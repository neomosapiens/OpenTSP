//! The fork-join pool of the MXM (`slices::pool`): every chunk runs exactly once, whatever the workers do (spinning,
//! parked, woken late, absent), jobs follow one another without a chunk of one running with the function of another,
//! two callers at once are both served, and a panic in a chunk comes back to the caller.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tsp_iss::slices::pool::Pool;

fn counters(n: usize) -> Vec<AtomicUsize> {
    (0..n).map(|_| AtomicUsize::new(0)).collect()
}

#[test]
fn every_chunk_runs_exactly_once_for_every_pool_width() {
    for workers in [0usize, 1, 2, 3, 7] {
        let pool = Pool::new(workers);
        assert_eq!(pool.width(), workers + 1);
        for chunks in [0usize, 1, 2, 3, 7, 8, 64, 1000] {
            let hits = counters(chunks);
            pool.run(chunks, &|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "workers {workers} chunks {chunks}");
        }
    }
}

#[test]
fn many_short_jobs_in_a_row_keep_their_own_closures() {
    let pool = Pool::new(3);
    for job in 0..20_000usize {
        let hits = counters(8);
        let tag = job % 251;
        pool.run(8, &|i| {
            hits[i].fetch_add(tag + 1, Ordering::Relaxed);
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == tag + 1), "job {job}");
    }
}

/// More worker threads than cores (several pools at once), so that a worker is often descheduled between reading a job
/// and claiming a chunk of it: it must not claim a chunk of the next job with the function (or the chunk count) of the
/// one it read.
#[test]
fn a_late_worker_never_runs_a_chunk_of_the_next_job_with_the_old_function() {
    std::thread::scope(|s| {
        for p in 0..6usize {
            s.spawn(move || {
                let pool = Pool::new(12);
                for job in 0..30_000usize {
                    let chunks = 2 + (job + p) % 5;
                    let hits = counters(chunks);
                    let tag = job % 251;
                    pool.run(chunks, &|i| {
                        hits[i].fetch_add(tag + 1, Ordering::Relaxed);
                    });
                    assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == tag + 1), "pool {p} job {job}");
                }
            });
        }
    });
}

#[test]
fn workers_that_parked_between_jobs_are_woken_or_left_behind_without_harm() {
    let pool = Pool::new(3);
    for round in 0..40usize {
        std::thread::sleep(Duration::from_millis(if round % 2 == 0 { 3 } else { 0 }));
        let hits = counters(16);
        pool.run(16, &|i| {
            hits[i].fetch_add(1, Ordering::Relaxed);
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "round {round}");
    }
}

#[test]
fn two_callers_at_once_both_get_every_chunk() {
    let pool = Pool::new(3);
    std::thread::scope(|s| {
        for t in 0..4usize {
            let pool = &pool;
            s.spawn(move || {
                for j in 0..2_000usize {
                    let hits = counters(8);
                    pool.run(8, &|i| {
                        hits[i].fetch_add(t + 1, Ordering::Relaxed);
                    });
                    assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == t + 1), "caller {t} job {j}");
                }
            });
        }
    });
}

#[test]
fn a_panic_in_a_chunk_reaches_the_caller_and_the_pool_stays_usable() {
    let pool = Pool::new(2);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.run(8, &|i| {
            if i == 5 {
                panic!("chunk 5");
            }
        })
    }));
    assert!(r.is_err());
    let hits = counters(8);
    pool.run(8, &|i| {
        hits[i].fetch_add(1, Ordering::Relaxed);
    });
    assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
}

#[test]
fn the_shared_pool_has_at_least_the_caller() {
    assert!(Pool::shared().width() >= 1);
}

// ---- the adaptive pool (`--matrix-threads auto`) ----

use tsp_iss::slices::pool::{HOLD_OFF, PROBE_JOBS, SAVE_MIN_NS};

/// Spin for about `us` microseconds (a chunk of real work).
fn spin(us: u64) {
    let t = std::time::Instant::now();
    while t.elapsed() < Duration::from_micros(us) {
        std::hint::spin_loop();
    }
}

#[test]
fn an_adaptive_pool_runs_every_chunk_once_in_every_mode() {
    let pool = Pool::new_adaptive(3);
    for round in 0..4 * PROBE_JOBS as usize {
        let chunks = 2 + round % 7;
        let hits = counters(chunks);
        pool.run(chunks, &|i| {
            hits[i].fetch_add(1, Ordering::Relaxed);
            if round % 2 == 0 {
                spin(20);
            }
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "round {round}");
    }
}

#[test]
fn an_adaptive_pool_stops_splitting_products_cheaper_than_a_fork() {
    let pool = Pool::new_adaptive(3);
    assert_eq!((pool.width(), pool.splitting()), (4, None), "it measures first");
    for _ in 0..PROBE_JOBS + 2 {
        pool.run(8, &|_| spin(1)); // eight chunks of about a microsecond: the join costs more than the work
    }
    assert_eq!(pool.splitting(), Some(false), "a split that saves less than {SAVE_MIN_NS} ns a fork does not pay");
    assert_eq!(pool.width(), 1, "no worker is woken for the next products");
    // after HOLD_OFF decisions it measures again
    for _ in 0..HOLD_OFF {
        pool.width();
    }
    assert_eq!(pool.splitting(), None);
    assert_eq!(pool.width(), 4);
}

#[test]
fn an_adaptive_pool_keeps_splitting_products_that_pay() {
    if std::thread::available_parallelism().map_or(1, |n| n.get()) < 2 {
        return; // one core: a split saves nothing, the pool is right to stop
    }
    let pool = Pool::new_adaptive(3);
    for _ in 0..PROBE_JOBS + 2 {
        pool.run(8, &|_| spin(300)); // 2.4 ms of work over four threads
    }
    assert_eq!(pool.splitting(), Some(true));
    assert_eq!(pool.width(), 4);
}

#[test]
fn a_pool_that_is_not_adaptive_never_changes_its_width() {
    let pool = Pool::new(3);
    for _ in 0..PROBE_JOBS * 3 {
        pool.run(8, &|_| {});
    }
    assert_eq!((pool.width(), pool.splitting()), (4, None));
}

#[test]
fn a_probe_that_ran_serially_does_not_count() {
    // one chunk is never forked: the probe window does not move, the pool keeps measuring
    let pool = Pool::new_adaptive(3);
    for _ in 0..4 * PROBE_JOBS {
        pool.run(1, &|_| {});
    }
    assert_eq!(pool.splitting(), None);
}
