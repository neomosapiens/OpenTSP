//! The bounded schedule cache: a least-recently-used cache of `Arc<Schedule>` with an entry limit and a
//! byte budget, counters (hits, misses, divergences, evictions, retained bytes), recording on the second sight of a key,
//! and, in a platform, a divergent replay that falls back to the normal driver instead of ending the run.
//!
//! The cache tests need no replay build; the tests of served runs are for `--features replay`
//! (`just iss-replay-test`).
use tsp_iss::driver::replay::{parse_size, Lookup, RecordOn, ReplayCache, Schedule};

fn key(n: u8) -> [u8; 32] {
    [n; 32]
}

/// A schedule of about `ticks * 12` bytes (`cycles` and `text`) for key `n`.
fn schedule(n: u8, ticks: usize) -> Schedule {
    Schedule { key: key(n), cycles: vec![0; ticks], text: vec![0; ticks], ..Schedule::default() }
}

fn expect_hit(c: &ReplayCache, n: u8) {
    assert!(matches!(c.begin(&key(n), RecordOn::First), Lookup::Hit(_)), "key {n} should be stored");
}

#[test]
fn the_entry_limit_evicts_the_least_recently_used_schedule() {
    let c = ReplayCache::new(usize::MAX, 2);
    c.insert(schedule(1, 10));
    c.insert(schedule(2, 10));
    expect_hit(&c, 1); // 1 is now the most recently used, 2 the least
    c.insert(schedule(3, 10));
    assert!(c.contains(&key(1)) && c.contains(&key(3)) && !c.contains(&key(2)), "2 was the least recently used");
    let n = c.counters();
    assert_eq!((n.schedules, n.evictions, n.recordings, n.hits), (2, 1, 3, 1));
    assert_eq!(n.max_schedules, 2);
}

#[test]
fn the_byte_budget_bounds_the_retained_bytes() {
    let one = schedule(0, 1000).bytes();
    let c = ReplayCache::new(one * 2 + one / 2, 100); // room for two
    for n in 1..=5 {
        c.insert(schedule(n, 1000));
        let k = c.counters();
        assert!(k.retained_bytes <= k.budget_bytes, "after {n} insertions: {k:?}");
    }
    let k = c.counters();
    assert_eq!((k.schedules, k.evictions), (2, 3));
    assert_eq!(k.retained_bytes as usize, 2 * one, "retained bytes follow the stored schedules");
    assert!(c.contains(&key(4)) && c.contains(&key(5)));
    // one schedule bigger than the whole budget is not stored and evicts nothing
    c.insert(schedule(9, 100_000));
    let k = c.counters();
    assert_eq!((k.schedules, k.evictions, k.oversize), (2, 3, 1));
    assert!(!c.contains(&key(9)));
    // and the key is not recorded again: that would cost every run and keep nothing
    assert!(matches!(c.begin(&key(9), RecordOn::First), Lookup::TooBig));
    assert!(matches!(c.begin(&key(9), RecordOn::Second), Lookup::TooBig));
    // a schedule inserted again replaces itself instead of counting twice
    c.insert(schedule(5, 1000));
    assert_eq!((c.counters().schedules, c.counters().retained_bytes as usize), (2, 2 * one));
}

#[test]
fn a_key_is_recorded_on_its_second_sight_and_served_from_the_third() {
    let c = ReplayCache::new(usize::MAX, 4);
    assert!(matches!(c.begin(&key(7), RecordOn::Second), Lookup::FirstSight), "the first sight costs nothing");
    assert!(matches!(c.begin(&key(7), RecordOn::Second), Lookup::Record), "the second records");
    c.insert(schedule(7, 10));
    assert!(matches!(c.begin(&key(7), RecordOn::Second), Lookup::Hit(_)), "the third is served");
    let k = c.counters();
    assert_eq!((k.misses, k.hits, k.recordings, k.schedules), (2, 1, 1, 1));
    // another key starts again at its first sight; the policy `First` records at once
    assert!(matches!(c.begin(&key(8), RecordOn::Second), Lookup::FirstSight));
    assert!(matches!(c.begin(&key(9), RecordOn::First), Lookup::Record));
    // a key seen once and never again keeps no schedule
    assert!(!c.contains(&key(8)));
}

#[test]
fn a_divergence_drops_the_schedule_and_is_counted() {
    let c = ReplayCache::new(usize::MAX, 4);
    c.insert(schedule(1, 10));
    c.diverged(&key(1));
    let k = c.counters();
    assert_eq!((k.divergences, k.schedules, k.retained_bytes), (1, 0, 0));
    // the next sight records again (the schedule no longer describes the text)
    assert!(matches!(c.begin(&key(1), RecordOn::Second), Lookup::Record));
    c.shadow(true);
    c.shadow(false);
    assert_eq!((c.counters().shadow_checks, c.counters().shadow_mismatches), (2, 1));
}

#[test]
fn sizes_are_bytes_with_an_optional_binary_suffix() {
    assert_eq!(parse_size("1048576"), Some(1 << 20));
    assert_eq!(parse_size("512k"), Some(512 << 10));
    assert_eq!(parse_size("64M"), Some(64 << 20));
    assert_eq!(parse_size(" 2g "), Some(2 << 30));
    for bad in ["", "m", "-1", "1.5g", "12x", "99999999999999999999g"] {
        assert_eq!(parse_size(bad), None, "{bad:?}");
    }
}

#[test]
fn a_schedule_counts_what_it_retains() {
    let a = schedule(1, 0).bytes();
    assert_eq!(schedule(1, 1000).bytes() - a, 1000 * (8 + 4));
}

#[cfg(feature = "replay")]
mod served {
    use super::*;

    /// Run `body` on a thread with a stack large enough for the machines of a served run (a machine of the widest lane count
    /// and the replay bookkeeping around it do not fit the 2 MiB of a test thread twice over).
    fn on_big_stack(body: fn()) {
        let t = std::thread::Builder::new().stack_size(32 << 20).spawn(body).unwrap();
        if let Err(e) = t.join() {
            std::panic::resume_unwind(e);
        }
    }

    use ots_profile::Profile;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tsp_iss::driver::replay::{Began, Mode, Op};
    use tsp_iss::links::NoLinks;
    use tsp_iss::machine::Machine;
    use tsp_iss::platform::{Platform, PlatformError};

    const LIMIT: u64 = 6000;

    fn names() -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/replay"))
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|n| n.ends_with(".txt"))
            .collect();
        v.sort();
        v
    }

    fn data(name: &str) -> String {
        std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/replay").join(name)).unwrap()
    }

    fn profile_of(name: &str) -> &'static str {
        if name.starts_with("ref") {
            "ref"
        } else {
            "mini"
        }
    }

    /// The same program with every preload byte changed (the data of another request; the text is the same).
    fn flip(prog: &str) -> String {
        prog.lines()
            .map(|l| {
                if l.starts_with("S ") {
                    let mut f: Vec<String> = l.split(' ').map(String::from).collect();
                    for b in f.iter_mut().skip(3) {
                        *b = format!("{:x}", u8::from_str_radix(b, 16).unwrap() ^ 0x5a);
                    }
                    f.join(" ")
                } else {
                    l.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// What a host sees of one run: the result of every advance, the records it pops after each, then the cycle, the
    /// diagnostics and the SRAM.
    type Seen = (Vec<String>, u64, String, [u8; 32]);

    /// Reset, load the program text (`Q` lines) and data (`S` lines) of the harness protocol, start and advance in steps of
    /// `step` cycles, popping the error FIFO after each.
    fn serve(p: &mut Platform, prog: &str, step: u64) -> Seen {
        p.reset().unwrap();
        let hex = |s: &str| u64::from_str_radix(s, 16).unwrap();
        for l in prog.lines() {
            let f: Vec<&str> = l.split(' ').collect();
            match f[0] {
                "S" => {
                    let bytes: Vec<u8> = f[3..].iter().map(|b| hex(b) as u8).collect();
                    p.write_sram_word(hex(f[1]) as u8, hex(f[2]) as u32, &bytes).unwrap();
                }
                "Q" => {
                    p.select_iq(hex(f[1]) as u16).unwrap();
                    let units: Vec<u32> = f[2..].iter().map(|u| hex(u) as u32).collect();
                    p.write_iq_window(&units).unwrap();
                }
                _ => {}
            }
        }
        p.start().unwrap();
        let mut steps = Vec::new();
        while p.cycle() < LIMIT {
            let r = p.advance(step);
            steps.push(format!("{r:?}"));
            while let Some(rec) = p.err_pop() {
                steps.push(format!("pop {} {} {}", rec.cycle, rec.queue, rec.code.as_str()));
            }
            if r.is_err() || p.status().quiescent {
                break;
            }
        }
        (steps, p.cycle(), format!("{:?}", p.diagnostics()), p.sram_digest())
    }

    fn platform(name: &str) -> Platform {
        let mut p = Platform::new(&Profile::load(profile_of(name)).unwrap(), 0).unwrap();
        p.set_replay_sessions(false);
        p
    }

    fn session(name: &str, cache: &Arc<ReplayCache>) -> Platform {
        let mut p = platform(name);
        p.set_replay_cache(cache.clone());
        p.set_replay_sessions(true);
        p
    }

    fn new_cache() -> Arc<ReplayCache> {
        Arc::new(ReplayCache::new(1 << 30, 16))
    }

    /// The programs a platform can record (no link): with their schedule from a session of two runs.
    fn recorded() -> Vec<(String, Schedule)> {
        let mut out = Vec::new();
        for n in names() {
            let cache = new_cache();
            let mut p = session(&n, &cache);
            let prog = data(&n);
            serve(&mut p, &prog, 50);
            serve(&mut p, &prog, 50);
            if let Some(s) = cache.schedules().pop() {
                if s.ticks() > 40 {
                    out.push((n, (*s).clone()));
                }
            }
        }
        out
    }

    #[test]
    fn the_third_run_of_a_program_is_served_and_each_run_is_the_run_without_sessions() {
        on_big_stack(the_third_run_of_a_program_is_served_and_each_run_is_the_run_without_sessions_body);
    }

    fn the_third_run_of_a_program_is_served_and_each_run_is_the_run_without_sessions_body() {
        let cache = new_cache();
        let name = "mini-3.txt";
        let (mut plain, mut sess) = (platform(name), session(name, &cache));
        let prog = data(name);
        let mut modes = Vec::new();
        for k in 0..4 {
            let input = if k % 2 == 0 { prog.clone() } else { flip(&prog) };
            let a = serve(&mut plain, &input, 50);
            let b = serve(&mut sess, &input, 50);
            assert_eq!(a, b, "run {k}");
            modes.push(sess.replay_mode());
        }
        // run 0 is the first sight (normal), run 1 records, runs 2 and 3 replay; the chip is back to normal after its run
        let k = cache.counters();
        assert_eq!((k.misses, k.hits, k.recordings, k.schedules, k.divergences), (2, 2, 1, 1, 0), "{k:?} {modes:?}");
        assert!(k.retained_bytes > 0);
    }

    #[test]
    fn a_divergent_replay_gives_the_normal_driver_result_not_an_error() {
        on_big_stack(a_divergent_replay_gives_the_normal_driver_result_not_an_error_body);
    }

    fn a_divergent_replay_gives_the_normal_driver_result_not_an_error_body() {
        let recs = recorded();
        assert!(recs.len() >= 6, "only {} programs recorded", recs.len());
        for (n, mut s) in recs {
            let mid = s.ticks() / 2;
            s.text[mid] ^= 1; // the text of the stored schedule is not the text of the run
            let cache = new_cache();
            cache.insert(s);
            let prog = data(&n);
            let want = serve(&mut platform(&n), &prog, 50);
            let mut sess = session(&n, &cache);
            let got = serve(&mut sess, &prog, 50);
            assert_eq!(want, got, "{n}: the fallback differs from the normal driver");
            let k = cache.counters();
            assert_eq!((k.divergences, k.hits, k.schedules), (1, 1, 0), "{n}: {k:?}");
            assert_eq!(sess.replay_mode(), Mode::Off, "{n}: the run after the fallback is the normal driver");
            // the key is forgotten: the next run records again, the one after is served
            serve(&mut sess, &prog, 50);
            assert_eq!(cache.counters().schedules, 1, "{n}");
            let again = serve(&mut sess, &prog, 50);
            assert_eq!(want, again, "{n}: a run served after the fallback differs");
        }
    }

    #[test]
    fn without_the_snapshot_a_divergence_is_the_model_fault() {
        on_big_stack(without_the_snapshot_a_divergence_is_the_model_fault_body);
    }

    fn without_the_snapshot_a_divergence_is_the_model_fault_body() {
        let (n, mut s) = recorded().into_iter().next().unwrap();
        let mid = s.ticks() / 2;
        s.text[mid] ^= 1;
        let cache = new_cache();
        cache.insert(s);
        let mut p = session(&n, &cache);
        p.set_replay_snapshot(false);
        let (steps, ..) = serve(&mut p, &data(&n), 50);
        let last = steps.iter().find(|l| l.starts_with("Err")).unwrap_or_else(|| panic!("{n}: {steps:?}"));
        assert!(last.contains("replay divergence"), "{n}: {last}");
    }

    #[test]
    fn a_stored_schedule_that_misses_a_diagnostic_is_caught_by_the_shadow_check() {
        on_big_stack(a_stored_schedule_that_misses_a_diagnostic_is_caught_by_the_shadow_check_body);
    }

    fn a_stored_schedule_that_misses_a_diagnostic_is_caught_by_the_shadow_check_body() {
        let mut checked = 0;
        for (n, mut s) in recorded() {
            let Some(i) = s.ops.iter().position(|o| matches!(o, Op::Diag(..))) else { continue };
            s.ops.remove(i); // a replay would now be silent about one diagnostic of the driver
            for off in s.ops_off.iter_mut().skip(1) {
                if *off as usize > i {
                    *off -= 1;
                }
            }
            let cache = new_cache();
            cache.insert(s);
            let prog = data(&n);
            let want = serve(&mut platform(&n), &prog, 50);
            let mut sess = session(&n, &cache);
            sess.set_replay_shadow(1);
            let got = serve(&mut sess, &prog, 50);
            // the host read the wrong FIFO while the run went on (the check is an audit at the end of the run: it cannot
            // unsay that); the result it leaves, the cycle count, the diagnostics and the SRAM, is the normal driver's
            assert_eq!(
                (&want.1, &want.2, &want.3),
                (&got.1, &got.2, &got.3),
                "{n}: the shadow check must repair the result"
            );
            let k = cache.counters();
            assert_eq!((k.shadow_checks, k.shadow_mismatches), (1, 1), "{n}: {k:?}");
            checked += 1;
        }
        assert!(checked >= 1, "no recorded program has a driver diagnostic");
    }

    #[test]
    fn a_shadow_check_of_a_good_schedule_agrees() {
        on_big_stack(a_shadow_check_of_a_good_schedule_agrees_body);
    }

    fn a_shadow_check_of_a_good_schedule_agrees_body() {
        let cache = new_cache();
        let name = "mini-3.txt";
        let (mut plain, mut sess) = (platform(name), session(name, &cache));
        sess.set_replay_shadow(1);
        for k in 0..4 {
            let input = if k % 2 == 0 { data(name) } else { flip(&data(name)) };
            assert_eq!(serve(&mut plain, &input, 50), serve(&mut sess, &input, 50), "run {k}");
        }
        let k = cache.counters();
        assert_eq!((k.hits, k.shadow_checks, k.shadow_mismatches), (2, 2, 0), "{k:?}");
    }

    #[test]
    fn the_session_with_a_small_budget_evicts_and_stays_exact() {
        on_big_stack(the_session_with_a_small_budget_evicts_and_stays_exact_body);
    }

    fn the_session_with_a_small_budget_evicts_and_stays_exact_body() {
        // two programs, room for one schedule: each is recorded, evicts the other, and every run is the plain run
        let cache = Arc::new(ReplayCache::new(1 << 30, 1));
        let recs = recorded();
        let (a, b) = (recs[0].0.as_str(), recs[1].0.as_str());
        assert_ne!(profile_of(a), "", "two programs that record");
        let (mut pa, mut pb) = (session(a, &cache), session(b, &cache));
        let (mut qa, mut qb) = (platform(a), platform(b));
        for round in 0..3 {
            assert_eq!(serve(&mut pa, &data(a), 50), serve(&mut qa, &data(a), 50), "a, round {round}");
            assert_eq!(serve(&mut pa, &data(a), 50), serve(&mut qa, &data(a), 50), "a again, round {round}");
            assert_eq!(serve(&mut pb, &data(b), 50), serve(&mut qb, &data(b), 50), "b, round {round}");
            assert_eq!(serve(&mut pb, &data(b), 50), serve(&mut qb, &data(b), 50), "b again, round {round}");
        }
        let k = cache.counters();
        assert!(k.schedules <= 1 && k.evictions >= 2, "{k:?}");
    }

    #[test]
    fn a_session_machine_records_on_the_second_sight_and_replays_the_third() {
        let name = "mini-3.txt";
        let cache = new_cache();
        let machine = || {
            let mut m = Machine::new(&Profile::load("mini").unwrap(), 0, Box::new(NoLinks)).unwrap();
            for l in data(name).lines() {
                let f: Vec<&str> = l.split(' ').collect();
                let hex = |s: &str| u64::from_str_radix(s, 16).unwrap();
                match f[0] {
                    "S" => m
                        .preload(
                            hex(f[1]) as u8,
                            hex(f[2]) as u32,
                            &f[3..].iter().map(|b| hex(b) as u8).collect::<Vec<_>>(),
                        )
                        .unwrap(),
                    "Q" => m
                        .load_image(hex(f[1]) as u16, &f[2..].iter().map(|u| hex(u) as u32).collect::<Vec<_>>())
                        .unwrap(),
                    _ => {}
                }
            }
            m
        };
        let mut seen = Vec::new();
        for _ in 0..3 {
            let mut m = machine();
            seen.push(m.replay_session_begin_in(&cache, RecordOn::Second));
            m.run(LIMIT, &mut tsp_iss::observe::NoObserver).unwrap();
            m.replay_session_end();
        }
        assert_eq!(seen, [Began::Normal, Began::Recording, Began::Replaying]);
    }

    #[test]
    fn a_replayed_run_reports_the_text_left_at_completion() {
        on_big_stack(a_replayed_run_reports_the_text_left_at_completion_body);
    }

    fn a_replayed_run_reports_the_text_left_at_completion_body() {
        // queue 0 holds a truncated parcel: the completion check raises IQ-004. A replay must raise it too.
        let prog = "Q 0 90000010 0\n";
        let cache = new_cache();
        let mut plain = platform("mini");
        let mut sess = session("mini", &cache);
        let want = serve(&mut plain, prog, 50);
        assert!(want.2.contains("Iq004"), "the program must raise IQ-004: {}", want.2);
        for k in 0..3 {
            assert_eq!(want, serve(&mut sess, prog, 50), "run {k}");
        }
        assert_eq!(cache.counters().hits, 1);
    }

    #[test]
    fn a_fault_that_is_not_a_divergence_still_reports_as_an_error() {
        // a platform that is not running refuses advance, with or without sessions
        let mut p = session("mini", &new_cache());
        assert!(matches!(p.advance(1), Err(PlatformError::WrongPhase(_))));
    }

    /// The real bucket (`python3 tools/iss_bench_prep.py`; `ISS_BENCH_DIR`, default `/tmp/iss-bench`; `ISS_BENCH_BUCKET`,
    /// default `P8_4_128`): the clean run, served, and 40 single-bit faults of its weight and text words, each compared with
    /// the normal driver (cycle count, diagnostics, SRAM). A fault in a word that is fetched as text makes the replay diverge:
    /// the platform falls back to the normal driver and the result is the normal driver's (the 2026-11-19 verification saw
    /// exit 4 for these in the harness). Prints the counters and the times. Run by `just iss-replay-bucket`.
    #[test]
    #[ignore = "needs the bucket inputs of tools/iss_bench_prep.py; run by just iss-replay-bucket"]
    fn a_real_bucket_with_single_bit_faults_falls_back_exactly() {
        on_big_stack(a_real_bucket_with_single_bit_faults_falls_back_exactly_body);
    }

    fn a_real_bucket_with_single_bit_faults_falls_back_exactly_body() {
        use std::time::Instant;
        let dir = PathBuf::from(std::env::var("ISS_BENCH_DIR").unwrap_or_else(|_| "/tmp/iss-bench".into()));
        let bucket = std::env::var("ISS_BENCH_BUCKET").unwrap_or_else(|_| "P8_4_128".into());
        let obj = std::fs::read(dir.join(format!("{bucket}.elf"))).expect("the bucket object");
        let pkg = std::fs::read(dir.join(format!("{bucket}.safetensors"))).expect("the bucket weights");
        let (name, images, externs) = ots_format::read_object(&obj, Some("ref"), None).unwrap();
        let tensors = ots_format::read_weights(&pkg, Some(&name), None).unwrap();
        let words: Vec<(u8, u32, Vec<u8>)> = ots_format::bind_weights(&externs, &tensors)
            .unwrap()
            .iter()
            .map(|w| (w.slice as u8, w.addr as u32, w.word.to_vec()))
            .collect();
        let load = |p: &mut Platform| {
            p.reset().unwrap();
            for (q, units) in images.iter() {
                p.select_iq(q).unwrap();
                p.write_iq_window(units).unwrap();
            }
            for (s, a, w) in &words {
                p.write_sram_word(*s, *a, w).unwrap();
            }
        };
        let finish = |p: &mut Platform, fault: Option<(u8, u32, u8, u8)>| -> Seen {
            p.start().unwrap();
            if let Some((s, a, l, b)) = fault {
                let _ = p.service_flip_sram(s, a, l, b); // at cycle 0, as an `F sram 0` line
            }
            let r = p.run_to_done(1 << 30).map(|c| c.to_string()).map_err(|e| format!("{e:?}"));
            (vec![format!("{r:?}")], p.cycle(), format!("{:?}", p.diagnostics()), p.sram_digest())
        };
        let profile = Profile::load("ref").unwrap();
        let mut plain = Platform::new(&profile, 0).unwrap();
        plain.set_replay_sessions(false);
        let cache = new_cache();
        let mut sess = Platform::new(&profile, 0).unwrap();
        sess.set_replay_cache(cache.clone());
        sess.set_replay_sessions(true);
        let timed = |p: &mut Platform, fault: Option<(u8, u32, u8, u8)>| {
            load(p);
            let t = Instant::now();
            let r = finish(p, fault);
            (r, t.elapsed().as_secs_f64() * 1e3)
        };
        let (want, plain_ms) = timed(&mut plain, None);
        let (a, first_ms) = timed(&mut sess, None);
        let (b, record_ms) = timed(&mut sess, None);
        let (c, served_ms) = timed(&mut sess, None);
        assert_eq!((&want, &want, &want), (&a, &b, &c), "the clean runs");
        // the cost of the SRAM snapshot that the divergence fallback needs: served runs with and without it, interleaved
        let (mut with, mut without) = (Vec::new(), Vec::new());
        for _ in 0..5 {
            sess.set_replay_snapshot(true);
            with.push(timed(&mut sess, None).1);
            sess.set_replay_snapshot(false);
            without.push(timed(&mut sess, None).1);
        }
        sess.set_replay_snapshot(true);
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        eprintln!(
            "{bucket}: plain {plain_ms:.1} ms, first sight {first_ms:.1}, recording {record_ms:.1}, served {served_ms:.1}; served with the snapshot {:.1} ms, without {:.1} ms (medians of 5, interleaved)",
            med(&mut with),
            med(&mut without)
        );
        let mut x: u64 = 0x2545_f491_4f6c_dd1d;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let (mut same, mut fell_back, mut fb_ms) = (0, 0, 0.0);
        for k in 0..40 {
            let (s, a, _) = &words[(next() % words.len() as u64) as usize];
            let fault = (*s, *a, (next() % 20) as u8, (next() % 138) as u8);
            let before = cache.counters().divergences;
            let (want, _) = timed(&mut plain, Some(fault));
            let (got, ms) = timed(&mut sess, Some(fault));
            if want != got {
                let (a, b) = (plain.sram_words(), sess.sram_words());
                let differ: Vec<_> = a
                    .iter()
                    .filter(|w| !b.iter().any(|v| v == *w))
                    .map(|w| (w.0, w.1))
                    .chain(b.iter().filter(|w| !a.iter().any(|v| v == *w)).map(|w| (w.0, w.1)))
                    .take(8)
                    .collect();
                eprintln!(
                    "fault {k} {fault:?}: diverged {} (before {before}); words that differ: {differ:?}",
                    cache.counters().divergences
                );
            }
            assert_eq!(want, got, "fault {k} {fault:?}");
            if cache.counters().divergences > before {
                fell_back += 1;
                fb_ms += ms;
                // the schedule was dropped: serve the clean run twice more so that it records and is served again
                timed(&mut sess, None);
                timed(&mut sess, None);
            } else {
                same += 1;
            }
        }
        let k = cache.counters();
        eprintln!(
            "{bucket}: 40 faults: {same} served exactly, {fell_back} fell back (mean {:.1} ms); counters {k:?}",
            if fell_back > 0 { fb_ms / fell_back as f64 } else { 0.0 }
        );
        assert!(fell_back >= 1 && same >= 1, "the faults should hit both text and data: {same} / {fell_back}");
        assert_eq!(k.divergences, fell_back);
    }
}
