//! A small persistent pool for fork-join over independent chunks of one unit instruction (the output lanes of an MXM
//! ABC). The workers spin on one atomic word for a while after their last job and then park; the caller never waits
//! for a worker to wake: it claims chunks itself, so a sleeping pool costs one `unpark` check and the result is that
//! of the serial loop whatever the workers do. Nothing depends on the pool for correctness: [`Pool::run`] with no
//! worker, or with the pool busy in another thread, runs every chunk on the calling thread in order.
//!
//! The state word packs the job number (high 32 bits) and the next chunk to claim (low 32 bits); a chunk is claimed by
//! a compare-and-swap on it. The caller first moves the word to the next job number with every chunk claimed
//! ([`CLOSED`]), then writes the job's words (function, data, chunk count), then opens the word (next chunk 0). A
//! worker reads the words only after it has loaded an open state, and the compare-and-swap fails if the word moved in
//! between, so a worker that wakes late (it still holds the previous job number, or the closed state) cannot claim a
//! chunk of the next job with the function of the last one, nor with a mix of the two.
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle, Thread};
use std::time::Instant;

/// Spin iterations of an idle worker before it parks (about 150 us on the target machine).
const SPIN: u32 = 30_000;

/// The most threads a pool has (`TSP_ISS_THREADS` and `--matrix-threads` are clamped to it).
pub const MAX_WIDTH: usize = 16;

/// An adaptive pool (`--matrix-threads auto`) measures the first [`PROBE_JOBS`] forks it makes: the time of every chunk and the
/// wall time of every fork. The split pays when it **saves** at least [`SAVE_MIN_NS`] per fork on average (the chunk times
/// minus the wall time): a dense f16 product of a model saves about 25 us, while a sparse program's products (the decode step
/// of a language model: about 5 us of work, 3 us of fork and join) save about 2 us, a speedup of 1.6 on paper that is no gain
/// and burns the cores of the workers, which spin between jobs. When it does not pay the pool answers `width() == 1` for the
/// next [`HOLD_OFF`] decisions (every `width()` call is one), so no worker is woken and none spins, and probes again after
/// that, since the activations may become dense; a split that pays is kept for [`HOLD_ON`] decisions before the next probe.
pub const PROBE_JOBS: u32 = 24;
pub const SAVE_MIN_NS: u64 = 10_000;
pub const HOLD_OFF: u32 = 2048;
pub const HOLD_ON: u32 = 16384;

const PROBING: u32 = 0;
const PAYS: u32 = 1;
const PAYS_NOT: u32 = 2;

#[derive(Default)]
struct Adaptive {
    on: AtomicBool,
    mode: AtomicU32,
    jobs: AtomicU32,
    work_ns: AtomicU64,
    wall_ns: AtomicU64,
    /// Decisions (`width()` calls) since the mode was set.
    held: AtomicU32,
}

static POOL: OnceLock<Pool> = OnceLock::new();
/// The width asked for by [`Pool::configure`].
static WIDTH: OnceLock<usize> = OnceLock::new();

/// The next-chunk value of a job that is being set up (and of none that can be claimed).
const CLOSED: u64 = 0xffff_ffff;

struct Shared {
    /// Job number << 32 | the next chunk to claim.
    state: AtomicU64,
    /// Chunks finished in the current job.
    done: AtomicUsize,
    /// The current job as plain words (the function, its data, the chunk count), written before `state` publishes it.
    call: AtomicUsize,
    data: AtomicUsize,
    chunks: AtomicUsize,
    sleepers: AtomicUsize,
    quit: AtomicBool,
    panicked: AtomicBool,
}

pub struct Pool {
    adapt: Adaptive,
    shared: Arc<Shared>,
    threads: Vec<Thread>,
    handles: Vec<JoinHandle<()>>,
    /// One caller at a time forks; a second one runs its chunks itself.
    busy: Mutex<()>,
}

impl Shared {
    /// Claim and run chunks of job `seq` until none is left.
    fn work(&self, seq: u64) {
        loop {
            let cur = self.state.load(Ordering::Acquire);
            let next = cur & 0xffff_ffff;
            if cur >> 32 != seq || next == CLOSED {
                return;
            }
            // The words of job `seq` (`state` was loaded open, so they were written before it); if the caller has moved
            // on since, the compare-exchange below fails and they are not used.
            let (call, data, chunks) = (
                self.call.load(Ordering::Relaxed),
                self.data.load(Ordering::Relaxed),
                self.chunks.load(Ordering::Relaxed),
            );
            if next >= chunks as u64 {
                return;
            }
            if self.state.compare_exchange_weak(cur, cur + 1, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                // SAFETY: the claim succeeded on the state loaded above, which has not changed since, so the words are
                // those of job `seq`: `call` is the `unsafe fn(*const (), usize)` of its `run`, `data` its closure,
                // which `run` keeps alive (with the memory the closure writes) until every claimed chunk has counted
                // itself in `done`.
                let r = catch_unwind(AssertUnwindSafe(|| unsafe {
                    let call = std::mem::transmute::<usize, unsafe fn(*const (), usize)>(call);
                    call(data as *const (), next as usize)
                }));
                if r.is_err() {
                    self.panicked.store(true, Ordering::Relaxed);
                }
                self.done.fetch_add(1, Ordering::AcqRel);
            }
        }
    }
}

fn worker(shared: Arc<Shared>) {
    let mut seen = shared.state.load(Ordering::Acquire) >> 32;
    loop {
        let mut spins = 0;
        let seq = loop {
            if shared.quit.load(Ordering::Relaxed) {
                return;
            }
            let cur = shared.state.load(Ordering::Acquire);
            if cur >> 32 != seen && cur & 0xffff_ffff != CLOSED {
                break cur >> 32;
            }
            spins += 1;
            if spins < SPIN {
                std::hint::spin_loop();
            } else {
                shared.sleepers.fetch_add(1, Ordering::SeqCst);
                if shared.state.load(Ordering::SeqCst) >> 32 == seen && !shared.quit.load(Ordering::SeqCst) {
                    thread::park();
                }
                shared.sleepers.fetch_sub(1, Ordering::SeqCst);
                spins = 0;
            }
        };
        seen = seq;
        shared.work(seq);
    }
}

impl Pool {
    /// A pool with `workers` threads besides the caller; 0 gives a serial pool.
    pub fn new(workers: usize) -> Pool {
        let shared = Arc::new(Shared {
            state: AtomicU64::new(0),
            done: AtomicUsize::new(0),
            call: AtomicUsize::new(0),
            data: AtomicUsize::new(0),
            chunks: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0),
            quit: AtomicBool::new(false),
            panicked: AtomicBool::new(false),
        });
        let handles: Vec<JoinHandle<()>> = (0..workers)
            .map(|i| {
                let s = shared.clone();
                thread::Builder::new().name(format!("tsp-iss-pool-{i}")).spawn(move || worker(s)).expect("pool thread")
            })
            .collect();
        let threads = handles.iter().map(|h| h.thread().clone()).collect();
        Pool { adapt: Adaptive::default(), shared, threads, handles, busy: Mutex::new(()) }
    }

    /// A pool of `workers` threads besides the caller that measures its own forks and stops splitting when that does not
    /// pay (see [`PROBE_JOBS`]).
    pub fn new_adaptive(workers: usize) -> Pool {
        let p = Pool::new(workers);
        p.adapt.on.store(true, Ordering::Relaxed);
        p
    }

    /// The threads the pool has, the caller included, whatever an adaptive pool answers for [`Pool::width`] just now.
    pub fn threads(&self) -> usize {
        self.handles.len() + 1
    }

    /// Whether the pool measures its forks (`--matrix-threads auto`).
    pub fn is_adaptive(&self) -> bool {
        self.adapt.on.load(Ordering::Relaxed)
    }

    /// For an adaptive pool: `Some(true)` while its split pays, `Some(false)` while it does not, `None` while it measures;
    /// `None` also for a pool that is not adaptive.
    pub fn splitting(&self) -> Option<bool> {
        if !self.adapt.on.load(Ordering::Relaxed) {
            return None;
        }
        match self.adapt.mode.load(Ordering::Relaxed) {
            PAYS => Some(true),
            PAYS_NOT => Some(false),
            _ => None,
        }
    }

    /// The forks the pool has measured since its last decision (0 for a pool that has forked nothing: an adaptive pool on a program whose
    /// products are all below the fork threshold never measures, so it never decides).
    pub fn measured_forks(&self) -> usize {
        self.adapt.jobs.load(Ordering::Relaxed) as usize
    }

    fn adapt_set(&self, mode: u32) {
        self.adapt.jobs.store(0, Ordering::Relaxed);
        self.adapt.work_ns.store(0, Ordering::Relaxed);
        self.adapt.wall_ns.store(0, Ordering::Relaxed);
        self.adapt.held.store(0, Ordering::Relaxed);
        self.adapt.mode.store(mode, Ordering::Relaxed);
    }

    /// One fork has been measured: after [`PROBE_JOBS`] of them, decide.
    fn adapt_record(&self, work_ns: u64, wall_ns: u64) {
        let a = &self.adapt;
        a.work_ns.fetch_add(work_ns, Ordering::Relaxed);
        a.wall_ns.fetch_add(wall_ns, Ordering::Relaxed);
        if a.jobs.fetch_add(1, Ordering::Relaxed) + 1 >= PROBE_JOBS && a.mode.load(Ordering::Relaxed) == PROBING {
            let (work, wall) = (a.work_ns.load(Ordering::Relaxed), a.wall_ns.load(Ordering::Relaxed));
            let saved = work.saturating_sub(wall) / u64::from(PROBE_JOBS);
            self.adapt_set(if saved >= SAVE_MIN_NS { PAYS } else { PAYS_NOT });
        }
    }

    /// The process-wide pool: the width set by [`Pool::configure`] (`tsp-iss run --matrix-threads`), else `TSP_ISS_THREADS`
    /// threads in all, else 1 (no worker, nothing to decide; a width above 1 is adaptive: the measurement on the P(8,4,128) benchmark found no gain from four threads, see
    /// `task-45-report.md`; a dense matrix program does scale, 3.1x at six threads (`opentsp-iss/README.md`), which `--matrix-threads auto` is for).
    /// A product is split over the workers only when its estimated serial time is above `SPLIT_MIN_NS` (`slices/mxm.rs`).
    pub fn shared() -> &'static Pool {
        POOL.get_or_init(|| {
            let cores = thread::available_parallelism().map_or(1, |n| n.get());
            let want = WIDTH
                .get()
                .copied()
                .or_else(|| std::env::var("TSP_ISS_THREADS").ok().and_then(|v| v.parse::<usize>().ok()))
                .unwrap_or(1);
            let workers = want.clamp(1, MAX_WIDTH).min(cores.max(1)) - 1;
            // every width above 1 is adaptive: a wrong estimate of what a split saves cannot slow a program
            if workers > 0 {
                Pool::new_adaptive(workers)
            } else {
                Pool::new(workers)
            }
        })
    }

    /// [`Pool::configure`] (`--matrix-threads auto`): the shared pool of any width above 1 is adaptive, it measures its first
    /// forks and stops splitting when that does not pay.
    pub fn configure_adaptive(width: usize) -> Result<(), usize> {
        Pool::configure(width)
    }

    /// Set the width of the process-wide pool, before its first use: `Ok` when the pool will have `width` threads in all
    /// (clamped to 1..=16 and to the cores), `Err(width in use)` when the pool exists already, with another width.
    pub fn configure(width: usize) -> Result<(), usize> {
        let _ = WIDTH.set(width);
        let cores = thread::available_parallelism().map_or(1, |n| n.get());
        let wanted = WIDTH.get().copied().unwrap_or(1).clamp(1, MAX_WIDTH).min(cores.max(1));
        let have = Pool::shared().width();
        if have == wanted {
            Ok(())
        } else {
            Err(have)
        }
    }

    /// Threads that can work on a job, the caller included; for an adaptive pool 1 while splitting does not pay (every
    /// call is a decision whether to fork, and counts towards the next probe).
    pub fn width(&self) -> usize {
        let full = self.handles.len() + 1;
        if !self.adapt.on.load(Ordering::Relaxed) {
            return full;
        }
        match self.adapt.mode.load(Ordering::Relaxed) {
            PAYS_NOT => {
                if self.adapt.held.fetch_add(1, Ordering::Relaxed) >= HOLD_OFF {
                    self.adapt_set(PROBING);
                    return full;
                }
                1
            }
            PAYS => {
                if self.adapt.held.fetch_add(1, Ordering::Relaxed) >= HOLD_ON {
                    self.adapt_set(PROBING);
                }
                full
            }
            _ => full,
        }
    }

    /// Run `f(0)`, ..., `f(chunks - 1)`, each exactly once, on the pool's threads and the caller's. Returns when all
    /// have returned. With no worker, with a busy pool, or for one chunk, they run on the caller in order.
    pub fn run<F: Fn(usize) + Sync>(&self, chunks: usize, f: &F) {
        if self.adapt.on.load(Ordering::Relaxed) && self.adapt.mode.load(Ordering::Relaxed) == PROBING {
            // a probe: the time of every chunk and of the whole fork
            let work = AtomicU64::new(0);
            let timed = |i: usize| {
                let t = Instant::now();
                f(i);
                work.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            };
            let t0 = Instant::now();
            if self.fork(chunks, &timed) {
                self.adapt_record(work.load(Ordering::Relaxed), t0.elapsed().as_nanos() as u64);
            }
        } else {
            self.fork(chunks, f);
        }
    }

    /// [`Pool::run`]; `true` when the chunks were forked over the workers, `false` when the caller ran them all.
    fn fork<F: Fn(usize) + Sync>(&self, chunks: usize, f: &F) -> bool {
        let serial = |f: &F| {
            (0..chunks).for_each(f);
            false
        };
        if chunks <= 1 || self.handles.is_empty() {
            return serial(f);
        }
        let Ok(_guard) = self.busy.try_lock() else { return serial(f) };
        unsafe fn call<F: Fn(usize)>(data: *const (), i: usize) {
            // SAFETY: `data` is the `&F` of the `run` that published this job, alive until it returns.
            unsafe { (*(data as *const F))(i) }
        }
        assert!(chunks < CLOSED as usize, "too many chunks");
        let s = &*self.shared;
        let seq = (s.state.load(Ordering::Relaxed) >> 32) + 1;
        // close the word: a worker holding the previous job number can no longer claim anything
        s.state.store(seq << 32 | CLOSED, Ordering::SeqCst);
        s.done.store(0, Ordering::Relaxed);
        s.call.store(call::<F> as unsafe fn(*const (), usize) as usize, Ordering::Relaxed);
        s.data.store(f as *const F as usize, Ordering::Relaxed);
        s.chunks.store(chunks, Ordering::Relaxed);
        s.state.store(seq << 32, Ordering::SeqCst);
        if s.sleepers.load(Ordering::SeqCst) > 0 {
            for t in &self.threads {
                t.unpark();
            }
        }
        s.work(seq);
        let mut spins = 0u32;
        while s.done.load(Ordering::Acquire) < chunks {
            spins += 1;
            if spins < 20_000 {
                std::hint::spin_loop();
            } else {
                thread::yield_now(); // a worker holding a chunk was descheduled
            }
        }
        assert!(!s.panicked.swap(false, Ordering::Relaxed), "a chunk of a pooled job panicked");
        true
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.quit.store(true, Ordering::SeqCst);
        for t in &self.threads {
            t.unpark();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}
