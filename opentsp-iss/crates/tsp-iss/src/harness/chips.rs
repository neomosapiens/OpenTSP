//! Several chips in one process: lock-step windows (`tsp_sim.c` `run_chips`).
//!
//! `g` is the smallest next event over all chips and `L` the smallest latency of a connected link
//! (1 when none is). Every chip runs the window `[g, g + L)`: a vector sent at `t >= g` arrives at
//! `t + L >= g + L`, so no vector can arrive inside the window that sent it and the result equals
//! cycle-by-cycle execution. Vectors sent in the window are routed into the receivers' FIFOs, in
//! chip order and sending order, at the window's end. Each chip's trace and results are buffered
//! and printed in chip order after a `C <chip>` line.
//!
//! With `jobs` > 1 the chips of a window run on that many threads (the caller's and `jobs - 1`
//! workers that live for the whole run, parked on a barrier between windows). The chips of a window
//! are independent and everything between windows (routing, delivery, the next window) is done by
//! the caller's thread in chip order, so the bytes do not depend on `jobs`.
use super::links::{ChipLinks, VecMsg};
use super::obs::Recorder;
use super::output::TraceObserver;
use super::stats::{Counters, StatsObserver};
use crate::machine::{Machine, RunError};
use ots_isa::Code;
use std::any::Any;
use std::io;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};

/// One chip of a multi-chip run: the machine, its end of the links, and its buffered output.
pub struct ChipRun<const L: usize = { crate::geom::MAX_LANES }> {
    pub machine: Machine<L>,
    pub links: Arc<Mutex<ChipLinks>>,
    /// The chip's trace, then (when it finishes) its `E` and `W` lines.
    pub out: Vec<u8>,
    /// The state of the chip's [`TraceObserver`] between windows.
    pub diags: u64,
    pub first: Option<(Code, u64)>,
    pub error: Option<io::Error>,
    /// Cycles ticked so far, and the last of them.
    pub ticks: u64,
    pub last_cycle: Option<u64>,
    /// `--lint`: the chip's trace carries the `T lint` lines.
    pub lint: bool,
    /// `--stats`: what the chip's run counted.
    pub counts: Option<Counters>,
    /// `--run-profile-json` and the `--obs-*` flags: what the chip's run recorded, with the wall time of its setup and
    /// its run loop.
    pub obs: Option<super::obs::ObsState>,
    pub setup_ns: u64,
    pub run_ns: u64,
}

impl<const L: usize> ChipRun<L> {
    pub fn new(machine: Machine<L>, links: Arc<Mutex<ChipLinks>>) -> ChipRun<L> {
        ChipRun {
            machine,
            links,
            out: Vec::new(),
            diags: 0,
            first: None,
            error: None,
            ticks: 0,
            last_cycle: None,
            lint: false,
            counts: None,
            obs: None,
            setup_ns: 0,
            run_ns: 0,
        }
    }

    /// Tick every cycle of this chip below `wend`.
    // model: c_emulator/tsp_sim.c run_until
    fn run_until(&mut self, wend: u128, max_cycles: u64, trace: bool) -> Result<(), RunError> {
        let mut obs = TraceObserver::resume(&mut self.out, trace, self.diags, self.first, self.error.take());
        obs.lint = self.lint;
        let ran = std::time::Instant::now();
        let r = match (self.counts.as_mut(), self.obs.as_mut()) {
            (Some(counts), None) => {
                self.machine.run_horizon(Some(wend), max_cycles, &mut StatsObserver { inner: &mut obs, counts })
            }
            (None, None) => self.machine.run_horizon(Some(wend), max_cycles, &mut obs),
            (counts, Some(st)) => {
                st.resume();
                let r = match counts {
                    Some(counts) => self.machine.run_horizon(
                        Some(wend),
                        max_cycles,
                        &mut Recorder { inner: &mut StatsObserver { inner: &mut obs, counts }, st },
                    ),
                    None => self.machine.run_horizon(Some(wend), max_cycles, &mut Recorder { inner: &mut obs, st }),
                };
                st.suspend();
                self.run_ns += ran.elapsed().as_nanos() as u64;
                r
            }
        };
        (self.diags, self.first, self.error) = (obs.diags, obs.first, obs.error.take());
        r.map(|s| {
            self.ticks += s.ticks;
            self.last_cycle = s.last_cycle.or(self.last_cycle);
        })
    }

    /// The `E` and `W` lines, after the last window (`print_results`).
    pub fn finish(&mut self, words: &[(u8, u32, [u8; crate::geom::MAX_LANES])], tail: &str, trace: bool) {
        let mut obs = TraceObserver::resume(&mut self.out, trace, self.diags, self.first, self.error.take());
        // the end-of-run records (IQ-004, the parked queues) go to the recording too, as in a one-chip run
        match self.obs.as_mut() {
            Some(st) => self.machine.finish(&mut Recorder { inner: &mut obs, st }),
            None => self.machine.finish(&mut obs),
        }
        obs.finish_lanes(words, self.machine.geom().vlanes as usize);
        obs.tail(tail);
        (self.diags, self.first, self.error) = (obs.diags, obs.first, obs.error.take());
    }
}

/// How a multi-chip run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Every chip ran to its end and is quiescent: the chips still need [`ChipRun::finish`].
    Done,
    /// The smallest next event, `cycle`, is beyond `--max-cycles` (the traces so far are printed,
    /// then exit 3).
    OverBudget { cycle: u128 },
}

/// Runs the chips of one window; `Err` holds the failures in chip order.
type RunWindow<'a> = dyn FnMut(u128) -> Result<(), Vec<RunError>> + 'a;

/// Run the chips in lock-step windows of `window` cycles until none has anything left to do, on
/// `jobs` threads (`jobs` of 0 or 1: the caller's thread only).
/// `Err`: chips failed inside a window (their own `--max-cycles` stop or an internal fault; the
/// reference's chips run the window in parallel, so every chip that fails in it reports, and the
/// run exits with the first one's code and no output) or one is not quiescent at the end (exit 4).
/// A panic in a chip propagates to the caller, as it does on one thread.
// model: c_emulator/tsp_sim.c run_chips
pub fn run_chips<const L: usize>(
    chips: &mut [ChipRun<L>],
    window: u64,
    max_cycles: u64,
    trace: bool,
    jobs: usize,
) -> Result<Outcome, Vec<RunError>> {
    let threads = jobs.min(chips.len()).max(1);
    let cells: Vec<Mutex<&mut ChipRun<L>>> = chips.iter_mut().map(Mutex::new).collect();
    if threads == 1 {
        let mut run = |wend: u128| {
            let failed: Vec<RunError> =
                cells.iter().filter_map(|c| c.lock().unwrap().run_until(wend, max_cycles, trace).err()).collect();
            if failed.is_empty() {
                Ok(())
            } else {
                Err(failed)
            }
        };
        return drive(&cells, window, max_cycles, &mut run);
    }
    let pool = Pool {
        cells: &cells,
        dedicated: threads == cells.len(),
        wend: Mutex::new(0u128),
        next: AtomicUsize::new(0),
        stop: AtomicBool::new(false),
        start: Barrier::new(threads),
        done: Barrier::new(threads),
        errors: Mutex::new(Vec::new()),
        panic: Mutex::new(None),
        max_cycles,
        trace,
    };
    std::thread::scope(|s| {
        for id in 1..threads {
            let pool = &pool;
            s.spawn(move || loop {
                pool.start.wait();
                if pool.stop.load(Ordering::SeqCst) {
                    break;
                }
                pool.work(id);
                pool.done.wait();
            });
        }
        let r = catch_unwind(AssertUnwindSafe(|| drive(&cells, window, max_cycles, &mut |wend| pool.window(wend))));
        // release the workers, whatever happened
        pool.stop.store(true, Ordering::SeqCst);
        pool.start.wait();
        r.unwrap_or_else(|p| resume_unwind(p))
    })
}

/// The state shared by the threads of [`run_chips`].
struct Pool<'a, const L: usize> {
    cells: &'a [Mutex<&'a mut ChipRun<L>>],
    wend: Mutex<u128>,
    /// With one thread per chip, keep its working set on the same worker across windows.
    dedicated: bool,
    /// The next chip nobody has taken in this window.
    next: AtomicUsize,
    stop: AtomicBool,
    start: Barrier,
    done: Barrier,
    errors: Mutex<Vec<(usize, RunError)>>,
    /// The payload of the panic of the lowest chip index so far (the one a single thread would meet first).
    panic: Mutex<Option<(usize, Box<dyn Any + Send>)>>,
    max_cycles: u64,
    trace: bool,
}

impl<const L: usize> Pool<'_, L> {
    /// Take chips until none is left, and run the window on each.
    fn work(&self, worker: usize) {
        let wend = *self.wend.lock().unwrap();
        if self.dedicated {
            self.run_chip(worker, wend);
            return;
        }
        loop {
            let i = self.next.fetch_add(1, Ordering::SeqCst);
            if i >= self.cells.len() {
                break;
            }
            self.run_chip(i, wend);
        }
    }

    fn run_chip(&self, i: usize, wend: u128) {
        let mut chip = self.cells[i].lock().unwrap();
        match catch_unwind(AssertUnwindSafe(|| chip.run_until(wend, self.max_cycles, self.trace))) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => self.errors.lock().unwrap().push((i, e)),
            Err(p) => {
                let mut kept = self.panic.lock().unwrap();
                if kept.as_ref().is_none_or(|k| i < k.0) {
                    *kept = Some((i, p));
                }
            }
        }
    }

    /// One window on every thread; the failures in chip order.
    fn window(&self, wend: u128) -> Result<(), Vec<RunError>> {
        *self.wend.lock().unwrap() = wend;
        self.next.store(0, Ordering::SeqCst);
        self.start.wait();
        self.work(0);
        self.done.wait();
        // re-raised here, on the caller's thread; `run_chips` releases the workers first
        if let Some((_, p)) = self.panic.lock().unwrap().take() {
            resume_unwind(p);
        }
        let mut errors = std::mem::take(&mut *self.errors.lock().unwrap());
        errors.sort_by_key(|e| e.0);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.into_iter().map(|e| e.1).collect())
        }
    }
}

/// The windows: run one on every chip, route what the chips sent, find the next.
fn drive<const L: usize>(
    chips: &[Mutex<&mut ChipRun<L>>],
    window: u64,
    max_cycles: u64,
    run_window: &mut RunWindow,
) -> Result<Outcome, Vec<RunError>> {
    let k = chips.len();
    let mut wend = 0u128;
    loop {
        run_window(wend)?;
        // route the vectors sent in the window, in chip order and sending order
        let mut route: Vec<Vec<VecMsg>> = vec![Vec::new(); k];
        for c in chips {
            for m in c.lock().unwrap().links.lock().unwrap().take_outbound() {
                let to = m.chip;
                route[to].push(m);
            }
        }
        // Deliver first, then each chip's next time (a delivered vector may bring an
        // overrun to report); every chip has ticked every cycle below wend
        for (c, msgs) in chips.iter().zip(route) {
            let chip = c.lock().unwrap();
            let mut links = chip.links.lock().unwrap();
            for m in msgs {
                links.deliver(m);
            }
        }
        let g = chips.iter().filter_map(|c| c.lock().unwrap().machine.report_time()).min();
        match g {
            None => break,
            Some(g) if g > max_cycles.into() => return Ok(Outcome::OverBudget { cycle: g }),
            Some(g) => wend = g + u128::from(window),
        }
    }
    for c in chips {
        if !c.lock().unwrap().machine.quiescent() {
            return Err(vec![RunError::Internal("model not quiescent with no event left".into())]);
        }
    }
    Ok(Outcome::Done)
}
