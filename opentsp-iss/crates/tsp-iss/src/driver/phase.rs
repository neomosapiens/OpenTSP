//! Per-phase time counters of the tick loop, under the cargo feature `phases` (off by default; without it every
//! function here is empty and the optimiser removes the calls). AArch64 uses its architectural virtual counter. Other hosts use
//! elapsed monotonic nanoseconds. The report includes the clock frequency and the
//! calibration (the cost of an empty `begin`/`end` pair). Phases are exclusive: time spent in unit semantics
//! (`sem_*`) inside a phase is taken out of it and counted under `semantics`.

#[cfg(feature = "phases")]
mod on {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    pub const NAMES: [&str; N] = [
        "enqueue loop",
        "retire+commit_due",
        "execute_due (excl. semantics)",
        "control",
        "wake reset, cal_take, visit set",
        "visit: skip test (wake > c)",
        "visit: head_brief/earliest/head",
        "visit: repeat iteration",
        "visit: parcel_held, config end",
        "visit: pop, dispatch notes",
        "visit: dispatch() (excl. semantics)",
        "visit: wake_of, set_wake, sets",
        "unit_state_flush",
        "model_next_event",
        "try_tick total (contains 0-12; not summed)",
        "semantics (execute)",
        "srf_write (commit)",
        "Iq::parse (inside enqueue)",
        "  of it: decode_text (inclusive)",
        "  of it: classify_items (inclusive)",
        "(calibration)",
    ];
    pub const N: usize = 21;
    pub const SEM: usize = 15;
    pub const SRF: usize = 16;
    pub const PARSE: usize = 17;
    pub const DEC: usize = 18;
    pub const CLS: usize = 19;
    static T: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
    static C: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
    // A phase excludes nested work from its own thread, never another chip's.
    // T/C remain atomic process-wide aggregates; marks stay on their creating thread.
    thread_local! {
        static SEMT: Cell<u64> = const { Cell::new(0) };
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub fn now() -> u64 {
        let t: u64;
        unsafe { core::arch::asm!("isb", "mrs {t}, cntvct_el0", t = out(reg) t, options(nomem, nostack)) };
        t
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[inline(always)]
    pub fn now() -> u64 {
        use std::{sync::OnceLock, time::Instant};
        static ORIGIN: OnceLock<Instant> = OnceLock::new();
        ORIGIN.get_or_init(Instant::now).elapsed().as_nanos() as u64
    }

    /// An instrumented interval belongs to its creating thread: semantic
    /// exclusion is thread-local. The zero-sized marker prevents Send/Sync
    /// without allocation or a runtime thread-identity check.
    #[derive(Clone, Copy)]
    pub struct Mark(u64, u64, std::marker::PhantomData<std::rc::Rc<()>>);

    #[inline(always)]
    pub fn begin() -> Mark {
        Mark(now(), SEMT.with(Cell::get), std::marker::PhantomData)
    }

    #[inline(always)]
    pub fn end(i: usize, m: Mark) {
        let t = now() - m.0;
        let s = SEMT.with(Cell::get) - m.1;
        T[i].fetch_add(t.saturating_sub(s), Relaxed);
        C[i].fetch_add(1, Relaxed);
    }

    /// Time inside the unit semantics, counted under `SEM` or `SRF` and left out of the enclosing phases.
    #[inline(always)]
    pub fn sem_end(i: usize, m: Mark) {
        let t = now() - m.0;
        T[i].fetch_add(t, Relaxed);
        C[i].fetch_add(1, Relaxed);
        SEMT.with(|total| total.set(total.get().wrapping_add(t)));
    }

    /// Successful run-loop ticks represented by these process-cumulative counters.
    /// Phase 14 ends exactly after RunSummary.ticks advances, including idle ticks
    /// and ticks before an IRQ stop, but excluding failed ticks/arrival-only events.
    pub fn completed_ticks() -> u64 {
        C[14].load(Relaxed)
    }

    /// Print a cumulative snapshot. `ticks` must cover the same cumulative work;
    /// normal Machine::run uses completed_ticks(), including prior run_until work.
    /// Counters aggregate all threads; nested semantic subtraction is thread-local.
    /// Snapshots during active workers are approximate, not atomic or per-machine.
    pub fn report(ticks: u64) {
        #[cfg(target_arch = "aarch64")]
        let freq: u64 = {
            let f: u64;
            unsafe { core::arch::asm!("mrs {f}, cntfrq_el0", f = out(reg) f, options(nomem, nostack)) };
            f
        };
        #[cfg(not(target_arch = "aarch64"))]
        let freq = 1_000_000_000u64;
        let n = 1_000_000u64;
        let t0 = now();
        for _ in 0..n {
            let m = begin();
            end(N - 1, m);
        }
        let cal = (now() - t0) as f64 / freq as f64 / n as f64 * 1e9;
        // Both calibration time and calls accumulate across reports.
        let rec = T[N - 1].load(Relaxed) as f64 / freq as f64 / C[N - 1].load(Relaxed) as f64 * 1e9;
        let ns = |t: u64| t as f64 / freq as f64 * 1e9;
        eprintln!("phases: counter {freq} Hz, {ticks} ticks, empty begin/end pair {cal:.1} ns, of which recorded in a phase {rec:.1} ns (subtracted per call below)");
        let per_tick = |ns: f64| if ticks == 0 { 0.0 } else { ns / 1e3 / ticks as f64 };
        let mut sum = 0.0;
        for i in 0..N - 1 {
            let c = C[i].load(Relaxed);
            let t = ns(T[i].load(Relaxed)) - rec * c as f64;
            if !matches!(i, 14 | 18 | 19) {
                sum += t;
            }
            eprintln!(
                "phase {:>2} {:<40} {:>9.1} ms {:>8.4} us/tick {:>9} calls {:>7.1} ns/call",
                i,
                NAMES[i],
                t / 1e6,
                per_tick(t),
                c,
                if c > 0 { t / c as f64 } else { 0.0 }
            );
        }
        eprintln!("phase sum {:.1} ms, {:.4} us/tick", sum / 1e6, per_tick(sum));
    }
}

#[cfg(feature = "phases")]
pub use on::{begin, completed_ticks, end, report, sem_end, Mark, CLS, DEC, PARSE, SEM, SRF};

#[cfg(not(feature = "phases"))]
mod off {
    #[derive(Clone, Copy)]
    pub struct Mark;
    pub const SEM: usize = 0;
    pub const SRF: usize = 0;
    pub const PARSE: usize = 0;
    pub const DEC: usize = 0;
    pub const CLS: usize = 0;
    #[inline(always)]
    pub fn begin() -> Mark {
        Mark
    }
    #[inline(always)]
    pub fn end(_: usize, _: Mark) {}
    #[inline(always)]
    pub fn sem_end(_: usize, _: Mark) {}
    pub fn completed_ticks() -> u64 {
        0
    }
    pub fn report(_: u64) {}
}

#[cfg(not(feature = "phases"))]
pub use off::{begin, completed_ticks, end, report, sem_end, Mark, CLS, DEC, PARSE, SEM, SRF};
