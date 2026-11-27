//! `tsp-iss bench [--program F] [--repeat N]`: run the checked-in benchmark programs
//! (`benches/programs/`, written by `gen_bench.py`) on profile `ref` without a trace, and print one
//! line per program:
//!
//! ```text
//! <name> repeat=<N> cycles=<C> ticks=<T> seconds=<S> cycles_per_second=<R> dispatches=<D> ns_per_dispatch=<P> ns_per_visited_tick=<V>
//! ```
//!
//! With `--elements` each line ends with `elements=<E> ns_per_element=<L>`: `E` is the instructions executed, counted once
//! more by an untimed run with the activity counters on (a loop or a Repeat executes its body once per element with one
//! dispatch, so `ns_per_dispatch` says little for them and `L = seconds / E` is the rate to compare them by; the timed
//! runs count nothing).
//!
//! `cycles` is the simulated cycles (the last cycle ticked, plus one: idle cycles count), `ticks` the
//! cycles ticked, `seconds` the best of the `N` runs (machine construction and program load included,
//! as the wall time of `tsp_sim`), and `R = cycles / seconds`; `D` is the parcels dispatched, `P = seconds / D` and
//! `V = seconds / ticks` in nanoseconds: the two rates that do not depend on how many idle cycles a program has, so
//! that programs of different length can be compared (`0` when `D` is 0). `--program F` runs the Q-line text file
//! `F` instead of the checked-in set.
//!
//! Every run goes through [`execute`] with the statistics on (a `StatsObserver` counts dispatches, waits
//! and fetches), so the numbers include the cost of the stats path; a run without `--stats` is a little faster.
use super::{check_geom, fail, load_named};
use crate::harness::cli::Options;
use crate::harness::{execute, input, strerror, RunStats};
use std::ffi::OsString;
use std::io::Write;
use std::time::Instant;

const CMD: &str = "bench";
const USAGE: &str = "usage: tsp-iss bench [--program F] [--repeat N] [--matrix-threads N|auto] [--elements]";

/// The checked-in programs, in report order. `b0`..`b4` and `idle` are the design's (§4.10); the others are the coverage
/// benchmark programs: dense MXM, a valid elementwise VXM, transcendentals and barriers, each with zero
/// diagnostics (`gen_bench.py`).
const PROGRAMS: [(&str, &str); 11] = [
    ("b0", include_str!("../../benches/programs/b0.txt")),
    ("b1", include_str!("../../benches/programs/b1.txt")),
    ("b2", include_str!("../../benches/programs/b2.txt")),
    ("b3", include_str!("../../benches/programs/b3.txt")),
    ("b4", include_str!("../../benches/programs/b4.txt")),
    ("idle", include_str!("../../benches/programs/idle.txt")),
    ("dense_f16", include_str!("../../benches/programs/dense_f16.txt")),
    ("dense_i8", include_str!("../../benches/programs/dense_i8.txt")),
    ("vxm_valid", include_str!("../../benches/programs/vxm_valid.txt")),
    ("transc", include_str!("../../benches/programs/transc.txt")),
    ("syncnotify", include_str!("../../benches/programs/syncnotify.txt")),
];

/// The cycle limit of a benchmark run (the longest program needs 10^8 + a few).
const MAX_CYCLES: u64 = 4_000_000_000;

pub fn bench(args: &[OsString], stdout: &mut dyn Write, stderr: &mut dyn Write) -> i32 {
    let mut program: Option<OsString> = None;
    let mut repeat = 1usize;
    let mut threads: Option<super::threads::MatrixThreads> = None;
    let mut elements = false;
    let mut i = 0;
    while i < args.len() {
        let value = args.get(i + 1);
        match args[i].to_str() {
            Some("--elements") => {
                elements = true;
                i += 1;
                continue;
            }
            Some("--program") if value.is_some() => program = value.cloned(),
            Some("--repeat") if value.is_some() => match value.and_then(|v| v.to_str()?.parse::<usize>().ok()) {
                Some(n) if n >= 1 => repeat = n,
                _ => return fail(stderr, CMD, USAGE),
            },
            Some("--matrix-threads") if value.is_some() => {
                match value.and_then(|v| super::threads::MatrixThreads::parse(v.to_str()?)) {
                    Some(t) => threads = Some(t),
                    None => return fail(stderr, CMD, USAGE),
                }
            }
            _ => return fail(stderr, CMD, USAGE),
        }
        i += 2;
    }
    // nothing reads the activity counters here
    crate::slices::activity::set_default_enabled(false);
    let profile = match load_named("ref") {
        Ok(p) => p,
        Err(m) => return fail(stderr, CMD, &m),
    };
    let geom = match check_geom(&profile, "ref") {
        Ok(g) => g,
        Err(m) => return fail(stderr, CMD, &m),
    };
    let set: Vec<(String, String)> = match &program {
        None => PROGRAMS.iter().map(|(n, t)| (n.to_string(), t.to_string())).collect(),
        Some(f) => match std::fs::read_to_string(f) {
            Ok(t) => {
                let name = std::path::Path::new(f).file_stem().map_or_else(String::new, |s| s.to_string_lossy().into());
                vec![(name, t)]
            }
            Err(e) => return fail(stderr, CMD, &format!("{}: {}", f.to_string_lossy(), strerror(&e))),
        },
    };
    if let Some(t) = threads {
        use super::threads::{auto_width, MatrixThreads};
        let (width, auto) = match t {
            MatrixThreads::Width(n) => (n, false),
            MatrixThreads::Auto => {
                // the pool serves every program of the set: wide when one of them holds an ABC that pays for a split
                let mut all = Vec::new();
                for (name, src) in &set {
                    match input::parse(src.as_bytes(), input::Limits { geom: &geom, chips: 1, topo: None }) {
                        Ok(p) => all.extend(p.chips.into_iter().map(|c| c.images)),
                        Err(e) => return fail(stderr, CMD, &format!("{name}: {e}")),
                    }
                }
                let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
                (auto_width(&profile, &geom, &all.iter().collect::<Vec<_>>(), cores, 1), true)
            }
        };
        let set = if auto {
            crate::slices::pool::Pool::configure_adaptive(width)
        } else {
            crate::slices::pool::Pool::configure(width)
        };
        if let Err(have) = set {
            let _ = writeln!(stderr, "tsp-iss: {CMD}: --matrix-threads: the pool has {have} threads already");
        }
    }
    let mut opts = Options::new(OsString::from("ref"));
    opts.max_cycles = MAX_CYCLES;
    for (name, src) in &set {
        let limits = input::Limits { geom: &geom, chips: 1, topo: None };
        let prog = match input::parse(src.as_bytes(), limits) {
            Ok(p) => p,
            Err(e) => return fail(stderr, CMD, &format!("{name}: {e}")),
        };
        let mut best = f64::INFINITY;
        let mut stats = RunStats::default();
        for _ in 0..repeat {
            stats = RunStats::default();
            let mut err = Vec::new();
            let t = Instant::now();
            let code = execute(
                &profile,
                &geom,
                &prog,
                &opts,
                opts.link_mode().expect("no link options"),
                Some(&mut stats),
                &mut std::io::sink(),
                &mut err,
            );
            let s = t.elapsed().as_secs_f64();
            if code != 0 {
                let _ = stderr.write_all(&err);
                return fail(stderr, CMD, &format!("{name}: the run failed (exit {code})"));
            }
            best = best.min(s);
        }
        let counted = if elements {
            // the executed elements: one more run, untimed, with the counters on
            crate::slices::activity::set_default_enabled(true);
            let mut counted = RunStats::default();
            let code = execute(
                &profile,
                &geom,
                &prog,
                &opts,
                opts.link_mode().expect("no link options"),
                Some(&mut counted),
                &mut std::io::sink(),
                &mut Vec::new(),
            );
            crate::slices::activity::set_default_enabled(false);
            if code != 0 {
                return fail(stderr, CMD, &format!("{name}: the counting run failed (exit {code})"));
            }
            counted.chips[0].activity.as_ref().map(|a| a.issued.iter().sum::<u64>())
        } else {
            None
        };
        let chip = &stats.chips[0];
        let cycles = chip.last_cycle.map_or(0, |c| c + 1);
        let rate = if cycles == 0 { 0.0 } else { cycles as f64 / best };
        let dispatches: u64 = chip.counts.dispatches.iter().sum();
        let per = |n: u64| if n == 0 { 0.0 } else { best * 1e9 / n as f64 };
        let tail = counted.map_or(String::new(), |e| format!(" elements={e} ns_per_element={:.2}", per(e)));
        let _ = writeln!(
            stdout,
            "{name} repeat={repeat} cycles={cycles} ticks={} seconds={best:.6} cycles_per_second={rate:.0} \
             dispatches={dispatches} ns_per_dispatch={:.1} ns_per_visited_tick={:.1}{tail}",
            chip.ticks,
            per(dispatches),
            per(chip.ticks)
        );
    }
    0
}
