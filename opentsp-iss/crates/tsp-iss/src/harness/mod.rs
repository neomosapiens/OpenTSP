//! The `tsp_sim`-compatible harness: options (`cli`), the program on
//! stdin (`input`), the output (`output`), the link wiring (`links`), and the order of checks of
//! `tsp_sim.c` `main`:
//!
//! arguments → schema check of the configuration → model check (`timing_config_error`) →
//! `link_latency_min <= link_latency <= link_latency_max` → `vlanes <= 512` → the link options
//! exclude one another → input → link setup → run.
//!
//! Exit codes: 0 success; 2 input, option or configuration error; 3 `--max-cycles` exceeded;
//! 4 internal error (a Rust panic is caught at the top level). Every message on stderr starts
//! with `tsp-iss:`; stdout is byte-identical to `tsp_sim`.
pub mod chips;
pub mod cli;
pub mod cnum;
pub mod input;
pub mod links;
pub mod obs;
pub mod output;
pub mod stats;
pub mod topology;

use crate::geom::{Geom, MAX_LANES, NARROW_LANES};
use crate::machine::{Machine, RunError};
use cli::CliError;
use ots_profile::Profile;
use output::TraceObserver;
use std::ffi::{OsStr, OsString};
use std::io::{self, Read, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::Arc;

/// The program name in messages.
pub const PROG: &str = "tsp-iss";

/// Read a configuration file and validate it exactly as `tsp_config_load` and
/// `tsp_config_model_check` do. The error is the text after `tsp-iss: config <path>: `.
pub fn load_config(path: &Path) -> Result<Profile, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("cannot open: {}", strerror(&e)))?;
    // tsp_sim refuses files of 2^25 bytes or more
    const LIMIT: u64 = 1 << 25;
    let mut bytes = Vec::new();
    Read::by_ref(&mut f).take(LIMIT).read_to_end(&mut bytes).map_err(|_| "read error".to_string())?;
    if bytes.len() as u64 >= LIMIT {
        return Err("file too large".into());
    }
    if bytes.contains(&0) {
        return Err("NUL byte in file".into());
    }
    Profile::from_validated_bytes(&bytes).map_err(|e| e.message)
}

/// The C library's message for an I/O error (Rust appends ` (os error N)`).
pub(crate) fn strerror(e: &io::Error) -> String {
    let s = e.to_string();
    match s.rfind(" (os error ") {
        Some(i) => s[..i].to_string(),
        None => s,
    }
}

pub(crate) fn say(err: &mut dyn Write, msg: &str) {
    let _ = writeln!(err, "{PROG}: {msg}");
}

/// The whole program: `args` are the arguments after the program name. Returns the exit code.
pub fn main_with(args: &[OsString], stdin: &mut dyn Read, stdout: &mut dyn Write, stderr: &mut dyn Write) -> i32 {
    if args.len() == 1 && (args[0] == "--help" || args[0] == "help") {
        // the line `tsp_sim --help` prints, then the subcommands of this binary
        let help = cli::USAGE.replace("[--lint]", "[--observe] [--lint]");
        if let Err(e) = writeln!(stdout, "{help}\n\n{}", cli::SUBCOMMAND_HELP) {
            say(stderr, &format!("cannot write standard output: {}", strerror(&e)));
            return 2;
        }
        return 0;
    }
    if cli::is_subcommand(args) {
        return crate::info::native(args, stdout, stderr);
    }
    compat(args, stdin, stdout, stderr)
}

/// Read and check the topology file of `--topology`; it gives the number of chips (`tsp_sim.c` `main`).
pub(crate) fn load_topology(opts: &mut cli::Options, geom: &Geom) -> Result<Option<Arc<topology::Topology>>, String> {
    let Some(p) = &opts.topology else { return Ok(None) };
    if opts.chips_given {
        return Err("--topology gives the number of chips: do not combine it with --chips".into());
    }
    let path = Path::new(p);
    let text = std::fs::read(path).map_err(|e| format!("--topology {}: {}", path.display(), strerror(&e)))?;
    let t = topology::Topology::parse(&text, geom).map_err(|m| format!("topology {}: {m}", path.display()))?;
    opts.chips = t.chips as u32;
    Ok(Some(Arc::new(t)))
}

fn compat(args: &[OsString], stdin: &mut dyn Read, stdout: &mut dyn Write, stderr: &mut dyn Write) -> i32 {
    let mut opts = match cli::parse(args) {
        Ok(o) => o,
        Err(e) => return refuse_args(&e, stderr),
    };
    let cfg_path = Path::new(&opts.config);
    let profile = match load_config(cfg_path) {
        Ok(p) => p,
        Err(m) => {
            say(stderr, &format!("config {}: {m}", cfg_path.display()));
            return 2;
        }
    };
    let geom = Geom::new(&profile);
    if let Err(m) = geom.check_link_latency() {
        say(stderr, &format!("config {}: {m}", cfg_path.display()));
        return 2;
    }
    if let Err(m) = geom.check_lanes() {
        say(stderr, &format!("config {}: {m}", cfg_path.display()));
        return 2;
    }
    let topo = match load_topology(&mut opts, &geom) {
        Ok(t) => t,
        Err(m) => {
            say(stderr, &m);
            return 2;
        }
    };
    let mode = match opts.link_mode() {
        Ok(m) => m,
        Err(e) => return refuse_args(&e, stderr),
    };
    if let Err(m) = check_on_irq(&opts) {
        say(stderr, m);
        return 2;
    }
    let mut raw = Vec::new();
    if let Err(e) = stdin.read_to_end(&mut raw) {
        say(stderr, &format!("cannot read standard input: {}", strerror(&e)));
        return 2;
    }
    let prog = match input::parse(&raw, input::Limits { geom: &geom, chips: opts.chips, topo: topo.as_ref() }) {
        Ok(p) => p,
        Err(e) => {
            say(stderr, &e.to_string());
            return 2;
        }
    };
    execute(&profile, &geom, &prog, &opts, mode, None, stdout, stderr)
}

/// `--on-irq stop` and `replay` need one chip (`tsp_sim.c` `main`).
// model: c_emulator/tsp_sim.c main
fn check_on_irq(opts: &cli::Options) -> Result<(), &'static str> {
    if opts.on_irq != cli::OnIrq::Continue && opts.chips > 1 {
        return Err("--on-irq stop and replay need --chips 1");
    }
    Ok(())
}

/// What a run did, per chip (`tsp-iss run --stats`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunStats {
    pub chips: Vec<ChipStats>,
    /// Wall time of [`execute`] from the machines' construction to the end of the run.
    pub wall_seconds: f64,
    /// What `--run-profile-json` and the `--obs-*` flags recorded, per chip.
    pub obs: Vec<ObsChip>,
}

/// What the recording of one chip's run holds, and the wall time of its parts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObsChip {
    pub state: obs::ObsState,
    /// Building the machine and loading its images, in nanoseconds.
    pub setup_ns: u64,
    /// The run loop, in nanoseconds.
    pub run_ns: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChipStats {
    /// Dispatches, wait cycles, fetch bytes and diagnostics per code.
    pub counts: stats::Counters,
    /// What each unit class did (`slices::activity`).
    pub activity: Option<crate::slices::activity::Activity>,
    /// Per queue that ever held text: queue id, fewest units held after a dispatch, most held.
    pub iq: Vec<(u16, usize, usize)>,
    /// Cycles ticked (idle cycles are skipped).
    pub ticks: u64,
    /// The last cycle ticked, if any.
    pub last_cycle: Option<u64>,
    /// Diagnostics reported.
    pub diagnostics: u64,
    /// The first diagnostic: its code and cycle.
    pub first: Option<(ots_isa::Code, u64)>,
}

/// Everything after the input: wire the links, build the machines of the program, run them and print
/// (the tail of `tsp_sim.c` `main`). Shared by the compatible mode and `tsp-iss run`. `prog` must
/// have been checked against `geom` (the input parser and the loaders do); `stats`, when given, is
/// filled when the run completes.
#[allow(clippy::too_many_arguments)] // the state a chip run shares with its caller, one argument each
pub fn execute(
    profile: &Profile,
    geom: &Geom,
    prog: &input::Program,
    opts: &cli::Options,
    mode: cli::LinkMode,
    mut stats: Option<&mut RunStats>,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> i32 {
    if let Err(m) = check_on_irq(opts) {
        say(stderr, m);
        return 2;
    }
    let (code, irq) = execute_once(profile, geom, prog, opts, mode.clone(), stats.as_deref_mut(), stdout, stderr);
    if opts.on_irq != cli::OnIrq::Replay || code != 0 || !irq {
        return code;
    }
    // --on-irq replay (c_emulator/tsp_sim.c replay_run): the run raised an error interrupt; run again from reset with the
    // faults marked `p` only. The first run's output is followed by `T replay start`, the second run's and the verdict.
    let _ = stdout.write_all(b"T replay start\n");
    let mut again = prog.clone();
    again.host.retain(|h| h.persist || !is_fault(&h.kind));
    let keep: Vec<bool> = again.link_persist.clone();
    let mut k = keep.iter();
    again.link_faults.retain(|_| *k.next().unwrap());
    again.link_persist.retain(|&p| p);
    let (code, irq) = execute_once(profile, geom, &again, opts, mode, stats, stdout, stderr);
    if code != 0 {
        return code;
    }
    let _ = stdout.write_all(if irq { b"T replay persistent\n" } else { b"T replay resolved\n" });
    let _ = stdout.flush();
    if irq {
        5
    } else {
        0
    }
}

/// An `F` line (`F sram`, `F srf`, `F iq`) as opposed to a line of the host's script.
fn is_fault(k: &crate::host::HostKind) -> bool {
    use crate::host::HostKind::*;
    matches!(k, FlipSram { .. } | FlipSrf { .. } | FlipIq { .. } | FlipMxm { .. })
}

/// One run of the program: the exit code and whether an error interrupt reached the host (one chip only). A profile of at most
/// [`NARROW_LANES`] lanes runs on machines whose per-cycle records are that wide, a wider one on the widest.
#[allow(clippy::too_many_arguments)]
fn execute_once(
    profile: &Profile,
    geom: &Geom,
    prog: &input::Program,
    opts: &cli::Options,
    mode: cli::LinkMode,
    stats: Option<&mut RunStats>,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> (i32, bool) {
    if geom.vlanes as usize <= NARROW_LANES {
        execute_lanes::<NARROW_LANES>(profile, geom, prog, opts, mode, stats, stdout, stderr)
    } else {
        execute_lanes::<MAX_LANES>(profile, geom, prog, opts, mode, stats, stdout, stderr)
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_lanes<const L: usize>(
    profile: &Profile,
    geom: &Geom,
    prog: &input::Program,
    opts: &cli::Options,
    mode: cli::LinkMode,
    mut stats: Option<&mut RunStats>,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> (i32, bool) {
    let started = std::time::Instant::now();
    let plan = match match &prog.topo {
        Some(t) => links::LinkPlan::with_topology(mode, geom, t.clone()),
        None => links::LinkPlan::new(mode, geom, opts.chips as usize),
    } {
        Ok(p) => Arc::new(p),
        Err(m) => {
            say(stderr, &m);
            return (2, false);
        }
    };
    let k = opts.chips as usize;
    let mut ctl = host_events(prog);
    let single_chip = k == 1 && prog.chips.len() == 1;
    let mut chips: Vec<chips::ChipRun<L>> = Vec::with_capacity(if single_chip { 0 } else { k });
    for (c, chip) in prog.chips.iter().enumerate() {
        let built = std::time::Instant::now();
        let start = prog.starts.get(c).copied().unwrap_or(0);
        let host = std::mem::take(&mut ctl.host[c]);
        let faults = prog.link_faults.iter().filter(|f| f.chip as usize == c).copied().collect();
        match build_machine(profile, geom, &plan, c, chip, host, start, faults) {
            Ok(mut m) => {
                if opts.observe {
                    m.machine.iq.retag_images();
                    m.machine.iq.observe_sites = true;
                }
                m.machine.set_ctl(std::mem::take(&mut ctl.pkts[c]));
                m.machine.set_tick_every_cycle(opts.tick_every_cycle);
                if opts.obs.any() {
                    m.obs = Some(obs::ObsState::new(&opts.obs));
                    m.setup_ns = built.elapsed().as_nanos() as u64;
                }
                if single_chip {
                    let code = run_one(&mut m, prog, opts, stats.as_deref_mut(), stdout, stderr);
                    if let Some(st) = stats {
                        st.wall_seconds = started.elapsed().as_secs_f64();
                    }
                    return (code, m.machine.error_irq_seen());
                }
                chips.push(m);
            }
            Err(m) => {
                say(stderr, &m);
                return (2, false);
            }
        }
    }
    if k > 1 && plan.window() == 0 {
        say(stderr, "--chips needs link latencies >= 1");
        return (2, false);
    }
    let mut window = plan.window();
    if k > 1 && opts.window > 0 {
        // --window N: a shorter window, for testing; the results do not depend on it
        window = window.min(opts.window);
    }
    let (code, irq) = if k == 1 {
        let c = run_one(&mut chips[0], prog, opts, stats.as_deref_mut(), stdout, stderr);
        (c, chips[0].machine.error_irq_seen())
    } else {
        (run_many(chips, prog, opts, window, stats.as_deref_mut(), stdout, stderr), false)
    };
    if let Some(st) = stats {
        st.wall_seconds = started.elapsed().as_secs_f64();
    }
    (code, irq)
}

/// One chip: it prints as it runs (with every link looped back). The observer is fixed by `--trace`, so a run
/// without it carries no per-event test of the flag.
fn run_one<const L: usize>(
    chip: &mut chips::ChipRun<L>,
    prog: &input::Program,
    opts: &cli::Options,
    stats: Option<&mut RunStats>,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> i32 {
    if opts.trace {
        run_one_with(chip, prog, opts, stats, stderr, TraceObserver::traced(stdout))
    } else {
        run_one_with(chip, prog, opts, stats, stderr, TraceObserver::untraced(stdout))
    }
}

fn run_one_with<const L: usize, const MODE: u8>(
    chip: &mut chips::ChipRun<L>,
    prog: &input::Program,
    opts: &cli::Options,
    stats: Option<&mut RunStats>,
    stderr: &mut dyn Write,
    mut obs: TraceObserver<'_, MODE>,
) -> i32 {
    let mut code = 0;
    obs.lint = opts.lint;
    let mut counts = stats.as_ref().map(|_| stats::Counters::default());
    replay_begin(&mut chip.machine, opts, stderr);
    chip.machine.set_stop_on_irq(opts.on_irq == cli::OnIrq::Stop);
    let ran = std::time::Instant::now();
    let res = match (counts.as_mut(), chip.obs.as_mut()) {
        (Some(counts), None) => {
            chip.machine.run(opts.max_cycles, &mut stats::StatsObserver { inner: &mut obs, counts })
        }
        (None, None) => chip.machine.run(opts.max_cycles, &mut obs),
        (counts, Some(st)) => {
            st.resume();
            let r = match counts {
                Some(counts) => chip.machine.run(
                    opts.max_cycles,
                    &mut obs::Recorder { inner: &mut stats::StatsObserver { inner: &mut obs, counts }, st },
                ),
                None => chip.machine.run(opts.max_cycles, &mut obs::Recorder { inner: &mut obs, st }),
            };
            st.suspend();
            chip.run_ns = ran.elapsed().as_nanos() as u64;
            r
        }
    };
    // `tsp_sim` stops (exit 5) before its end-of-run check: a stopped run never reaches the unused `F link` rule
    let unused = if res.is_ok() && !chip.machine.stopped_at_irq() { unused_link_fault(chip, 0) } else { None };
    if res.is_ok() && unused.is_none() {
        obs.finish_lanes(&dump_words(chip, prog, 0), chip.machine.geom().vlanes as usize);
        obs.tail(&host_tail(&chip.machine, prog, 0));
        if chip.machine.stopped_at_irq() {
            code = 5; // --on-irq stop: the results as they stand, exit status 5
        } else {
            replay_end(&mut chip.machine, opts, stderr);
        }
    }
    if let (Ok(sum), Some(st)) = (&res, stats) {
        st.chips = vec![ChipStats {
            counts: counts.unwrap_or_default(),
            activity: Some(chip.machine.activity.clone()),
            iq: stats::iq_levels(&chip.machine),
            ticks: sum.ticks,
            last_cycle: sum.last_cycle,
            diagnostics: obs.diags,
            first: obs.first,
        }];
        st.obs = chip
            .obs
            .take()
            .map(|mut state| {
                state.finish();
                ObsChip { state, setup_ns: chip.setup_ns, run_ns: chip.run_ns }
            })
            .into_iter()
            .collect();
    }
    if let Some(e) = obs.error.take() {
        say(stderr, &format!("cannot write standard output: {}", strerror(&e)));
        code = 4;
    }
    let _ = obs.flush();
    drop(obs);
    match res {
        Err(e @ RunError::MaxCycles { .. }) => {
            say(stderr, &e.to_string());
            3
        }
        Err(e @ RunError::Input(_)) => {
            say(stderr, &e.to_string());
            2
        }
        Err(e) => {
            say(stderr, &e.to_string());
            4
        }
        Ok(_) => match unused {
            Some(m) => {
                say(stderr, &m);
                2
            }
            None => code,
        },
    }
}

/// An `F link` line of chip `c` that marked no vector is an input error (`finish_check`).
// model: c_emulator/tsp_sim.c finish_check
fn unused_link_fault<const L: usize>(chip: &chips::ChipRun<L>, c: usize) -> Option<String> {
    let (link, cycle) = chip.links.lock().unwrap().unused_fault()?;
    Some(format!("F link: no vector is sent on link {link} in cycle {cycle:x} (chip {c})"))
}

/// `--replay-runs`: set the machine up before its run; a schedule that does not fit runs
/// the normal driver, with a line on stderr.
fn replay_begin<const L: usize>(m: &mut Machine<L>, opts: &cli::Options, stderr: &mut dyn Write) {
    use crate::driver::replay::Began;
    use cli::ReplayOpt;
    let refused =
        |stderr: &mut dyn Write, why: String| say(stderr, &format!("replay: {why}: the run takes the normal driver"));
    match &opts.replay {
        ReplayOpt::None => {}
        ReplayOpt::Session => {
            if m.replay_session_begin() == Began::Normal {
                let why = if crate::driver::replay::ENABLED {
                    "this run is not recordable"
                } else {
                    "this build has no schedule replay (build with `--features replay`)"
                };
                refused(stderr, why.into());
            }
        }
    }
}

/// After a run that finished: write the recorded schedule, or keep it in the process.
fn replay_end<const L: usize>(m: &mut Machine<L>, opts: &cli::Options, stderr: &mut dyn Write) {
    use cli::ReplayOpt;
    match &opts.replay {
        ReplayOpt::Session => {
            say(stderr, &format!("replay: this run was {}", mode_word(m)));
            m.replay_session_end();
        }
        ReplayOpt::None => {}
    }
}

fn mode_word<const L: usize>(m: &Machine<L>) -> &'static str {
    use crate::driver::replay::Mode;
    match m.replay_mode() {
        Mode::Off => "normal (nothing recorded or replayed)",
        Mode::Record => "recorded",
        Mode::Replay => "replayed",
        Mode::Check => "replayed under TSP_ISS_REPLAY_CHECK (the normal driver compared with the schedule per tick)",
    }
}

/// What the host's script and the `P` lines give the chips: per chip its host events and its `P` lines.
struct CtlSetup {
    host: Vec<Vec<crate::host::HostEvent>>,
    pkts: Vec<Vec<crate::host::PktEvent>>,
}

/// The host's script split among the chips: the `H` lines in input order, each for the chip it acts on (every chip has
/// its own host interface); and the `P` lines, each for the chip whose control unit sends it.
// model: c_emulator/tsp_sim.c host_setup, pk_setup
fn host_events(prog: &input::Program) -> CtlSetup {
    use crate::host::{HostEvent, PktEvent};
    let n = prog.chips.len();
    let mut host: Vec<Vec<HostEvent>> = vec![Vec::new(); n];
    for h in &prog.host {
        host[h.chip as usize].push(HostEvent { cycle: h.cycle, kind: h.kind.clone() });
    }
    let mut pkts: Vec<Vec<PktEvent>> = vec![Vec::new(); n];
    for p in &prog.pkts {
        pkts[p.chip as usize].push(PktEvent { cycle: p.cycle, id: p.id, link: p.link, kind: p.kind });
    }
    CtlSetup { host, pkts }
}

/// The `Y` lines (the vectors PciSend gave the host) and the `X` lines (the `R` lines) of chip `c`, after its `W` lines.
// model: c_emulator/tsp_sim.c print_results
pub(crate) fn host_tail<const L: usize>(m: &Machine<L>, prog: &input::Program, c: usize) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let vl = m.geom().vlanes as usize;
    for (cycle, b) in &m.host().egress {
        let _ = write!(s, "Y {cycle}");
        output::append_hex(&mut s, &b[..vl], true);
        s.push('\n');
    }
    for (_, a) in prog.rcsr.iter().filter(|r| r.0 as usize == c) {
        let _ = writeln!(s, "X {a:03x} {:x}", m.csr_read(*a));
    }
    s
}

/// The words of the `D` lines of chip `c`, in input order.
fn dump_words<const L: usize>(
    chip: &chips::ChipRun<L>,
    prog: &input::Program,
    c: usize,
) -> Vec<(u8, u32, [u8; crate::geom::MAX_LANES])> {
    prog.dumps
        .iter()
        .filter(|d| d.chip as usize == c)
        .map(|d| (d.slice, d.addr, chip.machine.sram_word(d.slice, d.addr).expect("dump words are validated")))
        .collect()
}

/// The threads of a multi-chip run: `--jobs`, else the available cores; one per chip at most.
fn jobs_of(opts: &cli::Options, chips: usize) -> usize {
    let n = match opts.jobs {
        0 => std::thread::available_parallelism().map_or(1, |n| n.get()),
        n => n,
    };
    n.min(chips)
}

/// Several chips in lock-step windows; each chip's output is printed in chip order after a
/// `C <chip>` line.
fn run_many<const L: usize>(
    mut chips: Vec<chips::ChipRun<L>>,
    prog: &input::Program,
    opts: &cli::Options,
    window: u64,
    stats: Option<&mut RunStats>,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> i32 {
    let mut over: Option<u128> = None;
    for c in chips.iter_mut() {
        c.lint = opts.lint;
        c.counts = stats.as_ref().map(|_| stats::Counters::default());
    }
    let jobs = jobs_of(opts, chips.len());
    match chips::run_chips(&mut chips, window, opts.max_cycles, opts.trace, jobs) {
        // chips stopped inside a window: each says why, the reference exits with the first one's
        // code and prints nothing
        Err(es) => {
            for e in &es {
                say(stderr, &e.to_string());
            }
            return match es[0] {
                RunError::MaxCycles { .. } => 3,
                RunError::Input(_) => 2,
                RunError::Internal(_) => 4,
            };
        }
        Ok(chips::Outcome::Done) => {
            if let Some(m) = chips.iter().enumerate().find_map(|(c, chip)| unused_link_fault(chip, c)) {
                say(stderr, &m);
                return 2;
            }
            for (c, chip) in chips.iter_mut().enumerate() {
                let words = dump_words(chip, prog, c);
                let tail = host_tail(&chip.machine, prog, c);
                chip.finish(&words, &tail, opts.trace);
            }
        }
        Ok(chips::Outcome::OverBudget { cycle }) => over = Some(cycle),
    }
    if let Some(st) = stats {
        st.chips = chips
            .iter_mut()
            .map(|c| ChipStats {
                counts: c.counts.take().unwrap_or_default(),
                activity: Some(c.machine.activity.clone()),
                iq: stats::iq_levels(&c.machine),
                ticks: c.ticks,
                last_cycle: c.last_cycle,
                diagnostics: c.diags,
                first: c.first,
            })
            .collect();
        st.obs = chips
            .iter_mut()
            .filter_map(|c| {
                c.obs.take().map(|mut state| {
                    state.finish();
                    ObsChip { state, setup_ns: c.setup_ns, run_ns: c.run_ns }
                })
            })
            .collect();
    }
    let mut code = 0;
    for (c, chip) in chips.iter_mut().enumerate() {
        let _ = writeln!(stdout, "C {c}");
        let _ = stdout.write_all(&chip.out);
        if let Some(e) = chip.error.take() {
            say(stderr, &format!("cannot write standard output: {}", strerror(&e)));
            code = 4;
        }
    }
    let _ = stdout.flush();
    if let Some(cycle) = over {
        say(stderr, &RunError::MaxCycles { cycle, limit: opts.max_cycles }.to_string());
        return 3;
    }
    code
}

fn refuse_args(e: &CliError, stderr: &mut dyn Write) -> i32 {
    say(stderr, e.text());
    2
}

#[allow(clippy::too_many_arguments)] // the pieces of one chip: its input, host script, start and link faults
fn build_machine<const L: usize>(
    profile: &Profile,
    geom: &Geom,
    plan: &Arc<links::LinkPlan>,
    chip: usize,
    input: &input::ChipInput,
    host: Vec<crate::host::HostEvent>,
    start: u64,
    faults: Vec<crate::links::LinkFault>,
) -> Result<chips::ChipRun<L>, String> {
    let port = links::PlanLinks::new(plan.clone(), chip, geom);
    let state = port.state();
    state.lock().unwrap().set_faults(faults);
    let mut m = Machine::<L>::with_lanes(profile, chip as u16, Box::new(port))?;
    for (q, units) in input.images.iter() {
        m.load_image(q, units).map_err(|e| e.to_string())?;
    }
    for p in &input.preloads {
        m.preload(p.slice, p.addr, &p.bytes).map_err(|e| e.to_string())?;
    }
    if !host.is_empty() || start != 0 {
        m.set_host_script(host, start);
    }
    Ok(chips::ChipRun::new(m, state))
}

/// Run `f`, turning a panic into exit code 4 with one line on stderr (the top level of the binary).
pub fn guarded(f: impl FnOnce() -> i32) -> i32 {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(4)
}

/// The process entry point: real argv, stdin, stdout (buffered) and stderr.
pub fn run_process() -> i32 {
    std::panic::set_hook(Box::new(|info| {
        let loc = info.location().map(|l| format!(" at {}:{}", l.file(), l.line())).unwrap_or_default();
        let msg = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_default();
        let msg = msg.replace('\n', " ");
        // One write, and no `eprintln!` (which panics when standard error is closed): the hook runs in whichever thread
        // panicked, so it must not need anything another thread holds.
        let _ = io::stderr().write_all(format!("{PROG}: internal error: {msg}{loc}\n").as_bytes());
    }));
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    guarded(|| {
        let mut out = io::BufWriter::new(io::stdout().lock());
        // Standard error is not locked for the whole run: a worker thread that panics writes the hook's line, and would
        // wait for that lock for ever while this thread waits for the worker.
        let code = main_with(&args, &mut io::stdin().lock(), &mut out, &mut io::stderr());
        let _ = out.flush();
        code
    })
}

/// A lossy rendering of an argument for messages.
pub fn show(a: &OsStr) -> String {
    a.to_string_lossy().into_owned()
}
