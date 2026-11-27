//! The command line of the `tsp_sim`-compatible mode (`tsp_sim.c` `main`):
//!
//! `tsp-iss <config.json> [--trace] [--observe] [--lint] [--on-irq stop|continue|replay] [--max-cycles N] [--chips K]
//!          [--link-latency N | --link-latency-seed S | --link-jitter-seed S]`
//!
//! The hidden option `--tick-every-cycle` (not in `tsp_sim`) ticks every cycle, not only the model's
//! events; the hidden option `--jobs N` (1..1024) runs the chips of `--chips` on N threads (default: the
//! available cores, at most one per chip); the output does not depend on it. Options are checked in argument order, as `tsp_sim` does. The numbers are read with C's
//! `strtoull`/`strtol`/`strtoll` (`cnum`).
use super::cnum::{strtol0, strtoull0};
use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::OsStrExt;

/// The frozen error usage line of `tsp_sim`. Its successful `--help` also advertises site observation.
pub const USAGE: &str = "usage: tsp-iss <config.json> [--trace] [--lint] [--on-irq stop|continue|replay] [--max-cycles N] [--chips K | --topology FILE] [--window N] [--link-latency N | --link-latency-seed S | --link-jitter-seed S]";

/// The native subcommands; an `argv[1]` equal to one of them is a subcommand.
pub const SUBCOMMANDS: [&str; 5] = ["run", "vectors", "bench", "info", "verify"];

/// What `--help` and `help` print after the usage line of the compatible mode.
pub const SUBCOMMAND_HELP: &str =
    "subcommands (`tsp-iss <subcommand> ...`; each prints its own usage on a bad argument):
  run      run an object (.elf), a manifest (.otsm) or a Q text with weight packages
  bench    run the benchmark programs and print simulated cycles per second
  vectors  run the golden vectors of a conformance file in-process
  info     print the derived profile (--config F | --profile P)
  verify   run an object without data; exit 1 on a diagnostic of a schedule class";

/// Default `--max-cycles`.
pub const DEFAULT_MAX_CYCLES: u64 = 100_000_000;

/// How the links get their latency (the three link options exclude one another).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LinkMode {
    /// `c2c.link_latency` on every link.
    Nominal,
    /// `--link-latency N`.
    Fixed(u64),
    /// `--link-latency-seed S`: each directed link's latency drawn from the range.
    Seed(u64),
    /// `--link-jitter-seed S`: the latency varies per transfer.
    Jitter(u64),
    /// `--link-latency-map C:L=N,...` (hidden; `tsp-iss verify` uses it): the directed link `L` of chip `C` has latency `N`,
    /// every other link the nominal latency.
    Map(Vec<(u32, u32, u64)>),
}

/// The host runtime's policy for the error interrupt (`--on-irq`; `tsp_sim.c`, microarchitecture.md §16.2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OnIrq {
    /// The chip does not stop (the default).
    #[default]
    Continue,
    /// End the run at the end of the cycle of the first error interrupt (exit 5).
    Stop,
    /// Run again from reset, with the faults marked `p` only, after a run that raised an error interrupt.
    Replay,
}

/// What `tsp-iss run` does with a recorded schedule (`driver::replay`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ReplayOpt {
    #[default]
    None,
    /// `--replay-runs N`: the in-process cache (the first run records, the later ones replay).
    Session,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Options {
    pub config: OsString,
    pub trace: bool,
    pub observe: bool,
    /// `--lint`: print the simulator lints (`T lint`).
    pub lint: bool,
    /// `--on-irq`.
    pub on_irq: OnIrq,
    pub max_cycles: u64,
    pub chips: u32,
    /// `--chips` was given.
    pub chips_given: bool,
    /// `--topology FILE`: the wiring of the chips (`harness::topology`); it gives the number of chips.
    pub topology: Option<OsString>,
    /// `--window N`: lock-step windows of at most N cycles (0: the smallest link latency); the output does not depend on it.
    pub window: u64,
    /// `--tick-every-cycle` (hidden): tick every cycle, not only the model's events.
    pub tick_every_cycle: bool,
    /// `--jobs N`: threads for the chips of `--chips` (0: the available cores).
    pub jobs: usize,
    /// What `run` records besides the trace (`--run-profile-json`, `--obs-*`).
    pub obs: super::obs::ObsConfig,
    pub replay: ReplayOpt,
    lat_fixed: Option<u64>,
    lat_seed: Option<u64>,
    jit_seed: Option<u64>,
    lat_map: Option<Vec<(u32, u32, u64)>>,
}

/// An argument error: exit 2, with the usage line or a message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CliError {
    Usage,
    Message(String),
}

impl CliError {
    /// The line printed on stderr (without the program prefix).
    pub fn text(&self) -> &str {
        match self {
            CliError::Usage => USAGE,
            CliError::Message(m) => m,
        }
    }
}

/// Is `args[0]` a native subcommand name?
pub fn is_subcommand(args: &[OsString]) -> bool {
    args.first().and_then(|a| a.to_str()).is_some_and(|a| SUBCOMMANDS.contains(&a))
}

/// A non-negative number of a link option: `strtol` or `strtoll`, no trailing junk, no overflow.
/// `tsp_sim` also tests `errno`: the BSD libc sets `EINVAL` when no digit was converted, so on Apple
/// targets the empty string is a usage error there, while glibc accepts it as 0.
fn nonneg(a: &OsStr) -> Result<u64, CliError> {
    let (v, end, overflow) = strtol0(a.as_bytes());
    let einval = cfg!(target_vendor = "apple") && a.is_empty();
    if end != a.len() || overflow || v < 0 || einval {
        return Err(CliError::Usage);
    }
    Ok(v as u64)
}

/// `C:L=N,C:L=N,...`: the latency `N` of the directed link `L` of chip `C`.
fn latency_map(a: &OsStr) -> Result<Vec<(u32, u32, u64)>, CliError> {
    let text = a.to_str().ok_or(CliError::Usage)?;
    let num = |t: &str| t.parse::<u64>().map_err(|_| CliError::Usage);
    text.split(',')
        .map(|e| {
            let (cl, n) = e.split_once('=').ok_or(CliError::Usage)?;
            let (c, l) = cl.split_once(':').ok_or(CliError::Usage)?;
            Ok((
                u32::try_from(num(c)?).map_err(|_| CliError::Usage)?,
                u32::try_from(num(l)?).map_err(|_| CliError::Usage)?,
                num(n)?,
            ))
        })
        .collect()
}

/// The option `name` of either mode (`--trace`, `--lint`, `--tick-every-cycle`, `--jobs`, `--max-cycles`,
/// `--chips` and the link options),
/// checked as `tsp_sim` checks it. `Ok(Some(n))`: the option took `n` more arguments; `Ok(None)`: not
/// a shared option; `value` is the argument after `name`, if any.
pub fn common_option(o: &mut Options, name: &[u8], value: Option<&OsString>) -> Result<Option<usize>, CliError> {
    match (name, value) {
        (b"--observe", _) => o.observe = true,
        (b"--trace", _) => o.trace = true,
        (b"--lint", _) => o.lint = true,
        (b"--tick-every-cycle", _) => o.tick_every_cycle = true,
        (b"--on-irq", Some(a)) => {
            o.on_irq = match a.as_bytes() {
                b"stop" => OnIrq::Stop,
                b"continue" => OnIrq::Continue,
                b"replay" => OnIrq::Replay,
                _ => return Err(CliError::Message("--on-irq must be stop, continue or replay".into())),
            };
            return Ok(Some(1));
        }
        (b"--jobs", Some(a)) => {
            let (n, end, _) = strtol0(a.as_bytes());
            if end != a.len() || !(1..=1024).contains(&n) {
                return Err(CliError::Message("--jobs must be a number from 1 to 1024".into()));
            }
            o.jobs = n as usize;
            return Ok(Some(1));
        }
        (b"--max-cycles", Some(a)) => {
            // strtoull accepts leading space and a sign ("-1" is 2^64 - 1): digits only
            let b = a.as_bytes();
            if !b.first().is_some_and(u8::is_ascii_digit) {
                return Err(CliError::Usage);
            }
            let n = strtoull0(b);
            if n.end != b.len() || n.overflow {
                return Err(CliError::Usage);
            }
            o.max_cycles = n.mag;
            return Ok(Some(1));
        }
        (b"--chips", Some(a)) => {
            let (k, end, _) = strtol0(a.as_bytes());
            if end != a.len() || !(1..=264).contains(&k) {
                return Err(CliError::Message(
                    "--chips must be 1..264 (33 nodes of 8; a larger system needs --topology)".into(),
                ));
            }
            o.chips = k as u32;
            o.chips_given = true;
            return Ok(Some(1));
        }
        (b"--topology", Some(a)) => {
            o.topology = Some(a.clone());
            return Ok(Some(1));
        }
        (b"--window", Some(a)) => {
            let w = nonneg(a)?;
            if w < 1 {
                return Err(CliError::Usage);
            }
            o.window = w;
            return Ok(Some(1));
        }
        (b"--link-latency", Some(a)) => {
            o.lat_fixed = Some(nonneg(a)?);
            return Ok(Some(1));
        }
        (b"--link-latency-seed", Some(a)) => {
            o.lat_seed = Some(nonneg(a)?);
            return Ok(Some(1));
        }
        (b"--link-jitter-seed", Some(a)) => {
            o.jit_seed = Some(nonneg(a)?);
            return Ok(Some(1));
        }
        (b"--link-latency-map", Some(a)) => {
            o.lat_map = Some(latency_map(a)?);
            return Ok(Some(1));
        }
        _ => return Ok(None),
    }
    Ok(Some(0))
}

/// Parse the arguments after the program name.
pub fn parse(args: &[OsString]) -> Result<Options, CliError> {
    let Some((config, rest)) = args.split_first() else { return Err(CliError::Usage) };
    let mut o = Options::new(config.clone());
    let mut i = 0;
    while i < rest.len() {
        match common_option(&mut o, rest[i].as_bytes(), rest.get(i + 1))? {
            Some(n) => i += n,
            None => return Err(CliError::Usage),
        }
        i += 1;
    }
    Ok(o)
}

impl Options {
    /// The defaults of every option, for the configuration file `config`.
    pub fn new(config: OsString) -> Options {
        Options {
            config,
            trace: false,
            observe: false,
            lint: false,
            on_irq: OnIrq::Continue,
            max_cycles: DEFAULT_MAX_CYCLES,
            chips: 1,
            chips_given: false,
            topology: None,
            window: 0,
            tick_every_cycle: false,
            jobs: 0,
            obs: Default::default(),
            replay: ReplayOpt::None,
            lat_fixed: None,
            lat_seed: None,
            jit_seed: None,
            lat_map: None,
        }
    }

    /// The link mode. `tsp_sim` checks this after the configuration (so a bad configuration is
    /// reported first): at most one of the three link options may be given.
    pub fn link_mode(&self) -> Result<LinkMode, CliError> {
        match (self.lat_fixed, self.lat_seed, self.jit_seed, &self.lat_map) {
            (None, None, None, None) => Ok(LinkMode::Nominal),
            (Some(n), None, None, None) => Ok(LinkMode::Fixed(n)),
            (None, Some(s), None, None) => Ok(LinkMode::Seed(s)),
            (None, None, Some(s), None) => Ok(LinkMode::Jitter(s)),
            (None, None, None, Some(m)) => Ok(LinkMode::Map(m.clone())),
            _ => Err(CliError::Usage),
        }
    }
}
