//! `tsp-iss run`:
//!
//! ```text
//! tsp-iss run [--profile P | --config F] [--weights W.safetensors ...] [--dump [CHIP:]SLICE:ADDR ...] [--inject [CHIP:]FLINE ...]
//!             [--host FILE] [--trace] [--observe] [--lint] [--on-irq stop|continue|replay] [--stats[=FILE]] [--run-profile-json FILE] [--max-cycles N] [--chips K | --topology FILE] [--jobs N]
//!             [--link-latency N | --link-latency-seed S | --link-jitter-seed S] <prog>
//! ```
//!
//! `<prog>` is an `.elf` object (revision 2), a Q-line text file (first non-blank byte `Q`) or an
//! `.otsm` manifest ([`load_manifest`]: one object per chip). The object, the weight package, the
//! manifest and the text are read by `ots-format` exactly as `ots-asm objinput` and `mobjinput` read
//! them; the `S` lines they would print are the SRAM preloads, the `Q` lines the initial IQ images,
//! and the run is the compatible mode's ([`crate::harness::execute`]) on the same machine setup, so
//! the output is the reference harness format. `--dump` adds the `D` lines (hex, as the harness takes
//! them). Weights are never part of an object: they come from `--weights`, one package per chip in
//! chip order for a manifest. A single-chip object/text program can name multiple packages;
//! their tensor names must be unique and all are bound together before cycle zero. A manifest sets `--chips` to its chip count and refuses another
//! `--chips`. `--host FILE` adds the host's lines of the harness text (`H`, `P`, with the `C` lines that choose the chip) to the
//! program, as if they followed its `Q` lines on the input of the compatible mode.
//!
//! The profile is `--config F` (a configuration file, checked as `tsp_sim` checks it), else
//! `--profile P` (a shipped profile), else the one the object or manifest names; a text program has none of its
//! own, so it needs one of the two options. An object or manifest must be for the profile named.
use super::obs_out::{print, report_text, write_files, ObsOut, Report};
use super::{check_geom, fail, load_file, load_named};
use crate::harness::cli::{common_option, CliError, Options, ReplayOpt};
use crate::harness::input::{add_fault_line, check_word, parse as parse_input, ChipInput, Dump, Limits, Program};
use crate::harness::obs::Probe;
use crate::harness::stats::{queue_of, REASONS};
use crate::harness::{execute, obs, strerror, ChipStats, RunStats};
use ots_format::text::{is_text_input, parse_images, Images};
use ots_format::{bind_weights, read_manifest, read_object, read_weights, Extern};
use ots_isa::{AsmError, Code};
use ots_profile::Profile;
use serde_json::{json, Value};
use std::ffi::{OsStr, OsString};
use std::io::Write;
use std::os::unix::ffi::OsStrExt;

const CMD: &str = "run";

const USAGE: &str = "usage: tsp-iss run [--profile P | --config F] [--weights W.safetensors ...] [--dump [CHIP:]SLICE:ADDR ...] [--inject [CHIP:]FLINE ...] [--host FILE] [--trace] [--observe] [--lint] [--on-irq stop|continue|replay] [--stats[=FILE]] [--energy TABLE [--clock-mhz N]] [--run-profile-json FILE] [--obs-out DIR [--obs-summary] [--obs-diagnostics] [--obs-utilization] [--obs-events] [--obs-probe SPEC ...]] [--obs-report compact|medium|verbose] [--max-cycles N] [--chips K | --topology FILE] [--jobs N] [--matrix-threads N|auto] [--link-latency N | --link-latency-seed S | --link-jitter-seed S] <prog.elf | prog.otsm | prog.txt>";

/// What a program gives one chip: its text images and the externs its weights fill.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChipLoad {
    pub images: Images,
    pub externs: Vec<Extern>,
    /// The SRAM words of the `S` lines of a text program: slice, address, bytes (the dense benchmark programs preload their
    /// matrices this way). An object takes its words from a weight package instead.
    pub sram: Vec<(u64, u64, Vec<u8>)>,
    pub embedded: std::collections::BTreeMap<String, ots_format::weights::Tensor>,
}

struct Args {
    opts: Options,
    chips_given: bool,
    profile: Option<String>,
    config: Option<OsString>,
    weights: Vec<OsString>,
    dumps: Vec<(Option<u64>, u64, u64)>,
    /// `--inject [CHIP:]FLINE`: the `F` lines of the harness, from the command line.
    injects: Vec<(Option<u64>, String)>,
    /// `--host FILE`: the host's `H` and `P` lines (and the `C` lines that choose the chip) of the harness, from a file.
    host: Option<OsString>,
    /// `--stats`: `Some(None)` to stderr, `Some(Some(file))` to the file.
    stats: Option<Option<OsString>>,
    /// `--energy TABLE`: the energy table (`slices::energy`) that prices the activity counters, and `--clock-mhz N` (default 900).
    energy: Option<OsString>,
    clock_mhz: Option<f64>,
    /// `--run-profile-json FILE`.
    run_profile: Option<OsString>,
    /// `--obs-out`, `--obs-summary`, `--obs-report`: what is written and printed besides the recording.
    out: ObsOut,
    /// `--matrix-threads N|auto`: the width of the pool that splits large matrix products (default: 1, or `TSP_ISS_THREADS`).
    matrix_threads: Option<super::threads::MatrixThreads>,
    /// `--replay-runs N`: the number of runs in this process.
    runs: u32,
    prog: OsString,
}

/// Where the time before the machine goes, in nanoseconds (the `load` of the run profile).
#[derive(Default)]
struct LoadNs {
    read_program: u64,
    decode_program: u64,
    read_weights_file: u64,
    verify_weights: u64,
    bind_weights: u64,
    build_input: u64,
}

fn ns(t: std::time::Instant) -> u64 {
    t.elapsed().as_nanos() as u64
}

/// A hex number, `0x` accepted, no sign.
fn hex(s: &str) -> Option<u64> {
    let t = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    if t.is_empty() || !t.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(t, 16).ok()
}

/// `[CHIP:]SLICE:ADDR`, all hex.
fn dump_spec(s: &str) -> Option<(Option<u64>, u64, u64)> {
    let f: Vec<&str> = s.split(':').collect();
    match f.as_slice() {
        [a, b] => Some((None, hex(a)?, hex(b)?)),
        [c, a, b] => Some((Some(hex(c)?), hex(a)?, hex(b)?)),
        _ => None,
    }
}

/// `[CHIP:]FLINE`: the chip in hex (default 0) and an `F` line of the harness without its `F` (`F sram ...`: `sram ...`).
fn inject_spec(s: &str) -> (Option<u64>, String) {
    if let Some((c, rest)) = s.split_once(':') {
        if let Some(chip) = hex(c.trim()) {
            return (Some(chip), rest.to_string());
        }
    }
    (None, s.to_string())
}

fn parse(args: &[OsString]) -> Result<Args, String> {
    let usage = || USAGE.to_string();
    let mut a = Args {
        opts: Options::new(OsString::new()),
        chips_given: false,
        profile: None,
        config: None,
        weights: Vec::new(),
        dumps: Vec::new(),
        injects: Vec::new(),
        host: None,
        stats: None,
        energy: None,
        clock_mhz: None,
        run_profile: None,
        out: ObsOut::default(),
        matrix_threads: None,
        runs: 1,
        prog: OsString::new(),
    };
    let mut prog: Option<OsString> = None;
    let mut i = 0;
    while i < args.len() {
        let name = args[i].as_bytes();
        let next = args.get(i + 1);
        let text = |v: Option<&OsString>| -> Result<String, String> {
            v.and_then(|v| v.to_str()).map(String::from).ok_or_else(usage)
        };
        match name {
            b"--profile" => {
                a.profile = Some(text(next)?);
                i += 1;
            }
            b"--config" => {
                a.config = Some(next.cloned().ok_or_else(usage)?);
                i += 1;
            }
            b"--weights" => {
                a.weights.push(next.cloned().ok_or_else(usage)?);
                i += 1;
            }
            b"--dump" => {
                let s = text(next)?;
                a.dumps.push(dump_spec(&s).ok_or_else(|| format!("--dump {s:?}: expected [CHIP:]SLICE:ADDR in hex"))?);
                i += 1;
            }
            b"--inject" => {
                let s = text(next)?;
                a.injects.push(inject_spec(&s));
                i += 1;
            }
            b"--host" => {
                a.host = Some(next.cloned().ok_or_else(usage)?);
                i += 1;
            }
            b"--run-profile-json" => {
                a.run_profile = Some(next.cloned().ok_or_else(usage)?);
                a.opts.obs.profile = true;
                i += 1;
            }
            b"--obs-out" => {
                a.out.dir = Some(next.cloned().ok_or_else(usage)?);
                i += 1;
            }
            b"--obs-summary" => {
                a.out.summary = true;
                a.opts.obs.summary = true;
            }
            b"--obs-diagnostics" => a.opts.obs.diagnostics = true,
            b"--obs-utilization" => {
                a.out.utilization = true;
                a.opts.obs.utilization = true;
            }
            b"--obs-events" => a.opts.obs.events = true,
            b"--obs-probe" => {
                a.opts.obs.probes.push(Probe::parse(&text(next)?)?);
                i += 1;
            }
            b"--srf-snapshot" => {
                let s = text(next)?;
                for c in s.split(',') {
                    a.opts.obs.probes.push(Probe::parse(&format!("srf_snapshot@{c}"))?);
                }
                i += 1;
            }
            b"--obs-bin" => {
                let s = text(next)?;
                a.opts.obs.bin = s
                    .parse::<u64>()
                    .ok()
                    .filter(|n| *n >= 1)
                    .ok_or_else(|| format!("--obs-bin {s:?}: expected a number of cycles, 1 or more"))?;
                i += 1;
            }
            b"--obs-report" => {
                let s = text(next)?;
                a.out.report = Some(
                    Report::parse(&s)
                        .ok_or_else(|| format!("--obs-report {s:?}: expected compact, medium or verbose"))?,
                );
                a.opts.obs.utilization = true;
                i += 1;
            }
            b"--stats" => a.stats = Some(None),
            b"--energy" => {
                a.energy = Some(next.cloned().ok_or_else(usage)?);
                i += 1;
            }
            b"--clock-mhz" => {
                let s = text(next)?;
                a.clock_mhz =
                    Some(s.parse::<f64>().ok().filter(|f| f.is_finite() && *f > 0.0 && *f <= 100_000.0).ok_or_else(
                        || format!("--clock-mhz {s:?}: expected a clock in MHz, above 0 and at most 100000"),
                    )?);
                i += 1;
            }
            b"--matrix-threads" => {
                let v = text(next)?;
                a.matrix_threads = Some(
                    super::threads::MatrixThreads::parse(&v)
                        .ok_or_else(|| format!("--matrix-threads {v:?}: expected a number from 1 to 16, or auto"))?,
                );
                i += 1;
            }
            b"--replay-runs" => {
                let n = text(next)?.parse::<u32>().ok().filter(|n| (1..=1000).contains(n)).ok_or_else(usage)?;
                a.runs = n;
                a.opts.replay = ReplayOpt::Session;
                i += 1;
            }
            _ if name.starts_with(b"--stats=") => a.stats = Some(Some(OsStr::from_bytes(&name[8..]).to_os_string())),
            _ => match common_option(&mut a.opts, name, next) {
                Ok(Some(n)) => {
                    a.chips_given |= name == b"--chips";
                    i += n;
                }
                Ok(None) if !name.starts_with(b"-") && prog.is_none() => prog = Some(args[i].clone()),
                Ok(None) => return Err(usage()),
                Err(CliError::Usage) => return Err(usage()),
                Err(CliError::Message(m)) => return Err(m),
            },
        }
        i += 1;
    }
    a.prog = prog.ok_or_else(usage)?;
    if a.clock_mhz.is_some() && a.energy.is_none() {
        return Err("--clock-mhz needs --energy".into());
    }
    a.out.check(&a.opts.obs)?;
    Ok(a)
}

/// An `ots-format` error naming the file when it has no location of its own.
fn asm_err(e: &AsmError, path: &OsStr) -> String {
    e.render(&path.to_string_lossy())
}

/// Read an `.otsm` manifest (`ots_format::read_manifest`) into one [`ChipLoad`] per chip, in chip
/// order (chip i is entry i), and the profile name of its objects; the checks are those of
/// `ots-asm mobjinput`. `profile` and `config` are `--profile` and `--config`.
pub fn load_manifest(
    data: &[u8],
    profile: Option<&str>,
    config: Option<&str>,
) -> Result<(String, Vec<ChipLoad>), AsmError> {
    let (name, objects) = read_manifest(data, profile, config)?;
    let mut chips = Vec::with_capacity(objects.len());
    for o in objects {
        let (_, images, externs) = read_object(o, Some(&name), config)?;
        let embedded = ots_format::object_preloads(o, &name, &externs, config, 1)?;
        chips.push(ChipLoad { images, externs, sram: Vec::new(), embedded });
    }
    Ok((name, chips))
}

/// What `<prog>` is.
struct Loaded {
    /// The profile named by the program, if it names one.
    name: Option<String>,
    chips: Vec<ChipLoad>,
    manifest: bool,
}

fn load_program(a: &Args, data: &[u8], machine_profile: &Option<Profile>) -> Result<Loaded, String> {
    let config = match &a.config {
        Some(c) => Some(c.to_str().ok_or_else(|| "--config: not UTF-8".to_string())?),
        None => None,
    };
    if data.starts_with(b"OTSM") {
        let (name, chips) = load_manifest(data, a.profile.as_deref(), config).map_err(|e| asm_err(&e, &a.prog))?;
        return Ok(Loaded { name: Some(name), chips, manifest: true });
    }
    if is_text_input(data) {
        if machine_profile.is_none() {
            return Err("a text program has no profile of its own: give --profile or --config".into());
        }
        let txt = std::str::from_utf8(data)
            .ok()
            .filter(|t| t.is_ascii())
            .ok_or_else(|| asm_err(&AsmError::new(Code::Asm003, "text input is not ASCII"), &a.prog))?;
        // the `Q` lines are the program, the `S` lines preload SRAM words (each kind with the other's lines blanked, so that a
        // message names the line of the file); every other line is refused as before
        let is_s = |l: &str| l.trim_start().starts_with('S');
        let queues: Vec<&str> = txt.lines().map(|l| if is_s(l) { "" } else { l }).collect();
        let images = parse_images(&queues.join("\n"), &a.prog.to_string_lossy()).map_err(|e| asm_err(&e, &a.prog))?;
        let sram = match machine_profile {
            Some(p) if txt.lines().any(is_s) => {
                text_sram(txt, p).map_err(|m| format!("{}: {m}", a.prog.to_string_lossy()))?
            }
            _ => Vec::new(),
        };
        return Ok(Loaded {
            name: a.profile.clone(),
            chips: vec![ChipLoad { images, sram, ..ChipLoad::default() }],
            manifest: false,
        });
    }
    let (name, images, externs) = read_object(data, a.profile.as_deref(), config).map_err(|e| asm_err(&e, &a.prog))?;
    let embedded = ots_format::object_preloads(data, &name, &externs, config, 1).map_err(|e| asm_err(&e, &a.prog))?;
    Ok(Loaded {
        name: Some(name),
        chips: vec![ChipLoad { images, externs, sram: Vec::new(), embedded }],
        manifest: false,
    })
}

/// An SRAM word to preload: slice, address, bytes.
type Preload = (u64, u64, Vec<u8>);

/// The `S` lines of a text program, read by the input parser of the compatible mode (its checks and messages), with the other
/// lines blanked so that a message names the line of the file.
fn text_sram(txt: &str, profile: &Profile) -> Result<Vec<Preload>, String> {
    let only_s: String =
        txt.lines().map(|l| if l.trim_start().starts_with('S') { l } else { "" }).collect::<Vec<_>>().join("\n");
    let geom = crate::geom::Geom::new(profile);
    let p = crate::harness::input::parse(
        only_s.as_bytes(),
        crate::harness::input::Limits { geom: &geom, chips: 1, topo: None },
    )
    .map_err(|e| format!("line {}: {}", e.line, e.msg))?;
    Ok(p.chips[0].preloads.iter().map(|w| (w.slice as u64, w.addr as u64, w.bytes.clone())).collect())
}

/// The weight package of each chip bound to its externs (`objinput --weights`): the SRAM words, in extern order.
fn bind(a: &Args, l: &Loaded, config: Option<&str>, lt: &mut LoadNs) -> Result<Vec<Vec<Preload>>, String> {
    let k = l.chips.len();
    if l.manifest && !a.weights.is_empty() && a.weights.len() != k {
        return Err(asm_err(
            &AsmError::new(Code::Asm004, format!(
                "--weights given {} times: the manifest has {k} chips and needs one weight package per chip (--weights {k} times)",
                a.weights.len()
            )),
            &a.prog,
        ));
    }
    let mut out = Vec::with_capacity(k);
    for (c, chip) in l.chips.iter().enumerate() {
        if a.weights.is_empty() && chip.embedded.is_empty() {
            out.push(chip.sram.clone());
            continue;
        }
        let paths = if l.manifest && !a.weights.is_empty() { &a.weights[c..c + 1] } else { &a.weights[..] };
        let mut load = |w: &OsString| {
            let t = std::time::Instant::now();
            let data = std::fs::read(w).map_err(|e| format!("{}: {}", w.to_string_lossy(), strerror(&e)))?;
            lt.read_weights_file += ns(t);
            let t = std::time::Instant::now();
            let package = read_weights(&data, l.name.as_deref(), config).map_err(|e| asm_err(&e, w))?;
            lt.verify_weights += ns(t);
            Ok::<_, String>(package)
        };
        let mut pk = chip.embedded.clone();
        for path in paths {
            ots_format::merge_preloads(&mut pk, load(path)?).map_err(|e| asm_err(&e, path))?;
        }
        let w = if paths.len() == 1 { &paths[0] } else { &a.prog };
        let t = std::time::Instant::now();
        let pre = bind_weights(&chip.externs, &pk).map_err(|e| asm_err(&e, w))?;
        let mut words = chip.sram.clone();
        words.extend(pre.iter().map(|p| (p.slice, p.addr, p.word.to_vec())));
        out.push(words);
        lt.bind_weights += ns(t);
    }
    Ok(out)
}

/// The width `--matrix-threads` asks for: the number, or what `auto` makes of the program (`threads::auto_width`); `None`
/// without the option.
fn matrix_width(a: &Args, profile: &Profile, geom: &crate::geom::Geom, loaded: &Loaded) -> Option<usize> {
    use super::threads::{auto_width, MatrixThreads};
    match a.matrix_threads? {
        MatrixThreads::Width(n) => Some(n),
        MatrixThreads::Auto => {
            let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
            let chips = a.opts.chips.max(1) as usize;
            let jobs = if a.opts.jobs == 0 { cores } else { a.opts.jobs }.min(chips);
            let images: Vec<&Images> = loaded.chips.iter().map(|c| &c.images).collect();
            Some(auto_width(profile, geom, &images, cores, jobs))
        }
    }
}

/// `tsp-iss run ...`; `args` are the arguments after `run`.
pub fn run(args: &[OsString], stdout: &mut dyn Write, stderr: &mut dyn Write) -> i32 {
    let started = std::time::Instant::now();
    let mut a = match parse(args) {
        Ok(a) => a,
        Err(m) => return fail(stderr, CMD, &m),
    };
    // the activity counters are read only by `--stats` and `--energy`: any other run does not pay for them
    crate::slices::activity::set_default_enabled(a.stats.is_some() || a.energy.is_some());
    // the energy table is read before the run: a bad table fails at once
    let energy = match &a.energy {
        Some(f) => match crate::slices::energy::Table::load(std::path::Path::new(f)) {
            Ok(t) => Some(t),
            Err(m) => return fail(stderr, CMD, &format!("--energy {}: {m}", f.to_string_lossy())),
        },
        None => None,
    };
    // the machine's profile: --config, else --profile
    let machine_profile = match (&a.config, &a.profile) {
        (Some(c), _) => match load_file(c) {
            Ok(p) => Some(p),
            Err(m) => return fail(stderr, CMD, &m),
        },
        (None, Some(n)) => match load_named(n) {
            Ok(p) => Some(p),
            Err(m) => return fail(stderr, CMD, &m),
        },
        (None, None) => None,
    };
    let mut lt = LoadNs::default();
    let t = std::time::Instant::now();
    let data = match std::fs::read(&a.prog) {
        Ok(d) => d,
        Err(e) => return fail(stderr, CMD, &format!("{}: {}", a.prog.to_string_lossy(), strerror(&e))),
    };
    lt.read_program = ns(t);
    let t = std::time::Instant::now();
    let loaded = match load_program(&a, &data, &machine_profile) {
        Ok(l) => l,
        Err(m) => return fail(stderr, CMD, &m),
    };
    lt.decode_program = ns(t);
    let profile = match machine_profile {
        Some(p) => p,
        None => match load_named(loaded.name.as_deref().unwrap_or("ref")) {
            Ok(p) => p,
            Err(m) => return fail(stderr, CMD, &m),
        },
    };
    let origin = a
        .config
        .as_ref()
        .map(|c| c.to_string_lossy().into_owned())
        .or_else(|| a.profile.clone())
        .or_else(|| loaded.name.clone())
        .unwrap_or_default();
    let label = loaded.name.clone().or_else(|| a.profile.clone()).unwrap_or_else(|| "config".to_string());
    let geom = match check_geom(&profile, &origin) {
        Ok(g) => g,
        Err(m) => return fail(stderr, CMD, &m),
    };
    let mode = match a.opts.link_mode() {
        Ok(m) => m,
        Err(_) => return fail(stderr, CMD, USAGE),
    };
    let topo = match crate::harness::load_topology(&mut a.opts, &geom) {
        Ok(t) => t,
        Err(m) => return fail(stderr, CMD, &m),
    };
    if loaded.manifest {
        let k = loaded.chips.len() as u32;
        if topo.is_some() && a.opts.chips != k {
            return fail(
                stderr,
                CMD,
                &format!("the topology has {} chips but the manifest has {k} chips", a.opts.chips),
            );
        }
        if a.chips_given && a.opts.chips != k {
            return fail(stderr, CMD, &format!("--chips {} but the manifest has {k} chips", a.opts.chips));
        }
        if topo.is_none() && k > 264 {
            return fail(
                stderr,
                CMD,
                &format!("the manifest has {k} chips: more than 264 chips need --topology FILE (the built-in wiring is for at most 264)"),
            );
        }
        a.opts.chips = k;
    }
    let config = a.config.as_ref().and_then(|c| c.to_str());
    let preloads = match bind(&a, &loaded, config, &mut lt) {
        Ok(p) => p,
        Err(m) => return fail(stderr, CMD, &m),
    };

    // the program of the compatible mode: the images and the preloads of each chip, and the dumps
    let t = std::time::Instant::now();
    let mut prog =
        Program { topo: topo.clone(), chips: vec![ChipInput::default(); a.opts.chips as usize], ..Program::default() };
    for (c, (chip, pre)) in loaded.chips.iter().zip(&preloads).enumerate() {
        let input = &mut prog.chips[c];
        for (q, units) in chip.images.iter() {
            if let Err(m) = input.add_image(&geom, q, units) {
                return fail(stderr, CMD, &format!("chip {c}: {m}"));
            }
        }
        for (slice, addr, word) in pre {
            if let Err(m) = input.add_preload(&geom, *slice, *addr, word) {
                return fail(stderr, CMD, &format!("chip {c}: {m}"));
            }
        }
    }
    for &(chip, slice, addr) in &a.dumps {
        let chip = chip.unwrap_or(0);
        if chip >= a.opts.chips as u64 {
            return fail(stderr, CMD, &format!("--dump: chip {chip:x} out of range (--chips)"));
        }
        if let Err(m) = check_word(&geom, slice, addr) {
            return fail(stderr, CMD, &format!("--dump {slice:x}:{addr:x}: {m}"));
        }
        prog.dumps.push(Dump { chip: chip as u32, slice: slice as u8, addr: addr as u32 });
    }

    for (chip, line) in &a.injects {
        let chip = chip.unwrap_or(0);
        if chip >= a.opts.chips as u64 {
            return fail(stderr, CMD, &format!("--inject: chip {chip:x} out of range (--chips)"));
        }
        if let Err(m) = add_fault_line(&mut prog, chip as usize, line, &geom) {
            return fail(stderr, CMD, &format!("--inject {line:?}: {m}"));
        }
    }

    if let Some(f) = &a.host {
        let text = match std::fs::read(f) {
            Ok(t) => t,
            Err(e) => return fail(stderr, CMD, &format!("--host {}: {}", f.to_string_lossy(), strerror(&e))),
        };
        let h = match parse_input(&text, Limits { geom: &geom, chips: a.opts.chips, topo: topo.as_ref() }) {
            Ok(h) => h,
            Err(e) => return fail(stderr, CMD, &format!("--host {}: {e}", f.to_string_lossy())),
        };
        prog.host.extend(h.host);
        prog.pkts.extend(h.pkts);
        prog.link_faults.extend(h.link_faults);
        prog.link_persist.extend(h.link_persist);
        if h.starts.iter().any(|&s| s != 0) {
            prog.starts = h.starts;
        }
    }

    lt.build_input = ns(t);

    if let Some(w) = matrix_width(&a, &profile, &geom, &loaded) {
        let auto = a.matrix_threads == Some(super::threads::MatrixThreads::Auto);
        let set = if auto {
            crate::slices::pool::Pool::configure_adaptive(w)
        } else {
            crate::slices::pool::Pool::configure(w)
        };
        if let Err(have) = set {
            let _ = writeln!(
                stderr,
                "{}: {CMD}: --matrix-threads: the pool has {have} threads already",
                crate::harness::PROG
            );
        }
    }
    let mut stats = RunStats::default();
    let mut code = 0;
    for run in 1..=a.runs {
        // the runs before the last print nothing on stdout: the output is that of the last one
        stats = RunStats::default();
        let t0 = std::time::Instant::now();
        code = if run < a.runs {
            execute(&profile, &geom, &prog, &a.opts, mode.clone(), Some(&mut stats), &mut std::io::sink(), stderr)
        } else {
            execute(&profile, &geom, &prog, &a.opts, mode.clone(), Some(&mut stats), stdout, stderr)
        };
        if a.runs > 1 {
            let ticks: u64 = stats.chips.iter().map(|c| c.ticks).sum();
            let last = stats.chips.iter().filter_map(|c| c.last_cycle).max().unwrap_or(0);
            let _ = writeln!(
                stderr,
                "{}: run {run}/{}: wall {:.6} s, run phase {:.6} s, {ticks} ticks, last cycle {last}",
                crate::harness::PROG,
                a.runs,
                t0.elapsed().as_secs_f64(),
                stats.wall_seconds
            );
        }
        if code != 0 {
            break;
        }
    }
    if code == 0 {
        if let Some(dest) = &a.run_profile {
            let r = write_profile(dest, &label, &lt, started, &stats, stderr);
            if r != 0 {
                return r;
            }
        }
        if a.opts.obs.any() {
            let files = match write_files(CMD, &a.out, &a.opts.obs, &label, &stats) {
                Ok(f) => f,
                Err(m) => {
                    let _ = writeln!(stderr, "{m}");
                    return 4;
                }
            };
            if let Some(level) = a.out.report.or(a.out.summary.then_some(Report::Compact)) {
                print(stderr, &report_text(level, &label, &stats, &files));
            }
        }
        let priced = energy.as_ref().map(|t| {
            let chips: Vec<_> = stats
                .chips
                .iter()
                .filter_map(|c| {
                    c.activity
                        .clone()
                        .map(|act| (act, c.counts.fetch_bytes, c.last_cycle.map_or(0, |l| u128::from(l) + 1)))
                })
                .collect();
            crate::slices::energy::estimate_run(t, &chips, a.clock_mhz.unwrap_or(900.0))
        });
        if let Some(e) = &priced {
            print(stderr, &crate::slices::energy::summary(e));
        }
        if let Some(dest) = &a.stats {
            return write_stats(dest, &label, &stats, priced, stderr);
        }
    }
    code
}

/// Write a JSON document to `dest`: exit 4 and a line on stderr when that fails.
fn write_json(dest: &OsStr, doc: &Value, stderr: &mut dyn Write) -> i32 {
    let text = serde_json::to_string_pretty(doc).expect("json") + "\n";
    match std::fs::write(dest, text) {
        Ok(()) => 0,
        Err(e) => {
            let _ = writeln!(
                stderr,
                "{}: {CMD}: cannot write {}: {}",
                crate::harness::PROG,
                dest.to_string_lossy(),
                strerror(&e)
            );
            4
        }
    }
}

/// `--run-profile-json`: the run profile (`harness::obs`), from the start of `run` to now.
fn write_profile(
    dest: &OsStr,
    profile: &str,
    lt: &LoadNs,
    started: std::time::Instant,
    st: &RunStats,
    stderr: &mut dyn Write,
) -> i32 {
    let load = json!({
        "read_program_ns": lt.read_program,
        "decode_program_ns": lt.decode_program,
        "read_weights_file_ns": lt.read_weights_file,
        "verify_weights_ns": lt.verify_weights,
        "bind_weights_ns": lt.bind_weights,
        "build_input_ns": lt.build_input,
    });
    let chips: Vec<obs::ChipProfile> = st
        .obs
        .iter()
        .zip(&st.chips)
        .map(|(o, c)| obs::ChipProfile {
            st: &o.state,
            setup_ns: o.setup_ns,
            run_ns: o.run_ns,
            ticks: c.ticks,
            last_cycle: c.last_cycle,
        })
        .collect();
    let doc = obs::profile_json(profile, ns(started), load, &chips);
    write_json(dest, &doc, stderr)
}

/// `{"parked": n, ...}` for a wait-cycle row.
fn wait_object(w: &[u128; 7], only_nonzero: bool) -> Value {
    let mut m = serde_json::Map::new();
    for ((_, name), n) in REASONS.iter().zip(w) {
        if !only_nonzero || *n > 0 {
            m.insert((*name).into(), json!(n));
        }
    }
    Value::Object(m)
}

/// The activity counters of every chip added up.
fn activity_total(st: &RunStats) -> Option<crate::slices::activity::Activity> {
    let mut it = st.chips.iter().filter_map(|c| c.activity.as_ref());
    let mut total = it.next()?.clone();
    for a in it {
        total.add(a);
    }
    Some(total)
}

/// One chip's entry of the summary: the totals and the queues that did anything.
fn chip_json(c: &ChipStats) -> Value {
    let n = &c.counts;
    let mut queues = Vec::new();
    for s in 0..crate::geom::QUEUES {
        let q = queue_of(s);
        let level = c.iq.iter().find(|l| l.0 == q);
        let (dispatches, waits) = (n.dispatches[s], &n.waits[s]);
        if level.is_none() && dispatches == 0 && waits.iter().all(|w| *w == 0) {
            continue;
        }
        let (min, max) = level.map_or((0, 0), |l| (l.1, l.2));
        queues.push(json!({
            "queue": q,
            "name": ots_isa::queue_name(q),
            "dispatches": dispatches,
            "wait_cycles": wait_object(waits, true),
            "iq_min": min,
            "iq_max": max,
        }));
    }
    json!({
        "ticks": c.ticks,
        "last_cycle": c.last_cycle,
        "diagnostics": c.diagnostics,
        "first_diagnostic": c.first.map(|(code, cycle)| json!({"code": code.as_str(), "cycle": cycle})),
        "dispatches": n.dispatches.iter().sum::<u64>(),
        "fetch_bytes": n.fetch_bytes,
        "wait_cycles": wait_object(&n.wait_totals(), false),
        "diagnostics_by_code": n.by_code,
        "queues": queues,
        "activity": c.activity.as_ref().map(|a| a.to_json()),
    })
}

/// What the pool does with the products: `fixed` (a plain pool splits by the size of the product), `idle` (an adaptive pool that has
/// forked no product, for example the i8 products, which are all below the fork threshold: nothing to measure or decide), `measuring`
/// (forks measured, decision pending), `on` or `off` (after its last probe: a split that saves time, or one that does not).
fn matrix_split(pool: &crate::slices::pool::Pool) -> &'static str {
    match (pool.is_adaptive(), pool.splitting()) {
        (false, _) => "fixed",
        (true, None) if pool.measured_forks() == 0 => "idle",
        (true, None) => "measuring",
        (true, Some(true)) => "on",
        (true, Some(false)) => "off",
    }
}

/// The version of the `--stats` JSON document: raised when a key changes meaning or goes away.
const STATS_SCHEMA: u64 = 1;

/// The `--stats` summary (JSON; never on stdout): the totals over the chips, then one entry per chip.
fn write_stats(
    dest: &Option<OsString>,
    profile: &str,
    st: &RunStats,
    energy: Option<Value>,
    stderr: &mut dyn Write,
) -> i32 {
    let mut waits = [0u128; 7];
    let mut by_code: std::collections::BTreeMap<&str, u64> = std::collections::BTreeMap::new();
    for c in &st.chips {
        for (a, b) in waits.iter_mut().zip(c.counts.wait_totals()) {
            *a += b;
        }
        for (code, n) in &c.counts.by_code {
            *by_code.entry(code).or_insert(0) += n;
        }
    }
    let doc = json!({
        "schema": STATS_SCHEMA,
        "profile": profile,
        "chips": st.chips.len(),
        "ticks": st.chips.iter().map(|c| c.ticks).sum::<u64>(),
        "last_cycle": st.chips.iter().filter_map(|c| c.last_cycle).max(),
        "diagnostics": st.chips.iter().map(|c| c.diagnostics).sum::<u64>(),
        "dispatches": st.chips.iter().map(|c| c.counts.dispatches.iter().sum::<u64>()).sum::<u64>(),
        "fetch_bytes": st.chips.iter().map(|c| c.counts.fetch_bytes).sum::<u64>(),
        "wait_cycles": wait_object(&waits, false),
        "diagnostics_by_code": by_code,
        "activity": activity_total(st).map(|a| a.to_json()),
        "energy": energy,
        "wall_seconds": st.wall_seconds,
        "matrix_threads": crate::slices::pool::Pool::shared().threads(),
        "matrix_split": matrix_split(crate::slices::pool::Pool::shared()),
        "replay": crate::harness::stats::replay_json(
            crate::driver::replay::ENABLED,
            &crate::driver::replay::ReplayCache::global().counters(),
        ),
        "per_chip": st.chips.iter().map(chip_json).collect::<Vec<_>>(),
    });
    let text = serde_json::to_string_pretty(&doc).expect("json") + "\n";
    match dest {
        None => {
            let _ = stderr.write_all(text.as_bytes());
            0
        }
        Some(f) => match std::fs::write(f, text) {
            Ok(()) => 0,
            Err(e) => {
                let _ = writeln!(
                    stderr,
                    "{}: {CMD}: cannot write {}: {}",
                    crate::harness::PROG,
                    f.to_string_lossy(),
                    strerror(&e)
                );
                4
            }
        },
    }
}
