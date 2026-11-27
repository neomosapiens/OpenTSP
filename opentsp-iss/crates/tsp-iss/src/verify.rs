//! `tsp-iss verify <obj.elf | manifest.otsm> [--weights W.safetensors ...] [--config C.json] [--profile P] [--host FILE]
//! [--topology FILE] [--max-cycles N] [--allow-parked]`: the data-independent check of a program (`--profile` is for a program that names none: a Q text or a
//! manifest, as for `tsp-iss run`; `--topology FILE` is passed on to the runs, as a manifest of more than 264 chips needs one).
//!
//! The program runs on the ISS without data (no SRAM preload but the optional weight packages, no faults), and every
//! diagnostic of the run is printed, one line each, in the order the model reports them, with its class:
//!
//! ```text
//! OTS-E-TIM-001 cycle 43 queue 88 schedule
//! OTS-E-IQ-004 cycle 60 queue 12 data-dependent
//! OTS-E-DEC-003 cycle 52 queue 44 data-fetched
//! ```
//!
//! A manifest is judged on every chip: each line then names its chip, and the program runs at the shortest and at the
//! longest `c2c.link_latency` of its profile on every link (the receive rules need the longest, the overrun rule the
//! shortest), so a line also names the latencies at which it occurs. Those two runs decide the whole latency domain of the
//! specification (every directed link with its own latency, varying per transfer) whenever the schedule does not depend on a
//! latency: the only instruction whose timing does is DeskewRecv, which is released at the first epoch boundary after its
//! vector arrives, and the arrival moves with the latency. The runs therefore also record where every DeskewRecv was
//! released. When the two runs release one at different cycles, the schedule of the program depends on the latencies, and
//! a hazard can need one link fast and another slow, or a latency between the bounds (the epoch boundary falls inside the
//! range). `verify` then runs every combination of the two bounds over the directed links the program sends on (the
//! ones that name a hazard are printed with `link-latency C:L=N+...`, chip, link, latency) up to
//! [`MAX_CORNER_LINKS`] links; a hazard found is exit 1, and a clean result is still undecided (exit 3): the latencies
//! between the bounds and the per-transfer jitter were not judged. A lone object that sends or receives on a C2C link is
//! refused (exit 2): the vectors it receives come from other chips, so the manifest of the program is the unit that can be
//! judged. A lone object that only talks to the host (PciSend, PciReceive) is judged as it is.
//!
//! The class of a code is the `Class` column of `microarchitecture.md` section 8 ([`SCHEDULE_CODES`] is its `schedule`
//! set; a test reads the table). A DEC-003 of a unit that did not come from the object's own image (fetched from SRAM, or
//! injected by the host) is `data-fetched`: its units decode to whatever the fetched words are.
//!
//! A queue that ends the run parked for good with text it never dispatched (a HostWait no doorbell released, a
//! DeskewRecv whose vector never arrived, a Sync nothing releases) is printed as `parked queue Q cycle C CAUSE`: what lies
//! behind the wait was not judged. `--host FILE` takes the `H` lines (and `C` lines) of the harness, so a producer verifies
//! under its intended doorbell and DMA timing; `--allow-parked` accepts the parked text.
//!
//! The exit status is three-valued: 0 when the program is clean (no schedule code and no data code but the end-of-run
//! IQ-004, nothing parked unless `--allow-parked`), 1 when a schedule code occurs that no data code precedes, or text is
//! parked, and 3 when the verdict is undecided: a data code occurred and no schedule code precedes it, so the schedule codes
//! that follow depend on data this run did not have, and a schedule code that the data code hid (the instruction it
//! suppressed would have raised one) was never raised either (give `--weights`); or text fetched from SRAM did not decode
//! (a `data-fetched` DEC-003) and no schedule code was raised, so what that text would have dispatched was never judged
//! without its data; or the schedule of a manifest depends on the link latencies and no hazard was found at the latencies
//! that were run (above); or the run needs more than `--max-cycles` (the option of `tsp-iss run`, passed on) and no hazard was found by
//! the runs before it (a hazard already found is exit 1 whatever a later run does). The message
//! names the first data code and how to supply the data. 2 is
//! an input error (a missing or malformed object, a bad option, a lone object with C2C traffic) and 4 an internal error.
//! There is no second implementation of the checks: the schedule of a program is judged by the ISS itself.
//!
//! The run records its diagnostics through `tsp-iss run --obs-out DIR --obs-diagnostics` into a scratch directory under
//! the system's temporary directory, which the run creates at its end and which is removed afterwards: no trace is
//! produced, so a verification costs little more than the run.
use crate::harness::PROG;
use serde_json::Value;
use std::ffi::{OsStr, OsString};
use std::io::Write;
use std::path::PathBuf;

/// The usage line.
pub const USAGE: &str = "usage: tsp-iss verify <obj.elf | manifest.otsm> [--weights W.safetensors ...] [--config C.json] [--profile P] [--host FILE] [--topology FILE] [--max-cycles N] [--allow-parked]";

/// The codes of class `schedule`, the `schedule` entries of the `Class` column of the diagnostic table in
/// `microarchitecture.md` section 8: raised by a rule that reads only the text, the configuration and the timing state
/// (the host script and the traffic of other chips included), so a run without data finds each of them if the program has it.
/// `tests/verify.rs` reads the table and compares. The `data` codes (IQ-002, IQ-004, SXM-001) and the `fault` codes
/// (ECC-001, ECC-002, ECC-003, LNK-004, which need an injected fault) are printed and never fail a program.
pub const SCHEDULE_CODES: [&str; 17] = [
    "OTS-E-DEC-003",
    "OTS-E-QUE-001",
    "OTS-E-QUE-002",
    "OTS-E-IQ-003",
    "OTS-E-TIM-001",
    "OTS-E-DIR-001",
    "OTS-E-SRF-002",
    "OTS-E-MEM-001",
    "OTS-E-OPR-001",
    "OTS-E-OPR-002",
    "OTS-E-RES-001",
    "OTS-E-RES-002",
    "OTS-E-RES-003",
    "OTS-E-LNK-001",
    "OTS-E-LNK-002",
    "OTS-E-LNK-003",
    "OTS-E-HIO-001",
];

/// The end-of-run record: it reports text left in a queue after the machine has stopped, so it removes no effect of any
/// run-time instruction and decides nothing.
const IQ_END_OF_RUN: &str = "OTS-E-IQ-004";

/// The codes of class `fault`.
pub const FAULT_CODES: [&str; 4] = ["OTS-E-ECC-001", "OTS-E-ECC-002", "OTS-E-ECC-003", "OTS-E-LNK-004"];

/// Is `code` a diagnostic of class `schedule` (in the table; see [`classify`] for the origin of a DEC-003)?
pub fn is_schedule_code(code: &str) -> bool {
    SCHEDULE_CODES.contains(&code)
}

/// What a diagnostic says about the program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// The program breaks a placement or scheduling rule.
    Schedule,
    /// A rule that reads data values raised it (`IQ-002`, `IQ-004`, `SXM-001`).
    Data,
    /// A DEC-003 of a unit that was fetched from SRAM or injected: data by origin.
    DataFetched,
    /// An integrity event; it needs an injected fault.
    Fault,
}

impl Class {
    /// The word printed at the end of the line.
    pub fn label(self) -> &'static str {
        match self {
            Class::Schedule => "schedule",
            Class::Data => "data-dependent",
            Class::DataFetched => "data-fetched",
            Class::Fault => "fault",
        }
    }

    fn is_data(self) -> bool {
        matches!(self, Class::Data | Class::DataFetched)
    }
}

/// One diagnostic of a run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub cycle: u64,
    pub queue: u64,
    pub code: String,
    /// The unit came from a fetch or the host, not from the object's image.
    pub fetched: bool,
    /// The instruction that raised it makes it data (a Gather or Scatter tile address, an `sxselect` select).
    pub data: bool,
}

/// The class of a diagnostic: the table's class of its code, except that a fetched DEC-003 is data, and so is an OPR-001
/// that the ISS marks as raised by a Gather or Scatter tile address or an `sxselect` select.
pub fn classify(d: &Diagnostic) -> Class {
    if d.data {
        Class::Data
    } else if d.fetched && d.code == "OTS-E-DEC-003" {
        Class::DataFetched
    } else if is_schedule_code(&d.code) {
        Class::Schedule
    } else if FAULT_CODES.contains(&d.code.as_str()) {
        Class::Fault
    } else {
        Class::Data
    }
}

/// A queue that ended the run parked with undispatched text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parked {
    pub queue: u64,
    pub since: u64,
    pub cause: String,
}

/// What one chip's part of a run's `diagnostics.json` holds: the listed diagnostics in report order, the total the run
/// reported, whether a schedule-class diagnostic occurs among all of them (the counts per code are complete even when the
/// list is cut) and the parked queues.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChipFound {
    pub chip: u64,
    pub listed: Vec<Diagnostic>,
    pub total: u64,
    pub schedule: bool,
    /// How many diagnostics came from fetched or injected text (all codes, the list may be cut).
    pub fetched: u64,
    /// A code of class `data` (the end-of-run record IQ-004 excepted) that the counts per code show, whether or not the
    /// list holds it: the list may be cut.
    pub data: Option<String>,
    pub parked: Vec<Parked>,
    /// Every DeskewRecv that found its vector: queue, cycle of dispatch, cycle of release.
    pub released: Vec<(u64, u64, u64)>,
    /// The links the chip sent on (Send, Transmit) and named in a Receive or DeskewRecv.
    pub sent: Vec<u64>,
    pub received: Vec<u64>,
}

/// The diagnostics of every chip of one run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Found {
    pub chips: Vec<ChipFound>,
}

fn read_chip(chip: &Value) -> Option<ChipFound> {
    let fetched = chip.get("by_code_fetched").and_then(Value::as_object);
    let by_data = chip.get("by_code_data").and_then(Value::as_object);
    let listed = chip
        .get("diagnostics")?
        .as_array()?
        .iter()
        .map(|d| {
            Some(Diagnostic {
                cycle: d.get("cycle")?.as_u64()?,
                queue: d.get("queue")?.as_u64()?,
                code: d.get("code")?.as_str()?.to_string(),
                fetched: d.get("fetched").and_then(Value::as_bool).unwrap_or(false),
                data: d.get("data").and_then(Value::as_bool).unwrap_or(false),
            })
        })
        .collect::<Option<Vec<_>>>()?;
    // a DEC-003 that came from fetched text, and an OPR-001 that the instruction makes data, are not schedule codes: take those
    // out of the count of their code
    let schedule = chip.get("by_code")?.as_object()?.iter().any(|(code, n)| {
        let count = |m: Option<&serde_json::Map<String, Value>>| {
            m.and_then(|f| f.get(code)).and_then(Value::as_u64).unwrap_or(0)
        };
        is_schedule_code(code) && n.as_u64().unwrap_or(0) > count(fetched) + count(by_data)
    });
    let fetched_total: u64 = fetched.map_or(0, |f| f.values().filter_map(Value::as_u64).sum());
    // a code of class `data` by its table entry, or one that the instruction makes data (an OPR-001 of a Gather or Scatter tile
    // address or an `sxselect` select: its count is in `by_code_data`)
    let data = chip.get("by_code")?.as_object()?.iter().find_map(|(code, n)| {
        let probe = Diagnostic { cycle: 0, queue: 0, code: code.clone(), fetched: false, data: false };
        let by_instruction = by_data.and_then(|m| m.get(code)).and_then(Value::as_u64).unwrap_or(0) > 0;
        (n.as_u64().unwrap_or(0) > 0 && (classify(&probe) == Class::Data || by_instruction) && code != IQ_END_OF_RUN)
            .then(|| code.clone())
    });
    let parked = match chip.get("parked") {
        None => Vec::new(),
        Some(p) => p
            .as_array()?
            .iter()
            .map(|p| {
                Some(Parked {
                    queue: p.get("queue")?.as_u64()?,
                    since: p.get("since")?.as_u64()?,
                    cause: p.get("cause")?.as_str()?.to_string(),
                })
            })
            .collect::<Option<Vec<_>>>()?,
    };
    let released = match chip.get("deskew_released") {
        None => Vec::new(),
        Some(r) => r
            .as_array()?
            .iter()
            .map(|e| {
                let e = e.as_array()?;
                Some((e.first()?.as_u64()?, e.get(1)?.as_u64()?, e.get(2)?.as_u64()?))
            })
            .collect::<Option<Vec<_>>>()?,
    };
    let links = |key: &str| -> Option<Vec<u64>> {
        match chip.get(key) {
            None => Some(Vec::new()),
            Some(l) => l.as_array()?.iter().map(Value::as_u64).collect(),
        }
    };
    Some(ChipFound {
        chip: chip.get("chip")?.as_u64()?,
        listed,
        total: chip.get("total")?.as_u64()?,
        schedule,
        fetched: fetched_total,
        data,
        parked,
        released,
        sent: links("links_sent")?,
        received: links("links_received")?,
    })
}

/// Read the `diagnostics.json` that `--obs-diagnostics` writes; `None` when it is not that document.
pub fn read_diagnostics(text: &str) -> Option<Found> {
    let doc: Value = serde_json::from_str(text).ok()?;
    let chips = doc.get("chips")?.as_array()?.iter().map(read_chip).collect::<Option<Vec<_>>>()?;
    (!chips.is_empty()).then_some(Found { chips })
}

/// The verdict on one run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// No schedule code and no data code (the end-of-run record IQ-004 excepted).
    Clean,
    /// A schedule code that no data code precedes.
    Hazard,
    /// A data code that removes an effect occurred and no schedule code precedes it: the schedule codes that the run raised
    /// after it can be its cascade, and the schedule codes it hid (the instruction that would have raised one did not
    /// take effect) were never raised. `schedule` is the first schedule code that followed, `None` when there is none, or
    /// when the list was cut before it: the run did not have the data it depends on.
    Undecided { chip: u64, data: Diagnostic, schedule: Option<(u64, Diagnostic)> },
}

/// Judge one run. A data code precedes a schedule code when it was reported earlier on the same chip or at an earlier
/// cycle on another chip (a vector needs at least a cycle to cross a link); the end-of-run record IQ-004 removes no effect
/// and decides nothing.
///
/// A run is decided only when no data code occurred before its first schedule code. A data code that removes an effect (a
/// Permute that did not run because its map is not bijective, a unit that was lost or did not decode) can hide the schedule
/// code that the lost effect would have raised, so a run with a data code and no schedule code after it is undecided as
/// well: whatever it did not raise depends on data this run did not have.
pub fn judge(found: &Found) -> Verdict {
    // the first schedule code: by cycle, then chip, then report order
    let mut first: Option<(u64, u64, usize, &Diagnostic)> = None;
    for c in &found.chips {
        for (i, d) in c.listed.iter().enumerate() {
            if classify(d) == Class::Schedule && first.is_none_or(|(cy, ch, ix, _)| (d.cycle, c.chip, i) < (cy, ch, ix))
            {
                first = Some((d.cycle, c.chip, i, d));
            }
        }
    }
    let cut = found.chips.iter().any(|c| c.schedule) && first.is_none();
    // the earliest data code that precedes it (any data code when there is no schedule code, or when it is out of the list)
    let mut data: Option<(u64, u64, usize, &Diagnostic)> = None;
    for c in &found.chips {
        for (i, d) in c.listed.iter().enumerate() {
            if !classify(d).is_data() || d.code == IQ_END_OF_RUN {
                continue;
            }
            let precedes = match first {
                None => true,
                Some((cy, ch, ix, _)) => d.cycle < cy || (d.cycle == cy && c.chip == ch && i < ix),
            };
            if precedes && data.is_none_or(|(cy, ch, ix, _)| (d.cycle, c.chip, i) < (cy, ch, ix)) {
                data = Some((d.cycle, c.chip, i, d));
            }
        }
    }
    if let Some((_, chip, _, d)) = data {
        return Verdict::Undecided { chip, data: d.clone(), schedule: first.map(|(_, ch, _, s)| (ch, s.clone())) };
    }
    if first.is_some() {
        return Verdict::Hazard;
    }
    // no listed schedule code and no listed data code: the counts decide for what the list does not show. A data code that
    // is counted but not listed (the list was cut) may precede a schedule code that is counted but not listed, and text
    // fetched from SRAM that did not decode was never judged; the stated code is a stand-in for the record that is not listed.
    let unlisted = found.chips.iter().find_map(|c| match (&c.data, c.fetched > 0) {
        (Some(code), _) => {
            Some((c.chip, Diagnostic { cycle: 0, queue: 0, code: code.clone(), fetched: false, data: false }))
        }
        (None, true) => {
            Some((c.chip, Diagnostic { cycle: 0, queue: 0, code: "OTS-E-DEC-003".into(), fetched: true, data: false }))
        }
        (None, false) => None,
    });
    match (unlisted, cut) {
        (Some((chip, data)), _) => Verdict::Undecided { chip, data, schedule: None },
        (None, true) => Verdict::Hazard,
        (None, false) => Verdict::Clean,
    }
}

fn usage(stderr: &mut dyn Write) -> i32 {
    let _ = writeln!(stderr, "{PROG}: verify: {USAGE}");
    2
}

/// The name of a scratch directory that the run creates at its end (`--obs-out` makes it) and that is removed when this
/// goes out of scope; an interrupt during the run therefore leaves nothing behind.
struct Scratch(PathBuf);

impl Scratch {
    fn new(n: usize) -> Scratch {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        Scratch(std::env::temp_dir().join(format!("tsp-iss-verify-{}-{nanos}-{n}", std::process::id())))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The most directed links whose latency bounds `verify` combines (2^N runs) when a schedule depends on the latencies.
pub const MAX_CORNER_LINKS: usize = 10;

/// How the program is run: whether it is a manifest (every line names its chip and latencies) and the shortest and longest
/// link latency of its profile (`None` for a lone object, which runs at its profile's own).
struct Plan {
    manifest: bool,
    bounds: Option<(u64, u64)>,
}

/// The profile the program is for: `--config`, else `--profile`, else the name the object or manifest carries.
fn profile_of(config: Option<&OsStr>, profile: Option<&str>, name: &str) -> Result<ots_profile::Profile, String> {
    match (config, profile) {
        (Some(c), _) => crate::native::load_file(c),
        (None, Some(p)) => crate::native::load_named(p),
        (None, None) => crate::native::load_named(name),
    }
}

/// Look at the object or manifest: a manifest runs at the shortest and the longest link latency of its profile. What
/// cannot be read here is left to the run, which reports it with its own message and status.
fn plan(object: &OsStr, config: Option<&OsStr>, profile: Option<&str>) -> Plan {
    let plain = Plan { manifest: false, bounds: None };
    let Ok(data) = std::fs::read(object) else { return plain };
    if !data.starts_with(b"OTSM") {
        return plain;
    }
    let config_str = config.and_then(|c| c.to_str());
    let Ok((name, _)) = ots_format::read_manifest(&data, profile, config_str) else { return plain };
    let Ok(p) = profile_of(config, profile, &name) else { return plain };
    let g = crate::geom::Geom::new(&p);
    Plan { manifest: true, bounds: Some((g.link_latency_min, g.link_latency_max)) }
}

/// The latencies of one run: every link at one value, or the listed directed links at the given values.
#[derive(Clone, Debug)]
enum Setting {
    Plain,
    Uniform(u64),
    /// Directed links `(chip, link, latency)`.
    Links(Vec<(u64, u64, u64)>),
}

impl Setting {
    fn args(&self) -> Vec<OsString> {
        match self {
            Setting::Plain => Vec::new(),
            Setting::Uniform(l) => vec!["--link-latency".into(), l.to_string().into()],
            Setting::Links(m) => {
                let list: Vec<String> = m.iter().map(|(c, l, n)| format!("{c}:{l}={n}")).collect();
                vec!["--link-latency-map".into(), list.join(",").into()]
            }
        }
    }

    /// What a line names after `link-latency`: the value, or the directed links with their values.
    fn label(&self) -> Option<String> {
        match self {
            Setting::Plain => None,
            Setting::Uniform(l) => Some(l.to_string()),
            Setting::Links(m) => Some(m.iter().map(|(c, l, n)| format!("{c}:{l}={n}")).collect::<Vec<_>>().join("+")),
        }
    }
}

/// One line of output with the latencies at which it occurred.
struct Line {
    text: String,
    latencies: Vec<String>,
}

fn add_line(lines: &mut Vec<Line>, text: String, latency: Option<String>) {
    let at = match lines.iter().position(|l| l.text == text) {
        Some(i) => i,
        None => {
            lines.push(Line { text, latencies: Vec::new() });
            lines.len() - 1
        }
    };
    if let Some(l) = latency {
        lines[at].latencies.push(l);
    }
}

/// What the runs found so far.
#[derive(Default)]
struct Collected {
    diag_lines: Vec<Line>,
    parked_lines: Vec<Line>,
    hazard: bool,
    undecided: Option<Verdict>,
    unlisted: u64,
}

impl Collected {
    fn add(&mut self, found: &Found, manifest: bool, setting: &Setting) {
        let latency = setting.label().filter(|_| manifest);
        for c in &found.chips {
            let who = if manifest { format!(" chip {}", c.chip) } else { String::new() };
            for d in &c.listed {
                let text = format!("{}{who} cycle {} queue {} {}", d.code, d.cycle, d.queue, classify(d).label());
                add_line(&mut self.diag_lines, text, latency.clone());
            }
            for p in &c.parked {
                let text = format!("parked{who} queue {} cycle {} {}", p.queue, p.since, p.cause);
                add_line(&mut self.parked_lines, text, latency.clone());
            }
            self.unlisted += c.total.saturating_sub(c.listed.len() as u64);
        }
        match judge(found) {
            Verdict::Clean => {}
            Verdict::Hazard => self.hazard = true,
            v => {
                self.undecided.get_or_insert(v);
            }
        }
    }

    /// Print the lines (and the count of those not listed); `false` when standard output cannot be written.
    fn print(&self, stdout: &mut dyn Write, stderr: &mut dyn Write) -> bool {
        for l in self.diag_lines.iter().chain(&self.parked_lines) {
            let mut text = l.text.clone();
            if !l.latencies.is_empty() {
                text += &format!(" link-latency {}", l.latencies.join(","));
            }
            if writeln!(stdout, "{text}").is_err() {
                let _ = writeln!(stderr, "{PROG}: verify: cannot write standard output");
                return false;
            }
        }
        if self.unlisted > 0 {
            let _ = writeln!(stderr, "{PROG}: verify: {} more diagnostics are counted but not listed", self.unlisted);
        }
        true
    }
}

/// Where each DeskewRecv was released and what ended the run parked, per chip: the timing that a link latency can change.
type Signature = Vec<(u64, Vec<(u64, u64, u64)>, Vec<(u64, u64, String)>)>;

fn signature(found: &Found) -> Signature {
    found
        .chips
        .iter()
        .map(|c| {
            let mut r = c.released.clone();
            r.sort_unstable();
            let mut p: Vec<_> = c.parked.iter().map(|p| (p.queue, p.since, p.cause.clone())).collect();
            p.sort();
            (c.chip, r, p)
        })
        .collect()
}

/// A DeskewRecv that is released at different cycles in two runs (`None`: never).
struct Moved {
    chip: u64,
    queue: u64,
    since: u64,
    first: Option<u64>,
    second: Option<u64>,
}

/// The first place where two runs differ in the release of a DeskewRecv, or in what ended the run parked.
fn first_difference(a: &Found, b: &Found) -> Option<Moved> {
    for (ca, cb) in a.chips.iter().zip(&b.chips) {
        let key = |r: &(u64, u64, u64)| (r.0, r.1);
        let mut keys: Vec<(u64, u64)> = ca.released.iter().chain(&cb.released).map(key).collect();
        keys.sort_unstable();
        keys.dedup();
        for (queue, since) in keys {
            let at = |c: &ChipFound| c.released.iter().find(|r| key(r) == (queue, since)).map(|r| r.2);
            let (first, second) = (at(ca), at(cb));
            if first != second {
                return Some(Moved { chip: ca.chip, queue, since, first, second });
            }
        }
        if ca.parked != cb.parked {
            let p = ca.parked.first().or(cb.parked.first())?;
            return Some(Moved { chip: ca.chip, queue: p.queue, since: p.since, first: None, second: None });
        }
    }
    None
}

/// Run the object once at `setting`; `Err(code)` is the run's own failure, its message already on `stderr`.
fn one_run(
    run_args: &[OsString],
    object: &OsStr,
    setting: &Setting,
    n: usize,
    stderr: &mut dyn Write,
) -> Result<Result<Found, ()>, i32> {
    let scratch = Scratch::new(n);
    let mut one = run_args.to_vec();
    one.extend(setting.args());
    one.extend(["--obs-out".into(), scratch.0.clone().into_os_string(), "--obs-diagnostics".into()]);
    one.push(object.to_os_string());
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let code = crate::native::run(&one, &mut out, &mut err);
    if code != 0 {
        // the run's own message, under this command's name
        let msg = String::from_utf8_lossy(&err);
        let msg = msg.replace(&format!("{PROG}: run: "), &format!("{PROG}: verify: "));
        // every message of the run carries this command's name (the --max-cycles one has none of its own)
        let msg = match msg.strip_prefix(&format!("{PROG}: ")) {
            Some(rest) if !rest.starts_with("verify: ") => format!("{PROG}: verify: {rest}"),
            _ => msg.clone(),
        };
        let _ = stderr.write_all(msg.as_bytes());
        return Err(code);
    }
    let found = std::fs::read_to_string(scratch.0.join("diagnostics.json")).ok().and_then(|t| read_diagnostics(&t));
    Ok(found.ok_or(()))
}

/// `args` are the arguments after `verify`.
pub fn verify(args: &[OsString], stdout: &mut dyn Write, stderr: &mut dyn Write) -> i32 {
    let mut object: Option<&OsStr> = None;
    let mut run_args: Vec<OsString> = Vec::new();
    let (mut config, mut profile): (Option<&OsStr>, Option<String>) = (None, None);
    let mut allow_parked = false;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_os_str();
        if a == "--weights"
            || a == "--config"
            || a == "--profile"
            || a == "--host"
            || a == "--max-cycles"
            || a == "--topology"
        {
            let Some(v) = args.get(i + 1) else { return usage(stderr) };
            if a == "--config" {
                config = Some(v.as_os_str());
            } else if a == "--profile" {
                profile = v.to_str().map(String::from);
            }
            run_args.push(a.to_os_string());
            run_args.push(v.clone());
            i += 2;
        } else if a == "--allow-parked" {
            allow_parked = true;
            i += 1;
        } else if a.to_string_lossy().starts_with("--") || object.is_some() {
            return usage(stderr);
        } else {
            object = Some(a);
            i += 1;
        }
    }
    let Some(object) = object else { return usage(stderr) };
    // the advice of an undecided verdict: asking for the data is wrong when --weights was given
    let weights_given = run_args.iter().any(|a| a == "--weights");
    let give = if weights_given {
        "--weights was given: the code was raised on that data, and what it removed or what follows it is not judged"
    } else {
        "give --weights for the program's data"
    };
    let plan = plan(object, config, profile.as_deref());

    let mut acc = Collected::default();
    let mut runs: Vec<(Setting, Found)> = Vec::new();
    let settings: Vec<Setting> = match plan.bounds {
        Some((lo, hi)) if lo != hi => vec![Setting::Uniform(lo), Setting::Uniform(hi)],
        Some((lo, _)) => vec![Setting::Uniform(lo)],
        None => vec![Setting::Plain],
    };
    // a failing run still shows what the earlier runs found
    macro_rules! run {
        ($setting:expr) => {{
            let setting: Setting = $setting;
            match one_run(&run_args, object, &setting, runs.len(), stderr) {
                Err(code) => {
                    let printed = acc.print(stdout, stderr);
                    // a hazard already found decides the object (exit 1), even when a later run exceeds --max-cycles
                    return if code == 3 && printed && acc.hazard { 1 } else { code };
                }
                Ok(Err(())) => {
                    acc.print(stdout, stderr);
                    let _ = writeln!(stderr, "{PROG}: verify: the run left no readable diagnostics.json");
                    return 4;
                }
                Ok(Ok(found)) => {
                    acc.add(&found, plan.manifest, &setting);
                    runs.push((setting, found));
                }
            }
        }};
    }
    for s in settings {
        run!(s);
    }
    if !plan.manifest && runs[0].1.chips.iter().any(|c| !c.sent.is_empty() || !c.received.is_empty()) {
        let _ = writeln!(
            stderr,
            "{PROG}: verify: {}: the object sends or receives on a C2C link: the vectors it receives and the links it \
             sends on belong to other chips, so it cannot be judged alone; give the manifest of the program \
             (ots-asm mpack -o M.otsm <chip0.elf> <chip1.elf> ...)",
            object.to_string_lossy()
        );
        return 2;
    }

    // the schedule depends on the latencies when the two bounds release a DeskewRecv at different cycles
    let mut dependent: Option<String> = None;
    if let (Some((lo, hi)), true) = (plan.bounds.filter(|(l, h)| l != h), runs.len() == 2) {
        if signature(&runs[0].1) != signature(&runs[1].1) {
            let (found_lo, found_hi) = (&runs[0].1, &runs[1].1);
            let mut links: Vec<(u64, u64)> = Vec::new();
            for f in [found_lo, found_hi] {
                for c in &f.chips {
                    links.extend(c.sent.iter().map(|l| (c.chip, *l)));
                }
            }
            links.sort_unstable();
            links.dedup();
            let when = |c: Option<u64>| c.map_or("never".to_string(), |c| format!("at cycle {c}"));
            let what = match first_difference(found_lo, found_hi) {
                Some(Moved { chip, queue, since, first, second }) if first != second => format!(
                    "chip {chip} queue {queue} (a DeskewRecv dispatched at cycle {since}) is released {} with every link at {lo} and {} with every link at {hi}",
                    when(first),
                    when(second)
                ),
                Some(Moved { chip, queue, since, .. }) => {
                    format!("chip {chip} queue {queue} parks at cycle {since} at one of the bounds only")
                }
                None => "the text parked at the two bounds differs".to_string(),
            };
            let done = if links.len() > MAX_CORNER_LINKS {
                format!(
                    "the {} links in use are more than the {MAX_CORNER_LINKS} whose bounds are combined, so only the two uniform runs were made",
                    links.len()
                )
            } else {
                let k = links.len();
                for mask in 1..(1u64 << k).saturating_sub(1) {
                    let map = links
                        .iter()
                        .enumerate()
                        .map(|(b, (c, l))| (*c, *l, if mask >> b & 1 == 1 { hi } else { lo }))
                        .collect();
                    run!(Setting::Links(map));
                }
                format!(
                    "every combination of {lo} and {hi} over the {k} link{} in use was run",
                    if k == 1 { "" } else { "s" }
                )
            };
            dependent = Some(format!(
                "undecided: the schedule depends on the link latencies: {what}; {done}, but the latencies between the bounds and the per-transfer jitter were not judged"
            ));
        }
    }

    if !acc.print(stdout, stderr) {
        return 2;
    }
    let parked = !acc.parked_lines.is_empty();
    if acc.hazard {
        return 1;
    }
    if parked && !allow_parked {
        let _ = writeln!(
            stderr,
            "{PROG}: verify: text is parked behind a wait that nothing released, so what follows it was not judged; give \
             --host FILE (H lines that ring the doorbell or deliver the vector) or --allow-parked"
        );
        return 1;
    }
    if let Some(Verdict::Undecided { chip, data, schedule }) = acc.undecided {
        let at = |d: &Diagnostic| format!("{} (chip {chip}) at cycle {} queue {}", d.code, d.cycle, d.queue);
        if schedule.is_none() && classify(&data) == Class::DataFetched {
            let _ = writeln!(
                stderr,
                "{PROG}: verify: undecided: text fetched from SRAM did not decode ({}): its words are data, so the text \
                 could not be judged without them ({})",
                at(&data),
                if weights_given { give } else { "give the data with --weights (the package that holds the text)" }
            );
            return 3;
        }
        let Some((c, s)) = schedule else {
            let _ = writeln!(
                stderr,
                "{PROG}: verify: undecided: the data code {} removed or changed an effect that depends on data this run did not \
                 have, so a schedule code it hid was never raised ({give})",
                at(&data)
            );
            return 3;
        };
        let later = format!("{} (chip {c}) at cycle {} queue {}", s.code, s.cycle, s.queue);
        let _ = writeln!(
            stderr,
            "{PROG}: verify: undecided: the data code {} precedes the first schedule code, {later}, which depends on the \
             data this run did not have ({give})",
            at(&data)
        );
        return 3;
    }
    if let Some(m) = dependent {
        let _ = writeln!(stderr, "{PROG}: verify: {m}");
        return 3;
    }
    0
}
