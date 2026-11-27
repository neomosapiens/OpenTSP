//! `tsp-iss vectors`: run the golden vectors of a conformance file in-process.
//!
//! ```text
//! tsp-iss vectors <conformance.json> [--config F] [--only NAME ...] [--jobs N]
//! ```
//!
//! The file is `opentsp.conformance.model` (`opentsp-arch-spec/conformance/model/<profile>.json`):
//! each vector gives the initial IQ text per queue, the SRAM words to preload and what a run must
//! leave: the diagnostics in order (cycle, queue, code), the dumped words (`vlanes` bytes) and the
//! trace: the `events` (the canonical `T srf` and `T sram` lines with their values, when the
//! vector lists them) and their digest `trace_sha256`. Each vector runs on one chip with every link
//! looped back at the nominal latency, as `tsp_sim` ran it. The file's `config_sha256` is the digest
//! of the configuration it was generated from ([`config_digest`]); a file whose digest differs from
//! the profile in use is refused.
//!
//! One line per vector in file order, `pass <name>` or `FAIL <name>: <why>`, then
//! `<n> vectors: <p> pass, <f> FAIL`. Exit 0 when all pass, 1 when any fails, 2 on a usage or input
//! error (an unreadable file, a file that is not a conformance file or is of another configuration, an
//! unknown `--only` name, a configuration `tsp_sim` would refuse). `--config` stands for the file of the profile the vectors
//! name (default: the shipped profile of that name). The output does not depend on `--jobs`.
//!
//! A panic while a vector runs fails that vector (`FAIL <name>: internal error: ...`) and the run goes on; the
//! panic hook also writes its line on standard error. `TSP_ISS_VECTORS_PANIC=<name>` makes the named vector panic
//! (a test of this path; no vector file can set it).
use super::{check_geom, fail, load_file, load_named};
use crate::geom::{Geom, MAX_LANES, NARROW_LANES};
use crate::harness::links::PlanLinks;
use crate::harness::output::TraceObserver;
use crate::harness::{strerror, PROG};
use crate::machine::Machine;
use crate::observe::{Observer, Row};
use crate::Diag;
use ots_profile::Profile;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::ffi::OsString;
use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;

const CMD: &str = "vectors";

const USAGE: &str = "usage: tsp-iss vectors <conformance.json> [--config F] [--only NAME ...] [--jobs N]";

/// The cycle budget of a vector (the harness default).
const MAX_CYCLES: u64 = crate::harness::cli::DEFAULT_MAX_CYCLES;

/// The diagnostics in order, and the trace lines as `tsp-iss --trace` writes them.
struct Collect<'a> {
    diags: Vec<(u64, u16, Diag)>,
    trace: TraceObserver<'a>,
    /// The host's external events (`T dma`, `T irq`, `T csr` without the `T `), in output order.
    ext: Vec<String>,
}

impl Observer for Collect<'_> {
    fn srf(&mut self, c: u64, row: Row, stream: u8, gap: u8, v: &[u8]) {
        self.trace.srf(c, row, stream, gap, v);
    }
    fn sram(&mut self, c: u64, slice: u8, addr: u32, v: &[u8]) {
        self.trace.sram(c, slice, addr, v);
    }
    fn diag(&mut self, c: u64, queue: u16, code: Diag) {
        self.diags.push((c, queue, code));
    }
    fn ecc(&mut self, e: &crate::ecc::EccEvent) {
        let q = if e.queue == 144 { 65535 } else { e.queue };
        self.ext.push(format!(
            "ecc {} queue {q} {} {}",
            e.cycle,
            if e.spare {
                "spare"
            } else if e.unc {
                "unc"
            } else {
                "corr"
            },
            e.place()
        ));
    }
    fn scrub(&mut self, c: u64, slice: u8, addr: u32, sl: u8, unc: bool) {
        self.ext.push(format!("ecc {c} scrub slice {slice} addr {addr} sl {sl}{}", if unc { " unc" } else { "" }));
    }
    fn mxm_corr(&mut self, c: u64, queue: u16, plane: u8, lane: u16) {
        self.ext.push(format!("ecc {c} queue {queue} mxm corr plane {plane} lane {lane}"));
    }
    fn link_fec(&mut self, c: u64, link: u8, unc: bool, flit: u8) {
        self.ext.push(format!("ecc {c} link {link} {} flit {flit}", if unc { "unc" } else { "corr" }));
    }
    fn host_dma(&mut self, c: u64, what: crate::observe::HostDma) {
        self.ext.push(format!("dma {c} {}", what.word()));
    }
    fn host_irq(&mut self, c: u64, vector: u8) {
        self.ext.push(format!("irq {c} msix {vector}"));
    }
    fn host_csr(&mut self, c: u64, addr: u64, value: u64) {
        self.ext.push(format!("csr {c} {addr:03x} {value:x}"));
    }
    fn host_csr_remote(&mut self, c: u64, chip: u16, addr: u64, value: u64) {
        self.ext.push(format!("csr {c} {chip} {addr:03x} {value:x}"));
    }
    fn host_pkt(&mut self, c: u64, ctrl: bool, link: u8) {
        self.ext.push(format!("pkt {c} {} {link}", if ctrl { "ctrl" } else { "csr" }));
    }
}

fn sha256_hex(b: &[u8]) -> String {
    Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
}

/// The `T srf` and `T sram` lines in canonical order (`tools/gen_model_vectors.py` `canonical_trace`;
/// `microarchitecture.md` §12): each cycle's run of consecutive `T srf` lines sorted by (row, stream,
/// gap), E before W; every other line keeps its order. Lines that are neither are dropped.
pub fn canonical_trace(text: &str) -> Vec<String> {
    let key = |l: &String| {
        let t: Vec<&str> = l.split(' ').collect();
        let (row, st) = t[3].split_once(':').unwrap_or((t[3], "0"));
        (row.to_string(), st.parse::<u64>().unwrap_or(0), t[5].parse::<u64>().unwrap_or(0))
    };
    let (mut out, mut run): (Vec<String>, Vec<String>) = (Vec::new(), Vec::new());
    let flush = |out: &mut Vec<String>, run: &mut Vec<String>| {
        run.sort_by_key(key);
        out.append(run);
    };
    for l in text.lines() {
        if l.starts_with("T srf ") {
            if run.first().is_some_and(|r| r.split(' ').nth(2) != l.split(' ').nth(2)) {
                flush(&mut out, &mut run);
            }
            run.push(l.to_string());
        } else {
            flush(&mut out, &mut run);
            if l.starts_with("T sram ") {
                out.push(l.to_string());
            }
        }
    }
    flush(&mut out, &mut run);
    out
}

/// The digest of a configuration: SHA-256 (hex) of its JSON with sorted keys and no spaces
/// (`gen_model_vectors.py` `config_digest`), so it does not depend on how the file is written.
pub fn config_digest(profile: &Profile) -> String {
    fn canon(v: &Value, out: &mut String) {
        match v {
            Value::Object(m) => {
                let mut keys: Vec<&String> = m.keys().collect();
                keys.sort();
                out.push('{');
                for (i, k) in keys.into_iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&Value::String(k.clone()).to_string());
                    out.push(':');
                    canon(&m[k], out);
                }
                out.push('}');
            }
            Value::Array(a) => {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    canon(x, out);
                }
                out.push(']');
            }
            x => out.push_str(&x.to_string()),
        }
    }
    let mut s = String::new();
    canon(profile.value(), &mut s);
    sha256_hex(s.as_bytes())
}

/// `Ok` when the file's `config_sha256` is the digest of `profile`.
pub fn check_config_digest(doc: &Value, profile: &Profile) -> Result<(), String> {
    let got = config_digest(profile);
    match doc["config_sha256"].as_str() {
        Some(want) if want == got => Ok(()),
        Some(want) => Err(format!("config_sha256 {want} is not the digest {got} of the configuration in use: the vectors are of another configuration")),
        None => Err("no config_sha256: not a conformance file".to_string()),
    }
}

fn hex_bytes(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) || !s.is_ascii() {
        return None;
    }
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()).collect()
}

/// `"slice:addr"` (decimal).
fn key_pair(k: &str) -> Option<(u8, u32)> {
    let (a, b) = k.split_once(':')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

fn show_diags(d: &[(u64, u16, String)]) -> String {
    let items: Vec<String> = d.iter().map(|(c, q, code)| format!("{code}@{c}/q{q}")).collect();
    format!("[{}]", items.join(", "))
}

/// Run one vector on a fresh machine and compare everything it lists (diagnostics, words, `events`,
/// `trace_sha256`); `Err(why)` when it does not match.
pub fn check_vector(profile: &Profile, geom: &Geom, v: &Value) -> Result<(), String> {
    if geom.vlanes as usize <= NARROW_LANES {
        check_vector_lanes::<NARROW_LANES>(profile, geom, v)
    } else {
        check_vector_lanes::<MAX_LANES>(profile, geom, v)
    }
}

/// [`check_vector`] on a machine of `L` lanes (`L` of at least the profile's `vlanes`); the verdict does not depend on `L`.
pub fn check_vector_lanes<const L: usize>(profile: &Profile, geom: &Geom, v: &Value) -> Result<(), String> {
    let bad = |what: &str| format!("malformed vector: {what}");
    let text = v["text"].as_object().ok_or_else(|| bad("text"))?;
    let sram = v["sram"].as_object().ok_or_else(|| bad("sram"))?;
    let expect = v["expect"].as_object().ok_or_else(|| bad("expect"))?;
    let diags = expect.get("diagnostics").and_then(Value::as_array).ok_or_else(|| bad("expect.diagnostics"))?;
    let words = expect.get("words").and_then(Value::as_object).ok_or_else(|| bad("expect.words"))?;

    let port = PlanLinks::loopback(geom);
    let link_state = port.state();
    let mut m = Machine::<L>::with_lanes(profile, 0, Box::new(port)).map_err(|e| format!("machine: {e}"))?;
    for (q, units) in text {
        let q: u16 = q.parse().map_err(|_| bad("queue number"))?;
        let units: Option<Vec<u32>> = units
            .as_str()
            .ok_or_else(|| bad("text unit list"))?
            .split_whitespace()
            .map(|u| u32::from_str_radix(u, 16).ok())
            .collect();
        m.load_image(q, &units.ok_or_else(|| bad("text unit"))?).map_err(|e| format!("load: {e}"))?;
    }
    for (k, w) in sram {
        let (s, a) = key_pair(k).ok_or_else(|| bad("sram key"))?;
        let bytes = w.as_str().and_then(hex_bytes).ok_or_else(|| bad("sram word"))?;
        m.preload(s, a, &bytes).map_err(|e| format!("preload: {e}"))?;
    }
    // the host's script: the lines as the harness reads them (`H ...`, `R ...`)
    let host_lines: Vec<&str> = match v.get("host") {
        None | Some(Value::Null) => Vec::new(),
        Some(h) => h
            .as_array()
            .and_then(|a| a.iter().map(Value::as_str).collect::<Option<Vec<_>>>())
            .ok_or_else(|| bad("host"))?,
    };
    let hprog = crate::harness::input::parse(
        host_lines.join("\n").as_bytes(),
        crate::harness::input::Limits { geom, chips: 1, topo: None },
    )
    .map_err(|e| format!("host script: {e}"))?;
    link_state.lock().unwrap().set_faults(hprog.link_faults.iter().copied());
    let events: Vec<crate::host::HostEvent> =
        hprog.host.iter().map(|h| crate::host::HostEvent { cycle: h.cycle, kind: h.kind.clone() }).collect();
    let start = hprog.starts.first().copied().unwrap_or(0);
    let pkts = hprog
        .pkts
        .iter()
        .map(|p| crate::host::PktEvent { cycle: p.cycle, id: p.id, link: p.link, kind: p.kind })
        .collect();
    m.set_ctl(pkts);
    if !events.is_empty() || start != 0 {
        m.set_host_script(events, start);
    }
    let mut trace = Vec::new();
    let mut obs = Collect { diags: Vec::new(), trace: TraceObserver::new(&mut trace, true), ext: Vec::new() };
    m.run(MAX_CYCLES, &mut obs).map_err(|e| format!("run: {e}"))?;
    let run_diags = std::mem::take(&mut obs.diags);
    let mut ext = std::mem::take(&mut obs.ext);
    drop(obs);
    ext.extend(crate::harness::host_tail(&m, &hprog, 0).lines().map(str::to_string));

    let mut want = Vec::new();
    for d in diags {
        match (d["cycle"].as_u64(), d["queue"].as_u64(), d["code"].as_str()) {
            (Some(c), Some(q), Some(code)) => want.push((c, q as u16, code.to_string())),
            _ => return Err(bad("expect.diagnostics entry")),
        }
    }
    let got: Vec<(u64, u16, String)> = run_diags.iter().map(|(c, q, d)| (*c, *q, d.as_str().to_string())).collect();
    if got != want {
        return Err(format!("diagnostics: got {}, want {}", show_diags(&got), show_diags(&want)));
    }
    let vl = geom.vlanes as usize;
    for (k, w) in words {
        let (s, a) = key_pair(k).ok_or_else(|| bad("word key"))?;
        let want = w.as_str().ok_or_else(|| bad("word"))?;
        let word = m.sram_word(s, a).map_err(|e| format!("dump {k}: {e}"))?;
        let hex: String = word[..vl].iter().map(|b| format!("{b:02x}")).collect();
        if hex != want {
            return Err(format!("word {k}: got {hex}, want {want}"));
        }
    }
    let want_ext: Vec<&str> = match expect.get("ext_events") {
        None => Vec::new(),
        Some(e) => e
            .as_array()
            .and_then(|a| a.iter().map(Value::as_str).collect::<Option<Vec<_>>>())
            .ok_or_else(|| bad("expect.ext_events"))?,
    };
    if ext.iter().map(String::as_str).ne(want_ext.iter().copied()) {
        return Err(format!("ext_events: got {ext:?}, want {want_ext:?}"));
    }
    let events = canonical_trace(&String::from_utf8_lossy(&trace));
    if let Some(want) = expect.get("events") {
        let want: Option<Vec<&str>> = want.as_array().and_then(|a| a.iter().map(Value::as_str).collect());
        let want = want.ok_or_else(|| bad("expect.events"))?;
        let got: Vec<&str> = events.iter().map(|e| &e[2..]).collect();
        if got != want {
            return Err(format!(
                "events: got {} events, want {}: {}",
                got.len(),
                want.len(),
                first_difference(&got, &want)
            ));
        }
    }
    let sha = sha256_hex(events.iter().map(|e| format!("{e}\n")).collect::<String>().as_bytes());
    match expect.get("trace_sha256").and_then(Value::as_str) {
        Some(want) if want == sha => Ok(()),
        want => Err(format!("trace_sha256: got {sha}, want {want:?}")),
    }
}

/// Where two event lists first differ, abbreviated.
fn first_difference(got: &[&str], want: &[&str]) -> String {
    let cut = |s: &str| match s.char_indices().nth(60) {
        Some((i, _)) => format!("{}...", &s[..i]),
        None => s.to_string(),
    };
    match (0..got.len().max(want.len())).find(|&i| got.get(i) != want.get(i)) {
        Some(i) => format!("event {i}: got {:?}, want {:?}", got.get(i).map(|s| cut(s)), want.get(i).map(|s| cut(s))),
        None => String::new(),
    }
}

/// A vector's result with a panic turned into a failure of that vector. `inject`, the value of
/// `TSP_ISS_VECTORS_PANIC`, names a vector whose check panics (a test of the panic path; no vector file sets it).
fn run_guarded(profile: &Profile, geom: &Geom, v: &Value, inject: Option<&str>) -> Result<(), String> {
    match catch_unwind(AssertUnwindSafe(|| {
        if inject.is_some() && inject == v["name"].as_str() {
            panic!("injected panic in vector {}", inject.unwrap_or_default());
        }
        check_vector(profile, geom, v)
    })) {
        Ok(r) => r,
        Err(p) => {
            let msg = p
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            Err(format!("internal error: {}", msg.replace('\n', " ")))
        }
    }
}

struct Args {
    file: OsString,
    config: Option<OsString>,
    only: BTreeSet<String>,
    jobs: usize,
}

fn parse(args: &[OsString]) -> Result<Args, String> {
    let usage = || USAGE.to_string();
    let mut a = Args { file: OsString::new(), config: None, only: BTreeSet::new(), jobs: 1 };
    let mut file = None;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].to_str().ok_or_else(usage)?;
        let next = args.get(i + 1);
        match arg {
            "--config" => {
                a.config = Some(next.cloned().ok_or_else(usage)?);
                i += 1;
            }
            "--only" => {
                a.only.insert(next.and_then(|n| n.to_str()).ok_or_else(usage)?.to_string());
                i += 1;
            }
            "--jobs" => {
                let n = next
                    .and_then(|n| n.to_str())
                    .and_then(|n| n.parse::<usize>().ok())
                    .filter(|n| (1..=1024).contains(n))
                    .ok_or_else(|| "--jobs needs a number from 1 to 1024".to_string())?;
                a.jobs = n;
                i += 1;
            }
            _ if !arg.starts_with('-') && file.is_none() => file = Some(args[i].clone()),
            _ => return Err(usage()),
        }
        i += 1;
    }
    a.file = file.ok_or_else(usage)?;
    Ok(a)
}

/// `tsp-iss vectors ...`; `args` are the arguments after `vectors`.
pub fn vectors(args: &[OsString], stdout: &mut dyn Write, stderr: &mut dyn Write) -> i32 {
    let a = match parse(args) {
        Ok(a) => a,
        Err(m) => return fail(stderr, CMD, &m),
    };
    let path = a.file.to_string_lossy().into_owned();
    let data = match std::fs::read(&a.file) {
        Ok(d) => d,
        Err(e) => return fail(stderr, CMD, &format!("{path}: {}", strerror(&e))),
    };
    let doc: Value = match serde_json::from_slice(&data) {
        Ok(d) => d,
        Err(e) => return fail(stderr, CMD, &format!("{path}: not JSON: {e}")),
    };
    let (Some(pname), Some(all)) = (doc["profile"].as_str(), doc["vectors"].as_array()) else {
        return fail(stderr, CMD, &format!("{path}: not a conformance file (needs \"profile\" and \"vectors\")"));
    };
    let mut names = Vec::with_capacity(all.len());
    for v in all {
        match v["name"].as_str() {
            Some(n) => names.push(n),
            None => return fail(stderr, CMD, &format!("{path}: a vector has no name")),
        }
    }
    for n in &a.only {
        if !names.contains(&n.as_str()) {
            return fail(stderr, CMD, &format!("{path}: no vector named {n}"));
        }
    }
    let selected: Vec<&Value> =
        all.iter().zip(&names).filter(|(_, n)| a.only.is_empty() || a.only.contains(**n)).map(|(v, _)| v).collect();

    let (profile, origin) = match &a.config {
        Some(c) => (load_file(c), c.to_string_lossy().into_owned()),
        None => (load_named(pname), pname.to_string()),
    };
    let profile = match profile {
        Ok(p) => p,
        Err(m) => return fail(stderr, CMD, &m),
    };
    let geom = match check_geom(&profile, &origin) {
        Ok(g) => g,
        Err(m) => return fail(stderr, CMD, &m),
    };
    if let Err(m) = check_config_digest(&doc, &profile) {
        return fail(stderr, CMD, &format!("{path}: {m} ({origin})"));
    }

    // the workers run the vectors in any order; the lines are printed in file order as they complete
    let n = selected.len();
    let next = AtomicUsize::new(0);
    let inject = std::env::var("TSP_ISS_VECTORS_PANIC").ok();
    let (tx, rx) = mpsc::channel::<(usize, Result<(), String>)>();
    let mut failed = 0usize;
    let mut write_err = false;
    std::thread::scope(|s| {
        for _ in 0..a.jobs.min(n.max(1)) {
            let tx = tx.clone();
            let (next, selected, profile, geom, inject) = (&next, &selected, &profile, &geom, inject.as_deref());
            s.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                if tx.send((i, run_guarded(profile, geom, selected[i], inject))).is_err() {
                    break;
                }
            });
        }
        drop(tx);
        let mut ready: Vec<Option<Result<(), String>>> = (0..n).map(|_| None).collect();
        let mut at = 0;
        for (i, r) in rx {
            ready[i] = Some(r);
            while at < n && ready[at].is_some() {
                let name = selected[at]["name"].as_str().unwrap_or("");
                let line = match ready[at].take().unwrap() {
                    Ok(()) => format!("pass {name}\n"),
                    Err(why) => {
                        failed += 1;
                        format!("FAIL {name}: {}\n", why.replace('\n', " "))
                    }
                };
                write_err |= stdout.write_all(line.as_bytes()).is_err();
                at += 1;
            }
        }
    });
    write_err |= writeln!(stdout, "{n} vectors: {} pass, {failed} FAIL", n - failed).is_err();
    write_err |= stdout.flush().is_err();
    if write_err {
        let _ = writeln!(stderr, "{PROG}: {CMD}: cannot write standard output");
        return 4;
    }
    i32::from(failed > 0)
}
