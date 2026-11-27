//! The files and the text of the `--obs-*` flags of `tsp-iss run`: `manifest.json`, `summary.json`,
//! `diagnostics.json`, `utilization.json`, `events.jsonl` and `probes.json` in `--obs-out DIR`, and the text of
//! `--obs-summary` and `--obs-report` on stderr (stdout stays the trace). The recording is `harness::obs`.
use crate::harness::obs::{self, ObsConfig, UNITS};
use crate::harness::stats::REASONS;
use crate::harness::{strerror, ObsChip, RunStats};
use ots_isa::ENTRIES;
use serde_json::{json, Value};
use std::ffi::OsString;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// The depth of `--obs-report`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Report {
    Compact,
    Medium,
    Verbose,
}

impl Report {
    pub fn parse(s: &str) -> Option<Report> {
        match s {
            "compact" => Some(Report::Compact),
            "medium" => Some(Report::Medium),
            "verbose" => Some(Report::Verbose),
            _ => None,
        }
    }
}

/// What the flags ask for beyond the recording itself.
#[derive(Clone, Debug, Default)]
pub struct ObsOut {
    /// `--obs-out DIR`.
    pub dir: Option<OsString>,
    /// `--obs-summary`: `summary.json` and its text on stderr.
    pub summary: bool,
    /// `--obs-utilization` given (the recording also runs for `--obs-report`, which writes no file).
    pub utilization: bool,
    /// `--obs-report LEVEL`: the text on stderr.
    pub report: Option<Report>,
}

impl ObsOut {
    /// Whether a file goes to `--obs-out`: the flags that write files need the directory.
    pub fn check(&self, cfg: &ObsConfig) -> Result<(), String> {
        let files = self.summary || cfg.diagnostics || cfg.events || !cfg.probes.is_empty() || self.utilization;
        if files && self.dir.is_none() {
            return Err("--obs-summary, --obs-diagnostics, --obs-utilization, --obs-events and --obs-probe write files: give --obs-out DIR".into());
        }
        Ok(())
    }
}

fn io_err(cmd: &str, path: &Path, e: &std::io::Error) -> String {
    format!("{}: {cmd}: cannot write {}: {}", crate::harness::PROG, path.display(), strerror(e))
}

fn put(cmd: &str, dir: &Path, name: &str, bytes: &[u8], files: &mut Vec<String>) -> Result<(), String> {
    let p = dir.join(name);
    std::fs::write(&p, bytes).map_err(|e| io_err(cmd, &p, &e))?;
    files.push(name.to_string());
    Ok(())
}

fn doc(chips: Vec<Value>, extra: Value) -> Vec<u8> {
    let mut d = json!({"schema": obs::OBS_SCHEMA, "chips": chips});
    if let (Some(m), Some(e)) = (d.as_object_mut(), extra.as_object()) {
        for (k, v) in e {
            m.insert(k.clone(), v.clone());
        }
    }
    (serde_json::to_string_pretty(&d).expect("json") + "\n").into_bytes()
}

fn cycles_of(st: &RunStats, i: usize) -> u128 {
    st.chips.get(i).and_then(|c| c.last_cycle).map_or(0, |l| u128::from(l) + 1)
}

/// The artifacts of a finished run into `out.dir`; the error is the line for stderr (exit 4).
pub fn write_files(
    cmd: &str,
    out: &ObsOut,
    cfg: &ObsConfig,
    label: &str,
    st: &RunStats,
) -> Result<Vec<String>, String> {
    let mut files = Vec::new();
    let Some(dir) = &out.dir else { return Ok(files) };
    let dir = Path::new(dir);
    std::fs::create_dir_all(dir).map_err(|e| io_err(cmd, dir, &e))?;
    // Keep one publisher per directory and invalidate the previous listing
    // before any of its files can be overwritten.
    let directory = File::open(dir).map_err(|e| io_err(cmd, dir, &e))?;
    directory.try_lock().map_err(|e| io_err(cmd, dir, &e.into()))?;
    let manifest_file = dir.join("manifest.json");
    match std::fs::remove_file(&manifest_file) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(io_err(cmd, &manifest_file, &e)),
    }
    if out.summary {
        let chips: Vec<Value> = st.obs.iter().enumerate().map(|(i, o)| summary_chip(i, o, st)).collect();
        put(
            cmd,
            dir,
            "summary.json",
            &doc(chips, json!({"profile": label, "wall_seconds": st.wall_seconds})),
            &mut files,
        )?;
    }
    if cfg.diagnostics {
        let chips = st.obs.iter().enumerate().map(|(i, o)| obs::diagnostics_json(i, &o.state)).collect();
        put(cmd, dir, "diagnostics.json", &doc(chips, json!({})), &mut files)?;
    }
    if out.utilization {
        let sparse = st.obs.iter().any(|o| o.state.util.as_ref().is_some_and(|u| u.sparse_timeline()));
        let chips = st
            .obs
            .iter()
            .enumerate()
            .map(|(i, o)| {
                if sparse {
                    obs::sparse_utilization_json(i, &o.state, cycles_of(st, i))
                } else {
                    obs::utilization_json(i, &o.state, cycles_of(st, i))
                }
            })
            .collect();
        let format = if sparse { json!({"schema": "opentsp.utilization.sparse.v1"}) } else { json!({}) };
        put(cmd, dir, "utilization.json", &doc(chips, format), &mut files)?;
    }
    if cfg.events {
        let path = dir.join("events.jsonl");
        let write = || -> std::io::Result<()> {
            let mut output = BufWriter::new(File::create(&path)?);
            for (i, o) in st.obs.iter().enumerate() {
                for line in o.state.events.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
                    write!(output, "{{\"chip\":{i},")?;
                    output.write_all(&line[1..])?;
                    output.write_all(b"\n")?;
                }
            }
            output.flush()
        };
        write().map_err(|e| io_err(cmd, &path, &e))?;
        files.push("events.jsonl".into());
    }
    if !cfg.probes.is_empty() {
        let chips = st
            .obs
            .iter()
            .enumerate()
            .map(|(i, o)| {
                let mut v = obs::probes_json(i, &o.state, cfg);
                let last = st.chips.get(i).and_then(|c| c.last_cycle);
                if let Some(ps) = v["probes"].as_array_mut() {
                    for p in ps {
                        if let Some(at) = p["cycle"].as_u64() {
                            p["beyond_run"] = json!(last.is_none_or(|l| at > l));
                        }
                    }
                }
                v
            })
            .collect();
        put(cmd, dir, "probes.json", &doc(chips, json!({})), &mut files)?;
    }
    let truncated: Vec<Value> = st
        .obs
        .iter()
        .enumerate()
        .map(|(i, o)| json!({"chip": i, "events": o.state.events_truncated, "diagnostics": o.state.diag_total > o.state.diags.len() as u64 && cfg.diagnostics}))
        .collect();
    let mut listed = files.clone();
    listed.push("manifest.json".into());
    let manifest = doc(
        vec![],
        json!({
            "profile": label,
            "files": listed,
            "limits": {"max_events": obs::MAX_EVENTS, "max_diagnostics": obs::MAX_DIAGNOSTICS},
            "truncated": truncated,
        }),
    );
    put(cmd, dir, "manifest.json", &manifest, &mut files)?;
    Ok(files)
}

fn summary_chip(i: usize, o: &ObsChip, st: &RunStats) -> Value {
    let s = &o.state;
    let c = st.chips.get(i);
    let mut v = json!({
        "chip": i,
        "cycles": cycles_of(st, i),
        "ticks": c.map(|c| c.ticks),
        "dispatches": s.dispatches,
        "executions": s.executions,
        "diagnostics": s.diag_total,
        "diagnostics_by_code": c.map(|c| c.counts.by_code.clone()),
    });
    if let Some(u) = &s.util {
        v["units"] = json!(UNITS
            .iter()
            .zip(&u.units)
            .map(|(n, b)| (n.to_string(), json!({"executions": b.executions, "active_cycles": b.active_cycles})))
            .collect::<serde_json::Map<_, _>>());
    }
    v
}

/// The text of `--obs-summary` and `--obs-report`, without any wall time, so that it is the same on every run of the
/// same program.
pub fn report_text(level: Report, label: &str, st: &RunStats, files: &[String]) -> String {
    use std::fmt::Write;
    let mut t = String::new();
    for (i, o) in st.obs.iter().enumerate() {
        let s = &o.state;
        let c = st.chips.get(i);
        let cycles = cycles_of(st, i);
        let ticks = c.map_or(0, |c| c.ticks);
        let _ = writeln!(t, "chip {i} ({label}): {cycles} cycles, {ticks} ticks ({} idle), {} dispatches, {} executions, {} diagnostics",
            cycles.saturating_sub(u128::from(ticks)), s.dispatches, s.executions, s.diag_total);
        let Some(u) = &s.util else { continue };
        let busiest = UNITS.iter().zip(&u.units).max_by_key(|(_, b)| b.active_cycles);
        if let Some((n, b)) = busiest {
            if b.active_cycles > 0 {
                let _ = writeln!(
                    t,
                    "  busiest unit {n}: {} active cycles ({:.1}% of the run)",
                    b.active_cycles,
                    100.0 * b.active_cycles as f64 / cycles.max(1) as f64
                );
            }
        }
        if level >= Report::Medium {
            for (n, b) in UNITS.iter().zip(&u.units).filter(|(_, b)| b.executions > 0) {
                let _ = writeln!(
                    t,
                    "  unit {n}: {} executions, {} active cycles, cycles {}..{}",
                    b.executions,
                    b.active_cycles,
                    b.first_cycle.unwrap_or(0),
                    b.last_cycle.unwrap_or(0)
                );
            }
            let mut qs: Vec<(usize, &obs::Busy)> =
                u.queues.iter().enumerate().filter(|(_, b)| b.executions > 0).collect();
            qs.sort_by_key(|(q, b)| (std::cmp::Reverse(b.active_cycles), *q));
            for (q, b) in qs.iter().take(10) {
                let id = if *q >= 144 { 0xffff } else { *q as u16 };
                let _ = writeln!(
                    t,
                    "  queue {id} ({}): {} executions, {} active cycles",
                    ots_isa::queue_name(id).unwrap_or("?"),
                    b.executions,
                    b.active_cycles
                );
            }
            if let Some(c) = c {
                let w = c.counts.wait_totals();
                let waits: Vec<String> =
                    REASONS.iter().zip(w).filter(|(_, n)| *n > 0).map(|((_, name), n)| format!("{name} {n}")).collect();
                if !waits.is_empty() {
                    let _ = writeln!(t, "  wait cycles: {}", waits.join(", "));
                }
            }
        }
        if level >= Report::Verbose {
            let mut ops: Vec<(usize, u64)> = u.ops.iter().copied().enumerate().filter(|(_, n)| *n > 0).collect();
            ops.sort_by_key(|(e, n)| (std::cmp::Reverse(*n), *e));
            for (e, n) in ops.iter().take(15) {
                let _ = writeln!(t, "  opcode {} ({}): {n}", ENTRIES[*e].name, ENTRIES[*e].slice);
            }
            if let Some(c) = c {
                for (code, n) in &c.counts.by_code {
                    let _ = writeln!(t, "  diagnostic {code}: {n}");
                }
            }
            let busy_bins = u.busy_bins();
            let _ = writeln!(
                t,
                "  timeline: {busy_bins} of {} bins of {} cycles executed something",
                u.timeline_bins(),
                s.cfg.bin
            );
        }
    }
    if level >= Report::Verbose && !files.is_empty() {
        let _ = writeln!(t, "artifacts: {}", files.join(", "));
    }
    t
}

/// Print the text to stderr.
pub fn print(stderr: &mut dyn Write, text: &str) {
    let _ = stderr.write_all(text.as_bytes());
}
