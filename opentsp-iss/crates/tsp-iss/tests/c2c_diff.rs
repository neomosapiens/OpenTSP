//! Multi-chip runs on `tsp_sim` and on the ISS with `--trace`: the exit code and the whole
//! stdout (`C <chip>` blocks, `T srf`, `T diag`, `E`) must be byte-identical, and so must the
//! messages on stderr.
//!
//! - `random_c2c_programs_*`: random Send, Receive and Transmit programs on K = 1, 2, 3, 5, 8, 16
//!   chips (one node, several nodes), with every latency option (nominal, `--link-latency`,
//!   `--link-latency-seed`, `--link-jitter-seed`) and sometimes a small `--max-cycles`. Receives
//!   are timed near the arrival of a vector, so the "arrived" and "not yet" branches both occur.
//! - `c2c_directed_*_through_both`: `test/c2c/c2c_directed.py` run with a wrapper that runs both
//!   programs on every input and fails on the first difference (data path, Deskew, topologies,
//!   seeds, jitter, harness errors).
//!
//! - `random_c2c_programs_link_tight`: the same on `test/link_tight` (`cycles_per_vector` 3,
//!   `rx_depth` 2), with bursts on few links, so OTS-E-LNK-001 (a busy link) and
//!   OTS-E-LNK-002 (a full receive FIFO, also in a cycle with no tick and across chips) occur.
//!
//! (`$ISS_DIFF_N` programs per profile, default 120; seeds are fixed.)
mod common;
use common::*;
use ots_isa::{encode, Insn};
use std::fmt::Write as _;
use tsp_iss::harness::links::link_peer;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn chance(&mut self, pct: u64) -> bool {
        self.below(100) < pct
    }
}

const TX: [u16; 4] = [126, 127, 140, 141];
const RX: [u16; 4] = [128, 129, 142, 143];

fn units_of(i: &Insn) -> Vec<u32> {
    let w = encode(i);
    if i.entry().width == 64 {
        vec![(w >> 32) as u32, w as u32]
    } else {
        vec![w as u32]
    }
}

/// One generated program and the arguments it runs with.
struct Case {
    args: Vec<String>,
    text: String,
}

/// The timed instructions of one chip: per queue, (cycle, instruction).
type ChipQueues = Vec<(u16, Vec<(u64, Insn)>)>;

/// `poll`: some of the receives are DeskewRecv (, the polling loop), at times from the first cycles to several epochs
/// before or after the vector's arrival, and some chips start later (`H start`). The generator draws nothing more when it is
/// false, so the programs of the other tests stay as they were.
fn case_with(r: &mut Rng, links: usize, tight: bool, poll: bool) -> Case {
    let k = [1usize, 2, 3, 5, 8, 16][r.below(6) as usize];
    // timed items per (chip, queue): (cycle, instruction)
    let mut q: Vec<ChipQueues> = vec![Vec::new(); k];
    let add = |q: &mut Vec<ChipQueues>, c: usize, qn: u16, t: u64, i: Insn| match q[c].iter_mut().find(|e| e.0 == qn) {
        Some(e) => e.1.push((t, i)),
        None => q[c].push((qn, vec![(t, i)])),
    };
    let (lat, mut args): (u64, Vec<String>) = match r.below(5) {
        0 => (216, vec![]),
        1 => {
            let n = 209 + r.below(20);
            (n, vec!["--link-latency".into(), n.to_string()])
        }
        2 => (209 + r.below(20), vec!["--link-latency-seed".into(), (r.below(50) + 1).to_string()]),
        3 => (209 + r.below(20), vec!["--link-jitter-seed".into(), (r.below(50) + 1).to_string()]),
        _ => (216, vec![]),
    };
    if k > 1 || r.chance(30) {
        args.push("--chips".into());
        args.push(k.to_string());
    }
    if r.chance(15) {
        args.push("--max-cycles".into());
        args.push(["100", "230", "260", "500"][r.below(4) as usize].into());
    }
    // tight: bursts on two links overrun the FIFOs and meet busy links
    let (events, used) = if tight { (4 + r.below(12), links.min(2)) } else { (1 + r.below(8), links) };
    for _ in 0..events {
        let c = r.below(k as u64) as usize;
        let l = r.below(used as u64) as usize;
        let (peer, pl) = match link_peer(k, links, c, l) {
            Some((p, pl)) if r.chance(92) => (p, pl),
            _ => (r.below(k as u64) as usize, r.below(links as u64) as usize),
        };
        let t = 1 + r.below(if tight { 12 } else { 20 });
        let tx = TX[r.below(4) as usize];
        let ins = match r.below(4) {
            0 => Insn::Send { chip_id: peer as u32, link: l as u32, stream: r.below(64) as u32 },
            _ => Insn::Transmit { chip_id: peer as u32, link: l as u32 },
        };
        add(&mut q, c, tx, t, ins);
        // the receive on the peer, around the arrival of that vector
        let delta = r.below(7) as i64 - 3;
        let near = (t as i64 + lat as i64 + delta).max(1) as u64;
        let rx = RX[r.below(4) as usize];
        if poll && r.chance(50) {
            let at = [1, 5, near, near + 1, near + 252, near + r.below(600)][r.below(6) as usize];
            add(
                &mut q,
                peer,
                rx,
                at,
                Insn::DeskewRecv { chip_id: c as u32, link: pl as u32, stream: r.below(64) as u32 },
            );
            add(
                &mut q,
                peer,
                rx,
                at + 1 + r.below(700),
                Insn::Receive { chip_id: c as u32, link: pl as u32, stream: r.below(64) as u32 },
            );
        } else {
            add(
                &mut q,
                peer,
                rx,
                near,
                Insn::Receive { chip_id: c as u32, link: pl as u32, stream: r.below(64) as u32 },
            );
        }
        if r.chance(25) {
            let again = near + 1 + r.below(3);
            add(
                &mut q,
                peer,
                rx,
                again,
                Insn::Receive { chip_id: c as u32, link: pl as u32, stream: r.below(64) as u32 },
            );
        }
        if r.chance(15) {
            add(&mut q, c, tx, t + 1 + r.below(30), Insn::Deskew);
        }
        if r.chance(10) {
            add(&mut q, c, tx, t + 1 + r.below(10), Insn::RuntimeDeskew { t: r.below(30) as u32 });
        }
    }
    let mut text = String::new();
    for (c, queues) in q.iter_mut().enumerate() {
        if k > 1 || c > 0 {
            writeln!(text, "C {c:x}").unwrap();
        }
        for (qn, items) in queues.iter_mut() {
            items.sort_by_key(|e| e.0);
            let mut units: Vec<u32> = Vec::new();
            let mut now = 0u64;
            for (t, i) in items.iter() {
                // a NOP of n idles n cycles, the instruction takes one
                if *t > now + 1 {
                    units.extend(units_of(&Insn::NOP { n: (*t - now - 1).min(1000) as u32 }));
                }
                now = (*t).max(now + 1);
                units.extend(units_of(i));
            }
            write!(text, "Q {qn:x}").unwrap();
            for u in units {
                write!(text, " {u:x}").unwrap();
            }
            text.push('\n');
        }
        if poll && r.chance(35) {
            writeln!(text, "H start {:x}", 1 + r.below(1500)).unwrap();
        }
    }
    Case { args, text }
}

/// The `--chips` count of a case's arguments (1 without the option).
fn k_of(args: &[String]) -> usize {
    args.iter().position(|a| a == "--chips").map_or(1, |p| args[p + 1].parse().unwrap())
}

fn diff_profile(name: &str, seed: u64, n: usize) {
    diff_profile_with(name, seed, n, false)
}

fn diff_profile_with(name: &str, seed: u64, n: usize, poll: bool) {
    let cfg = config(name).to_str().unwrap().to_string();
    let links = serde_json::from_slice::<serde_json::Value>(&std::fs::read(config(name)).unwrap()).unwrap()["c2c"]
        ["links"]
        .as_u64()
        .unwrap() as usize;
    let cases: Vec<Case> = {
        let mut r = Rng(seed);
        (0..n).map(|_| case_with(&mut r, links, name == "test/link_tight", poll)).collect()
    };
    let bad = std::sync::Mutex::new(Vec::new());
    let seen: std::sync::Mutex<Vec<String>> = Default::default();
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= cases.len() {
                    break;
                }
                let mut args: Vec<&str> = vec![cfg.as_str(), "--trace"];
                if i.is_multiple_of(4) && iss_extra().is_empty() {
                    args.push("--observe");
                }
                args.extend(cases[i].args.iter().map(String::as_str));
                let ours = run_iss(&args, &cases[i].text);
                let theirs = run_sim(&args, &cases[i].text);
                seen.lock().unwrap().push(text(&theirs.stdout));
                // the reference's chips run a window in parallel: when several stop in it, the
                // order of their messages is a race, and the parent may kill one before it writes
                // so each of its lines must be one of ours, and both say something
                let (theirs_err, ours_err) = (norm_stderr(&text(&theirs.stderr)), text(&ours.stderr));
                let ours_lines: std::collections::BTreeSet<&str> = ours_err.lines().collect();
                let err_ok = if theirs.status.code() == Some(3) && k_of(&cases[i].args) > 1 {
                    theirs_err.lines().all(|l| ours_lines.contains(l)) && theirs_err.is_empty() == ours_err.is_empty()
                } else {
                    theirs_err == ours_err
                };
                if ours.status.code() != theirs.status.code() || ours.stdout != theirs.stdout || !err_ok {
                    bad.lock().unwrap().push(format!(
                        "program {i} ({name}) {:?}:\n{}\nours (exit {:?}):\n{}\n{}\ntsp_sim (exit {:?}):\n{}\n{}",
                        cases[i].args,
                        cases[i].text,
                        ours.status.code(),
                        text(&ours.stdout),
                        text(&ours.stderr),
                        theirs.status.code(),
                        text(&theirs.stdout),
                        text(&theirs.stderr)
                    ));
                }
            });
        }
    });
    let bad = bad.into_inner().unwrap();
    assert!(bad.is_empty(), "{} of {n} programs differ on {name}; first:\n{}", bad.len(), bad[0]);
    if name == "test/link_tight" {
        for code in ["OTS-E-LNK-001", "OTS-E-LNK-002"] {
            assert!(seen.lock().unwrap().iter().any(|o| o.contains(code)), "no program of {name} reports {code}");
        }
    }
}

fn count() -> usize {
    std::env::var("ISS_DIFF_N").ok().and_then(|s| s.parse().ok()).unwrap_or(120)
}

#[test]
fn random_c2c_programs_match_tsp_sim_ref() {
    diff_profile("ref", 11, count());
}

#[test]
fn random_c2c_programs_match_tsp_sim_mini() {
    diff_profile("mini", 12, count());
}

/// Run `c2c_directed.py` with a program that runs `tsp_sim` and the ISS on the same arguments and
/// input, answers with the output of `tsp_sim` and writes a line to `$MISMATCH` on any difference.
fn directed_through_both(name: &str) {
    script_through_both("test/c2c/c2c_directed.py", "all pass", name);
}

/// [`directed_through_both`] for another script of the specification (`pass` is the line it ends with when all checks pass).
fn script_through_both(script_rel: &str, pass: &str, name: &str) {
    let stem = std::path::Path::new(script_rel).file_stem().unwrap().to_string_lossy().into_owned();
    let dir =
        std::env::temp_dir().join(format!("iss-c2c-both-{}-{stem}-{}", std::process::id(), name.replace('/', "_")));
    std::fs::create_dir_all(&dir).unwrap();
    let wrapper = dir.join("both.sh");
    let mismatch = dir.join("mismatch.txt");
    std::fs::write(
        &wrapper,
        format!(
            r#"#!/bin/bash
in=$(mktemp {d}/t.XXXXXX); cat > "$in"
a=$(mktemp {d}/t.XXXXXX); b=$(mktemp {d}/t.XXXXXX)
{sim} "$@" < "$in" > "$a.out" 2> "$a.raw"; ca=$?
{iss} "$@" {extra} < "$in" > "$b.out" 2> "$b.err"; cb=$?
sed -e 's/^tsp_sim:/tsp-iss:/' -e 's#^usage: [^ ]*#tsp-iss: usage: tsp-iss#' "$a.raw" > "$a.err"
if [ $ca -ne $cb ] || ! cmp -s "$a.out" "$b.out" || ! cmp -s "$a.err" "$b.err"; then
  echo "args: $* exit tsp_sim $ca iss $cb" >> {m}
  cp "$in" {d}/last-input.txt
fi
cat "$a.out"; cat "$a.err" >&2
rm -f "$in" "$a" "$b" "$a.out" "$a.raw" "$a.err" "$b.out" "$b.err"
exit $ca
"#,
            d = dir.display(),
            sim = tsp_sim().display(),
            iss = iss(),
            extra = iss_extra().join(" "),
            m = mismatch.display()
        ),
    )
    .unwrap();
    let script = spec().join(script_rel);
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".into());
    let out = std::process::Command::new("bash")
        .args([
            "-c",
            &format!(
                "chmod +x {w} && {py} {s} {w} {c}",
                w = wrapper.display(),
                py = python,
                s = script.display(),
                c = config(name).display()
            ),
        ])
        .output()
        .unwrap();
    let so = text(&out.stdout);
    let diffs = std::fs::read_to_string(&mismatch).unwrap_or_default();
    assert!(diffs.is_empty(), "tsp_sim and the ISS differ on {name}:\n{diffs}");
    if name == "test/link_tight" {
        // two LNK-001 checks of the script assume rx_depth >= 3 and fail on tsp_sim itself
        // the comparison above is the ISS's part
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    assert!(
        out.status.success() && so.contains(pass),
        "{script_rel} on {name} failed:\n{}\n{}",
        so.lines().filter(|l| !l.starts_with("PASS")).collect::<Vec<_>>().join("\n"),
        text(&out.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// : random programs with DeskewRecv polling loops and chips that start late, on 1 to 16 chips and every latency option.
#[test]
fn random_deskewrecv_programs_match_tsp_sim_ref() {
    diff_profile_with("ref", 21, count(), true);
}

#[test]
fn random_deskewrecv_programs_match_tsp_sim_mini() {
    diff_profile_with("mini", 22, count(), true);
}

#[test]
fn random_deskewrecv_programs_link_tight() {
    diff_profile_with("test/link_tight", 23, count(), true);
}

#[test]
fn random_c2c_programs_link_tight() {
    diff_profile("test/link_tight", 13, count());
}

#[test]
fn c2c_directed_ref_through_both() {
    directed_through_both("ref");
}

#[test]
fn c2c_directed_mini_through_both() {
    directed_through_both("mini");
}

#[test]
fn c2c_directed_link_tight_through_both() {
    directed_through_both("test/link_tight");
}

/// : `deskewrecv_directed.py` (the polling loop on one chip and on two chips started 1000 cycles apart) with a program
/// that runs `tsp_sim` and the ISS on every input: the outputs are byte-identical and every check passes.
#[test]
fn deskewrecv_directed_ref_through_both() {
    script_through_both("test/c2c/deskewrecv_directed.py", "deskewrecv: all pass", "ref");
}

#[test]
fn deskewrecv_directed_mini_through_both() {
    script_through_both("test/c2c/deskewrecv_directed.py", "deskewrecv: all pass", "mini");
}

#[test]
fn deskewrecv_directed_link_tight_through_both() {
    script_through_both("test/c2c/deskewrecv_directed.py", "deskewrecv: all pass", "test/link_tight");
}

#[test]
fn deskewrecv_directed_ref_slow_through_both() {
    script_through_both("test/c2c/deskewrecv_directed.py", "deskewrecv: all pass", "test/ref_slow");
}
