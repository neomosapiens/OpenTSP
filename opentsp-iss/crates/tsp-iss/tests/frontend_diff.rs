//! Random text for the front end (control instructions, bundles, NOP prefixes, undecodable units,
//! instructions on queues that do not issue them, Sync and Notify on data queues and the control
//! channel, Repeat, Deskew, the MEM instructions on every queue: references, AGT records,
//! SetSegment; Ifetch and IfetchM; VXM instructions) run on `tsp_sim` and on the ISS with
//! `--trace`: the exit code and the whole stdout (`T srf`, `T sram`, `T diag`, `E`) must be byte-identical.
//! (`$ISS_DIFF_N` programs per profile, default 120; seeds are fixed.)
mod common;
use common::*;
use ots_isa::{encode, Insn};
use std::fmt::Write as _;

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

/// Queues the programs use: MEM west and east, VXM, MXM, SXM, C2C, and the control channel.
const QUEUES: &[u16] = &[0, 3, 44, 87, 88, 100, 104, 116, 126, 130, 143, 0xffff];

fn units_of(i: &Insn, bundle: bool) -> Vec<u32> {
    let w = encode(i);
    let b = if bundle { 0x8000_0000 } else { 0 };
    if i.entry().width == 64 {
        vec![((w >> 32) as u32) | b, w as u32]
    } else {
        vec![(w as u32) | b]
    }
}

fn item(r: &mut Rng, q: u16) -> Vec<u32> {
    let bundle = r.chance(15);
    let insn = match r.below(26) {
        0 | 1 => Insn::NOP { n: 1 + r.below(8) as u32 },
        2 => Insn::Sync,
        3 => Insn::Notify,
        4 | 5 => Insn::Repeat { n: 1 + r.below(5) as u32, d: r.below(4) as u32 },
        6 => Insn::Config { v: r.below(24) as u32 },
        7 | 8 => Insn::SetVL { n: 1 + r.below(20) as u32 },
        9 => Insn::Deskew,
        10 => Insn::RuntimeDeskew { t: r.below(40) as u32 },
        11 => {
            // undecodable: slice tag 6 or 7, a reserved bit, NOP 0 (other unused encodings may decode
            // to slice instructions)
            let pick = [0x6FFF_FFFFu32, 0x7000_0000, 0x0040_0001, 0x0000_0000];
            return vec![pick[r.below(pick.len() as u64) as usize] | if bundle { 0x8000_0000 } else { 0 }];
        }
        // two-unit memory instructions are QUE-002 on every queue but a MEM queue; on a MEM queue
        // they reach the MEM slots: references, AGT records, classification and holds
        12 if q != 0xffff => {
            Insn::Read { addr: r.below(64) as u32, dst: r.below(32) as u32, thread: r.below(2) as u32 }
        }
        13 if q != 0xffff => Insn::Step { a: r.below(100) as u32 },
        14 if q != 0xffff => {
            Insn::Write { addr: r.below(64) as u32, src: r.below(64) as u32, thread: r.below(2) as u32 }
        }
        15 if q != 0xffff => Insn::Gather {
            dst: r.below(64) as u32,
            map: r.below(64) as u32,
            k: r.below(16) as u32,
            thread: r.below(2) as u32,
        },
        16 if q != 0xffff => Insn::Scatter {
            src: r.below(64) as u32,
            map: r.below(64) as u32,
            k: r.below(16) as u32,
            thread: r.below(2) as u32,
        },
        17 if q != 0xffff => Insn::Countdown { d: r.below(4) as u32 },
        18 if q != 0xffff => Insn::Iterations { n: r.below(4) as u32 },
        19 if q != 0xffff => Insn::StreamStep { s: r.below(32) as u32 },
        20 if q != 0xffff => Insn::SetSegment { n: r.below(70) as u32 },
        // the fetches; no feeder, so Ifetch is TIM-001 (or a direction or alignment error) and
        // IfetchM reads the empty SRAM (an all-zero word is not padding: undecodable text); on the
        // control channel and on non-MEM queues IfetchM is QUE-002
        21 => Insn::Ifetch { stream: r.below(64) as u32 },
        22 => Insn::IfetchM { addr: r.below(8) as u32 },
        // VXM instructions on every queue: QUE-002 off the VXM queues; on them, no feeder, so
        // TIM-001 (or a direction or alignment error)
        23 => Insn::Add {
            r#type: r.below(8) as u32,
            dst: r.below(64) as u32,
            lhs: r.below(64) as u32,
            rhs: r.below(64) as u32,
            sg4: r.below(2) as u32,
            lhs_chain: r.below(2) as u32,
            rhs_chain: r.below(2) as u32,
        },
        24 => Insn::Convert {
            r#type: r.below(11) as u32,
            dst: r.below(64) as u32,
            lhs: r.below(64) as u32,
            dtype: r.below(11) as u32,
            lhs_chain: r.below(2) as u32,
        },
        25 => Insn::Reduce {
            r#type: r.below(8) as u32,
            dst: r.below(64) as u32,
            lhs: r.below(64) as u32,
            mode: r.below(3) as u32,
            lhs_chain: r.below(2) as u32,
        },
        _ => Insn::SetVL { n: 2 },
    };
    units_of(&insn, bundle)
}

fn program(r: &mut Rng) -> String {
    let mut out = String::new();
    let nq = 1 + r.below(5) as usize;
    for _ in 0..nq {
        let q = QUEUES[r.below(QUEUES.len() as u64) as usize];
        let n_items = 1 + r.below(12) as usize;
        let mut units: Vec<u32> = Vec::new();
        for _ in 0..n_items {
            units.extend(item(r, q));
        }
        // occasionally cut the text in the middle of an instruction or bundle
        if r.chance(10) {
            units.pop();
        }
        write!(out, "Q {q:x}").unwrap();
        for u in units {
            write!(out, " {u:x}").unwrap();
        }
        out.push('\n');
    }
    out
}

fn diff_profile(name: &str, seed: u64, n: usize) {
    let cfg = config(name).to_str().unwrap().to_string();
    let progs: Vec<String> = {
        let mut r = Rng(seed);
        (0..n).map(|_| program(&mut r)).collect()
    };
    if let Ok(dir) = std::env::var("ISS_DIFF_SAVE") {
        for (i, p) in progs.iter().enumerate() {
            std::fs::write(format!("{dir}/{name}-{i}.txt"), p).unwrap();
        }
    }
    let bad = std::sync::Mutex::new(Vec::new());
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= progs.len() {
                    break;
                }
                let args = [cfg.as_str(), "--trace"];
                let ours = run_iss(&args, &progs[i]);
                let theirs = run_sim(&args, &progs[i]);
                if ours.status.code() != theirs.status.code() || ours.stdout != theirs.stdout {
                    bad.lock().unwrap().push(format!(
                        "program {i} ({name}):\n{}\nours (exit {:?}):\n{}\ntsp_sim (exit {:?}):\n{}",
                        progs[i],
                        ours.status.code(),
                        text(&ours.stdout),
                        theirs.status.code(),
                        text(&theirs.stdout)
                    ));
                }
            });
        }
    });
    let bad = bad.into_inner().unwrap();
    assert!(bad.is_empty(), "{} of {n} programs differ on {name}; first:\n{}", bad.len(), bad[0]);
}

fn count() -> usize {
    std::env::var("ISS_DIFF_N").ok().and_then(|s| s.parse().ok()).unwrap_or(120)
}

#[test]
fn random_control_programs_match_tsp_sim_ref() {
    diff_profile("ref", 1, count());
}

#[test]
fn random_control_programs_match_tsp_sim_mini() {
    diff_profile("mini", 2, count());
}

// The test profile with longer MEM, VXM and SXM latencies
#[test]
fn random_control_programs_match_tsp_sim_ref_slow() {
    diff_profile("test/ref_slow", 3, count() / 2);
}
