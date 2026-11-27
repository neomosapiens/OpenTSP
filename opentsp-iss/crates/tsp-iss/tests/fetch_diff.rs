//! Random fetch programs run on `tsp_sim` and on the ISS with `--trace --lint`: the exit code and
//! the whole stdout (`T srf`, `T sram`, `T diag`, `T lint`, the `E` line and the dumped words) must be
//! byte-identical. The programs fetch text with `IfetchM` (words of the issuing MEM slice, at
//! addresses inside, at the end of and outside the slice and the segment) and with `Ifetch` (a
//! stream pair that two feeder Reads put on the fetching queue's read gap, at the right cycle or a
//! cycle off, on any queue type). The fetched text is random: NOP prefixes, bundles that continue
//! across the two words, SetVL, Config, Reads, a further IfetchM now and then (a fetch from fetched text; a loop ends at `--max-cycles`),
//! undecodable units and padding (0xFFFFFFFF), so the fill rate, the latency, the padding share
//! of the rate, the Parcel stage on fetched text and the conditions that hold or reject an
//! IfetchM are all reached. A dangling bundle unit at the end of an image makes the next fetch
//! arrive after the parcel's earliest cycle (the lint OTS-W-IQ-001); an image filled with NOPs makes
//! a payload overflow the IQ (the diagnostic OTS-E-IQ-002).
//! (`$ISS_DIFF_N` programs per profile, default 120; seeds are fixed, `$ISS_DIFF_SEED` = k shifts them
//! by 1000 k for a longer campaign.)
mod common;
use common::*;
use ots_isa::{encode, Insn};
use ots_profile::Profile;
use std::fmt::Write as _;
use tsp_iss::layout::{read_gap, write_gap, RowId, UnitKind};
use tsp_iss::Geom;

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
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len() as u64) as usize]
    }
}

struct Env {
    geom: Geom,
    vlanes: usize,
    word_units: usize,
    cap: usize,
    d_func: u64,
    d_skew_cu: u64,
}

/// The configuration file of a profile: the shipped ones, or `config/test/<name>.json`.
fn cfg_path(name: &str) -> std::path::PathBuf {
    if matches!(name, "ref" | "mini") {
        config(name)
    } else {
        spec().join("config/test").join(format!("{name}.json"))
    }
}

fn env(name: &str) -> Env {
    let p = if matches!(name, "ref" | "mini") {
        Profile::load(name).unwrap()
    } else {
        Profile::load_file(name, cfg_path(name)).unwrap()
    };
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(cfg_path(name)).unwrap()).unwrap();
    let vlanes = 16 * v["geometry"]["superlanes"].as_u64().unwrap() as usize;
    Env {
        geom: Geom::new(&p),
        vlanes,
        word_units: vlanes / 4,
        cap: v["icu"]["iq_depth"].as_u64().unwrap() as usize / 4,
        d_func: v["timing"]["mem"]["d_func"].as_u64().unwrap(),
        d_skew_cu: v["timing"]["cu"]["d_skew"].as_u64().unwrap(),
    }
}

const PAD: u32 = 0xFFFF_FFFF;

/// Ifetch instructions generated with both feeders in place (most of them at the right cycle).
static TIMED_IFETCH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn units_of(i: &Insn, bundle: bool) -> Vec<u32> {
    let w = encode(i);
    let b = if bundle { 0x8000_0000 } else { 0 };
    if i.entry().width == 64 {
        vec![((w >> 32) as u32) | b, w as u32]
    } else {
        vec![(w as u32) | b]
    }
}

/// One queue's records: `(cycle, instruction)`, lowered to text with NOP prefixes and bundle bits;
/// `tail` is raw text appended to the image.
#[derive(Default)]
struct Plan {
    q: Vec<QueuePlan>,
}

/// One queue: its number, its timed instructions and its raw tail.
type QueuePlan = (u16, Vec<(u64, Insn)>, Vec<u32>);

impl Plan {
    fn slot(&mut self, q: u16) -> &mut QueuePlan {
        if !self.q.iter().any(|(x, _, _)| *x == q) {
            self.q.push((q, vec![], vec![]));
        }
        self.q.iter_mut().find(|(x, _, _)| *x == q).unwrap()
    }
    fn add(&mut self, q: u16, c: u64, i: Insn) {
        self.slot(q).1.push((c, i));
    }
    fn tail(&mut self, q: u16, us: &[u32]) {
        self.slot(q).2.extend_from_slice(us);
    }
    fn lower(&self) -> String {
        let mut out = String::new();
        for (q, recs, tail) in &self.q {
            let mut recs = recs.clone();
            recs.sort_by_key(|(c, _)| *c); // stable: records of one cycle keep their order
            let mut units: Vec<u32> = Vec::new();
            let mut last: i64 = -1;
            let mut k = 0;
            while k < recs.len() {
                let c = recs[k].0;
                let mut j = k;
                while j < recs.len() && recs[j].0 == c {
                    j += 1;
                }
                let gap = c as i64 - last - 1;
                if gap > 0 {
                    units.push(encode(&Insn::NOP { n: gap as u32 }) as u32);
                }
                for (n, (_, i)) in recs[k..j].iter().enumerate() {
                    units.extend(units_of(i, n + 1 < j - k));
                }
                last = c as i64;
                k = j;
            }
            units.extend_from_slice(tail);
            write!(out, "Q {q:x}").unwrap();
            for u in units {
                write!(out, " {u:x}").unwrap();
            }
            out.push('\n');
        }
        out
    }
}

fn s_line(slice: u16, addr: u32, bytes: &[u8]) -> String {
    let mut l = format!("S {slice:x} {addr:x}");
    for b in bytes {
        write!(l, " {b:x}").unwrap();
    }
    l
}

/// An SRAM word (or a stream value) holding the units, little endian, padded with 0xFFFFFFFF.
fn word_of(us: &[u32], word_units: usize) -> Vec<u8> {
    let mut b = Vec::new();
    for k in 0..word_units {
        b.extend_from_slice(&us.get(k).copied().unwrap_or(PAD).to_le_bytes());
    }
    b
}

/// Random text: instructions, bundles (the last one may be cut by the end of the payload),
/// NOP prefixes, undecodable units and padding. At most `max` units.
fn fetched_text(r: &mut Rng, max: usize, mem_q: bool) -> Vec<u32> {
    let mut us: Vec<u32> = Vec::new();
    let items = 1 + r.below(10);
    for _ in 0..items {
        let bundle = r.chance(15);
        let u: Vec<u32> = match r.below(14) {
            0 | 1 => units_of(&Insn::NOP { n: 1 + r.below(6) as u32 }, bundle),
            2 | 3 => units_of(&Insn::SetVL { n: 1 + r.below(20) as u32 }, bundle),
            4 => units_of(&Insn::Config { v: r.below(24) as u32 }, bundle),
            5 => vec![r.pick(&[0x6FFF_FFFFu32, 0x7000_0000, 0x0040_0001, 0x0000_0000])],
            6 => vec![PAD],
            7 if mem_q => units_of(
                &Insn::Read { addr: r.below(8) as u32, dst: r.below(32) as u32, thread: r.below(2) as u32 },
                bundle,
            ),
            8 if mem_q && r.chance(15) => units_of(&Insn::IfetchM { addr: r.below(6) as u32 }, bundle),
            9 if mem_q => units_of(
                &Insn::Write { addr: r.below(8) as u32, src: r.below(32) as u32, thread: r.below(2) as u32 },
                bundle,
            ),
            10 => units_of(&Insn::Sync, bundle),
            11 => units_of(&Insn::Notify, bundle),
            12 => units_of(&Insn::Repeat { n: 1 + r.below(4) as u32, d: r.below(3) as u32 }, bundle),
            _ => units_of(&Insn::SetVL { n: 2 }, bundle),
        };
        us.extend(u);
    }
    us.truncate(max);
    us
}

fn unit_of(s: u16) -> UnitKind {
    if s < 44 {
        UnitKind::MemW
    } else {
        UnitKind::MemE
    }
}

fn program(r: &mut Rng, e: &Env) -> String {
    let wps = e.geom.words_per_slice as u32;
    let mut plan = Plan::default();
    let mut lines: Vec<String> = Vec::new();
    let mut dumps: Vec<(u16, u32)> = Vec::new();
    let slices: Vec<u16> = (0..88u16).filter(|s| e.geom.slice_present(*s as u64)).collect();
    let two = 2 * e.word_units;
    // IfetchM from a MEM queue: the text is preloaded at a, a + 1 (and the other words get other text)
    let n_m = 1 + r.below(3);
    for _ in 0..n_m {
        let s = r.pick(&slices);
        let a = match r.below(10) {
            0 => wps - 1, // a + 1 is outside the slice
            1 => wps - 2, // the last pair
            2 => (wps + r.below(4) as u32).min(8191),
            3 => 6 + r.below(3) as u32, // around a small segment
            _ => 1 + r.below(5) as u32,
        };
        let c = 1 + r.below(12);
        if a < wps {
            lines.push(s_line(s, a, &word_of(&fetched_text(r, e.word_units, true), e.word_units)));
        }
        if a + 1 < wps {
            lines.push(s_line(s, a + 1, &word_of(&fetched_text(r, e.word_units, true), e.word_units)));
        }
        if r.chance(20) {
            plan.add(s, c.saturating_sub(1), Insn::SetSegment { n: r.pick(&[0, 3, 7, 8, 9, 200, wps]) });
        }
        plan.add(s, c, Insn::IfetchM { addr: a });
        if r.chance(25) {
            // a second fetch right behind: its payload queues behind the first
            plan.add(s, c + r.below(4), Insn::IfetchM { addr: (a + 2).min(8191) });
            if a + 2 < wps {
                lines.push(s_line(s, a + 2, &word_of(&fetched_text(r, e.word_units, true), e.word_units)));
            }
            if a + 3 < wps {
                lines.push(s_line(s, a + 3, &word_of(&fetched_text(r, e.word_units, true), e.word_units)));
            }
        }
        if r.chance(15) {
            // a reference of the same parcel (QUE-001), or of the cycle (bank rule)
            let rd = Insn::Read { addr: r.below(8) as u32, dst: r.below(32) as u32, thread: r.below(2) as u32 };
            let wr = Insn::Write { addr: r.below(8) as u32, src: r.below(32) as u32, thread: r.below(2) as u32 };
            plan.add(s, c, if r.chance(50) { rd } else { wr });
        }
        if r.chance(15) {
            plan.add(
                s,
                c + r.below(3),
                Insn::Write { addr: r.pick(&[a, a + 1, wps / 2 + a % 4]), src: r.below(8) as u32, thread: 0 },
            );
        }
        if r.chance(25) {
            // underflow: a dangling bundle unit, completed by the fetched text after its earliest cycle
            plan.tail(s, &units_of(&Insn::SetVL { n: 3 }, true));
        }
        if r.chance(10) {
            // overflow: an IQ filled with NOP units, so the payload does not fit
            let fill = vec![encode(&Insn::NOP { n: 1 }) as u32; e.cap.saturating_sub(2 + r.below(8) as usize)];
            plan.tail(s, &fill);
        }
        dumps.push((s, a.min(wps - 1)));
    }
    // Ifetch from a stream pair: two feeder Reads, then the fetch on a queue of any type
    let n_s = r.below(3);
    for _ in 0..n_s {
        let f = r.pick(&slices);
        let fu = unit_of(f);
        let row = if r.chance(70) { RowId::E } else { RowId::W };
        let stream = 2 * r.below(15) as u32;
        let code = |s: u32| s | if row == RowId::W { 32 } else { 0 };
        let t = 6 + r.below(10);
        // the fetching queue: a MEM queue (the feeders upstream of it), or any other
        let fq: u16 = if r.chance(70) { f } else { r.pick(&[88, 100, 104, 116, 126, 130, 143, 128, 142]) };
        let mem_q = fq < 88;
        let read_at: Option<i64> = if mem_q { read_gap(&e.geom, fu, f as usize, row).map(|g| g as i64) } else { None };
        let (a_slices, wanted) = (if mem_q { slices.clone() } else { vec![] }, mem_q);
        let mut ok = false;
        if wanted {
            for k in 0..2u32 {
                // a feeder upstream of the fetching slice, if there is one
                let cands: Vec<u16> = a_slices
                    .iter()
                    .copied()
                    .filter(|a| match (write_gap(&e.geom, unit_of(*a), *a as usize, row), read_at) {
                        (Some(ga), Some(gb)) => gb - ga as i64 > 0,
                        _ => false,
                    })
                    .collect();
                if cands.is_empty() {
                    break;
                }
                let a = r.pick(&cands);
                let ga = write_gap(&e.geom, unit_of(a), a as usize, row).unwrap() as i64;
                let transit = read_at.unwrap() - ga;
                let delta = if r.chance(80) { 0 } else { r.below(3) as i64 - 1 };
                let t_read = t as i64 + e.d_skew_cu as i64 - e.d_func as i64 - transit + delta;
                if t_read < 0 {
                    break;
                }
                let addr = 1 + k;
                lines.push(s_line(a, addr, &word_of(&fetched_text(r, e.word_units, true), e.word_units)));
                plan.add(a, t_read as u64, Insn::Read { addr, dst: code(stream + k), thread: 0 });
                if k == 1 {
                    ok = true;
                }
            }
        }
        if ok {
            TIMED_IFETCH.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        plan.add(fq, t, Insn::Ifetch { stream: code(stream) });
        if r.chance(10) && fq < 88 {
            // an odd stream: the pair is misaligned
            plan.add(fq, t + 1 + r.below(3), Insn::Ifetch { stream: code(stream + 1) });
        }
        if r.chance(20) {
            plan.tail(fq, &units_of(&Insn::SetVL { n: 4 }, true));
        }
    }
    // a little free-standing traffic on the queues that fetch, so the fetched text meets other parcels
    if r.chance(40) {
        let s = r.pick(&slices);
        plan.add(s, 3 + r.below(10), Insn::SetVL { n: 1 + r.below(20) as u32 });
    }
    let _ = two;
    let mut out = lines.join("\n");
    out.push('\n');
    out.push_str(&plan.lower());
    dumps.sort();
    dumps.dedup();
    for (s, a) in dumps {
        if (a as u64) < e.geom.words_per_slice {
            writeln!(out, "D {s:x} {a:x}").unwrap();
        }
    }
    out
}

fn diff_profile(name: &str, seed: u64, n: usize) {
    let cfg = cfg_path(name).to_str().unwrap().to_string();
    let e = env(name);
    let _ = e.vlanes;
    let progs: Vec<String> = {
        let mut r =
            Rng(seed + std::env::var("ISS_DIFF_SEED").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(0) * 1000);
        (0..n).map(|_| program(&mut r, &e)).collect()
    };
    if let Ok(dir) = std::env::var("ISS_DIFF_SAVE") {
        for (i, p) in progs.iter().enumerate() {
            std::fs::write(format!("{dir}/fetch-{name}-{i}.txt"), p).unwrap();
        }
    }
    let bad = std::sync::Mutex::new(Vec::new());
    let next = std::sync::atomic::AtomicUsize::new(0);
    // how many programs reach each fetch-related report, from the reference's output
    let seen: std::sync::Mutex<std::collections::BTreeMap<&'static str, usize>> = Default::default();
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= progs.len() {
                    break;
                }
                // fetched text may fetch itself again, without end: a bounded run (exit 3 compares too)
                let mut args = vec![cfg.as_str(), "--trace", "--lint", "--max-cycles", "4000"];
                if i.is_multiple_of(4) && iss_extra().is_empty() {
                    args.push("--observe");
                }
                let ours = run_iss(&args, &progs[i]);
                let theirs = run_sim(&args, &progs[i]);
                let out = text(&theirs.stdout);
                {
                    let mut m = seen.lock().unwrap();
                    for (k, pat) in [
                        ("IQ-001", "OTS-W-IQ-001"),
                        ("IQ-002", "OTS-E-IQ-002"),
                        ("OPR-001", "OTS-E-OPR-001"),
                        ("TIM-001", "OTS-E-TIM-001"),
                        ("QUE-001", "OTS-E-QUE-001"),
                        ("MEM-001", "OTS-E-MEM-001"),
                        ("srf", "T srf"),
                    ] {
                        if out.contains(pat) {
                            *m.entry(k).or_default() += 1;
                        }
                    }
                }
                if ours.status.code() != theirs.status.code() || ours.stdout != theirs.stdout {
                    bad.lock().unwrap().push(format!(
                        "program {i} ({name}):\n{}\nours (exit {:?}):\n{}\ntsp_sim (exit {:?}):\n{}",
                        progs[i],
                        ours.status.code(),
                        text(&ours.stdout),
                        theirs.status.code(),
                        out
                    ));
                }
            });
        }
    });
    let bad = bad.into_inner().unwrap();
    let seen = seen.into_inner().unwrap();
    let timed = TIMED_IFETCH.load(std::sync::atomic::Ordering::SeqCst);
    eprintln!(
        "fetch_diff {name}: {n} programs, {} differ; programs with: {seen:?}; {timed} Ifetch with feeders",
        bad.len()
    );
    assert!(bad.is_empty(), "{} of {n} programs differ on {name}; first:\n{}", bad.len(), bad[0]);
    // the generator must reach the conditions it exists for
    for k in ["IQ-001", "IQ-002", "OPR-001", "QUE-001"] {
        assert!(
            seen.get(k).copied().unwrap_or(0) > 0,
            "no program of {name} reports {k}: the generator lost it ({seen:?})"
        );
    }
}

fn count() -> usize {
    std::env::var("ISS_DIFF_N").ok().and_then(|s| s.parse().ok()).unwrap_or(120)
}

#[test]
fn random_fetch_programs_match_tsp_sim_ref() {
    diff_profile("ref", 21, count());
}

#[test]
fn random_fetch_programs_match_tsp_sim_mini() {
    diff_profile("mini", 22, count());
}

// The test profiles: few slices, small SRAM (absent slices are not used by the generator).
#[test]
fn random_fetch_programs_match_tsp_sim_f3_profiles() {
    for (k, name) in ["mid_min", "mini_small", "ref_small", "ref_slow"].iter().enumerate() {
        diff_profile(name, 31 + k as u64, count() / 2);
    }
}
