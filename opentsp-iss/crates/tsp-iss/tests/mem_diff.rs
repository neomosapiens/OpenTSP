//! Random MEM programs run on `tsp_sim` and on the ISS with `--trace`: the exit code and the
//! whole stdout (`T srf`, `T sram`, `T diag`, the `E` line and the dumped words) must be
//! byte-identical. The programs use Read, Write, Gather and Scatter on both liquid threads with
//! random AGT configurations (nested loops, negative steps, StreamStep, a fifth level, stray
//! records), Read/Write pairs, SetVL, Config, SetSegment, Repeat and Sync/Notify; a producer
//! slice Reads a preloaded word onto a stream and a consumer slice Writes (or Gathers,
//! Scatters) it at the cycle the stream reaches it, so values move, and the dumped words show it.
//! (`$ISS_DIFF_N` programs per profile, default 120; seeds are fixed.)
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
    d_func: u64,
    d_skew: u64,
}

fn env(name: &str) -> Env {
    let p = match name.strip_prefix("test/") {
        Some(n) => Profile::load_file(n, config(name)).unwrap(),
        None => Profile::load(name).unwrap(),
    };
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(config(name)).unwrap()).unwrap();
    Env {
        geom: Geom::new(&p),
        vlanes: 16 * v["geometry"]["superlanes"].as_u64().unwrap() as usize,
        d_func: v["timing"]["mem"]["d_func"].as_u64().unwrap(),
        d_skew: v["timing"]["mem"]["d_skew"].as_u64().unwrap(),
    }
}

/// One queue's records: `(cycle, instruction)`, lowered to text with NOP prefixes and bundle bits.
#[derive(Default)]
struct Plan {
    q: Vec<(u16, Vec<(u64, Insn)>)>,
}

impl Plan {
    fn add(&mut self, q: u16, c: u64, i: Insn) {
        match self.q.iter_mut().find(|(x, _)| *x == q) {
            Some((_, v)) => v.push((c, i)),
            None => self.q.push((q, vec![(c, i)])),
        }
    }

    fn lower(&self) -> String {
        let mut out = String::new();
        for (q, recs) in &self.q {
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
                    let b = if n + 1 < j - k { 0x8000_0000 } else { 0 };
                    let w = encode(i);
                    if i.entry().width == 64 {
                        units.push(((w >> 32) as u32) | b);
                        units.push(w as u32);
                    } else {
                        units.push((w as u32) | b);
                    }
                }
                last = c as i64;
                k = j;
            }
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

/// The AGT records of a loop: a random subset in the order of the fields, with a chance of a
/// second level, a fifth level or an out-of-order field.
fn agt_records(r: &mut Rng, plan: &mut Plan, q: u16, c0: u64, gather: bool) -> u64 {
    let mut c = c0;
    let levels = if r.chance(8) { 5 } else { 1 + r.below(2) };
    for _ in 0..levels {
        let mut fields = vec![];
        if r.chance(40) {
            fields.push(Insn::Countdown { d: r.below(4) as u32 });
        }
        if r.chance(80) {
            fields.push(Insn::Step { a: if r.chance(30) { 0xFFFF - r.below(6) as u32 } else { r.below(40) as u32 } });
        }
        if r.chance(25) && !gather {
            fields.push(Insn::StreamStep { s: r.below(4) as u32 });
        }
        fields.push(Insn::Iterations { n: r.below(5) as u32 });
        for f in fields {
            c += 1;
            plan.add(q, c, f);
        }
    }
    c
}

fn program(r: &mut Rng, e: &Env, name: &str) -> String {
    let wps = e.geom.words_per_slice as u32;
    let bank = 1u32 << e.geom.bank_bit;
    let sp = e.geom.superlanes as usize;
    let mut plan = Plan::default();
    let mut lines: Vec<String> = Vec::new();
    let mut dumps: Vec<(u16, u32)> = Vec::new();
    let slices: Vec<u16> = (0..88u16).filter(|s| e.geom.slice_present(*s as u64)).collect();
    let addr = |r: &mut Rng| -> u32 {
        match r.below(10) {
            0 => wps - 1 - r.below(2) as u32,         // at the end of the slice
            1 => (wps + r.below(8) as u32).min(8191), // outside the slice (a 13-bit field holds at most 8191)
            2 | 3 => bank + r.below(8) as u32,        // the other bank
            _ => r.below(8) as u32,
        }
    };
    let mut preloaded: Vec<(u16, u32)> = Vec::new();
    let mut pre = |r: &mut Rng, lines: &mut Vec<String>, s: u16, a: u32, map: bool| {
        if a >= wps || preloaded.contains(&(s, a)) {
            return;
        }
        preloaded.push((s, a));
        let mut w = vec![0u8; e.vlanes];
        for g in 0..sp {
            if map {
                let t = if r.chance(15) {
                    wps + r.below(4) as u32
                } else {
                    r.below(6) as u32 + if r.chance(30) { bank } else { 0 }
                };
                w[16 * g] = t as u8;
                w[16 * g + 1] = (t >> 8) as u8;
                for l in 2..16 {
                    w[16 * g + l] = r.below(256) as u8;
                }
            } else {
                for l in 0..16 {
                    w[16 * g + l] = 1 + r.below(255) as u8;
                }
            }
        }
        lines.push(s_line(s, a, &w));
    };
    // the producer/consumer templates
    let n_tpl = 1 + r.below(3);
    for _ in 0..n_tpl {
        let (mut a, mut b) = (r.pick(&slices), r.pick(&slices));
        let row = if r.chance(50) { RowId::E } else { RowId::W };
        let transit_of = |a: u16, b: u16| -> Option<i64> {
            let ga = write_gap(&e.geom, unit_of(a), a as usize, row)? as i64;
            let gb = read_gap(&e.geom, unit_of(b), b as usize, row)? as i64;
            let _ = row;
            Some(gb - ga) // gaps grow along the flow on both rows
        };
        // the consumer must be downstream of the producer (else the read is a direction error)
        let Some(mut transit) = transit_of(a, b) else { continue };
        if transit < 0 && r.chance(90) {
            std::mem::swap(&mut a, &mut b);
            transit = -transit;
        }
        let stream = r.below(30) as u32;
        let code = |s: u32| s | if row == RowId::W { 32 } else { 0 };
        let t_read = 10 + r.below(10);
        let map = r.chance(30);
        let pa = if map { r.below(4) as u32 } else { addr(r) };
        let th = r.below(2) as u32;
        pre(r, &mut lines, a, pa, map);
        plan.add(a, t_read, Insn::Read { addr: pa, dst: code(stream), thread: th });
        // the consumer's dispatch cycle, from the commit cycle of the read; a delta moves it off
        let delta = if r.chance(75) { 0 } else { r.below(3) as i64 - 1 };
        let t_use = t_read as i64 + e.d_func as i64 + transit - e.d_skew as i64 + delta;
        if t_use < 0 {
            continue;
        }
        let t_use = t_use as u64;
        let tb = r.below(2) as u32;
        let target = addr(r);
        match r.below(5) {
            0 | 1 => {
                plan.add(b, t_use, Insn::Write { addr: target, src: code(stream), thread: tb });
                if r.chance(30) {
                    let end = agt_records(r, &mut plan, b, t_use, false);
                    let _ = end;
                }
                dumps.push((b, target));
                dumps.push((b, target + 1));
                dumps.push((b, (target + 8).min(wps - 1)));
            }
            2 => {
                // a Gather whose map is the produced word: the destination stream is read back by a Write
                let ds = r.below(30) as u32;
                plan.add(
                    b,
                    t_use,
                    Insn::Gather { dst: code(ds), map: code(stream), k: r.below(16) as u32, thread: tb },
                );
                for _ in 0..3 {
                    let a = addr(r);
                    pre(r, &mut lines, b, a, false);
                    dumps.push((b, a));
                }
                for a in 0..6 {
                    pre(r, &mut lines, b, a, false);
                    pre(r, &mut lines, b, a + bank, false);
                }
                let c = if r.chance(30) { agt_records(r, &mut plan, b, t_use, true) } else { t_use };
                let _ = c;
            }
            3 => {
                // a Scatter of the produced word through itself as the map (src = map): lanes land by tile
                plan.add(
                    b,
                    t_use,
                    Insn::Scatter { src: code(stream), map: code(stream), k: r.below(16) as u32, thread: tb },
                );
                for a in 0..6 {
                    pre(r, &mut lines, b, a, false);
                    dumps.push((b, a));
                    dumps.push((b, a + bank));
                }
                if r.chance(30) {
                    agt_records(r, &mut plan, b, t_use, true);
                }
            }
            _ => {
                // a Read/Write pair on the consumer: the Write stores the stream, the Read loads a word
                let ra = addr(r);
                plan.add(b, t_use, Insn::Write { addr: target, src: code(stream), thread: tb });
                plan.add(b, t_use, Insn::Read { addr: ra, dst: code(r.below(30) as u32), thread: r.below(2) as u32 });
                dumps.push((b, target));
            }
        }
    }
    // free-standing MEM instructions
    let n_free = r.below(5);
    for _ in 0..n_free {
        let s = r.pick(&slices);
        let q = s;
        let c = 5 + r.below(40);
        let i = match r.below(16) {
            0 => Insn::Read { addr: addr(r), dst: r.below(64) as u32, thread: r.below(2) as u32 },
            1 => Insn::Write { addr: addr(r), src: r.below(64) as u32, thread: r.below(2) as u32 },
            2 => Insn::Gather {
                dst: r.below(64) as u32,
                map: r.below(64) as u32,
                k: r.below(16) as u32,
                thread: r.below(2) as u32,
            },
            3 => Insn::Scatter {
                src: r.below(64) as u32,
                map: r.below(64) as u32,
                k: r.below(16) as u32,
                thread: r.below(2) as u32,
            },
            4 => Insn::SetSegment { n: r.pick(&[0, 1, 4, 16, 100, 255, 256, wps, 8192]) },
            5 => Insn::SetVL { n: 1 + r.below(20) as u32 },
            6 => Insn::Config { v: r.below(24) as u32 },
            7 => Insn::Countdown { d: r.below(4) as u32 },
            8 => Insn::Step { a: r.below(9) as u32 },
            9 => Insn::Iterations { n: r.below(4) as u32 },
            10 => Insn::StreamStep { s: r.below(32) as u32 },
            11 => Insn::PowerConfig { v: r.below(100) as u32 },
            12 => Insn::DeepSleep,
            13 => Insn::Repeat { n: 1 + r.below(4) as u32, d: r.below(3) as u32 },
            14 => Insn::NOP { n: 1 + r.below(5) as u32 },
            _ => {
                if r.chance(50) {
                    Insn::Sync
                } else {
                    Insn::Notify
                }
            }
        };
        if let Insn::Read { addr: a, .. } | Insn::Write { addr: a, .. } = i {
            pre(r, &mut lines, s, a, false);
            dumps.push((s, a));
        }
        plan.add(q, c, i);
        if let Insn::Read { .. } | Insn::Write { .. } | Insn::Gather { .. } | Insn::Scatter { .. } = i {
            if r.chance(60) {
                agt_records(r, &mut plan, q, c, false);
            }
        }
    }
    // SetVL / Config / SetSegment ahead of a template on the consumer's queue, now and then
    if r.chance(40) {
        let (q, c0) = (plan.q[0].0, plan.q[0].1[0].0);
        if c0 > 3 {
            let pick = r.below(3);
            let i = match pick {
                0 => Insn::SetVL { n: 1 + r.below(sp as u64) as u32 },
                1 => Insn::Config { v: r.below(sp as u64 + 2) as u32 },
                _ => Insn::SetSegment { n: r.pick(&[0, 2, 5, 40, wps]) },
            };
            plan.add(q, 1 + r.below(c0 - 2), i);
        }
    }
    let _ = name;
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

fn unit_of(s: u16) -> UnitKind {
    if s < 44 {
        UnitKind::MemW
    } else {
        UnitKind::MemE
    }
}

fn diff_profile(name: &str, seed: u64, n: usize) {
    let cfg = config(name).to_str().unwrap().to_string();
    let e = env(name);
    let progs: Vec<String> = {
        let mut r = Rng(seed);
        (0..n).map(|_| program(&mut r, &e, name)).collect()
    };
    let bad = std::sync::Mutex::new(Vec::new());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let moved = std::sync::atomic::AtomicUsize::new(0);
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
                if std::env::var("ISS_DIFF_SHOW").is_ok() && i < 6 {
                    eprintln!("--- program {i} ({name})\n{}\n{}", progs[i], text(&theirs.stdout));
                }
                if text(&theirs.stdout).contains("T sram") {
                    moved.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
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
    let moved = moved.into_inner();
    eprintln!("mem_diff {name}: {n} programs, {moved} wrote SRAM, {} differ", bad.len());
    assert!(bad.is_empty(), "{} of {n} programs differ on {name}; first:\n{}", bad.len(), bad[0]);
    assert!(moved * 4 >= n, "only {moved} of {n} programs write SRAM: the generator lost its templates");
}

fn count() -> usize {
    std::env::var("ISS_DIFF_N").ok().and_then(|s| s.parse().ok()).unwrap_or(120)
}

#[test]
fn random_mem_programs_match_tsp_sim_ref() {
    diff_profile("ref", 11, count());
}

#[test]
fn random_mem_programs_match_tsp_sim_mini() {
    diff_profile("mini", 12, count());
}

// The test profile with longer MEM, VXM and SXM latencies
#[test]
fn random_mem_programs_match_tsp_sim_ref_slow() {
    diff_profile("test/ref_slow", 13, count() / 2);
}
