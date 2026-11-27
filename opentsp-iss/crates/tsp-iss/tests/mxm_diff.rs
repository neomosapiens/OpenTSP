//! Random MXM programs run on `tsp_sim` and on the ISS with `--trace`: the exit code and the
//! whole stdout (`T srf`, `T sram`, `T diag`, the `E` line and the dumped words) must be
//! byte-identical.
//!
//! A program has one or two flows, each on one MXM plane: weights loaded by LW groups and installed
//! by IW, or installed from streams by IWS; one or two ABCs; ACCs that write the products. A flow
//! may use a type the plane cannot take (f16 on an odd plane, a mismatched ABC type), overfill the
//! buffer, run an instruction a cycle too early or an ABC the cycle after an f16 ABC, or have a
//! "twin": the same instruction at the same cycle from another queue of the plane (OTS-E-RES-003
//! and the withdrawn ACC writes), and sends a stray instruction to the plane's f16 partner.
//! The operands are put on the stream rows by feeder Reads of MEM slices (a group of 16 streams
//! needs 16 slices; a slice serves several cycles), preloaded with values that stress the kernels
//! (i8/u8 extremes, f16 specials, subnormals and moderate magnitudes). The ACC results are
//! observed by Writes of MEM slices and dumped with `D` lines: the stream values are not in the trace
//! so the words are what the comparison sees.
//! (`$ISS_DIFF_N` programs per profile, default 150; seeds are fixed, `$ISS_DIFF_SEED` = k shifts
//! them by 1000 k for a longer campaign.)
mod common;
use common::*;
use ots_isa::{encode, Insn};
use ots_profile::Profile;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use tsp_iss::layout::{read_gap, write_gap, RowId, UnitKind};
use tsp_iss::{Geom, Machine, NoLinks};

struct Rng(u64, bool);

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
    /// A fault knob: fires only in a noisy program (a clean one has only the faults the semantics
    /// make: a result read too early, a type mismatch of the program's own making are knobs too).
    fn fault(&mut self, pct: u64) -> bool {
        self.1 && self.chance(pct)
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len() as u64) as usize]
    }
}

const I8: u32 = 0;
const F16: u32 = 1;
const U8: u32 = 2;

fn units_of(i: &Insn) -> Vec<u32> {
    let w = encode(i);
    if i.entry().width == 64 {
        vec![(w >> 32) as u32, w as u32]
    } else {
        vec![w as u32]
    }
}

/// Text of one queue from `(cycle, instruction)` records, with NOP prefixes.
fn lower(q: u16, recs: &mut [(u64, Insn)]) -> String {
    recs.sort_by_key(|(c, _)| *c);
    let mut units: Vec<u32> = Vec::new();
    let mut last: i64 = -1;
    for (c, i) in recs.iter() {
        assert!(*c as i64 > last, "two records of queue {q} in one cycle");
        let gap = *c as i64 - last - 1;
        if gap > 0 {
            units.push(encode(&Insn::NOP { n: gap as u32 }) as u32);
        }
        units.extend(units_of(i));
        last = *c as i64;
    }
    let mut out = format!("Q {q:x}");
    for u in units {
        write!(out, " {u:x}").unwrap();
    }
    out.push('\n');
    out
}

fn s_line(slice: u16, addr: u32, bytes: &[u8]) -> String {
    let mut l = format!("S {slice:x} {addr:x}");
    for b in bytes {
        write!(l, " {b:x}").unwrap();
    }
    l
}

fn cfg_path(name: &str) -> std::path::PathBuf {
    if matches!(name, "ref" | "mini") {
        config(name)
    } else {
        spec().join("config/test").join(format!("{name}.json"))
    }
}

struct Env {
    geom: Geom,
    machine: Machine,
    vlanes: usize,
    mxm_rows: usize,
    words: u32,
}

fn env(name: &str) -> Env {
    let p = if matches!(name, "ref" | "mini") {
        Profile::load(name).unwrap()
    } else {
        Profile::load_file(name, cfg_path(name)).unwrap()
    };
    let geom = Geom::new(&p);
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(cfg_path(name)).unwrap()).unwrap();
    Env {
        vlanes: 16 * v["geometry"]["superlanes"].as_u64().unwrap() as usize,
        mxm_rows: v["mxm"]["rows"].as_u64().unwrap() as usize,
        words: geom.words_per_slice as u32,
        machine: Machine::new(&p, 0, Box::new(NoLinks)).unwrap(),
        geom,
    }
}

fn unit(pl: usize) -> UnitKind {
    if pl < 2 {
        UnitKind::MxmW
    } else {
        UnitKind::MxmE
    }
}

/// The row the plane reads (its hemisphere's inward row) and the row it writes.
fn in_row(pl: usize) -> RowId {
    if pl < 2 {
        RowId::W
    } else {
        RowId::E
    }
}

fn out_row(pl: usize) -> RowId {
    if pl < 2 {
        RowId::E
    } else {
        RowId::W
    }
}

fn row_bit(r: RowId) -> u32 {
    if r == RowId::W {
        32
    } else {
        0
    }
}

/// The 16 byte values of an i8/u8 row or the 8-bit planes of f16 data.
fn i8_value(r: &mut Rng) -> u8 {
    const EXT: [u8; 10] = [0, 1, 2, 0x7f, 0x80, 0xff, 0xfe, 3, 0x55, 0x81];
    if r.chance(60) {
        r.pick(&EXT)
    } else {
        r.next() as u8
    }
}

/// An f16 without NaN and infinity (exponent field below 31): zeros, ones, subnormals, extremes of
/// the finite range, moderate magnitudes (the dot product of 320 terms then stays finite and rounds),
/// and any finite bit pattern.
fn f16_value(r: &mut Rng) -> u16 {
    const SP: [u16; 12] =
        [0x0000, 0x8000, 0x3c00, 0xbc00, 0x0001, 0x03ff, 0x0400, 0x7bff, 0xfbff, 0x4000, 0x3800, 0x3555];
    match r.below(20) {
        0..=2 => r.pick(&SP),
        3..=15 => {
            let e = 9 + r.below(12) as u16;
            ((r.below(2) as u16) << 15) | (e << 10) | (r.next() as u16 & 0x3ff)
        }
        _ => {
            let e = r.below(31) as u16;
            ((r.below(2) as u16) << 15) | (e << 10) | (r.next() as u16 & 0x3ff)
        }
    }
}

/// A lane that makes the whole row special: NaN, infinities (a few vectors in a program only: one
/// NaN turns every product that uses it into the canonical NaN).
fn f16_special(r: &mut Rng) -> u16 {
    r.pick(&[0x7c00, 0xfc00, 0x7e00, 0x7c01, 0xfe00])
}

struct Prog {
    lines: Vec<String>,
    recs: BTreeMap<u16, Vec<(u64, Insn)>>,
    dumps: Vec<(u16, u32)>,
    /// (queue, cycle) taken by a record
    taken: BTreeSet<(u16, u64)>,
    next_addr: BTreeMap<u16, u32>,
}

impl Prog {
    fn addr(&mut self, e: &Env, s: u16) -> Option<u32> {
        let a = self.next_addr.entry(s).or_insert(1);
        if *a >= e.words {
            return None;
        }
        *a += 1;
        Some(*a - 1)
    }

    /// Add a record if its queue is free at the cycle.
    fn put(&mut self, q: u16, c: i64, i: Insn) -> bool {
        if c < 0 || !self.taken.insert((q, c as u64)) {
            return false;
        }
        self.recs.entry(q).or_default().push((c as u64, i));
        true
    }

    /// Feed `planes` (one byte vector per stream, ids `base ..`) of row `row` so that they reach the
    /// reader at `gm` in cycle `t_exec`: stream k from its own slice (a slice serves one stream per
    /// event), read at the cycle that lands it.
    #[allow(clippy::too_many_arguments)] // the coordinates of one feeder event
    fn feed(&mut self, r: &mut Rng, e: &Env, row: RowId, gm: u8, base: u32, planes: &[Vec<u8>], t_exec: i64) {
        let mem_read = e.machine.latency_of_instr(&Insn::Read { addr: 1, dst: 0, thread: 0 });
        let mut used: BTreeSet<u16> = BTreeSet::new();
        // one feeder of the event a cycle off now and then (OTS-E-TIM-001)
        let off = if r.fault(5) { Some(r.below(planes.len() as u64) as usize) } else { None };
        for (k, bytes) in planes.iter().enumerate() {
            let st = base + k as u32;
            if ((base & 31) as usize + k) > 31 {
                break;
            }
            let cands: Vec<(u16, u8, i64)> = (0..88u16)
                .filter(|s| e.geom.slice_present(*s as u64) && !used.contains(s))
                .filter_map(|s| {
                    let u = if s < 44 { UnitKind::MemW } else { UnitKind::MemE };
                    let g = read_gap(&e.geom, u, s as usize, row)?;
                    let tr = gm as i64 - g as i64;
                    let t_read = t_exec - mem_read.d_func as i64 - tr;
                    (tr > 0 && t_read >= 0 && !self.taken.contains(&(s, t_read as u64))).then_some((s, g, t_read))
                })
                .collect();
            if cands.is_empty() {
                continue;
            }
            let (s, _, t_read) = r.pick(&cands);
            let Some(a) = self.addr(e, s) else { continue };
            used.insert(s);
            self.lines.push(s_line(s, a, bytes));
            let t_read = if off == Some(k) { t_read + 1 } else { t_read };
            self.put(s, t_read, Insn::Read { addr: a, dst: st, thread: 0 });
        }
    }

    /// Observe the four planes of a result at `dst` (row `row`, written at `gw` at cycle `t_commit`).
    fn observe(&mut self, r: &mut Rng, e: &Env, row: RowId, gw: u8, dst: u32, t_commit: i64) {
        let mem_write = e.machine.latency_of_instr(&Insn::Write { addr: 1, src: 0, thread: 0 });
        let mut used: BTreeSet<u16> = BTreeSet::new();
        for b in 0..4u32 {
            if (dst & 31) + b > 31 {
                break;
            }
            let cands: Vec<(u16, i64)> = (0..88u16)
                .filter(|s| e.geom.slice_present(*s as u64) && !used.contains(s))
                .filter_map(|s| {
                    let u = if s < 44 { UnitKind::MemW } else { UnitKind::MemE };
                    let g = read_gap(&e.geom, u, s as usize, row)?;
                    let tr = g as i64 - gw as i64;
                    let tw = t_commit + tr - mem_write.d_skew as i64;
                    (tr > 0 && tw >= 0 && !self.taken.contains(&(s, tw as u64))).then_some((s, tw))
                })
                .collect();
            if cands.is_empty() {
                continue;
            }
            let (o, tw) = r.pick(&cands);
            let Some(a) = self.addr(e, o) else { continue };
            used.insert(o);
            self.lines.push(s_line(o, a, &vec![0x5a; e.vlanes]));
            let tw = if r.fault(5) { tw + r.below(3) as i64 - 1 } else { tw };
            if self.put(o, tw, Insn::Write { addr: a, src: dst + b, thread: 0 }) {
                self.dumps.push((o, a));
            }
        }
    }
}

fn mxm_queue(r: &mut Rng, p: &mut Prog, pl: usize, c: i64, i: Insn) -> Option<u16> {
    let mut qs = [104 + 3 * pl as u16, 105 + 3 * pl as u16, 106 + 3 * pl as u16];
    let k = r.below(3) as usize;
    qs.rotate_left(k);
    qs.into_iter().find(|q| p.put(*q, c, i))
}

/// One instruction record at cycle `c` on a queue of plane `pl`, with its twin sometimes.
fn emit(r: &mut Rng, p: &mut Prog, pl: usize, c: i64, i: Insn) {
    let q = mxm_queue(r, p, pl, c, i);
    if let Some(q) = q {
        if r.fault(10) {
            // the same instruction from another queue of the plane, in the same cycle
            for o in 104 + 3 * pl as u16..107 + 3 * pl as u16 {
                if o != q && p.put(o, c, i) {
                    break;
                }
            }
        }
    }
}

fn lat(e: &Env, i: &Insn) -> (i64, i64) {
    let l = e.machine.latency_of_instr(i);
    (l.d_func as i64, l.d_skew as i64)
}

/// `n` stream values of a weight group: i8/u8 rows, or f16 row pairs (lo, hi planes).
fn weight_group(r: &mut Rng, e: &Env, wt: u32, n: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    if wt == F16 {
        for _ in 0..n / 2 {
            let mut vals: Vec<u16> = (0..e.vlanes).map(|_| f16_value(r)).collect();
            if r.chance(4) {
                let l = r.below(e.vlanes as u64) as usize;
                vals[l] = f16_special(r);
            }
            out.push(vals.iter().map(|v| *v as u8).collect());
            out.push(vals.iter().map(|v| (*v >> 8) as u8).collect());
        }
    } else {
        for _ in 0..n {
            out.push((0..e.vlanes).map(|_| i8_value(r)).collect());
        }
    }
    out
}

fn activation(r: &mut Rng, e: &Env, t: u32) -> Vec<Vec<u8>> {
    weight_group(r, e, t, if t == F16 { 2 } else { 1 })
}

fn group_field(row: RowId, id: u32) -> u32 {
    (row_bit(row) + id) / 16
}

/// A flow on plane `pl`.
fn flow(r: &mut Rng, e: &Env, p: &mut Prog, pl: usize, seen: &mut BTreeMap<&'static str, usize>) -> bool {
    let row = if r.fault(4) { out_row(pl) } else { in_row(pl) };
    let gm = read_gap(&e.geom, unit(pl), 0, in_row(pl)).unwrap();
    let gw = write_gap(&e.geom, unit(pl), 0, out_row(pl)).unwrap();
    // the weight type: f16 on an odd plane is the (rare) violation
    let wt = if pl.is_multiple_of(2) {
        r.pick(&[I8, I8, U8, F16, F16])
    } else if r.fault(15) {
        F16
    } else {
        r.pick(&[I8, U8])
    };
    let stream_mode = r.chance(35);
    let max_groups = (e.mxm_rows / 16).max(1);
    let mut groups = 1 + r.below(max_groups.min(3) as u64) as usize;
    if r.fault(30) {
        groups += 1; // a group past mxm.rows: OTS-E-RES-002
    }
    let mut t: i64 = 60 + r.below(10) as i64;
    let mut last_commit = t;
    for _ in 0..groups {
        let id = r.pick(&[0u32, 16]);
        let w = weight_group(r, e, wt, 16);
        let ty = if r.fault(5) { r.pick(&[I8, U8]) } else { wt }; // a mismatching type now and then
        let (ins, name) = if stream_mode {
            (Insn::IWS { r#type: ty, src: group_field(row, id) }, "IWS")
        } else {
            (Insn::LW { r#type: ty, src: group_field(row, id) }, "LW")
        };
        let (f, s) = lat(e, &ins);
        p.feed(r, e, row, gm, row_bit(row) + id, &w, t + s);
        emit(r, p, pl, t, ins);
        *seen.entry(name).or_default() += 1;
        last_commit = t + f;
        t += 1 + if r.chance(25) { r.below(3) as i64 } else { 0 };
    }
    if !stream_mode {
        let ty = if r.fault(5) { r.pick(&[I8, U8, F16]) } else { wt };
        let ins = Insn::IW { r#type: ty };
        let ti = last_commit + if r.fault(25) { r.below(3) as i64 - 1 } else { 0 };
        emit(r, p, pl, ti, ins);
        *seen.entry("IW").or_default() += 1;
        last_commit = ti + lat(e, &ins).0;
    }
    // ABCs
    let n_abc = 1 + r.chance(30) as usize;
    let mut ta = last_commit + if r.fault(20) { r.below(3) as i64 - 1 } else { 0 };
    let mut abc_commit = ta;
    let mut abc_type = wt;
    let mut abc_times = Vec::new();
    for _ in 0..n_abc {
        let at = if r.fault(12) {
            r.pick(&[I8, U8, F16])
        } else if wt == F16 || !r.chance(25) {
            wt
        } else {
            r.pick(&[I8, U8])
        };
        let width = if at == F16 { 2 } else { 1 };
        let id = (r.below(16) as u32 * 2) & !(width - 1);
        let ids = if r.fault(6) { id + 1 } else { id }; // a misaligned f16 operand
        let ins = Insn::ABC { r#type: at, src: row_bit(row) + ids };
        let (f, s) = lat(e, &ins);
        let act = activation(r, e, at);
        p.feed(r, e, row, gm, row_bit(row) + id, &act, ta + s);
        emit(r, p, pl, ta, ins);
        *seen.entry("ABC").or_default() += 1;
        abc_commit = ta + f;
        abc_type = at;
        abc_times.push(ta);
        // a second ABC the next cycle (the f16 plane is busy) or a few later
        ta += if r.chance(30) { 1 } else { 2 + r.below(4) as i64 };
    }
    // ACCs
    let n_acc = if abc_type == F16 { 2 + r.below(2) as usize } else { 1 + r.below(3) as usize };
    let mut prev_acc: Option<u32> = None;
    let mut tc = abc_commit + if r.fault(15) { r.below(3) as i64 - 1 } else { 0 };
    for _ in 0..n_acc {
        let ty = if r.fault(12) {
            r.pick(&[2u32, 3])
        } else if abc_type == F16 {
            3
        } else {
            2
        };
        let dst0 = (r.below(8) as u32) * 4;
        let dst_row = if r.fault(5) { in_row(pl) } else { out_row(pl) };
        let dst = (dst0 + if r.fault(6) { 1 } else { 0 }) | row_bit(dst_row);
        // a later ACC mostly accumulates into the accumulator the earlier one wrote
        let acc = match prev_acc {
            Some(a) if r.chance(65) => a,
            _ => r.below(2) as u32,
        };
        let mode = if prev_acc.is_some() && r.chance(60) { 1 } else { r.below(2) as u32 };
        prev_acc = Some(acc);
        let ins = Insn::ACC { r#type: ty, dst, mode, acc };
        let (f, _) = lat(e, &ins);
        emit(r, p, pl, tc, ins);
        *seen.entry("ACC").or_default() += 1;
        p.observe(r, e, dst_row, gw, dst, tc + f);
        tc += 1 + if r.chance(30) { r.below(3) as i64 } else { 0 };
    }
    // a stray instruction to the f16 partner (the odd plane of the pair)
    if pl.is_multiple_of(2) && r.fault(40) {
        let c = 60 + r.below(40) as i64;
        let ins = match r.below(3) {
            0 => Insn::LW { r#type: r.pick(&[I8, U8]), src: group_field(in_row(pl + 1), 0) },
            1 => Insn::ABC { r#type: I8, src: row_bit(in_row(pl + 1)) },
            _ => Insn::ACC { r#type: 2, dst: row_bit(out_row(pl + 1)), mode: 0, acc: 0 },
        };
        emit(r, p, pl + 1, c, ins);
    }
    wt == F16 && pl.is_multiple_of(2)
}

fn program(r: &mut Rng, e: &Env, seen: &mut BTreeMap<&'static str, usize>) -> (String, bool) {
    let mut p = Prog {
        lines: vec![],
        recs: BTreeMap::new(),
        dumps: vec![],
        taken: BTreeSet::new(),
        next_addr: BTreeMap::new(),
    };
    let first = r.below(4) as usize;
    let mut f16 = flow(r, e, &mut p, first, seen);
    if r.chance(30) {
        let second = r.below(4) as usize;
        f16 |= flow(r, e, &mut p, second, seen);
    }
    let mut out = p.lines.join("\n");
    out.push('\n');
    for (q, mut recs) in p.recs {
        out.push_str(&lower(q, &mut recs));
    }
    for (s, a) in p.dumps {
        writeln!(out, "D {s:x} {a:x}").unwrap();
    }
    (out, f16)
}

fn diff_profile(name: &str, seed: u64, n: usize) {
    let cfg = cfg_path(name).to_str().unwrap().to_string();
    let e = env(name);
    let mut seen: BTreeMap<&'static str, usize> = BTreeMap::new();
    let progs: Vec<(String, bool)> = {
        let mut r = Rng(
            seed + std::env::var("ISS_DIFF_SEED").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(0) * 1000,
            false,
        );
        (0..n)
            .map(|_| {
                r.1 = r.chance(50); // half of the programs are noisy
                program(&mut r, &e, &mut seen)
            })
            .collect()
    };
    if let Ok(dir) = std::env::var("ISS_DIFF_SAVE") {
        for (i, (p, _)) in progs.iter().enumerate() {
            std::fs::write(format!("{dir}/mxm-{name}-{i}.txt"), p).unwrap();
        }
    }
    let bad = std::sync::Mutex::new(Vec::new());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let clean = std::sync::atomic::AtomicUsize::new(0);
    let f16_results = std::sync::atomic::AtomicUsize::new(0);
    let diags: std::sync::Mutex<BTreeMap<String, usize>> = Default::default();
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= progs.len() {
                    break;
                }
                let args = [cfg.as_str(), "--trace", "--max-cycles", "4000"];
                let ours = run_iss(&args, &progs[i].0);
                let theirs = run_sim(&args, &progs[i].0);
                let out = text(&theirs.stdout);
                // a program whose results were written: an observed word that is not the sentinel
                let sentinel = "5a".repeat(e.vlanes);
                let zeros = "00".repeat(e.vlanes);
                let wrote = |l: &&str| l.starts_with("W ") && !l.contains(&sentinel);
                if out.lines().filter(wrote).any(|l| !l.ends_with(&zeros)) {
                    clean.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if progs[i].1 {
                        f16_results.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    }
                }
                {
                    let mut m = diags.lock().unwrap();
                    for l in out.lines().filter(|l| l.starts_with("T diag")) {
                        *m.entry(l.rsplit(' ').next().unwrap().to_string()).or_default() += 1;
                    }
                }
                if ours.status.code() != theirs.status.code() || ours.stdout != theirs.stdout {
                    bad.lock().unwrap().push(format!(
                        "program {i} ({name}):\n{}\nours (exit {:?}):\n{}\ntsp_sim (exit {:?}):\n{}",
                        progs[i].0,
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
    let clean = clean.into_inner();
    let f16_results = f16_results.into_inner();
    let diags = diags.into_inner().unwrap();
    eprintln!("mxm_diff {name}: {n} programs, {} differ, {clean} wrote a nonzero result ({f16_results} of them f16 flows); diagnostics {diags:?}; instructions drawn {seen:?}", bad.len());
    assert!(
        bad.is_empty(),
        "{} of {n} programs differ on {name}; first:\n{}",
        bad.len(),
        bad[0].chars().take(6000).collect::<String>()
    );
    // the generator must reach what it exists for (the profiles with at least 16 slices can load weights)
    let slices = (0..88u64).filter(|s| e.geom.slice_present(*s)).count();
    if slices >= 16 && n >= 60 {
        assert!(clean * 4 >= n, "only {clean} of {n} programs of {name} wrote a nonzero result: the generator drifted");
        assert!(
            f16_results * 20 >= n,
            "only {f16_results} of {n} programs of {name} wrote an f16 result: the generator drifted"
        );
        for k in ["OTS-E-RES-001", "OTS-E-RES-003"] {
            assert!(diags.contains_key(k), "{name}: no {k} ({diags:?})");
        }
        assert!(diags.contains_key("OTS-E-TIM-001"), "{name}: no TIM-001 ({diags:?})");
    }
    // (a plane of 320 rows would need 21 groups of 16 feeders: only the small matrices overfill)
    if e.mxm_rows <= 48 && slices >= 16 && n >= 60 {
        assert!(diags.contains_key("OTS-E-RES-002"), "{name}: no RES-002 ({diags:?})");
    }
}

fn count() -> usize {
    std::env::var("ISS_DIFF_N").ok().and_then(|s| s.parse().ok()).unwrap_or(150)
}

#[test]
fn random_mxm_programs_match_tsp_sim_ref() {
    diff_profile("ref", 91, count());
}

#[test]
fn random_mxm_programs_match_tsp_sim_mini() {
    diff_profile("mini", 92, count());
}

#[test]
fn random_mxm_programs_match_tsp_sim_f3_profiles() {
    for (k, name) in ["mid_min", "mini_small", "ref_small", "ref_slow"].iter().enumerate() {
        diff_profile(name, 101 + k as u64, count() / 2);
    }
}
