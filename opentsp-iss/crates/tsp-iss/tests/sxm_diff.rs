//! Random SXM programs run on `tsp_sim` and on the ISS with `--trace`: the exit code and the
//! whole stdout (`T srf`, `T sram`, `T diag`, the `E` line and the dumped words) must be
//! byte-identical.
//!
//! A program has one to three scenarios, each on one SXM (west or east):
//! - a plain instruction (Shift, Permute and Distribute with the reset identity maps, Rotate 3 or 4,
//!   Transpose, sxselect) at a random cycle;
//! - maps: one or two LoadMap / LoadMapWide (an 8-bit or 10-bit entry map: a permutation, a
//!   permutation with a repeated entry or entries at or beyond `vlanes`, or the in-superlane
//!   entries of a Distribute), from the PRM or a DST queue, a twin load of the same map in the same
//!   cycle (OTS-E-RES-003), and a Permute or Distribute at the ready cycle, a cycle before or after
//!   (OTS-E-RES-001, OTS-E-SXM-001);
//! - a collision: two SXM instructions writing one stream in one cycle (the SEL and TSP OR merge,
//!   and OTS-E-SRF-002 for every other pair).
//!
//! The operands are put on the stream rows by feeder Reads of MEM slices (a group of 16 streams
//! needs 16 slices), preloaded with random bytes (sxselect's selector mostly 0..2). The results are
//! observed by Writes of MEM slices and dumped with `D` lines: the stream values are not in the
//! trace, so the words are what the comparison sees.
//! (`$ISS_DIFF_N` programs per profile, default 150; seeds are fixed, `$ISS_DIFF_SEED` = k shifts
//! them by 1000 k for a longer campaign; `$ISS_DIFF_SAVE` = dir writes the programs.)
mod common;
use common::*;
use ots_isa::{encode, Insn};
use ots_profile::Profile;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use tsp_iss::layout::{issuer_of, read_gap, write_gap, RowId, UnitKind};
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
        words: geom.words_per_slice as u32,
        machine: Machine::new(&p, 0, Box::new(NoLinks)).unwrap(),
        geom,
    }
}

/// Byte values: the extremes and random ones.
fn i8_value(r: &mut Rng) -> u8 {
    const EXT: [u8; 10] = [0, 1, 2, 0x7f, 0x80, 0xff, 0xfe, 3, 0x55, 0x81];
    if r.chance(60) {
        r.pick(&EXT)
    } else {
        r.next() as u8
    }
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

    /// Observe the `n` streams of a result at `dst` (row `row`, written at `gw` at cycle `t_commit`).
    #[allow(clippy::too_many_arguments)] // the coordinates of one observer event
    fn observe(&mut self, r: &mut Rng, e: &Env, row: RowId, gw: u8, dst: u32, t_commit: i64, n: u32) {
        let mem_write = e.machine.latency_of_instr(&Insn::Write { addr: 1, src: 0, thread: 0 });
        let mut used: BTreeSet<u16> = BTreeSet::new();
        for b in 0..n {
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

fn row_bit(r: RowId) -> u32 {
    if r == RowId::W {
        32
    } else {
        0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Shift,
    Permute,
    Distribute,
    LoadMap,
    LoadMapWide,
    Rotate,
    Transpose,
    Select,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Shift => "Shift",
            Kind::Permute => "Permute",
            Kind::Distribute => "Distribute",
            Kind::LoadMap => "LoadMap",
            Kind::LoadMapWide => "LoadMapWide",
            Kind::Rotate => "Rotate",
            Kind::Transpose => "Transpose",
            Kind::Select => "SelectS",
        }
    }
}

/// The queues of SXM `h` that take `kind`: DST 0..3, TSP 4, 5, SHF 6, 7, PRM 8, SEL 9 from 116 + 14 h.
fn queues(kind: Kind, h: usize) -> Vec<u16> {
    let base = 116 + 14 * h as u16;
    match kind {
        Kind::Shift => vec![base + 6, base + 7],
        Kind::Permute => vec![base + 8],
        Kind::Distribute => (0..4).map(|k| base + k).collect(),
        Kind::LoadMap | Kind::LoadMapWide => vec![base + 8, base, base + 1, base + 2, base + 3],
        Kind::Rotate | Kind::Transpose => vec![base + 4, base + 5],
        Kind::Select => vec![base + 9],
    }
}

fn in_row(h: usize) -> RowId {
    if h == 0 {
        RowId::W
    } else {
        RowId::E
    }
}

fn opposite(r: RowId) -> RowId {
    if r == RowId::W {
        RowId::E
    } else {
        RowId::W
    }
}

fn lat(e: &Env, i: &Insn) -> (i64, i64) {
    let l = e.machine.latency_of_instr(i);
    (l.d_func as i64, l.d_skew as i64)
}

fn bytes(r: &mut Rng, e: &Env) -> Vec<u8> {
    (0..e.vlanes).map(|_| i8_value(r)).collect()
}

/// What a map scenario fixes for its loads: the entries of the map.
#[derive(Clone)]
struct MapData {
    entries: Vec<u16>,
}

impl MapData {
    fn lo(&self, e: &Env) -> Vec<u8> {
        (0..e.vlanes).map(|l| self.entries.get(l).copied().unwrap_or(0) as u8).collect()
    }
    /// The high part: bits 9..8 of the entry, with random bits above them (only the low two count).
    fn hi(&self, r: &mut Rng, e: &Env) -> Vec<u8> {
        (0..e.vlanes)
            .map(|l| {
                ((self.entries.get(l).copied().unwrap_or(0) >> 8) as u8)
                    | (((r.below(64) as u8) << 2) * (r.chance(30) as u8))
            })
            .collect()
    }
    fn wide(&self) -> bool {
        self.entries.iter().any(|e| *e > 255)
    }
}

struct Ctx<'a> {
    e: &'a Env,
    seen: &'a mut BTreeMap<&'static str, usize>,
}

/// One SXM instruction of `kind` on SXM `h` dispatched at cycle `t`, its operands fed, its result at
/// `dst` observed (when `watch`). Returns its commit cycle.
#[allow(clippy::too_many_arguments)] // the coordinates of one instruction
fn gen_one(
    r: &mut Rng,
    p: &mut Prog,
    cx: &mut Ctx,
    kind: Kind,
    h: usize,
    t: i64,
    dst: u32,
    map: u32,
    data: Option<&MapData>,
    watch: bool,
    fixed: Option<u32>,
) -> i64 {
    let e = cx.e;
    let q = r.pick(&queues(kind, h));
    let unit = issuer_of(q as usize).unit;
    let rin = in_row(h);
    let rout = if kind == Kind::Distribute { rin } else { opposite(rin) };
    let gm = read_gap(&e.geom, unit, 0, rin).expect("SXM read gap");
    let gw = write_gap(&e.geom, unit, 0, rout).unwrap_or(0);
    // a destination on the wrong row now and then (OTS-E-DIR-001)
    let drow = if r.fault(4) { opposite(rout) } else { rout };
    // `fixed`: the source stream is a result already on the row (not fed)
    let src = fixed.unwrap_or_else(|| r.below(32) as u32);
    let sref = |id: u32| row_bit(rin) + id;
    let mut nout = 1;
    type Feed = Vec<(u32, Vec<Vec<u8>>)>;
    let (ins, nsrc_feed): (Insn, Feed) = match kind {
        Kind::Shift => {
            let amount = if r.chance(30) { r.below(3) } else { r.below(e.vlanes as u64) } as u32;
            (
                Insn::Shift { dst: row_bit(drow) + dst, src: sref(src), dir: r.below(2) as u32, amount },
                if fixed.is_some() { vec![] } else { vec![(src, vec![bytes(r, e)])] },
            )
        }
        Kind::Permute => (
            Insn::Permute { dst: row_bit(drow) + dst, src: sref(src), map },
            if fixed.is_some() { vec![] } else { vec![(src, vec![bytes(r, e)])] },
        ),
        Kind::Distribute => (
            Insn::Distribute { dst: row_bit(drow) + dst, src: sref(src), map },
            if fixed.is_some() { vec![] } else { vec![(src, vec![bytes(r, e)])] },
        ),
        Kind::LoadMap => {
            let d = data.expect("map data");
            (Insn::LoadMap { src: sref(src), map }, vec![(src, vec![d.lo(e)])])
        }
        Kind::LoadMapWide => {
            let d = data.expect("map data");
            let mut s2 = r.below(32) as u32;
            if s2 == src {
                s2 = (s2 + 1) & 31;
            }
            let hi = d.hi(r, e);
            (Insn::LoadMapWide { src: sref(src), map, src2: sref(s2) }, vec![(src, vec![d.lo(e)]), (s2, vec![hi])])
        }
        Kind::Rotate => {
            let n = r.pick(&[3u32, 4]);
            // the destination streams dst..dst+n*n must exist (a few overrun: OTS-E-OPR-002)
            let dmax = 32 - n * n;
            let d = if r.fault(8) { dst & 31 } else { dst % (dmax + 1) };
            nout = n * n;
            (
                Insn::Rotate { dst: row_bit(drow) + d, src: sref(src), n },
                if fixed.is_some() { vec![] } else { vec![(src, vec![bytes(r, e)])] },
            )
        }
        Kind::Transpose => {
            let sg = r.below(2) as u32 * 16;
            let dg = dst & 16;
            nout = 16;
            let planes = (0..16).map(|_| bytes(r, e)).collect();
            (Insn::Transpose { dst: (row_bit(drow) + dg) / 16, src: (row_bit(rin) + sg) / 16 }, vec![(sg, planes)])
        }
        Kind::Select => {
            let mut ids: Vec<u32> = vec![src];
            while ids.len() < 4 {
                let i = r.below(32) as u32;
                if !ids.contains(&i) {
                    ids.push(i);
                }
            }
            let sel: Vec<u8> =
                (0..e.vlanes).map(|_| if r.fault(3) { r.next() as u8 } else { r.below(3) as u8 }).collect();
            (
                Insn::SelectS {
                    dst: row_bit(drow) + dst,
                    a: sref(ids[0]),
                    b: sref(ids[1]),
                    sel: sref(ids[2]),
                    c: sref(ids[3]),
                },
                {
                    let mut v = vec![(ids[1], vec![bytes(r, e)]), (ids[3], vec![bytes(r, e)]), (ids[2], vec![sel])];
                    if fixed.is_none() {
                        v.push((ids[0], vec![bytes(r, e)]));
                    }
                    v
                },
            )
        }
    };
    let (f, s) = lat(e, &ins);
    for (base, planes) in &nsrc_feed {
        p.feed(r, e, rin, gm, row_bit(rin) + base, planes, t + s);
    }
    // the instruction on a free queue of the unit class (twins are the caller's business)
    if !p.put(q, t, ins) {
        return t + f;
    }
    *cx.seen.entry(kind.name()).or_default() += 1;
    if watch && !matches!(kind, Kind::LoadMap | Kind::LoadMapWide) {
        let d0 = match ins {
            Insn::Shift { dst, .. }
            | Insn::Permute { dst, .. }
            | Insn::Distribute { dst, .. }
            | Insn::Rotate { dst, .. }
            | Insn::SelectS { dst, .. } => dst,
            Insn::Transpose { dst, .. } => dst * 16,
            _ => unreachable!(),
        };
        p.observe(r, e, drow, gw, d0, t + f, nout);
    }
    t + f
}

/// A Distribute at `t` whose result is observed through a consumer on the next station: the DST
/// writes the inward row at its own gap, which no MEM slice can read, so an SXM instruction of the
/// same hemisphere (reading one position further) takes it, a cycle after the commit.
fn distribute_chain(r: &mut Rng, p: &mut Prog, cx: &mut Ctx, h: usize, t: i64, dst: u32, map: u32) {
    let tc = gen_one(r, p, cx, Kind::Distribute, h, t, dst, map, None, false, None);
    if !r.chance(90) {
        return;
    }
    let kind = r.pick(&[Kind::Shift, Kind::Select, Kind::Rotate, Kind::Permute, Kind::Shift]);
    let d2 = dst_id(r);
    let m2 = r.below(16) as u32;
    let skew = lat(cx.e, &sample(kind)).1;
    // the station is one position on: a value written at cycle tc at the DST's gap is read there in tc + 1
    let t2 = tc + 1 - skew + if r.fault(10) { r.below(3) as i64 - 1 } else { 0 };
    gen_one(r, p, cx, kind, h, t2, d2, m2, None, true, Some(dst));
}

fn shuffle(r: &mut Rng, v: &mut [u16]) {
    for i in (1..v.len()).rev() {
        let j = r.below(i as u64 + 1) as usize;
        v.swap(i, j);
    }
}

fn map_entries(r: &mut Rng, e: &Env, distribute: bool) -> Vec<u16> {
    let vl = e.vlanes;
    if distribute {
        // mostly in-superlane entries; sometimes a map whose entries all fit LoadMap's 8 bits
        let narrow = r.chance(40);
        return (0..vl)
            .map(|_| {
                if r.chance(8) {
                    r.pick(&[15u16, 16, 17, 31, 32])
                } else if r.chance(80) {
                    r.below(16) as u16
                } else if narrow {
                    16 + r.below(240) as u16
                } else {
                    16 + r.below(1008) as u16
                }
            })
            .collect();
    }
    let mut v: Vec<u16> = (0..vl as u16).collect();
    shuffle(r, &mut v);
    if r.fault(30) {
        // a repeated entry (OTS-E-SXM-001)
        let (a, b) = (r.below(vl as u64) as usize, r.below(vl as u64) as usize);
        v[a] = v[b];
    }
    if r.chance(20) {
        // entries at or beyond vlanes select zero (and make the map need LoadMapWide)
        for _ in 0..1 + r.below(3) {
            let a = r.below(vl as u64) as usize;
            v[a] = if r.chance(50) {
                r.pick(&[vl as u16, vl as u16 + 1, 511, 512, 1023]).min(1023)
            } else {
                (vl as u16 + r.below(1024 - vl as u64) as u16).min(1023)
            };
        }
    }
    v
}

fn dst_id(r: &mut Rng) -> u32 {
    r.below(32) as u32
}

fn scenario(r: &mut Rng, p: &mut Prog, cx: &mut Ctx) {
    let h = r.below(2) as usize;
    let e = cx.e;
    let t0 = 60 + r.below(40) as i64;
    let sl = e.geom.superlanes as i64;
    match r.below(10) {
        0..=3 => {
            // plain instructions, the maps at reset (identity, bijective, ready)
            for _ in 0..1 + r.below(2) {
                let kind = r.pick(&[
                    Kind::Shift,
                    Kind::Permute,
                    Kind::Distribute,
                    Kind::Rotate,
                    Kind::Transpose,
                    Kind::Select,
                    Kind::Select,
                ]);
                let t = t0 + r.below(12) as i64;
                let (d, m) = (dst_id(r), r.below(16) as u32);
                if kind == Kind::Distribute {
                    distribute_chain(r, p, cx, h, t, d, m);
                } else {
                    gen_one(r, p, cx, kind, h, t, d, m, None, true, None);
                }
            }
        }
        4..=7 => {
            // maps
            let distribute = r.chance(40);
            let m = r.below(3) as u32;
            let loads = 1 + r.chance(20) as usize;
            let mut t = t0;
            let mut ready = t0;
            for _ in 0..loads {
                let data = MapData { entries: map_entries(r, e, distribute) };
                let kind = if data.wide() || r.chance(40) { Kind::LoadMapWide } else { Kind::LoadMap };
                gen_one(r, p, cx, kind, h, t, 0, m, Some(&data), false, None);
                if r.fault(15) {
                    // a twin load of the same map in the same cycle: OTS-E-RES-003, neither takes effect
                    let twin = MapData { entries: map_entries(r, e, distribute) };
                    gen_one(
                        r,
                        p,
                        cx,
                        if twin.wide() { Kind::LoadMapWide } else { Kind::LoadMap },
                        h,
                        t,
                        0,
                        m,
                        Some(&twin),
                        false,
                        None,
                    );
                }
                ready = t + sl; // exec cycle + superlanes (d_skew of the load is added by the model)
                t += 1 + r.below(3) as i64;
            }
            let ld = lat(e, &Insn::LoadMap { src: 0, map: 0 });
            ready += ld.1;
            for _ in 0..1 + r.chance(30) as usize {
                if distribute {
                    // usable the cycle after the load's: a Distribute in the load's cycle sees the old map
                    let td = (t0 + r.pick(&[0i64, 1, 1, 2, 4])).max(0);
                    let d = dst_id(r);
                    distribute_chain(r, p, cx, h, td, d, m);
                } else {
                    let tp = (ready + r.pick(&[-2i64, -1, 0, 0, 0, 1, 3, 25])).max(0);
                    let d = dst_id(r);
                    gen_one(r, p, cx, Kind::Permute, h, tp, d, m, None, true, None);
                }
            }
        }
        _ => {
            // a collision of two writers of one stream in one cycle
            let kinds = [Kind::Select, Kind::Rotate, Kind::Transpose, Kind::Shift, Kind::Permute];
            let k1 = r.pick(&kinds);
            let k2 = r.pick(&[Kind::Rotate, Kind::Select, Kind::Transpose, Kind::Shift, Kind::Select]);
            let d = r.pick(&[0u32, 16]) + if r.chance(30) { r.below(3) as u32 } else { 0 };
            let d = if matches!(k1, Kind::Transpose) || matches!(k2, Kind::Transpose) { d & 16 } else { d };
            let commit = t0 + 25;
            let f1 = lat(e, &sample(k1)).0;
            let f2 = lat(e, &sample(k2)).0;
            let (m1, m2) = (r.below(16) as u32, r.below(16) as u32);
            gen_one(r, p, cx, k1, h, commit - f1, d, m1, None, true, None);
            gen_one(r, p, cx, k2, h, commit - f2, d, m2, None, false, None);
        }
    }
}

/// An instruction of the kind, for its latency.
fn sample(k: Kind) -> Insn {
    match k {
        Kind::Shift => Insn::Shift { dst: 0, src: 0, dir: 0, amount: 0 },
        Kind::Permute => Insn::Permute { dst: 0, src: 0, map: 0 },
        Kind::Distribute => Insn::Distribute { dst: 0, src: 0, map: 0 },
        Kind::LoadMap => Insn::LoadMap { src: 0, map: 0 },
        Kind::LoadMapWide => Insn::LoadMapWide { src: 0, map: 0, src2: 0 },
        Kind::Rotate => Insn::Rotate { dst: 0, src: 0, n: 3 },
        Kind::Transpose => Insn::Transpose { dst: 0, src: 0 },
        Kind::Select => Insn::SelectS { dst: 0, a: 0, b: 0, sel: 0, c: 0 },
    }
}

fn program(r: &mut Rng, e: &Env, seen: &mut BTreeMap<&'static str, usize>) -> String {
    let mut p = Prog {
        lines: vec![],
        recs: BTreeMap::new(),
        dumps: vec![],
        taken: BTreeSet::new(),
        next_addr: BTreeMap::new(),
    };
    let mut cx = Ctx { e, seen };
    for _ in 0..1 + r.below(3) {
        scenario(r, &mut p, &mut cx);
    }
    let mut out = p.lines.join("\n");
    out.push('\n');
    for (q, mut recs) in p.recs {
        out.push_str(&lower(q, &mut recs));
    }
    for (s, a) in p.dumps {
        writeln!(out, "D {s:x} {a:x}").unwrap();
    }
    out
}

fn diff_profile(name: &str, seed: u64, n: usize) {
    let cfg = cfg_path(name).to_str().unwrap().to_string();
    let e = env(name);
    let mut seen: BTreeMap<&'static str, usize> = BTreeMap::new();
    let progs: Vec<String> = {
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
        for (i, p) in progs.iter().enumerate() {
            std::fs::write(format!("{dir}/sxm-{name}-{i}.txt"), p).unwrap();
        }
    }
    let bad = std::sync::Mutex::new(Vec::new());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let wrote_nonzero = std::sync::atomic::AtomicUsize::new(0);
    let diags: std::sync::Mutex<BTreeMap<String, usize>> = Default::default();
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= progs.len() {
                    break;
                }
                let args = [cfg.as_str(), "--trace", "--max-cycles", "4000"];
                let ours = run_iss(&args, &progs[i]);
                let theirs = run_sim(&args, &progs[i]);
                let out = text(&theirs.stdout);
                // a program whose results were written: an observed word that is not the sentinel
                let sentinel = "5a".repeat(e.vlanes);
                let zeros = "00".repeat(e.vlanes);
                let wrote = |l: &&str| l.starts_with("W ") && !l.contains(&sentinel);
                if out.lines().filter(wrote).any(|l| !l.ends_with(&zeros)) {
                    wrote_nonzero.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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
    let wrote_nonzero = wrote_nonzero.into_inner();
    let diags = diags.into_inner().unwrap();
    eprintln!("sxm_diff {name}: {n} programs, {} differ, {wrote_nonzero} wrote a nonzero result; diagnostics {diags:?}; instructions drawn {seen:?}", bad.len());
    assert!(
        bad.is_empty(),
        "{} of {n} programs differ on {name}; first:\n{}",
        bad.len(),
        bad[0].chars().take(6000).collect::<String>()
    );
    // the generator must reach what it exists for (the profiles with at least 16 slices can feed 16 streams)
    let slices = (0..88u64).filter(|s| e.geom.slice_present(*s)).count();
    if slices >= 16 && n >= 60 {
        assert!(
            wrote_nonzero * 4 >= n,
            "only {wrote_nonzero} of {n} programs of {name} wrote a nonzero result: the generator drifted"
        );
        // (a map is ready the cycle after its load on a one-superlane profile: no early Permute is possible)
        let early = if e.geom.superlanes >= 2 { vec!["OTS-E-RES-001"] } else { vec![] };
        for k in early.into_iter().chain([
            "OTS-E-RES-003",
            "OTS-E-SXM-001",
            "OTS-E-SRF-002",
            "OTS-E-OPR-001",
            "OTS-E-TIM-001",
        ]) {
            assert!(diags.contains_key(k), "{name}: no {k} ({diags:?})");
        }
        for k in ["Shift", "Permute", "Distribute", "Rotate", "Transpose", "LoadMap", "LoadMapWide", "SelectS"] {
            assert!(seen.contains_key(k), "{name}: no {k} drawn ({seen:?})");
        }
    }
}

fn count() -> usize {
    std::env::var("ISS_DIFF_N").ok().and_then(|s| s.parse().ok()).unwrap_or(150)
}

#[test]
fn random_sxm_programs_match_tsp_sim_ref() {
    diff_profile("ref", 111, count());
}

#[test]
fn random_sxm_programs_match_tsp_sim_mini() {
    diff_profile("mini", 112, count());
}

#[test]
fn random_sxm_programs_match_tsp_sim_f3_profiles() {
    for (k, name) in ["mid_min", "mini_small", "ref_small", "ref_slow"].iter().enumerate() {
        diff_profile(name, 121 + k as u64, count() / 2);
    }
}
