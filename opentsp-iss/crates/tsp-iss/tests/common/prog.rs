//! The program builder of the generated differential tests (`mixed_diff`, `order_diff`): a seeded
//! generator, the profile's geometry, the records of a program per queue with the feeder Reads and
//! observer Writes that put values on the stream rows, and the reporting-order pairs of a trace.
use super::{config, spec};
use ots_isa::{encode, EntryId, Insn, Isa};
use ots_profile::Profile;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use tsp_iss::layout::{issuer_of, read_gap, RowId, UnitKind};
use tsp_iss::{Geom, Machine, NoLinks};

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }
    pub fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    pub fn chance(&mut self, pct: u64) -> bool {
        self.below(100) < pct
    }
    pub fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len() as u64) as usize]
    }
    pub fn offset(&mut self, pct: u64) -> i64 {
        if self.chance(pct) {
            self.below(3) as i64 - 1
        } else {
            0
        }
    }
}

pub fn units_of(i: &Insn, more: bool) -> Vec<u32> {
    let w = encode(i);
    let b = if more { 0x8000_0000 } else { 0 };
    if i.entry().width == 64 {
        vec![((w >> 32) as u32) | b, w as u32]
    } else {
        vec![(w as u32) | b]
    }
}

pub fn s_line(slice: u16, addr: u32, bytes: &[u8]) -> String {
    let mut l = format!("S {slice:x} {addr:x}");
    for b in bytes {
        write!(l, " {b:x}").unwrap();
    }
    l
}

pub fn cfg_path(name: &str) -> std::path::PathBuf {
    if matches!(name, "ref" | "mini") {
        config(name)
    } else {
        spec().join("config/test").join(format!("{name}.json"))
    }
}

pub struct Env {
    pub geom: Geom,
    pub isa: Isa,
    pub machine: Machine,
    pub vlanes: usize,
    pub words: u32,
    pub mxm_rows: usize,
    pub superlanes: u64,
    pub links: usize,
    pub link_latency: u64,
    pub rx_depth: u64,
    pub cycles_per_vector: u64,
}

pub fn env(name: &str) -> Env {
    let p = if matches!(name, "ref" | "mini") {
        Profile::load(name).unwrap()
    } else {
        Profile::load_file(name, cfg_path(name)).unwrap()
    };
    let geom = Geom::new(&p);
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(cfg_path(name)).unwrap()).unwrap();
    let superlanes = v["geometry"]["superlanes"].as_u64().unwrap();
    Env {
        vlanes: 16 * superlanes as usize,
        words: geom.words_per_slice as u32,
        mxm_rows: v["mxm"]["rows"].as_u64().unwrap() as usize,
        superlanes,
        links: v["c2c"]["links"].as_u64().unwrap() as usize,
        link_latency: v["c2c"]["link_latency"].as_u64().unwrap(),
        rx_depth: v["c2c"]["rx_depth"].as_u64().unwrap_or(8),
        cycles_per_vector: v["c2c"]["cycles_per_vector"].as_u64().unwrap_or(1),
        isa: Isa::new(&p),
        machine: Machine::new(&p, 0, Box::new(NoLinks)).unwrap(),
        geom,
    }
}

pub fn lat(e: &Env, i: &Insn) -> (i64, i64) {
    let l = e.machine.latency_of_instr(i);
    (l.d_func as i64, l.d_skew as i64)
}

pub fn row_bit(r: RowId) -> u32 {
    if r == RowId::W {
        32
    } else {
        0
    }
}

pub fn other(r: RowId) -> RowId {
    if r == RowId::W {
        RowId::E
    } else {
        RowId::W
    }
}

pub fn unit_of(q: u16) -> Option<UnitKind> {
    (q < 144).then(|| issuer_of(q as usize).unit)
}

/// The queues an entry may issue on (the control channel as 0xffff), in a profile.
pub fn queues_of(e: &Env, id: EntryId) -> Vec<u16> {
    let en = id.entry();
    (0..=144usize)
        .filter(|b| en.queues[b / 64] >> (b % 64) & 1 == 1)
        .map(|b| if b == 144 { 0xffff } else { b as u16 })
        .filter(|q| *q > 87 || e.geom.slice_present(*q as u64))
        .collect()
}

pub fn bytes(r: &mut Rng, e: &Env) -> Vec<u8> {
    const EXT: [u8; 8] = [0, 1, 2, 0x7f, 0x80, 0xff, 0xfe, 0x55];
    (0..e.vlanes).map(|_| if r.chance(50) { r.pick(&EXT) } else { r.below(256) as u8 }).collect()
}

/// The records of a program: per queue, the parcels (a cycle and one or more instructions).
#[derive(Default)]
pub struct Prog {
    pub lines: Vec<String>,
    pub recs: BTreeMap<u16, Vec<(u64, Vec<Insn>)>>,
    pub dumps: Vec<(u16, u32)>,
    pub taken: BTreeSet<(u16, u64)>,
    pub next_addr: BTreeMap<u16, u32>,
    /// No cycle offsets in the feeders and observers (a directed program).
    pub exact: bool,
}

impl Prog {
    pub fn addr(&mut self, e: &Env, s: u16) -> Option<u32> {
        let a = self.next_addr.entry(s).or_insert(1);
        if *a >= e.words {
            return None;
        }
        *a += 1;
        Some(*a - 1)
    }

    /// A parcel at cycle `c` of queue `q`, if the queue is free then.
    pub fn put_all(&mut self, q: u16, c: i64, group: Vec<Insn>) -> bool {
        if c < 0 || !self.taken.insert((q, c as u64)) {
            return false;
        }
        self.recs.entry(q).or_default().push((c as u64, group));
        true
    }

    pub fn put(&mut self, q: u16, c: i64, i: Insn) -> bool {
        self.put_all(q, c, vec![i])
    }

    /// Feed `n` streams from stream `base` (0..31, without the row bit) of `row` so that they reach
    /// the reader (gap `gm`) at `t_exec`, each with random bytes from its own MEM slice.
    #[allow(clippy::too_many_arguments)] // the coordinates of one feeder event
    pub fn feed(&mut self, r: &mut Rng, e: &Env, row: RowId, gm: u8, base: u32, n: u32, t_exec: i64) {
        let mut used: BTreeSet<u16> = BTreeSet::new();
        for k in 0..n {
            if base + k > 31 {
                break;
            }
            let v = bytes(r, e);
            self.feed_data(r, e, row, gm, base + k, &v, t_exec, &mut used);
        }
    }

    /// Feed stream `st` of `row` with `data` (one byte per lane) from a MEM slice not in `used`.
    #[allow(clippy::too_many_arguments)] // the coordinates of one feeder event
    pub fn feed_data(
        &mut self,
        r: &mut Rng,
        e: &Env,
        row: RowId,
        gm: u8,
        st: u32,
        data: &[u8],
        t_exec: i64,
        used: &mut BTreeSet<u16>,
    ) {
        let mem_read = lat(e, &Insn::Read { addr: 1, dst: 0, thread: 0 });
        let cands: Vec<(u16, i64)> = (0..88u16)
            .filter(|s| e.geom.slice_present(*s as u64) && !used.contains(s))
            .filter_map(|s| {
                let u = if s < 44 { UnitKind::MemW } else { UnitKind::MemE };
                let g = read_gap(&e.geom, u, s as usize, row)?;
                let tr = gm as i64 - g as i64;
                let t_read = t_exec - mem_read.0 - tr;
                (tr > 0 && t_read >= 0 && !self.taken.contains(&(s, t_read as u64))).then_some((s, t_read))
            })
            .collect();
        if cands.is_empty() {
            return;
        }
        let (s, t_read) = r.pick(&cands);
        let Some(a) = self.addr(e, s) else { return };
        used.insert(s);
        self.lines.push(s_line(s, a, data));
        let t_read = t_read + if self.exact { 0 } else { r.offset(4) };
        self.put(s, t_read, Insn::Read { addr: a, dst: row_bit(row) + st, thread: r.below(2) as u32 });
    }

    /// Observe `n` streams from stream `dst` (0..31, without the row bit) of `row`, written at gap `gw` at `t_commit`, by Writes of MEM slices.
    #[allow(clippy::too_many_arguments)] // the coordinates of one observer event
    pub fn observe(&mut self, r: &mut Rng, e: &Env, row: RowId, gw: u8, dst: u32, n: u32, t_commit: i64) {
        let mem_write = lat(e, &Insn::Write { addr: 1, src: 0, thread: 0 });
        let mut used: BTreeSet<u16> = BTreeSet::new();
        for k in 0..n {
            if dst + k > 31 {
                break;
            }
            let cands: Vec<(u16, i64)> = (0..88u16)
                .filter(|s| e.geom.slice_present(*s as u64) && !used.contains(s))
                .filter_map(|s| {
                    let u = if s < 44 { UnitKind::MemW } else { UnitKind::MemE };
                    let g = read_gap(&e.geom, u, s as usize, row)?;
                    let tr = g as i64 - gw as i64;
                    let tw = t_commit + tr - mem_write.1;
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
            if self.put(
                o,
                tw + if self.exact { 0 } else { r.offset(3) },
                Insn::Write { addr: a, src: row_bit(row) + dst + k, thread: 0 },
            ) {
                self.dumps.push((o, a));
            }
        }
    }

    pub fn text(&mut self) -> String {
        let mut out = self.lines.join("\n");
        out.push('\n');
        for (q, recs) in self.recs.iter_mut() {
            recs.sort_by_key(|(c, _)| *c);
            let mut units: Vec<u32> = Vec::new();
            let mut last: i64 = -1;
            for (c, group) in recs.iter() {
                let gap = *c as i64 - last - 1;
                if gap > 0 {
                    units.push(encode(&Insn::NOP { n: gap as u32 }) as u32);
                }
                for (k, i) in group.iter().enumerate() {
                    units.extend(units_of(i, k + 1 < group.len()));
                }
                last = *c as i64;
            }
            write!(out, "Q {q:x}").unwrap();
            for u in units {
                write!(out, " {u:x}").unwrap();
            }
            out.push('\n');
        }
        for (s, a) in &self.dumps {
            writeln!(out, "D {s:x} {a:x}").unwrap();
        }
        out
    }
}

/// The reporting step of §8 a diagnostic belongs to, from its code and queue (steps 3 and 5 are
/// told apart only by the order, so both count as 5).
pub fn step_of(code: &str, queue: &str) -> u8 {
    match (code, queue) {
        ("OTS-E-LNK-002", _) => 0,
        ("OTS-E-SRF-002", _) => 2,
        ("OTS-E-RES-003", _) => 6,
        (_, "65535") => 4,
        _ => 5,
    }
}

/// The ordered pairs of (code, code) and of (step, step) of different diagnostics in one cycle,
/// in the order they are printed.
pub fn pairs(out: &str, codes: &mut BTreeSet<(String, String)>, steps: &mut BTreeSet<(u8, u8)>) {
    let diags: Vec<(u64, String, u8)> = out
        .lines()
        .filter(|l| l.starts_with("T diag "))
        .map(|l| {
            let f: Vec<&str> = l.split(' ').collect();
            (f[2].parse().unwrap(), f[5].to_string(), step_of(f[5], f[4]))
        })
        .collect();
    for (i, a) in diags.iter().enumerate() {
        for b in diags[i + 1..].iter().take_while(|b| b.0 == a.0) {
            if a.1 != b.1 {
                codes.insert((a.1.clone(), b.1.clone()));
            }
            if a.2 != b.2 {
                steps.insert((a.2, b.2));
            }
        }
    }
}
