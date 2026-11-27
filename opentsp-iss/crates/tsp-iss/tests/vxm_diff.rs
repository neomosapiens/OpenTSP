//! Random VXM programs run on `tsp_sim` and on the ISS with `--trace`: the exit code and the
//! whole stdout (`T srf`, `T sram`, `T diag`, the `E` line and the dumped words) must be
//! byte-identical.
//!
//! A program has one to three VXM instructions, each on its own VXM queue (88..103), drawn from
//! all 35 catalogue entries with random legal types, modes, group (`.sg4`) and chain bits and
//! stream operands (a misaligned stream now and then). The operands are put on the stream rows by
//! feeder Reads of MEM slices upstream of the VXM, preloaded with values of the element type that
//! stress the kernels (zeros, signs, extremes, NaN, infinities, subnormals, random); a feeder is a
//! cycle off now and then (OTS-E-TIM-001). A later instruction may read the result of an earlier
//! one as a chained operand (the `.chain` bit), dispatched at the cycle that works, or a cycle
//! off. The results are observed by Writes of MEM slices downstream of the VXM and dumped with `D`
//! lines: the stream values are not in the trace, so the words are what the comparison sees.
//! (`$ISS_DIFF_N` programs per profile, default 150; seeds are fixed, `$ISS_DIFF_SEED` = k shifts
//! them by 1000 k for a longer campaign.)
mod common;
use common::*;
use ots_isa::{encode, EntryId, Insn, Isa, MAX_VARS};
use ots_profile::Profile;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use tsp_iss::layout::{read_gap, write_gap, RowId, UnitKind};
use tsp_iss::{Geom, Machine, NoLinks};
use tsp_numerics::lanes::{conv_dst, conv_src};
use tsp_numerics::EType;

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

fn units_of(i: &Insn, bundle: bool) -> Vec<u32> {
    let w = encode(i);
    let b = if bundle { 0x8000_0000 } else { 0 };
    if i.entry().width == 64 {
        vec![((w >> 32) as u32) | b, w as u32]
    } else {
        vec![(w as u32) | b]
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
        units.extend(units_of(i, false));
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
    isa: Isa,
    machine: Machine,
    vlanes: usize,
    gv_r: [u8; 2],
    gv_w: [u8; 2],
}

fn row_of(code: u32) -> RowId {
    if code & 32 == 0 {
        RowId::E
    } else {
        RowId::W
    }
}

fn row_ix(r: RowId) -> usize {
    (r == RowId::W) as usize
}

fn env(name: &str) -> Env {
    let p = if matches!(name, "ref" | "mini") {
        Profile::load(name).unwrap()
    } else {
        Profile::load_file(name, cfg_path(name)).unwrap()
    };
    let geom = Geom::new(&p);
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(cfg_path(name)).unwrap()).unwrap();
    let vlanes = 16 * v["geometry"]["superlanes"].as_u64().unwrap() as usize;
    let g = |f: fn(&Geom, UnitKind, usize, RowId) -> Option<u8>, r| f(&geom, UnitKind::Vxm, 0, r).unwrap();
    Env {
        gv_r: [g(read_gap, RowId::E), g(read_gap, RowId::W)],
        gv_w: [g(write_gap, RowId::E), g(write_gap, RowId::W)],
        isa: Isa::new(&p),
        machine: Machine::new(&p, 0, Box::new(NoLinks)).unwrap(),
        geom,
        vlanes,
    }
}

fn ety(code: u32) -> EType {
    EType::from_code(code as u8).unwrap()
}

/// A value of type `t` (its low `8 * width` bits), biased to the cases that matter.
fn value(r: &mut Rng, t: EType) -> u32 {
    let n = t.nbits();
    let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
    let v = if t.is_float() {
        let sp: &[u32] = if t == EType::F16 {
            &[
                0x0000, 0x8000, 0x3c00, 0xbc00, 0x7c00, 0xfc00, 0x7e00, 0x7c01, 0x0001, 0x03ff, 0x0400, 0x7bff, 0xfbff,
                0x4000, 0x3800, 0x3555, 0xc200, 0x5640,
            ]
        } else {
            &[
                0,
                0x8000_0000,
                0x3f80_0000,
                0xbf80_0000,
                0x7f80_0000,
                0xff80_0000,
                0x7fc0_0000,
                0x7f80_0001,
                1,
                0x007f_ffff,
                0x0080_0000,
                0x7f7f_ffff,
                0xff7f_ffff,
                0x4000_0000,
                0x3f00_0000,
                0x4049_0fdb,
                0xc2c8_0000,
                0x4b00_0000,
            ]
        };
        match r.below(10) {
            0..=3 => r.pick(sp),
            4..=6 => {
                // small magnitudes: a modest exponent
                let e = if t == EType::F16 { 8 + r.below(14) as u32 } else { 118 + r.below(20) as u32 };
                let (eb, mb) = if t == EType::F16 { (5, 10) } else { (8, 23) };
                let _ = eb;
                ((r.below(2) as u32) << (n - 1)) | (e << mb) | (r.next() as u32 & ((1 << mb) - 1))
            }
            _ => r.next() as u32,
        }
    } else if t.is_bool() {
        {
            let x = r.next() as u32;
            r.pick(&[0, 1, 0xff, 0x100, 0x8000_0000, x])
        }
    } else {
        let ext = [0u32, 1, 2, mask, mask >> 1, (mask >> 1) + 1, mask - 1, 3, 0x55 & mask, 0x80 & mask];
        if r.chance(60) {
            r.pick(&ext)
        } else {
            r.next() as u32
        }
    };
    v & mask
}

/// What an instruction reads and writes.
struct Shape {
    /// operand names with their stream plane counts
    srcs: Vec<(&'static str, usize)>,
    /// the element type of the operands, the number of planes of the result and its element type
    st: EType,
    dst_planes: usize,
    dt: EType,
    /// elements per 32-bit word of an operand (sg4: 4 / width)
    ws: usize,
}

fn shape(i: &Insn) -> Shape {
    let e = i.entry();
    let name = e.name;
    let get = |n: &str| i.get(n).unwrap();
    let names: Vec<&'static str> =
        ["lhs", "rhs", "mask"].into_iter().filter(|n| e.vars.iter().any(|v| v.name == *n)).collect();
    if let Some(pos) = [
        "Conv_I8_F32",
        "Conv_F32_I8",
        "Conv_I8_F16",
        "Conv_F16_I8",
        "Conv_I8_I32",
        "Conv_I32_I8",
        "Conv_F16_F32",
        "Conv_F32_F16",
    ]
    .iter()
    .position(|n| *n == name)
    {
        let op = 25 + pos as u32;
        let (s, d) = (conv_src(op), conv_dst(op));
        return Shape { srcs: vec![("lhs", s.width() as usize)], st: s, dst_planes: d.width() as usize, dt: d, ws: 1 };
    }
    let st = ety(get("type"));
    let w = st.width() as usize;
    match name {
        "Convert" => {
            let d = ety(get("dtype"));
            Shape { srcs: vec![("lhs", w)], st, dst_planes: d.width() as usize, dt: d, ws: 1 }
        }
        "Reduce" => Shape { srcs: vec![("lhs", w)], st, dst_planes: w, dt: st, ws: 1 },
        _ => {
            let g = i.get("sg4").unwrap_or(0) == 1;
            let p = if g { 4 } else { w };
            Shape {
                srcs: names.into_iter().map(|n| (n, p)).collect(),
                st,
                dst_planes: p,
                dt: st,
                ws: if g { w } else { 4 },
            }
        }
    }
}

/// The element type of an instruction's result, for chaining.
fn result_type(i: &Insn) -> u32 {
    let s = shape(i);
    s.dt.code() as u32
}

struct Prog {
    lines: Vec<String>,
    recs: BTreeMap<u16, Vec<(u64, Insn)>>,
    dumps: Vec<(u16, u32)>,
    used: Vec<u16>,
    /// free 4-stream groups per row
    groups: [Vec<u32>; 2],
}

impl Prog {
    fn slice(&mut self, r: &mut Rng, e: &Env, row: RowId, upstream: bool) -> Option<(u16, u8)> {
        let ix = row_ix(row);
        let cands: Vec<(u16, u8)> = (0..88u16)
            .filter(|s| e.geom.slice_present(*s as u64) && !self.used.contains(s))
            .filter_map(|s| {
                let u = if s < 44 { UnitKind::MemW } else { UnitKind::MemE };
                let g = read_gap(&e.geom, u, s as usize, row)?;
                let ok =
                    if upstream { (e.gv_r[ix] as i64) - (g as i64) > 0 } else { (g as i64) - (e.gv_w[ix] as i64) > 0 };
                ok.then_some((s, g))
            })
            .collect();
        if cands.is_empty() {
            return None;
        }
        let c = r.pick(&cands);
        self.used.push(c.0);
        Some(c)
    }
}

/// Random fields of one VXM entry: legal for the profile.
fn random_insn(
    r: &mut Rng,
    e: &Env,
    id: EntryId,
    forced_type: Option<u32>,
    dst: u32,
    srcs: &dyn Fn(&str) -> (u32, bool),
) -> Insn {
    let en = id.entry();
    for _ in 0..200 {
        let mut f = [0u32; MAX_VARS];
        for (i, v) in en.vars.iter().enumerate() {
            f[i] = match v.name {
                "type" => match (forced_type, v.legal) {
                    (Some(t), _) if r.chance(70) => t,
                    (_, Some(l)) => r.pick(l),
                    _ => r.below(11) as u32,
                },
                "dst" => dst,
                "lhs" | "rhs" | "mask" => srcs(v.name).0,
                "lhs_chain" | "rhs_chain" | "mask_chain" => srcs(&v.name[..v.name.len() - 6]).1 as u32,
                "sg4" => r.chance(30) as u32,
                "mode" => match en.name {
                    "Cmp" => r.below(6) as u32,
                    "Reduce" => r.below(3) as u32,
                    _ => 0,
                },
                "dtype" => r.below(11) as u32,
                other => panic!("unexpected VXM operand {other}"),
            };
        }
        if e.isa.fields_legal(en, &f) {
            return Insn::from_fields(id, &f);
        }
    }
    panic!("no legal fields for {}", en.name);
}

fn program(r: &mut Rng, e: &Env, seen: &mut BTreeMap<&'static str, usize>) -> String {
    let vxm: Vec<EntryId> = EntryId::ALL.iter().copied().filter(|i| i.entry().slice == "VXM").collect();
    let mut p = Prog {
        lines: vec![],
        recs: BTreeMap::new(),
        dumps: vec![],
        used: vec![],
        groups: [(0..8).map(|g| 4 * g).collect(), (0..8).map(|g| 4 * g).collect()],
    };
    let n = 1 + r.below(3) as usize;
    // (dispatch cycle, instruction) of the earlier instructions, for chaining
    let mut done: Vec<(u64, Insn, u32)> = vec![];
    for k in 0..n {
        let id = r.pick(&vxm);
        // a chained operand: lhs is the destination of an earlier instruction
        let chain_of =
            if !done.is_empty() && r.chance(45) { Some(done[r.below(done.len() as u64) as usize]) } else { None };
        fn take(r: &mut Rng, p: &mut Prog, _hint: Option<RowId>) -> Option<u32> {
            let row = if r.chance(60) { RowId::E } else { RowId::W };
            let ix = row_ix(row);
            if p.groups[ix].is_empty() {
                return None;
            }
            let at = r.below(p.groups[ix].len() as u64) as usize;
            Some(p.groups[ix].remove(at) | if row == RowId::W { 32 } else { 0 })
        }
        let Some(dst) = take(r, &mut p, None) else { break };
        let mut sstream: BTreeMap<&str, (u32, bool)> = BTreeMap::new();
        for nm in ["lhs", "rhs", "mask"] {
            if id.entry().vars.iter().any(|v| v.name == nm) {
                let s = if let Some(chain) = chain_of.as_ref().filter(|_| nm == "lhs") {
                    (chain.2, true)
                } else if let Some(s) = (r.chance(8).then_some(dst)).filter(|_| nm == "lhs") {
                    (s, false) // in place: the result replaces the operand's stream
                } else {
                    match take(r, &mut p, None) {
                        Some(s) => (s, false),
                        None => (0, false),
                    }
                };
                // a misaligned stream now and then (OTS-E-OPR-002, or a shifted group)
                let s = if r.chance(5) && s.0 & 31 < 31 { (s.0 + 1, s.1) } else { s };
                sstream.insert(nm, s);
            }
        }
        let forced = chain_of.as_ref().map(|c| result_type(&c.1));
        let dstv = if r.chance(4) && dst & 31 < 31 { dst + 1 } else { dst };
        let insn = random_insn(r, e, id, forced, dstv, &|n| sstream.get(n).copied().unwrap_or((0, false)));
        *seen.entry(insn.entry().name).or_default() += 1;
        let sh = shape(&insn);
        let lat = e.machine.latency_of_instr(&insn);
        let mem_read = e.machine.latency_of_instr(&Insn::Read { addr: 1, dst: 0, thread: 0 });
        let mem_write = e.machine.latency_of_instr(&Insn::Write { addr: 1, src: 0, thread: 0 });
        let (d_func_v, d_skew_v) = (lat.d_func as i64, lat.d_skew as i64);
        let t: i64 = match &chain_of {
            // the consumer reads at the cycle after the producer's commit
            Some((tp, pi, _)) => {
                let pl = e.machine.latency_of_instr(pi);
                *tp as i64 + pl.d_func as i64 - d_skew_v + 1 + if r.chance(12) { r.below(3) as i64 - 1 } else { 0 }
            }
            None => 28 + r.below(10) as i64 + (k as i64) * r.below(6) as i64,
        };
        let t = t.max(1);
        let q = 88 + k as u16;
        // feeders
        for (nm, planes) in &sh.srcs {
            let (s, chain) = sstream[nm];
            if chain {
                continue;
            }
            let row = row_of(s);
            let elems: Vec<u32> = (0..e.vlanes)
                .map(|_| {
                    let mut x = 0u32;
                    for part in 0..sh.ws.min(4) {
                        let w = sh.st.width();
                        if *planes == 4 && w < 4 {
                            x |= value(r, sh.st) << (8 * w * part as u32);
                        } else {
                            x = value(r, sh.st);
                            break;
                        }
                    }
                    x
                })
                .collect();
            for b in 0..*planes {
                let st = s + b as u32;
                if (s & 31) + b as u32 > 31 {
                    break;
                }
                let Some((a, ga)) = p.slice(r, e, row, true) else { continue };
                let bytes: Vec<u8> = elems.iter().map(|x| (x >> (8 * b)) as u8).collect();
                p.lines.push(s_line(a, 1, &bytes));
                let transit = e.gv_r[row_ix(row)] as i64 - ga as i64;
                let delta = if r.chance(4) { r.below(3) as i64 - 1 } else { 0 };
                let t_read = t + d_skew_v - mem_read.d_func as i64 - transit + delta;
                if t_read < 0 {
                    continue;
                }
                p.recs.entry(a).or_default().push((t_read as u64, Insn::Read { addr: 1, dst: st, thread: 0 }));
            }
        }
        p.recs.entry(q).or_default().push((t as u64, insn));
        // observers
        let drow = row_of(dstv);
        for b in 0..sh.dst_planes {
            if (dstv & 31) + b as u32 > 31 {
                break;
            }
            let Some((o, go)) = p.slice(r, e, drow, false) else { continue };
            p.lines.push(s_line(o, 2, &vec![0x5a; e.vlanes]));
            let transit = go as i64 - e.gv_w[row_ix(drow)] as i64;
            let delta = if r.chance(3) { r.below(3) as i64 - 1 } else { 0 };
            let tw = t + d_func_v + transit - mem_write.d_skew as i64 + delta;
            if tw < 0 {
                continue;
            }
            p.recs.entry(o).or_default().push((tw as u64, Insn::Write { addr: 2, src: dstv + b as u32, thread: 0 }));
            p.dumps.push((o, 2));
        }
        done.push((t as u64, insn, dstv));
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
        let mut r =
            Rng(seed + std::env::var("ISS_DIFF_SEED").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(0) * 1000);
        (0..n).map(|_| program(&mut r, &e, &mut seen)).collect()
    };
    if let Ok(dir) = std::env::var("ISS_DIFF_SAVE") {
        for (i, p) in progs.iter().enumerate() {
            std::fs::write(format!("{dir}/vxm-{name}-{i}.txt"), p).unwrap();
        }
    }
    let bad = std::sync::Mutex::new(Vec::new());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let clean = std::sync::atomic::AtomicUsize::new(0);
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
                let wrote = out.lines().any(|l| l.starts_with("W ") && !l.contains(&sentinel));
                if wrote {
                    clean.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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
    let clean = clean.into_inner();
    let diags = diags.into_inner().unwrap();
    eprintln!("vxm_diff {name}: {n} programs, {} differ, {clean} wrote a result; diagnostics {diags:?}; entries {} of 35 drawn", bad.len(), seen.len());
    assert!(bad.is_empty(), "{} of {n} programs differ on {name}; first:\n{}", bad.len(), bad[0]);
    // the generator must reach what it exists for
    assert!(clean * 3 >= n, "only {clean} of {n} programs of {name} wrote a result: the generator drifted");
    assert!(
        diags.contains_key("OTS-E-TIM-001") && diags.contains_key("OTS-E-OPR-002"),
        "{name}: no TIM-001 or OPR-002 ({diags:?})"
    );
    assert!(seen.len() >= 30 || n < 100, "{name}: only {} entries drawn", seen.len());
}

fn count() -> usize {
    std::env::var("ISS_DIFF_N").ok().and_then(|s| s.parse().ok()).unwrap_or(150)
}

#[test]
fn random_vxm_programs_match_tsp_sim_ref() {
    diff_profile("ref", 71, count());
}

#[test]
fn random_vxm_programs_match_tsp_sim_mini() {
    diff_profile("mini", 72, count());
}

#[test]
fn random_vxm_programs_match_tsp_sim_f3_profiles() {
    for (k, name) in ["mid_min", "mini_small", "ref_small", "ref_slow"].iter().enumerate() {
        diff_profile(name, 81 + k as u64, count() / 2);
    }
}
