//! Programs that mix every unit on many queues, run on `tsp_sim` and on the ISS with `--trace`:
//! the exit code and the whole stdout (`T srf`, `T sram`, `T diag`, `E`, `W`) must be
//! byte-identical, and the ISS must give the same output ticking every cycle.
//!
//! One program is a set of flows placed at random cycles on random queues, sharing the stream rows
//! and the MEM slices (so their results collide, hold and report in the same cycles):
//! - an instruction of any catalogue entry on a queue of its unit (or, now and then, of another
//!   unit), with feeder Reads for its operand streams and a Write for its result;
//! - an MXM weight, activation and accumulate chain, sometimes with a twin from another queue of
//!   the plane in the same cycle (RES-003) or with too many weight groups (RES-002);
//! - SXM map loads (bijective or not, narrow or wide, sometimes twice in one cycle), and a Permute
//!   or Distribute around the cycle the map is ready;
//! - a Send and the Receive timed near its arrival, bursts on one link, Deskew;
//! - MEM loops (the AGT records), Repeat after an instruction, parcels of two instructions,
//!   Sync and Notify on several queues.
//!
//! `pair_coverage_floor` states what the reference run on `ref` and `mini` must reach (the runs print
//! what they reached): at least 30 ordered pairs of different codes and 3 of the reporting steps of
//! `microarchitecture.md` §8 in one cycle, among them control channel before queues (4, 5) and
//! queues before state changes (5, 6), and every code the generator exists to combine. The seeds
//! 0 to 6 of `$ISS_DIFF_SEED` reach 44 to 62 code pairs. The pairs `order_diff` states (steps 0 to 3)
//! are not required of a random run.
//! (`$ISS_DIFF_N` programs per profile, default 150; seeds are fixed, `$ISS_DIFF_SEED` = k shifts
//! them by 1000 k; `$ISS_DIFF_SAVE` = dir writes the programs.)
mod common;
use common::prog::*;
use common::*;
use ots_isa::{EntryId, Insn, Syntax, MAX_VARS};
use std::collections::{BTreeMap, BTreeSet};
use tsp_iss::harness::links::link_peer;
use tsp_iss::layout::{issuer_of, read_gap, write_gap, RowId, UnitKind};

/// A stream operand value: `row` and `idx` (aligned to `align`) as the kind encodes them.
fn stream_code(group: u32, row: RowId, idx: u32) -> u32 {
    (row_bit(row) + idx) / group
}

/// Operand fields of an entry, legal for the profile (the last of 60 tries is used as drawn), and
/// the stream operands (name, row, first stream).
fn fields_for(r: &mut Rng, e: &Env, id: EntryId) -> ([u32; MAX_VARS], Vec<(String, RowId, u32)>) {
    let en = id.entry();
    let draw = |r: &mut Rng| {
        let mut f = [0u32; MAX_VARS];
        let mut named = Vec::new();
        for (i, v) in en.vars.iter().enumerate() {
            let k = v.kind();
            f[i] = if k.syntax == Syntax::Stream {
                let row = if r.chance(50) { RowId::E } else { RowId::W };
                let align = [1u32, 2, 4, 4, 8][r.below(5) as usize].max(k.group);
                // a small pool of streams, so the results of several flows meet
                let pool = if r.chance(65) { r.pick(&[0u32, 4, 8, 16, 20]) } else { r.below(32) as u32 };
                let idx = (pool / align) * align;
                named.push((v.name.to_string(), row, idx));
                stream_code(k.group, row, idx)
            } else if let Some(l) = v.legal {
                r.pick(l)
            } else if k.symbol {
                r.below(4) as u32
            } else {
                match v.name {
                    "chip_id" | "link" | "map" | "acc" => r.below(2) as u32 * (r.chance(60) as u32),
                    _ => r.below((1u64 << v.nbits.min(5)).min(24)) as u32,
                }
            };
        }
        (f, named)
    };
    for _ in 0..59 {
        let (f, named) = draw(r);
        if e.isa.fields_legal(en, &f) {
            return (f, named);
        }
    }
    draw(r)
}

/// A catalogue entry on a random queue with its operands fed and its result observed.
fn generic_flow(r: &mut Rng, e: &Env, p: &mut Prog, seen: &mut BTreeMap<&'static str, usize>) {
    let ids: Vec<EntryId> = EntryId::ALL
        .iter()
        .copied()
        .filter(|i| !matches!(i.entry().name, "NOP" | "Sync" | "Notify" | "Repeat" | "DeepSleep" | "PowerConfig"))
        .collect();
    let id = r.pick(&ids);
    let own = queues_of(e, id);
    let q = if r.chance(6) {
        0xffff
    } else if r.chance(8) || own.is_empty() {
        r.pick(&[0u16, 3, 44, 88, 95, 104, 110, 116, 120, 125, 126, 128, 130, 135, 140, 142, 0xffff])
    } else {
        r.pick(&own)
    };
    let (f, rows) = fields_for(r, e, id);
    let insn = Insn::from_fields(id, &f);
    *seen.entry(id.entry().name).or_default() += 1;
    let (d_func, d_skew) = lat(e, &insn);
    let t = 62 + r.below(14) as i64;
    let unit = unit_of(q);
    let inst = if q < 88 { q as usize } else { 0 };
    for (name, row, idx) in rows.iter() {
        // the streams the instruction writes are observed, the others fed
        let writes = matches!((id.entry().name, name.as_str()), ("Receive", "stream") | (_, "dst"));
        let planes = if id.entry().slice == "VXM" { r.pick(&[1u32, 2, 4]) } else { 1 + r.below(2) as u32 };
        let Some(u) = unit else { continue };
        if writes {
            if r.chance(60) {
                if let Some(gw) = write_gap(&e.geom, u, inst, *row) {
                    p.observe(r, e, *row, gw, *idx, planes, t + d_func);
                }
            }
        } else if let Some(gm) = read_gap(&e.geom, u, inst, *row) {
            p.feed(r, e, *row, gm, *idx, planes, t + d_skew);
        }
    }
    let mut group = vec![insn];
    // a parcel of two on a MEM queue now and then
    if q < 88 && r.chance(12) {
        let (a, s, th) = (r.below(4) as u32, r.below(32) as u32, r.below(2) as u32);
        group.push(match r.below(3) {
            0 => Insn::Read { addr: a, dst: s, thread: th },
            1 => Insn::Write { addr: a, src: s, thread: th },
            _ => Insn::Countdown { d: a },
        });
    }
    if p.put_all(q, t, group) && q != 0xffff && r.chance(20) {
        p.put(q, t + 1, Insn::Repeat { n: 1 + r.below(4) as u32, d: r.below(3) as u32 });
        *seen.entry("Repeat").or_default() += 1;
    }
}

fn group_field(row: RowId, id: u32) -> u32 {
    (row_bit(row) + id) / 16
}

/// An MXM plane's chain: weights, installation, activations, accumulation.
fn mxm_flow(r: &mut Rng, e: &Env, p: &mut Prog, seen: &mut BTreeMap<&'static str, usize>) {
    let pl = r.below(4) as usize;
    let unit = if pl < 2 { UnitKind::MxmW } else { UnitKind::MxmE };
    let (rin, rout) = if pl < 2 { (RowId::W, RowId::E) } else { (RowId::E, RowId::W) };
    let gm = read_gap(&e.geom, unit, 0, rin).unwrap();
    let gw = write_gap(&e.geom, unit, 0, rout).unwrap();
    let qs: Vec<u16> = (0..3).map(|k| 104 + 3 * pl as u16 + k).collect();
    let emit = |r: &mut Rng, p: &mut Prog, c: i64, i: Insn| {
        let k = r.below(3) as usize;
        let mut order = qs.clone();
        order.rotate_left(k);
        if let Some(q) = order.into_iter().find(|q| p.put(*q, c, i)) {
            if r.chance(12) {
                // a twin from another queue of the plane in the same cycle (RES-003)
                if let Some(o) = qs.iter().copied().find(|o| *o != q) {
                    p.put(o, c, i);
                }
            }
        }
    };
    let wt = r.pick(&[0u32, 0, 2, 1, 1]);
    let stream_mode = r.chance(30);
    let mut groups = 1 + r.below(2) as usize;
    if e.mxm_rows <= 48 && r.chance(35) {
        groups = e.mxm_rows / 16 + 1; // past mxm.rows: RES-002
    }
    let mut t: i64 = 64 + r.below(10) as i64;
    let mut commit = t;
    for _ in 0..groups {
        let id = r.pick(&[0u32, 16]);
        let ty = if r.chance(5) { r.pick(&[0u32, 2]) } else { wt };
        let ins = if stream_mode {
            Insn::IWS { r#type: ty, src: group_field(rin, id) }
        } else {
            Insn::LW { r#type: ty, src: group_field(rin, id) }
        };
        let (f, s) = lat(e, &ins);
        p.feed(r, e, rin, gm, id, 16, t + s);
        emit(r, p, t, ins);
        *seen.entry(ins.entry().name).or_default() += 1;
        commit = t + f;
        t += 1 + r.below(2) as i64 * r.chance(30) as i64;
    }
    if !stream_mode {
        let ti = commit + r.offset(25);
        emit(r, p, ti, Insn::IW { r#type: wt });
        commit = ti + lat(e, &Insn::IW { r#type: wt }).0;
    }
    let width = if wt == 1 { 2 } else { 1 };
    let id = (r.below(16) as u32 * 2) & !(width - 1);
    let abc = Insn::ABC { r#type: wt, src: row_bit(rin) + id };
    let (f, s) = lat(e, &abc);
    let ta = commit + r.offset(20);
    p.feed(r, e, rin, gm, id, width, ta + s);
    emit(r, p, ta, abc);
    let mut tc = ta + f + r.offset(15);
    for k in 0..1 + r.below(3) {
        let ty = if wt == 1 { 3 } else { 2 };
        let dst = (r.below(8) as u32) * 4 + row_bit(rout);
        let acc = Insn::ACC { r#type: ty, dst, mode: if k > 0 { 1 } else { 0 }, acc: r.below(2) as u32 };
        emit(r, p, tc, acc);
        if r.chance(50) {
            p.observe(r, e, rout, gw, dst & 31, 4, tc + lat(e, &acc).0);
        }
        tc += 1 + r.below(2) as i64;
    }
}

fn shuffle(r: &mut Rng, v: &mut [u16]) {
    for i in (1..v.len()).rev() {
        v.swap(i, r.below(i as u64 + 1) as usize);
    }
}

/// SXM map loads and the Permute and Distribute that use them.
fn sxm_flow(r: &mut Rng, e: &Env, p: &mut Prog, seen: &mut BTreeMap<&'static str, usize>) {
    let h = r.below(2) as usize;
    let base = 116 + 14 * h as u16;
    let (rin, rout) = if h == 0 { (RowId::W, RowId::E) } else { (RowId::E, RowId::W) };
    let n = e.vlanes;
    let mut entries: Vec<u16> = (0..n as u16).collect();
    shuffle(r, &mut entries);
    if r.chance(35) {
        // not bijective: an entry repeated (SXM-001 at the Permute)
        let (a, b) = (r.below(n as u64) as usize, r.below(n as u64) as usize);
        entries[a] = entries[b];
    }
    if r.chance(30) {
        for _ in 0..1 + r.below(4) {
            let l = r.below(n as u64) as usize;
            entries[l] = n as u16 + r.below(300) as u16; // masked lanes
        }
    }
    let wide = entries.iter().any(|x| *x > 255);
    let map = r.below(4) as u32;
    let t: i64 = 64 + r.below(10) as i64;
    let load_q = |r: &mut Rng| r.pick(&[base + 8, base, base + 1, base + 2, base + 3]);
    let mut load = |r: &mut Rng, p: &mut Prog, t: i64| -> Option<i64> {
        let q = load_q(r);
        let src = r.below(31) as u32;
        let lo: Vec<u8> = (0..n).map(|l| entries[l] as u8).collect();
        let hi: Vec<u8> = (0..n).map(|l| (entries[l] >> 8) as u8).collect();
        let unit = unit_of(q)?;
        let gm = read_gap(&e.geom, unit, 0, rin)?;
        let (ins, wide_src) = if wide || r.chance(30) {
            let s2 = (src + 1 + r.below(30) as u32) % 32;
            (Insn::LoadMapWide { src: row_bit(rin) + src, map, src2: row_bit(rin) + s2 }, Some(s2))
        } else {
            (Insn::LoadMap { src: row_bit(rin) + src, map }, None)
        };
        let (f, s) = lat(e, &ins);
        // the feeder planes carry the entries: stream `src` the low byte, `src2` the high part
        let mut used = BTreeSet::new();
        for (st, v) in [(Some(src), lo), (wide_src, hi)] {
            if let Some(st) = st {
                p.feed_data(r, e, rin, gm, st, &v, t + s, &mut used);
            }
        }
        *seen.entry(ins.entry().name).or_default() += 1;
        p.put(q, t, ins).then_some(t + s + f)
    };
    let ready = load(r, p, t).map(|_| t + e.superlanes as i64 + lat(e, &Insn::LoadMap { src: 0, map: 0 }).1);
    if r.chance(25) {
        load(r, p, t); // a second load of the map in the same cycle: RES-003
    }
    let Some(ready) = ready else { return };
    for _ in 0..1 + r.below(2) {
        let tp = ready + r.below(5) as i64 - 2;
        let src = r.below(32) as u32;
        let (q, ins, row_out) = if r.chance(70) {
            (base + 8, Insn::Permute { dst: row_bit(rout) + r.below(32) as u32, src: row_bit(rin) + src, map }, rout)
        } else {
            (
                base + r.below(4) as u16,
                Insn::Distribute { dst: row_bit(rin) + r.below(32) as u32, src: row_bit(rin) + src, map },
                rin,
            )
        };
        let unit = unit_of(q).unwrap();
        *seen.entry(ins.entry().name).or_default() += 1;
        if let Some(gm) = read_gap(&e.geom, unit, 0, rin) {
            p.feed(r, e, rin, gm, src, 1, tp + lat(e, &ins).1);
        }
        if r.chance(50) {
            if let (Some(gw), Insn::Permute { dst, .. } | Insn::Distribute { dst, .. }) =
                (write_gap(&e.geom, unit, 0, row_out), ins)
            {
                p.observe(r, e, row_out, gw, dst & 31, 1, tp + lat(e, &ins).0);
            }
        }
        p.put(q, tp, ins);
    }
}

/// Sends, the Receives near their arrival, bursts on one link.
fn c2c_flow(r: &mut Rng, e: &Env, p: &mut Prog, seen: &mut BTreeMap<&'static str, usize>) {
    let h = r.below(2) as u16;
    let base = 116 + 14 * h;
    let rin = if h == 0 { RowId::W } else { RowId::E };
    let rout = other(rin);
    let l = r.below(e.links as u64) as usize;
    let (peer, pl) = link_peer(1, e.links, 0, l).unwrap_or((0, l));
    let tx = [base + 10, base + 11];
    let rx = [base + 12, base + 13];
    let t0: i64 = 64 + r.below(20) as i64;
    // a burst longer than the receive FIFO, whose vectors are mostly not received: LNK-002
    let sends = if r.chance(45) { 1 + r.below(e.rx_depth + 3) } else { 1 };
    let received = if r.chance(25) { 0 } else { sends };
    let mut t = t0;
    for _ in 0..sends {
        let stream = r.below(32) as u32;
        let ins = if r.chance(75) {
            Insn::Send { chip_id: peer as u32, link: l as u32, stream: row_bit(rin) + stream }
        } else {
            Insn::Transmit { chip_id: peer as u32, link: l as u32 }
        };
        let q = r.pick(&tx);
        if let (Insn::Send { .. }, Some(gm)) = (ins, read_gap(&e.geom, issuer_of(q as usize).unit, 0, rin)) {
            p.feed(r, e, rin, gm, stream, 1, t + lat(e, &ins).1);
        }
        if p.put(q, t, ins) {
            *seen.entry(ins.entry().name).or_default() += 1;
        }
        // mostly spaced so the link takes each (cycles_per_vector); LNK-001 otherwise
        t += if r.chance(85) { e.cycles_per_vector as i64 } else { 1 } + r.below(2) as i64;
    }
    for k in 0..received + r.chance(25) as u64 {
        let tr = t0 + e.link_latency as i64 + k as i64 * r.below(3) as i64 + r.below(7) as i64 - 3;
        let dst = r.below(32) as u32;
        let ins = Insn::Receive { chip_id: peer as u32, link: pl as u32, stream: row_bit(rout) + dst };
        let q = r.pick(&rx);
        if p.put(q, tr, ins) {
            *seen.entry("Receive").or_default() += 1;
            if r.chance(50) {
                let (dfunc, _) = lat(e, &ins);
                let unit = issuer_of(q as usize).unit;
                if let Some(gw) = write_gap(&e.geom, unit, 0, rout) {
                    p.observe(r, e, rout, gw, dst, 1, tr + dfunc);
                }
            }
        }
    }
    if r.chance(15) {
        p.put(r.pick(&tx), t0 + 1 + r.below(40) as i64, Insn::Deskew);
    }
}

/// A MEM queue's AGT loop: records, then a reference that runs it.
fn agt_flow(r: &mut Rng, e: &Env, p: &mut Prog, seen: &mut BTreeMap<&'static str, usize>) {
    let slices: Vec<u16> = (0..88u16).filter(|s| e.geom.slice_present(*s as u64)).collect();
    let q = r.pick(&slices);
    let mut c: i64 = 8 + r.below(30) as i64;
    let levels = if r.chance(8) { 5 } else { 1 + r.below(2) };
    for _ in 0..levels {
        let mut fields = vec![];
        if r.chance(40) {
            fields.push(Insn::Countdown { d: r.below(4) as u32 });
        }
        if r.chance(80) {
            fields.push(Insn::Step { a: if r.chance(30) { 0xFFFF - r.below(6) as u32 } else { r.below(40) as u32 } });
        }
        if r.chance(25) {
            fields.push(Insn::StreamStep { s: r.below(4) as u32 });
        }
        fields.push(Insn::Iterations { n: r.below(5) as u32 });
        for f in fields {
            p.put(q, c, f);
            c += 1;
        }
    }
    let a = r.below(8) as u32;
    let first = if r.chance(70) {
        Insn::Read { addr: a, dst: r.below(32) as u32, thread: 0 }
    } else {
        Insn::Write { addr: a, src: r.below(32) as u32, thread: r.below(2) as u32 }
    };
    *seen.entry(first.entry().name).or_default() += 1;
    p.put(q, c, first);
    for a in 0..8 {
        let v = bytes(r, e);
        p.lines.push(s_line(q, a, &v));
    }
}

/// Sync on some queues, a Notify on another.
fn barrier_flow(r: &mut Rng, p: &mut Prog) {
    let all: Vec<u16> = (0..144u16).collect();
    let t = 5 + r.below(40) as i64;
    for _ in 0..1 + r.below(3) {
        let q = r.pick(&all);
        p.put(q, t + r.below(10) as i64, Insn::Sync);
    }
    let any = r.pick(&all);
    let q = r.pick(&[any, 0xffff]);
    p.put(q, t + r.below(30) as i64, Insn::Notify);
}

fn program(r: &mut Rng, e: &Env, seen: &mut BTreeMap<&'static str, usize>) -> String {
    let mut p = Prog::default();
    for _ in 0..5 + r.below(10) {
        match r.below(20) {
            0..=7 => generic_flow(r, e, &mut p, seen),
            8..=10 => mxm_flow(r, e, &mut p, seen),
            11..=13 => sxm_flow(r, e, &mut p, seen),
            14..=15 => c2c_flow(r, e, &mut p, seen),
            16..=17 => agt_flow(r, e, &mut p, seen),
            _ => barrier_flow(r, &mut p),
        }
    }
    p.text()
}

/// What the reference programs of a profile reached.
#[derive(Default)]
struct Reached {
    codes: BTreeSet<(String, String)>,
    steps: BTreeSet<(u8, u8)>,
    diags: BTreeMap<String, usize>,
    /// How many of each instruction the generator drew.
    drawn: BTreeMap<&'static str, usize>,
}

fn diff_profile(name: &str, seed: u64, n: usize) -> Reached {
    let cfg = cfg_path(name).to_str().unwrap().to_string();
    let e = env(name);
    let mut seen: BTreeMap<&'static str, usize> = BTreeMap::new();
    let progs: Vec<String> = {
        let mut r = Rng::new(
            seed + std::env::var("ISS_DIFF_SEED").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(0) * 1000,
        );
        (0..n).map(|_| program(&mut r, &e, &mut seen)).collect()
    };
    if let Ok(dir) = std::env::var("ISS_DIFF_SAVE") {
        for (i, p) in progs.iter().enumerate() {
            std::fs::write(format!("{dir}/mixed-{}-{i}.txt", name.replace('/', "_")), p).unwrap();
        }
    }
    let reached = std::sync::Mutex::new(Reached::default());
    let bad = parallel(&progs, |prog| {
        let lint = prog.len() % 3 == 0;
        let mut args = vec![cfg.as_str(), "--trace", "--max-cycles", "6000"];
        if lint {
            args.push("--lint");
        }
        let ours = run_iss(&args, prog);
        let theirs = run_sim(&args, prog);
        let out = text(&theirs.stdout);
        {
            let mut m = reached.lock().unwrap();
            let m = &mut *m;
            pairs(&out, &mut m.codes, &mut m.steps);
            for l in out.lines().filter(|l| l.starts_with("T diag")) {
                *m.diags.entry(l.rsplit(' ').next().unwrap().to_string()).or_default() += 1;
            }
        }
        let mut every = args.clone();
        every.push("--tick-every-cycle");
        let ticked = run_bin(iss(), &every, prog.as_bytes());
        if ours.status.code() == theirs.status.code()
            && ours.stdout == theirs.stdout
            && ticked.status.code() == theirs.status.code()
            && ticked.stdout == theirs.stdout
        {
            return Ok(());
        }
        Err(format!(
            "({name}):\n{prog}\nours (exit {:?}):\n{}\nticking every cycle (exit {:?}):\n{}\ntsp_sim (exit {:?}):\n{out}",
            ours.status.code(),
            text(&ours.stdout),
            ticked.status.code(),
            text(&ticked.stdout),
            theirs.status.code(),
        ))
    });
    let mut reached = reached.into_inner().unwrap();
    reached.drawn = seen;
    let seen = &reached.drawn;
    eprintln!(
        "mixed_diff {name}: {n} programs, {} differ; {} code pairs {:?}, step pairs {:?}; diagnostics {:?}; instructions drawn {seen:?}",
        bad.len(),
        reached.codes.len(),
        reached.codes,
        reached.steps,
        reached.diags
    );
    assert!(
        bad.is_empty(),
        "{} of {n} programs differ on {name}; first:\n{}",
        bad.len(),
        bad[0].chars().take(6000).collect::<String>()
    );
    reached
}

fn count() -> usize {
    std::env::var("ISS_DIFF_N").ok().and_then(|s| s.parse().ok()).unwrap_or(150)
}

/// The floors of the pair coverage of one profile's programs (only when the run is long enough to
/// reach them): the codes the generator exists to combine, and the pairs of steps.
fn pair_coverage_floor(name: &str, n: usize, got: &Reached) {
    if n < 100 {
        eprintln!("mixed_diff {name}: pair floor skipped ({n} < 100)");
        return;
    }
    // the units the programs exist to mix: LW (MXM weights), ABC and ACC (MXM), Permute (SXM), Send and
    // Receive (C2C), Iterations (MEM loops) and Repeat; a draw count of 0 is a generator that lost one
    for entry in ["LW", "ABC", "ACC", "Permute", "Send", "Receive", "Iterations", "Repeat"] {
        assert!(got.drawn.get(entry).copied().unwrap_or(0) > 0, "{name}: no {entry} drawn: {:?}", got.drawn);
    }
    assert!(got.codes.len() >= 30, "{name}: {} code pairs: {:?}", got.codes.len(), got.codes);
    assert!(got.steps.len() >= 3, "{name}: {} step pairs: {:?}", got.steps.len(), got.steps);
    for pair in [(4, 5), (5, 6)] {
        assert!(got.steps.contains(&pair), "{name}: no step pair {pair:?}: {:?}", got.steps);
    }
    for code in [
        "OTS-E-TIM-001",
        "OTS-E-DEC-003",
        "OTS-E-QUE-001",
        "OTS-E-QUE-002",
        "OTS-E-RES-001",
        "OTS-E-RES-003",
        "OTS-E-SRF-002",
        "OTS-E-SXM-001",
        "OTS-E-LNK-001",
    ] {
        assert!(got.diags.contains_key(code), "{name}: no {code}: {:?}", got.diags);
    }
}

#[test]
fn mixed_programs_match_tsp_sim_ref() {
    let got = diff_profile("ref", 201, count());
    pair_coverage_floor("ref", count(), &got);
}

#[test]
fn mixed_programs_match_tsp_sim_mini() {
    let got = diff_profile("mini", 202, count());
    pair_coverage_floor("mini", count(), &got);
}

#[test]
fn mixed_programs_match_tsp_sim_f3_and_timing_profiles() {
    for (k, name) in ["mid_min", "mini_small", "ref_small", "ref_slow", "link_tight"].iter().enumerate() {
        diff_profile(name, 211 + k as u64, count() / 2);
    }
}
