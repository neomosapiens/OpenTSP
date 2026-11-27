//! Tests of `slices/sxm`, moved from the module's in-source test block (they use the crate's public API).
//! Ports of the SXM unit tests of the Sail model (`unit_tests/test_slices.sail`:
//! `test_sxm_rotate_range`, `test_sxm_cross_timing`, `test_sxm_map_flag_and_ready`;
//! `test_streams.sail`: the reset maps), and the ISS's own tests of each instruction, the map
//! staging, the Permute ready cycle and bijectivity, and sxselect's partial effect.
use ots_isa::Insn;
use tsp_iss::geom::MAX_LANES;
use tsp_iss::layout::issuer_of;
use tsp_iss::observe::Observer;
use tsp_iss::slices::sxm::*;
use tsp_iss::slices::{Exec, Issue};
use tsp_iss::srf::Vec320;
use tsp_iss::{Diag, Machine};

use ots_profile::Profile;
use tsp_iss::layout::{read_gap, write_gap, RowId, UnitKind};
use tsp_iss::links::NoLinks;
use tsp_iss::srf::SrfLookup;

fn machine(name: &str) -> Machine {
    let mut m = Machine::new(&Profile::load(name).unwrap(), 0, Box::new(NoLinks)).unwrap();
    m.edit_state(|state| state.activity.set_enabled(true)).unwrap(); // the tests read the activity counters, which are off unless asked for
    m
}

#[derive(Default)]
struct Rec {
    diags: Vec<(u64, u16, Diag)>,
}

impl Observer for Rec {
    fn diag(&mut self, c: u64, queue: u16, code: Diag) {
        self.diags.push((c, queue, code));
    }
}

/// `issue_on` + `execute_test`: issue `i` on queue `q` as dispatched at cycle `c`.
fn exec(m: &mut Machine, q: usize, c: u64, i: &Insn) -> Exec {
    let ctx = Issue {
        cycle: c,
        queue: q,
        element: 0,
        addr_offset: 0,
        stream_offset: 0,
        thread: 0,
        lat: m.latency_of_instr(i),
    };
    m.execute(i, &ctx, &mut Rec::default())
}

fn tick_to(m: &mut Machine, a: u64, b: u64) -> Rec {
    let mut o = Rec::default();
    for c in a..=b {
        m.tick(c, &mut o);
    }
    o
}

fn diag(d: Diag) -> Exec {
    Exec::Diagnostic(d)
}

// West SXM queues (core/queues.sail): DST 116..119, TSP 120, 121, SHF 122, 123, PRM 124, SEL 125;
// the east SXM's are the same + 14.
const Q_DST: usize = 116;
const Q_TSP: usize = 120;
const Q_SHF: usize = 122;
const Q_PRM: usize = 124;
const Q_SEL: usize = 125;
const EAST: usize = 14;

fn unit_of(q: usize) -> UnitKind {
    issuer_of(q).unit
}

/// The row the unit of queue `q` reads its operands from, and the row it writes.
fn rows_of(q: usize) -> (RowId, RowId) {
    let west = q < 130;
    let dst = matches!(unit_of(q), UnitKind::SxmWDst | UnitKind::SxmEDst);
    let rin = if west { RowId::W } else { RowId::E };
    let rout = if dst {
        rin
    } else if west {
        RowId::E
    } else {
        RowId::W
    };
    (rin, rout)
}

/// A stream reference (row bit + stream) on the given row.
fn sref(r: RowId, id: u32) -> u32 {
    id + if r == RowId::W { 32 } else { 0 }
}

/// Feed stream `id` of queue `q`'s input row so that a read at its gap in cycle `t` finds `v`.
fn feed(m: &mut Machine, q: usize, id: u32, v: &Vec320, t: u64) {
    let (rin, _) = rows_of(q);
    let i = issuer_of(q);
    let g = read_gap(m.geom(), i.unit, i.slice as usize, rin).unwrap();
    assert!(g > 0 && t > 0);
    m.edit_state(|state| state.srf.write(rin, id as u8, g - 1, t - 1, v)).unwrap();
}

/// The value queue `q` committed on stream `id` of its output row in cycle `commit` (after the
/// ticks up to it).
fn result(m: &Machine, q: usize, id: u32, commit: u64) -> Option<Vec320> {
    let (_, rout) = rows_of(q);
    let i = issuer_of(q);
    let g = write_gap(m.geom(), i.unit, i.slice as usize, rout).unwrap();
    match m.srf().read(rout, id as u8, g + 1, commit as i64 + 1) {
        (SrfLookup::Found, v) => Some(v),
        _ => None,
    }
}

/// Lane l holds `f(l)` (lanes at or beyond `vlanes` stay zero).
fn lanes(m: &Machine, f: impl Fn(usize) -> u8) -> Vec320 {
    let mut v = [0u8; MAX_LANES];
    for (l, x) in v.iter_mut().enumerate().take(m.geom().vlanes as usize) {
        *x = f(l);
    }
    v
}

fn shift(dir: u32, n: u32, s: u32, d: u32) -> Insn {
    Insn::Shift { dst: d, src: s, dir, amount: n }
}
fn permute(map: u32, s: u32, d: u32) -> Insn {
    Insn::Permute { dst: d, src: s, map }
}
fn distribute(map: u32, s: u32, d: u32) -> Insn {
    Insn::Distribute { dst: d, src: s, map }
}
fn loadmap(map: u32, s: u32) -> Insn {
    Insn::LoadMap { src: s, map }
}
fn loadmapwide(map: u32, s: u32, s2: u32) -> Insn {
    Insn::LoadMapWide { src: s, map, src2: s2 }
}

/// Run `i` on queue `q` at cycle `c` (operands fed for `c + d_skew`), tick to its commit and return
/// the result of stream `id`.
fn run1(m: &mut Machine, q: usize, c: u64, i: &Insn, ins: &[(u32, Vec320)], id: u32) -> (Exec, Option<Vec320>) {
    let lat = m.latency_of_instr(i);
    for (s, v) in ins {
        feed(m, q, *s, v, c + lat.d_skew as u64);
    }
    let r = exec(m, q, c, i);
    let commit = c + lat.d_func as u64;
    m.unit_state_flush(c, &mut Rec::default());
    tick_to(m, 0, commit);
    (r, result(m, q, id, commit))
}

// model: unit_tests/test_streams.sail test_reset_state (the SXM part)
#[test]
fn reset_maps_are_identity_bijective_and_ready() {
    let m = machine("ref");
    for x in 0..2 {
        for k in 0..16 {
            assert!(m.sxm().bijective(x, k) && m.sxm().ready(x, k) == 0);
            assert!((0..320).all(|l| m.sxm().map(x, k)[l] as usize == l));
        }
    }
}

// model: unit_tests/test_slices.sail test_sxm_rotate_range
#[test]
fn rotate_needs_n_squared_streams_on_the_row() {
    let mut m = machine("ref");
    assert_eq!(exec(&mut m, 120, 50, &Insn::Rotate { dst: 0b010001, src: 0b100000, n: 4 }), diag(Diag::Opr002));
    assert_eq!(exec(&mut m, 120, 50, &Insn::Rotate { dst: 0b011000, src: 0b100000, n: 3 }), diag(Diag::Opr002));
}

// model: unit_tests/test_slices.sail test_sxm_cross_timing
#[test]
fn shift_and_permute_have_the_cross_latency() {
    let m = machine("ref");
    let cross = m.latency_of_instr(&shift(1, 16, 0b100000, 0b000101));
    assert_eq!(m.latency_of_instr(&permute(0, 0b100000, 0b000101)), cross);
    assert!(cross.d_func as u64 >= m.geom().superlanes);
    let plain = m.latency_of_instr(&distribute(0, 0b100000, 0b100001));
    assert_eq!(m.latency_of_instr(&Insn::Transpose { dst: 1, src: 3 }), plain);
    assert!(plain.d_func < cross.d_func);
}

// model: unit_tests/test_slices.sail test_sxm_map_flag_and_ready
#[test]
fn staged_map_carries_flag_and_ready_and_a_clash_drops_all_three() {
    let mut m = machine("ref");
    let mut mv = identity_map();
    mv[1] = mv[2];
    m.edit_state(|state| state.sxm.staged.push(Staged { q: 138, sxm: 1, map: 3, value: mv, bij: false, ready: 50 }))
        .unwrap();
    assert!(m.sxm().bijective(1, 3) && m.sxm().ready(1, 3) == 0); // staged, not yet applied
    m.unit_state_flush(30, &mut Rec::default());
    assert!(!m.sxm().bijective(1, 3) && m.sxm().ready(1, 3) == 50 && m.sxm().map(1, 3)[1] == 2);
    assert!(m.sxm().bijective(0, 3) && m.sxm().ready(0, 3) == 0 && m.sxm().bijective(1, 4) && m.sxm().ready(1, 4) == 0);
    m.edit_state(|state| state.sxm.staged.push(Staged { q: 138, sxm: 1, map: 4, value: mv, bij: false, ready: 77 }))
        .unwrap();
    m.edit_state(|state| state.sxm.staged.push(Staged { q: 131, sxm: 1, map: 4, value: mv, bij: false, ready: 78 }))
        .unwrap();
    let mut o = Rec::default();
    m.unit_state_flush(31, &mut o);
    assert!(m.sxm().bijective(1, 4) && m.sxm().ready(1, 4) == 0 && m.sxm().map(1, 4)[1] == 1);
    // RES-003 once per queue, in queue order
    assert_eq!(o.diags, vec![(31, 131, Diag::Res003), (31, 138, Diag::Res003)]);
}

#[test]
fn loads_of_different_maps_or_sxms_do_not_clash_and_the_oldest_of_one_queue_stands() {
    let mut m = machine("ref");
    let id = identity_map();
    let mut a = id;
    a[0] = 5;
    let mut b = id;
    b[0] = 6;
    // queue 124 stages map 2 twice: the oldest stands (the model applies its list newest first)
    m.edit_state(|state| state.sxm.staged.push(Staged { q: 124, sxm: 0, map: 2, value: a, bij: true, ready: 10 }))
        .unwrap();
    m.edit_state(|state| state.sxm.staged.push(Staged { q: 124, sxm: 0, map: 2, value: b, bij: true, ready: 11 }))
        .unwrap();
    // a different map, and the other SXM's map 2: no clash
    m.edit_state(|state| state.sxm.staged.push(Staged { q: 116, sxm: 0, map: 3, value: b, bij: true, ready: 12 }))
        .unwrap();
    m.edit_state(|state| state.sxm.staged.push(Staged { q: 138, sxm: 1, map: 2, value: b, bij: true, ready: 13 }))
        .unwrap();
    let mut o = Rec::default();
    m.unit_state_flush(7, &mut o);
    assert!(o.diags.is_empty());
    assert_eq!((m.sxm().map(0, 2)[0], m.sxm().ready(0, 2)), (5, 10));
    assert_eq!((m.sxm().map(0, 3)[0], m.sxm().ready(0, 3)), (6, 12));
    assert_eq!((m.sxm().map(1, 2)[0], m.sxm().ready(1, 2)), (6, 13));
}

#[test]
fn shift_moves_lanes_across_superlanes_with_zero_fill() {
    for prof in ["ref", "mini"] {
        let mut m = machine(prof);
        let vl = m.geom().vlanes as usize;
        let v = lanes(&m, |l| (l % 250) as u8 + 1);
        let s = sref(RowId::W, 3);
        for (dir, n) in [(0u32, 0u32), (0, 1), (0, 5), (1, 3), (1, 15)] {
            let n = n.min(vl as u32 - 1);
            let (r, out) = run1(&mut m, Q_SHF, 60, &shift(dir, n, s, 7), &[(3, v)], 7);
            assert_eq!(r, Exec::Issued);
            let want = lanes(&m, |l| {
                let j = if dir == 0 { l as i64 - n as i64 } else { l as i64 + n as i64 };
                if (0..vl as i64).contains(&j) {
                    v[j as usize]
                } else {
                    0
                }
            });
            assert_eq!(out.unwrap(), want, "{prof} dir {dir} n {n}");
            m.reset();
        }
    }
}

#[test]
fn shift_of_an_unwritten_stream_is_tim_001_and_an_operand_on_the_wrong_row_dir_001() {
    let mut m = machine("ref");
    assert_eq!(exec(&mut m, Q_SHF, 60, &shift(0, 1, sref(RowId::W, 3), 7)), diag(Diag::Tim001));
    // the west shifter reads the W row: an operand on the E row is read upstream of it
    let mut m = machine("ref");
    assert_eq!(exec(&mut m, Q_SHF, 60, &shift(0, 1, sref(RowId::E, 3), 7)), diag(Diag::Dir001));
}

#[test]
fn east_shifter_reads_east_and_writes_west() {
    let mut m = machine("ref");
    let q = Q_SHF + EAST;
    let v = lanes(&m, |l| (l as u8).wrapping_add(1));
    // output row W: a destination reference with the W bit
    let (r, out) = run1(&mut m, q, 60, &shift(1, 2, sref(RowId::E, 3), sref(RowId::W, 9)), &[(3, v)], 9);
    assert_eq!(r, Exec::Issued);
    let o = out.unwrap();
    assert_eq!(&o[..4], &[3, 4, 5, 6]);
}

fn set_map(m: &mut Machine, x: usize, k: usize, vals: &[u16], ready: u64) {
    let mut mv = identity_map();
    for (l, e) in vals.iter().enumerate() {
        mv[l] = *e;
    }
    m.edit_state(|state| state.sxm.maps[x][k] = mv).unwrap();
    m.edit_state(|state| state.sxm.bij[x][k] = Machine::<512>::sxm_map_distinct(&mv, state.geom.vlanes as usize))
        .unwrap();
    m.edit_state(|state| state.sxm.ready[x][k] = ready).unwrap();
}

#[test]
fn permute_gathers_by_map_entry_and_masks_entries_at_or_beyond_vlanes() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    let v = lanes(&m, |l| (l % 200) as u8 + 1);
    let rev: Vec<u16> = (0..vl).map(|l| (vl - 1 - l) as u16).collect();
    set_map(&mut m, 0, 5, &rev, 0);
    let (r, out) = run1(&mut m, Q_PRM, 60, &permute(5, sref(RowId::W, 1), 2), &[(1, v)], 2);
    assert_eq!(r, Exec::Issued);
    assert_eq!(out.unwrap(), lanes(&m, |l| v[vl - 1 - l]));
    // an entry at or beyond vlanes (the map carries the mask) gives zero
    let mut m = machine("ref");
    let mut e: Vec<u16> = (0..vl as u16).collect();
    e[3] = 700;
    e[4] = 320;
    set_map(&mut m, 0, 1, &e, 0);
    let (r, out) = run1(&mut m, Q_PRM, 60, &permute(1, sref(RowId::W, 1), 2), &[(1, v)], 2);
    assert_eq!(r, Exec::Issued);
    assert_eq!(out.unwrap(), lanes(&m, |l| if l == 3 || l == 4 { 0 } else { v[l] }));
}

#[test]
fn permute_before_the_map_is_ready_is_res_001_and_a_non_bijective_map_sxm_001() {
    let mut m = machine("ref");
    let v = lanes(&m, |l| l as u8);
    set_map(&mut m, 0, 2, &[1, 0], 100);
    let lat = m.latency_of_instr(&permute(2, sref(RowId::W, 1), 2));
    feed(&mut m, Q_PRM, 1, &v, 99 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, Q_PRM, 99, &permute(2, sref(RowId::W, 1), 2)), diag(Diag::Res001));
    feed(&mut m, Q_PRM, 1, &v, 100 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, Q_PRM, 100, &permute(2, sref(RowId::W, 1), 2)), Exec::Issued);
    // a repeated source lane: SXM-001, no effect
    set_map(&mut m, 0, 3, &[1, 1], 0);
    assert!(!m.sxm().bijective(0, 3));
    feed(&mut m, Q_PRM, 1, &v, 120 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, Q_PRM, 120, &permute(3, sref(RowId::W, 1), 2)), diag(Diag::Sxm001));
    // the operand check comes first: an unwritten stream is TIM-001 whatever the map
    assert_eq!(exec(&mut m, Q_PRM, 130, &permute(3, sref(RowId::W, 9), 2)), diag(Diag::Tim001));
    assert_eq!(exec(&mut m, Q_PRM, 5, &permute(2, sref(RowId::W, 9), 2)), diag(Diag::Tim001));
}

#[test]
fn loadmap_stages_the_map_with_flag_and_ready_cycle() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    // lane l holds the entry 255 - l % 256: LoadMap gives 8-bit entries, and lanes 256.. repeat 0..
    let rev = lanes(&m, |l| (255 - l % 256) as u8);
    let (r, _) = run1(&mut m, Q_PRM, 40, &loadmap(4, sref(RowId::W, 6)), &[(6, rev)], 0);
    assert_eq!(r, Exec::Issued);
    assert!(m.sxm().map(0, 4)[..vl].iter().enumerate().all(|(l, e)| *e as usize == 255 - l % 256));
    assert!(m.sxm().map(0, 4)[vl..].iter().all(|e| *e == 0));
    assert_eq!(m.sxm().bijective(0, 4), vl <= 256); // 320 lanes repeat entries
    assert_eq!(m.sxm().ready(0, 4), 40 + m.latency_of_instr(&loadmap(0, 0)).d_skew as u64 + m.geom().superlanes);
    // the east map 4 is untouched
    assert!(m.sxm().bijective(1, 4) && m.sxm().ready(1, 4) == 0 && m.sxm().map(1, 4)[7] == 7);
    // the load is staged: no effect in the cycle itself
    let mut m = machine("ref");
    let lat = m.latency_of_instr(&loadmap(4, 0));
    feed(&mut m, Q_PRM, 6, &rev, 40 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, Q_PRM, 40, &loadmap(4, sref(RowId::W, 6))), Exec::Issued);
    assert_eq!(m.sxm().map(0, 4)[0], 0);
}

#[test]
fn loadmapwide_takes_two_bits_of_the_second_stream_as_the_high_part() {
    let mut m = machine("ref");
    let lo = lanes(&m, |l| l as u8);
    let hi = lanes(&m, |l| (l / 256) as u8 | 0xFC); // bits above the low two are ignored
    let (r, _) = run1(&mut m, Q_DST, 40, &loadmapwide(2, sref(RowId::W, 6), sref(RowId::W, 7)), &[(6, lo), (7, hi)], 0);
    assert_eq!(r, Exec::Issued);
    assert!((0..320).all(|l| m.sxm().map(0, 2)[l] as usize == l));
    assert!(m.sxm().bijective(0, 2));
    // a high part of 1 on lane 0: entry 256, a lane index below 320 that another lane also holds
    let mut m = machine("ref");
    let hi1 = lanes(&m, |l| if l == 0 { 1 } else { (l / 256) as u8 });
    let (r, _) =
        run1(&mut m, Q_DST, 40, &loadmapwide(2, sref(RowId::W, 6), sref(RowId::W, 7)), &[(6, lo), (7, hi1)], 0);
    assert_eq!(r, Exec::Issued);
    assert_eq!(m.sxm().map(0, 2)[0], 256);
    assert!(!m.sxm().bijective(0, 2)); // lane 256 also holds 256
                                       // a missing second operand: the first error
    let mut m = machine("ref");
    let lat = m.latency_of_instr(&loadmapwide(2, 0, 0));
    feed(&mut m, Q_DST, 6, &lo, 40 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, Q_DST, 40, &loadmapwide(2, sref(RowId::W, 6), sref(RowId::W, 7))), diag(Diag::Tim001));
    assert!(m.sxm().staged.is_empty());
}

#[test]
fn distribute_works_inside_each_superlane_and_the_map_is_usable_the_next_cycle() {
    let mut m = machine("ref");
    let v = lanes(&m, |l| (l as u8).wrapping_add(1));
    // reverse the 16 lanes of every superlane; entries >= 16 select zero
    let mut vals: Vec<u16> = (0..320).map(|l| (15 - l % 16) as u16).collect();
    vals[2] = 16;
    vals[18] = 300;
    set_map(&mut m, 0, 1, &vals, 0);
    let (r, out) = run1(&mut m, Q_DST, 60, &distribute(1, sref(RowId::W, 1), sref(RowId::W, 2)), &[(1, v)], 2);
    assert_eq!(r, Exec::Issued);
    let o = out.unwrap();
    for l in 0..320 {
        let want = if l == 2 || l == 18 { 0 } else { v[16 * (l / 16) + 15 - l % 16] };
        assert_eq!(o[l], want, "lane {l}");
    }
    // a Distribute right after the map's load: the load applies at the end of its cycle
    let mut m = machine("ref");
    let lo = lanes(&m, |l| (15 - l % 16) as u8);
    let data = lanes(&m, |l| (l as u8).wrapping_add(1));
    let lat = m.latency_of_instr(&loadmap(1, 0));
    feed(&mut m, Q_DST, 6, &lo, 50 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, Q_DST, 50, &loadmap(1, sref(RowId::W, 6))), Exec::Issued);
    m.unit_state_flush(50, &mut Rec::default());
    feed(&mut m, Q_DST, 1, &data, 51 + lat.d_skew as u64);
    let ld = m.latency_of_instr(&distribute(1, 0, 0));
    assert_eq!(exec(&mut m, Q_DST, 51, &distribute(1, sref(RowId::W, 1), sref(RowId::W, 2))), Exec::Issued);
    tick_to(&mut m, 0, 51 + ld.d_func as u64);
    let o = result(&m, Q_DST, 2, 51 + ld.d_func as u64).unwrap();
    assert_eq!(o[0], data[15]);
}

#[test]
fn rotate_builds_the_n_by_n_tile_rotations_in_each_superlane() {
    for n in [3usize, 4] {
        let mut m = machine("ref");
        let v = lanes(&m, |l| (l % 100) as u8 + 1);
        let dst = 10u32;
        let lat = m.latency_of_instr(&Insn::Rotate { dst: 0, src: 0, n: 3 });
        feed(&mut m, Q_TSP, 4, &v, 70 + lat.d_skew as u64);
        let r = exec(&mut m, Q_TSP, 70, &Insn::Rotate { dst, src: sref(RowId::W, 4), n: n as u32 });
        assert_eq!(r, Exec::Issued);
        tick_to(&mut m, 0, 70 + lat.d_func as u64);
        for t in 0..n * n {
            let o = result(&m, Q_TSP, dst + t as u32, 70 + lat.d_func as u64).unwrap();
            let (tr, tc) = (t / n, t % n);
            for g in 0..20 {
                for rr in 0..16 {
                    let want = if rr < n * n {
                        let (r, c) = (rr / n, rr % n);
                        v[16 * g + ((r + tr) % n) * n + (c + tc) % n]
                    } else {
                        0
                    };
                    assert_eq!(o[16 * g + rr], want, "n {n} tile {t} lane {rr}");
                }
            }
        }
        // the stream after the last output is not written
        assert!(result(&m, Q_TSP, dst + (n * n) as u32, 70 + lat.d_func as u64).is_none());
    }
}

#[test]
fn transpose_exchanges_rows_and_lanes_inside_each_superlane() {
    let mut m = machine("ref");
    let lat = m.latency_of_instr(&Insn::Transpose { dst: 0, src: 0 });
    // stream r (0..15, west row) lane 16 g + c = 16 r + c + g
    let ins: Vec<Vec320> = (0..16).map(|r| lanes(&m, |l| (16 * r + l % 16 + l / 16) as u8)).collect();
    for (r, v) in ins.iter().enumerate() {
        feed(&mut m, Q_TSP, r as u32, v, 30 + lat.d_skew as u64);
    }
    // the field holds the stream / 16: W row, group 0 is 2; destination E group 1 is 1
    assert_eq!(exec(&mut m, Q_TSP, 30, &Insn::Transpose { dst: 1, src: 2 }), Exec::Issued);
    tick_to(&mut m, 0, 30 + lat.d_func as u64);
    for c in 0..16 {
        let o = result(&m, Q_TSP, 16 + c as u32, 30 + lat.d_func as u64).unwrap();
        for g in 0..20 {
            for r in 0..16 {
                assert_eq!(o[16 * g + r], ins[r][16 * g + c], "g {g} r {r} c {c}");
            }
        }
    }
    // a missing input stream: TIM-001, nothing scheduled
    let mut m = machine("ref");
    for r in 0..15 {
        feed(&mut m, Q_TSP, r, &[1; MAX_LANES], 30 + lat.d_skew as u64);
    }
    assert_eq!(exec(&mut m, Q_TSP, 30, &Insn::Transpose { dst: 1, src: 2 }), diag(Diag::Tim001));
}

#[test]
fn sxselect_picks_per_lane_and_a_bad_selector_zeroes_only_its_lane() {
    let mut m = machine("ref");
    let lat = m.latency_of_instr(&Insn::SelectS { dst: 0, a: 0, b: 0, sel: 0, c: 0 });
    let a = lanes(&m, |_| 0x11);
    let b = lanes(&m, |_| 0x22);
    let c = lanes(&m, |_| 0x33);
    let sel = lanes(&m, |l| match l % 5 {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        _ => 255,
    });
    for (id, v) in [(1, &a), (2, &b), (3, &c), (4, &sel)] {
        feed(&mut m, Q_SEL, id, v, 20 + lat.d_skew as u64);
    }
    let w = |id| sref(RowId::W, id);
    let r = exec(&mut m, Q_SEL, 20, &Insn::SelectS { dst: 9, a: w(1), b: w(2), sel: w(4), c: w(3) });
    assert_eq!(r, diag(Diag::Opr001)); // once, whatever the number of bad lanes
    tick_to(&mut m, 0, 20 + lat.d_func as u64);
    let o = result(&m, Q_SEL, 9, 20 + lat.d_func as u64).unwrap();
    for l in 0..320 {
        let want = [0x11, 0x22, 0x33, 0, 0][l % 5];
        assert_eq!(o[l], want, "lane {l}");
    }
    // all selectors good: no diagnostic
    let mut m = machine("ref");
    for (id, v) in [(1, &a), (2, &b), (3, &c)] {
        feed(&mut m, Q_SEL, id, v, 20 + lat.d_skew as u64);
    }
    let sel3 = lanes(&m, |l| (l % 3) as u8);
    feed(&mut m, Q_SEL, 4, &sel3, 20 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, Q_SEL, 20, &Insn::SelectS { dst: 9, a: w(1), b: w(2), sel: w(4), c: w(3) }), Exec::Issued);
}

#[test]
fn sxselect_with_a_bad_destination_is_dir_001_and_a_bad_selector_opr_001_over_it() {
    // the write error (a destination on the wrong row has no write gap: DIR-001) comes from
    // sxm_emit; a bad selector reports OPR-001 over it (`if bad then Diagnostic(OPR_RANGE) else r`)
    let mut m = machine("ref");
    let lat = m.latency_of_instr(&Insn::SelectS { dst: 0, a: 0, b: 0, sel: 0, c: 0 });
    let ones = lanes(&m, |_| 1);
    for id in 1..=4 {
        feed(&mut m, Q_SEL, id, &ones, 20 + lat.d_skew as u64);
    }
    let w = |id| sref(RowId::W, id);
    // the west select writes the E row; a W-row destination is not its output row
    let r = exec(&mut m, Q_SEL, 20, &Insn::SelectS { dst: w(9), a: w(1), b: w(2), sel: w(4), c: w(3) });
    assert_eq!(r, diag(Diag::Dir001));
    // a bad selector reports OPR-001 even then
    let mut m = machine("ref");
    let mut sel = ones;
    sel[0] = 9;
    for id in [1, 2, 3] {
        feed(&mut m, Q_SEL, id, &ones, 20 + lat.d_skew as u64);
    }
    feed(&mut m, Q_SEL, 4, &sel, 20 + lat.d_skew as u64);
    let r = exec(&mut m, Q_SEL, 20, &Insn::SelectS { dst: w(9), a: w(1), b: w(2), sel: w(4), c: w(3) });
    assert_eq!(r, diag(Diag::Opr001));
}

/// (a, b, c, sel, the streams fed on the W row, the diagnostic that stands)
type Case = (u32, u32, u32, u32, &'static [u32], Diag);

#[test]
fn sxselect_reports_the_error_of_the_first_bad_operand_in_the_order_a_b_c_sel() {
    // operand read errors: a stream never written is TIM-001, one on the row the unit does not read is DIR-001
    let lat = machine("ref").latency_of_instr(&Insn::SelectS { dst: 0, a: 0, b: 0, sel: 0, c: 0 });
    let w = |id| sref(RowId::W, id);
    let e = |id| sref(RowId::E, id);
    // (a, b, c, sel, streams fed on the W row, the diagnostic that stands)
    let cases: [Case; 6] = [
        (w(1), w(2), w(3), e(4), &[2, 3], Diag::Tim001), // a missing, sel wrong row: a first
        (w(1), e(2), w(3), w(4), &[1, 4], Diag::Dir001), // b wrong row, c missing: b first
        (w(1), w(2), w(3), e(4), &[1, 2], Diag::Tim001), // c missing, sel wrong row: c first
        (w(1), w(2), w(3), w(4), &[1, 2, 3], Diag::Tim001), // only sel missing
        (e(1), w(2), w(3), w(4), &[2, 3, 4], Diag::Dir001), // only a wrong row
        (w(1), w(2), e(3), w(4), &[1, 2], Diag::Dir001), // c wrong row, sel missing: c first
    ];
    for (a, b, c, sel, fed, want) in cases {
        let mut m = machine("ref");
        let ones = lanes(&m, |_| 1);
        for &id in fed {
            feed(&mut m, Q_SEL, id, &ones, 20 + lat.d_skew as u64);
        }
        let r = exec(&mut m, Q_SEL, 20, &Insn::SelectS { dst: 9, a, b, sel, c });
        assert_eq!(r, diag(want), "a {a} b {b} c {c} sel {sel}");
    }
}

#[test]
fn a_transpose_counts_sixteen_vectors_in_and_out_on_its_unit() {
    let mut m = machine("ref");
    let lat = m.latency_of_instr(&Insn::Transpose { dst: 0, src: 0 });
    for r in 0..16 {
        feed(&mut m, Q_TSP, r, &[1; MAX_LANES], 30 + lat.d_skew as u64);
    }
    assert_eq!(exec(&mut m, Q_TSP, 30, &Insn::Transpose { dst: 1, src: 2 }), Exec::Issued);
    let a = m.activity();
    assert_eq!((a.sxm_in_vectors, a.sxm_out_vectors), (16, 16));
    assert_eq!(a.issued[ots_isa::EntryId::Transpose.index()], 1);
    assert_eq!(a.srf_writes.iter().sum::<u64>(), 16, "sixteen result streams");
    assert_eq!(a.active_cycles[3], 1, "the SXM class was active in one cycle");
    // a rejected Transpose (a missing input stream) counts nowhere
    let mut m = machine("ref");
    for r in 0..15 {
        feed(&mut m, Q_TSP, r, &[1; MAX_LANES], 30 + lat.d_skew as u64);
    }
    assert!(matches!(exec(&mut m, Q_TSP, 30, &Insn::Transpose { dst: 1, src: 2 }), Exec::Diagnostic(_)));
    assert_eq!((m.activity().sxm_in_vectors, m.activity().srf_writes), (0, [0, 0]));
}
