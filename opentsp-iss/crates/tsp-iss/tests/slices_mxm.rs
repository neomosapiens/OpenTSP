//! Tests of `slices/mxm`, moved from the module's in-source test block (they use the crate's public API).
//! Ports of the MXM unit tests of the Sail model (`unit_tests/test_vxm_mxm.sail`:
//! `test_mxm_f16_abc_two_cycles`, `test_mxm_mixed_sign`, and `test_slices.sail`:
//! `test_mxm_stage_readiness`, `test_mxm_stage_conflict`, `test_mxm_f16_plane_pair`,
//! `test_mxm_operand_checks`), and the ISS's own tests of the plane
//! rules: the weight buffer (RES-002), the f16 plane pair (RES-001), the start-of-cycle staging
//! and `unit_state_flush` (RES-003), the ACC commit drop, accumulators and the ready cycles.
use ots_isa::Insn;
use std::sync::Arc;
use tsp_iss::geom::MAX_LANES;
use tsp_iss::observe::Observer;
use tsp_iss::slices::mxm::*;
use tsp_iss::slices::{Exec, Issue};
use tsp_iss::srf::Vec320;
use tsp_iss::{Diag, Machine};

use ots_profile::Profile;
use tsp_iss::layout::{read_gap, RowId, UnitKind};
use tsp_iss::links::NoLinks;
use tsp_iss::srf::SrfLookup;

fn machine(name: &str) -> Machine {
    Machine::new(&Profile::load(name).unwrap(), 0, Box::new(NoLinks)).unwrap()
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

/// `slice_test_set_issue_context` + `execute_test`: issue `i` on queue `q` as dispatched at cycle `c`.
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

const I8: u32 = 0;
const F16: u32 = 1;
const U8: u32 = 2;
const A_I32: u32 = 2;
const A_F32: u32 = 3;

fn lw(t: u32, group: u32) -> Insn {
    Insn::LW { r#type: t, src: group }
}
fn iw(t: u32) -> Insn {
    Insn::IW { r#type: t }
}
fn iws(t: u32, group: u32) -> Insn {
    Insn::IWS { r#type: t, src: group }
}
fn abc(t: u32, s: u32) -> Insn {
    Insn::ABC { r#type: t, src: s }
}
fn acc(t: u32, dst: u32, mode: u32, a: u32) -> Insn {
    Insn::ACC { r#type: t, dst, mode, acc: a }
}

/// Plane `pl`'s queues: LW, ABC, ACC at 104 + 3 pl + 0, 1, 2.
fn q_lw(pl: usize) -> usize {
    104 + 3 * pl
}
fn q_abc(pl: usize) -> usize {
    105 + 3 * pl
}
fn q_acc(pl: usize) -> usize {
    106 + 3 * pl
}

/// The stream row an MXM plane reads its inputs from: its hemisphere's inward row (W for the west
/// planes, E for the east).
fn in_row(pl: usize) -> RowId {
    if pl < 2 {
        RowId::W
    } else {
        RowId::E
    }
}

fn unit(pl: usize) -> UnitKind {
    if pl < 2 {
        UnitKind::MxmW
    } else {
        UnitKind::MxmE
    }
}

/// Feed stream `id` of the plane's input row so that a read at the MXM gap in cycle `t` finds `v`
/// (written one register upstream in cycle t - 1).
fn feed(m: &mut Machine, pl: usize, id: u8, v: &Vec320, t: u64) {
    let g = read_gap(m.geom(), unit(pl), 0, in_row(pl)).unwrap();
    assert!(g > 0 && t > 0);
    m.edit_state(|state| state.srf.write(in_row(pl), id, g - 1, t - 1, v)).unwrap();
}

fn row_ref(pl: usize, id: u32) -> u32 {
    id + if in_row(pl) == RowId::W { 32 } else { 0 }
}

/// Lanes 0, 1 = a, b; every other lane 0.
fn lanes2(a: u8, b: u8) -> Vec320 {
    let mut v = [0u8; MAX_LANES];
    v[0] = a;
    v[1] = b;
    v
}

/// `slice_test_install_mxm_rows`: a plane with `n` installed rows of type `t`, ready since cycle 0.
fn install(m: &mut Machine, pl: usize, t: WType, n: usize) {
    m.edit_state(|state| {
        let p = &mut state.mxm.planes[pl];
        p.wtype = t;
        p.rows = n;
        p.ready_iw = 0;
        let vals = Arc::make_mut(&mut p.weights);
        vals.resize(if t.is_pair() { 2 * n } else { n }, [0; MAX_LANES]);
    })
    .unwrap();
}

fn abc_entry(dispatch: u64, commit: u64, ptype: WType) -> AbcEntry {
    AbcEntry {
        dispatch,
        commit: commit.into(),
        ptype,
        q: 105,
        product: Arc::new([0; MAX_LANES]),
        wide: None,
        poison: 0,
    }
}

fn diag(d: Diag) -> Exec {
    Exec::Diagnostic(d)
}

// model: unit_tests/test_vxm_mxm.sail test_mxm_f16_abc_occupancy
#[test]
fn test_mxm_f16_abc_occupancy() {
    let mut m = machine("ref");
    install(&mut m, 0, WType::F16, 4);
    m.edit_state(|state| state.mxm.planes[0].abc = vec![abc_entry(99, 119, WType::F16)]).unwrap();
    let s = 0b100010; // W:2
    let occupancy = m.geom().mxm_occupancy_f16;
    // The next cycle is busy only for profiles with occupancy greater than one.
    let next = diag(if occupancy > 1 { Diag::Res001 } else { Diag::Tim001 });
    assert_eq!(exec(&mut m, 106, 100, &abc(F16, s)), next);
    // another queue of the same plane
    assert_eq!(exec(&mut m, 104, 100, &abc(F16, s)), next);
    // At the boundary the plane is free and reaches the missing-operand check.
    assert_eq!(exec(&mut m, 106, 99 + occupancy, &abc(F16, s)), diag(Diag::Tim001));
    // An i8 ABC does not hold the plane: a second one in the next cycle reaches its read.
    let mut m = machine("ref");
    install(&mut m, 0, WType::I8, 4);
    m.edit_state(|state| state.mxm.planes[0].abc = vec![abc_entry(99, 119, WType::I8)]).unwrap();
    assert_eq!(exec(&mut m, 106, 100, &abc(I8, s)), diag(Diag::Tim001));
}

// model: unit_tests/test_vxm_mxm.sail test_mxm_mixed_sign
#[test]
fn test_mxm_mixed_sign() {
    // Weights row 0 = [0xFF, 0x02], activations [0xFF, 0x03]:
    //   s x s: (-1)(-1) + 2*3 = 7      s x u: (-1)(255) + 6 = -249
    //   u x s: 255(-1) + 6 = -249      u x u: 255*255 + 6 = 65031
    for (wt, at, expect) in [
        (WType::I8, I8, 7u32),
        (WType::I8, U8, 0xFFFF_FF07),
        (WType::U8, I8, 0xFFFF_FF07),
        (WType::U8, U8, 0x0000_FE07),
    ] {
        let mut m = machine("ref");
        install(&mut m, 0, wt, 1);
        m.edit_state(|state| Arc::make_mut(&mut state.mxm.planes[0].weights)[0] = lanes2(0xFF, 0x02)).unwrap();
        let lat = m.latency_of_instr(&abc(at, 0));
        feed(&mut m, 0, 4, &lanes2(0xFF, 0x03), 50 + lat.d_skew as u64);
        assert_eq!(exec(&mut m, 106, 50, &abc(at, 0b100100)), Exec::Issued);
        m.unit_state_flush(50, &mut Rec::default());
        let e = &m.mxm().planes[0].abc[0];
        assert_eq!(e.product[0], expect, "{wt:?} x {at}");
        assert_eq!(e.ptype, if at == I8 { WType::I8 } else { WType::U8 });
    }
    // f16 weights with an i8 ABC (and the reverse) stay OTS-E-RES-001.
    let mut m = machine("ref");
    install(&mut m, 0, WType::F16, 1);
    assert_eq!(exec(&mut m, 106, 50, &abc(I8, 0b100100)), diag(Diag::Res001));
    let mut m = machine("ref");
    install(&mut m, 0, WType::I8, 1);
    assert_eq!(exec(&mut m, 106, 50, &abc(F16, 0b100100)), diag(Diag::Res001));
}

/// Lanes 0, 1, 2 = a, b, c; every other lane 0.
fn lanes3(a: u8, b: u8, c: u8) -> Vec320 {
    let mut v = lanes2(a, b);
    v[2] = c;
    v
}

fn abc8(fmt: u32, s: u32) -> Insn {
    Insn::ABC8 { fmt, src: s }
}

// model: unit_tests/test_vxm_mxm.sail test_mxm_fp8_abc
#[test]
fn test_mxm_fp8_abc() {
    // (weights, activations, format, product of row 0): 6.5 in e4m3 and in e5m2; lanes 0 and 1 cancel (max * max - max * max)
    // and lane 2 leaves the exact 2^-9 (e4m3) or 2^-16 (e5m2), which an f32 accumulation in lane order would have lost
    let cases = [
        (lanes2(0x38, 0x40), lanes2(0x30, 0x44), 0, 0x40D0_0000u32),
        (lanes2(0x3C, 0x40), lanes2(0x38, 0x42), 1, 0x40D0_0000),
        (lanes3(0x7E, 0x7E, 0x01), lanes3(0x7E, 0xFE, 0x38), 0, 0x3B00_0000),
        (lanes3(0x7B, 0x7B, 0x01), lanes3(0x7B, 0xFB, 0x3C), 1, 0x3780_0000),
    ];
    for (k, (w, x, fmt, want)) in cases.into_iter().enumerate() {
        let mut m = machine("ref");
        install(&mut m, 0, WType::Fp8, 1);
        m.edit_state(|state| Arc::make_mut(&mut state.mxm.planes[0].weights)[0] = w).unwrap();
        let lat = m.latency_of_instr(&abc8(fmt, 0));
        feed(&mut m, 0, 4, &x, 50 + lat.d_skew as u64);
        assert_eq!(exec(&mut m, 106, 50, &abc8(fmt, 0b100100)), Exec::Issued, "case {k}");
        m.unit_state_flush(50, &mut Rec::default());
        let e = &m.mxm().planes[0].abc[0];
        assert_eq!(e.ptype, WType::Fp8);
        assert_eq!(e.product[0], want, "case {k}");
        assert_eq!(e.product[1], 0, "a row that is not installed");
    }
    // fp8 weights need an fp8 ABC and the reverse
    let mut m = machine("ref");
    install(&mut m, 0, WType::Fp8, 1);
    assert_eq!(exec(&mut m, 106, 50, &abc(I8, 0b100100)), diag(Diag::Res001));
    assert_eq!(exec(&mut m, 106, 50, &abc(F16, 0b100100)), diag(Diag::Res001));
    let mut m = machine("ref");
    install(&mut m, 0, WType::I8, 1);
    assert_eq!(exec(&mut m, 106, 50, &abc8(0, 0b100100)), diag(Diag::Res001));
    // specials: e5m2 infinity times zero is NaN, an infinity gives an infinity, an e4m3 NaN is the canonical f32 NaN
    for (k, (w, x, fmt, want)) in [
        (lanes2(0x7C, 0x38), lanes2(0x00, 0x38), 1, 0x7FC0_0000u32),
        (lanes2(0x7C, 0x38), lanes2(0xBC, 0x38), 1, 0xFF80_0000),
        (lanes2(0x7F, 0x38), lanes2(0x38, 0x38), 0, 0x7FC0_0000),
    ]
    .into_iter()
    .enumerate()
    {
        let mut m = machine("ref");
        install(&mut m, 0, WType::Fp8, 1);
        m.edit_state(|state| Arc::make_mut(&mut state.mxm.planes[0].weights)[0] = w).unwrap();
        let lat = m.latency_of_instr(&abc8(fmt, 0));
        feed(&mut m, 0, 4, &x, 50 + lat.d_skew as u64);
        assert_eq!(exec(&mut m, 106, 50, &abc8(fmt, 0b100100)), Exec::Issued);
        m.unit_state_flush(50, &mut Rec::default());
        assert_eq!(m.mxm().planes[0].abc[0].product[0], want, "special {k}");
    }
    // ACC f32 follows an fp8 ABC; ACC i32 does not
    let mut m = machine("ref");
    m.edit_state(|state| state.mxm.planes[0].wtype = WType::Fp8).unwrap();
    m.edit_state(|state| state.mxm.planes[0].rows = 1).unwrap();
    m.edit_state(|state| state.mxm.planes[0].abc = vec![abc_entry(20, 40, WType::Fp8)]).unwrap();
    assert_eq!(exec(&mut m, 106, 60, &acc(A_I32, 0b001000, 0, 0)), diag(Diag::Opr001));
}

// model: unit_tests/test_vxm_mxm.sail test_mxm_fp8_pair_survives_next_abc
#[test]
fn an_fp8_pair_keeps_both_products_when_the_next_abc_is_accepted_in_its_commit_cycle() {
    let path = format!("{}/../../../opentsp-arch-spec/config/test/fp8_rate2.json", env!("CARGO_MANIFEST_DIR"));
    let profile = Profile::load_file("fp8_rate2", path).unwrap();
    let mut m = Machine::new(&profile, 0, Box::new(NoLinks)).unwrap();
    install(&mut m, 0, WType::Fp8, 1);
    let product = |v: u32| Arc::new([v; MAX_LANES]);
    let e1 =
        AbcEntry { dispatch: 30, commit: 50, ptype: WType::Fp8, q: 105, product: product(1), wide: None, poison: 0 };
    let e2 = AbcEntry { q: 106, product: product(2), ..e1.clone() };
    m.edit_state(|state| state.mxm.planes[0].abc = vec![e2, e1]).unwrap();
    let lat = m.latency_of_instr(&abc8(0, 0));
    feed(&mut m, 0, 4, &[0u8; MAX_LANES], 50 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, 104, 50, &abc8(0, 0b100100)), Exec::Issued);
    m.unit_state_flush(50, &mut Rec::default());
    let qs: Vec<usize> = m.mxm().planes[0].abc.iter().map(|e| e.q).collect();
    assert_eq!(qs, vec![104, 106, 105], "the new entry in front of both products of the pair");
    assert_eq!(exec(&mut m, 106, 51, &acc(A_F32, 0b001000, 0, 0)), Exec::Issued);
    assert_eq!(exec(&mut m, 105, 51, &acc(A_F32, 0b010000, 0, 1)), Exec::Issued);
    m.unit_state_flush(51, &mut Rec::default());
    let p = &m.mxm().planes[0];
    assert_eq!((p.acc[0][0], p.acc[1][0]), (1, 2), "even: lower queue, odd: higher queue");
}

/// Lanes `k < n` of a bf16 plane pair: `lo` and `hi` bytes.
fn bf_lane0(lo: u8, hi: u8) -> (Vec320, Vec320) {
    let (mut l, mut h) = ([0u8; MAX_LANES], [0u8; MAX_LANES]);
    l[0] = lo;
    h[0] = hi;
    (l, h)
}

// model: unit_tests/test_bf16.sail test_bf16_abc, test_bf16_plane_pair
#[test]
fn test_mxm_bf16_abc() {
    const W_BF16: u32 = 4;
    let mut m = machine("ref");
    install(&mut m, 0, WType::Bf16, 1);
    let (wl, wh) = bf_lane0(0xC0, 0x3F); // [1.5]
    m.edit_state(|state| {
        let w = Arc::make_mut(&mut state.mxm.planes[0].weights);
        w[0] = wl;
        w[1] = wh;
    })
    .unwrap();
    let (xl, xh) = bf_lane0(0x00, 0x40); // [2.0]
    let lat = m.latency_of_instr(&abc(W_BF16, 0));
    feed(&mut m, 0, 4, &xl, 50 + lat.d_skew as u64);
    feed(&mut m, 0, 5, &xh, 50 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, 106, 50, &abc(W_BF16, 0b100100)), Exec::Issued);
    m.unit_state_flush(50, &mut Rec::default());
    let e = &m.mxm().planes[0].abc[0];
    assert_eq!(e.ptype, WType::Bf16);
    assert_eq!((e.product[0], e.product[1]), (0x4040_0000, 0));
    // bf16 weights need a bf16 ABC and the reverse
    let mut m = machine("ref");
    install(&mut m, 0, WType::Bf16, 1);
    for t in [F16, I8] {
        assert_eq!(exec(&mut m, 106, 50, &abc(t, 0b100100)), diag(Diag::Res001));
    }
    assert_eq!(exec(&mut m, 106, 50, &abc8(0, 0b100100)), diag(Diag::Res001));
    let mut m = machine("ref");
    install(&mut m, 0, WType::F16, 1);
    assert_eq!(exec(&mut m, 106, 50, &abc(W_BF16, 0b100100)), diag(Diag::Res001));
    // ACC f32 follows a bf16 ABC; ACC i32 does not
    let mut m = machine("ref");
    m.edit_state(|state| state.mxm.planes[0].wtype = WType::Bf16).unwrap();
    m.edit_state(|state| state.mxm.planes[0].rows = 1).unwrap();
    m.edit_state(|state| state.mxm.planes[0].abc = vec![abc_entry(20, 40, WType::Bf16)]).unwrap();
    assert_eq!(exec(&mut m, 106, 60, &acc(A_I32, 0b001000, 0, 0)), diag(Diag::Opr001));
    // the plane pair: a bf16 weight plane holds its odd partner, and bf16 loads do not run on an odd plane
    let mut m = machine("ref");
    assert_eq!(exec(&mut m, 107, 10, &iw(W_BF16)), diag(Diag::Res001));
    m.edit_state(|state| state.mxm.planes[0].wtype = WType::Bf16).unwrap();
    m.edit_state(|state| state.mxm.planes[0].rows = 16).unwrap();
    m.edit_state(|state| state.mxm.planes[1].buf_rows = 16).unwrap();
    assert_eq!(exec(&mut m, 107, 10, &iw(I8)), diag(Diag::Res001));
}

// model: unit_tests/test_slices.sail test_mxm_stage_readiness
#[test]
fn test_mxm_stage_readiness() {
    // Each MXM stage needs the previous stage COMMITTED (dispatch + d_func); changes are staged and
    // take effect at the end of the cycle.
    let mut m = machine("ref");
    m.edit_state(|state| state.mxm.planes[0].buf_rows = 16).unwrap();
    m.edit_state(|state| state.mxm.planes[0].ready_lw = 120).unwrap(); // the last LW commits at cycle 120
    assert_eq!(exec(&mut m, 105, 119, &iw(I8)), diag(Diag::Res001)); // west plane 0, IW one cycle early
    assert_eq!(exec(&mut m, 105, 120, &iw(I8)), Exec::Issued);
    assert_eq!(m.mxm().planes[0].rows, 0); // staged, not yet applied
    m.unit_state_flush(120, &mut Rec::default());
    assert_eq!((m.mxm().planes[0].rows, m.mxm().planes[0].buf_rows), (16, 0));
    let d_iw = m.latency_of_instr(&iw(I8)).d_func as u64;
    assert_eq!(m.mxm().planes[0].ready_iw, (120 + d_iw).into());
    assert_eq!(exec(&mut m, 106, 119 + d_iw, &abc(I8, 0b111000)), diag(Diag::Res001));
    // ACC before any ABC has committed (D5).
    m.edit_state(|state| state.mxm.planes[0].abc = vec![abc_entry(280, 300, WType::I8)]).unwrap();
    assert_eq!(exec(&mut m, 106, 299, &acc(A_I32, 0b001000, 0, 0)), diag(Diag::Res001));
}

// model: unit_tests/test_slices.sail test_mxm_stage_conflict
#[test]
fn test_mxm_stage_conflict() {
    // two queues staging the same plane part in one cycle are RES-003; neither applies.
    let mut m = machine("ref");
    m.edit_state(|state| state.mxm.planes[0].buf_rows = 16).unwrap();
    assert_eq!(exec(&mut m, 104, 10, &iw(I8)), Exec::Issued);
    assert_eq!(exec(&mut m, 105, 10, &iw(I8)), Exec::Issued);
    m.unit_state_flush(10, &mut Rec::default());
    assert_eq!((m.mxm().planes[0].rows, m.mxm().planes[0].buf_rows), (0, 16));
}

// model: unit_tests/test_slices.sail test_mxm_f16_plane_pair
#[test]
fn test_mxm_f16_plane_pair() {
    // f16 weights use the even plane of a pair; the odd partner is busy meanwhile.
    let mut m = machine("ref");
    assert_eq!(exec(&mut m, 107, 10, &iw(F16)), diag(Diag::Res001)); // west plane 1
    m.edit_state(|state| state.mxm.planes[0].wtype = WType::F16).unwrap();
    m.edit_state(|state| state.mxm.planes[0].rows = 16).unwrap();
    m.edit_state(|state| state.mxm.planes[1].buf_rows = 16).unwrap();
    assert_eq!(exec(&mut m, 107, 10, &iw(I8)), diag(Diag::Res001)); // plane 0 holds f16
    assert_eq!(exec(&mut m, 109, 10, &acc(A_I32, 0b001000, 0, 0)), diag(Diag::Res001));
}

// model: unit_tests/test_slices.sail test_mxm_operand_checks
#[test]
fn test_mxm_operand_checks() {
    // IW with no buffered rows, and ACC with a mismatched result type, are rejected.
    let mut m = machine("ref");
    assert_eq!(exec(&mut m, 104, 10, &iw(I8)), diag(Diag::Res001));
    m.edit_state(|state| state.mxm.planes[0].abc = vec![abc_entry(0, 5, WType::I8)]).unwrap();
    assert_eq!(exec(&mut m, 106, 10, &acc(A_F32, 0b001000, 0, 0)), diag(Diag::Opr001));
    // ACC destinations are 4 aligned streams (destinations are checked too).
    assert_eq!(exec(&mut m, 106, 10, &acc(A_I32, 0b000010, 0, 0)), diag(Diag::Opr002));
}

/// 16 stream values (a group of 16), value k = `base + k` in every lane.
fn group(base: u8) -> [Vec320; 16] {
    std::array::from_fn(|k| [base.wrapping_add(k as u8); MAX_LANES])
}

/// Feed the 16 streams of `g` (0 or 16, plus the row bit) for a read in cycle `t`.
fn feed16(m: &mut Machine, pl: usize, g: u32, vals: &[Vec320; 16], t: u64) {
    for (k, v) in vals.iter().enumerate() {
        feed(m, pl, (g + k as u32) as u8, v, t);
    }
}

#[test]
fn lw_fills_the_buffer_and_overflow_is_res_002() {
    let mut m = machine("mini"); // mxm.rows = 16: the buffer holds one group
    let lat = m.latency_of_instr(&lw(I8, 0));
    let g = 2; // W row: stream ref 32 + 0 → field (32 + 0) / 16 = 2
    feed16(&mut m, 0, 0, &group(1), 10 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, q_lw(0), 10, &lw(I8, g)), Exec::Issued);
    m.unit_state_flush(10, &mut Rec::default());
    let p = &m.mxm().planes[0];
    assert_eq!((p.buf_rows, p.btype, p.ready_lw), (16, WType::I8, 10 + u128::from(lat.d_func)));
    assert_eq!(p.buf[3][0], 4);
    // a second LW would pass mxm.rows
    feed16(&mut m, 0, 0, &group(1), 20 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, q_lw(0), 20, &lw(I8, g)), diag(Diag::Res002));
    // a different type with rows buffered is RES-001, before the capacity check
    feed16(&mut m, 0, 0, &group(1), 21 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, q_lw(0), 21, &lw(U8, g)), diag(Diag::Res001));
    // IW before the LW committed: RES-001; after: the matrix is installed and the buffer empties
    assert_eq!(exec(&mut m, q_lw(0), 10, &iw(I8)), diag(Diag::Res001));
    let ready = 10 + u64::from(lat.d_func);
    assert_eq!(exec(&mut m, q_lw(0), ready, &iw(U8)), diag(Diag::Res001)); // wrong type
    assert_eq!(exec(&mut m, q_lw(0), ready, &iw(I8)), Exec::Issued);
    m.unit_state_flush(ready, &mut Rec::default());
    let p = &m.mxm().planes[0];
    assert_eq!((p.buf_rows, p.rows, p.wtype, p.iw_open), (0, 16, WType::I8, false));
    assert_eq!(p.weights[5][7], 6);
}

#[test]
fn an_f16_load_needs_an_even_plane_and_holds_its_partner() {
    let mut m = machine("ref");
    // f16 on plane 1 or 3 is RES-001, before the operand is read
    assert_eq!(exec(&mut m, q_lw(1), 10, &lw(F16, 2)), diag(Diag::Res001));
    assert_eq!(exec(&mut m, q_abc(3), 10, &abc(F16, 0)), diag(Diag::Res001));
    // while plane 0 buffers f16 rows, plane 1 is busy for every type and every instruction
    m.edit_state(|state| {
        let p = &mut state.mxm.planes[0];
        p.btype = WType::F16;
        p.buf_rows = 16;
    })
    .unwrap();
    assert_eq!(exec(&mut m, q_lw(1), 10, &lw(I8, 2)), diag(Diag::Res001));
    assert_eq!(exec(&mut m, q_acc(1), 10, &acc(A_I32, 0, 0, 0)), diag(Diag::Res001));
    // plane 3 is not affected by plane 0
    assert_eq!(exec(&mut m, q_lw(3), 10, &lw(I8, 0)), diag(Diag::Tim001));
    // installed f16 weights hold the partner too
    let mut m = machine("ref");
    install(&mut m, 2, WType::F16, 2);
    assert_eq!(exec(&mut m, q_lw(3), 10, &lw(I8, 0)), diag(Diag::Res001));
}

#[test]
fn abc_computes_every_installed_row_and_acc_writes_four_planes() {
    let mut m = machine("ref");
    install(&mut m, 0, WType::I8, 3);
    m.edit_state(|state| {
        let w = Arc::make_mut(&mut state.mxm.planes[0].weights);
        w[0] = lanes2(1, 2);
        w[1] = lanes2(3, 4);
        w[2] = lanes2(0xff, 0x80); // -1, -128
    })
    .unwrap();
    let lat = m.latency_of_instr(&abc(I8, 0));
    feed(&mut m, 0, 4, &lanes2(5, 6), 50 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, q_abc(0), 50, &abc(I8, row_ref(0, 4))), Exec::Issued);
    m.unit_state_flush(50, &mut Rec::default());
    let e = m.mxm().planes[0].abc[0].clone();
    assert_eq!((e.dispatch, e.commit), (50, 50 + u128::from(lat.d_func)));
    assert_eq!(&e.product[..4], &[17, 39, (-5i32 - 768) as u32, 0]); // lane r >= rows is zero
                                                                     // ACC before the ABC committed: nothing to use
    let la = m.latency_of_instr(&acc(A_I32, 0, 0, 0));
    assert_eq!(exec(&mut m, q_acc(0), 50, &acc(A_I32, 0, 0, 0)), diag(Diag::Res001));
    // a type mismatch is OPR-001
    let at = e.commit;
    assert_eq!(exec(&mut m, q_acc(0), at.try_into().unwrap(), &acc(A_F32, 8, 0, 0)), diag(Diag::Opr001));
    // replace into accumulator 1, then accumulate: acc[1] += product (i32 wrap)
    let dst = 8; // E:8..11, an ACC result row of the west planes
    assert_eq!(exec(&mut m, q_acc(0), at.try_into().unwrap(), &acc(A_I32, dst, 0, 1)), Exec::Issued);
    m.unit_state_flush(at.try_into().unwrap(), &mut Rec::default());
    assert_eq!(&m.mxm().planes[0].acc[1][..3], &[17, 39, (-773i32) as u32]);
    assert_eq!(m.mxm().planes[0].acc[0][0], 0);
    assert_eq!(exec(&mut m, q_acc(0), (at + 1).try_into().unwrap(), &acc(A_I32, dst, 1, 1)), Exec::Issued);
    m.unit_state_flush((at + 1).try_into().unwrap(), &mut Rec::default());
    assert_eq!(&m.mxm().planes[0].acc[1][..3], &[34, 78, (-1546i32) as u32]);
    // four byte planes were scheduled for commit at at + 1 + d_func - ... : check the stream file
    let o = tick_to(&mut m, 0, (at + 1 + u128::from(la.d_func) + 1).try_into().unwrap());
    assert!(o.diags.is_empty(), "{:?}", o.diags);
    let g = tsp_iss::layout::write_gap(m.geom(), UnitKind::MxmW, 0, RowId::E).unwrap();
    let (k, v) = m.srf().read(RowId::E, dst as u8, g + 1, i128::try_from(at + u128::from(la.d_func) + 1).unwrap());
    assert_eq!(k, SrfLookup::Found);
    assert_eq!(v[0], 17);
    // accumulating into f32 needs f16 weights: ACC.f32 on an i8 product is OPR-001 (above); an
    // unwritable destination (misaligned) changes nothing
    assert_eq!(exec(&mut m, q_acc(0), (at + 2).try_into().unwrap(), &acc(A_I32, dst + 1, 0, 0)), diag(Diag::Opr002));
    m.unit_state_flush((at + 2).try_into().unwrap(), &mut Rec::default());
    assert_eq!(m.mxm().planes[0].acc[0][0], 0);
}

#[test]
fn two_queues_changing_one_part_clash_and_neither_has_an_effect() {
    // LW on queue 104 and the second LW of the same plane from queue 105 (ABC queue can issue LW
    // too: every MXM queue takes every MXM instruction) in one cycle: RES-003 for both.
    let mut m = machine("ref");
    let lat = m.latency_of_instr(&lw(I8, 0));
    feed16(&mut m, 0, 0, &group(1), 10 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, 104, 10, &lw(I8, 2)), Exec::Issued);
    assert_eq!(exec(&mut m, 105, 10, &lw(I8, 2)), Exec::Issued);
    let mut o = Rec::default();
    m.unit_state_flush(10, &mut o);
    assert_eq!(o.diags, vec![(10, 104, Diag::Res003), (10, 105, Diag::Res003)]);
    assert_eq!(m.mxm().planes[0].buf_rows, 0);
    assert!(m.mxm().staged.is_empty());
    // different parts of one plane do not clash: an LW (buffer) and an IWS (weights)
    let mut m = machine("ref");
    feed16(&mut m, 0, 0, &group(1), 10 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, 104, 10, &lw(I8, 2)), Exec::Issued);
    assert_eq!(exec(&mut m, 105, 10, &iws(I8, 2)), Exec::Issued);
    let mut o = Rec::default();
    m.unit_state_flush(10, &mut o);
    assert!(o.diags.is_empty());
    let p = &m.mxm().planes[0];
    assert_eq!((p.buf_rows, p.rows, p.iw_open), (16, 16, true));
    // the same queue changing one part twice does not clash
    let mut m = machine("ref");
    feed16(&mut m, 0, 0, &group(1), 10 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, 104, 10, &lw(I8, 2)), Exec::Issued);
    assert_eq!(exec(&mut m, 104, 10, &lw(I8, 2)), Exec::Issued);
    let mut o = Rec::default();
    m.unit_state_flush(10, &mut o);
    assert!(o.diags.is_empty());
    // two ACCs into different accumulators do not clash; into one they do, and their stream
    // writes are withdrawn
    let mut m = machine("ref");
    install(&mut m, 0, WType::I8, 1);
    m.edit_state(|state| state.mxm.planes[0].abc = vec![abc_entry(1, 2, WType::I8)]).unwrap();
    let la = m.latency_of_instr(&acc(A_I32, 0, 0, 0));
    let c = 50;
    assert_eq!(exec(&mut m, 106, c, &acc(A_I32, 0, 0, 0)), Exec::Issued);
    assert_eq!(exec(&mut m, 105, c, &acc(A_I32, 4, 0, 1)), Exec::Issued); // the other accumulator
    let mut o = Rec::default();
    m.unit_state_flush(c, &mut o);
    assert!(o.diags.is_empty());
    let mut m = machine("ref");
    install(&mut m, 0, WType::I8, 1);
    m.edit_state(|state| state.mxm.planes[0].abc = vec![abc_entry(1, 2, WType::I8)]).unwrap();
    assert_eq!(exec(&mut m, 106, c, &acc(A_I32, 0, 0, 1)), Exec::Issued);
    assert_eq!(exec(&mut m, 105, c, &acc(A_I32, 4, 0, 1)), Exec::Issued);
    let slot = ((c + la.d_func as u64 - la.d_skew as u64) % tsp_iss::driver::PENDING_SLOTS as u64) as usize;
    let before = m.driver().pending[slot].len();
    assert_eq!(before, 8);
    let mut o = Rec::default();
    m.unit_state_flush(c, &mut o);
    assert_eq!(o.diags, vec![(c, 105, Diag::Res003), (c, 106, Diag::Res003)]);
    assert_eq!(m.driver().pending[slot].len(), 0);
}

#[test]
fn iws_appends_rows_until_an_abc_closes_the_matrix() {
    let mut m = machine("mini");
    let lat = m.latency_of_instr(&iws(I8, 0));
    feed16(&mut m, 0, 0, &group(1), 10 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, q_lw(0), 10, &iws(I8, 2)), Exec::Issued);
    m.unit_state_flush(10, &mut Rec::default());
    assert_eq!((m.mxm().planes[0].rows, m.mxm().planes[0].iw_open), (16, true));
    // a second install into the open matrix would pass mxm.rows (16)
    feed16(&mut m, 0, 0, &group(1), 11 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, q_lw(0), 11, &iws(I8, 2)), diag(Diag::Res002));
    // a different type into the open matrix: RES-001
    assert_eq!(exec(&mut m, q_lw(0), 11, &iws(U8, 2)), diag(Diag::Res001));
}

#[test]
fn an_abc_after_the_last_install_starts_a_new_matrix() {
    let mut m = machine("mini");
    let lat = m.latency_of_instr(&iws(I8, 0));
    feed16(&mut m, 0, 0, &group(1), 10 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, q_lw(0), 10, &iws(I8, 2)), Exec::Issued);
    m.unit_state_flush(10, &mut Rec::default());
    // an ABC dispatched at cycle 12 closes the matrix: the next IWS starts again at row 0
    m.edit_state(|state| state.mxm.planes[0].abc = vec![abc_entry(12, 12 + lat.d_func as u64, WType::I8)]).unwrap();
    feed16(&mut m, 0, 0, &group(40), 13 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, q_lw(0), 13, &iws(I8, 2)), Exec::Issued);
    m.unit_state_flush(13, &mut Rec::default());
    let p = &m.mxm().planes[0];
    assert_eq!((p.rows, p.iw_open, p.iws_cycle, p.weights[0][0]), (16, true, 13, 40));
}

#[test]
fn an_f16_matrix_is_row_pairs_and_the_dot_product_rounds_once() {
    let mut m = machine("ref");
    let lat = m.latency_of_instr(&iws(F16, 0));
    // row 0: weights [1.0, 2.0] (0x3c00, 0x4000): stream 0 = low bytes, stream 1 = high bytes
    let mut g = group(0);
    g[0] = [0; MAX_LANES];
    g[1] = [0; MAX_LANES];
    g[1][0] = 0x3c;
    g[1][1] = 0x40;
    feed16(&mut m, 0, 0, &g, 10 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, q_lw(0), 10, &iws(F16, 2)), Exec::Issued);
    m.unit_state_flush(10, &mut Rec::default());
    let p = &m.mxm().planes[0];
    assert_eq!((p.rows, p.wtype), (8, WType::F16)); // 16 stream values = 8 rows
                                                    // x = [3.0 (0x4200), 0.5 (0x3800)]: 3 + 1 = 4.0
    let ready = 10 + u64::from(lat.d_func);
    let al = m.latency_of_instr(&abc(F16, 0));
    let (xlo, mut xhi) = ([0u8; MAX_LANES], [0u8; MAX_LANES]);
    (xhi[0], xhi[1]) = (0x42, 0x38);
    feed(&mut m, 0, 4, &xlo, ready + al.d_skew as u64);
    feed(&mut m, 0, 5, &xhi, ready + al.d_skew as u64);
    assert_eq!(exec(&mut m, q_abc(0), ready, &abc(F16, row_ref(0, 4))), Exec::Issued);
    m.unit_state_flush(ready, &mut Rec::default());
    let e = &m.mxm().planes[0].abc[0];
    assert_eq!(e.product[0], 0x4080_0000);
    // the product has mxm.rows lanes, 0 beyond the installed rows
    assert_eq!(e.product[8], 0);
    // The shipped profile accepts another initiation next cycle; no new operand was supplied.
    assert_eq!(exec(&mut m, q_abc(0), ready + 1, &abc(F16, row_ref(0, 4))), diag(Diag::Tim001));
    // ACC.f32: replace, then accumulate with one f32 add (4.0 + 4.0 = 8.0); an i32 ACC is OPR-001
    let at = ready + al.d_func as u64;
    assert_eq!(exec(&mut m, q_acc(0), at, &acc(A_I32, 8, 0, 0)), diag(Diag::Opr001));
    assert_eq!(exec(&mut m, q_acc(0), at, &acc(A_F32, 8, 0, 0)), Exec::Issued);
    m.unit_state_flush(at, &mut Rec::default());
    assert_eq!(m.mxm().planes[0].acc[0][0], 0x4080_0000);
    assert_eq!(exec(&mut m, q_acc(0), at + 1, &acc(A_F32, 8, 1, 0)), Exec::Issued);
    m.unit_state_flush(at + 1, &mut Rec::default());
    assert_eq!(m.mxm().planes[0].acc[0][0], 0x4100_0000);
    assert_eq!(m.mxm().planes[0].acc[1][0], 0, "the other accumulator is untouched");
}

#[test]
fn the_inf_nan_flags_of_an_f16_matrix_follow_the_weights_they_were_built_from() {
    // an ABC scans the weight rows once; a matrix that changes afterwards (here a row turned into Inf behind the plane's
    // back, as `Arc::make_mut` on the field does, and by a new IW below) must not be judged by the old flags
    let mut m = machine("ref");
    install(&mut m, 0, WType::F16, 4);
    m.edit_state(|state| {
        let p = &mut state.mxm.planes[0];
        let w = Arc::make_mut(&mut p.weights);
        (w[0][0], w[1][0]) = (0x00, 0x3c); // row 0: 1.0 in lane 0
        (w[2][0], w[3][0]) = (0x00, 0x40); // row 1: 2.0 in lane 0
    })
    .unwrap();
    let al = m.latency_of_instr(&abc(F16, 0));
    let (xlo, mut xhi) = ([0u8; MAX_LANES], [0u8; MAX_LANES]);
    xhi[0] = 0x3c; // x = 1.0 in lane 0
    let run = |m: &mut Machine, c: u64| -> (u32, u32) {
        feed(m, 0, 4, &xlo, c + al.d_skew as u64);
        feed(m, 0, 5, &xhi, c + al.d_skew as u64);
        assert_eq!(exec(m, q_abc(0), c, &abc(F16, row_ref(0, 4))), Exec::Issued);
        m.unit_state_flush(c, &mut Rec::default());
        let e = &m.mxm().planes[0].abc[0];
        (e.product[0], e.product[1])
    };
    assert_eq!(run(&mut m, 10), (0x3f80_0000, 0x4000_0000));
    assert!(m.mxm().planes[0].row_exponents.is_some(), "the first f16 ABC scans the matrix");
    // row 0 becomes +Inf (0x7c00) in place
    m.edit_state(|state| {
        let w = Arc::make_mut(&mut state.mxm.planes[0].weights);
        w[1][0] = 0x7c;
    })
    .unwrap();
    assert_eq!(run(&mut m, 20), (0x7f80_0000, 0x4000_0000));
    // and a NaN in row 1 (0x7e00): the product is the canonical NaN
    m.edit_state(|state| {
        let w = Arc::make_mut(&mut state.mxm.planes[0].weights);
        w[3][0] = 0x7e;
    })
    .unwrap();
    assert_eq!(run(&mut m, 30), (0x7f80_0000, 0x7fc0_0000));
    // the matrix back to finite: the flags follow
    m.edit_state(|state| {
        let w = Arc::make_mut(&mut state.mxm.planes[0].weights);
        (w[1][0], w[3][0]) = (0x3c, 0x40);
    })
    .unwrap();
    assert_eq!(run(&mut m, 40), (0x3f80_0000, 0x4000_0000));
}

#[test]
fn a_stream_append_keeps_the_original_prefix_when_the_same_queue_installs_twice() {
    let mut m = machine("ref");
    let append = iws(I8, 2);
    let lat = m.latency_of_instr(&append);
    feed16(&mut m, 0, 0, &group(1), 10 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, 104, 10, &append), Exec::Issued);
    m.unit_state_flush(10, &mut Rec::default());
    m.edit_state(|state| {
        let p = &mut state.mxm.planes[0];
        p.buf = Arc::new(group(80).to_vec());
        p.buf_rows = 16;
        p.btype = WType::I8;
    })
    .unwrap();

    feed16(&mut m, 0, 0, &group(40), 11 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, 104, 11, &append), Exec::Issued);
    assert_eq!(exec(&mut m, 104, 11, &iw(I8)), Exec::Issued);
    let mut observed = Rec::default();
    m.unit_state_flush(11, &mut observed);
    let p = &m.mxm().planes[0];
    assert_eq!((p.rows, p.weights[0][0], p.weights[16][0]), (32, 1, 40));
    assert_eq!(p.buf_rows, 0);
    assert!(observed.diags.is_empty());
}
