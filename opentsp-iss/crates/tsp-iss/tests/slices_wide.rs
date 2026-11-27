//! The wide accumulator and the block-scaled type on the ISS (extensions wide_acc and block_fp8; ports of
//! `unit_tests/test_ext_types.sail`: `test_acc_x128`, `test_abc_keeps_exact_sum`, `test_ext_decode_gate`).
use ots_isa::Insn;
use ots_profile::Profile;
use std::sync::Arc;
use tsp_iss::geom::MAX_LANES;
use tsp_iss::links::NoLinks;
use tsp_iss::observe::Observer;
use tsp_iss::slices::mxm::*;
use tsp_iss::slices::{Exec, Issue};
use tsp_iss::{Diag, Machine};
use tsp_numerics::mxm::Wide;

const A_X128: u32 = 0;
const A_F32: u32 = 3;

fn machine() -> Machine {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../opentsp-arch-spec/config/test/mini_ext.json");
    Machine::new(&Profile::load_file("mini_ext", path).unwrap(), 0, Box::new(NoLinks)).unwrap()
}

#[derive(Default)]
struct Rec;
impl Observer for Rec {}

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
    m.execute(i, &ctx, &mut Rec)
}

fn acc(t: u32, dst: u32, mode: u32, a: u32) -> Insn {
    Insn::ACC { r#type: t, dst, mode, acc: a }
}

fn entry(ptype: WType, product0: u32, wide0: Wide) -> AbcEntry {
    let mut product = [0u32; MAX_LANES];
    product[0] = product0;
    let mut wide = vec![Wide::default(); MAX_LANES];
    wide[0] = wide0;
    AbcEntry {
        dispatch: 20,
        commit: 40,
        ptype,
        q: 105,
        product: Arc::new(product),
        wide: Some(Arc::new(wide)),
        poison: 0,
    }
}

/// Three accumulations of 2^-25 on 1.0 give 1 + 2^-23 in x128 and 1.0 in f32.
// model: unit_tests/test_ext_types.sail test_acc_x128
#[test]
fn acc_x128_keeps_the_exact_sum() {
    let mut m = machine();
    m.edit_state(|state| {
        let p = &mut state.mxm.planes[0];
        p.wtype = WType::Fp8;
        p.rows = 1;
        p.abc = vec![entry(WType::Fp8, 0x3300_0000, Wide { v: 1 << 23, sp: 0, nz: false })];
        let mut w = vec![Wide::default(); MAX_LANES];
        w[0] = Wide { v: 1 << 48, sp: 0, nz: false };
        p.acc128[0] = Arc::new(w);
        let mut f = [0u32; MAX_LANES];
        f[0] = 0x3F80_0000;
        p.acc[0] = Arc::new(f);
    })
    .unwrap();
    for i in 0..3u64 {
        assert_eq!(exec(&mut m, 106, 60 + 2 * i, &acc(A_X128, 8, 1, 0)), Exec::Issued);
        m.unit_state_flush(60 + 2 * i, &mut Rec);
        assert_eq!(exec(&mut m, 106, 61 + 2 * i, &acc(A_F32, 16, 1, 0)), Exec::Issued);
        m.unit_state_flush(61 + 2 * i, &mut Rec);
    }
    let p = &m.mxm().planes[0];
    assert_eq!(p.acc128[0][0].v, (1 << 48) + 3 * (1 << 23));
    assert_eq!(p.acc128[0][0].read(), 0x3F80_0001);
    assert_eq!(p.acc[0][0], 0x3F80_0000);
    // replace overwrites the other accumulator and leaves the first alone
    assert_eq!(exec(&mut m, 106, 70, &acc(A_X128, 8, 0, 1)), Exec::Issued);
    m.unit_state_flush(70, &mut Rec);
    let p = &m.mxm().planes[0];
    assert_eq!(p.acc128[1][0].v, 1 << 23);
    assert_eq!(p.acc128[0][0].v, (1 << 48) + 3 * (1 << 23));
}

/// An integer product has no wide form; a bfp8 product has.
#[test]
fn acc_x128_needs_a_floating_product() {
    let mut m = machine();
    m.edit_state(|state| {
        let p = &mut state.mxm.planes[0];
        p.wtype = WType::I8;
        p.rows = 1;
        p.abc = vec![entry(WType::I8, 0, Wide::default())];
    })
    .unwrap();
    assert_eq!(exec(&mut m, 106, 60, &acc(A_X128, 8, 0, 0)), Exec::Diagnostic(Diag::Opr001));
    let mut m = machine();
    m.edit_state(|state| {
        let p = &mut state.mxm.planes[0];
        p.wtype = WType::Bfp8;
        p.rows = 1;
        p.abc = vec![entry(WType::Bfp8, 0x3F80_0000, Wide { v: 1 << 48, sp: 0, nz: false })];
    })
    .unwrap();
    assert_eq!(exec(&mut m, 106, 60, &acc(A_X128, 8, 0, 0)), Exec::Issued);
}

/// A pair type holds the plane pair: bfp8 weights block the odd plane, like f16.
#[test]
fn bfp8_takes_the_plane_pair() {
    let mut m = machine();
    m.edit_state(|state| {
        let p = &mut state.mxm.planes[1];
        p.wtype = WType::Bfp8;
        p.rows = 1;
    })
    .unwrap();
    assert_eq!(exec(&mut m, 107, 50, &Insn::ABC { r#type: 5, src: 0b100100 }), Exec::Diagnostic(Diag::Res001));
}
