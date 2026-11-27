//! Tests of `slices/vxm`, moved from the module's in-source test block (they use the crate's public API).
//! Ports of the VXM unit tests of the Sail model (`unit_tests/test_vxm_mxm.sail`:
//! `test_vxm_alu_chain_gelu`, `test_vxm_alu_chain_timing`, `test_vxm_unchained_read_ignores_station`,
//! `test_vxm_chain_encoding`, `test_vxm_sg4`, `test_skew_reads_after_dispatch`,
//! `test_skew_execute_stage`, `test_timing_entry_table`), and the
//! ISS's own tests of each operation family, Reduce, Convert and the Mask/Select/Cmp semantics.
use ots_isa::Insn;
use tsp_iss::observe::Observer;
use tsp_iss::operands::Planes;
use tsp_iss::slices::{Exec, Issue};
use tsp_iss::Machine;

use ots_isa::{encode, Code as Diag, EntryId, Isa};
use ots_profile::Profile;
use tsp_iss::layout::{read_gap, RowId, UnitKind};
use tsp_iss::links::NoLinks;
use tsp_iss::operands::{one_plane, words_to_planes};
use tsp_iss::srf::SrfLookup;

/// The vector length of the `ref` profile, the one the tests build their operands for.
const VL: usize = 320;

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

/// The context of an instruction issued on queue `q` at cycle `c` (`slice_test_set_issue_context`).
fn issue(m: &Machine, q: usize, c: u64, i: &Insn) -> Issue {
    Issue { cycle: c, queue: q, element: 0, addr_offset: 0, stream_offset: 0, thread: 0, lat: m.latency_of_instr(i) }
}

/// `execute_test`: issue `i` on queue `q` at cycle `c`.
fn exec(m: &mut Machine, q: usize, c: u64, i: &Insn) -> Exec {
    let ctx = issue(m, q, c, i);
    m.execute(i, &ctx, &mut Rec::default())
}

fn vl(m: &Machine) -> usize {
    m.geom().vlanes as usize
}

fn vxm_gap(m: &Machine) -> u8 {
    read_gap(m.geom(), UnitKind::Vxm, 0, RowId::E).unwrap()
}

/// `slice_test_write_vxm_operand`: the operand on E:s..s+3 reaches the VXM in cycle `rc` (a writer one register upstream
/// wrote it in cycle rc - 1).
fn vfeed(m: &mut Machine, s: u8, p: &Planes, rc: u64) {
    let g = vxm_gap(m);
    if g > 0 && rc > 0 {
        for (b, plane) in p.iter().enumerate() {
            m.edit_state(|state| state.srf.write(RowId::E, s + b as u8, g - 1, rc - 1, plane)).unwrap();
        }
    }
}

/// `slice_test_first_four_f32`: four f32 lanes, the others zero.
fn f32x4(a: u32, b: u32, c: u32, d: u32) -> Planes {
    let mut xs = [0u32; tsp_iss::geom::MAX_LANES];
    xs[..4].copy_from_slice(&[a, b, c, d]);
    words_to_planes(&xs, VL)
}

/// `slice_test_broadcast_f32`.
fn bcast(k: u32) -> Planes {
    words_to_planes(&[k; tsp_iss::geom::MAX_LANES], VL)
}

fn tick_to(m: &mut Machine, a: u64, b: u64) {
    let mut o = Rec::default();
    for c in a..=b {
        m.tick(c, &mut o);
    }
}

/// `slice_test_read_f32_lane`: f32 lane `l` (0..3) of streams s..s+3 on row E just past the VXM in cycle `t`.
fn lane(m: &Machine, s: u8, l: usize, t: u64) -> u32 {
    let g = vxm_gap(m);
    assert!(g < 43);
    let mut x = 0u32;
    for b in 0..4usize {
        let (k, p) = m.srf().read(RowId::E, s + b as u8, g + 1, t as i64);
        assert_eq!(k, SrfLookup::Found, "chain result missing");
        x |= (p[l] as u32) << (8 * b);
    }
    x
}

/// The 8-bit lane 0 of stream `s` just past the VXM in cycle `t`.
fn lane8(m: &Machine, s: u8, t: u64) -> Option<u8> {
    let g = vxm_gap(m);
    let (k, p) = m.srf().read(RowId::E, s, g + 1, t as i64);
    (k == SrfLookup::Found).then_some(p[0])
}

fn vs(s: u32, c: u32) -> (u32, u32) {
    (s, c)
}

fn un(id: EntryId, t: u32, d: u32, a: (u32, u32), g: u32) -> Insn {
    let f = [t, d, a.0, g, a.1, 0, 0, 0, 0, 0];
    let mut fields = [0u32; ots_isa::MAX_VARS];
    fields[..f.len().min(ots_isa::MAX_VARS)].copy_from_slice(&f[..f.len().min(ots_isa::MAX_VARS)]);
    Insn::from_fields(id, &fields)
}

fn bin(id: EntryId, t: u32, d: u32, a: (u32, u32), b: (u32, u32), g: u32) -> Insn {
    let mut f = [0u32; ots_isa::MAX_VARS];
    f[..7].copy_from_slice(&[t, d, a.0, b.0, g, a.1, b.1]);
    Insn::from_fields(id, &f)
}

fn mul(t: u32, d: u32, a: (u32, u32), b: (u32, u32)) -> Insn {
    Insn::Mul { r#type: t, dst: d, lhs: a.0, rhs: b.0, sg4: 0, lhs_chain: a.1, rhs_chain: b.1 }
}
fn add(t: u32, d: u32, a: (u32, u32), b: (u32, u32)) -> Insn {
    Insn::Add { r#type: t, dst: d, lhs: a.0, rhs: b.0, sg4: 0, lhs_chain: a.1, rhs_chain: b.1 }
}
fn neg(t: u32, d: u32, a: (u32, u32)) -> Insn {
    Insn::Neg { r#type: t, dst: d, lhs: a.0, sg4: 0, lhs_chain: a.1 }
}

const F32: u32 = 3;
const I8: u32 = 0;
const F16: u32 = 1;
const I16: u32 = 5;
const I32: u32 = 2;
const U8: u32 = 4;

// model: unit_tests/test_vxm_mxm.sail test_vxm_alu_chain_gelu
#[test]
fn test_vxm_alu_chain_gelu() {
    let mut m = machine("ref");
    let lat = m.latency_of_instr(&mul(F32, 0, vs(0, 0), vs(0, 0)));
    let step = (lat.d_func - lat.d_skew + 1) as u64; // consumer dispatch - producer dispatch
    let x = f32x4(0x3f800000, 0xbfc00000, 0x40000000, 0x3e800000); // 1.0, -1.5, 2.0, 0.25
    let ie = 0b010100; // E:20, the intermediate
    let ic = vs(ie, 1); // E:20.chain
    let ex = vs(0b000000, 0); // E:0..3 x
    let ek = vs(0b000100, 0); // E:4..7 a constant
    let t0 = 40u64;
    vfeed(&mut m, 0, &x, t0 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, 88, t0, &mul(F32, ie, ex, ex)), Exec::Issued); // x^2
    for k in 1..=8u64 {
        let t = t0 + k * step;
        tick_to(&mut m, t - step + lat.d_skew as u64, t + lat.d_skew as u64 - 1);
        let rc = t + lat.d_skew as u64;
        let q = 88 + k as usize;
        let r = match k {
            1 => {
                vfeed(&mut m, 0, &x, rc);
                exec(&mut m, q, t, &mul(F32, ie, ic, ex)) // x^3
            }
            2 => {
                vfeed(&mut m, 4, &bcast(0x3d372713), rc);
                exec(&mut m, q, t, &mul(F32, ie, ic, ek)) // * 0.044715
            }
            3 => {
                vfeed(&mut m, 0, &x, rc);
                exec(&mut m, q, t, &add(F32, ie, ic, ex)) // + x
            }
            4 => {
                vfeed(&mut m, 4, &bcast(0x3f4c422a), rc);
                exec(&mut m, q, t, &mul(F32, ie, ic, ek)) // * sqrt(2/pi)
            }
            5 => exec(&mut m, q, t, &un(EntryId::TanH, F32, ie, ic, 0)), // tanh
            6 => {
                vfeed(&mut m, 4, &bcast(0x3f800000), rc);
                exec(&mut m, q, t, &add(F32, ie, ic, ek)) // + 1
            }
            7 => {
                vfeed(&mut m, 0, &x, rc);
                exec(&mut m, q, t, &mul(F32, ie, ic, ex)) // * x
            }
            _ => {
                vfeed(&mut m, 4, &bcast(0x3f000000), rc);
                exec(&mut m, q, t, &mul(F32, 0b011000, ic, ek)) // * 0.5 -> E:24
            }
        };
        assert_eq!(r, Exec::Issued, "k = {k}");
    }
    let t8 = t0 + 8 * step;
    tick_to(&mut m, t8 + lat.d_skew as u64, t8 + lat.d_func as u64);
    let seen = t8 + lat.d_func as u64 + 1;
    assert_eq!(lane(&m, 24, 0, seen), 0x3f57585c); // gelu(1.0)
    assert_eq!(lane(&m, 24, 1, seen), 0xbdcdad6c); // gelu(-1.5)
    assert_eq!(lane(&m, 24, 2, seen), 0x3ffa3042); // gelu(2.0)
    assert_eq!(lane(&m, 24, 3, seen), 0x3e19447f); // gelu(0.25)
}

// model: unit_tests/test_vxm_mxm.sail test_vxm_alu_chain_timing
#[test]
fn test_vxm_alu_chain_timing() {
    let mut m = machine("ref");
    let lat = m.latency_of_instr(&neg(I8, 0, vs(0, 0)));
    let (dsk, dfn) = (lat.d_skew as u64, lat.d_func as u64);
    let step = dfn - dsk + 1;
    let ie = 0b001000; // E:8 (i8)
    for dt in 0..=2u64 {
        m.reset();
        let mut xs = [0u32; tsp_iss::geom::MAX_LANES];
        xs[0] = 5;
        let p = words_to_planes(&xs, VL);
        let g = vxm_gap(&m);
        if g > 0 {
            m.edit_state(|state| state.srf.write(RowId::E, 0, g - 1, 40 + dsk - 1, &p[0])).unwrap();
        }
        assert_eq!(exec(&mut m, 88, 40, &neg(I8, ie, vs(0, 0))), Exec::Issued);
        let t = 40 + step + dt - 1;
        tick_to(&mut m, 40, t + dsk - 1);
        let r = exec(&mut m, 89, t, &add(I8, 0b001100, vs(ie, 1), vs(ie, 1)));
        if dt == 1 {
            assert_eq!(r, Exec::Issued);
            tick_to(&mut m, t + dsk, t + dfn);
            assert_eq!(lane8(&m, 12, t + dfn + 1), Some(0xF6)); // -5 + -5
        } else {
            assert_eq!(r, Exec::Diagnostic(Diag::Tim001), "dt = {dt}");
        }
    }
}

// model: unit_tests/test_vxm_mxm.sail test_vxm_unchained_read_ignores_station
#[test]
fn test_vxm_unchained_read_ignores_station() {
    let mut m = machine("ref");
    let lat = m.latency_of_instr(&neg(I8, 0, vs(0, 0)));
    let (dsk, dfn) = (lat.d_skew as u64, lat.d_func as u64);
    let step = dfn - dsk + 1;
    let e0 = vs(0, 0);
    let one = |x: u8, m: &Machine| {
        one_plane({
            let mut v = [0u8; tsp_iss::geom::MAX_LANES];
            v[0] = x;
            let _ = m;
            v
        })
    };
    // producer: neg.i8 E:0, E:0 at 40 with x = 5 fed on E:0
    let feed1 = |m: &mut Machine, x: u8, rc: u64| {
        let g = vxm_gap(m);
        let p = one(x, m);
        m.edit_state(|state| state.srf.write(RowId::E, 0, g - 1, rc - 1, &p[0])).unwrap();
    };
    feed1(&mut m, 5, 40 + dsk);
    assert_eq!(exec(&mut m, 88, 40, &neg(I8, 0, e0)), Exec::Issued);
    let t = 40 + step;
    tick_to(&mut m, 40, t + dsk - 1);
    // consumer one cycle after the commit, E:0 fed with 7 from upstream: the unchained read is 7
    feed1(&mut m, 7, t + dsk);
    assert_eq!(exec(&mut m, 89, t, &add(I8, 0b001100, e0, e0)), Exec::Issued);
    tick_to(&mut m, t + dsk, t + dfn);
    assert_eq!(lane8(&m, 12, t + dfn + 1), Some(0x0E), "7 + 7, not -5 + -5");
    // the same consumer with no value fed for its cycle: TIM-001 for the plain read, the chained read finds -5
    m.reset();
    feed1(&mut m, 5, 40 + dsk);
    assert_eq!(exec(&mut m, 88, 40, &neg(I8, 0, e0)), Exec::Issued);
    tick_to(&mut m, 40, t + dsk - 1);
    assert_eq!(exec(&mut m, 89, t, &add(I8, 0b001100, e0, e0)), Exec::Diagnostic(Diag::Tim001));
    m.reset();
    feed1(&mut m, 5, 40 + dsk);
    assert_eq!(exec(&mut m, 88, 40, &neg(I8, 0, e0)), Exec::Issued);
    tick_to(&mut m, 40, t + dsk - 1);
    assert_eq!(exec(&mut m, 89, t, &add(I8, 0b001100, vs(0, 1), vs(0, 1))), Exec::Issued);
}

// model: unit_tests/test_vxm_mxm.sail test_vxm_chain_encoding
#[test]
fn test_vxm_chain_encoding() {
    let m = machine("ref");
    let isa: &Isa = m.isa();
    let i = Insn::Select {
        r#type: I8,
        dst: 0b001000,
        lhs: 0b000000,
        rhs: 0b000100,
        mask: 0b100000,
        sg4: 0,
        lhs_chain: 1,
        rhs_chain: 0,
        mask_chain: 1,
    };
    let w = encode(&i);
    assert_eq!((w >> 19) & 7, 0b101);
    assert_eq!(w & ((1 << 19) - 1), 0);
    assert_eq!(isa.decode(64, w), Some(i));
    assert_eq!(ots_isa::render(&i), "select.i8 E:8, E:0.chain, E:4, W:0.chain");
    let n = encode(&neg(I8, 0b001000, vs(0, 1)));
    assert_eq!((n >> 19) & 7, 0b100);
    // a chain bit for an operand the instruction does not have, and a reserved bit, do not decode
    assert!(isa.decode(64, n | (1 << 20)).is_none());
    assert!(isa.decode(64, n | (1 << 17)).is_none());
}

// model: unit_tests/test_vxm_mxm.sail test_vxm_sg4
#[test]
fn test_vxm_sg4() {
    let mut m = machine("ref");
    let isa: Isa = Isa::new(&Profile::load("ref").unwrap());
    let i = Insn::Max { r#type: I8, dst: 0b100000, lhs: 0b100000, rhs: 0b100100, sg4: 1, lhs_chain: 0, rhs_chain: 0 };
    let w = encode(&i);
    assert!((w >> 18) & 1 == 1 && w & ((1 << 18) - 1) == 0);
    assert_eq!(isa.decode(64, w), Some(i));
    assert_eq!(ots_isa::render(&i), "max.i8.sg4 W:0, W:0, W:4");
    let c =
        Insn::Cmp { r#type: I16, dst: 0b001000, lhs: 0, rhs: 0b000100, mode: 2, sg4: 1, lhs_chain: 0, rhs_chain: 1 };
    assert_eq!(ots_isa::render(&c), "cmp.lt.i16.sg4 E:8, E:0, E:4.chain");
    // Convert and Reduce change or fold widths: bit 18 stays reserved there.
    let cv = encode(&Insn::Convert { r#type: I8, dst: 0b001000, lhs: 0, dtype: F32, lhs_chain: 0 });
    assert!(isa.decode(64, cv | (1 << 18)).is_none());
    let rd = encode(&Insn::Reduce { r#type: I8, dst: 0b001000, lhs: 0, mode: 0, lhs_chain: 0 });
    assert!(isa.decode(64, rd | (1 << 18)).is_none());
    // Execution: lane 0 of a = 0x7f0180ff, b = 0x80027f00 (bytes, low first: a = -1, -128, 1, 127;
    // b = 0, 127, 2, -128).
    let lat = m.latency_of_instr(&i);
    let a = f32x4(0x7f0180ff, 0, 0, 0);
    let b = f32x4(0x80027f00, 0, 0, 0);
    let (ea, eb, ed) = (vs(0, 0), vs(0b000100, 0), 0b001000);
    let t0 = 40u64;
    let seen = t0 + lat.d_func as u64 + 1;
    for k in 0..4 {
        m.reset();
        vfeed(&mut m, 0, &a, t0 + lat.d_skew as u64);
        vfeed(&mut m, 4, &b, t0 + lat.d_skew as u64);
        let ins = match k {
            0 => bin(EntryId::Max, I8, ed, ea, eb, 1),
            1 => bin(EntryId::Max, I16, ed, ea, eb, 1),
            2 => bin(EntryId::Max, I32, ed, ea, eb, 1),
            _ => Insn::Cmp { r#type: I8, dst: ed, lhs: ea.0, rhs: eb.0, mode: 2, sg4: 1, lhs_chain: 0, rhs_chain: 0 },
        };
        assert_eq!(exec(&mut m, 88, t0, &ins), Exec::Issued);
        tick_to(&mut m, t0, t0 + lat.d_func as u64);
        let want = [0x7f027f00u32, 0x7f017f00, 0x7f0180ff, 0x00ffffff][k];
        assert_eq!(lane(&m, 8, 0, seen), want, "k = {k}");
    }
    // A group operand must be SG4-aligned (OTS-E-OPR-002).
    m.reset();
    assert_eq!(exec(&mut m, 88, t0, &bin(EntryId::Add, I8, ed, vs(1, 0), eb, 1)), Exec::Diagnostic(Diag::Opr002));
}

// model: unit_tests/test_vxm_mxm.sail test_skew_reads_after_dispatch
#[test]
fn test_skew_reads_after_dispatch() {
    let mut m = machine("ref");
    let i = neg(I8, 0b001000, vs(0, 0));
    let lat = m.latency_of_instr(&i);
    assert!(lat.d_skew > 0, "the profile's VXM d_skew is positive");
    let mut v = [0u8; tsp_iss::geom::MAX_LANES];
    v[0] = 5;
    let g = vxm_gap(&m);
    // The operand is visible at the VXM in cycle 40 + d_skew: the read finds it.
    m.edit_state(|state| state.srf.write(RowId::E, 0, g - 1, 40 + lat.d_skew as u64 - 1, &v)).unwrap();
    assert_eq!(exec(&mut m, 88, 40, &i), Exec::Issued);
    // Visible only in cycle 40 - d_skew (the opposite sign): nothing to read.
    m.reset();
    m.edit_state(|state| state.srf.write(RowId::E, 0, g - 1, 40 - lat.d_skew as u64 - 1, &v)).unwrap();
    assert_eq!(exec(&mut m, 88, 40, &i), Exec::Diagnostic(Diag::Tim001));
}

// model: unit_tests/test_vxm_mxm.sail test_skew_execute_stage
#[test]
fn test_skew_execute_stage() {
    let mut m = machine("ref");
    let i = neg(I8, 0b001000, vs(0, 0));
    let lat = m.latency_of_instr(&i);
    m.run_element(88, i, 40, 0, 0, 0, &mut Rec::default());
    assert!(!m.quiescent(), "the element waits in the execute ring");
    assert_eq!(m.next_event(40), Some(40 + u128::from(lat.d_skew)));
    let mut v = [0u8; tsp_iss::geom::MAX_LANES];
    v[0] = 5;
    let g = vxm_gap(&m);
    m.edit_state(|state| state.srf.write(RowId::E, 0, g - 1, 40 + lat.d_skew as u64 - 1, &v)).unwrap();
    tick_to(&mut m, 41, 40 + lat.d_func as u64);
    assert_eq!(lane8(&m, 8, 40 + lat.d_func as u64 + 1), Some(0xFB)); // -5
    assert!(m.quiescent());
}

// model: unit_tests/test_vxm_mxm.sail test_timing_entry_table
#[test]
fn test_timing_entry_table() {
    let m = machine("ref");
    let ent = |i: &Insn| ots_isa::entry(i.id()).name;
    let sub_sat = bin(EntryId::Sub_sat, I8, 0b001000, vs(0, 0), vs(0b000100, 0), 0);
    assert_eq!(ent(&sub_sat), "Sub_sat");
    assert_eq!(ent(&Insn::Conv_F16_I8 { r#type: F16, dst: 0b001000, lhs: 0, lhs_chain: 0 }), "Conv_F16_I8");
    assert_eq!(ent(&Insn::Convert { r#type: F16, dst: 8, lhs: 0, dtype: I8, lhs_chain: 0 }), "Convert");
    // `timing_of` the VXM entries: every one has the VXM class and the same row as its class
    let class = m.latency_of_instr(&add(F32, 8, vs(0, 0), vs(4, 0)));
    for id in EntryId::ALL.iter().filter(|i| i.entry().slice == "VXM") {
        assert_eq!(id.entry().timing, ots_profile::TimingClass::Vxm, "{id:?}");
        assert_eq!(m.timing().row(id.index()), class, "{id:?}");
    }
}

// model: unit_tests/test_vxm_mxm.sail test_timing_type_rows
#[test]
fn test_timing_type_rows() {
    use ots_profile::eltype_name;
    let m = machine("ref");
    let code = |name: &str| ots_profile::ELTYPES.iter().find(|(n, _)| *n == name).unwrap().1 as u32;
    // the element type an instruction is timed by: its `type` operand named, or "" for an instruction without one
    let type_of = |i: &Insn| i.get("type").and_then(eltype_name).unwrap_or("");
    let (d, a, b) = (0b001000, vs(0, 0), vs(0b000100, 0));
    let max = |t: u32| bin(EntryId::Max, t, d, a, b, 0);
    assert_eq!(type_of(&max(I8)), "i8");
    assert_eq!(type_of(&max(F32)), "f32");
    assert_eq!(type_of(&un(EntryId::Exp, F16, d, a, 0)), "f16");
    assert_eq!(type_of(&Insn::Conv_F16_I8 { r#type: F16, dst: d, lhs: 0, lhs_chain: 0 }), "f16");
    assert_eq!(type_of(&Insn::Convert { r#type: U8, dst: d, lhs: 0, dtype: F32, lhs_chain: 0 }), "u8");
    assert_eq!(type_of(&Insn::Reduce { r#type: code("e4m3"), dst: d, lhs: 0, mode: 0, lhs_chain: 0 }), "e4m3");
    assert_eq!(type_of(&Insn::NOP { n: 1 }), "");
    // which pairs of entry and type exist
    let takes = |e: &str, t: &str| m.timing().entry_typed(e, t).is_some();
    assert!(takes("Max", "i8") && takes("Max", "e5m2"));
    assert!(!takes("Max", "b8") && !takes("Add_sat", "f32"));
    assert!(takes("Exp", "f16") && !takes("Exp", "i32"));
    assert!(takes("Conv_F16_I8", "f16") && !takes("Conv_F16_I8", "i8"));
    assert!(takes("Cmp", "b32") && takes("Convert", "b8") && !takes("Reduce", "b8"));
    assert!(!takes("Read", "i8") && !takes("Max", "q9") && !takes("Nothing", "i8"));
    // the row of an instruction is its typed row, else its entry's row; no profile has a row for Max at e5m2
    assert_eq!(Some(m.latency_of_instr(&max(I8))), m.timing().entry_typed("Max", "i8"));
    assert_eq!(Some(m.latency_of_instr(&max(code("e5m2")))), m.timing().entry("Max"));
}

/// An override replaces the class's row for that entry only (`timing.overrides`): Exp commits 4
/// cycles later than the other VXM entries and reads 2 cycles later.
#[test]
fn a_timing_override_applies_to_its_entry_only() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../opentsp-arch-spec/config/ref.json");
    let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
    v["timing"]["overrides"] = serde_json::json!([{ "entry": "Exp", "d_func": 9, "d_skew": 4 }]);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ref_override.json");
    std::fs::write(&path, serde_json::to_vec(&v).unwrap()).unwrap();
    let p = Profile::load_file("ref_override", &path).unwrap();
    let mut m = Machine::new(&p, 0, Box::new(NoLinks)).unwrap();
    let exp = un(EntryId::Exp, F32, 16, vs(0, 0), 0);
    let class = m.latency_of_instr(&add(F32, 8, vs(0, 0), vs(4, 0)));
    assert_eq!((class.d_func, class.d_skew), (5, 2));
    let l = m.latency_of_instr(&exp);
    assert_eq!((l.d_func, l.d_skew), (9, 4));
    assert_eq!(m.latency_of_instr(&un(EntryId::Log, F32, 16, vs(0, 0), 0)), class);
    // the result is there at dispatch + 9 + 1 (`run` ticks to the entry's own d_func)
    let z = words(|_| 0);
    assert_eq!(run(&mut m, &exp, &z, &z, &z).unwrap()[0], 0x3f80_0000);
}

/// A profile row whose entry is `<Entry>.<type>` is the row of that element type only: it beats the
/// entry's own row, which beats the class row; the other types of the entry keep the entry's row.
#[test]
fn a_per_type_timing_row_applies_to_its_type_only() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../opentsp-arch-spec/config/ref.json");
    let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
    v["timing"]["overrides"].as_array_mut().unwrap().extend([
        serde_json::json!({ "entry": "Max.i8", "d_func": 3, "d_skew": 2 }),
        serde_json::json!({ "entry": "Max.f32", "d_func": 9, "d_skew": 4 }),
        serde_json::json!({ "entry": "Min", "d_func": 7, "d_skew": 1 }),
        serde_json::json!({ "entry": "Min.u8", "d_func": 2, "d_skew": 1 }),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ref_typed.json");
    std::fs::write(&path, serde_json::to_vec(&v).unwrap()).unwrap();
    let p = Profile::load_file("ref_typed", &path).unwrap();
    let mut m = Machine::new(&p, 0, Box::new(NoLinks)).unwrap();
    let lat = |m: &Machine, id, t| {
        let l = m.latency_of_instr(&bin(id, t, 16, vs(0, 0), vs(4, 0), 0));
        (l.d_func, l.d_skew)
    };
    assert_eq!(lat(&m, EntryId::Max, I8), (3, 2));
    assert_eq!(lat(&m, EntryId::Max, F32), (9, 4));
    assert_eq!(lat(&m, EntryId::Max, F16), (5, 2), "a type without a row keeps the class row");
    assert_eq!(lat(&m, EntryId::Add, I8), (5, 2), "another entry is untouched");
    assert_eq!(lat(&m, EntryId::Min, I8), (7, 1), "the entry's own row covers its other types");
    assert_eq!(lat(&m, EntryId::Min, U8), (2, 1), "the typed row beats the entry row");
    // the typed row moves the commit: max.i8 of 5 and 9 is there at dispatch + 3 + 1
    let z = words(|_| 0);
    let r = run(&mut m, &bin(EntryId::Max, I8, 16, vs(0, 0), vs(4, 0), 0), &words(|_| 5), &words(|_| 9), &z).unwrap();
    assert_eq!(r[0], 9);
    let one_two = (words(|_| 0x3f80_0000), words(|_| 0x4000_0000));
    let r = run(&mut m, &bin(EntryId::Max, F32, 16, vs(0, 0), vs(4, 0), 0), &one_two.0, &one_two.1, &z).unwrap();
    assert_eq!(r[0], 0x4000_0000);
}

// ---------------------------------------------------------------------------------------
// Operation families, on operands fed at the VXM and read back just past it.

/// Run `i` on queue 88 at cycle 40 with the operands fed at E:0, E:4 and E:8 (the third operand),
/// tick to its commit and return the result planes of E:16.. as 32-bit lanes, or the diagnostic.
fn run(m: &mut Machine, i: &Insn, a: &Planes, b: &Planes, c: &Planes) -> Result<Vec<u32>, Diag> {
    m.reset();
    let lat = m.latency_of_instr(i);
    let rc = 40 + lat.d_skew as u64;
    vfeed(m, 0, a, rc);
    vfeed(m, 4, b, rc);
    vfeed(m, 8, c, rc);
    match exec(m, 88, 40, i) {
        Exec::Issued => {}
        Exec::Diagnostic(d) => return Err(d),
    }
    tick_to(m, 40, 40 + lat.d_func as u64);
    let t = 40 + lat.d_func as u64 + 1;
    let g = vxm_gap(m);
    let n = vl(m);
    let mut out = vec![0u32; n];
    for b in 0..4u8 {
        if let (SrfLookup::Found, p) = m.srf().read(RowId::E, 16 + b, g + 1, t as i64) {
            for l in 0..n {
                out[l] |= (p[l] as u32) << (8 * b);
            }
        }
    }
    Ok(out)
}

fn words(f: impl Fn(usize) -> u32) -> Planes {
    let mut xs = [0u32; tsp_iss::geom::MAX_LANES];
    for (l, x) in xs.iter_mut().enumerate().take(VL) {
        *x = f(l);
    }
    words_to_planes(&xs, VL)
}

#[test]
fn binary_operations_per_type() {
    let mut m = machine("ref");
    let a = words(|l| 0x7f00_0000 | (l as u32).wrapping_mul(0x0101_0101));
    let b = words(|l| 0x0100_0000 | (l as u32));
    let z = words(|_| 0);
    let op = |id, t, g| bin(id, t, 16, vs(0, 0), vs(4, 0), g);
    // add.i8 wraps, add_sat.i8 saturates: lane 3 is 3 + 3 = 6 (and 0x7f lane: 0x7f + 0x01)
    let r = run(&mut m, &op(EntryId::Add, I8, 0), &a, &b, &z).unwrap();
    assert_eq!(r[1] & 0xff, 0x01 + 0x01);
    let sat = run(&mut m, &op(EntryId::Add_sat, I8, 0), &words(|_| 0x7f), &words(|_| 0x01), &z).unwrap();
    assert_eq!(sat[0], 0x7f, "saturated");
    let wrap = run(&mut m, &op(EntryId::Add, I8, 0), &words(|_| 0x7f), &words(|_| 0x01), &z).unwrap();
    assert_eq!(wrap[0], 0x80, "wrapped");
    // i16 and i32 take two and four planes
    let r = run(&mut m, &op(EntryId::Mul, I16, 0), &words(|_| 300), &words(|_| 200), &z).unwrap();
    assert_eq!(r[0], 60000);
    let r = run(&mut m, &op(EntryId::Sub, I32, 0), &words(|_| 5), &words(|_| 9), &z).unwrap();
    assert_eq!(r[0], 0xffff_fffc);
    // f32 add: 1.5 + 2.25 = 3.75; f16 add: 1.0 + 1.0 = 2.0
    let r = run(&mut m, &op(EntryId::Add, F32, 0), &words(|_| 0x3fc0_0000), &words(|_| 0x4010_0000), &z).unwrap();
    assert_eq!(r[0], 0x4070_0000);
    let r = run(&mut m, &op(EntryId::Add, F16, 0), &words(|_| 0x3c00), &words(|_| 0x3c00), &z).unwrap();
    assert_eq!(r[0], 0x4000);
    // bitwise on integers, logical on booleans (b8 = 8): any non-zero byte is true, so xor of two
    // true values is false
    let r = run(&mut m, &op(EntryId::Xor, I8, 0), &words(|_| 0xff), &words(|_| 0x0f), &z).unwrap();
    assert_eq!(r[0] & 0xff, 0xf0);
    let r = run(&mut m, &op(EntryId::Xor, 8, 0), &words(|_| 0xff), &words(|_| 0x0f), &z).unwrap();
    assert_eq!(r[0] & 0xff, 0);
}

#[test]
fn unary_operations_and_transcendentals() {
    let mut m = machine("ref");
    let z = words(|_| 0);
    let u = |id, t| un(id, t, 16, vs(0, 0), 0);
    let r = run(&mut m, &u(EntryId::Neg, I8), &words(|_| 5), &z, &z).unwrap();
    assert_eq!(r[0], 0xfb);
    let r = run(&mut m, &u(EntryId::Abs, I8), &words(|_| 0xfb), &z, &z).unwrap();
    assert_eq!(r[0], 5);
    let r = run(&mut m, &u(EntryId::Abs, I8), &words(|_| 0x80), &z, &z).unwrap();
    assert_eq!(r[0], 0x80, "Abs wraps at the minimum");
    let r = run(&mut m, &u(EntryId::ReLU, I8), &words(|_| 0xfb), &z, &z).unwrap();
    assert_eq!(r[0], 0);
    let r = run(&mut m, &u(EntryId::Not, I8), &words(|_| 0x0f), &z, &z).unwrap();
    assert_eq!(r[0], 0xf0);
    // exp(0) = 1, log(1) = 0, tanh(0) = 0 on f32; rsqrt(1) = 1
    let one = words(|_| 0x3f80_0000);
    assert_eq!(run(&mut m, &u(EntryId::Exp, F32), &z, &z, &z).unwrap()[0], 0x3f80_0000);
    assert_eq!(run(&mut m, &u(EntryId::Log, F32), &one, &z, &z).unwrap()[0], 0);
    assert_eq!(run(&mut m, &u(EntryId::TanH, F32), &z, &z, &z).unwrap()[0], 0);
    assert_eq!(run(&mut m, &u(EntryId::RSqrt, F32), &one, &z, &z).unwrap()[0], 0x3f80_0000);
    // the f16 table and the i8 transcendentals: exp.f16(0) = 1.0
    assert_eq!(run(&mut m, &u(EntryId::Exp, F16), &z, &z, &z).unwrap()[0], 0x3c00);
    assert_eq!(run(&mut m, &u(EntryId::Exp, I8), &z, &z, &z).unwrap()[0], 1);
}

#[test]
fn mask_select_and_cmp() {
    let mut m = machine("ref");
    let a = words(|l| 10 + l as u32);
    let b = words(|l| 100 + l as u32);
    let msk = words(|l| (l % 2) as u32);
    let z = words(|_| 0);
    // Mask: dst = rhs != 0 ? lhs : 0
    let i = Insn::Mask { r#type: I8, dst: 16, lhs: 0, mask: 4, sg4: 0, lhs_chain: 0, mask_chain: 0 };
    let r = run(&mut m, &i, &a, &msk, &z).unwrap();
    assert_eq!((r[0], r[1], r[2], r[3]), (0, 11, 0, 13));
    // Select: dst = mask != 0 ? lhs : rhs, the mask from the third operand
    let i = Insn::Select {
        r#type: I8,
        dst: 16,
        lhs: 0,
        rhs: 4,
        mask: 8,
        sg4: 0,
        lhs_chain: 0,
        rhs_chain: 0,
        mask_chain: 0,
    };
    let r = run(&mut m, &i, &a, &b, &msk).unwrap();
    assert_eq!((r[0], r[1], r[2], r[3]), (100, 11, 102, 13));
    // Cmp: all ones or zero; lt on signed i8: 0x80 (-128) < 1
    for (mode, want) in [(0, 0u32), (1, 0xff), (2, 0xff), (3, 0xff), (4, 0), (5, 0)] {
        let i = Insn::Cmp { r#type: I8, dst: 16, lhs: 0, rhs: 4, mode, sg4: 0, lhs_chain: 0, rhs_chain: 0 };
        let r = run(&mut m, &i, &words(|_| 0x80), &words(|_| 1), &z).unwrap();
        assert_eq!(r[0], want, "mode {mode}");
    }
}

#[test]
fn conversions_and_convert() {
    let mut m = machine("ref");
    let z = words(|_| 0);
    let cv = |id, t, s: u32| match id {
        EntryId::Convert => Insn::Convert { r#type: t, dst: 16, lhs: 0, dtype: s, lhs_chain: 0 },
        _ => Insn::from_fields(id, &{
            let mut f = [0u32; ots_isa::MAX_VARS];
            f[..4].copy_from_slice(&[t, 16, 0, 0]);
            f
        }),
    };
    // i8 -> f32 (4 result planes from 1 source plane), and back with saturation
    let r = run(&mut m, &cv(EntryId::Conv_I8_F32, I8, 0), &words(|_| 0xfe), &z, &z).unwrap();
    assert_eq!(r[0], 0xc000_0000, "-2.0");
    let r = run(&mut m, &cv(EntryId::Conv_F32_I8, F32, 0), &words(|_| 0x4348_0000), &z, &z).unwrap();
    assert_eq!(r[0], 0x7f, "200.0 saturates to 127");
    let r = run(&mut m, &cv(EntryId::Conv_I8_F16, I8, 0), &words(|_| 3), &z, &z).unwrap();
    assert_eq!(r[0], 0x4200, "3.0 as f16");
    let r = run(&mut m, &cv(EntryId::Conv_F16_F32, F16, 0), &words(|_| 0x3c00), &z, &z).unwrap();
    assert_eq!(r[0], 0x3f80_0000);
    let r = run(&mut m, &cv(EntryId::Conv_I8_I32, I8, 0), &words(|_| 0xff), &z, &z).unwrap();
    assert_eq!(r[0], 0xffff_ffff, "sign extension");
    // Convert: any source type to any destination type: f32 -> u8 clamps below zero, u8 -> f16
    let r = run(&mut m, &cv(EntryId::Convert, F32, 4), &words(|_| 0xc000_0000), &z, &z).unwrap();
    assert_eq!(r[0], 0, "-2.0 -> u8 saturates at 0");
    let r = run(&mut m, &cv(EntryId::Convert, 4, F16), &words(|_| 255), &z, &z).unwrap();
    assert_eq!(r[0], 0x5bf8, "255 as f16");
    let r = run(&mut m, &cv(EntryId::Convert, I16, I8), &words(|_| 0x1234), &z, &z).unwrap();
    assert_eq!(r[0], 0x7f, "i16 -> i8 saturates");
}

#[test]
fn reduce_folds_each_superlane_group_into_its_first_lane() {
    let mut m = machine("ref");
    let z = words(|_| 0);
    let a = words(|l| 1 + (l / 16) as u32 + (l % 16) as u32);
    let red = |mode, t| Insn::Reduce { r#type: t, dst: 16, lhs: 0, mode, lhs_chain: 0 };
    let r = run(&mut m, &red(0, I8), &a, &z, &z).unwrap();
    for g in 0..m.geom().superlanes as usize {
        let sum: u32 = (0..16).map(|j| 1 + g as u32 + j).sum();
        assert_eq!(r[16 * g], sum & 0xff, "group {g}");
        assert!((1..16).all(|j| r[16 * g + j] == 0), "the other lanes are zero");
    }
    let r = run(&mut m, &red(1, I8), &a, &z, &z).unwrap();
    assert_eq!((r[0], r[16]), (16, 17), "max");
    let r = run(&mut m, &red(2, I8), &a, &z, &z).unwrap();
    assert_eq!((r[0], r[16]), (1, 2), "min");
    // f32: an ascending fold rounds at each step: 2^24 + 1 + 1 + ... stays 2^24
    let big = words(|l| if l % 16 == 0 { 0x4b80_0000 } else { 0x3f80_0000 });
    let r = run(&mut m, &red(0, F32), &big, &z, &z).unwrap();
    assert_eq!(r[0], 0x4b80_0000, "each +1 is rounded away");
}

#[test]
fn operand_errors_are_diagnostics_with_no_effect() {
    let mut m = machine("ref");
    let z = words(|_| 0);
    // an unfed operand: TIM-001, and nothing is scheduled
    m.reset();
    let i = add(I8, 16, vs(0, 0), vs(4, 0));
    assert_eq!(exec(&mut m, 88, 40, &i), Exec::Diagnostic(Diag::Tim001));
    assert_eq!(m.driver().pending.iter().map(|s| s.len()).sum::<usize>(), 0);
    // a misaligned multi-byte operand (f32 at E:1) or destination: OPR-002
    assert_eq!(run(&mut m, &add(F32, 16, vs(1, 0), vs(4, 0)), &z, &z, &z), Err(Diag::Opr002));
    assert_eq!(run(&mut m, &add(F32, 17, vs(0, 0), vs(4, 0)), &z, &z, &z), Err(Diag::Opr002));
    // the second operand's error is reported after the first's (the first failing check wins)
    assert_eq!(run(&mut m, &add(F32, 16, vs(1, 0), vs(5, 0)), &z, &z, &z), Err(Diag::Opr002));
}

#[test]
fn conversions_and_reductions_count_under_their_source_type() {
    let mut m = machine("ref");
    let a = bcast(1);
    let cv = Insn::Conv_I8_F16 { r#type: I8, dst: 8, lhs: 0, lhs_chain: 0 };
    let rd = Insn::Reduce { r#type: I8, dst: 8, lhs: 0, mode: 0, lhs_chain: 0 };
    for (c, i) in [(40u64, &cv), (60, &rd)] {
        let at = c + m.latency_of_instr(i).d_skew as u64;
        vfeed(&mut m, 0, &a, at);
        assert_eq!(exec(&mut m, 88, c, i), Exec::Issued, "{i:?}");
    }
    let i8_code = 0;
    assert_eq!(m.activity().vxm[EntryId::Conv_I8_F16.index()][i8_code], 320);
    assert_eq!(m.activity().vxm[EntryId::Reduce.index()][i8_code], 320);
    assert_eq!(m.activity().vxm_ops_of_type(i8_code), 640);
}
