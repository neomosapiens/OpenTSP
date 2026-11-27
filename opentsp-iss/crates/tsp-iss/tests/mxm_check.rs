//! The sum check of an ABC (`microarchitecture.md` sec. 18): the check itself on exact sums (integer, f16, fp8), the
//! `F mxm` input line and the pending fault. The whole path through the machine is held to `tsp_sim` by
//! `opentsp-arch-spec/test/directed/mxm_check.py` run with the ISS, and by the golden vectors (`ecc-*`).
use ots_profile::Profile;
use tsp_iss::geom::Geom;
use tsp_iss::harness::input::{parse, Limits};
use tsp_iss::host::HostKind;
use tsp_iss::slices::mxm::{Mxm, MxmFault, WType};
use tsp_iss::slices::mxm_check::{mxm_check, Verdict};
use tsp_iss::srf::Vec320;

fn geom(name: &str) -> Geom {
    Geom::new(&Profile::load(name).unwrap())
}

fn err(input: &str) -> String {
    let g = geom("ref");
    parse(input.as_bytes(), Limits { geom: &g, chips: 1, topo: None }).unwrap_err().msg
}

fn lanes(a: &[u8]) -> Vec320 {
    let mut v = [0u8; tsp_iss::geom::MAX_LANES];
    v[..a.len()].copy_from_slice(a);
    v
}

/// Weights rows [1, 2], [3, 4], [5, 6] and the activation [10, 20] (products 50, 110, 170), as `test_mxm_check`.
fn i8_case(faults: &[(usize, u32)]) -> Verdict {
    let w = [lanes(&[1, 2]), lanes(&[3, 4]), lanes(&[5, 6])];
    let x =
        [lanes(&[10, 20]), [0; tsp_iss::geom::MAX_LANES], [0; tsp_iss::geom::MAX_LANES], [0; tsp_iss::geom::MAX_LANES]];
    mxm_check(&w, 3, 2, WType::I8, WType::I8, 0, &x, faults)
}

// model: unit_tests/test_vxm_mxm.sail test_mxm_check
// model: slices/mxm/mxm_insts.sail mxm_check
#[test]
fn one_flipped_lane_is_located_and_two_are_not() {
    assert_eq!(i8_case(&[]), Verdict::Clean);
    for (lane, bit) in [(0, 0), (1, 5), (2, 40), (2, 95), (0, 63), (1, 94)] {
        assert_eq!(i8_case(&[(lane, bit)]), Verdict::Corrected(lane), "lane {lane} bit {bit}");
    }
    // two bits of one lane are one wrong lane
    assert_eq!(i8_case(&[(2, 3), (2, 9)]), Verdict::Corrected(2));
    // two lanes are uncorrectable, whichever bits
    assert_eq!(i8_case(&[(0, 3), (2, 9)]), Verdict::Uncorrectable);
    assert_eq!(i8_case(&[(0, 4), (1, 4)]), Verdict::Uncorrectable);
    // a lane beyond the installed rows carries no product
    assert_eq!(i8_case(&[(7, 3)]), Verdict::Clean);
}

// model: slices/mxm/mxm_insts.sail chk_elem, fval_int, flip96
#[test]
fn the_floating_sums_are_exact() {
    // f16: 1.0 = 0x3C00, 2.0 = 0x4000, 0.5 = 0x3800; NaN (0x7E00) and infinity (0x7C00) add nothing to either side
    let one = [0x00u8, 0x3C];
    let two = [0x00u8, 0x40];
    let half = [0x00u8, 0x38];
    let nan = [0x00u8, 0x7E];
    let inf = [0x00u8, 0x7C];
    let row = |a: [u8; 2], b: [u8; 2], c: [u8; 2]| (lanes(&[a[0], b[0], c[0]]), lanes(&[a[1], b[1], c[1]]));
    let (r0l, r0h) = row(one, two, half);
    let (r1l, r1h) = row(two, half, one);
    let (r2l, r2h) = row(half, one, two);
    let w = [r0l, r0h, r1l, r1h, r2l, r2h];
    let (xl, xh) = row(one, one, one);
    let x = [xl, xh, [0; tsp_iss::geom::MAX_LANES], [0; tsp_iss::geom::MAX_LANES]];
    let f16 = |faults: &[(usize, u32)]| mxm_check(&w, 3, 3, WType::F16, WType::F16, 0, &x, faults);
    assert_eq!(f16(&[]), Verdict::Clean);
    assert_eq!(f16(&[(1, 30)]), Verdict::Corrected(1));
    assert_eq!(f16(&[(1, 30), (2, 30)]), Verdict::Uncorrectable);
    // a NaN and an infinity in the activation: the finite lanes still check
    let (xl, xh) = row(one, nan, inf);
    let x2 = [xl, xh, [0; tsp_iss::geom::MAX_LANES], [0; tsp_iss::geom::MAX_LANES]];
    assert_eq!(mxm_check(&w, 3, 3, WType::F16, WType::F16, 0, &x2, &[(0, 7)]), Verdict::Corrected(0));
    // fp8 e4m3: 1.0 = 0x38, 2.0 = 0x40; e5m2: 1.0 = 0x3C
    let w8 = [lanes(&[0x38, 0x40]), lanes(&[0x40, 0x38])];
    let x8 = [
        lanes(&[0x38, 0x38]),
        [0; tsp_iss::geom::MAX_LANES],
        [0; tsp_iss::geom::MAX_LANES],
        [0; tsp_iss::geom::MAX_LANES],
    ];
    assert_eq!(mxm_check(&w8, 2, 2, WType::Fp8, WType::Fp8, 0, &x8, &[(1, 20)]), Verdict::Corrected(1));
    let w5 = [lanes(&[0x3C, 0x40]), lanes(&[0x40, 0x3C])];
    let x5 = [
        lanes(&[0x3C, 0x3C]),
        [0; tsp_iss::geom::MAX_LANES],
        [0; tsp_iss::geom::MAX_LANES],
        [0; tsp_iss::geom::MAX_LANES],
    ];
    assert_eq!(mxm_check(&w5, 2, 2, WType::Fp8, WType::Fp8, 1, &x5, &[(0, 20), (1, 21)]), Verdict::Uncorrectable);
}

// model: c_emulator/tsp_sim.c read_mxm_fault
#[test]
fn an_f_mxm_line_is_read_and_checked_in_the_order_of_the_harness() {
    let g = geom("ref");
    let p = parse(b"F mxm 1c8 0 5 3f\nF mxm 1c8 3 13f 5f t\n", Limits { geom: &g, chips: 1, topo: None }).unwrap();
    let kinds: Vec<_> = p.host.iter().map(|h| (h.cycle, h.kind.clone())).collect();
    assert_eq!(
        kinds,
        vec![
            (0x1c8, HostKind::FlipMxm { plane: 0, lane: 5, bit: 0x3f }),
            (0x1c8, HostKind::FlipMxm { plane: 3, lane: 0x13f, bit: 0x5f }),
        ]
    );
    assert!(err("F mxm 10 4 0 0\n").contains("plane out of range (0..3)"));
    assert!(err("F mxm 10 0 140 0\n").contains("lane out of range (>= mxm.rows)"));
    assert!(err("F mxm 10 0 0 60\n").contains("bit out of range (0..5f)"));
    assert!(err("F mxm 10 0 0\n").contains("missing bit"));
    assert!(err("F mxm 10 0 0 1 x\n").contains("F: expected t or p at the end of the line"));
    assert!(err("F mxm 10 0 0 1\nF mxm 10 0 0 1\n").contains("two lines flip the same bit in one cycle"));
}

// model: postlude/harness_api.sail model_inject_mxm, model_mxm_pending
#[test]
fn a_fault_no_abc_took_is_pending_with_the_plane_of_the_last_line() {
    let mut m = Mxm::new();
    assert_eq!(m.pending(), None);
    m.inject(MxmFault { plane: 1, lane: 2, bit: 3, cycle: 0x20 });
    m.inject(MxmFault { plane: 2, lane: 0, bit: 9, cycle: 0x20 });
    m.inject(MxmFault { plane: 0, lane: 0, bit: 1, cycle: 0x30 });
    assert_eq!(m.pending(), Some((2, 0x20)), "the earliest cycle, the plane of the last line in it");
}

/// An `F mxm` line that no ABC took ends the run in the cycle in which it is stale, as `tsp_sim` does: nothing of the
/// later trace is printed (a read of slice 0x2c would put its stream value on the trace at cycle 0xaf).
// model: c_emulator/tsp_sim.c main loop after the tick
#[test]
fn a_stale_fault_stops_the_run_in_its_cycle() {
    use tsp_iss::harness::cli::{LinkMode, Options};
    use tsp_iss::harness::execute;
    let p = Profile::load("ref").unwrap();
    let g = Geom::new(&p);
    let run = |extra: &str| {
        let src = format!("S 2c 3 1 2 3\nQ 2c 000000aa 10000003 80000000\n{extra}");
        let prog = parse(src.as_bytes(), Limits { geom: &g, chips: 1, topo: None }).unwrap();
        let mut o = Options::new("cfg".into());
        o.trace = true;
        o.chips = 1;
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = execute(&p, &g, &prog, &o, LinkMode::Nominal, None, &mut out, &mut err);
        (code, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
    };
    let (code, out, err) = run("");
    assert_eq!((code, err.as_str()), (0, ""));
    assert!(out.contains("T srf "), "the program has a trace after cycle 0x50: {out}");
    let (code, out, err) = run("F mxm 50 0 0 0\n");
    assert_eq!((code, out.as_str()), (2, ""), "stopped in cycle 0x50, before any trace line");
    assert_eq!(err, "tsp-iss: F mxm: no ABC executes on plane 0 in cycle 50 (chip 0)\n");
}
