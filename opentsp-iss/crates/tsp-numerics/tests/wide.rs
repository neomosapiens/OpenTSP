//! The wide accumulator lane and the exact dot products of the floating-point MXM types
//! (`slices/mxm/mxm_insts.sail`: `wlane_add`, `wlane_read`, `f16_exact`, `fp8_exact`, `bfp8_exact`, `dsum_round`): the same
//! cases as the Sail unit tests of `test_ext_types.sail`.
use tsp_numerics::fp::{FMT_E4M3, FMT_E5M2};
use tsp_numerics::mxm::{bfp8_exact, dsum_round, f16_dot, f16_exact, fp8_dot, fp8_exact, Wide};

fn lane(v: i128) -> Wide {
    Wide { v, sp: 0, nz: false }
}

#[test]
fn three_tiny_additions_round_once() {
    let one = lane(1 << 48);
    let tiny = lane(1 << 23); // 2^-25
    let s3 = one.plus(tiny).plus(tiny).plus(tiny);
    assert_eq!(s3.v, (1 << 48) + 3 * (1 << 23));
    assert_eq!(s3.read(), 0x3F80_0001);
    // every step rounded to f32 loses the 2^-25 each time
    let mut f = 0x3F80_0000u32;
    for _ in 0..3 {
        f = tsp_numerics::fp::fp_add(tsp_numerics::fp::FMT32, f, 0x3300_0000);
    }
    assert_eq!(f, 0x3F80_0000);
}

// model: unit_tests/test_ext_types.sail test_wlane_arithmetic
#[test]
fn wraps_at_128_bits() {
    assert_eq!(lane(i128::MAX).plus(lane(1)).v, i128::MIN);
}

#[test]
fn specials_are_sticky_and_zero_has_a_sign() {
    let one = lane(1 << 48);
    let (pinf, ninf, nan) =
        (Wide { v: 0, sp: 1, nz: false }, Wide { v: 0, sp: 2, nz: false }, Wide { v: 0, sp: 3, nz: false });
    assert_eq!(pinf.plus(one).read(), 0x7F80_0000);
    assert_eq!(one.plus(ninf).read(), 0xFF80_0000);
    assert_eq!(pinf.plus(ninf).read(), 0x7FC0_0000);
    assert_eq!(nan.plus(one).read(), 0x7FC0_0000);
    assert_eq!(pinf.plus(ninf).plus(pinf).read(), 0x7FC0_0000);
    let nz = Wide { v: 0, sp: 0, nz: true };
    assert_eq!(nz.read(), 0x8000_0000);
    assert_eq!(nz.plus(nz).read(), 0x8000_0000);
    assert_eq!(nz.plus(Wide::default()).read(), 0);
    assert_eq!(one.plus(lane(-(1 << 48))).read(), 0);
}

#[test]
fn read_out_rounds_to_nearest_even() {
    assert_eq!(lane((1 << 48) + (1 << 24)).read(), 0x3F80_0000); // a tie goes to the even 1.0
    assert_eq!(lane((1 << 48) + 3 * (1 << 24)).read(), 0x3F80_0002); // a tie goes to the even 1 + 2^-22
    assert_eq!(lane(-((1 << 48) + (1 << 24) + 1)).read(), 0xBF80_0001);
    assert_eq!(lane(i128::MAX).read(), 0x6700_0000); // 2^79
}

fn vec16(a: &[(usize, u8)]) -> Vec<u8> {
    let mut v = vec![0u8; 16];
    for &(i, b) in a {
        v[i] = b;
    }
    v
}

// model: unit_tests/test_ext_types.sail test_abc_keeps_exact_sum
#[test]
fn exact_products_round_like_the_dot_products() {
    // f16: [2^-24, 1.0] . [2^-24, 1.0] = 1 + 2^-48 exactly; the f32 product rounds to 1.0, the exact sum keeps it
    let (wlo, whi) = (vec16(&[(0, 0x01)]), vec16(&[(1, 0x3C)]));
    let d = f16_exact(&wlo, &whi, &wlo, &whi);
    assert_eq!(d.sum, (1 << 48) + 1);
    assert_eq!(dsum_round(&d), 0x3F80_0000);
    assert_eq!(dsum_round(&d), f16_dot(&wlo, &whi, &wlo, &whi));
    // fp8: [448, 448, 2^-9] . [448, -448, 1] = 2^-9
    let w = vec16(&[(0, 0x7E), (1, 0x7E), (2, 0x01)]);
    let x = vec16(&[(0, 0x7E), (1, 0xFE), (2, 0x38)]);
    let d = fp8_exact(FMT_E4M3, &w, &x);
    assert_eq!(d.sum, 1 << 39);
    assert_eq!(dsum_round(&d), fp8_dot(FMT_E4M3, &w, &x));
    assert_eq!(dsum_round(&fp8_exact(FMT_E5M2, &w, &x)), fp8_dot(FMT_E5M2, &w, &x));
}

// model: unit_tests/test_ext_types.sail test_bfp8_abc
#[test]
fn bfp8_scales_apply_per_block() {
    // lane 0: 1.0 * 2^3 times 2.0 * 2^-2 = 4; lane 1: 0.5 * 2^3 times 0.5 * 2^-2 = 0.5 (the scale of lane 1 is not read);
    // the upper nibble of a scale byte is not read
    let wd = vec16(&[(0, 0x38), (1, 0x30)]);
    let ws = vec16(&[(0, 0xA3), (1, 0x05)]);
    let xd = vec16(&[(0, 0x40), (1, 0x30)]);
    let xs = vec16(&[(0, 0x0E), (1, 0x07)]);
    assert_eq!(dsum_round(&bfp8_exact(&wd, &ws, &xd, &xs)), 0x4090_0000); // 4.5

    // the largest element and scale on both operands: 448^2 * 2^14
    let big = vec16(&[(0, 0x7E)]);
    let sc = vec16(&[(0, 0x07)]);
    assert_eq!(dsum_round(&bfp8_exact(&big, &sc, &big, &sc)), 0x4F44_0000);
    // an e4m3 NaN gives the NaN
    let nan = vec16(&[(0, 0x7F)]);
    assert_eq!(dsum_round(&bfp8_exact(&nan, &sc, &big, &sc)), 0x7FC0_0000);
}

// model: unit_tests/test_ext_types.sail test_bfp8_second_block
#[test]
fn bfp8_second_block_uses_its_own_scale() {
    let mut wd = vec![0u8; 32];
    let mut ws = vec![0u8; 32];
    let mut xd = vec![0u8; 32];
    let mut xs = vec![0u8; 32];
    wd[16] = 0x38;
    ws[16] = 0x08; // -8
    xd[16] = 0x38;
    xs[16] = 0x07; // 7
    assert_eq!(dsum_round(&bfp8_exact(&wd, &ws, &xd, &xs)), 0x3F00_0000); // 2^-1
}
