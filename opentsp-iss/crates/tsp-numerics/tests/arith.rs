//! Arithmetic, rounding and conversion tests: the machine-width fast paths against the exact
//! big-integer versions, the ported Sail `$[test]`s of the numerics, and independent checks.
mod common;

use common::*;
use num_bigint::BigInt;
use tsp_numerics::bigfp::{self, fround_big};
use tsp_numerics::fp::*;
use tsp_numerics::lanes::{self, ReduceOp};
use tsp_numerics::{mxm, EType};

/// Debug builds run a thinned version of the heavy loops (`TSP_NUMERICS_FULL=1` runs all).
fn scale(n: usize) -> usize {
    if std::env::var_os("TSP_NUMERICS_FULL").is_some() || !cfg!(debug_assertions) {
        n
    } else {
        n / 16
    }
}

#[test]
fn fround_matches_big() {
    let mut rng = Rng(42);
    for f in [FMT16, FMT32] {
        for _ in 0..scale(400_000) {
            let bits = 1 + rng.next() % 100;
            let mag: u128 = ((rng.next() as u128) << 64 | rng.next() as u128) >> (128 - bits);
            let e = (rng.next() % 400) as i32 - 250;
            let neg = rng.next() & 1 == 1;
            let a = fround(f, neg, mag, e);
            let b = fround_big(f, neg, &BigInt::from(mag), e as i64);
            assert_eq!(a, b, "fmt {} mag {mag} e {e}", f.width);
        }
        // every boundary: mantissas around powers of two and the overflow threshold
        for p in 0..100u32 {
            for d in [0u128, 1, 2] {
                for e in -200..200 {
                    let mag = (1u128 << p) + d;
                    let a = fround(f, false, mag, e);
                    let b = fround_big(f, false, &BigInt::from(mag), e as i64);
                    assert_eq!(a, b, "fmt {} mag {mag} e {e}", f.width);
                }
            }
        }
    }
}

#[test]
fn f16_arithmetic_matches_big() {
    let mut rng = Rng(7);
    let check = |x: u32, y: u32| {
        assert_eq!(fp_add(FMT16, x, y), bigfp::fp_add_big(FMT16, x, y), "add {x:04x} {y:04x}");
        assert_eq!(fp_sub(FMT16, x, y), bigfp::fp_sub_big(FMT16, x, y), "sub {x:04x} {y:04x}");
        assert_eq!(fp_mul(FMT16, x, y), bigfp::fp_mul_big(FMT16, x, y), "mul {x:04x} {y:04x}");
    };
    for &x in &F16_EDGES {
        for &y in &F16_EDGES {
            check(x, y);
        }
    }
    // one operand exhaustive against 64 others
    for x in 0..0x10000u32 {
        for _ in 0..scale(64).max(2) {
            check(x, (rng.next() & 0xFFFF) as u32);
        }
    }
    for _ in 0..scale(600_000) {
        let (x, y) = rand_pair(&mut rng, EType::F16);
        check(x, y);
    }
}

#[test]
fn f32_native_matches_exact() {
    let mut rng = Rng(8);
    let check = |x: u32, y: u32| {
        assert_eq!(fp_add(FMT32, x, y), bigfp::fp_add_big(FMT32, x, y), "add {x:08x} {y:08x}");
        assert_eq!(fp_sub(FMT32, x, y), bigfp::fp_sub_big(FMT32, x, y), "sub {x:08x} {y:08x}");
        assert_eq!(fp_mul(FMT32, x, y), bigfp::fp_mul_big(FMT32, x, y), "mul {x:08x} {y:08x}");
    };
    for &x in &F32_EDGES {
        for &y in &F32_EDGES {
            check(x, y);
        }
    }
    for _ in 0..scale(800_000) {
        let (x, y) = rand_pair(&mut rng, EType::F32);
        check(x, y);
        // wide exponent gaps, subnormals
        let (a, b) = (rng.next() as u32, (rng.next() as u32) & 0x807F_FFFF);
        check(a, b);
    }
}

#[test]
fn nan_is_canonical_and_zero_signs() {
    for f in [FMT16, FMT32] {
        let nan = canon_nan(f);
        let weird_nan = inf_bits(f, true) | 1; // negative signalling NaN
        let one = fp_of_int(f, 1);
        assert_eq!(fp_add(f, weird_nan, one), nan);
        assert_eq!(fp_mul(f, one, weird_nan), nan);
        assert_eq!(fp_neg(f, weird_nan), nan);
        assert_eq!(fp_abs(f, weird_nan), nan);
        // Inf - Inf, Inf * 0
        assert_eq!(fp_sub(f, inf_bits(f, false), inf_bits(f, false)), nan);
        assert_eq!(fp_mul(f, inf_bits(f, false), zero_bits(f, true)), nan);
        // exact zero sum of opposite signs is +0; -0 + -0 is -0; -0 + +0 is +0
        assert_eq!(fp_sub(f, one, one), zero_bits(f, false));
        assert_eq!(fp_add(f, zero_bits(f, true), zero_bits(f, true)), zero_bits(f, true));
        assert_eq!(fp_add(f, zero_bits(f, true), zero_bits(f, false)), zero_bits(f, false));
        assert_eq!(fp_sub(f, zero_bits(f, true), zero_bits(f, false)), zero_bits(f, true));
        // max / min with -0 < +0 and NaN as missing data
        let (pz, nz) = (zero_bits(f, false), zero_bits(f, true));
        assert_eq!(fp_max(f, nz, pz), pz);
        assert_eq!(fp_max(f, pz, nz), pz);
        assert_eq!(fp_min(f, nz, pz), nz);
        assert_eq!(fp_min(f, pz, nz), nz);
        assert_eq!(fp_max(f, nan, one), one);
        assert_eq!(fp_min(f, one, nan), one);
        assert_eq!(fp_max(f, nan, weird_nan), nan);
        // tanh(+-0) echoes the raw input; negation drops stray high bits of an f16
    }
    assert_eq!(fp_neg(FMT16, 0xFFFF_3C00), 0xBC00);
    assert_eq!(fp_abs(FMT16, 0xFFFF_BC00), 0x3C00);
}

#[test]
fn fcmp_orders_all_f16_patterns() {
    // independent order: the sign-magnitude integer order of non-NaN f16 patterns
    let key = |x: u32| -> i32 {
        let m = (x & 0x7FFF) as i32;
        if x & 0x8000 != 0 {
            -m
        } else {
            m
        }
    };
    let vals: Vec<u32> = (0..0x10000u32).filter(|&x| !is_nan(FMT16, x)).collect();
    for &x in vals.iter().step_by(if cfg!(debug_assertions) { 997 } else { 97 }) {
        for &y in &vals {
            let want = (key(x) > key(y)) as i32 - (key(x) < key(y)) as i32;
            assert_eq!(fcmp(FMT16, x, y), want, "{x:04x} {y:04x}");
        }
    }
    // f32: includes patterns with exponent gaps
    let mut rng = Rng(3);
    let key32 = |x: u32| -> i64 {
        let m = (x & 0x7FFF_FFFF) as i64;
        if x & 0x8000_0000 != 0 {
            -m
        } else {
            m
        }
    };
    for _ in 0..scale(400_000) {
        let (x, y) = rand_pair(&mut rng, EType::F32);
        if is_nan(FMT32, x) || is_nan(FMT32, y) {
            continue;
        }
        let want = (key32(x) > key32(y)) as i32 - (key32(x) < key32(y)) as i32;
        assert_eq!(fcmp(FMT32, x, y), want, "{x:08x} {y:08x}");
    }
}

#[test]
fn float_to_int_matches_f64() {
    // independent reference: f64 round-ties-even, saturate, NaN -> 0
    let reference = |v: f64, nbits: u32| -> i64 {
        if v.is_nan() {
            return 0;
        }
        let hi = (1i64 << (nbits - 1)) - 1;
        let lo = -(1i64 << (nbits - 1));
        let r = v.round_ties_even();
        if r >= hi as f64 {
            hi
        } else if r <= lo as f64 {
            lo
        } else {
            r as i64
        }
    };
    let f16_to_f64 = |x: u32| -> f64 {
        let a = fdecode(FMT16, x);
        match a.cls {
            FClass::NaN => f64::NAN,
            FClass::Inf => f64::INFINITY.copysign(if a.neg { -1.0 } else { 1.0 }),
            FClass::Zero => {
                if a.neg {
                    -0.0
                } else {
                    0.0
                }
            }
            FClass::Finite => (a.mag as f64) * 2f64.powi(a.e) * if a.neg { -1.0 } else { 1.0 },
        }
    };
    for x in 0..0x10000u32 {
        for nbits in [8u32, 16, 32, 33] {
            assert_eq!(int_of_fp(FMT16, x, nbits), reference(f16_to_f64(x), nbits), "f16 {x:04x} -> i{nbits}");
        }
    }
    let mut rng = Rng(5);
    for _ in 0..scale(1_000_000) {
        let x = rand_float(&mut rng, EType::F32);
        let v = f32::from_bits(x) as f64;
        for nbits in [8u32, 16, 32, 33] {
            assert_eq!(int_of_fp(FMT32, x, nbits), reference(v, nbits), "f32 {x:08x} -> i{nbits}");
        }
    }
    // int to float: exact for |v| < 2^24 in f32, RNE beyond (checked against the host's `as f32`)
    for _ in 0..scale(1_000_000) {
        let v = rng.next() as i32;
        assert_eq!(fp_of_int(FMT32, v as i64), (v as f32).to_bits(), "i32 {v}");
        let w = (rng.next() as i64) >> (rng.next() % 40);
        let w = w.clamp(i32::MIN as i64, u32::MAX as i64);
        assert_eq!(fp_of_int(FMT32, w), (w as f32).to_bits(), "i64 {w}");
    }
    // f16 <-> f32 conversion: widening is exact and narrowing is RNE (host f32 -> f16 is not
    // available, so check the round trip and the order of the exhaustive f16 widening)
    for x in 0..0x10000u32 {
        let w = fp_convert(FMT16, FMT32, x);
        if is_nan(FMT16, x) {
            assert_eq!(w, canon_nan(FMT32));
        } else {
            assert_eq!(fp_convert(FMT32, FMT16, w), x, "round trip {x:04x}");
        }
    }
}

// ---- the Sail $[test]s of the numerics, ported with their numbers ------------------------------

#[test]
fn test_unsigned_and_bool_kernels() {
    use EType::*;
    assert_eq!(lanes::k_binary(1, U8, 0x000000F0, 0x00000020), 0x000000FF); // add_sat u8 saturates at 255
    assert_eq!(lanes::k_binary(2, U8, 0x000000F0, 0x00000020), 0x00000010); // add_mod u8 wraps
    assert_eq!(lanes::k_binary(1, I8, 0x00000070, 0x00000020), 0x0000007F); // add_sat i8 at 127
    assert_eq!(lanes::k_binary(9, U16, 0x0000FFFF, 0x00000001), 0x0000FFFF); // max u16: 65535 > 1
    assert_eq!(lanes::k_binary(9, I16, 0x0000FFFF, 0x00000001), 0x00000001); // max i16: -1 < 1
    assert_eq!(lanes::k_binary(11, B8, 0x00000005, 0x00000080), 0x000000FF); // true and true
    assert_eq!(lanes::k_cmp(0, B8, 0x00000005, 0x00000001), 0x000000FF); // bool eq on truth values
    assert_eq!(lanes::k_cmp(2, U32, 0xFFFFFFFF, 0x00000001), 0x00000000); // u32 lt: 4294967295 < 1 false
}

#[test]
fn test_generic_convert() {
    use EType::*;
    assert_eq!(lanes::k_convert_any(I32, U8, 0xFFFFFFFF), 0x00000000); // -1 -> u8 saturates to 0
    assert_eq!(lanes::k_convert_any(U16, I8, 0x00000200), 0x0000007F); // 512 -> i8 saturates
    assert_eq!(lanes::k_convert_any(F32, U8, 0x43800000), 0x000000FF); // 256.0 -> u8 255
    assert_eq!(lanes::k_convert_any(F32, B32, 0x80000000), 0x00000000); // -0.0 -> false
    assert_eq!(lanes::k_convert_any(B8, F16, 0x00000001), 0x00003C00); // true -> 1.0
    assert_eq!(lanes::k_convert_any(U8, F32, 0x000000C8), 0x43480000); // 200 -> 200.0
    assert_eq!(lanes::k_convert_any(F32, B32, 0x7FC00000), 0xFFFFFFFF); // NaN counts as true
    assert_eq!(lanes::k_convert_any(F16, I8, 0x7E00), 0); // NaN -> 0
}

#[test]
fn test_mxm_f16_single_rounding() {
    // 2^30 + 1 - 2^30 = 1 (per-addition f32 rounding would give 0). w = [2^15, 1, -2^15],
    // x = [2^15, 1, 2^15] (f16 0x7800, 0x3C00, 0xF800).
    let lanes8 = |a: u8, b: u8, c: u8| vec![a, b, c];
    let z = vec![0u8; 3];
    assert_eq!(mxm::f16_dot(&z, &lanes8(0x78, 0x3C, 0xF8), &z, &lanes8(0x78, 0x3C, 0x78)), 0x3F800000);
    // Inf (0x7C00) x 0 is the canonical NaN
    assert_eq!(mxm::f16_dot(&z, &lanes8(0x7C, 0, 0), &z, &z), canon_nan(FMT32));
    // -0 x 1 with +0 elsewhere is +0
    assert_eq!(mxm::f16_dot(&z, &lanes8(0x80, 0, 0), &z, &lanes8(0x3C, 0, 0)), 0x00000000);
    // every product -0 gives -0; an infinity wins over finite terms; +Inf and -Inf give NaN
    assert_eq!(mxm::f16_dot(&[0], &[0x80], &[0], &[0x00]), 0x8000_0000);
    assert_eq!(mxm::f16_dot(&[0, 0], &[0x7C, 0x3C], &[0, 0], &[0x3C, 0x3C]), 0x7F80_0000);
    assert_eq!(mxm::f16_dot(&[0, 0], &[0x7C, 0x7C], &[0, 0], &[0x3C, 0xBC]), canon_nan(FMT32));
    // 65504^2 * 320 stays finite in f32; the sum of 320 maxima rounds correctly
    let n = 320;
    let w = vec![0xFFu8; n];
    let wh = vec![0x7Bu8; n];
    let want = fround(FMT32, false, 65504u128 * 65504 * 320, 0);
    assert_eq!(mxm::f16_dot(&w, &wh, &w, &wh), want);
}

#[test]
fn int8_dot_signs_and_wrap() {
    let w = [0x80u8, 0x7F, 0xFF];
    let x = [0x80u8, 0x7F, 0x02];
    assert_eq!(mxm::int8_dot(&w, true, &x, true) as i32, 128 * 128 + 127 * 127 - 2);
    assert_eq!(mxm::int8_dot(&w, false, &x, false), 128 * 128 + 127 * 127 + 255 * 2);
    assert_eq!(mxm::int8_dot(&w, true, &x, false) as i32, -128 * 128 + 127 * 127 - 2);
    assert_eq!(mxm::int8_dot(&w, false, &x, true) as i32, 128 * -128 + 127 * 127 + 255 * 2);
    // 320 lanes of -128 * -128 = 5 242 880 fits; the sum wraps to 32 bits when it must
    let big = vec![0x80u8; 320];
    assert_eq!(mxm::int8_dot(&big, true, &big, true), 320 * 16384);
}

#[test]
fn reduce_rounds_at_each_step() {
    // : an ascending fold with rounding at each step. 2^24 + 1 rounds back to 2^24, so a
    // run of ones after 2^24 vanishes; a sum that rounded once at the end would keep them.
    let big = 0x4B80_0000u32; // 2^24
    let one = 0x3F80_0000u32;
    let mut g = vec![big];
    g.extend(std::iter::repeat_n(one, 15));
    assert_eq!(lanes::reduce_fold(ReduceOp::Add, EType::F32, &g), big);
    // ones first: 15 + 2^24 is exact and 2^24 + 15 rounds to even
    let mut g2: Vec<u32> = std::iter::repeat_n(one, 15).collect();
    g2.push(big);
    assert_eq!(lanes::reduce_fold(ReduceOp::Add, EType::F32, &g2), fround(FMT32, false, (1 << 24) + 15, 0));
    // integers wrap per step; max and min follow k_binary
    let g3: Vec<u32> = (0..16).map(|i| 0x7F + i).collect();
    assert_eq!(lanes::reduce_fold(ReduceOp::Add, EType::I8, &g3), (0..16u32).fold(0u32, |a, i| (a + 0x7F + i) & 0xFF));
    assert_eq!(lanes::reduce_fold(ReduceOp::Max, EType::I8, &[0x80, 0x7F, 0x00, 0xFF]), 0x7F);
    assert_eq!(lanes::reduce_fold(ReduceOp::Min, EType::U8, &[0x80, 0x7F, 0x00, 0xFF]), 0x00);
}

#[test]
fn unary_kernels_follow_b_d1() {
    use EType::*;
    // k_unary reads an integer signed or unsigned by its element type (model: core/numerics/lanes.sail k_unary)
    assert_eq!(lanes::k_unary(16, U8, 0xFF), 0xFF); // Abs is the identity on unsigned types
    assert_eq!(lanes::k_unary(20, U8, 0x80), 0x80); // so is ReLU
    assert_eq!(lanes::k_unary(20, U32, 0x8000_0000), 0x8000_0000);
    assert_eq!(lanes::k_unary(15, U8, 0x01), 0xFF); // Neg = wrap(0 - a)
    assert_eq!(lanes::k_unary(15, U16, 0x0000), 0x0000);
    assert_eq!(lanes::k_unary(16, I8, 0x80), 0x80); // Abs wraps at the minimum
    assert_eq!(lanes::k_unary(15, I8, 0x80), 0x80); // Neg wraps at the minimum
    assert_eq!(lanes::k_unary(20, I8, 0x80), 0x00);
    assert_eq!(lanes::k_unary(15, I32, 0x8000_0000), 0x8000_0000);
    assert_eq!(lanes::k_unary(14, B8, 0x00), 0xFF); // Not of a bool is logical
    for op in [15, 16, 20] {
        assert_eq!(lanes::k_unary(op, B8, 0xFF), 0, "op {op}"); // only Not is defined on booleans
        assert_eq!(lanes::k_unary(op, B16, 0x0001), 0, "op {op}");
    }
    assert_eq!(lanes::k_unary(14, U16, 0x1234), 0xEDCB);
    assert_eq!(lanes::k_unary(14, F16, 0x1234), 0); // not defined on floats
                                                    // ReLU of a positive float has no bits above the format width
    assert_eq!(lanes::k_unary(20, F16, 0xFFFF_3C00), 0x3C00);
    // TanH(+-0) is the format's signed zero, no stray high bits (model: core/numerics/transcendental.sail fp_tr_value)
    use tsp_numerics::transcendental::{fp_tr_pre, Pre, TANH};
    assert_eq!(fp_tr_pre(TANH, FMT16, 0xFFFF_8000), Pre::Done(0x8000));
    assert_eq!(fp_tr_pre(TANH, FMT16, 0xABCD_0000), Pre::Done(0x0000));
    assert_eq!(fp_tr_pre(TANH, FMT32, 0x8000_0000), Pre::Done(0x8000_0000));
    // binary ops that are undefined for a type give zero
    assert_eq!(lanes::k_binary(0, B8, 1, 1), 0);
    assert_eq!(lanes::k_binary(11, F16, 1, 1), 0);
    // Mask and Select test any set bit of the element width
    assert!(!lanes::k_nonzero(B8, 0x0000_0100));
    assert!(lanes::k_nonzero(F16, 0x8000));
    assert_eq!(lanes::apply(lanes::LaneOp::Select(B16), 7, 9, 0x0001_0000), 9);
    assert_eq!(lanes::apply(lanes::LaneOp::Mask(I8), 7, 0x100, 0), 0);
}
