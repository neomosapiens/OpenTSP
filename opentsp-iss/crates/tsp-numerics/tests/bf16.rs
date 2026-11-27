//! The extension type bf16 (`microarchitecture.md` sec. 5 and 6): the landmarks of `unit_tests/test_bf16.sail`, the
//! type code, and the exact matrix product `bf16_dot`.
// model: unit_tests/test_bf16.sail test_bf16_format, test_bf16_dot

use tsp_numerics::fp::{canon_nan, fdecode, inf_bits, zero_bits, FClass, FMT_BF16};
use tsp_numerics::lanes::{k_binary, k_cmp, k_convert_any, k_unary};
use tsp_numerics::mxm::{bf16_dot, f16_dot};
use tsp_numerics::EType;

// model: unit_tests/test_bf16.sail test_bf16_format
#[test]
fn type_code_and_width() {
    assert_eq!(EType::from_code(16), Some(EType::Bf16));
    assert_eq!(EType::Bf16.code(), 16);
    assert_eq!((EType::Bf16.width(), EType::Bf16.name(), EType::Bf16.is_float()), (2, "bf16", true));
    // the codes between the base set and bf16, and above it, are reserved
    for c in (13..16).chain(17..32) {
        assert_eq!(EType::from_code(c), None, "code {c}");
    }
    for (i, t) in EType::ALL.iter().chain(EType::EXTENSIONS.iter()).enumerate() {
        assert_eq!(t.index(), i);
        assert_eq!(EType::from_code(t.code()), Some(*t));
    }
}

// model: unit_tests/test_bf16.sail test_bf16_format
#[test]
fn format_landmarks() {
    let one = fdecode(FMT_BF16, 0x3F80);
    assert_eq!((one.cls, one.mag, one.e), (FClass::Finite, 128, -7));
    assert_eq!(fdecode(FMT_BF16, 0x7F80).cls, FClass::Inf);
    assert!(fdecode(FMT_BF16, 0xFF80).neg);
    assert_eq!(fdecode(FMT_BF16, 0x7FC1).cls, FClass::NaN);
    let tiny = fdecode(FMT_BF16, 0x0001);
    assert_eq!((tiny.mag, tiny.e), (1, -133));
    assert_eq!(canon_nan(FMT_BF16), 0x7FC0);
    assert_eq!((inf_bits(FMT_BF16, false), inf_bits(FMT_BF16, true)), (0x7F80, 0xFF80));
    assert_eq!((zero_bits(FMT_BF16, true), zero_bits(FMT_BF16, false)), (0x8000, 0));
}

// model: unit_tests/test_bf16.sail test_bf16_format
#[test]
fn conversions_and_arithmetic() {
    use EType::*;
    assert_eq!(k_convert_any(F32, Bf16, 0x3F80_0000), 0x3F80);
    assert_eq!(k_convert_any(F32, Bf16, 0x3F80_8000), 0x3F80); // the tie above 1.0 rounds to even (down)
    assert_eq!(k_convert_any(F32, Bf16, 0x3F81_8000), 0x3F82); // the tie above 0x3F81 rounds to even (up)
    assert_eq!(k_convert_any(F32, Bf16, 0x3F80_8001), 0x3F81);
    assert_eq!(k_convert_any(F32, Bf16, 0x7F7F_FFFF), 0x7F80); // the largest binary32 rounds to infinity
    assert_eq!(k_convert_any(F32, Bf16, 0x7FC0_0000), 0x7FC0);
    assert_eq!(k_convert_any(Bf16, F32, 0x3F80), 0x3F80_0000);
    assert_eq!(k_convert_any(Bf16, F32, 0x0001), 0x0001_0000);
    assert_eq!(k_convert_any(F32, Bf16, 0x0001_0000), 0x0001);
    assert_eq!(k_convert_any(Bf16, F16, 0x3F80), 0x3C00);
    assert_eq!(k_convert_any(I8, Bf16, 3), 0x4040);
    assert_eq!(k_convert_any(Bf16, I8, 0x4040), 3);
    assert_eq!(k_binary(0, Bf16, 0x3F80, 0x3F80), 0x4000);
    assert_eq!(k_binary(6, Bf16, 0x3FC0, 0x4000), 0x4040);
    assert_eq!(k_binary(0, Bf16, 0x3F80, 0x3B80), 0x3F80); // 1 + 2^-8 ties to even
    assert_eq!(k_binary(0, Bf16, 0x7F7F, 0x7F7F), 0x7F80);
    assert_eq!(k_binary(0, Bf16, 0x7FC0, 0x3F80), 0x7FC0);
    // exponents 260 apart: the smaller operand is rounded away (an `i128` alignment would overflow)
    assert_eq!(k_binary(0, Bf16, 0x0001, 0x3F80), 0x3F80);
    assert_eq!(k_binary(3, Bf16, 0x0001, 0x3F80), 0xBF80);
    assert_eq!(k_binary(0, Bf16, 0x0001, 0x7F7F), 0x7F7F);
    assert_eq!(k_binary(0, Bf16, 0x8001, 0x0001), 0x0000);
    assert_eq!((k_unary(15, Bf16, 0x3F80), k_unary(16, Bf16, 0xBF80)), (0xBF80, 0x3F80));
    assert_eq!(k_cmp(2, Bf16, 0xBF80, 0x3F80), 0xFFFF);
    assert_eq!(k_cmp(1, Bf16, 0x7FC0, 0x7FC0), 0xFFFF);
}

/// A vector of `n` lanes of which the first lanes are `v`.
fn lanes(v: &[u8], n: usize) -> Vec<u8> {
    let mut o = vec![0u8; n];
    o[..v.len()].copy_from_slice(v);
    o
}

// model: unit_tests/test_bf16.sail test_bf16_dot
#[test]
fn dot_product() {
    let n = 320;
    let dot = |w: &[(u8, u8)], x: &[(u8, u8)]| {
        let (wl, wh): (Vec<u8>, Vec<u8>) = w.iter().copied().unzip();
        let (xl, xh): (Vec<u8>, Vec<u8>) = x.iter().copied().unzip();
        bf16_dot(&lanes(&wl, n), &lanes(&wh, n), &lanes(&xl, n), &lanes(&xh, n))
    };
    // 1.5 * 2.0
    assert_eq!(dot(&[(0xC0, 0x3F)], &[(0x00, 0x40)]), 0x4040_0000);
    // [2^127, 1, -2^127] . [1, 1, 1] = 1: a binary32 accumulation gives 0
    assert_eq!(dot(&[(0x00, 0x7F), (0x80, 0x3F), (0x00, 0xFF)], &[(0x80, 0x3F); 3]), 0x3F80_0000);
    // the products are exact to 2^-266: 1 + 2^-12 * 2^-12 is a tie that rounds to 1; the smallest subnormal squared breaks it upward
    let tie = [(0x80, 0x3F), (0x80, 0x39)];
    assert_eq!(dot(&tie, &tie), 0x3F80_0000);
    let up = [(0x80, 0x3F), (0x80, 0x39), (0x01, 0x00)];
    assert_eq!(dot(&up, &up), 0x3F80_0001);
    // a product below 2^-259 is exact too: +(16 * 2^-133)^2 and -(2^-133 * 2^-125) are 2^-258 each and cancel, the tie stays a tie
    let w = [(0x80, 0x3F), (0x80, 0x39), (0x10, 0x00), (0x01, 0x00)];
    let x = [(0x80, 0x3F), (0x80, 0x39), (0x10, 0x00), (0x00, 0x81)];
    assert_eq!(dot(&w, &x), 0x3F80_0000);
    assert_eq!(dot(&[(0x01, 0x00)], &[(0x80, 0x3F)]), 0x0001_0000); // the smallest subnormal times 1.0
    assert_eq!(dot(&[(0x00, 0x7F)], &[(0x00, 0x7F)]), 0x7F80_0000); // 2^127 * 2^127 overflows
    assert_eq!(dot(&[(0x80, 0x7F)], &[(0x00, 0x00)]), 0x7FC0_0000); // infinity times zero
    assert_eq!(dot(&[(0x80, 0x7F)], &[(0x80, 0x3F)]), 0x7F80_0000);
    assert_eq!(dot(&[(0xC1, 0x7F)], &[(0x80, 0x3F)]), 0x7FC0_0000);
    // -inf + +inf
    assert_eq!(dot(&[(0x80, 0xFF), (0x80, 0x7F)], &[(0x80, 0x3F); 2]), 0x7FC0_0000);
    // zeros: -0 only when every product is -0
    assert_eq!(bf16_dot(&vec![0; n], &vec![0x80; n], &vec![0; n], &vec![0; n]), 0x8000_0000); // every product is -0 * +0
    assert_eq!(dot(&[(0x00, 0x80)], &[(0x80, 0x3F)]), 0x0000_0000); // -0 * 1.0 with +0 elsewhere
    assert_eq!(dot(&[(0x00, 0x80), (0x80, 0x3F)], &[(0x80, 0x3F), (0x00, 0x00)]), 0x0000_0000);
    // the f16 kernel is unchanged
    assert_eq!(f16_dot(&lanes(&[0x00], n), &lanes(&[0x3C], n), &lanes(&[0x00], n), &lanes(&[0x3C], n)), 0x3F80_0000);
}

#[test]
fn bf16_cross_limb_carry_and_borrow_spread_over_the_product_range() {
    let n = 512;
    let mut wlo = vec![0x7f; n];
    let mut whi = vec![0x7f; n];
    let mut xlo = vec![0x7f; n];
    let mut xhi = vec![0x7f; n];
    whi[n / 2..].fill(0xff);
    assert_eq!(bf16_dot(&wlo, &whi, &xlo, &xhi), 0);
    for k in [n / 2 - 1, n - 1] {
        wlo[k] = 1;
        whi[k] &= 0x80;
        xlo[k] = 0x80;
        xhi[k] = 0x3f;
    }
    wlo[n / 2 - 1] = 2;
    assert_eq!(bf16_dot(&wlo, &whi, &xlo, &xhi), 0x0001_0000);
}

#[test]
fn radix_bf16_matches_independent_signed_integer_sum_over_lengths_and_exponents() {
    use num_bigint::{BigInt, Sign};
    use tsp_numerics::bigfp::fround_big;
    use tsp_numerics::fp::FMT32;
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state as u16
    };
    for n in [0, 1, 2, 7, 8, 15, 16, 31, 64, 127, 319, 320, 511, 512, 513] {
        for seed in 0..64 {
            let mut sum = BigInt::from(0);
            let mut all_negzero = true;
            let (mut wlo, mut whi, mut xlo, mut xhi) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for lane in 0..n {
                let mut w = next();
                let mut x = next();
                if seed % 3 == 0 {
                    w &= !0x4000;
                    x &= !0x4000;
                }
                if w & 0x7f80 == 0x7f80 {
                    w &= !0x0080;
                }
                if x & 0x7f80 == 0x7f80 {
                    x &= !0x0080;
                }
                if seed % 7 == 0 && lane % 3 == 0 {
                    w &= 0x8000;
                }
                if seed % 8 == 0 && lane % 2 == 0 {
                    x &= 0x8000;
                }
                let (a, b) = (fdecode(FMT_BF16, u32::from(w)), fdecode(FMT_BF16, u32::from(x)));
                let neg = a.neg ^ b.neg;
                all_negzero &= neg && (a.cls == FClass::Zero || b.cls == FClass::Zero);
                if a.cls != FClass::Zero && b.cls != FClass::Zero {
                    let product = BigInt::from(a.mag * b.mag) << ((a.e + b.e + 266) as usize);
                    sum += if neg { -product } else { product };
                }
                let [lo, hi] = w.to_le_bytes();
                wlo.push(lo);
                whi.push(hi);
                let [lo, hi] = x.to_le_bytes();
                xlo.push(lo);
                xhi.push(hi);
            }
            let neg = sum.sign() == Sign::Minus;
            let expected = fround_big(FMT32, neg || all_negzero, &if neg { -sum } else { sum }, -266);
            assert_eq!(bf16_dot(&wlo, &whi, &xlo, &xhi), expected, "n={n} seed={seed}");
        }
    }
}

#[test]
fn prepared_bf16_matches_independent_signed_integer_sum_over_lengths_and_exponents() {
    use num_bigint::{BigInt, Sign};
    use tsp_numerics::bigfp::fround_big;
    use tsp_numerics::fp::FMT32;
    use tsp_numerics::mxm::{bf16_dot_prepared, Bf16Act};
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state as u16
    };
    for n in [0, 1, 2, 7, 8, 15, 16, 31, 64, 127, 319, 320, 511, 512, 513] {
        for seed in 0..64 {
            let mut sum = BigInt::from(0);
            let mut all_negzero = true;
            let (mut wlo, mut whi, mut xlo, mut xhi) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for lane in 0..n {
                let mut w = next();
                let mut x = next();
                if seed % 3 == 0 {
                    w &= !0x4000;
                    x &= !0x4000;
                }
                if w & 0x7f80 == 0x7f80 {
                    w &= !0x0080;
                }
                if x & 0x7f80 == 0x7f80 {
                    x &= !0x0080;
                }
                if seed % 7 == 0 && lane % 3 == 0 {
                    w &= 0x8000;
                }
                if seed % 8 == 0 && lane % 2 == 0 {
                    x &= 0x8000;
                }
                let (a, b) = (fdecode(FMT_BF16, u32::from(w)), fdecode(FMT_BF16, u32::from(x)));
                let neg = a.neg ^ b.neg;
                all_negzero &= neg && (a.cls == FClass::Zero || b.cls == FClass::Zero);
                if a.cls != FClass::Zero && b.cls != FClass::Zero {
                    let product = BigInt::from(a.mag * b.mag) << ((a.e + b.e + 266) as usize);
                    sum += if neg { -product } else { product };
                }
                let [lo, hi] = w.to_le_bytes();
                wlo.push(lo);
                whi.push(hi);
                let [lo, hi] = x.to_le_bytes();
                xlo.push(lo);
                xhi.push(hi);
            }
            let neg = sum.sign() == Sign::Minus;
            let expected = fround_big(FMT32, neg || all_negzero, &if neg { -sum } else { sum }, -266);
            assert_eq!(bf16_dot_prepared(&wlo, &whi, &Bf16Act::new(&xlo, &xhi)), expected, "n={n} seed={seed}");
        }
    }
}

#[test]
fn bf16_direct_rounding_keeps_distant_sticky_bits_and_f32_boundaries() {
    use tsp_numerics::mxm::{bf16_dot_prepared, Bf16Act};

    let power = |exponent: u16| exponent << 7;
    let check = |weight: [u16; 3], activation: [u16; 3], expected: u32| {
        let wlo = weight.map(|x| x as u8);
        let whi = weight.map(|x| (x >> 8) as u8);
        let xlo = activation.map(|x| x as u8);
        let xhi = activation.map(|x| (x >> 8) as u8);
        let prepared = Bf16Act::new(&xlo, &xhi);
        assert_eq!(bf16_dot(&wlo, &whi, &xlo, &xhi), expected);
        assert_eq!(bf16_dot_prepared(&wlo, &whi, &prepared), expected);
    };

    check([power(51), 0, 0], [power(53), 0, 0], 0);
    check([power(51), 1, 0], [power(53), 1, 0], 1);
    check([0x3f80, 0x3980, 0], [0x3f80, 0x3980, 0], 0x3f80_0000);
    check([0x3f80, 0x3980, 1], [0x3f80, 0x3980, 1], 0x3f80_0001);
    check([0x7f7f, 0, 0], [0x3f80, 0, 0], 0x7f7f_0000);
    check([0x7f7f, 0, 0], [0x4000, 0, 0], 0x7f80_0000);
    check([0x8000; 3], [0x3f80; 3], 0x8000_0000);
}

#[test]
fn packed_bf16_matches_independent_signed_integer_sum_over_lengths_and_exponents() {
    use num_bigint::{BigInt, Sign};
    use tsp_numerics::bigfp::fround_big;
    use tsp_numerics::fp::FMT32;
    use tsp_numerics::mxm::{Bf16Act, Bf16Weights};
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state as u16
    };
    for n in [0, 1, 2, 7, 8, 15, 16, 31, 64, 127, 319, 320, 511, 512, 513] {
        for seed in 0..64 {
            let mut sum = BigInt::from(0);
            let mut all_negzero = true;
            let (mut wlo, mut whi, mut xlo, mut xhi) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for lane in 0..n {
                let mut w = next();
                let mut x = next();
                if seed % 3 == 0 {
                    w &= !0x4000;
                    x &= !0x4000;
                }
                if w & 0x7f80 == 0x7f80 {
                    w &= !0x0080;
                }
                if x & 0x7f80 == 0x7f80 {
                    x &= !0x0080;
                }
                if seed % 7 == 0 && lane % 3 == 0 {
                    w &= 0x8000;
                }
                if seed % 8 == 0 && lane % 2 == 0 {
                    x &= 0x8000;
                }
                let (a, b) = (fdecode(FMT_BF16, u32::from(w)), fdecode(FMT_BF16, u32::from(x)));
                let neg = a.neg ^ b.neg;
                all_negzero &= neg && (a.cls == FClass::Zero || b.cls == FClass::Zero);
                if a.cls != FClass::Zero && b.cls != FClass::Zero {
                    let product = BigInt::from(a.mag * b.mag) << ((a.e + b.e + 266) as usize);
                    sum += if neg { -product } else { product };
                }
                let [lo, hi] = w.to_le_bytes();
                wlo.push(lo);
                whi.push(hi);
                let [lo, hi] = x.to_le_bytes();
                xlo.push(lo);
                xhi.push(hi);
            }
            let neg = sum.sign() == Sign::Minus;
            let expected = fround_big(FMT32, neg || all_negzero, &if neg { -sum } else { sum }, -266);
            let mut planes = [[0u8; 513]; 2];
            planes[0][..n].copy_from_slice(&wlo);
            planes[1][..n].copy_from_slice(&whi);
            let prepared = Bf16Weights::new(&planes, 1, n);
            assert_eq!(prepared.dot(0, &wlo, &whi, &Bf16Act::new(&xlo, &xhi)), expected, "n={n} seed={seed}");
        }
    }
}
