//! Edge cases of the numerics that the exhaustive and differential tests do not reach: the names
//! and properties of the element types, kernels given an opcode outside their table (the
//! result is 0), the points where the `f64` fast path of the transcendentals declines
//! (a zero or non-finite `f64` value, an `f64` subnormal) and the straight port takes over, the
//! integer square root and the rounding shift of the port, and `fround` with the whole `u128`
//! range of mantissas against the exact big-integer version.
mod common;

use common::*;
use num_bigint::BigInt;
use tsp_numerics::bigfp::fround_big;
use tsp_numerics::fp::{fdecode, fround, FClass, FVal, FMT16, FMT32};
use tsp_numerics::lanes;
use tsp_numerics::transcendental::{self as tr, EXP, LOG, RSQRT, TANH};
use tsp_numerics::{ziv, EType};

#[test]
fn element_types_have_names_codes_and_widths() {
    let names = ["i8", "f16", "i32", "f32", "u8", "i16", "u16", "u32", "b8", "b16", "b32", "e4m3", "e5m2"];
    let widths = [1, 2, 4, 4, 1, 2, 2, 4, 1, 2, 4, 1, 1];
    for (c, t) in EType::ALL.iter().enumerate() {
        assert_eq!(t.name(), names[c]);
        assert_eq!(t.code() as usize, c);
        assert_eq!(EType::from_code(c as u8), Some(*t));
        assert_eq!(t.width(), widths[c]);
        assert_eq!(t.nbits(), 8 * widths[c]);
        assert_eq!(t.is_unsigned(), ["u8", "u16", "u32"].contains(&names[c]));
        assert_eq!(t.is_bool(), names[c].starts_with('b'));
        assert_eq!(t.is_float(), names[c].starts_with('f') || names[c].starts_with('e'));
    }
    // 13..15 and 17..31 are reserved; 16 is the extension type bf16
    for c in (13..=15).chain(17..=31) {
        assert_eq!(EType::from_code(c), None, "code {c} is not a type");
    }
    assert_eq!(EType::from_code(16), Some(EType::Bf16));
}

#[test]
fn kernels_given_an_unknown_opcode_give_zero() {
    for t in EType::ALL {
        let ones = lanes::all_ones(t);
        assert_eq!(lanes::k_binary(99, t, ones, 1), 0, "binary {t:?}");
        assert_eq!(lanes::k_unary(99, t, ones), 0, "unary {t:?}");
    }
    // an opcode of the other table of the same type class
    assert_eq!(lanes::k_binary(14, EType::B8, 1, 1), 0);
    assert_eq!(lanes::k_unary(11, EType::B8, 1), 0);
    assert_eq!(lanes::k_unary(11, EType::F16, 0x3c00), 0);
    assert_eq!(lanes::k_convert(99, 0x3c00), 0);
    assert_eq!(lanes::k_convert(24, 0x3c00), 0);
    assert_eq!(lanes::k_convert(33, 0x3c00), 0);
}

fn finite(neg: bool, mag: u64, e: i32) -> FVal {
    FVal { cls: FClass::Finite, neg, mag, e }
}

#[test]
fn the_fast_path_declines_where_the_f64_value_is_not_usable() {
    // log(1) = 0 exactly: no rounding position to test
    let one = fdecode(FMT32, 1.0f32.to_bits());
    assert_eq!(tr::tr_fast(LOG, FMT32, &one), None);
    // exp(-740) is an f64 subnormal: declined, the port answers with the f32 zero it rounds to
    let small = finite(true, 740, 0);
    assert_eq!(tr::tr_fast(EXP, FMT32, &small), None);
    assert_eq!(tr::tr_eval(EXP, FMT32, &small), Some(0));
    assert_eq!(tr::tr_eval(EXP, FMT32, &small), ziv::tr_eval(EXP, FMT32, &small));
    // exp(800) overflows the f64: declined, and the port says +Inf for both formats
    let big = finite(false, 800, 0);
    assert_eq!(tr::tr_fast(EXP, FMT32, &big), None);
    assert_eq!(tr::tr_eval(EXP, FMT32, &big), Some(0x7f80_0000));
    assert_eq!(tr::tr_eval(EXP, FMT16, &big), Some(0x7c00));
    // the integer form: exp(800) is not an f64 either
    assert_eq!(tr::tr_fast_int(EXP, &big), None);
    // an f16 and an f32 input whose value in the f64 is exact are accepted far from a boundary
    assert!(tr::tr_fast(TANH, FMT16, &fdecode(FMT16, 0x3555)).is_some());
    assert!(tr::tr_fast_int(TANH, &finite(false, 100, 0)).is_some());
}

#[test]
fn the_integer_fast_path_declines_a_result_beyond_the_scaled_range() {
    // exp(100) = 2.7e43 does not fit an i64 (and saturates a u128): the answer is "decline", never a wrapped
    // value. The 2^-40 filter already declines every value from 2^39 up; the explicit bound (2^62) keeps the
    // conversion safe even if the filter changes.
    assert_eq!(tr::tr_fast_int(EXP, &finite(false, 100, 0)), None);
    assert_eq!(tr::tr_fast_int(EXP, &finite(false, 44, 0)), None, "exp(44) = 1.3e19 > 2^62");
    // a large result inside the bound is still answered, and correctly
    let r = tr::tr_fast_int(EXP, &finite(false, 20, 0));
    assert_eq!(r, ziv::tr_eval_int(EXP, &finite(false, 20, 0)));
    assert_eq!(r, Some(485_165_195));
}

#[test]
fn memoised_f16_table_is_used_only_for_16_bit_patterns() {
    // an f16 element is the low 16 bits of a lane: bits above them are not a table index
    for op in [TANH, EXP, RSQRT, LOG] {
        for x in [0x3c00u32, 0x4000, 0x1_3c00, 0xffff_4000] {
            assert_eq!(tr::fp_transcendental(op, FMT16, x), tr::fp_tr_value(op, FMT16, x).unwrap(), "op {op} x {x:x}");
            assert_eq!(
                ziv::fp_transcendental(op, FMT16, x),
                ziv::fp_tr_value(op, FMT16, x).unwrap(),
                "ziv op {op} x {x:x}"
            );
        }
        for x in [0x3f80_0000u32, 0x4000_0000, 0x3eaa_aaab] {
            assert_eq!(
                tr::fp_transcendental(op, FMT32, x),
                ziv::fp_transcendental(op, FMT32, x),
                "op {op} f32 x {x:x}"
            );
        }
    }
}

#[test]
fn integer_square_root_and_rounding_shift_of_the_port() {
    for (n, r) in [
        (0i64, 0i64),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 2),
        (15, 3),
        (16, 4),
        (1 << 40, 1 << 20),
        ((1 << 40) - 1, (1 << 20) - 1),
    ] {
        assert_eq!(ziv::isqrt(&BigInt::from(n)), BigInt::from(r), "isqrt {n}");
    }
    // not a positive number: 0
    assert_eq!(ziv::isqrt(&BigInt::from(-9)), BigInt::from(0));
    // a left shift is exact; a right shift rounds to nearest, ties to even, for either sign
    assert_eq!(ziv::rne_int(&BigInt::from(5), 3), BigInt::from(40));
    assert_eq!(ziv::rne_int(&BigInt::from(-5), 2), BigInt::from(-20));
    for (v, sh, want) in [
        (5i64, -1, 2),
        (7, -1, 4),
        (6, -2, 2),
        (10, -2, 2),
        (11, -2, 3),
        (-5, -1, -2),
        (-7, -1, -4),
        (-6, -2, -2),
        (0, -5, 0),
    ] {
        assert_eq!(ziv::rne_int(&BigInt::from(v), sh), BigInt::from(want), "rne_int {v} 2^{sh}");
    }
}

#[test]
fn fround_covers_every_mantissa_width_against_the_exact_version() {
    let mut rng = Rng(7);
    for f in [FMT16, FMT32] {
        for bits in 1..=127u32 {
            for _ in 0..40 {
                let mag: u128 =
                    (((rng.next() as u128) << 64) | rng.next() as u128) >> (128 - bits) | 1u128 << (bits - 1);
                for e in [-420, -300, -200, -150, -128, -100, -50, -20, 0, 20, 100] {
                    let neg = rng.next() & 1 == 1;
                    assert_eq!(
                        fround(f, neg, mag, e),
                        fround_big(f, neg, &BigInt::from(mag), e as i64),
                        "fmt {} bits {bits} e {e}",
                        f.width
                    );
                }
            }
            // the tie just above and below 2^127, shifted out entirely
            for d in [0u128, 1] {
                let mag = (1u128 << 127) + d;
                for e in -400..-100 {
                    assert_eq!(
                        fround(f, false, mag, e),
                        fround_big(f, false, &BigInt::from(mag), e as i64),
                        "fmt {} 2^127+{d} e {e}",
                        f.width
                    );
                }
            }
        }
    }
}
