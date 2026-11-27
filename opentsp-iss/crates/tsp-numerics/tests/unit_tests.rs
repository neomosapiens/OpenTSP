//! Ports of the numerics unit tests of the Sail model (`unit_tests/test_slices.sail`): the unary
//! kernels on unsigned and boolean elements, the float kernels' bits above the format width, and
//! the convergence of the transcendentals' Ziv loop.
use tsp_numerics::fp::{FMT16, FMT32};
use tsp_numerics::lanes::k_unary;
use tsp_numerics::transcendental::{self as tr, EXP, LOG, RSQRT, TANH};
use tsp_numerics::ziv;
use tsp_numerics::EType::*;

// The unary kernels read an unsigned element as unsigned and
// a boolean as a truth value. (No instruction encodes Neg, Abs or ReLU on an unsigned or boolean
// type, `unop_legal`, so these are kernel-level tests; the kernel is total and defined.)
// model: unit_tests/test_slices.sail test_unary_kernels_unsigned_bool
#[test]
fn test_unary_kernels_unsigned_bool() {
    // Abs and ReLU of an unsigned value are the value itself.
    assert_eq!(k_unary(16, U8, 0x0000_00FF), 0x0000_00FF);
    assert_eq!(k_unary(16, U8, 0x0000_0080), 0x0000_0080);
    assert_eq!(k_unary(20, U8, 0x0000_0080), 0x0000_0080);
    assert_eq!(k_unary(20, U8, 0x0000_00FF), 0x0000_00FF);
    assert_eq!(k_unary(16, U16, 0x0000_FFFE), 0x0000_FFFE);
    assert_eq!(k_unary(20, U16, 0x0000_8000), 0x0000_8000);
    assert_eq!(k_unary(16, U32, 0xFFFF_FFFF), 0xFFFF_FFFF);
    assert_eq!(k_unary(20, U32, 0x8000_0000), 0x8000_0000);
    // Neg of an unsigned value is the modular negation (2^n - x).
    assert_eq!(k_unary(15, U8, 0x0000_0001), 0x0000_00FF);
    assert_eq!(k_unary(15, U8, 0x0000_0000), 0x0000_0000);
    assert_eq!(k_unary(15, U8, 0x0000_0080), 0x0000_0080);
    assert_eq!(k_unary(15, U16, 0x0000_0001), 0x0000_FFFF);
    assert_eq!(k_unary(15, U32, 0x0000_0002), 0xFFFF_FFFE);
    // Not is a bitwise complement inside the element width.
    assert_eq!(k_unary(14, U8, 0x0000_00F0), 0x0000_000F);
    assert_eq!(k_unary(14, U16, 0x0000_FF00), 0x0000_00FF);
    // Signed types are unchanged.
    assert_eq!(k_unary(16, I8, 0x0000_00FF), 0x0000_0001);
    assert_eq!(k_unary(16, I8, 0x0000_0080), 0x0000_0080);
    assert_eq!(k_unary(20, I8, 0x0000_0080), 0x0000_0000);
    assert_eq!(k_unary(20, I16, 0x0000_8001), 0x0000_0000);
    assert_eq!(k_unary(15, I8, 0x0000_0001), 0x0000_00FF);
    // Booleans: only Not is defined (logical, canonical); every other unary kernel gives 0.
    assert_eq!(k_unary(14, B8, 0x0000_0000), 0x0000_00FF);
    assert_eq!(k_unary(14, B8, 0x0000_0005), 0x0000_0000);
    assert_eq!(k_unary(15, B8, 0x0000_00FF), 0x0000_0000);
    assert_eq!(k_unary(16, B16, 0x0000_FFFF), 0x0000_0000);
    assert_eq!(k_unary(20, B32, 0xFFFF_FFFF), 0x0000_0000);
}

// A float kernel result has no bits above the format width, whatever the lane
// input carried there; TanH(+-0) is the canonical signed zero of the format.
// model: unit_tests/test_slices.sail test_float_kernels_no_stray_bits
#[test]
fn test_float_kernels_no_stray_bits() {
    assert_eq!(tr::fp_tr_value(21, FMT16, 0x0000_0000), Some(0x0000_0000));
    assert_eq!(tr::fp_tr_value(21, FMT16, 0x0000_8000), Some(0x0000_8000));
    assert_eq!(tr::fp_tr_value(21, FMT16, 0x1234_0000), Some(0x0000_0000)); // stray high bits dropped
    assert_eq!(tr::fp_tr_value(21, FMT16, 0xABCD_8000), Some(0x0000_8000));
    assert_eq!(tr::fp_tr_value(21, FMT32, 0x0000_0000), Some(0x0000_0000));
    assert_eq!(tr::fp_tr_value(21, FMT32, 0x8000_0000), Some(0x8000_0000));
    assert_eq!(k_unary(20, F16, 0x0001_3C00), 0x0000_3C00); // ReLU of a positive f16 keeps 16 bits
    assert_eq!(k_unary(20, F32, 0x3F80_0000), 0x3F80_0000);
}

/// Release builds run every input like the Sail test; debug builds (the big-integer port is slow
/// there) every 13th, unless `TSP_NUMERICS_FULL` is set.
fn stride() -> usize {
    if std::env::var_os("TSP_NUMERICS_FULL").is_some() || !cfg!(debug_assertions) {
        1
    } else {
        13
    }
}

// The transcendentals' Ziv loop converges on every i8 and f16 input and on a stride
// of f32 inputs (every 2^18th bit pattern), so no result comes from a fallback. The loop is the
// straight port (`ziv`), not the memoised fast path, which would not reach it.
// model: unit_tests/test_slices.sail test_transcendental_converges
#[test]
fn test_transcendental_converges() {
    for op in [TANH, EXP, RSQRT, LOG] {
        for n in -128..=127 {
            assert!(ziv::i8_tr_value(op, n).is_some(), "op {op} i8 {n}");
        }
        for w in (0..=65535u32).step_by(stride()) {
            assert!(ziv::fp_tr_value(op, FMT16, w).is_some(), "op {op} f16 {w:#06x}");
        }
        for k in (0..=16383u32).step_by(stride()) {
            let x = 262144u32.wrapping_mul(k).wrapping_add(12345);
            assert!(ziv::fp_tr_value(op, FMT32, x).is_some(), "op {op} f32 {x:#010x}");
        }
    }
}
