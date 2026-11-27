//! Transcendentals: the fast path (f64 + 2^-40 filter, memoised tables) must equal the straight
//! port of `transcendental.sail` (Ziv's strategy on big integers) on every i8 and f16 input.
use tsp_numerics::fp::{FMT16, FMT32};
use tsp_numerics::transcendental::{self as tr, EXP, LOG, RSQRT, TANH};
use tsp_numerics::ziv;

const OPS: [u32; 4] = [TANH, EXP, RSQRT, LOG];

/// Release builds check every input; debug builds (the num-bigint port is slow there) every 13th.
fn stride() -> usize {
    if std::env::var_os("TSP_NUMERICS_FULL").is_some() || !cfg!(debug_assertions) {
        1
    } else {
        13
    }
}

#[test]
fn f16_exhaustive_matches_ziv() {
    for op in OPS {
        let mut x = 0u32;
        while x < 0x10000 {
            let fast = tr::fp_tr_value(op, FMT16, x);
            let slow = ziv::fp_tr_value(op, FMT16, x);
            assert_eq!(fast, slow, "op {op} f16 x={x:04x}");
            assert!(fast.is_some(), "op {op} f16 x={x:04x} did not converge");
            x += stride() as u32;
        }
    }
}

#[test]
fn i8_exhaustive_matches_ziv() {
    for op in OPS {
        for x in 0..256u32 {
            let v = (x as u8 as i8) as i64;
            assert_eq!(tr::i8_tr_value(op, v), ziv::i8_tr_value(op, v), "op {op} i8 v={v}");
            assert_eq!(tr::i8_transcendental(op, x), ziv::i8_transcendental(op, x));
        }
    }
}

#[test]
fn f32_edge_inputs_match_ziv() {
    let edges = [
        0x0000_0001u32,
        0x007f_ffff,
        0x0080_0000,
        0x3f80_0000,
        0x3f80_0001,
        0x3f7f_ffff,
        0x3380_0000,
        0x42b0_0000,
        0x42b1_7218,
        0x42b1_7217,
        0xc2ce_8ed0,
        0xc2cf_f1b4,
        0x4300_0000,
        0x7f7f_ffff,
        0x4080_0000,
        0x3e80_0000,
    ];
    for op in OPS {
        for &x in &edges {
            assert_eq!(tr::fp_tr_value(op, FMT32, x), ziv::fp_tr_value(op, FMT32, x), "op {op} f32 x={x:08x}");
        }
    }
}
