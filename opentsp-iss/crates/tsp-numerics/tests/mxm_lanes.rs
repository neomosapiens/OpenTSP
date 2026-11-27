//! `f16_dot` against a literal transcription of `slices/mxm/mxm_insts.sail` f16_dot, lane by lane.
//!
//! The shipped routine skips a lane whose operand is +0 and whose other operand is neither Inf nor NaN: such a lane
//! adds no term and only decides whether an exactly zero sum is -0. The reference below takes every lane through
//! the decode, so the two disagree the moment the skip is wrong about a sign, a subnormal, an infinity or a NaN.
use tsp_numerics::fp::{self, FClass, FMT16, FMT32};
use tsp_numerics::mxm::f16_dot;

/// The model's f16 dot product with no lane skipped.
fn reference(wlo: &[u8], whi: &[u8], xlo: &[u8], xhi: &[u8]) -> u32 {
    let (mut nan, mut pinf, mut ninf, mut all_negzero) = (false, false, false, true);
    let mut sum: i128 = 0;
    for k in 0..wlo.len() {
        let a = fp::fdecode(FMT16, (whi[k] as u32) << 8 | wlo[k] as u32);
        let b = fp::fdecode(FMT16, (xhi[k] as u32) << 8 | xlo[k] as u32);
        let neg = a.neg != b.neg;
        let inf_times_zero =
            (a.cls == FClass::Inf && b.cls == FClass::Zero) || (a.cls == FClass::Zero && b.cls == FClass::Inf);
        if a.cls == FClass::NaN || b.cls == FClass::NaN || inf_times_zero {
            nan = true;
        } else if a.cls == FClass::Inf || b.cls == FClass::Inf {
            if neg {
                ninf = true;
            } else {
                pinf = true;
            }
            all_negzero = false;
        } else if a.cls == FClass::Zero || b.cls == FClass::Zero {
            if !neg {
                all_negzero = false;
            }
        } else {
            all_negzero = false;
            let m = ((a.mag * b.mag) as i128) << (a.e + b.e + 48) as u32;
            sum = if neg { sum - m } else { sum + m };
        }
    }
    if nan || (pinf && ninf) {
        fp::canon_nan(FMT32)
    } else if pinf {
        fp::inf_bits(FMT32, false)
    } else if ninf {
        fp::inf_bits(FMT32, true)
    } else if sum == 0 {
        fp::zero_bits(FMT32, all_negzero)
    } else if sum < 0 {
        fp::fround(FMT32, true, (-sum) as u128, -48)
    } else {
        fp::fround(FMT32, false, sum as u128, -48)
    }
}

/// The four byte planes of a row of f16 values.
fn planes(w: &[u16], x: &[u16]) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    (
        w.iter().map(|v| *v as u8).collect(),
        w.iter().map(|v| (v >> 8) as u8).collect(),
        x.iter().map(|v| *v as u8).collect(),
        x.iter().map(|v| (v >> 8) as u8).collect(),
    )
}

fn same(w: &[u16], x: &[u16]) {
    let (wlo, whi, xlo, xhi) = planes(w, x);
    assert_eq!(f16_dot(&wlo, &whi, &xlo, &xhi), reference(&wlo, &whi, &xlo, &xhi), "w {w:x?} x {x:x?}");
}

/// Every f16 bit pattern against the values the lane skip turns on: +0, -0, both infinities, a NaN, a subnormal,
/// one and minus one, in both operand positions and as the only lane of a row.
#[test]
fn every_f16_against_the_values_that_decide_the_lane_skip() {
    let others: [u16; 9] = [0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0x0001, 0x8001, 0x3c00, 0xbc00];
    for v in 0..=u16::MAX {
        for o in others {
            same(&[v], &[o]);
            same(&[o], &[v]);
            // a zero lane next to a lane that carries the sum: the skip must not lose the sign of the zero
            same(&[v, o], &[o, v]);
        }
    }
}

/// Rows of 320 lanes of mixed zeros, subnormals, specials and ordinary values.
#[test]
fn rows_of_mixed_lanes_match_the_reference() {
    let pool: [u16; 14] = [
        0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0x1234,
    ];
    let mut s: u64 = 0x243f_6a88_85a3_08d3;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for case in 0..2_000 {
        // most lanes zero in the early cases (the padded activations of a step), dense later
        let zero_share = if case < 1_000 { 90 } else { 10 };
        let mk = |next: &mut dyn FnMut() -> u64| -> Vec<u16> {
            (0..320)
                .map(|_| {
                    let r = next();
                    if r % 100 < zero_share {
                        0
                    } else {
                        pool[(r >> 32) as usize % pool.len()]
                    }
                })
                .collect()
        };
        let (w, x) = (mk(&mut next), mk(&mut next));
        same(&w, &x);
    }
}
