//! TanH (21), Exp (22), RSqrt (23) and Log (24) on i8, f16 and f32 elements
//! (`core/numerics/transcendental.sail`).
//!
//! The model's contract is that the result is the exact function value rounded once, to nearest
//! even, to the destination format, so any method that provably returns the correctly rounded
//! value equals the Sail evaluator bit for bit. Evaluation order here:
//!
//! 1. the special-value tables of `fp_tr_value` and `i8_tr_value` ([`fp_tr_pre`], [`i8_tr_pre`]);
//! 2. the fast path: the function in `f64` (`exp`, `ln`, `tanh`, `1/sqrt`), accepted when its
//!    value is at least 2^-40 (relative to the value) away from every rounding boundary of the
//!    destination: the midpoints of the f16 or f32 grid, including the subnormal range and the
//!    overflow threshold, or the half-points of the integers for i8. The `f64` result is within
//!    a few 2^-53 (relative) of the true value on every libm that meets its documented accuracy
//!    (audited <= 2.9e-16 on this host), so 2^-40 leaves a margin of about 2^11;
//! 3. otherwise the straight port of the Sail evaluator on big integers ([`crate::ziv`]).
//!
//! i8 (256 inputs per op) and f16 (65536 per op) results are memoised in tables built on first
//! use. Where the exact value is a tie (rsqrt(4) = 0.5 on i8) the table of `i8_tr_pre` decides, as in Sail.

use crate::fault::model_fault;
use crate::fp::{self, FClass, FVal, Fmt};
use crate::lanes::{bits_of_s, s_of_bits, sat_s};
use crate::ziv;
use std::sync::OnceLock;

pub const TANH: u32 = 21;
pub const EXP: u32 = 22;
pub const RSQRT: u32 = 23;
pub const LOG: u32 = 24;

/// The relative distance below which the fast path declines: 2^-40.
const FILTER: f64 = 1.0 / (1u64 << 40) as f64;

/// The magnitude from which [`tr_fast_int`] declines: 2^62.
const FAST_INT_LIMIT: f64 = (1u64 << 62) as f64;

/// The outcome of the special-value tables: a result, or an input for the evaluator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pre {
    Done(u32),
    /// A finite, non-zero, in-domain input that needs `tr_eval`.
    Eval(FVal),
}

/// The integer counterpart of [`Pre`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IPre {
    Done(i64),
    Eval(FVal),
}

/// `fp_tr_value` up to (and excluding) `tr_eval`: NaN, infinities, zeros, domain errors, log(1)
/// and the saturating ranges. // model: core/numerics/transcendental.sail fp_tr_value
pub fn fp_tr_pre(op: u32, f: Fmt, x: u32) -> Pre {
    let a = fp::fdecode(f, x);
    match a.cls {
        FClass::NaN => Pre::Done(fp::canon_nan(f)),
        FClass::Inf => Pre::Done(if op == 21 {
            if a.neg {
                fp::fp_of_int(f, -1)
            } else {
                fp::fp_of_int(f, 1)
            }
        } else if op == 22 {
            if a.neg {
                fp::zero_bits(f, false)
            } else {
                fp::inf_bits(f, false)
            }
        } else if op == 23 {
            if a.neg {
                fp::canon_nan(f)
            } else {
                fp::zero_bits(f, false)
            }
        } else if a.neg {
            fp::canon_nan(f)
        } else {
            fp::inf_bits(f, false)
        }),
        FClass::Zero => Pre::Done(if op == 21 {
            fp::zero_bits(f, a.neg) // tanh(+-0) = the canonical +-0 (no stray bits)
        } else if op == 22 {
            fp::fp_of_int(f, 1)
        } else if op == 23 {
            fp::inf_bits(f, a.neg) // rsqrt(+-0) = +-Inf (IEEE rSqrt)
        } else {
            fp::inf_bits(f, true) // log(+-0) = -Inf
        }),
        FClass::Finite => {
            let top = a.e + fp::ilog2(a.mag as u128);
            if (op == 23 || op == 24) && a.neg {
                Pre::Done(fp::canon_nan(f))
            } else if op == 24 && a.e <= 0 && a.e >= -63 && a.mag == 1u64 << (-a.e) as u32 {
                Pre::Done(fp::zero_bits(f, false)) // log(1) = +0
            } else if op == 22 && top > 8 {
                Pre::Done(if a.neg { fp::zero_bits(f, false) } else { fp::inf_bits(f, false) })
            // |x| >= 512
            } else if op == 21 && top > 5 {
                Pre::Done(fp::fp_of_int(f, if a.neg { -1 } else { 1 })) // |x| >= 64
            } else {
                Pre::Eval(a)
            }
        }
    }
}

/// `i8_tr_value` up to `tr_eval_int`. // model: core/numerics/transcendental.sail i8_tr_value
pub fn i8_tr_pre(op: u32, v: i64) -> IPre {
    if v == 0 {
        IPre::Done(match op {
            22 => 1,
            23 => 127,
            24 => -128,
            _ => 0,
        })
    } else if (op == 23 || op == 24) && v < 0 {
        IPre::Done(0)
    } else if op == 22 && v > 5 {
        IPre::Done(127) // e^5 = 148.4 > 127
    } else if op == 22 && v < -1 {
        IPre::Done(0) // e^-1 = 0.37 -> 0; e^-2 smaller
    } else if op == 23 && v == 4 {
        // rsqrt(4) = 0.5 is the one exact tie among i8 inputs (1/sqrt(v) = n + 1/2 needs
        // v = 4 / (2n + 1)^2): the enclosure never leaves the tie, and RNE gives 0.
        IPre::Done(0)
    } else {
        IPre::Eval(FVal { cls: FClass::Finite, neg: v < 0, mag: v.unsigned_abs(), e: 0 })
    }
}

/// 2^e as an `f64`, for e in -1022..=1023.
fn pow2_f64(e: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&e));
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// The exact input value as an `f64` (f16 and f32 values are exact in it).
fn value_f64(a: &FVal) -> f64 {
    let v = a.mag as f64 * pow2_f64(a.e);
    if a.neg {
        -v
    } else {
        v
    }
}

fn apply_f64(op: u32, x: f64) -> f64 {
    match op {
        21 => x.tanh(),
        22 => x.exp(),
        23 => 1.0 / x.sqrt(),
        _ => x.ln(),
    }
}

/// Position of `av` (> 0) on the grid of quantum 2^qe, as `(floor, frac, distance of frac from
/// 1/2)`, accepted only when the distance to the nearest half-point exceeds `s * 2^-40` where
/// `s = av / 2^qe`. Returns `(floor + (frac > 1/2))` on success.
fn round_on_grid(av: f64, qe: i32) -> Option<u128> {
    let s = av * pow2_f64(-qe);
    let fl = s.floor();
    let fr = s - fl;
    let dist = (fr - 0.5).abs();
    if dist > s * FILTER {
        Some(fl as u128 + (fr > 0.5) as u128)
    } else {
        None
    }
}

/// The fast path for a float destination: `Some(bits)` when the `f64` value is far enough from
/// every rounding boundary of `f`. `a` is finite, non-zero and in domain.
pub fn tr_fast(op: u32, f: Fmt, a: &FVal) -> Option<u32> {
    let v = apply_f64(op, value_f64(a));
    if !v.is_finite() || v == 0.0 {
        return None;
    }
    let neg = v < 0.0;
    let av = v.abs();
    let ex = (av.to_bits() >> 52) as i32;
    if ex == 0 {
        return None; // an f64 subnormal: not reachable for f16 or f32 inputs
    }
    let top = ex - 1023; // floor(log2 av)
    let ee = top.max(f.emin);
    let qe = ee - f.mbits;
    let mq = round_on_grid(av, qe)?;
    Some(fp::fround(f, neg, mq, qe))
}

/// The fast path for the nearest integer (i8). `a` is a finite non-zero integer value. Declines
/// (`None`) a result of 2^62 or more, which an `i64` result could not hold.
pub fn tr_fast_int(op: u32, a: &FVal) -> Option<i64> {
    let v = apply_f64(op, value_f64(a));
    if !v.is_finite() || v.abs() >= FAST_INT_LIMIT {
        return None;
    }
    let mq = round_on_grid(v.abs(), 0)? as i64;
    Some(if v < 0.0 { -mq } else { mq })
}

/// `tr_eval`: the fast path, then the Ziv port. // model: core/numerics/transcendental.sail tr_eval
pub fn tr_eval(op: u32, f: Fmt, a: &FVal) -> Option<u32> {
    match tr_fast(op, f, a) {
        Some(r) => Some(r),
        None => ziv::tr_eval(op, f, a),
    }
}

/// `tr_eval_int`: the fast path, then the Ziv port. // model: core/numerics/transcendental.sail tr_eval_int
pub fn tr_eval_int(op: u32, a: &FVal) -> Option<i64> {
    match tr_fast_int(op, a) {
        Some(r) => Some(r),
        None => ziv::tr_eval_int(op, a),
    }
}

/// Transcendental of an FP element (format `f`); `None` only if the Ziv loop does not converge
/// (a model error). // model: core/numerics/transcendental.sail fp_tr_value
pub fn fp_tr_value(op: u32, f: Fmt, x: u32) -> Option<u32> {
    match fp_tr_pre(op, f, x) {
        Pre::Done(r) => Some(r),
        Pre::Eval(a) => tr_eval(op, f, &a),
    }
}

/// i8 transcendental on the exact integer input, before saturation. // model: core/numerics/transcendental.sail i8_tr_value
pub fn i8_tr_value(op: u32, v: i64) -> Option<i64> {
    match i8_tr_pre(op, v) {
        IPre::Done(r) => Some(r),
        IPre::Eval(a) => tr_eval_int(op, &a),
    }
}

fn fp_transcendental_eval(op: u32, f: Fmt, x: u32) -> u32 {
    fp_tr_value(op, f, x).unwrap_or_else(|| model_fault("fp_transcendental: Ziv loop did not converge"))
}

pub(crate) fn f16_table(op: u32) -> &'static [u16] {
    static TABLES: [OnceLock<Box<[u16]>>; 4] = [OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new()];
    TABLES[(op - 21) as usize]
        .get_or_init(|| (0..0x10000u32).map(|x| fp_transcendental_eval(op, fp::FMT16, x) as u16).collect())
}

fn i8_table(op: u32) -> &'static [u8] {
    static TABLES: [OnceLock<Box<[u8]>>; 4] = [OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new()];
    TABLES[(op - 21) as usize].get_or_init(|| (0..256u32).map(|x| i8_transcendental_uncached(op, x) as u8).collect())
}

/// Transcendental of an FP element. f16 results come from a memoised table; a model error
/// (Ziv loop not converging) is a [`crate::fault::ModelFault`]. // model: core/numerics/transcendental.sail fp_transcendental
pub fn fp_transcendental(op: u32, f: Fmt, x: u32) -> u32 {
    debug_assert!((21..=24).contains(&op));
    if f.width == 16 && x >> 16 == 0 {
        f16_table(op)[x as usize] as u32
    } else {
        if op == RSQRT && f.width == 32 {
            if let Some(r) = rsqrt_f32_fast(x) {
                return r;
            }
        }
        fp_transcendental_eval(op, f, x)
    }
}

/// The units in the last place of an `f64` (2^-29 of an `f32` grid step) that the `f64` value of
/// 1/sqrt(x) must keep from an `f32` rounding midpoint for [`rsqrt_f32_fast`] to accept it. The
/// `f64` error is at most 2 units (a correctly rounded sqrt and a correctly rounded quotient), so
/// 2^13 leaves a margin of 2^12 (the generic filter is 2^-40 of the value, which is about 2^13
/// units at the top of the `f32` binade).
const RSQRT_F32_MARGIN: u64 = 1 << 13;

/// 1/sqrt(x) for a positive normal `f32` `x`, rounded once to nearest even, without the generic
/// decode and the big-integer evaluator: `Some(bits)` when the `f64` value lies at least
/// [`RSQRT_F32_MARGIN`] units away from the midpoint between two `f32` values, `None` otherwise
/// (a sign, a subnormal, zero, infinity, NaN, or a value too close to a midpoint: the caller
/// falls back to the exact path). The result is a normal `f32` (1/sqrt of 2^-126..2^128 lies in
/// 2^-64..2^63.01), so `as f32` is the single RNE rounding.
// model: core/numerics/transcendental.sail fp_tr_value (the RSqrt of an f32 input)
pub fn rsqrt_f32_fast(x: u32) -> Option<u32> {
    let e = (x >> 23) & 0xff;
    if x >> 31 != 0 || e == 0 || e == 0xff {
        return None;
    }
    let v = 1.0 / (f32::from_bits(x) as f64).sqrt();
    let low = v.to_bits() & ((1u64 << 29) - 1);
    if low.abs_diff(1u64 << 28) <= RSQRT_F32_MARGIN {
        return None;
    }
    Some((v as f32).to_bits())
}

fn i8_transcendental_uncached(op: u32, x: u32) -> u32 {
    let r =
        i8_tr_value(op, s_of_bits(x, 8)).unwrap_or_else(|| model_fault("i8_transcendental: Ziv loop did not converge"));
    bits_of_s(sat_s(r as i128, 8), 8)
}

/// i8 transcendental on a lane value: evaluated on the exact integer, rounded to i8 (RNE) and
/// saturated; from a memoised table. // model: core/numerics/transcendental.sail i8_transcendental
pub fn i8_transcendental(op: u32, x: u32) -> u32 {
    debug_assert!((21..=24).contains(&op));
    i8_table(op)[(x & 0xFF) as usize] as u32
}
