//! The straight port of `core/numerics/transcendental.sail` on big integers.
//!
//! This is the model's own evaluator: fixed-point interval arithmetic with a rigorous error
//! bound, refined until the enclosing interval rounds to one destination value (Ziv's
//! strategy). It is the fallback of the fast path in [`crate::transcendental`] and the reference
//! that path is tested against. Names, argument order and the order of operations follow the
//! Sail source.

use crate::bigfp::{fround_big, ilog2, pow2i};
use crate::fp::{FClass, FVal, Fmt};
use crate::transcendental::{fp_tr_pre, i8_tr_pre, IPre, Pre};
use num_bigint::{BigInt, Sign};

/// A real interval `[lo, hi] * 2^-p`. // model: core/numerics/transcendental.sail struct ival
#[derive(Clone, Debug)]
pub struct Ival {
    pub lo: BigInt,
    pub hi: BigInt,
    pub p: i64,
}

fn big(v: i64) -> BigInt {
    BigInt::from(v)
}

fn is_zero(v: &BigInt) -> bool {
    v.sign() == Sign::NoSign
}

/// floor(a / b), b > 0. // model: core/numerics/transcendental.sail idiv_floor
pub fn idiv_floor(a: &BigInt, b: &BigInt) -> BigInt {
    let q = a / b; // truncates toward zero, as tdiv_int
    if &q * b > *a {
        q - 1
    } else {
        q
    }
}

/// ceil(a / b), b > 0. // model: core/numerics/transcendental.sail idiv_ceil
pub fn idiv_ceil(a: &BigInt, b: &BigInt) -> BigInt {
    -idiv_floor(&(-a), b)
}

/// The input `neg * mag * 2^e` as a fixed-point interval at precision p. // model: core/numerics/transcendental.sail fx_of
pub fn fx_of(neg: bool, mag: &BigInt, e: i64, p: i64) -> Ival {
    let s = e + p;
    if s >= 0 {
        let v = mag * pow2i(s);
        let v2 = if neg { -v } else { v };
        Ival { lo: v2.clone(), hi: v2, p }
    } else {
        let d = pow2i(-s);
        let sm = if neg { -mag } else { mag.clone() };
        Ival { lo: idiv_floor(&sm, &d), hi: idiv_ceil(&sm, &d), p }
    }
}

/// exp(y) for y = Y * 2^-p, as an enclosing interval at precision p (Taylor series with integer
/// truncation), for p >= 0 and |y * 2^-p| <= 1/2. // model: core/numerics/transcendental.sail exp_small
pub fn exp_small(y: &BigInt, p: i64) -> (BigInt, BigInt) {
    let one = pow2i(p);
    let mut term = one.clone();
    let mut sum = one.clone();
    let mut k: i64 = 1;
    let mut err: i64 = 2;
    while !is_zero(&term) && k <= p + 1 {
        term = (&term * y) / &one;
        term /= k;
        sum += &term;
        err += 2;
        k += 1;
    }
    (&sum - err, &sum + err)
}

/// ln(2) to precision p, enclosed: ln2 = 2 * atanh(1/3). // model: core/numerics/transcendental.sail ln2_iv
pub fn ln2_iv(p: i64) -> (BigInt, BigInt) {
    let one = pow2i(p + 8);
    let mut t = &one / 3i64;
    let mut sum = BigInt::from(0);
    let mut k: i64 = 0;
    while t.sign() == Sign::Plus {
        sum += &t / (2 * k + 1);
        t /= 9i64;
        k += 1;
    }
    let v = 2i64 * sum;
    let err = 4 * (k + 2);
    (idiv_floor(&(&v - err), &big(256)), idiv_ceil(&(&v + err), &big(256)))
}

/// exp(x) for a real interval x at precision p, as `(lo, hi, n)` meaning exp(x) in
/// `[lo, hi] * 2^(n - p)`. Range reduction x = n ln2 + r. // model: core/numerics/transcendental.sail exp_iv
pub fn exp_iv(x: &Ival) -> (BigInt, BigInt, i64) {
    let p = x.p;
    let (l2lo, l2hi) = ln2_iv(p);
    let mid = (&x.lo + &x.hi) / 2i64;
    let n = idiv_floor(&(mid + &l2lo / 2i64), &l2lo);
    // r = x - n ln2, enclosed
    let (rlo, rhi) = if n.sign() != Sign::Minus {
        (&x.lo - &n * &l2hi, &x.hi - &n * &l2lo)
    } else {
        (&x.lo - &n * &l2lo, &x.hi - &n * &l2hi)
    };
    let (alo, _) = exp_small(&rlo, p);
    let (_, bhi) = exp_small(&rhi, p);
    let n64 = i64::try_from(&n)
        .unwrap_or_else(|_| crate::fault::model_fault("exp_iv: the range reduction index does not fit an i64"));
    (alo, bhi, n64)
}

/// ln(x) for positive x = mag * 2^e at precision p. // model: core/numerics/transcendental.sail log_iv
pub fn log_iv(mag: &BigInt, e: i64, p: i64) -> (BigInt, BigInt) {
    let b = ilog2(mag);
    let k = b + e; // x = 2^k * m, m = mag / 2^b in [1, 2)
    let one = pow2i(p + 8);
    // z = (m-1)/(m+1) = (mag - 2^b) / (mag + 2^b)
    let num = (mag - pow2i(b)) * &one;
    let den = mag + pow2i(b);
    let z = num / den;
    let z2 = (&z * &z) / &one;
    let mut t = z.clone();
    let mut sum = BigInt::from(0);
    let mut j: i64 = 0;
    while t.sign() == Sign::Plus && j < 2000 {
        sum += &t / (2 * j + 1);
        t = (&t * &z2) / &one;
        j += 1;
    }
    let lnm = 2i64 * sum;
    let errm = 4 * (j + 3);
    let (l2lo, l2hi) = ln2_iv(p + 8);
    let lo = (if k >= 0 { k * &l2lo } else { k * &l2hi }) + &lnm - errm;
    let hi = (if k >= 0 { k * &l2hi } else { k * &l2lo }) + &lnm + errm;
    (idiv_floor(&lo, &big(256)), idiv_ceil(&hi, &big(256)))
}

/// Integer square root, floor. // model: core/numerics/transcendental.sail isqrt
pub fn isqrt(n: &BigInt) -> BigInt {
    if n.sign() != Sign::Plus {
        return BigInt::from(0);
    }
    let mut x = pow2i(ilog2(n) / 2 + 1);
    let mut y = (&x + n / &x) / 2i64;
    while y < x {
        x = y;
        y = (&x + n / &x) / 2i64;
    }
    x
}

/// Round the real number `[lo, hi] * 2^sh` to the format; `Some` only if both ends agree.
/// // model: core/numerics/transcendental.sail round_iv
pub fn round_iv(f: Fmt, lo: &BigInt, hi: &BigInt, sh: i64) -> Option<u32> {
    let a = if lo.sign() == Sign::Minus { fround_big(f, true, &(-lo), sh) } else { fround_big(f, false, lo, sh) };
    let b = if hi.sign() == Sign::Minus { fround_big(f, true, &(-hi), sh) } else { fround_big(f, false, hi, sh) };
    if a == b {
        Some(a)
    } else {
        None
    }
}

/// Enclosure of op(a) as `[lo, hi] * 2^sh` at working precision p. `a` is finite and in domain.
/// // model: core/numerics/transcendental.sail tr_interval
pub fn tr_interval(op: u32, a: &FVal, p: i64) -> (BigInt, BigInt, i64) {
    let mag = BigInt::from(a.mag);
    let e = a.e as i64;
    if op == 22 {
        let x = fx_of(a.neg, &mag, e, p);
        let (lo, hi, n) = exp_iv(&x);
        (lo, hi, n - p)
    } else if op == 24 {
        let (lo, hi) = log_iv(&mag, e, p);
        (lo, hi, -p)
    } else if op == 23 {
        // x = m2 * 2^e2 with e2 even; 1/sqrt(x) = 2^(2p) / sqrt(m2 * 2^(2p)) * 2^(-p - e2/2)
        let e2: i64 = if e % 2 == 0 { e } else { e - 1 };
        let m2: BigInt = if e % 2 == 0 { mag } else { mag * 2i64 };
        let s = isqrt(&(m2 * pow2i(2 * p))); // s <= sqrt(m2) 2^p < s + 1
        let num = pow2i(4 * p);
        (&num / (&s + 1), &num / &s + 1, -3 * p - e2 / 2)
    } else {
        // tanh(|x|) = 1 - 2 / (E + 1), E = exp(2|x|); odd function.
        let x = fx_of(false, &mag, e + 1, p);
        let (lo, hi, n) = exp_iv(&x);
        // exp_iv gives E = [lo, hi] * 2^(n - p); scaled by 2^p that is [lo, hi] * 2^n.
        let elo = if n >= 0 { &lo * pow2i(n) } else { idiv_floor(&lo, &pow2i(-n)) };
        let ehi = if n >= 0 { &hi * pow2i(n) } else { idiv_ceil(&hi, &pow2i(-n)) };
        let one = pow2i(p);
        let bigv = pow2i(2 * p);
        // E in [elo, ehi] * 2^-p ; 2/(E+1) = 2 * 2^p / (E' + 2^p)
        let tlo = &bigv - idiv_ceil(&(2i64 * &bigv * &one), &(&elo + &one));
        let thi = &bigv - idiv_floor(&(2i64 * &bigv * &one), &(&ehi + &one));
        let (lo2, hi2) = if a.neg { (-&thi, -&tlo) } else { (tlo, thi) };
        (lo2 - 1i64, hi2 + 1i64, -2 * p)
    }
}

/// Ziv loop: refine until the enclosure rounds to one value of the format. `None` only if 4096
/// bits do not suffice (a model error). // model: core/numerics/transcendental.sail tr_eval
pub fn tr_eval(op: u32, f: Fmt, a: &FVal) -> Option<u32> {
    let mut p: i64 = 64;
    let mut res = None;
    while res.is_none() && p <= 4096 {
        let (lo, hi, sh) = tr_interval(op, a, p);
        res = round_iv(f, &lo, &hi, sh);
        p *= 2;
    }
    res
}

/// RNE of `v * 2^sh` to an integer (v may be negative). // model: core/numerics/transcendental.sail rne_int
pub fn rne_int(v: &BigInt, sh: i64) -> BigInt {
    if sh >= 0 {
        v * pow2i(sh)
    } else {
        let d = pow2i(-sh);
        let q = idiv_floor(v, &d);
        let rem = v - &q * &d;
        let half = &d / 2i64;
        let odd = !is_zero(&(&q % 2i64));
        if rem > half || (rem == half && odd) {
            q + 1i64
        } else {
            q
        }
    }
}

/// Ziv loop to the nearest integer (RNE); `None` as for `tr_eval`. // model: core/numerics/transcendental.sail tr_eval_int
pub fn tr_eval_int(op: u32, a: &FVal) -> Option<i64> {
    let mut p: i64 = 64;
    let mut res = None;
    while res.is_none() && p <= 4096 {
        let (lo, hi, sh) = tr_interval(op, a, p);
        let x = rne_int(&lo, sh);
        let y = rne_int(&hi, sh);
        if x == y {
            res = Some(
                i64::try_from(&x)
                    .unwrap_or_else(|_| crate::fault::model_fault("tr_eval_int: the result does not fit an i64")),
            );
        }
        p *= 2;
    }
    res
}

/// Transcendental of an FP element, evaluated by the Ziv port alone. // model: core/numerics/transcendental.sail fp_tr_value
pub fn fp_tr_value(op: u32, f: Fmt, x: u32) -> Option<u32> {
    match fp_tr_pre(op, f, x) {
        Pre::Done(r) => Some(r),
        Pre::Eval(a) => {
            debug_assert!(a.cls == FClass::Finite);
            tr_eval(op, f, &a)
        }
    }
}

/// `fp_tr_value`, with a model error as a [`crate::fault::ModelFault`]. // model: core/numerics/transcendental.sail fp_transcendental
pub fn fp_transcendental(op: u32, f: Fmt, x: u32) -> u32 {
    fp_tr_value(op, f, x).unwrap_or_else(|| crate::fault::model_fault("fp_transcendental: Ziv loop did not converge"))
}

/// i8 transcendental on the exact integer input, by the Ziv port alone. // model: core/numerics/transcendental.sail i8_tr_value
pub fn i8_tr_value(op: u32, v: i64) -> Option<i64> {
    match i8_tr_pre(op, v) {
        IPre::Done(r) => Some(r),
        IPre::Eval(a) => tr_eval_int(op, &a),
    }
}

/// i8 transcendental on a lane value (zero-extended i8), result saturated to i8.
/// // model: core/numerics/transcendental.sail i8_transcendental
pub fn i8_transcendental(op: u32, x: u32) -> u32 {
    let v = crate::lanes::s_of_bits(x, 8);
    let r =
        i8_tr_value(op, v).unwrap_or_else(|| crate::fault::model_fault("i8_transcendental: Ziv loop did not converge"));
    crate::lanes::bits_of_s(crate::lanes::sat_s(r as i128, 8), 8)
}
