//! Exact arithmetic on big integers: `fround` for arbitrarily large magnitudes (the Ziv port
//! needs it) and `fadd_core` for any format. These follow `fp.sail` literally (no machine-width
//! shortcut) and are the reference against which the fast paths of [`crate::fp`] are tested.

use crate::fp::{canon_nan, fdecode, inf_bits, zero_bits, FClass, FVal, Fmt};
use num_bigint::{BigInt, Sign};

/// 2^n for n > 0, else 1. // model: core/numerics/fp.sail pow2i
pub fn pow2i(n: i64) -> BigInt {
    if n <= 0 {
        BigInt::from(1)
    } else {
        BigInt::from(1) << (n as usize)
    }
}

/// floor(log2(n)) for n >= 1. // model: core/numerics/fp.sail ilog2
pub fn ilog2(n: &BigInt) -> i64 {
    debug_assert!(n.sign() == Sign::Plus);
    n.bits() as i64 - 1
}

fn small(v: &BigInt) -> u64 {
    // a value below 2^64 has at most one digit
    debug_assert!(v.bits() <= 64);
    v.to_u64_digits().1.first().copied().unwrap_or(0)
}

/// Round `(-1)^neg * mag * 2^e` (mag >= 0) to the format, RNE. // model: core/numerics/fp.sail fround
pub fn fround_big(f: Fmt, neg: bool, mag: &BigInt, e: i64) -> u32 {
    if mag.sign() == Sign::NoSign {
        return zero_bits(f, neg);
    }
    let p = ilog2(mag);
    let ee = if p + e < f.emin as i64 { f.emin as i64 } else { p + e };
    let q = ee - f.mbits as i64;
    let mut m: BigInt;
    if e >= q {
        m = mag << ((e - q) as usize);
    } else {
        let d = (q - e) as usize;
        let lo = mag >> d;
        let rem = mag - (&lo << d);
        let half = BigInt::from(1) << (d - 1);
        let odd = (&lo & BigInt::from(1)) == BigInt::from(1);
        m = if rem > half || (rem == half && odd) { lo + 1 } else { lo };
    }
    let mut ex = ee;
    if m == BigInt::from(1) << (f.mbits as usize + 1) {
        m >>= 1usize;
        ex += 1;
    }
    let mv = small(&m);
    let sign: u64 = if neg { 1u64 << (f.width - 1) } else { 0 };
    if mv == 0 {
        zero_bits(f, neg)
    } else if mv < (1u64 << f.mbits) {
        (sign + mv) as u32
    } else if (f.ieee && ex + f.bias as i64 >= (1i64 << f.ebits) - 1)
        || (!f.ieee
            && (ex + f.bias as i64 > (1i64 << f.ebits) - 1
                || (ex + f.bias as i64 == (1i64 << f.ebits) - 1 && mv - (1u64 << f.mbits) == (1u64 << f.mbits) - 1)))
    {
        inf_bits(f, neg)
    } else {
        (sign + (ex + f.bias as i64) as u64 * (1u64 << f.mbits) + (mv - (1u64 << f.mbits))) as u32
    }
}

/// Sum or difference of two decoded values, any format. // model: core/numerics/fp.sail fadd_core
pub fn fadd_core_big(f: Fmt, a: FVal, b: FVal, negate_b: bool) -> u32 {
    let bneg = if negate_b { !b.neg } else { b.neg };
    if a.cls == FClass::NaN || b.cls == FClass::NaN {
        canon_nan(f)
    } else if a.cls == FClass::Inf && b.cls == FClass::Inf {
        if a.neg == bneg {
            inf_bits(f, a.neg)
        } else {
            canon_nan(f)
        }
    } else if a.cls == FClass::Inf {
        inf_bits(f, a.neg)
    } else if b.cls == FClass::Inf {
        inf_bits(f, bneg)
    } else if a.cls == FClass::Zero && b.cls == FClass::Zero {
        zero_bits(f, a.neg && bneg)
    } else {
        let e = if a.cls == FClass::Zero {
            b.e
        } else if b.cls == FClass::Zero {
            a.e
        } else {
            a.e.min(b.e)
        };
        let av = if a.cls == FClass::Zero { BigInt::from(0) } else { BigInt::from(a.mag) << (a.e - e) as usize };
        let bv = if b.cls == FClass::Zero { BigInt::from(0) } else { BigInt::from(b.mag) << (b.e - e) as usize };
        let s = (if a.neg { -av } else { av }) + (if bneg { -bv } else { bv });
        match s.sign() {
            Sign::NoSign => zero_bits(f, false),
            Sign::Minus => fround_big(f, true, &(-s), e as i64),
            Sign::Plus => fround_big(f, false, &s, e as i64),
        }
    }
}

/// x + y, exact. // model: core/numerics/fp.sail fp_add
pub fn fp_add_big(f: Fmt, x: u32, y: u32) -> u32 {
    fadd_core_big(f, fdecode(f, x), fdecode(f, y), false)
}

/// x - y, exact. // model: core/numerics/fp.sail fp_sub
pub fn fp_sub_big(f: Fmt, x: u32, y: u32) -> u32 {
    fadd_core_big(f, fdecode(f, x), fdecode(f, y), true)
}

/// x * y, exact. // model: core/numerics/fp.sail fp_mul
pub fn fp_mul_big(f: Fmt, x: u32, y: u32) -> u32 {
    let a = fdecode(f, x);
    let b = fdecode(f, y);
    let neg = a.neg != b.neg;
    let inf_times_zero =
        (a.cls == FClass::Inf && b.cls == FClass::Zero) || (a.cls == FClass::Zero && b.cls == FClass::Inf);
    if a.cls == FClass::NaN || b.cls == FClass::NaN || inf_times_zero {
        canon_nan(f)
    } else if a.cls == FClass::Inf || b.cls == FClass::Inf {
        inf_bits(f, neg)
    } else if a.cls == FClass::Zero || b.cls == FClass::Zero {
        zero_bits(f, neg)
    } else {
        fround_big(f, neg, &(BigInt::from(a.mag) * BigInt::from(b.mag)), (a.e + b.e) as i64)
    }
}
