//! Exact IEEE 754 binary16 / binary32 arithmetic (`core/numerics/fp.sail`).
//!
//! A finite value is `sign * mag * 2^e`. Operations compute the exact result and round once
//! (round to nearest even). NaN results are canonical. Values are raw bit patterns in a `u32`
//! (zero-extended for f16).
//!
//! Integer widths: f16 sums and products fit `i128` exactly. For f32, `fp_add`, `fp_sub` and
//! `fp_mul` use the host's IEEE binary32 operations (round to nearest even, no flush to zero on
//! x86-64 and aarch64; the build refuses other targets and [`crate::hostfp`] checks the
//! environment) and canonicalise NaN; the exact big-integer version is in
//! [`crate::bigfp`] and the tests compare the two.

/// A binary format. // model: core/numerics/fp.sail struct fmt
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fmt {
    pub mbits: i32,
    pub ebits: i32,
    pub bias: i32,
    pub emin: i32,
    pub width: i32,
    /// The top exponent code is reserved for infinities and NaNs (binary16, binary32, e5m2). A format with
    /// `ieee == false` (e4m3) has no infinity: only the all-ones pattern is NaN and an overflow saturates.
    pub ieee: bool,
}

/// binary16. // model: core/numerics/fp.sail FMT16
pub const FMT16: Fmt = Fmt { mbits: 10, ebits: 5, bias: 15, emin: -14, width: 16, ieee: true };
/// binary32. // model: core/numerics/fp.sail FMT32
pub const FMT32: Fmt = Fmt { mbits: 23, ebits: 8, bias: 127, emin: -126, width: 32, ieee: true };
/// FP8 e5m2 (IEEE style). // model: core/numerics/fp.sail FMT_E5M2
pub const FMT_E5M2: Fmt = Fmt { mbits: 2, ebits: 5, bias: 15, emin: -14, width: 8, ieee: true };
/// FP8 e4m3 (no infinity, saturating). // model: core/numerics/fp.sail FMT_E4M3
pub const FMT_E4M3: Fmt = Fmt { mbits: 3, ebits: 4, bias: 7, emin: -6, width: 8, ieee: false };
/// bfloat16, a profile extension. // model: core/numerics/fp.sail FMT_BF16
pub const FMT_BF16: Fmt = Fmt { mbits: 7, ebits: 8, bias: 127, emin: -126, width: 16, ieee: true };

/// Class of a decoded value. // model: core/numerics/fp.sail enum fclass
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FClass {
    Zero,
    Finite,
    Inf,
    NaN,
}

/// A decoded value: for `Finite`, `(-1)^neg * mag * 2^e`. // model: core/numerics/fp.sail struct fval
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FVal {
    pub cls: FClass,
    pub neg: bool,
    pub mag: u64,
    pub e: i32,
}

/// floor(log2(n)) for n >= 1. // model: core/numerics/fp.sail ilog2
pub fn ilog2(n: u128) -> i32 {
    debug_assert!(n >= 1);
    127 - n.leading_zeros() as i32
}

/// The canonical NaN of the format. // model: core/numerics/fp.sail canon_nan
pub fn canon_nan(f: Fmt) -> u32 {
    if f.width == 8 {
        if f.ieee {
            0x0000_007E
        } else {
            0x0000_007F
        }
    } else if f.width == 16 {
        if f.ebits == 8 {
            0x0000_7FC0
        } else {
            0x0000_7E00
        }
    } else {
        0x7FC0_0000
    }
}

/// An infinity; for e4m3, which has none, the saturation value (the largest finite magnitude, 448).
/// // model: core/numerics/fp.sail inf_bits
pub fn inf_bits(f: Fmt, neg: bool) -> u32 {
    if f.width == 8 {
        return match (f.ieee, neg) {
            (true, true) => 0x0000_00FC,
            (true, false) => 0x0000_007C,
            (false, true) => 0x0000_00FE,
            (false, false) => 0x0000_007E,
        };
    }
    match (f.width == 16, f.ebits == 8, neg) {
        (true, true, true) => 0x0000_FF80,
        (true, true, false) => 0x0000_7F80,
        (true, false, true) => 0x0000_FC00,
        (true, false, false) => 0x0000_7C00,
        (false, _, true) => 0xFF80_0000,
        (false, _, false) => 0x7F80_0000,
    }
}

/// A zero. // model: core/numerics/fp.sail zero_bits
pub fn zero_bits(f: Fmt, neg: bool) -> u32 {
    if f.width == 8 {
        return if neg { 0x0000_0080 } else { 0 };
    }
    match (f.width == 16, neg) {
        (true, true) => 0x0000_8000,
        (false, true) => 0x8000_0000,
        _ => 0,
    }
}

/// Decode a bit pattern. // model: core/numerics/fp.sail fdecode
pub fn fdecode(f: Fmt, x: u32) -> FVal {
    let neg = (x >> (f.width - 1)) & 1 == 1;
    let ex = (x >> f.mbits) & ((1u32 << f.ebits) - 1);
    let fr = x & ((1u32 << f.mbits) - 1);
    if f.ieee && ex == (1u32 << f.ebits) - 1 {
        FVal { cls: if fr == 0 { FClass::Inf } else { FClass::NaN }, neg, mag: 0, e: 0 }
    } else if !f.ieee && ex == (1u32 << f.ebits) - 1 && fr == (1u32 << f.mbits) - 1 {
        FVal { cls: FClass::NaN, neg, mag: 0, e: 0 }
    } else if ex == 0 && fr == 0 {
        FVal { cls: FClass::Zero, neg, mag: 0, e: 0 }
    } else if ex == 0 {
        FVal { cls: FClass::Finite, neg, mag: fr as u64, e: f.emin - f.mbits }
    } else {
        FVal { cls: FClass::Finite, neg, mag: (fr + (1u32 << f.mbits)) as u64, e: ex as i32 - f.bias - f.mbits }
    }
}

/// Is the pattern a NaN? // model: core/numerics/fp.sail is_nan
pub fn is_nan(f: Fmt, x: u32) -> bool {
    fdecode(f, x).cls == FClass::NaN
}

/// Round `(-1)^neg * mag * 2^e` (mag >= 0) to the format, RNE. // model: core/numerics/fp.sail fround
///
/// `mag` is below 2^127 in every use (f16 dot products: below 2^90).
pub fn fround(f: Fmt, neg: bool, mag: u128, e: i32) -> u32 {
    if mag == 0 {
        return zero_bits(f, neg);
    }
    let p = ilog2(mag);
    let ee = if p + e < f.emin { f.emin } else { p + e };
    let q = ee - f.mbits;
    let mut m: u128;
    if e >= q {
        m = mag << (e - q) as u32;
    } else {
        let d = (q - e) as u32;
        // rounding shift by d >= 1; d >= 128 leaves no bit of mag in the quotient
        if d >= 128 {
            m = if d == 128 && mag > (1u128 << 127) { 1 } else { 0 };
        } else {
            let lo = mag >> d;
            let rem = mag - (lo << d);
            let half = 1u128 << (d - 1);
            m = if rem > half || (rem == half && lo & 1 == 1) { lo + 1 } else { lo };
        }
    }
    let mut ex = ee;
    if m == 1u128 << (f.mbits + 1) {
        m /= 2;
        ex += 1;
    }
    let sign: u32 = if neg { 1u32 << (f.width - 1) } else { 0 };
    if m == 0 {
        zero_bits(f, neg)
    } else if m < (1u128 << f.mbits) {
        // subnormal: biased exponent 0
        sign + m as u32
    } else if (f.ieee && ex + f.bias >= (1 << f.ebits) - 1)
        || (!f.ieee
            && (ex + f.bias > (1 << f.ebits) - 1
                || (ex + f.bias == (1 << f.ebits) - 1 && m as u32 - (1u32 << f.mbits) == (1u32 << f.mbits) - 1)))
    {
        // IEEE formats overflow to infinity; e4m3 saturates to 448 (the all-ones fraction at the top exponent is NaN)
        inf_bits(f, neg)
    } else {
        sign + ((ex + f.bias) as u32) * (1u32 << f.mbits) + (m as u32 - (1u32 << f.mbits))
    }
}

/// Sum or difference of two decoded values, exact then rounded; f16 only (`i128`).
/// // model: core/numerics/fp.sail fadd_core
pub fn fadd_core(f: Fmt, a: FVal, b: FVal, negate_b: bool) -> u32 {
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
        debug_assert!((a.e - e).max(b.e - e) <= 100, "fadd_core is for f16 operands");
        let av: i128 = if a.cls == FClass::Zero { 0 } else { (a.mag as i128) << (a.e - e) as u32 };
        let bv: i128 = if b.cls == FClass::Zero { 0 } else { (b.mag as i128) << (b.e - e) as u32 };
        let s = (if a.neg { -av } else { av }) + (if bneg { -bv } else { bv });
        if s == 0 {
            zero_bits(f, false) // RNE: an exact zero sum of opposite signs is +0
        } else if s < 0 {
            fround(f, true, (-s) as u128, e)
        } else {
            fround(f, false, s as u128, e)
        }
    }
}

fn f32_canon(r: f32) -> u32 {
    if r.is_nan() {
        canon_nan(FMT32)
    } else {
        r.to_bits()
    }
}

/// x + y. // model: core/numerics/fp.sail fp_add
pub fn fp_add(f: Fmt, x: u32, y: u32) -> u32 {
    if f.width == 32 {
        f32_canon(f32::from_bits(x) + f32::from_bits(y))
    } else if f.ebits == 8 {
        // bf16: the exponents of two operands differ by up to 260, beyond the `i128` alignment of `fadd_core`
        crate::bigfp::fadd_core_big(f, fdecode(f, x), fdecode(f, y), false)
    } else {
        fadd_core(f, fdecode(f, x), fdecode(f, y), false)
    }
}

/// x - y. // model: core/numerics/fp.sail fp_sub
pub fn fp_sub(f: Fmt, x: u32, y: u32) -> u32 {
    if f.width == 32 {
        f32_canon(f32::from_bits(x) - f32::from_bits(y))
    } else if f.ebits == 8 {
        crate::bigfp::fadd_core_big(f, fdecode(f, x), fdecode(f, y), true)
    } else {
        fadd_core(f, fdecode(f, x), fdecode(f, y), true)
    }
}

/// x * y. // model: core/numerics/fp.sail fp_mul
pub fn fp_mul(f: Fmt, x: u32, y: u32) -> u32 {
    if f.width == 32 {
        return f32_canon(f32::from_bits(x) * f32::from_bits(y));
    }
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
        fround(f, neg, a.mag as u128 * b.mag as u128, a.e + b.e)
    }
}

/// Three-way comparison of non-NaN values: -1, 0, 1; -0 equals +0. // model: core/numerics/fp.sail fcmp
pub fn fcmp(f: Fmt, x: u32, y: u32) -> i32 {
    let a = fdecode(f, x);
    let b = fdecode(f, y);
    let sa: i32 = if a.cls == FClass::Zero {
        0
    } else if a.neg {
        -1
    } else {
        1
    };
    let sb: i32 = if b.cls == FClass::Zero {
        0
    } else if b.neg {
        -1
    } else {
        1
    };
    if sa != sb {
        if sa < sb {
            -1
        } else {
            1
        }
    } else if sa == 0 {
        0
    } else {
        // same sign, both non-zero
        let mag_cmp: i32 = if a.cls == FClass::Inf && b.cls == FClass::Inf {
            0
        } else if a.cls == FClass::Inf {
            1
        } else if b.cls == FClass::Inf {
            -1
        } else {
            // Compare mag * 2^e. The leading bit positions decide unless equal; then the
            // exponents differ by at most mbits + 1, so the aligned magnitudes fit a u128.
            let ta = ilog2(a.mag as u128) + a.e;
            let tb = ilog2(b.mag as u128) + b.e;
            if ta != tb {
                if ta < tb {
                    -1
                } else {
                    1
                }
            } else {
                let e = a.e.min(b.e);
                let av = (a.mag as u128) << (a.e - e) as u32;
                let bv = (b.mag as u128) << (b.e - e) as u32;
                if av < bv {
                    -1
                } else if av > bv {
                    1
                } else {
                    0
                }
            }
        };
        if sa > 0 {
            mag_cmp
        } else {
            -mag_cmp
        }
    }
}

/// maxNum with -0 < +0. // model: core/numerics/fp.sail fp_max
pub fn fp_max(f: Fmt, x: u32, y: u32) -> u32 {
    let nx = is_nan(f, x);
    let ny = is_nan(f, y);
    if nx && ny {
        canon_nan(f)
    } else if nx {
        y
    } else if ny {
        x
    } else {
        let c = fcmp(f, x, y);
        if c > 0 {
            x
        } else if c < 0 || fdecode(f, x).neg {
            y
        } else {
            x
        }
    }
}

/// minNum with -0 < +0. // model: core/numerics/fp.sail fp_min
pub fn fp_min(f: Fmt, x: u32, y: u32) -> u32 {
    let nx = is_nan(f, x);
    let ny = is_nan(f, y);
    if nx && ny {
        canon_nan(f)
    } else if nx {
        y
    } else if ny {
        x
    } else {
        let c = fcmp(f, x, y);
        if c < 0 {
            x
        } else if c > 0 {
            y
        } else if fdecode(f, x).neg {
            x
        } else {
            y
        }
    }
}

/// Negation; NaN becomes canonical. // model: core/numerics/fp.sail fp_neg
pub fn fp_neg(f: Fmt, x: u32) -> u32 {
    if is_nan(f, x) {
        canon_nan(f)
    } else {
        // (x + 2^(width-1)) mod 2^width, with x < 2^width
        (x ^ (1u32 << (f.width - 1))) & (((1u64 << f.width) - 1) as u32)
    }
}

/// Absolute value; NaN becomes canonical. // model: core/numerics/fp.sail fp_abs
pub fn fp_abs(f: Fmt, x: u32) -> u32 {
    if is_nan(f, x) {
        canon_nan(f)
    } else {
        x & (((1u64 << (f.width - 1)) - 1) as u32)
    }
}

/// Signed integer to float, RNE. // model: core/numerics/fp.sail fp_of_int
pub fn fp_of_int(f: Fmt, v: i64) -> u32 {
    if v < 0 {
        fround(f, true, (-(v as i128)) as u128, 0)
    } else {
        fround(f, false, v as u128, 0)
    }
}

/// Float to a signed integer of `nbits` bits: RNE, saturating, NaN to 0. // model: core/numerics/fp.sail int_of_fp
pub fn int_of_fp(f: Fmt, x: u32, nbits: u32) -> i64 {
    let a = fdecode(f, x);
    let hi: i64 = (1i64 << (nbits - 1)) - 1;
    let lo: i64 = -(1i64 << (nbits - 1));
    match a.cls {
        FClass::NaN => 0,
        FClass::Zero => 0,
        FClass::Inf => {
            if a.neg {
                lo
            } else {
                hi
            }
        }
        FClass::Finite => {
            let r: i128 = if a.e >= 0 {
                // mag < 2^24; anything at or above 2^64 saturates (nbits <= 33)
                if a.e >= 64 {
                    1i128 << 100
                } else {
                    (a.mag as i128) << a.e as u32
                }
            } else {
                let sh = (-a.e) as u32;
                // mag < 2^24 < half = 2^(sh-1) once sh >= 26: the quotient is 0 and so is the result
                if sh >= 100 {
                    0
                } else {
                    let d = 1u128 << sh;
                    let q = a.mag as u128 / d;
                    let rem = a.mag as u128 - q * d;
                    let half = d / 2;
                    (if rem > half || (rem == half && q % 2 == 1) { q + 1 } else { q }) as i128
                }
            };
            let s = if a.neg { -r } else { r };
            if s > hi as i128 {
                hi
            } else if s < lo as i128 {
                lo
            } else {
                s as i64
            }
        }
    }
}

/// Format conversion (f16 <-> f32), RNE, NaN canonical. // model: core/numerics/fp.sail fp_convert
pub fn fp_convert(src: Fmt, dst: Fmt, x: u32) -> u32 {
    let a = fdecode(src, x);
    match a.cls {
        FClass::NaN => canon_nan(dst),
        FClass::Inf => inf_bits(dst, a.neg),
        FClass::Zero => zero_bits(dst, a.neg),
        FClass::Finite => fround(dst, a.neg, a.mag as u128, a.e),
    }
}
