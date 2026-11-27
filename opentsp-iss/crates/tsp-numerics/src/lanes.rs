//! Per-lane VXM kernels for i8/f16/i32/f32 and the other element types
//! (`core/numerics/lanes.sail`, plus `apply` and the Reduce fold of `slices/vxm/vxm_insts.sail`).
//!
//! An element value is a `u32`, zero-extended for the narrower types. Opcodes are the VXM opcodes
//! of the encoding: binary 0..13, unary 14..24, conversions 25..32.

use crate::fp::{self, Fmt, FMT16, FMT32, FMT_BF16, FMT_E4M3, FMT_E5M2};
use crate::transcendental;
use crate::types::EType;

fn pow2(n: u32) -> i128 {
    1i128 << n
}

/// Wrap to a signed `nbits`-bit value. // model: core/numerics/lanes.sail wrap_s
pub fn wrap_s(v: i128, nbits: u32) -> i128 {
    let m = pow2(nbits);
    let u2 = v.rem_euclid(m);
    if u2 >= pow2(nbits - 1) {
        u2 - m
    } else {
        u2
    }
}

/// Saturate to a signed `nbits`-bit value. // model: core/numerics/lanes.sail sat_s
pub fn sat_s(v: i128, nbits: u32) -> i128 {
    let hi = pow2(nbits - 1) - 1;
    let lo = -pow2(nbits - 1);
    if v > hi {
        hi
    } else if v < lo {
        lo
    } else {
        v
    }
}

/// A signed value (in range) as `nbits` bits, zero-extended. // model: core/numerics/lanes.sail bits_of_s
pub fn bits_of_s(v: i128, nbits: u32) -> u32 {
    (if v < 0 { v + pow2(nbits) } else { v }) as u32
}

/// The signed value of the low `nbits` bits. // model: core/numerics/lanes.sail s_of_bits
pub fn s_of_bits(x: u32, nbits: u32) -> i64 {
    let u = (x as u64 & ((1u64 << nbits) - 1)) as i64;
    if u >= 1i64 << (nbits - 1) {
        u - (1i64 << nbits)
    } else {
        u
    }
}

/// The format of a float element type (f16 -> binary16, e4m3, e5m2, anything else binary32).
/// // model: core/numerics/lanes.sail fmt_of
pub fn fmt_of(t: EType) -> Fmt {
    match t {
        EType::F16 => FMT16,
        EType::Bf16 => FMT_BF16,
        EType::E4M3 => FMT_E4M3,
        EType::E5M2 => FMT_E5M2,
        _ => FMT32,
    }
}

/// An integer element's value, signed or unsigned per its type. // model: core/numerics/lanes.sail int_of
pub fn int_of(t: EType, x: u32) -> i128 {
    if t.is_unsigned() {
        (x as u64 & ((1u64 << t.nbits()) - 1)) as i128
    } else {
        s_of_bits(x, t.nbits()) as i128
    }
}

/// Wrap to the type's range. // model: core/numerics/lanes.sail wrap_t
pub fn wrap_t(t: EType, v: i128) -> i128 {
    if t.is_unsigned() {
        v.rem_euclid(pow2(t.nbits()))
    } else {
        wrap_s(v, t.nbits())
    }
}

/// Saturate to the type's range. // model: core/numerics/lanes.sail sat_t
pub fn sat_t(t: EType, v: i128) -> i128 {
    if t.is_unsigned() {
        if v < 0 {
            0
        } else if v >= pow2(t.nbits()) {
            pow2(t.nbits()) - 1
        } else {
            v
        }
    } else {
        sat_s(v, t.nbits())
    }
}

/// An in-range value as the type's bits. // model: core/numerics/lanes.sail bits_t
pub fn bits_t(t: EType, v: i128) -> u32 {
    if t.is_unsigned() {
        v as u32
    } else {
        bits_of_s(v, t.nbits())
    }
}

/// All ones of the element width. // model: core/numerics/lanes.sail all_ones
pub fn all_ones(t: EType) -> u32 {
    ((1u64 << t.nbits()) - 1) as u32
}

fn bool_bits(t: EType, b: bool) -> u32 {
    if b {
        all_ones(t)
    } else {
        0
    }
}

/// Binary arithmetic and logic, opcodes 0..13. // model: core/numerics/lanes.sail k_binary
pub fn k_binary(op: u32, t: EType, x: u32, y: u32) -> u32 {
    if t.is_bool() {
        let a = x & all_ones(t) != 0;
        let b = y & all_ones(t) != 0;
        match op {
            11 => bool_bits(t, a && b),
            12 => bool_bits(t, a || b),
            13 => bool_bits(t, a != b),
            _ => 0,
        }
    } else if t.is_float() {
        let f = fmt_of(t);
        match op {
            0 => fp::fp_add(f, x, y),
            3 => fp::fp_sub(f, x, y),
            6 => fp::fp_mul(f, x, y),
            9 => fp::fp_max(f, x, y),
            10 => fp::fp_min(f, x, y),
            _ => 0,
        }
    } else {
        let mask = all_ones(t);
        // Integer lanes: every value fits i64, and wrap-then-encode is the low `nbits` of the exact
        // result, so the i128 reference arithmetic (`wrap_t`, `sat_t`, `bits_t`) is needed only for
        // the saturating product, whose exact result may exceed i64.
        let (a, b) = (int64_of(t, x), int64_of(t, y));
        match op {
            0 | 2 => (a.wrapping_add(b) as u32) & mask,
            1 => bits64(t, sat64(t, a + b)),
            3 | 5 => (a.wrapping_sub(b) as u32) & mask,
            4 => bits64(t, sat64(t, a - b)),
            6 | 8 => (a.wrapping_mul(b) as u32) & mask,
            7 => bits_t(t, sat_t(t, a as i128 * b as i128)),
            9 => bits64(t, a.max(b)),
            10 => bits64(t, a.min(b)),
            11 => (x & y) & mask,
            12 => (x | y) & mask,
            13 => (x ^ y) & mask,
            _ => 0,
        }
    }
}

/// [`k_binary`] on a vector of 8-bit integer elements (`signed`: i8, else u8), opcodes 0..13, written
/// as plain slice loops so they vectorise. `a`, `b` and `out` have the same length.
pub fn bin8(op: u32, signed: bool, a: &[u8], b: &[u8], out: &mut [u8]) {
    macro_rules! lanes {
        (|$x:ident, $y:ident| $e:expr) => {
            for ((o, &$x), &$y) in out.iter_mut().zip(a).zip(b) {
                *o = $e;
            }
        };
    }
    match (op, signed) {
        (0 | 2, _) => lanes!(|x, y| x.wrapping_add(y)),
        (1, true) => lanes!(|x, y| (x as i8).saturating_add(y as i8) as u8),
        (1, false) => lanes!(|x, y| x.saturating_add(y)),
        (3 | 5, _) => lanes!(|x, y| x.wrapping_sub(y)),
        (4, true) => lanes!(|x, y| (x as i8).saturating_sub(y as i8) as u8),
        (4, false) => lanes!(|x, y| x.saturating_sub(y)),
        (6 | 8, _) => lanes!(|x, y| x.wrapping_mul(y)),
        (7, true) => lanes!(|x, y| ((x as i8 as i16) * (y as i8 as i16)).clamp(-128, 127) as u8),
        (7, false) => lanes!(|x, y| ((x as u16) * (y as u16)).min(255) as u8),
        (9, true) => lanes!(|x, y| (x as i8).max(y as i8) as u8),
        (9, false) => lanes!(|x, y| x.max(y)),
        (10, true) => lanes!(|x, y| (x as i8).min(y as i8) as u8),
        (10, false) => lanes!(|x, y| x.min(y)),
        (11, _) => lanes!(|x, y| x & y),
        (12, _) => lanes!(|x, y| x | y),
        (13, _) => lanes!(|x, y| x ^ y),
        _ => out.fill(0),
    }
}

/// [`int_of`] as an i64 (every integer element type is at most 32 bits).
#[inline]
fn int64_of(t: EType, x: u32) -> i64 {
    let bits = t.nbits();
    if t.is_unsigned() {
        (x as u64 & ((1u64 << bits) - 1)) as i64
    } else {
        s_of_bits(x, bits)
    }
}

/// [`sat_t`] on an i64.
#[inline]
fn sat64(t: EType, v: i64) -> i64 {
    let bits = t.nbits();
    let (lo, hi) =
        if t.is_unsigned() { (0, (1i64 << bits) - 1) } else { (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1) };
    v.clamp(lo, hi)
}

/// [`bits_t`] on an in-range i64.
#[inline]
fn bits64(t: EType, v: i64) -> u32 {
    (v as u32) & all_ones(t)
}

/// Unary operations: Not 14, Neg 15, Abs 16, ReLU 20. The transcendentals (21..24) are in
/// [`crate::transcendental`]. // model: core/numerics/lanes.sail k_unary
pub fn k_unary(op: u32, t: EType, x: u32) -> u32 {
    if t.is_float() {
        let f = fmt_of(t);
        match op {
            15 => fp::fp_neg(f, x),
            16 => fp::fp_abs(f, x),
            20 => {
                let a = fp::fdecode(f, x);
                if a.cls == fp::FClass::NaN {
                    fp::canon_nan(f)
                } else if a.neg || a.cls == fp::FClass::Zero {
                    fp::zero_bits(f, false)
                } else {
                    x & all_ones(t) // no bits above the format width
                }
            }
            _ => 0,
        }
    } else if t.is_bool() {
        // only Not is defined on a boolean (logical); every other unary kernel gives 0
        if op == 14 {
            bool_bits(t, x & all_ones(t) == 0)
        } else {
            0
        }
    } else {
        // signed or unsigned per the element type, like k_binary
        let a = int_of(t, x);
        match op {
            14 => !x & all_ones(t),
            15 => bits_t(t, wrap_t(t, -a)),
            16 => bits_t(t, wrap_t(t, if a < 0 { -a } else { a })),
            20 => bits_t(t, if a < 0 { 0 } else { a }),
            _ => 0,
        }
    }
}

/// Comparison, modes eq ne lt le gt ge = 0..5. Result: all ones or zero of the element width.
/// // model: core/numerics/lanes.sail k_cmp
pub fn k_cmp(mode: u32, t: EType, x: u32, y: u32) -> u32 {
    let truth: bool = if t.is_bool() {
        let a = x & all_ones(t) != 0;
        let b = y & all_ones(t) != 0;
        if mode == 0 {
            a == b
        } else {
            a != b
        }
    } else if t.is_float() {
        let f = fmt_of(t);
        if fp::is_nan(f, x) || fp::is_nan(f, y) {
            mode == 1
        } else {
            let c = fp::fcmp(f, x, y);
            match mode {
                0 => c == 0,
                1 => c != 0,
                2 => c < 0,
                3 => c <= 0,
                4 => c > 0,
                _ => c >= 0,
            }
        }
    } else {
        let a = int_of(t, x);
        let b = int_of(t, y);
        match mode {
            0 => a == b,
            1 => a != b,
            2 => a < b,
            3 => a <= b,
            4 => a > b,
            _ => a >= b,
        }
    };
    if truth {
        all_ones(t)
    } else {
        0
    }
}

/// Is any bit of the element set? (Mask, Select) // model: core/numerics/lanes.sail k_nonzero
pub fn k_nonzero(t: EType, x: u32) -> bool {
    x & all_ones(t) != 0
}

/// Source type of a conversion opcode 25..32. // model: core/numerics/lanes.sail conv_src
pub fn conv_src(op: u32) -> EType {
    match op {
        25 | 27 | 29 => EType::I8,
        26 | 32 => EType::F32,
        28 | 31 => EType::F16,
        _ => EType::I32,
    }
}

/// Destination type of a conversion opcode 25..32. // model: core/numerics/lanes.sail conv_dst
pub fn conv_dst(op: u32) -> EType {
    match op {
        26 | 28 | 30 => EType::I8,
        25 | 31 => EType::F32,
        27 | 32 => EType::F16,
        _ => EType::I32,
    }
}

/// Conversions, opcodes 25..32. // model: core/numerics/lanes.sail k_convert
pub fn k_convert(op: u32, x: u32) -> u32 {
    match op {
        25 => fp::fp_of_int(FMT32, s_of_bits(x, 8)),
        26 => bits_of_s(fp::int_of_fp(FMT32, x, 8) as i128, 8),
        27 => fp::fp_of_int(FMT16, s_of_bits(x, 8)),
        28 => bits_of_s(fp::int_of_fp(FMT16, x, 8) as i128, 8),
        29 => bits_of_s(s_of_bits(x, 8) as i128, 32),
        30 => bits_of_s(sat_s(s_of_bits(x, 32) as i128, 8), 8),
        31 => fp::fp_convert(FMT16, FMT32, x),
        32 => fp::fp_convert(FMT32, FMT16, x),
        _ => 0,
    }
}

/// Generic conversion between any two element types (VXM Convert): integer to integer
/// saturates; float to integer rounds to nearest even and saturates (NaN to 0); integer to
/// float and float to float round to nearest even; to bool: value != 0 (NaN counts as non-zero);
/// from bool: 1 or 0. // model: core/numerics/lanes.sail k_convert_any
pub fn k_convert_any(src: EType, dst: EType, x: u32) -> u32 {
    let truth: bool =
        if src.is_float() { fp::fdecode(fmt_of(src), x).cls != fp::FClass::Zero } else { x & all_ones(src) != 0 };
    if dst.is_bool() {
        if truth {
            all_ones(dst)
        } else {
            0
        }
    } else if dst.is_float() {
        if src.is_float() {
            fp::fp_convert(fmt_of(src), fmt_of(dst), x)
        } else {
            let v: i64 = if src.is_bool() { truth as i64 } else { int_of(src, x) as i64 };
            fp::fp_of_int(fmt_of(dst), v)
        }
    } else {
        let v: i128 = if src.is_bool() {
            truth as i128
        } else if src.is_float() {
            let f = fmt_of(src);
            if dst.is_unsigned() {
                (fp::int_of_fp(f, x, dst.nbits() + 1) as i128).max(0)
            } else {
                fp::int_of_fp(f, x, dst.nbits()) as i128
            }
        } else {
            int_of(src, x)
        };
        bits_t(dst, sat_t(dst, v))
    }
}

/// The per-lane operation of a VXM instruction. // model: slices/vxm/vxm_insts.sail union vxm_lane_op
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneOp {
    /// binary opcode 0..13
    Bin(u32, EType),
    /// unary opcode 14..24
    Un(u32, EType),
    /// dst = mask != 0 ? lhs : 0
    Mask(EType),
    /// dst = mask != 0 ? lhs : rhs
    Select(EType),
    /// comparison mode 0..5; all ones or zero
    Cmp(u32, EType),
    /// conversion opcode 25..32
    Conv(u32),
    /// generic conversion (Convert 34)
    Cvt(EType, EType),
}

/// Apply a lane operation to the operand elements `a`, `b` and the mask element `m`.
/// The transcendentals use the memoised fast path of [`crate::transcendental`].
/// // model: slices/vxm/vxm_insts.sail apply
pub fn apply(op: LaneOp, a: u32, b: u32, m: u32) -> u32 {
    match op {
        LaneOp::Bin(o, t) => k_binary(o, t, a, b),
        LaneOp::Un(o, t) => {
            if (21..=24).contains(&o) {
                if t.is_float() {
                    transcendental::fp_transcendental(o, fmt_of(t), a)
                } else {
                    transcendental::i8_transcendental(o, a)
                }
            } else {
                k_unary(o, t, a)
            }
        }
        LaneOp::Mask(t) => {
            if k_nonzero(t, b) {
                a
            } else {
                0
            }
        }
        LaneOp::Select(t) => {
            if k_nonzero(t, m) {
                a
            } else {
                b
            }
        }
        LaneOp::Cmp(o, t) => k_cmp(o, t, a, b),
        LaneOp::Conv(o) => k_convert(o, a),
        LaneOp::Cvt(st, dt) => k_convert_any(st, dt, a),
    }
}

/// binary16 to binary32, exact: [`fp::fp_convert`] `(FMT16, FMT32)` as integer operations. Bits above 15 are ignored.
pub fn f16_to_f32_bits(x: u32) -> u32 {
    let sign = (x & 0x8000) << 16;
    let e = (x >> 10) & 0x1f;
    let m = x & 0x3ff;
    if e == 0x1f {
        if m != 0 {
            fp::canon_nan(FMT32)
        } else {
            sign | 0x7f80_0000
        }
    } else if e != 0 {
        sign | ((e + 112) << 23) | (m << 13)
    } else if m == 0 {
        sign
    } else {
        // subnormal: m * 2^-24, normalised
        let p = 31 - m.leading_zeros();
        sign | ((p + 103) << 23) | ((m << (23 - p)) & 0x7f_ffff)
    }
}

/// binary32 to binary16, round to nearest even, NaN canonical: [`fp::fp_convert`] `(FMT32, FMT16)` as integer
/// operations.
pub fn f32_to_f16_bits(x: u32) -> u32 {
    let sign = (x >> 16) & 0x8000;
    let e = (x >> 23) & 0xff;
    let m = x & 0x7f_ffff;
    if e == 0xff {
        return if m != 0 { fp::canon_nan(FMT16) } else { sign | 0x7c00 };
    }
    let ex = e as i32 - 112; // the f16 exponent field of a normal result
    if ex >= 0x1f {
        return sign | 0x7c00;
    }
    let (mut q, rem, half) = if ex >= 1 {
        (((ex as u32) << 10) | (m >> 13), m & 0x1fff, 0x1000)
    } else if ex >= -10 {
        // an f16 subnormal (or zero): the 24-bit significand shifted down
        let mant = m | 0x80_0000;
        let shift = (14 - ex) as u32;
        (mant >> shift, mant & ((1 << shift) - 1), 1 << (shift - 1))
    } else {
        return sign;
    };
    if rem > half || (rem == half && q & 1 == 1) {
        q += 1; // a carry out of the significand lands in the exponent, up to infinity
    }
    sign | q
}

/// A binary32 maxNum (`max`) or minNum with -0 < +0, as [`fp::fp_max`] and [`fp::fp_min`] define them.
#[inline(always)]
fn f32_max_min(x: u32, y: u32, max: bool) -> u32 {
    let (fx, fy) = (f32::from_bits(x), f32::from_bits(y));
    if fx.is_nan() || fy.is_nan() {
        return match (fx.is_nan(), fy.is_nan()) {
            (true, true) => fp::canon_nan(FMT32),
            (true, false) => y,
            _ => x,
        };
    }
    let neg_x = x >> 31 == 1;
    if max {
        if fx > fy {
            x
        } else if fx < fy || neg_x {
            y
        } else {
            x
        }
    } else if fx < fy {
        x
    } else if fx > fy {
        y
    } else if neg_x {
        x
    } else {
        y
    }
}

/// [`apply`] on a vector of lanes: `out[l] = apply(op, a[l], b[l], m[l])` for every lane, with the f32 add, sub,
/// mul, max and min, the f16/f32 conversions and the f16 transcendentals (one table for the vector) as plain loops
/// and every other operation lane by lane through [`apply`]. The four slices have one length.
pub fn apply_vec(op: LaneOp, a: &[u32], b: &[u32], m: &[u32], out: &mut [u32]) {
    let n = out.len();
    assert!(a.len() == n && b.len() == n && m.len() == n, "apply_vec: operand lengths differ");
    macro_rules! each2 {
        (|$x:ident, $y:ident| $e:expr) => {
            for ((o, &$x), &$y) in out.iter_mut().zip(a).zip(b) {
                *o = $e;
            }
        };
    }
    fn canon(r: f32) -> u32 {
        if r.is_nan() {
            fp::canon_nan(FMT32)
        } else {
            r.to_bits()
        }
    }
    match op {
        LaneOp::Bin(0, EType::F32) => each2!(|x, y| canon(f32::from_bits(x) + f32::from_bits(y))),
        LaneOp::Bin(3, EType::F32) => each2!(|x, y| canon(f32::from_bits(x) - f32::from_bits(y))),
        LaneOp::Bin(6, EType::F32) => each2!(|x, y| canon(f32::from_bits(x) * f32::from_bits(y))),
        LaneOp::Bin(9, EType::F32) => each2!(|x, y| f32_max_min(x, y, true)),
        LaneOp::Bin(10, EType::F32) => each2!(|x, y| f32_max_min(x, y, false)),
        LaneOp::Conv(31) => {
            for (o, &x) in out.iter_mut().zip(a) {
                *o = f16_to_f32_bits(x);
            }
        }
        LaneOp::Conv(32) => {
            for (o, &x) in out.iter_mut().zip(a) {
                *o = f32_to_f16_bits(x);
            }
        }
        LaneOp::Un(o @ 21..=24, EType::F16) => {
            let tab = transcendental::f16_table(o);
            for (r, &x) in out.iter_mut().zip(a) {
                *r = if x >> 16 == 0 { tab[x as usize] as u32 } else { transcendental::fp_transcendental(o, FMT16, x) };
            }
        }
        _ => {
            for l in 0..n {
                out[l] = apply(op, a[l], b[l], m[l]);
            }
        }
    }
}

/// A Reduce operation. // model: slices/vxm/vxm_types.sail reduce_op
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Add,
    Max,
    Min,
}

impl ReduceOp {
    /// The binary opcode the fold uses: Add 0, Max 9, Min 10.
    pub fn kop(self) -> u32 {
        match self {
            ReduceOp::Add => 0,
            ReduceOp::Max => 9,
            ReduceOp::Min => 10,
        }
    }
}

/// The ascending fold of one superlane group: `acc = e[0]; acc = k_binary(kop, t, acc, e[j])`
/// for j = 1.., rounding at each step. `elems` is the group (16 lanes in the
/// model). // model: slices/vxm/vxm_insts.sail execute VXM_REDUCE
pub fn reduce_fold(op: ReduceOp, t: EType, elems: &[u32]) -> u32 {
    let kop = op.kop();
    if op == ReduceOp::Add && t == EType::F32 {
        // The host f32 add is the correctly rounded IEEE add the model defines at every step; a NaN, whichever
        // operand or invalid operation made it, is the one canonical NaN, and a NaN stays a NaN through the
        // remaining adds, so canonicalising once at the end equals canonicalising at each step.
        let mut acc = f32::from_bits(elems[0]);
        for &e in &elems[1..] {
            acc += f32::from_bits(e);
        }
        return if acc.is_nan() { 0x7FC0_0000 } else { acc.to_bits() };
    }
    let mut acc = elems[0];
    for &e in &elems[1..] {
        acc = k_binary(kop, t, acc, e);
    }
    acc
}
