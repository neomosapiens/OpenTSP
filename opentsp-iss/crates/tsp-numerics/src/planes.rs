//! The VXM lane kernels on byte planes: plane b holds byte b of every lane's element, so a lane's
//! `u32` is `p0 | p1 << 8 | p2 << 16 | p3 << 24`. [`apply_planes`] is [`crate::lanes::apply`] for every
//! lane of an operand set, reading the planes and writing the result planes without a word-array
//! detour for the hot operations (f32 add, sub, mul, max, min, and the f16/f32 conversions); on
//! AArch64 these run as NEON loops, and on x86-64 with SSE4.1 and F16C as SSE loops. The fallback uses
//! [`crate::lanes::apply_vec`]. Every result equals `lanes::apply` lane by lane: the per-lane routine
//! stays in the tree as the reference and `tests/planes_vec.rs` compares the two.

use crate::lanes::{self, LaneOp};

/// The most lanes of a vector: 32 superlanes of 16 lanes (`core/geometry.sail`).
pub const MAX_LANES: usize = 512;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use crate::types::EType;

/// The `n` elements of a `ws`-byte operand as words (`ws` <= 4): byte b of lane l is plane b.
pub fn planes_to_words(p: &[&[u8]; 4], ws: usize, n: usize, out: &mut [u32]) {
    assert!(ws <= 4 && out.len() >= n && p.iter().all(|x| x.len() >= n), "planes_to_words: lengths");
    let done = n & !15;
    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;
        while i < done {
            // SAFETY: every plane holds at least `n` >= i + 16 bytes and `out` at least `n` words (asserted above).
            unsafe { neon::to_words(p, ws, i, &mut out[i..i + 16]) };
            i += 16;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        out[..done].fill(0);
        for (b, plane) in p.iter().enumerate().take(ws) {
            for (o, &x) in out[..done].iter_mut().zip(&plane[..done]) {
                *o |= (x as u32) << (8 * b);
            }
        }
    }
    for l in done..n {
        out[l] = 0;
        for (b, plane) in p.iter().enumerate().take(ws) {
            out[l] |= (plane[l] as u32) << (8 * b);
        }
    }
}

/// Byte b of lane l of `xs` goes to plane b, for the first `wd` planes (`wd` <= 4) and the first `n` lanes.
pub fn words_to_planes(xs: &[u32], wd: usize, n: usize, out: &mut [&mut [u8]; 4]) {
    assert!(wd <= 4 && xs.len() >= n && out.iter().all(|x| x.len() >= n), "words_to_planes: lengths");
    let done = n & !15;
    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;
        while i < done {
            // SAFETY: `xs` holds at least `n` >= i + 16 words and every plane at least `n` bytes (asserted above).
            unsafe { neon::from_words(&xs[i..i + 16], wd, i, out) };
            i += 16;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (b, plane) in out.iter_mut().enumerate().take(wd) {
        for (o, &x) in plane[..done].iter_mut().zip(&xs[..done]) {
            *o = (x >> (8 * b)) as u8;
        }
    }
    for l in done..n {
        for (b, plane) in out.iter_mut().enumerate().take(wd) {
            plane[l] = (xs[l] >> (8 * b)) as u8;
        }
    }
}

/// `out[l] = lanes::apply(op, a[l], b[l], m[l])` for every lane `l` of the operand planes, where `ws` is the width
/// in bytes of the operands and `wd` the width of the result; only the first `wd` result planes are written. The
/// planes of `a`, `b` and `m` hold `n` lanes each (the planes at or above `ws` are not read).
pub fn apply_planes(
    op: LaneOp,
    ws: usize,
    wd: usize,
    a: &[&[u8]; 4],
    b: &[&[u8]; 4],
    m: &[&[u8]; 4],
    out: &mut [&mut [u8]; 4],
) {
    let n = a[0].len();
    assert!(ws <= 4 && wd <= 4, "apply_planes: widths");
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    if let LaneOp::Bin(k @ (0 | 3 | 6), EType::F16) = op {
        if ws == 2
            && wd == 2
            && (16..=MAX_LANES).contains(&n)
            && std::arch::is_aarch64_feature_detected!("fp16")
            && neon::half_environment_ok()
        {
            assert!(a.iter().chain(b).chain(m).all(|p| p.len() >= n), "planes_to_words: lengths");
            assert!(out.iter().all(|p| p.len() >= n), "words_to_planes: lengths");
            let done = n & !15;
            // SAFETY: the runtime checks establish FP16 support and the default
            // mode. All operands and outputs cover every full block.
            unsafe {
                match k {
                    0 => neon::half_binary::<0>(a, b, out, done),
                    3 => neon::half_binary::<3>(a, b, out, done),
                    6 => neon::half_binary::<6>(a, b, out, done),
                    _ => unreachable!(),
                }
            }
            tail(op, ws, wd, a, b, m, out, done, n);
            return;
        }
    }
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
        let fused = match op {
            LaneOp::Bin(o @ (0 | 3 | 6 | 9 | 10), EType::F32) if ws == 4 && wd == 4 => Some(o),
            LaneOp::Conv(31) if ws == 2 && wd == 4 => Some(31),
            LaneOp::Conv(32) if ws == 4 && wd == 2 => Some(32),
            _ => None,
        };
        #[cfg(target_arch = "x86_64")]
        let fused = fused.filter(|_| std::is_x86_feature_detected!("sse4.1") && std::is_x86_feature_detected!("f16c"));
        if let Some(k) = fused {
            let done = n & !15;
            assert!(
                a.iter().all(|x| x.len() >= n) && b.iter().all(|x| x.len() >= n) && out.iter().all(|x| x.len() >= n),
                "apply_planes: lengths"
            );
            if done > 0 {
                // SAFETY: every plane read or written holds at least `n` >= `done` bytes (asserted above), and the
                // kernel touches lanes below `done` only, in blocks of 16.
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    neon::fused(k, a, b, out, done)
                };
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    x86::fused(k, a, b, out, done)
                };
            }
            if done < n {
                tail(op, ws, wd, a, b, m, out, done, n);
            }
            return;
        }
    }
    let (mut wa, mut wb, mut wm, mut wo) = ([0u32; MAX_LANES], [0u32; MAX_LANES], [0u32; MAX_LANES], [0u32; MAX_LANES]);
    assert!(n <= MAX_LANES, "apply_planes: at most {MAX_LANES} lanes");
    planes_to_words(a, ws, n, &mut wa);
    planes_to_words(b, ws, n, &mut wb);
    planes_to_words(m, ws, n, &mut wm);
    lanes::apply_vec(op, &wa[..n], &wb[..n], &wm[..n], &mut wo[..n]);
    words_to_planes(&wo, wd, n, out);
}

/// Lanes `from..n` through the per-lane routine (the part of a vector that is not a multiple of 16).
#[allow(clippy::too_many_arguments)] // the arguments of apply_planes and the range
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn tail(
    op: LaneOp,
    ws: usize,
    wd: usize,
    a: &[&[u8]; 4],
    b: &[&[u8]; 4],
    m: &[&[u8]; 4],
    out: &mut [&mut [u8]; 4],
    from: usize,
    n: usize,
) {
    let word = |p: &[&[u8]; 4], l: usize| (0..ws).fold(0u32, |x, k| x | (p[k][l] as u32) << (8 * k));
    for l in from..n {
        let r = lanes::apply(op, word(a, l), word(b, l), word(m, l));
        for (k, plane) in out.iter_mut().enumerate().take(wd) {
            plane[l] = (r >> (8 * k)) as u8;
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    // Four consecutive bytes from one plane occupy the low byte of four words.
    #[target_feature(enable = "sse4.1,f16c")]
    unsafe fn load(p: &[&[u8]; 4], width: usize, at: usize) -> __m128i {
        let plane = |i: usize| {
            let word = i32::from_le_bytes(p[i][at..at + 4].try_into().unwrap());
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128(word))
        };
        let lo = _mm_or_si128(plane(0), _mm_slli_epi32::<8>(plane(1)));
        if width == 2 {
            lo
        } else {
            _mm_or_si128(lo, _mm_or_si128(_mm_slli_epi32::<16>(plane(2)), _mm_slli_epi32::<24>(plane(3))))
        }
    }

    #[target_feature(enable = "sse4.1,f16c")]
    pub unsafe fn fused(op: u32, a: &[&[u8]; 4], b: &[&[u8]; 4], out: &mut [&mut [u8]; 4], n: usize) {
        let nan = _mm_castsi128_ps(_mm_set1_epi32(0x7fc00000));
        let canonical = |v| _mm_blendv_ps(v, nan, _mm_cmpunord_ps(v, v));
        let selectors = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        for at in (0..n).step_by(4) {
            let x = load(a, if op == 31 { 2 } else { 4 }, at);
            let xf = _mm_castsi128_ps(x);
            let result = match op {
                31 => _mm_castps_si128(canonical(_mm_cvtph_ps(_mm_packus_epi32(x, x)))),
                32 => {
                    let half = _mm_cvtps_ph::<0>(xf);
                    let raw = _mm_cvtepu16_epi32(half);
                    _mm_blendv_epi8(raw, _mm_set1_epi32(0x7e00), _mm_castps_si128(_mm_cmpunord_ps(xf, xf)))
                }
                _ => {
                    let yf = _mm_castsi128_ps(load(b, 4, at));
                    let value = match op {
                        0 => _mm_add_ps(xf, yf),
                        3 => _mm_sub_ps(xf, yf),
                        6 => _mm_mul_ps(xf, yf),
                        _ => {
                            // Equal values differ only for signed zeros. A sign survives min, both signs must agree for max.
                            let (ordered, tie) = if op == 9 {
                                (_mm_max_ps(xf, yf), _mm_and_ps(xf, yf))
                            } else {
                                (_mm_min_ps(xf, yf), _mm_or_ps(xf, yf))
                            };
                            let v = _mm_blendv_ps(ordered, tie, _mm_cmpeq_ps(xf, yf));
                            _mm_blendv_ps(v, xf, _mm_cmpunord_ps(yf, yf))
                        }
                    };
                    _mm_castps_si128(canonical(value))
                }
            };
            for (byte, plane) in out.iter_mut().take(if op == 32 { 2 } else { 4 }).enumerate() {
                let pick = _mm_add_epi8(selectors, _mm_set1_epi8(byte as i8));
                let bytes = _mm_cvtsi128_si32(_mm_shuffle_epi8(result, pick)).to_le_bytes();
                plane[at..at + 4].copy_from_slice(&bytes);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    #[cfg(target_vendor = "apple")]
    pub(super) fn half_environment_ok() -> bool {
        let control: u64;
        // SAFETY: Reading this thread's floating-point control register does not modify it.
        unsafe {
            core::arch::asm!("mrs {control}, fpcr", control = out(reg) control, options(nostack, nomem));
        }
        control == 0
    }

    #[cfg(target_vendor = "apple")]
    #[target_feature(enable = "fp16")]
    pub(super) unsafe fn half_binary<const OP: u32>(
        a: &[&[u8]; 4],
        b: &[&[u8]; 4],
        out: &mut [&mut [u8]; 4],
        body: usize,
    ) {
        let flags: u64;
        core::arch::asm!("mrs {flags}, fpsr", flags = out(reg) flags, options(nostack, nomem));
        let nan_bits = vdupq_n_u16(0x7e00);
        let magnitude = vdupq_n_u16(0x7fff);
        let infinity = vdupq_n_u16(0x7c00);
        for i in (0..body).step_by(16) {
            let xa = vld1q_u8(a[0].as_ptr().add(i));
            let xb = vld1q_u8(a[1].as_ptr().add(i));
            let ya = vld1q_u8(b[0].as_ptr().add(i));
            let yb = vld1q_u8(b[1].as_ptr().add(i));
            let x = [vzip1q_u8(xa, xb), vzip2q_u8(xa, xb)];
            let y = [vzip1q_u8(ya, yb), vzip2q_u8(ya, yb)];
            let result: [uint8x16_t; 2] = std::array::from_fn(|j| {
                let value: uint16x8_t;
                match OP {
                    0 => core::arch::asm!("fadd {dst:v}.8h, {a:v}.8h, {b:v}.8h",
                        dst = lateout(vreg) value, a = in(vreg) x[j], b = in(vreg) y[j],
                        options(nostack, nomem)),
                    3 => core::arch::asm!("fsub {dst:v}.8h, {a:v}.8h, {b:v}.8h",
                        dst = lateout(vreg) value, a = in(vreg) x[j], b = in(vreg) y[j],
                        options(nostack, nomem)),
                    6 => core::arch::asm!("fmul {dst:v}.8h, {a:v}.8h, {b:v}.8h",
                        dst = lateout(vreg) value, a = in(vreg) x[j], b = in(vreg) y[j],
                        options(nostack, nomem)),
                    _ => unreachable!(),
                }
                let nan = vcgtq_u16(vandq_u16(value, magnitude), infinity);
                vreinterpretq_u8_u16(vbslq_u16(nan, nan_bits, value))
            });
            vst1q_u8(out[0].as_mut_ptr().add(i), vuzp1q_u8(result[0], result[1]));
            vst1q_u8(out[1].as_mut_ptr().add(i), vuzp2q_u8(result[0], result[1]));
        }
        core::arch::asm!("msr fpsr, {flags}", flags = in(reg) flags, options(nostack, nomem));
    }

    /// Sixteen lanes of four planes as four vectors of four words.
    #[inline(always)]
    unsafe fn zip4(p0: uint8x16_t, p1: uint8x16_t, p2: uint8x16_t, p3: uint8x16_t) -> [uint32x4_t; 4] {
        let z01 = vzipq_u8(p0, p1);
        let z23 = vzipq_u8(p2, p3);
        let w0 = vzipq_u16(vreinterpretq_u16_u8(z01.0), vreinterpretq_u16_u8(z23.0));
        let w1 = vzipq_u16(vreinterpretq_u16_u8(z01.1), vreinterpretq_u16_u8(z23.1));
        [
            vreinterpretq_u32_u16(w0.0),
            vreinterpretq_u32_u16(w0.1),
            vreinterpretq_u32_u16(w1.0),
            vreinterpretq_u32_u16(w1.1),
        ]
    }

    /// The four planes of sixteen words.
    #[inline(always)]
    unsafe fn unzip4(w: [uint32x4_t; 4]) -> [uint8x16_t; 4] {
        let e01 = vuzpq_u8(vreinterpretq_u8_u32(w[0]), vreinterpretq_u8_u32(w[1]));
        let e23 = vuzpq_u8(vreinterpretq_u8_u32(w[2]), vreinterpretq_u8_u32(w[3]));
        let even = vuzpq_u8(e01.0, e23.0); // planes 0 and 2
        let odd = vuzpq_u8(e01.1, e23.1); // planes 1 and 3
        [even.0, odd.0, even.1, odd.1]
    }

    /// Load lanes i..i+16 of planes `p` (the first `ws` are read) as words.
    #[inline(always)]
    unsafe fn load(p: &[&[u8]; 4], ws: usize, i: usize) -> [uint32x4_t; 4] {
        let z = vdupq_n_u8(0);
        let ld = |k: usize| if k < ws { vld1q_u8(p[k].as_ptr().add(i)) } else { z };
        zip4(ld(0), ld(1), ld(2), ld(3))
    }

    /// Store words as lanes i..i+16 of the first `wd` planes.
    #[inline(always)]
    unsafe fn store(out: &mut [&mut [u8]; 4], wd: usize, i: usize, w: [uint32x4_t; 4]) {
        let pl = unzip4(w);
        for (k, v) in pl.iter().enumerate().take(wd) {
            vst1q_u8(out[k].as_mut_ptr().add(i), *v);
        }
    }

    pub unsafe fn to_words(p: &[&[u8]; 4], ws: usize, i: usize, out: &mut [u32]) {
        let w = load(p, ws, i);
        for (k, v) in w.iter().enumerate() {
            vst1q_u32(out.as_mut_ptr().add(4 * k), *v);
        }
    }

    pub unsafe fn from_words(xs: &[u32], wd: usize, i: usize, out: &mut [&mut [u8]; 4]) {
        let w = [
            vld1q_u32(xs.as_ptr()),
            vld1q_u32(xs.as_ptr().add(4)),
            vld1q_u32(xs.as_ptr().add(8)),
            vld1q_u32(xs.as_ptr().add(12)),
        ];
        store(out, wd, i, w);
    }

    const CANON32: u32 = 0x7FC0_0000;

    /// The result of an f32 add, sub or mul with every NaN replaced by the canonical one.
    #[inline(always)]
    unsafe fn canon(r: float32x4_t) -> uint32x4_t {
        let nan = vmvnq_u32(vceqq_f32(r, r));
        vbslq_u32(nan, vdupq_n_u32(CANON32), vreinterpretq_u32_f32(r))
    }

    /// `lanes::f32_max_min` on four lanes (maxNum / minNum, -0 < +0, a NaN operand yields the other).
    #[inline(always)]
    unsafe fn max_min(x: uint32x4_t, y: uint32x4_t, max: bool) -> uint32x4_t {
        let (fx, fy) = (vreinterpretq_f32_u32(x), vreinterpretq_f32_u32(y));
        let (gt, lt) = (vcgtq_f32(fx, fy), vcltq_f32(fx, fy));
        let neg_x = vtstq_u32(x, vdupq_n_u32(0x8000_0000));
        // max: x when fx > fy, else y when fx < fy or x is negative (the tie of -0 and +0), else x
        // min: x when fx < fy, else y when fx > fy or x is not negative, else x
        let take_y =
            if max { vbicq_u32(vorrq_u32(lt, neg_x), gt) } else { vbicq_u32(vorrq_u32(gt, vmvnq_u32(neg_x)), lt) };
        let r = vbslq_u32(take_y, y, x);
        let xnan = vmvnq_u32(vceqq_f32(fx, fx));
        let ynan = vmvnq_u32(vceqq_f32(fy, fy));
        let r = vbslq_u32(xnan, y, r);
        let r = vbslq_u32(ynan, x, r);
        vbslq_u32(vandq_u32(xnan, ynan), vdupq_n_u32(CANON32), r)
    }

    /// binary16 (in the low 16 bits of each word) to binary32, as `lanes::f16_to_f32_bits`.
    #[inline(always)]
    unsafe fn f16_to_f32(x: uint32x4_t) -> uint32x4_t {
        let sign = vshlq_n_u32::<16>(vandq_u32(x, vdupq_n_u32(0x8000)));
        let e = vandq_u32(vshrq_n_u32::<10>(x), vdupq_n_u32(0x1f));
        let m = vandq_u32(x, vdupq_n_u32(0x3ff));
        let normal = vorrq_u32(vshlq_n_u32::<23>(vaddq_u32(e, vdupq_n_u32(112))), vshlq_n_u32::<13>(m));
        // a subnormal is m * 2^-24, exact in binary32 (m has at most 10 bits)
        let sub = vreinterpretq_u32_f32(vmulq_f32(vcvtq_f32_u32(m), vdupq_n_f32(1.0 / 16_777_216.0)));
        let m_zero = vceqq_u32(m, vdupq_n_u32(0));
        let special = vbslq_u32(m_zero, vorrq_u32(sign, vdupq_n_u32(0x7f80_0000)), vdupq_n_u32(CANON32));
        let r = vbslq_u32(vceqq_u32(e, vdupq_n_u32(0)), sub, normal);
        let r = vorrq_u32(r, sign);
        vbslq_u32(vceqq_u32(e, vdupq_n_u32(0x1f)), special, r)
    }

    /// binary32 to binary16 (round to nearest even, NaN canonical, in the low 16 bits of each word), as
    /// `lanes::f32_to_f16_bits`: the rounding by one integer add in the normal range and by one float add in the
    /// subnormal range.
    #[inline(always)]
    unsafe fn f32_to_f16(x: uint32x4_t) -> uint32x4_t {
        let sign = vandq_u32(x, vdupq_n_u32(0x8000_0000));
        let f = veorq_u32(x, sign); // |x|
                                    // |x| >= 2^16 (and every NaN, Inf): infinity, or the canonical NaN above 0x7f800000
        let big = vcgeq_u32(f, vdupq_n_u32(143 << 23));
        let nan = vcgtq_u32(f, vdupq_n_u32(255 << 23));
        // subnormal result (|x| < 2^-14): add 0.5 so the float adder rounds at the f16 subnormal grid
        let sub_bits = vsubq_u32(
            vreinterpretq_u32_f32(vaddq_f32(vreinterpretq_f32_u32(f), vdupq_n_f32(0.5))),
            vdupq_n_u32(126 << 23),
        );
        let is_sub = vcltq_u32(f, vdupq_n_u32(113 << 23));
        // normal result: rebias and add the rounding bias 0xfff plus the lowest kept bit
        let odd = vandq_u32(vshrq_n_u32::<13>(f), vdupq_n_u32(1));
        let nb = vaddq_u32(vaddq_u32(f, vdupq_n_u32(((15u32.wrapping_sub(127)) << 23).wrapping_add(0xfff))), odd);
        let normal = vshrq_n_u32::<13>(nb);
        let r = vbslq_u32(is_sub, sub_bits, normal);
        let inf_or_nan = vbslq_u32(nan, vdupq_n_u32(0x7e00), vdupq_n_u32(0x7c00));
        let r = vbslq_u32(big, inf_or_nan, r);
        let r = vorrq_u32(r, vshrq_n_u32::<16>(sign));
        // a NaN has no sign in the model
        vbslq_u32(nan, vdupq_n_u32(0x7e00), r)
    }

    /// Fused kernels over lanes 0..n (n a multiple of 16): `op` is the binary opcode 0, 3, 6, 9, 10 on f32, or the
    /// conversion 31 or 32.
    pub unsafe fn fused(op: u32, a: &[&[u8]; 4], b: &[&[u8]; 4], out: &mut [&mut [u8]; 4], n: usize) {
        let mut i = 0;
        match op {
            31 => {
                while i < n {
                    let x = load(a, 2, i);
                    store(out, 4, i, [f16_to_f32(x[0]), f16_to_f32(x[1]), f16_to_f32(x[2]), f16_to_f32(x[3])]);
                    i += 16;
                }
            }
            32 => {
                while i < n {
                    let x = load(a, 4, i);
                    let h = [f32_to_f16(x[0]), f32_to_f16(x[1]), f32_to_f16(x[2]), f32_to_f16(x[3])];
                    // two bytes per lane: the low byte of each word is plane 0, the next plane 1
                    store(out, 2, i, h);
                    i += 16;
                }
            }
            _ => {
                macro_rules! lanes {
                    (|$x:ident, $y:ident| $e:expr) => {
                        while i < n {
                            let xs = load(a, 4, i);
                            let ys = load(b, 4, i);
                            let mut r = [vdupq_n_u32(0); 4];
                            for k in 0..4 {
                                let ($x, $y) = (xs[k], ys[k]);
                                r[k] = $e;
                            }
                            store(out, 4, i, r);
                            i += 16;
                        }
                    };
                }
                let f = vreinterpretq_f32_u32;
                match op {
                    0 => lanes!(|x, y| canon(vaddq_f32(f(x), f(y)))),
                    3 => lanes!(|x, y| canon(vsubq_f32(f(x), f(y)))),
                    6 => lanes!(|x, y| canon(vmulq_f32(f(x), f(y)))),
                    9 => lanes!(|x, y| max_min(x, y, true)),
                    _ => lanes!(|x, y| max_min(x, y, false)),
                }
            }
        }
    }
}
