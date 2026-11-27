//! MXM dot products (`slices/mxm/mxm_insts.sail`).

use crate::fp::{self, FClass, FMT16, FMT32, FMT_BF16};
use std::sync::OnceLock;

/// The exact f16 dot product of weight row `w` (low and high byte planes) and activation `x`,
/// rounded once to f32. Lane k holds the f16 `whi[k] << 8 | wlo[k]`. All slices must have the
/// same length (the lane count, `vlanes`); lanes the machine does not have are simply absent.
///
/// Specials first: any NaN operand, Inf x 0, or +Inf and -Inf products give the canonical NaN;
/// otherwise an infinite product gives that infinity. An exact zero sum is -0 only when every
/// product is -0, else +0. The sum is exact in `i128` fixed point at 2^-48: a product is
/// `mag(<2^11) * mag(<2^11) * 2^(e_a + e_b)` with `e_a + e_b + 48` in 0..=58, so a term is below
/// 2^80 and 512 terms (32 superlanes) below 2^89. // model: slices/mxm/mxm_insts.sail f16_dot
pub fn f16_dot(wlo: &[u8], whi: &[u8], xlo: &[u8], xhi: &[u8]) -> u32 {
    f16_dot_prepared(wlo, whi, &F16Act::new(xlo, xhi))
}

/// One activation lane that is not zero: its index, significand, exponent + 24 (0..=29) and sign.
#[derive(Clone, Copy, Debug)]
struct ActLane {
    k: u32,
    mag: u32,
    e: u32,
    neg: u32,
}

/// Eight activation lanes from `at`, at least one not zero: significands (0 for a zero lane), exponents + 24 and
/// sign masks (0xffff when negative).
#[derive(Clone, Copy, Debug)]
struct ActChunk {
    at: u32,
    mag: [u16; 8],
    e: [u16; 8],
    neg: [u16; 8],
}

/// An f16 activation vector (low and high byte planes) decoded once for all the weight rows of an ABC.
#[derive(Debug)]
pub struct F16Act<'a> {
    xlo: &'a [u8],
    xhi: &'a [u8],
    special: bool,
    max_shift: u16,
    /// Every lane that is not zero (the scalar kernel), decoded on first use.
    lanes: OnceLock<Vec<ActLane>>,
    /// The whole chunks of eight lanes with a lane that is not zero, and the lanes past the last whole chunk that are
    /// not zero (the vector kernel).
    chunks: Vec<ActChunk>,
    tail: Vec<ActLane>,
}

/// The decoded lane `k` of an activation if it is not zero.
#[inline]
fn act_lane(xlo: &[u8], xhi: &[u8], k: usize) -> Option<ActLane> {
    let x = (xhi[k] as u32) << 8 | xlo[k] as u32;
    (x & 0x7fff != 0).then(|| {
        let ex = (x >> 10) & 31;
        let mag = (x & 0x3ff) | ((ex != 0) as u32) << 10;
        ActLane { k: k as u32, mag, e: ex.max(1) - 1, neg: x >> 15 }
    })
}

impl<'a> F16Act<'a> {
    /// The chunks of eight lanes that hold a lane that is not zero (the work of the vector kernel for one row).
    pub fn chunk_count(&self) -> usize {
        self.chunks.len() + self.tail.len()
    }

    pub fn new(xlo: &'a [u8], xhi: &'a [u8]) -> F16Act<'a> {
        let n = xlo.len();
        let xhi = &xhi[..n];
        let special = has_inf_or_nan(xhi);
        let mut max_shift = 0u16;
        let (mut chunks, mut tail) = (Vec::new(), Vec::new());
        if !special {
            let whole = n / 8 * 8;
            for at in (0..whole).step_by(8) {
                // eight lanes at once: a lane is not zero when its low byte or the low seven bits of its high byte are not
                let lo = u64::from_le_bytes(xlo[at..at + 8].try_into().unwrap());
                let hi = u64::from_le_bytes(xhi[at..at + 8].try_into().unwrap());
                if lo | (hi & 0x7f7f_7f7f_7f7f_7f7f) == 0 {
                    continue;
                }
                let mut c = ActChunk { at: at as u32, mag: [0; 8], e: [0; 8], neg: [0; 8] };
                for j in 0..8 {
                    if let Some(l) = act_lane(xlo, xhi, at + j) {
                        c.mag[j] = l.mag as u16;
                        c.e[j] = l.e as u16;
                        max_shift = max_shift.max(c.e[j]);
                        c.neg[j] = if l.neg == 1 { 0xffff } else { 0 };
                    }
                }
                chunks.push(c);
            }
            for lane in (whole..n).filter_map(|k| act_lane(xlo, xhi, k)) {
                max_shift = max_shift.max(lane.e as u16);
                tail.push(lane);
            }
        }
        F16Act { xlo, xhi, special, max_shift, lanes: OnceLock::new(), chunks, tail }
    }

    /// Every lane that is not zero, in lane order (empty when `special`).
    fn lanes(&self) -> &[ActLane] {
        self.lanes.get_or_init(|| {
            if self.special {
                Vec::new()
            } else {
                (0..self.xlo.len()).filter_map(|k| act_lane(self.xlo, self.xhi, k)).collect()
            }
        })
    }
}

/// Is a lane of an f16 plane pair Inf or NaN (the exponent bits of the high byte all ones)?
pub fn has_inf_or_nan(hi: &[u8]) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        neon::has_inf_or_nan(hi)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        hi.iter().fold(false, |s, &h| s | (h & 0x7c == 0x7c))
    }
}

/// Maximum raw f16 exponent in the high byte plane. Exponent 31 also
/// records that the ordinary finite kernel needs the reference path.
pub fn f16_row_max_exponent(high: &[u8]) -> u8 {
    #[cfg(target_arch = "aarch64")]
    {
        neon::row_max_exponent(high)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        high.iter().fold(0u8, |max, &byte| max.max(byte & 0x7c)) >> 2
    }
}

/// [`f16_dot`] of weight row `w` and a prepared activation, which fixes the lane count: the vector kernel on AArch64 or AVX-512DQ,
/// otherwise [`f16_dot_prepared_scalar`].
pub fn f16_dot_prepared(wlo: &[u8], whi: &[u8], act: &F16Act) -> u32 {
    f16_dot_prepared_with(wlo, whi, act, has_inf_or_nan(&whi[..act.xlo.len()]))
}

/// [`f16_dot_prepared`] for a row whose [`has_inf_or_nan`] (of `whi` over the activation's lanes) the caller already
/// knows as `row_special`: a matrix is scanned once, not once per ABC.
pub fn f16_dot_prepared_with(wlo: &[u8], whi: &[u8], act: &F16Act, row_special: bool) -> u32 {
    if row_special {
        let n = act.xlo.len();
        return f16_dot_reference(&wlo[..n], &whi[..n], act.xlo, act.xhi);
    }
    f16_dot_mode::<false>(wlo, whi, act)
}

/// Like [`f16_dot_prepared_with`], with the exact highest exponent of the
/// active weight row from [`f16_row_max_exponent`].
pub fn f16_dot_prepared_bounded(wlo: &[u8], whi: &[u8], act: &F16Act, row_exponent: u8) -> u32 {
    if row_exponent == 31 {
        let n = act.xlo.len();
        return f16_dot_reference(&wlo[..n], &whi[..n], act.xlo, act.xhi);
    }
    // A finite significand product is strictly smaller than 2^22.
    // For at most 512 terms and a shift of at most 32, the sum of all
    // absolute terms is strictly below 2^(22 + 32 + 9) = 2^63.
    if act.xlo.len() <= 512 && u16::from(row_exponent.saturating_sub(1)) + act.max_shift <= 32 {
        f16_dot_mode::<true>(wlo, whi, act)
    } else {
        f16_dot_mode::<false>(wlo, whi, act)
    }
}

fn f16_dot_mode<const SINGLE: bool>(wlo: &[u8], whi: &[u8], act: &F16Act) -> u32 {
    let n = act.xlo.len();
    let (wlo, whi) = (&wlo[..n], &whi[..n]);
    if act.special {
        return f16_dot_reference(wlo, whi, act.xlo, act.xhi);
    }
    #[cfg(target_arch = "aarch64")]
    let parts = neon::chunks::<SINGLE>(wlo, whi, &act.chunks);
    #[cfg(target_arch = "x86_64")]
    let parts = if std::is_x86_feature_detected!("avx512dq") {
        // SAFETY: AVX-512DQ includes AVX-512F. Every memory load is bounds checked in chunks.
        unsafe { x86::chunks::<SINGLE>(wlo, whi, &act.chunks) }
    } else {
        return scalar_finite(wlo, whi, act);
    };
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
        let (mut lo, mut hi) = parts;
        for l in &act.tail {
            let (a, b) = term(wlo, whi, l);
            lo = lo.wrapping_add(a);
            hi = hi.wrapping_add(b);
        }
        finish(lo, hi, wlo, whi, act)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    scalar_finite(wlo, whi, act)
}

/// One lane's term as its parts of the two accumulators (see [`f16_dot_prepared_scalar`]).
#[inline(always)]
fn term(wlo: &[u8], whi: &[u8], l: &ActLane) -> (i64, i64) {
    let k = l.k as usize;
    let w = (whi[k] as u32) << 8 | wlo[k] as u32;
    let ex = (w >> 10) & 31;
    let mag = (w & 0x3ff) | ((ex != 0) as u32) << 10;
    let s = ex.max(1) - 1 + l.e;
    let neg = ((w >> 15) ^ l.neg) as i64;
    let t = ((mag * l.mag) as i64) << (s & 31);
    let t = (t ^ -neg) + neg;
    let high = (s >> 5) as i64;
    (t & (high - 1), t & -high)
}

/// The scalar form of [`f16_dot_prepared`].
///
/// Without an Inf or NaN operand a term is `mag_w * mag_x < 2^22` shifted by `s = e_w + e_x + 48` in 0..=58. A term
/// with `s < 32` is summed shifted by `s` (below 2^53), one with `s >= 32` by `s - 32` (below 2^48) into a second
/// accumulator worth 2^32 of the first: neither of the two `i64` sums can overflow below 2^10 lanes. A zero activation
/// lane adds no term; it only matters when the sum is exactly zero, and then the sign rule is taken over all lanes.
pub fn f16_dot_prepared_scalar(wlo: &[u8], whi: &[u8], act: &F16Act) -> u32 {
    let n = act.xlo.len();
    debug_assert!(n < 1 << 10 && wlo.len() == n && whi.len() == n);
    let (wlo, whi) = (&wlo[..n], &whi[..n]);
    if act.special || has_inf_or_nan(whi) {
        return f16_dot_reference(wlo, whi, act.xlo, act.xhi);
    }
    scalar_finite(wlo, whi, act)
}

fn scalar_finite(wlo: &[u8], whi: &[u8], act: &F16Act) -> u32 {
    let (mut lo, mut hi) = (0i64, 0i64);
    for l in act.lanes() {
        let (a, b) = term(wlo, whi, l);
        lo += a;
        hi += b;
    }
    finish(lo, hi, wlo, whi, act)
}

/// The f32 of the sum `lo + hi * 2^32` (units of 2^-48) of a row without Inf or NaN.
fn finish(lo: i64, hi: i64, wlo: &[u8], whi: &[u8], act: &F16Act) -> u32 {
    let sum = lo as i128 + ((hi as i128) << 32);
    if sum == 0 {
        fp::zero_bits(FMT32, all_products_negative_zero(wlo, whi, act.xlo, act.xhi))
    } else {
        f32_of_sum(sum)
    }
}

/// Is every product of the lanes a zero of the negative sign? For each lane a magnitude is zero and the signs differ.
/// Written on bytes without a branch so that a block of 16 lanes is vectorised; a block that settles the answer ends
/// the scan (a padded row, the usual caller, ends in the first block).
fn all_products_negative_zero(wlo: &[u8], whi: &[u8], xlo: &[u8], xhi: &[u8]) -> bool {
    const B: usize = 16;
    let n = wlo.len();
    let (whi, xlo, xhi) = (&whi[..n], &xlo[..n], &xhi[..n]);
    let lane = |wl: u8, wh: u8, xl: u8, xh: u8| {
        let wz = (((wh & 0x7f) | wl) == 0) as u8;
        let xz = (((xh & 0x7f) | xl) == 0) as u8;
        (wz | xz) & ((wh ^ xh) >> 7)
    };
    let mut at = 0;
    while at + B <= n {
        let mut all = 1u8;
        for i in at..at + B {
            all &= lane(wlo[i], whi[i], xlo[i], xhi[i]);
        }
        if all == 0 {
            return false;
        }
        at += B;
    }
    (at..n).all(|i| lane(wlo[i], whi[i], xlo[i], xhi[i]) != 0)
}

/// The f32 nearest (ties to even) to `sum * 2^-48` for a non-zero `sum` below 2^127 in magnitude: [`fp::fround`] for the
/// range of a dot product, whose results are always normal (the value is at least 2^-48 and below 2^79); anything else
/// goes to `fround`. // model: core/numerics/fp.sail fround
pub fn f32_of_sum(sum: i128) -> u32 {
    let neg = sum < 0;
    let mag = sum.unsigned_abs();
    debug_assert!(mag != 0);
    let p = 127 - mag.leading_zeros(); // the leading bit of mag
    let mut e = p as i32 - 48;
    if !(-126..=127).contains(&e) {
        return fp::fround(FMT32, neg, mag, -48);
    }
    let mut m = if p <= 23 {
        (mag << (23 - p)) as u64
    } else {
        let d = p - 23;
        let lo = (mag >> d) as u64;
        let rem = mag & ((1u128 << d) - 1);
        let half = 1u128 << (d - 1);
        lo + (rem > half || (rem == half && lo & 1 == 1)) as u64
    };
    if m == 1 << 24 {
        m >>= 1;
        e += 1;
    }
    if e > 127 {
        return fp::inf_bits(FMT32, neg);
    }
    (neg as u32) << 31 | ((e + 127) as u32) << 23 | (m as u32 & 0x7f_ffff)
}

/// Eight exact integer products in 64-bit lanes. The two sums have the same scale and
/// bounds as the scalar kernel. The tail and final rounding are shared with NEON.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::ActChunk;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512dq")]
    pub unsafe fn chunks<const SINGLE: bool>(wlo: &[u8], whi: &[u8], chunks: &[ActChunk]) -> (i64, i64) {
        let (mut lo, mut hi) = (_mm512_setzero_si512(), _mm512_setzero_si512());
        let one = _mm512_set1_epi64(1);
        let mask31 = _mm512_set1_epi64(31);
        for chunk in chunks {
            let at = chunk.at as usize;
            let low = _mm_loadl_epi64(wlo[at..at + 8].as_ptr().cast());
            let high = _mm_loadl_epi64(whi[at..at + 8].as_ptr().cast());
            let weights = _mm_unpacklo_epi8(low, high);
            let w = _mm512_cvtepu16_epi64(weights);
            let ex = _mm512_and_si512(_mm512_srli_epi64::<10>(w), mask31);
            let mantissa = _mm512_and_si512(w, _mm512_set1_epi64(1023));
            let normal = _mm512_cmpneq_epi64_mask(ex, _mm512_setzero_si512());
            let magnitude = _mm512_mask_add_epi64(mantissa, normal, mantissa, _mm512_set1_epi64(1024));
            let load = |a: &[u16; 8]| _mm_loadu_si128(a.as_ptr().cast());
            let shift = _mm512_add_epi64(
                _mm512_sub_epi64(_mm512_max_epi64(ex, one), one),
                _mm512_cvtepu16_epi64(load(&chunk.e)),
            );
            let negative = _mm512_cmpgt_epi64_mask(
                _mm512_setzero_si512(),
                _mm512_cvtepi16_epi64(_mm_xor_si128(weights, load(&chunk.neg))),
            );
            let product = _mm512_mullo_epi64(magnitude, _mm512_cvtepu16_epi64(load(&chunk.mag)));
            let bits = if SINGLE { shift } else { _mm512_and_si512(shift, mask31) };
            let term = _mm512_sllv_epi64(product, bits);
            let signed = _mm512_mask_sub_epi64(term, negative, _mm512_setzero_si512(), term);
            if SINGLE {
                lo = _mm512_add_epi64(lo, signed);
            } else {
                let upper = _mm512_cmpgt_epi64_mask(shift, mask31);
                lo = _mm512_mask_add_epi64(lo, !upper, lo, signed);
                hi = _mm512_mask_add_epi64(hi, upper, hi, signed);
            }
        }
        (_mm512_reduce_add_epi64(lo), _mm512_reduce_add_epi64(hi))
    }
}

/// The NEON half of [`f16_dot_prepared`] (Advanced SIMD is part of every aarch64 target, so no run-time detection).
/// The terms are those of [`f16_dot_prepared_scalar`], eight lanes at a time, in 64-bit lanes.
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::ActChunk;
    use std::arch::aarch64::*;

    pub(super) struct E5WideChunk {
        mag: [i16; 8],
        shift: [u16; 8],
    }

    pub(super) fn e5_wide_chunks(lanes: &[super::Fp8Lane]) -> Vec<E5WideChunk> {
        lanes
            .chunks_exact(8)
            .map(|block| E5WideChunk {
                mag: std::array::from_fn(|i| i16::from(block[i].mag)),
                shift: std::array::from_fn(|i| u16::from(block[i].shift)),
            })
            .collect()
    }

    #[inline(never)]
    pub(super) fn e5_wide_dot(
        weights: &[u8],
        activation: &[super::Fp8Lane],
        chunks: &[E5WideChunk],
        table: &[super::Fp8Lane; 256],
    ) -> Option<i128> {
        debug_assert_eq!(weights.len(), activation.len());
        debug_assert_eq!(weights.len() / 8, chunks.len());
        // Each checked 8-byte weight block has a matching prepared 8-lane activation.
        let (mut low, mut high, tail) = unsafe {
            let mut low = vdupq_n_u64(0);
            let mut high = vdupq_n_u64(0);
            let mut special = vdup_n_u8(0);
            let mut blocks = weights.chunks_exact(8);
            for (weight, active) in blocks.by_ref().zip(chunks) {
                let wb = vld1_u8(weight.as_ptr());
                let ex = vand_u8(vshr_n_u8::<2>(wb), vdup_n_u8(31));
                special = vorr_u8(special, vceq_u8(ex, vdup_n_u8(31)));
                let normal = vcgt_u8(ex, vdup_n_u8(0));
                let mantissa = vorr_u8(vand_u8(wb, vdup_n_u8(3)), vand_u8(normal, vdup_n_u8(4)));
                let sign = vtst_u8(wb, vdup_n_u8(0x80));
                let coeff = vmovl_s8(vreinterpret_s8_u8(vsub_u8(veor_u8(mantissa, sign), sign)));
                let shift: uint8x8_t;
                std::arch::asm!(
                    "add {shift:v}.8b, {exponent:v}.8b, {normal:v}.8b",
                    shift = lateout(vreg) shift,
                    exponent = in(vreg) ex,
                    normal = in(vreg) normal,
                    options(pure, nomem, nostack, preserves_flags),
                );
                let shifts = vaddq_u16(vmovl_u8(shift), vld1q_u16(active.shift.as_ptr()));
                let mask = vreinterpretq_s16_u16(vcgeq_u16(shifts, vdupq_n_u16(32)));
                let x = vld1q_s16(active.mag.as_ptr());
                add4::<false>(
                    &mut low,
                    &mut high,
                    vmull_s16(vget_low_s16(coeff), vget_low_s16(x)),
                    vmovl_u16(vget_low_u16(shifts)),
                    vmovl_s16(vget_low_s16(mask)),
                );
                add4::<false>(
                    &mut low,
                    &mut high,
                    vmull_high_s16(coeff, x),
                    vmovl_high_u16(shifts),
                    vmovl_high_s16(mask),
                );
            }
            if vmaxv_u8(special) != 0 {
                return None;
            }
            (vaddvq_u64(low) as i64, vaddvq_u64(high) as i64, blocks.remainder())
        };
        for (&bits, x) in tail.iter().zip(&activation[weights.len() - tail.len()..]) {
            let w = table[bits as usize];
            if w.special {
                return None;
            }
            let shift = u32::from(w.shift) + u32::from(x.shift);
            let term = i64::from(w.mag) * i64::from(x.mag);
            if shift < 32 {
                low += term << shift;
            } else {
                high += term << (shift - 32);
            }
        }
        Some((i128::from(high) << 32) + i128::from(low))
    }

    // Classify sign, exponent and mantissa while all eight weights fit in one register.
    #[inline(always)]
    unsafe fn fp8_coefficients<const E4: bool>(bytes: uint8x8_t) -> (int32x4_t, int32x4_t) {
        let exponent = if E4 { vshr_n_u8::<3>(bytes) } else { vshr_n_u8::<2>(bytes) };
        let ex = vand_u8(exponent, vdup_n_u8(15));
        let normal = vcgt_u8(ex, vdup_n_u8(0));
        let frac = if E4 { 8 } else { 4 };
        let implicit = vand_u8(normal, vdup_n_u8(frac));
        let magnitude = vorr_u8(vand_u8(bytes, vdup_n_u8(frac - 1)), implicit);
        let negative = vtst_u8(bytes, vdup_n_u8(0x80));
        let signed = vreinterpret_s8_u8(vsub_u8(veor_u8(magnitude, negative), negative));
        // A saturating subtract changes the caller's FPSR.QC on subnormals.
        let shift: uint8x8_t;
        std::arch::asm!(
            "add {shift:v}.8b, {exponent:v}.8b, {normal:v}.8b",
            shift = lateout(vreg) shift,
            exponent = in(vreg) ex,
            normal = in(vreg) normal,
            options(pure, nomem, nostack, preserves_flags),
        );
        let values = vmovl_s8(signed);
        let shifts = vmovl_u8(shift);
        (
            vshlq_s32(vmovl_s16(vget_low_s16(values)), vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(shifts)))),
            vshlq_s32(vmovl_high_s16(values), vreinterpretq_s32_u32(vmovl_high_u16(shifts))),
        )
    }

    // Each E4M3 coefficient has magnitude less than 2^18. At most 512 terms
    // therefore fit in 45 magnitude bits. Narrow E5M2 coefficients are smaller.
    #[inline(never)]
    pub fn fp8_grid_dot<const E4: bool>(
        weights: &[u8],
        activation: &[i32],
        table: &[super::Fp8Lane; 256],
    ) -> Option<i64> {
        debug_assert_eq!(weights.len(), activation.len());
        debug_assert!(weights.len() <= 512);
        // Each vector load reads only a full checked block.
        let (mut sum, tail) = unsafe {
            let mut lo = vdupq_n_s64(0);
            let mut hi = vdupq_n_s64(0);
            let mut special = vdup_n_u8(0);
            let mut blocks = weights.chunks_exact(8);
            for (weight, active) in blocks.by_ref().zip(activation.chunks_exact(8)) {
                let wb = vld1_u8(weight.as_ptr());
                let xb = active.as_ptr();
                let unsupported = if E4 {
                    vceq_u8(vand_u8(wb, vdup_n_u8(0x7f)), vdup_n_u8(0x7f))
                } else {
                    vand_u8(wb, vdup_n_u8(0x40))
                };
                special = vorr_u8(special, unsupported);
                let (w0, w1) = fp8_coefficients::<E4>(wb);
                let x0 = vld1q_s32(xb);
                let x1 = vld1q_s32(xb.add(4));
                lo = vmlal_s32(lo, vget_low_s32(w0), vget_low_s32(x0));
                hi = vmlal_high_s32(hi, w0, x0);
                lo = vmlal_s32(lo, vget_low_s32(w1), vget_low_s32(x1));
                hi = vmlal_high_s32(hi, w1, x1);
            }
            if vmaxv_u8(special) != 0 {
                return None;
            }
            (vaddvq_s64(vaddq_s64(lo, hi)), blocks.remainder())
        };
        for (&bits, &x) in tail.iter().zip(&activation[weights.len() - tail.len()..]) {
            let w = table[bits as usize];
            if w.special || (!E4 && bits & 0x40 != 0) {
                return None;
            }
            let shift = if E4 { w.shift.saturating_sub(7) } else { w.shift };
            sum += i64::from(x) * (i64::from(w.mag) << shift);
        }
        Some(sum)
    }

    pub fn has_inf_or_nan(hi: &[u8]) -> bool {
        let mut blocks = hi.chunks_exact(64);
        // SAFETY: each load reads 16 bytes inside a `chunks_exact` chunk of 64 or 16 bytes.
        let any = unsafe {
            let m = vdupq_n_u8(0x7c);
            let hit = |p: *const u8| vceqq_u8(vandq_u8(vld1q_u8(p), m), m);
            // four independent accumulators: one chain of ORs would be latency bound
            let mut acc = [vdupq_n_u8(0); 4];
            for c in &mut blocks {
                for (i, a) in acc.iter_mut().enumerate() {
                    *a = vorrq_u8(*a, hit(c.as_ptr().add(16 * i)));
                }
            }
            let mut rest = blocks.remainder().chunks_exact(16);
            for c in &mut rest {
                acc[0] = vorrq_u8(acc[0], hit(c.as_ptr()));
            }
            let all = vorrq_u8(vorrq_u8(acc[0], acc[1]), vorrq_u8(acc[2], acc[3]));
            if vmaxvq_u8(all) != 0 {
                return true;
            }
            rest.remainder()
        };
        any.iter().any(|&h| h & 0x7c == 0x7c)
    }

    pub fn row_max_exponent(hi: &[u8]) -> u8 {
        let mut blocks = hi.chunks_exact(16);
        // SAFETY: every load reads one complete chunk, with unaligned loads.
        let widest = unsafe {
            let mask = vdupq_n_u8(0x7c);
            let mut max = vdupq_n_u8(0);
            for block in &mut blocks {
                max = vmaxq_u8(max, vandq_u8(vld1q_u8(block.as_ptr()), mask));
            }
            vmaxvq_u8(max)
        };
        blocks.remainder().iter().fold(widest, |max, value| max.max(value & 0x7c)) >> 2
    }

    /// Two lanes of terms (`m`: signed products, `s`: shifts 0..=58, `h`: all-ones when `s >= 32`).
    #[inline(always)]
    unsafe fn add2<const SINGLE: bool>(
        lo: &mut uint64x2_t,
        hi: &mut uint64x2_t,
        m: int64x2_t,
        s: uint64x2_t,
        h: uint64x2_t,
    ) {
        let shift = if SINGLE { s } else { vandq_u64(s, vdupq_n_u64(31)) };
        let signed = vreinterpretq_u64_s64(vshlq_s64(m, vreinterpretq_s64_u64(shift)));
        if SINGLE {
            *lo = vaddq_u64(*lo, signed);
        } else {
            *hi = vaddq_u64(*hi, vandq_u64(signed, h));
            *lo = vaddq_u64(*lo, vbicq_u64(signed, h));
        }
    }

    /// Four lanes: 32-bit products, shifts and masks widened to two pairs of 64-bit lanes.
    #[inline(always)]
    unsafe fn add4<const SINGLE: bool>(
        lo: &mut uint64x2_t,
        hi: &mut uint64x2_t,
        m: int32x4_t,
        s: uint32x4_t,
        h: int32x4_t,
    ) {
        let u = vreinterpretq_u64_s64;
        add2::<SINGLE>(lo, hi, vmovl_s32(vget_low_s32(m)), vmovl_u32(vget_low_u32(s)), u(vmovl_s32(vget_low_s32(h))));
        add2::<SINGLE>(lo, hi, vmovl_high_s32(m), vmovl_high_u32(s), u(vmovl_high_s32(h)));
    }

    /// The sums of the terms of `chunks` against the weight row, as (low, high) accumulators.
    pub fn chunks<const SINGLE: bool>(wlo: &[u8], whi: &[u8], chunks: &[ActChunk]) -> (i64, i64) {
        // SAFETY: the weight loads read the 8 bytes of a bounds-checked 8-byte subslice, the activation loads the
        // 8-element arrays of a chunk.
        unsafe {
            let (mut lo, mut hi) = (vdupq_n_u64(0), vdupq_n_u64(0));
            for c in chunks {
                let at = c.at as usize;
                let (l8, h8) = (&wlo[at..at + 8], &whi[at..at + 8]);
                let w = vorrq_u16(vmovl_u8(vld1_u8(l8.as_ptr())), vshll_n_u8::<8>(vld1_u8(h8.as_ptr())));
                let ex = vandq_u16(vshrq_n_u16::<10>(w), vdupq_n_u16(31));
                let mag = vorrq_u16(vandq_u16(w, vdupq_n_u16(0x3ff)), vandq_u16(vtstq_u16(ex, ex), vdupq_n_u16(0x400)));
                let e = vsubq_u16(vmaxq_u16(ex, vdupq_n_u16(1)), vdupq_n_u16(1));
                let xm = vld1q_u16(c.mag.as_ptr());
                let s = vaddq_u16(e, vld1q_u16(c.e.as_ptr()));
                let neg = veorq_u16(
                    vreinterpretq_u16_s16(vshrq_n_s16::<15>(vreinterpretq_s16_u16(w))),
                    vld1q_u16(c.neg.as_ptr()),
                );
                let high = vreinterpretq_s16_u16(vcgeq_u16(s, vdupq_n_u16(32)));
                // The magnitude fits in 11 bits. Apply the sign to all eight
                // lanes before widening, so the exact signed product needs no
                // per-lane sign correction after its 64-bit shift.
                let signed_mag = vreinterpretq_s16_u16(vsubq_u16(veorq_u16(mag, neg), neg));
                let xm = vreinterpretq_s16_u16(xm);
                add4::<SINGLE>(
                    &mut lo,
                    &mut hi,
                    vmull_s16(vget_low_s16(signed_mag), vget_low_s16(xm)),
                    vmovl_u16(vget_low_u16(s)),
                    vmovl_s16(vget_low_s16(high)),
                );
                add4::<SINGLE>(
                    &mut lo,
                    &mut hi,
                    vmull_high_s16(signed_mag, xm),
                    vmovl_high_u16(s),
                    vmovl_high_s16(high),
                );
            }
            (vaddvq_u64(lo) as i64, vaddvq_u64(hi) as i64)
        }
    }
}

/// The lane-by-lane form of [`f16_dot`], kept as the scalar reference of the fast kernel.
pub fn f16_dot_reference(wlo: &[u8], whi: &[u8], xlo: &[u8], xhi: &[u8]) -> u32 {
    debug_assert!(wlo.len() == whi.len() && wlo.len() == xlo.len() && wlo.len() == xhi.len());
    let mut nan = false;
    let mut pinf = false;
    let mut ninf = false;
    let mut all_negzero = true;
    let mut sum: i128 = 0; // in units of 2^-48 (f16 min exponent -24, squared)
    let n = wlo.len();
    let (whi, xlo, xhi) = (&whi[..n], &xlo[..n], &xhi[..n]); // one bounds check for the row, not four a lane
    for (((&wl, &wh), &xl), &xh) in wlo.iter().zip(whi).zip(xlo).zip(xhi) {
        let aw = (wh as u32) << 8 | wl as u32;
        let ax = (xh as u32) << 8 | xl as u32;
        // A +0 against an operand that is neither Inf nor NaN: no term, and the sum is -0 only if the product is
        // (the sign bits agree). This is the lane of a padded activation or an absent weight.
        if (aw == 0 || ax == 0) && (aw & 0x7c00) != 0x7c00 && (ax & 0x7c00) != 0x7c00 {
            all_negzero &= (aw >> 15) != (ax >> 15);
            continue;
        }
        let a = fp::fdecode(FMT16, aw);
        let b = fp::fdecode(FMT16, ax);
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

/// One row's integer dot product: exact products of the 8-bit lanes with the given signs, 32-bit
/// sum with wrap (INT8, UINT8). `ws` / `xs`: the weight / activation lanes are signed.
/// // model: slices/mxm/mxm_insts.sail int8_dot
pub fn int8_dot(wr: &[u8], ws: bool, xv: &[u8], xs: bool) -> u32 {
    debug_assert!(wr.len() == xv.len());
    let mut sum: i64 = 0;
    for k in 0..wr.len() {
        let a: i64 = if ws { wr[k] as i8 as i64 } else { wr[k] as i64 };
        let b: i64 = if xs { xv[k] as i8 as i64 } else { xv[k] as i64 };
        sum += a * b; // at most 512 * 2^16, no overflow
    }
    sum as u32 // wrap to 32 bits, two's complement
}

#[derive(Clone, Copy)]
struct Fp8Lane {
    mag: i8,
    shift: u8,
    neg: bool,
    special: bool,
}

fn fp8_finite_table(f: fp::Fmt) -> Option<&'static [Fp8Lane; 256]> {
    static E4: OnceLock<[Fp8Lane; 256]> = OnceLock::new();
    static E5: OnceLock<[Fp8Lane; 256]> = OnceLock::new();
    let slot = if f == fp::FMT_E4M3 {
        &E4
    } else if f == fp::FMT_E5M2 {
        &E5
    } else {
        return None;
    };
    Some(slot.get_or_init(|| {
        std::array::from_fn(|bits| {
            let v = fp::fdecode(f, bits as u32);
            let mag = v.mag as i8;
            Fp8Lane {
                mag: if v.neg { -mag } else { mag },
                shift: (v.e + 16) as u8,
                neg: v.neg,
                special: matches!(v.cls, FClass::Inf | FClass::NaN),
            }
        })
    }))
}

// With the upper exponent bit clear, an E5M2 magnitude is < 2. The
// full E4M3 magnitude is < 2^9. At most 512 products on the 2^-32 grid
// therefore occupy at most 59 bits even before cancellation.
fn fp8_lower_exponents(words: &[u8]) -> bool {
    let mut chunks = words.chunks_exact(8);
    let reduced = chunks.by_ref().fold(0u64, |bits, data| bits | u64::from_le_bytes(data.try_into().unwrap()));
    reduced & 0x4040_4040_4040_4040 == 0 && chunks.remainder().iter().all(|byte| byte & 0x40 == 0)
}

fn fp8_finite_dot<const NARROW: bool>(lanes: &[Fp8Lane; 256], wr: &[u8], xv: &[u8]) -> Option<u32> {
    let mut sum = 0i128;
    let mut small_sum = 0i64;
    let mut all_negzero = true;
    for (&w, &x) in wr.iter().zip(xv) {
        let a = lanes[w as usize];
        let b = lanes[x as usize];
        if a.special | b.special {
            return None;
        }
        let product = i16::from(a.mag) * i16::from(b.mag);
        let shift = u32::from(a.shift) + u32::from(b.shift);
        if NARROW {
            small_sum += i64::from(product) << shift;
        } else {
            sum += i128::from(product) << shift;
        }
        all_negzero &= (product == 0) & (a.neg ^ b.neg);
    }
    let sum = if NARROW { i128::from(small_sum) } else { sum };
    Some(if sum == 0 { fp::zero_bits(FMT32, all_negzero) } else { fp::fround(FMT32, sum < 0, sum.unsigned_abs(), -32) })
}

/// Finite FP8 activation data shared by the rows of one ABC8.
pub struct Fp8Act<'a> {
    f: fp::Fmt,
    original: &'a [u8],
    table: Option<&'static [Fp8Lane; 256]>,
    decoded: Vec<Fp8Lane>,
    narrow: bool,
    #[cfg(target_arch = "aarch64")]
    fixed_grid: Vec<i32>,
    #[cfg(target_arch = "aarch64")]
    e5_wide: OnceLock<Vec<neon::E5WideChunk>>,
}

impl<'a> Fp8Act<'a> {
    pub fn new(f: fp::Fmt, original: &'a [u8]) -> Self {
        let mut act = Self {
            f,
            original,
            table: fp8_finite_table(f),
            decoded: Vec::new(),
            narrow: false,
            #[cfg(target_arch = "aarch64")]
            fixed_grid: Vec::new(),
            #[cfg(target_arch = "aarch64")]
            e5_wide: OnceLock::new(),
        };
        if let Some(table) = act.table {
            act.decoded = original.iter().map(|&bits| table[bits as usize]).collect();
            if act.decoded.iter().any(|lane| lane.special) {
                act.table = None;
            }
            act.narrow = original.len() <= 512 && (f == fp::FMT_E4M3 || fp8_lower_exponents(original));
            #[cfg(target_arch = "aarch64")]
            if act.narrow && act.table.is_some() && original.len() >= 16 {
                let scale = if f == fp::FMT_E4M3 { 7 } else { 0 };
                act.fixed_grid =
                    act.decoded.iter().map(|x| i32::from(x.mag) << x.shift.saturating_sub(scale)).collect();
            }
        }
        act
    }

    pub fn dot(&self, wr: &[u8]) -> u32 {
        if let Some(table) = self.table {
            #[cfg(target_arch = "aarch64")]
            if !self.fixed_grid.is_empty() && self.fixed_grid.len() == wr.len() {
                let e4 = self.f == fp::FMT_E4M3;
                let sum = if e4 {
                    neon::fp8_grid_dot::<true>(wr, &self.fixed_grid, table)
                } else {
                    neon::fp8_grid_dot::<false>(wr, &self.fixed_grid, table)
                };
                if let Some(sum) = sum.filter(|&sum| sum != 0) {
                    let scale = if e4 { -18 } else { -32 };
                    return fp::fround(FMT32, sum < 0, u128::from(sum.unsigned_abs()), scale);
                }
            }
            if wr.len() == self.decoded.len() {
                let narrow = self.narrow && (self.f == fp::FMT_E4M3 || fp8_lower_exponents(wr));
                #[cfg(target_arch = "aarch64")]
                if !narrow && self.f == fp::FMT_E5M2 && (16..=512).contains(&wr.len()) {
                    let chunks = self.e5_wide.get_or_init(|| neon::e5_wide_chunks(&self.decoded));
                    if let Some(sum) = neon::e5_wide_dot(wr, &self.decoded, chunks, table).filter(|&sum| sum != 0) {
                        return fp::fround(FMT32, sum < 0, sum.unsigned_abs(), -32);
                    }
                }
                let bits = if narrow {
                    self.dot_finite::<true, false>(table, wr)
                } else if wr.len() <= 512 {
                    self.dot_finite::<false, true>(table, wr)
                } else {
                    self.dot_finite::<false, false>(table, wr)
                };
                if let Some(bits) = bits {
                    return bits;
                }
            }
        }
        fp8_dot(self.f, wr, self.original)
    }

    fn dot_finite<const NARROW: bool, const BUCKETS: bool>(&self, table: &[Fp8Lane; 256], wr: &[u8]) -> Option<u32> {
        let (mut sum, mut small, mut high) = (0i128, 0i64, 0i64);
        for (&bits, x) in wr.iter().zip(&self.decoded) {
            let w = table[bits as usize];
            if w.special {
                return None;
            }
            let magnitude = i16::from(w.mag) * i16::from(x.mag);
            let shift = u32::from(w.shift) + u32::from(x.shift);
            if NARROW {
                small += i64::from(magnitude) << shift;
            } else if BUCKETS {
                // The magnitude is at most 49 for E5M2 and at most 225
                // for E4M3. With <=512 products each bin fits in i64.
                if shift < 32 {
                    small += i64::from(magnitude) << shift;
                } else {
                    high += i64::from(magnitude) << (shift - 32);
                }
            } else {
                sum += i128::from(magnitude) << shift;
            }
        }
        let sum = if NARROW || BUCKETS { (i128::from(high) << 32) + i128::from(small) } else { sum };
        if sum != 0 {
            return Some(fp::fround(FMT32, sum < 0, sum.unsigned_abs(), -32));
        }
        let all_negzero = wr.iter().zip(&self.decoded).all(|(&bits, x)| {
            let w = table[bits as usize];
            (w.mag == 0 || x.mag == 0) && (w.neg ^ x.neg)
        });
        Some(fp::zero_bits(FMT32, all_negzero))
    }
}

/// The exact dot product of an FP8 weight row and an FP8 activation (`f` is `FMT_E4M3` or `FMT_E5M2`), rounded once to f32.
/// Every product is a multiple of 2^-32 (the e5m2 subnormal squared), so the sum of 512 products is an integer in `i128`.
/// NaN rules as [`f16_dot`]: any NaN operand, Inf x 0, or +Inf and -Inf products give the canonical NaN; an e4m3 operand has no
/// infinity. An exact zero sum is -0 only when every product is -0. // model: slices/mxm/mxm_insts.sail fp8_dot
pub fn fp8_dot(f: fp::Fmt, wr: &[u8], xv: &[u8]) -> u32 {
    debug_assert!(wr.len() == xv.len());
    if let Some(lanes) = fp8_finite_table(f) {
        let narrow = wr.len() <= 512 && (f == fp::FMT_E4M3 || (fp8_lower_exponents(wr) && fp8_lower_exponents(xv)));
        let value = if narrow { fp8_finite_dot::<true>(lanes, wr, xv) } else { fp8_finite_dot::<false>(lanes, wr, xv) };
        if let Some(bits) = value {
            return bits;
        }
    }
    let mut nan = false;
    let mut pinf = false;
    let mut ninf = false;
    let mut all_negzero = true;
    let mut sum: i128 = 0; // in units of 2^-32
    for (&w, &x) in wr.iter().zip(xv) {
        let a = fp::fdecode(f, w as u32);
        let b = fp::fdecode(f, x as u32);
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
            let m = ((a.mag * b.mag) as i128) << (a.e + b.e + 32) as u32;
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
        fp::fround(FMT32, true, (-sum) as u128, -32)
    } else {
        fp::fround(FMT32, false, sum as u128, -32)
    }
}

/// The exact dot product of one row before rounding: the special-value flags and the sum of the finite products in units of
/// 2^-48 (the smallest unit of an f16 product; the fp8 and bfp8 products are multiples of it). `negzero`: every product is -0
/// (or there is none that is not -0), which makes a zero sum -0. // model: slices/mxm/mxm_insts.sail struct dsum
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DSum {
    pub nan: bool,
    pub pinf: bool,
    pub ninf: bool,
    pub negzero: bool,
    pub sum: i128,
}

impl DSum {
    fn new() -> DSum {
        DSum { nan: false, pinf: false, ninf: false, negzero: true, sum: 0 }
    }

    /// Fold in the product of two decoded elements. // model: slices/mxm/mxm_insts.sail fp8_exact, f16_exact
    fn term(&mut self, a: fp::FVal, b: fp::FVal, shift: i32) {
        let neg = a.neg != b.neg;
        let inf_times_zero =
            (a.cls == FClass::Inf && b.cls == FClass::Zero) || (a.cls == FClass::Zero && b.cls == FClass::Inf);
        if a.cls == FClass::NaN || b.cls == FClass::NaN || inf_times_zero {
            self.nan = true;
        } else if a.cls == FClass::Inf || b.cls == FClass::Inf {
            if neg {
                self.ninf = true;
            } else {
                self.pinf = true;
            }
            self.negzero = false;
        } else if a.cls == FClass::Zero || b.cls == FClass::Zero {
            if !neg {
                self.negzero = false;
            }
        } else {
            self.negzero = false;
            let m = ((a.mag * b.mag) as i128) << (a.e + b.e + shift) as u32;
            self.sum = if neg { self.sum - m } else { self.sum + m };
        }
    }
}

/// Round the exact dot product ONCE to f32 (round to nearest even). // model: slices/mxm/mxm_insts.sail dsum_round
pub fn dsum_round(d: &DSum) -> u32 {
    if d.nan || (d.pinf && d.ninf) {
        fp::canon_nan(FMT32)
    } else if d.pinf {
        fp::inf_bits(FMT32, false)
    } else if d.ninf {
        fp::inf_bits(FMT32, true)
    } else if d.sum == 0 {
        fp::zero_bits(FMT32, d.negzero)
    } else if d.sum < 0 {
        fp::fround(FMT32, true, d.sum.unsigned_abs(), -48)
    } else {
        fp::fround(FMT32, false, d.sum as u128, -48)
    }
}

/// The exact dot product of an FP8 row and activation in the format `f`. // model: slices/mxm/mxm_insts.sail fp8_exact
pub fn fp8_exact(f: fp::Fmt, wr: &[u8], xv: &[u8]) -> DSum {
    let mut d = DSum::new();
    for (&w, &x) in wr.iter().zip(xv) {
        d.term(fp::fdecode(f, w as u32), fp::fdecode(f, x as u32), 48);
    }
    d
}

/// The exact f16 dot product of row `w` (low and high byte planes) and activation `x`.
/// // model: slices/mxm/mxm_insts.sail f16_exact
pub fn f16_exact(wlo: &[u8], whi: &[u8], xlo: &[u8], xhi: &[u8]) -> DSum {
    let mut d = DSum::new();
    for k in 0..wlo.len() {
        let a = fp::fdecode(FMT16, (whi[k] as u32) << 8 | wlo[k] as u32);
        let b = fp::fdecode(FMT16, (xhi[k] as u32) << 8 | xlo[k] as u32);
        d.term(a, b, 48);
    }
    d
}

/// The exponent of the block that holds lane `k` of a bfp8 scale vector: the low four bits of the byte at the block's first lane
/// (16 lanes per block), a two's-complement exponent -8..7. // model: slices/mxm/mxm_insts.sail blk_exp
pub fn blk_exp(scales: &[u8], k: usize) -> i32 {
    let n = (scales[16 * (k / 16)] & 0x0F) as i32;
    if n >= 8 {
        n - 16
    } else {
        n
    }
}

/// The exact dot product of a bfp8 weight row (e4m3 data `wd`, block scales `ws`) and a bfp8 activation (`xd`, `xs`): the value
/// of a lane is its e4m3 element times 2 to the scale of its block. e4m3 has no infinity.
/// // model: slices/mxm/mxm_insts.sail bfp8_exact
pub fn bfp8_exact(wd: &[u8], ws: &[u8], xd: &[u8], xs: &[u8]) -> DSum {
    let mut d = DSum::new();
    for k in 0..wd.len() {
        let a = fp::fdecode(fp::FMT_E4M3, wd[k] as u32);
        let b = fp::fdecode(fp::FMT_E4M3, xd[k] as u32);
        d.term(a, b, 48 + blk_exp(ws, k) + blk_exp(xs, k));
    }
    d
}

/// One lane of the wide accumulator (extension wide_acc) and of the exact product that feeds it: the exact value in units of
/// 2^-48 in a two's-complement `i128`, or a special value (`sp`: 0 finite, 1 +Inf, 2 -Inf, 3 NaN), and `nz`: the value is zero
/// and every term that made it was -0. // model: slices/mxm/mxm_regs.sail struct wlane
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Wide {
    pub v: i128,
    pub sp: u8,
    pub nz: bool,
}

impl Wide {
    /// The exact product as a lane. // model: slices/mxm/mxm_insts.sail dsum_wide
    pub fn of_dsum(d: &DSum) -> Wide {
        if d.nan || (d.pinf && d.ninf) {
            Wide { v: 0, sp: 3, nz: false }
        } else if d.pinf {
            Wide { v: 0, sp: 1, nz: false }
        } else if d.ninf {
            Wide { v: 0, sp: 2, nz: false }
        } else {
            Wide { v: d.sum, sp: 0, nz: d.sum == 0 && d.negzero }
        }
    }

    /// Exact addition, wrapping at 128 bits; a special value is sticky (NaN, or +Inf plus -Inf, gives NaN).
    /// // model: slices/mxm/mxm_insts.sail wlane_add
    pub fn plus(self, b: Wide) -> Wide {
        let a = self;
        if a.sp == 3 || b.sp == 3 || (a.sp == 1 && b.sp == 2) || (a.sp == 2 && b.sp == 1) {
            Wide { v: 0, sp: 3, nz: false }
        } else if a.sp != 0 {
            a
        } else if b.sp != 0 {
            b
        } else {
            Wide { v: a.v.wrapping_add(b.v), sp: 0, nz: a.nz && b.nz }
        }
    }

    /// The read-out: the sum rounded ONCE to f32 (round to nearest even); a zero sum is -0 only when every term was -0.
    /// // model: slices/mxm/mxm_insts.sail wlane_read
    pub fn read(self) -> u32 {
        match self.sp {
            3 => fp::canon_nan(FMT32),
            1 => fp::inf_bits(FMT32, false),
            2 => fp::inf_bits(FMT32, true),
            _ if self.v == 0 => fp::zero_bits(FMT32, self.nz),
            _ if self.v < 0 => fp::fround(FMT32, true, self.v.unsigned_abs(), -48),
            _ => fp::fround(FMT32, false, self.v as u128, -48),
        }
    }
}

// A BF16 product in units of 2^-266 is below 2^522. A sum of any
// usize-addressable number of products fits in ten 64-bit limbs.
const BF16_LIMBS: usize = 10;

fn bf16_add_mag(acc: &mut [u64; BF16_LIMBS], mag: u64, shift: usize) {
    let limb = shift / 64;
    let mut carry = (u128::from(mag) << (shift % 64)) + u128::from(acc[limb]);
    acc[limb] = carry as u64;
    carry >>= 64;
    for word in &mut acc[limb + 1..] {
        if carry == 0 {
            return;
        }
        carry += u128::from(*word);
        *word = carry as u64;
        carry >>= 64;
    }
    debug_assert_eq!(carry, 0);
}

fn bf16_round_difference(mut large: [u64; BF16_LIMBS], small: &[u64; BF16_LIMBS], neg: bool) -> u32 {
    let mut borrow = false;
    for (a, b) in large.iter_mut().zip(small) {
        let (d, first) = a.overflowing_sub(*b);
        let (d, second) = d.overflowing_sub(u64::from(borrow));
        *a = d;
        borrow = first || second;
    }
    debug_assert!(!borrow);
    let mut words = [0u32; 2 * BF16_LIMBS];
    for (lsw, pair) in large.into_iter().zip(words.chunks_exact_mut(2)) {
        pair[0] = lsw as u32;
        pair[1] = (lsw >> 32) as u32;
    }
    let abs = num_bigint::BigInt::from_slice(num_bigint::Sign::Plus, &words);
    crate::bigfp::fround_big(FMT32, neg, &abs, -266)
}

// Each product is below 2^16 times a power of two. With 512 terms,
// a radix-2^32 coefficient has magnitude below 2^(16+31+9).
// It can be accumulated in a signed machine word without carrying.
fn bf16_radix(wlo: &[u8], whi: &[u8], xlo: &[u8], xhi: &[u8]) -> Option<u32> {
    let mut bins = [0i64; 17];
    let mut all_negzero = true;
    for (((&wl, &wh), &xl), &xh) in wlo.iter().zip(whi).zip(xlo).zip(xhi) {
        let w = u16::from_le_bytes([wl, wh]);
        let x = u16::from_le_bytes([xl, xh]);
        let we = (w >> 7) & 0xff;
        let xe = (x >> 7) & 0xff;
        if we == 0xff || xe == 0xff {
            return None;
        }
        let neg = (w ^ x) & 0x8000 != 0;
        all_negzero &= (w & 0x7fff == 0 || x & 0x7fff == 0) && neg;
        let wm = (w & 0x7f) + u16::from(we != 0) * 128;
        let xm = (x & 0x7f) + u16::from(xe != 0) * 128;
        let shift = usize::from(we.max(1) + xe.max(1) - 2);
        let magnitude = (i64::from(wm) * i64::from(xm)) << (shift & 31);
        bins[shift >> 5] += if neg { -magnitude } else { magnitude };
    }

    let (neg, words) = bf16_radix_words(bins);
    Some(bf16_round_words(&words, neg || all_negzero))
}

fn bf16_radix_words(bins: [i64; 17]) -> (bool, [u32; 17]) {
    // Propagate signed carries from the lowest coefficient. A 17-word
    // result holds the at most 531-bit magnitude and its sign.
    let mut words = [0u32; 17];
    let mut carry = 0;
    for (word, coeff) in words.iter_mut().zip(bins) {
        let value = coeff + carry;
        *word = value as u32;
        carry = value >> 32;
    }
    let neg = carry < 0;
    debug_assert!(carry == 0 || carry == -1);
    if neg {
        let mut increment = true;
        for word in &mut words {
            (*word, increment) = (!*word).overflowing_add(u32::from(increment));
        }
        debug_assert!(!increment);
    }
    (neg, words)
}

/// Activation bits unpacked once for the independent rows of a BF16 ABC.
pub struct Bf16Act<'a> {
    xlo: &'a [u8],
    xhi: &'a [u8],
    sig: [i16; 512],
    shift: [u16; 512],
    supported: bool,
}

impl<'a> Bf16Act<'a> {
    pub fn new(xlo: &'a [u8], xhi: &'a [u8]) -> Self {
        let mut act = Self { xlo, xhi, sig: [0; 512], shift: [0; 512], supported: xlo.len() <= 512 };
        if xlo.len() != xhi.len() {
            act.supported = false;
            return act;
        }
        for (k, (&lo, &hi)) in xlo.iter().zip(xhi).take(512).enumerate() {
            let bits = u16::from_le_bytes([lo, hi]);
            let exp = (bits >> 7) & 0xff;
            act.supported &= exp != 0xff;
            let mag = ((bits & 0x7f) + u16::from(exp != 0) * 128) as i16;
            act.sig[k] = if bits & 0x8000 != 0 { -mag } else { mag };
            act.shift[k] = exp.saturating_sub(1);
        }
        act
    }
}

/// Exact f32-rounded BF16 dot sharing the activation unpack across rows.
pub fn bf16_dot_prepared(wlo: &[u8], whi: &[u8], act: &Bf16Act<'_>) -> u32 {
    let n = act.xlo.len();
    if act.supported && wlo.len() == n && whi.len() == n {
        if let Some(bits) = bf16_radix_prepared(wlo, whi, act) {
            return bits;
        }
    }
    bf16_dot(wlo, whi, act.xlo, act.xhi)
}

/// Fixed radix preserving every exponent bit of the mathematical product.
fn bf16_radix_prepared(wlo: &[u8], whi: &[u8], act: &Bf16Act<'_>) -> Option<u32> {
    let mut bins = [0i64; 17];
    let n = wlo.len();
    for (((&lo, &hi), &xs), &xe) in wlo.iter().zip(whi).zip(&act.sig[..n]).zip(&act.shift[..n]) {
        let w = u16::from_le_bytes([lo, hi]);
        let we = (w >> 7) & 0xff;
        if we == 0xff {
            return None;
        }
        let wm = (w & 0x7f) + u16::from(we != 0) * 128;
        let shift = usize::from(we.saturating_sub(1) + xe);
        let term = (i64::from(wm) * i64::from(xs)) << (shift & 31);
        bins[shift >> 5] += if w & 0x8000 != 0 { -term } else { term };
    }
    Some(bf16_round_prepared(bins, wlo, whi, act))
}

fn bf16_round_prepared(bins: [i64; 17], wlo: &[u8], whi: &[u8], act: &Bf16Act<'_>) -> u32 {
    let (neg, words) = bf16_radix_words(bins);
    let zero_neg = if words.iter().all(|&word| word == 0) {
        wlo.iter().zip(whi).zip(act.xlo).zip(act.xhi).all(|(((lo, hi), xlo), xhi)| {
            ((hi ^ xhi) & 0x80 != 0) && (*lo | (hi & 0x7f) == 0 || *xlo | (xhi & 0x7f) == 0)
        })
    } else {
        false
    };
    bf16_round_words(&words, neg || zero_neg)
}

/// Exact finite operand significands and base-2 exponents, prepacked once per installed matrix.
pub struct Bf16Weights {
    lanes: usize,
    packed: Vec<(i16, u16)>,
}

impl Bf16Weights {
    pub fn new<const W: usize>(planes: &[[u8; W]], rows: usize, lanes: usize) -> Self {
        let mut packed = Vec::with_capacity(rows * lanes);
        for pair in planes.chunks_exact(2).take(rows) {
            packed.extend(pair[0][..lanes].iter().zip(&pair[1][..lanes]).map(|(&lo, &hi)| {
                let bits = u16::from_le_bytes([lo, hi]);
                let exp = (bits >> 7) & 0xff;
                let mag = ((bits & 0x7f) + u16::from(exp != 0) * 128) as i16;
                (
                    if bits & 0x8000 != 0 { -mag } else { mag },
                    if exp == 0xff { u16::MAX } else { exp.saturating_sub(1) },
                )
            }));
        }
        Self { lanes, packed }
    }

    pub fn dot(&self, row: usize, wlo: &[u8], whi: &[u8], act: &Bf16Act<'_>) -> u32 {
        let lane = self.lanes;
        let offset = row * lane;
        if act.supported && act.xlo.len() == lane && self.packed.len() >= offset + lane {
            if let Some(bits) = self.dot_packed(&self.packed[offset..offset + lane], wlo, whi, act) {
                return bits;
            }
        }
        bf16_dot_prepared(wlo, whi, act)
    }

    fn dot_packed(&self, weights: &[(i16, u16)], wlo: &[u8], whi: &[u8], act: &Bf16Act<'_>) -> Option<u32> {
        let mut bins = [0i64; 17];
        let n = weights.len();
        for ((&(ws, we), &xs), &xe) in weights.iter().zip(&act.sig[..n]).zip(&act.shift[..n]) {
            if we == u16::MAX {
                return None;
            }
            let shift = usize::from(we + xe);
            bins[shift >> 5] += (i64::from(ws) * i64::from(xs)) << (shift & 31);
        }
        Some(bf16_round_prepared(bins, wlo, whi, act))
    }
}

/// Round a finite BF16 dot from its exact signed integer representation in units of 2^-266.
fn bf16_round_words(words: &[u32; 17], neg: bool) -> u32 {
    let Some(high) = words.iter().rposition(|&word| word != 0) else {
        return fp::zero_bits(FMT32, neg);
    };
    let top = (high * 32 + (31 - words[high].leading_zeros()) as usize) as i32;
    // The smallest binary32 quantum is 2^-149, or 2^117 units here.
    let shift = (top - 23).max(117) as usize;
    let word = shift / 32;
    let pair = u64::from(words[word]) | u64::from(words[word + 1]) << 32;
    let mut significand = (pair >> (shift % 32)) as u32;
    let guard = (shift - 1) / 32;
    let bit = 1u32 << ((shift - 1) % 32);
    let sticky = words[guard] & (bit - 1) != 0 || words[..guard].iter().any(|&word| word != 0);
    if words[guard] & bit != 0 && (sticky || significand & 1 == 1) {
        significand += 1;
    }
    let mut exponent = (top - 266).max(-126);
    if significand == 1 << 24 {
        significand >>= 1;
        exponent += 1;
    }
    let sign = u32::from(neg) << 31;
    if significand < 1 << 23 {
        sign | significand // zero or subnormal
    } else if exponent >= 128 {
        fp::inf_bits(FMT32, neg)
    } else {
        sign | (((exponent + 127) as u32) << 23) | (significand & 0x7f_ffff)
    }
}

/// The exact dot product of a bf16 weight row (low and high byte planes) and a bf16 activation, rounded once to f32. A product
/// is an integer multiple of 2^-266 (the smallest BF16 subnormal squared). The accumulator uses that integer grid. NaN
/// rules as [`f16_dot`]: any NaN operand, Inf x 0, or +Inf and -Inf products give the canonical NaN; an exact zero sum is -0
/// only when every product is -0. // model: slices/mxm/mxm_insts.sail bf16_dot
pub fn bf16_dot(wlo: &[u8], whi: &[u8], xlo: &[u8], xhi: &[u8]) -> u32 {
    debug_assert!(wlo.len() == whi.len() && wlo.len() == xlo.len() && wlo.len() == xhi.len());
    if wlo.len() <= 512 && wlo.len() == whi.len() && wlo.len() == xlo.len() && wlo.len() == xhi.len() {
        if let Some(bits) = bf16_radix(wlo, whi, xlo, xhi) {
            return bits;
        }
    }
    let mut nan = false;
    let mut pinf = false;
    let mut ninf = false;
    let mut all_negzero = true;
    let mut mags = [[0u64; BF16_LIMBS]; 2]; // positive and negative products
    for k in 0..wlo.len() {
        let a = fp::fdecode(FMT_BF16, (whi[k] as u32) << 8 | wlo[k] as u32);
        let b = fp::fdecode(FMT_BF16, (xhi[k] as u32) << 8 | xlo[k] as u32);
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
            bf16_add_mag(&mut mags[usize::from(neg)], a.mag * b.mag, (a.e + b.e + 266) as usize);
        }
    }
    if nan || (pinf && ninf) {
        fp::canon_nan(FMT32)
    } else if pinf {
        fp::inf_bits(FMT32, false)
    } else if ninf {
        fp::inf_bits(FMT32, true)
    } else {
        let [positive, negative] = mags;
        match positive.iter().rev().cmp(negative.iter().rev()) {
            std::cmp::Ordering::Equal => fp::zero_bits(FMT32, all_negzero),
            std::cmp::Ordering::Less => bf16_round_difference(negative, &positive, true),
            std::cmp::Ordering::Greater => bf16_round_difference(positive, &negative, false),
        }
    }
}
