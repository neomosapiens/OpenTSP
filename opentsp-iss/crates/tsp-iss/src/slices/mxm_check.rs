//! The sum check of an ABC (`microarchitecture.md` sec. 18; `slices/mxm/mxm_insts.sail` `mxm_check`).
//!
//! It runs only for an ABC with an injected fault (`F mxm`) pending for its plane: the arithmetic is exact, so without a fault
//! the actual sums equal the expected ones and the check could not find anything. The sums are exact integers: an integer
//! product sum as it is, an f16 one in units of 2^-48, an fp8 one in units of 2^-32, the finite products only.
use super::mxm::WType;
use crate::operands::Planes;
use crate::srf::Vec320;
use num_bigint::BigInt;
use tsp_numerics::fp::{self, FClass, FMT16, FMT_BF16, FMT_E4M3, FMT_E5M2};

/// What the check found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// No error.
    Clean,
    /// One lane was wrong; its product is corrected (the fault-free one).
    Corrected(usize),
    /// An error the check cannot correct (OTS-E-ECC-003).
    Uncorrectable,
}

/// The exact value of one element, in units of 2^-24 (f16) or 2^-16 (fp8) for the floating formats; `None` for a NaN or an
/// infinity. `sgn`: the integer element is signed (i8).
// model: slices/mxm/mxm_insts.sail chk_elem, fval_int
fn elem(t: WType, f: u32, lo: u8, hi: u8, sgn: bool) -> Option<i128> {
    let (v, s) = match t {
        WType::I8 | WType::U8 => return Some(if sgn { lo as i8 as i128 } else { lo as i128 }),
        WType::Fp8 => (fp::fdecode(if f & 1 == 0 { FMT_E4M3 } else { FMT_E5M2 }, lo as u32), 16),
        WType::F16 => (fp::fdecode(FMT16, lo as u32 | (hi as u32) << 8), 24),
        // `hi` is the scale byte of the lane's block: the low four bits, a two's-complement exponent
        WType::Bfp8 => (fp::fdecode(FMT_E4M3, lo as u32), 17 + ((hi << 4) as i8 >> 4) as i32),
        WType::Bf16 => unreachable!("a bf16 ABC is checked by `mxm_check_bf16` (its sums do not fit i128)"),
    };
    match v.cls {
        FClass::NaN | FClass::Inf => None,
        FClass::Zero => Some(0),
        FClass::Finite => {
            let m = (v.mag as i128) << (v.e + s) as u32;
            Some(if v.neg { -m } else { m })
        }
    }
}

/// The lane of the second stream value that holds the second byte of lane `k`: the lane itself (f16 high byte), or the first
/// lane of its block (bfp8 scale).
fn second_lane(t: WType, k: usize) -> usize {
    if t == WType::Bfp8 {
        16 * (k / 16)
    } else {
        k
    }
}

/// Flip bit `b` of the 96-bit two's-complement form of `p`.
// model: slices/mxm/mxm_insts.sail flip96
fn flip96(p: i128, b: u32) -> i128 {
    let m: i128 = 1 << 96;
    let u = p.rem_euclid(m) ^ (1i128 << b);
    if u >= m / 2 {
        u - m
    } else {
        u
    }
}

/// The check of one ABC of type `t` (`f`: the FP8 format) against the installed `rows` rows of `weights` (installed type
/// `wt`), with the flips `faults` = (lane, bit) in the exact sums. Primary row-major products are compared
/// with independently recomputed column-major nominal row products. The checker and captured operands are trusted;
/// every final multirow error is uncorrectable, independent of aggregate cancellation or injected fault count.
// model: slices/mxm/mxm_insts.sail mxm_check
#[allow(clippy::too_many_arguments)] // the operands of one ABC and its faults
pub fn mxm_check(
    weights: &[Vec320],
    rows: usize,
    vl: usize,
    wt: WType,
    t: WType,
    f: u32,
    x: &Planes,
    faults: &[(usize, u32)],
) -> Verdict {
    if t == WType::Bf16 {
        return mxm_check_bf16(weights, rows, vl, x, faults);
    }
    let ws = wt == WType::I8;
    let xs = t == WType::I8;
    let row = |r: usize| -> (&Vec320, Option<&Vec320>) {
        if t.is_pair() {
            (&weights[2 * r], Some(&weights[2 * r + 1]))
        } else {
            (&weights[r], None)
        }
    };
    let w_at = |r: usize, k: usize| -> Option<i128> {
        let (lo, hi) = row(r);
        elem(t, f, lo[k], hi.map_or(0, |h| h[second_lane(t, k)]), ws)
    };
    let xv: Vec<Option<i128>> =
        (0..vl).map(|k| elem(t, f, x[0][k], if t.is_pair() { x[1][second_lane(t, k)] } else { 0 }, xs)).collect();
    // Primary row-major products; the nominal checker recomputes them independently,
    // column-major from the captured operands. The checker is assumed unfaulted.
    let mut ps = vec![0i128; rows.max(1)];
    for (r, p) in ps.iter_mut().enumerate().take(rows) {
        for (k, xk) in xv.iter().enumerate().take(vl) {
            if let (Some(w), Some(xk)) = (w_at(r, k), *xk) {
                *p += w * xk;
            }
        }
    }
    let mut expected = vec![0i128; rows.max(1)];
    for (k, xk) in xv.iter().enumerate().take(vl) {
        let Some(xk) = *xk else { continue };
        for (r, nominal) in expected.iter_mut().enumerate().take(rows) {
            if let Some(w) = w_at(r, k) {
                *nominal += w * xk;
            }
        }
    }
    // the injected faults flip bits of the exact sums; a lane outside the installed rows carries no product
    for &(lane, bit) in faults {
        if lane < rows {
            ps[lane] = flip96(ps[lane], bit);
        }
    }
    classify_changed_rows(&ps[..rows], &expected[..rows])
}

/// The exact value of one bf16 element in units of 2^-133; `None` for a NaN or an infinity.
// model: slices/mxm/mxm_insts.sail chk_elem (W_BF16)
fn elem_bf16(lo: u8, hi: u8) -> Option<BigInt> {
    let v = fp::fdecode(FMT_BF16, lo as u32 | (hi as u32) << 8);
    match v.cls {
        FClass::NaN | FClass::Inf => None,
        FClass::Zero => Some(BigInt::from(0)),
        FClass::Finite => {
            let m = BigInt::from(v.mag) << (v.e + 133) as usize;
            Some(if v.neg { -m } else { m })
        }
    }
}

/// Flip bit `b` of the 96-bit two's-complement form of `p` (a bf16 sum exceeds i128, so this one is a big integer).
// model: slices/mxm/mxm_insts.sail flip96
fn flip96_big(p: &BigInt, b: u32) -> BigInt {
    let m = BigInt::from(1) << 96usize;
    let mut u = p % &m;
    if u < BigInt::from(0) {
        u += &m;
    }
    let u = u ^ (BigInt::from(1) << b as usize);
    if u >= (&m >> 1usize) {
        u - m
    } else {
        u
    }
}

/// [`mxm_check`] of a bf16 ABC: the same check on big integers (units of 2^-266 for a product).
// model: slices/mxm/mxm_insts.sail mxm_check (W_BF16)
fn mxm_check_bf16(weights: &[Vec320], rows: usize, vl: usize, x: &Planes, faults: &[(usize, u32)]) -> Verdict {
    let zero = BigInt::from(0);
    let w_at = |r: usize, k: usize| elem_bf16(weights[2 * r][k], weights[2 * r + 1][k]);
    let xv: Vec<Option<BigInt>> = (0..vl).map(|k| elem_bf16(x[0][k], x[1][k])).collect();
    let mut ps = vec![zero.clone(); rows.max(1)];
    for (r, p) in ps.iter_mut().enumerate().take(rows) {
        for (k, xk) in xv.iter().enumerate().take(vl) {
            if let (Some(w), Some(xk)) = (w_at(r, k), xk) {
                *p += w * xk;
            }
        }
    }
    let mut expected = vec![zero.clone(); rows.max(1)];
    for (k, xk) in xv.iter().enumerate().take(vl) {
        let Some(xk) = xk else { continue };
        for (r, nominal) in expected.iter_mut().enumerate().take(rows) {
            if let Some(w) = w_at(r, k) {
                *nominal += w * xk;
            }
        }
    }
    for &(lane, bit) in faults {
        if lane < rows {
            ps[lane] = flip96_big(&ps[lane], bit);
        }
    }
    classify_changed_rows(&ps[..rows], &expected[..rows])
}

/// Compare final row values against their independent nominal values. Fault count
/// is intentionally irrelevant: cancellations and repeated flips use the net value.
fn classify_changed_rows<T: PartialEq>(actual: &[T], expected: &[T]) -> Verdict {
    let mut wrong = actual.iter().zip(expected).enumerate().filter(|(_, (a, b))| a != b);
    match (wrong.next(), wrong.next()) {
        (None, _) => Verdict::Clean,
        (Some((row, _)), None) => Verdict::Corrected(row),
        (Some(_), Some(_)) => Verdict::Uncorrectable,
    }
}
