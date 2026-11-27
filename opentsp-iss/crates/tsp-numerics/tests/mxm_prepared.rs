//! `f16_dot_prepared` (the activation decoded once for every row of an ABC) and `f16_dot` against a literal
//! transcription of `slices/mxm/mxm_insts.sail` f16_dot.
//!
//! The fast kernel sums a lane's exact product into one of two 64-bit accumulators split at 2^32 of the 2^-48 fixed
//! point, visits only the activation lanes that are not zero, and leaves a row with an Inf or NaN operand, or an
//! exactly zero sum, to the lane-by-lane rules. The cases below aim at each of those: every exponent pair, the
//! accumulator split, the largest sums (the carry out of the low accumulator), cancellation to zero and to the
//! smallest residue, the sign of a zero sum, and specials on either side.
use tsp_numerics::fp::{self, FClass, FMT16, FMT32};
use tsp_numerics::mxm::{f16_dot, f16_dot_prepared, f16_dot_prepared_scalar, f16_dot_reference, F16Act};

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

fn split(v: &[u16]) -> (Vec<u8>, Vec<u8>) {
    (v.iter().map(|v| *v as u8).collect(), v.iter().map(|v| (v >> 8) as u8).collect())
}

/// Every kernel on one row: the prepared activation, `f16_dot` and the kept scalar routine.
fn same(w: &[u16], x: &[u16]) {
    let ((wlo, whi), (xlo, xhi)) = (split(w), split(x));
    let want = reference(&wlo, &whi, &xlo, &xhi);
    let act = F16Act::new(&xlo, &xhi);
    assert_eq!(f16_dot_prepared(&wlo, &whi, &act), want, "prepared: w {w:x?} x {x:x?}");
    assert_eq!(f16_dot_prepared_scalar(&wlo, &whi, &act), want, "scalar prepared: w {w:x?} x {x:x?}");
    assert_eq!(f16_dot(&wlo, &whi, &xlo, &xhi), want, "f16_dot: w {w:x?} x {x:x?}");
    assert_eq!(f16_dot_reference(&wlo, &whi, &xlo, &xhi), want, "scalar: w {w:x?} x {x:x?}");
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Every f16 bit pattern against the values that decide the special and zero paths, in both positions, alone and
/// beside a lane that carries a sum.
#[test]
fn every_f16_against_the_deciding_values() {
    let others: [u16; 11] = [0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0x7c01, 0x0001, 0x8001, 0x3c00, 0xbc00, 0x7bff];
    for v in 0..=u16::MAX {
        for o in others {
            same(&[v], &[o]);
            same(&[o], &[v]);
            same(&[v, o], &[o, v]);
            same(&[v, 0x3c00, o], &[0x3c00, v, 0x8000]);
        }
    }
}

/// Every pair of exponent fields (finite, so every term shift 0..=58 and the split at 32), with the extreme
/// significands, as a lone product and against a cancelling twin plus the smallest product.
#[test]
fn every_exponent_pair_and_the_accumulator_split() {
    for ea in 0..31u16 {
        for eb in 0..31u16 {
            for (fa, fb) in [(0x3ffu16, 0x3ffu16), (1, 1), (0x200, 0x155), (0, 1)] {
                let a = ea << 10 | fa;
                let b = eb << 10 | fb;
                same(&[a], &[b]);
                same(&[a, a | 0x8000], &[b, b]);
                same(&[a, a | 0x8000, 0x0001], &[b, b, 0x0001]);
                same(&[a, 0x0001], &[b | 0x8000, 0x0001]);
            }
        }
    }
}

/// The largest sums: 320 lanes of the largest product at every shift (each accumulator at its bound), all one sign,
/// alternating, and with the smallest product of the other sign (a borrow through every bit of both).
#[test]
fn carry_and_borrow_at_the_accumulator_bounds() {
    for e in 0..31u16 {
        let big = e << 10 | 0x3ff;
        let w = vec![big; 320];
        let x = vec![0x7bffu16; 320];
        same(&w, &x);
        let mut xn = x.clone();
        xn[319] = 0x0001;
        let mut wn = w.clone();
        wn[319] = 0x8001;
        same(&wn, &xn);
        let alt: Vec<u16> = (0..320).map(|k| if k % 2 == 0 { big } else { big | 0x8000 }).collect();
        same(&alt, &x);
        let mut alt1 = alt.clone();
        alt1.push(0x0001);
        let mut x1 = x.clone();
        x1.push(0x0001);
        same(&alt1, &x1);
        let mixed: Vec<u16> = (0..320).map(|k| if k < 160 { big } else { 0x83ff }).collect();
        same(&mixed, &x);
    }
}

/// A zero sum: -0 only when every product is -0, whatever the lanes the kernel skips.
#[test]
fn the_sign_of_a_zero_sum() {
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    for _ in 0..20_000 {
        let n = 1 + (r.next() % 40) as usize;
        let w: Vec<u16> = (0..n).map(|_| if r.next().is_multiple_of(2) { 0 } else { 0x8000 }).collect();
        let x: Vec<u16> = (0..n)
            .map(|_| match r.next() % 4 {
                0 => 0,
                1 => 0x8000,
                2 => 0x3c00,
                _ => 0xb555,
            })
            .collect();
        same(&w, &x);
        let mut wc = w.clone();
        let mut xc = x.clone();
        wc.extend([0x3c00, 0x3c00]);
        xc.extend([0x4000, 0xc000]);
        same(&wc, &xc);
    }
}

/// Random rows: uniform bit patterns, finite only, and the padded shapes of a step (most activation lanes zero),
/// many rows against one prepared activation.
#[test]
fn random_rows_against_one_activation() {
    let mut r = Rng(0x243f_6a88_85a3_08d3);
    for case in 0..3_000 {
        let n = [320usize, 160, 64, 1, 17, 8, 9, 23][case % 8];
        let zero_share = [0u64, 50, 90, 99][(case / 5) % 4];
        let finite = case % 3 != 0;
        let lane = |r: &mut Rng| -> u16 {
            let v = r.next();
            if v % 100 < zero_share {
                if v & 0x100 == 0 {
                    0
                } else {
                    0x8000
                }
            } else {
                let b = (v >> 16) as u16;
                if finite && b & 0x7c00 == 0x7c00 {
                    b & 0xfbff
                } else {
                    b
                }
            }
        };
        let x: Vec<u16> = (0..n).map(|_| lane(&mut r)).collect();
        let (xlo, xhi) = split(&x);
        let act = F16Act::new(&xlo, &xhi);
        for _ in 0..8 {
            let w: Vec<u16> = (0..n).map(|_| lane(&mut r)).collect();
            let (wlo, whi) = split(&w);
            assert_eq!(f16_dot_prepared(&wlo, &whi, &act), reference(&wlo, &whi, &xlo, &xhi), "w {w:x?} x {x:x?}");
            assert_eq!(f16_dot_prepared_scalar(&wlo, &whi, &act), reference(&wlo, &whi, &xlo, &xhi), "scalar w {w:x?}");
        }
    }
}
