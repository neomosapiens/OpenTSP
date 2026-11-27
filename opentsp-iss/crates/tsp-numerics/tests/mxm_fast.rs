//! The pieces of the f16 dot kernel that were made cheaper for the ABC: `f32_of_sum` (the sum in 2^-48 units to f32
//! without the general `fround`), the activation decoded by 8-lane words, the Inf/NaN scan with four accumulators, and
//! the zero-sum sign scan that stops at the first block that settles it. Each against the lane-by-lane reference
//! (`f16_dot_reference`, `fp::fround`) on the shapes that sit at the block boundaries (16, 64 and 8 lanes) and on the
//! values at the edges of the rounding.
use tsp_numerics::fp::{self, FMT32};
use tsp_numerics::mxm::{
    f16_dot_prepared, f16_dot_prepared_scalar, f16_dot_prepared_with, f16_dot_reference, f32_of_sum, has_inf_or_nan,
    F16Act,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn want(sum: i128) -> u32 {
    fp::fround(FMT32, sum < 0, sum.unsigned_abs(), -48)
}

#[test]
fn f32_of_sum_equals_fround_for_small_sums_and_every_length() {
    for m in 1..=(1i128 << 20) {
        assert_eq!(f32_of_sum(m), want(m), "{m}");
        assert_eq!(f32_of_sum(-m), want(-m), "-{m}");
    }
}

#[test]
fn f32_of_sum_equals_fround_at_the_rounding_boundaries() {
    let mut rng = Rng(0xf32_5001);
    for p in 0..=100u32 {
        // a power of two, its neighbours, and the ties and near-ties of the 24-bit significand at this length
        let base = 1u128 << p;
        let mut cands = vec![base, base + 1, base.wrapping_sub(1).max(1), base | 1];
        if p >= 25 {
            let d = p - 23;
            let half = 1u128 << (d - 1);
            for lo in [0u128, 1, (1 << 23) - 1, (1 << 24) - 1] {
                let top = base | (lo << d);
                cands.extend([top + half, top + half - 1, top + half + 1, top | (half - 1), top + (1 << d) - 1]);
            }
        }
        for _ in 0..2000 {
            let r = ((rng.next() as u128) << 64 | rng.next() as u128) >> (127 - p);
            cands.push(r | base);
        }
        for m in cands {
            if m == 0 || m >= 1 << 127 {
                continue;
            }
            let (a, b) = (m as i128, -(m as i128));
            assert_eq!(f32_of_sum(a), want(a), "{m:#x}");
            assert_eq!(f32_of_sum(b), want(b), "-{m:#x}");
        }
    }
}

/// An f16 from sign, exponent field and fraction.
fn h(s: u16, e: u16, f: u16) -> u16 {
    s << 15 | e << 10 | f
}

fn split(v: &[u16]) -> (Vec<u8>, Vec<u8>) {
    (v.iter().map(|v| *v as u8).collect(), v.iter().map(|v| (v >> 8) as u8).collect())
}

/// `f16_dot_prepared`, its scalar form and the reference agree on one row.
fn check(w: &[u16], x: &[u16], what: &str) {
    let (wlo, whi) = split(w);
    let (xlo, xhi) = split(x);
    let act = F16Act::new(&xlo, &xhi);
    let r = f16_dot_reference(&wlo, &whi, &xlo, &xhi);
    assert_eq!(f16_dot_prepared(&wlo, &whi, &act), r, "{what}: prepared, n {}", w.len());
    assert_eq!(f16_dot_prepared_scalar(&wlo, &whi, &act), r, "{what}: scalar, n {}", w.len());
}

const LENGTHS: [usize; 14] = [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 320];

#[test]
fn the_zero_sum_sign_scan_sees_a_violation_at_every_lane() {
    let neg_zero = h(1, 0, 0);
    for n in LENGTHS {
        // every product -0 (weight -0, activation +0): the sum is -0
        let w = vec![neg_zero; n];
        let x = vec![0u16; n];
        check(&w, &x, "all -0");
        for j in 0..n {
            // a +0 product at lane j: the sum is +0
            let mut w2 = w.clone();
            w2[j] = 0;
            check(&w2, &x, &format!("+0 product at lane {j}"));
            // a non-zero pair at lane j, cancelled by another at lane k: the sum is exactly zero and +0
            if n > 1 {
                let k = (j + 1 + n / 3) % n;
                if k != j {
                    let mut w3 = w.clone();
                    let mut x3 = x.clone();
                    w3[j] = h(0, 15, 0);
                    x3[j] = h(0, 15, 0);
                    w3[k] = h(1, 15, 0);
                    x3[k] = h(0, 15, 0);
                    check(&w3, &x3, &format!("cancelling pair at {j} and {k}"));
                }
            }
            // a -0 activation against a +0 weight: also a negative product
            let mut x4 = x.clone();
            x4[j] = neg_zero;
            let mut w4 = w.clone();
            w4[j] = 0;
            check(&w4, &x4, &format!("-0 activation at lane {j}"));
        }
    }
}

#[test]
fn the_activation_and_the_special_scans_at_every_block_boundary() {
    let mut rng = Rng(0xac7_0001);
    for n in LENGTHS {
        let finite =
            |rng: &mut Rng| h((rng.next() & 1) as u16, 1 + (rng.next() % 20) as u16, (rng.next() & 0x3ff) as u16);
        for density in [0u64, 1, 2, 8, 100] {
            let w: Vec<u16> = (0..n).map(|_| finite(&mut rng)).collect();
            let x: Vec<u16> = (0..n).map(|_| if rng.next() % 100 < density { finite(&mut rng) } else { 0 }).collect();
            check(&w, &x, &format!("density {density}"));
            // a non-zero activation at each lane alone, on a -0 background
            if density == 1 {
                for j in 0..n {
                    let mut x1 = vec![h(1, 0, 0); n];
                    x1[j] = finite(&mut rng);
                    check(&w, &x1, &format!("single activation {j}"));
                }
            }
        }
        // an Inf or NaN at each lane, on the weight side and on the activation side
        let w: Vec<u16> = (0..n).map(|_| finite(&mut rng)).collect();
        let x: Vec<u16> = (0..n).map(|_| finite(&mut rng)).collect();
        for j in 0..n {
            for special in [h(0, 31, 0), h(1, 31, 0), h(0, 31, 1), h(0, 31, 0x200), h(1, 31, 0x3ff)] {
                let mut w2 = w.clone();
                w2[j] = special;
                check(&w2, &x, &format!("weight special at {j}"));
                let mut x2 = x.clone();
                x2[j] = special;
                check(&w, &x2, &format!("activation special at {j}"));
            }
        }
    }
}

#[test]
fn rows_of_every_scale_round_like_the_reference() {
    // sums that reach each exponent range, so `f32_of_sum` sees every length through the kernel
    let mut rng = Rng(0x5ca1_e001);
    for _ in 0..20_000 {
        let n = 1 + (rng.next() % 320) as usize;
        let e0 = (rng.next() % 30) as u16;
        let w: Vec<u16> = (0..n)
            .map(|_| h((rng.next() & 1) as u16, (e0 + (rng.next() % 3) as u16).min(30), (rng.next() & 0x3ff) as u16))
            .collect();
        let x: Vec<u16> = (0..n)
            .map(|_| {
                if rng.next().is_multiple_of(4) {
                    0
                } else {
                    h((rng.next() & 1) as u16, (rng.next() % 31) as u16, (rng.next() & 0x3ff) as u16)
                }
            })
            .collect();
        check(&w, &x, "scale");
    }
}

#[test]
fn a_chunk_is_kept_for_every_non_zero_pattern_of_one_lane() {
    // one activation lane, by itself in its chunk, for every high byte (low byte zero) and every low byte (high byte
    // zero) that is not zero and not an Inf or NaN
    let n = 64;
    let w: Vec<u16> = (0..n).map(|k| h(0, 10 + (k % 5) as u16, (k * 37 % 1024) as u16)).collect();
    for hi in (1u16..=0xff).filter(|hi| hi & 0x7c != 0x7c) {
        for (lane, v) in [(3usize, hi << 8), (20, hi << 8 | 1), (n - 1, hi << 8)] {
            let mut x = vec![0u16; n];
            x[lane] = v;
            check(&w, &x, &format!("hi {hi:#x} at {lane}"));
        }
    }
    for lo in 1u16..=0xff {
        let mut x = vec![0u16; n];
        x[9] = lo;
        check(&w, &x, &format!("lo {lo:#x}"));
    }
}

#[test]
fn a_row_flag_known_in_advance_gives_the_same_dot_as_the_scan() {
    let mut rng = Rng(0x5ca9_f1a6);
    for n in [8usize, 16, 40, 128, 320] {
        for _ in 0..3000 {
            // weights: mostly finite f16 (exponent below 31), now and then an Inf or a NaN in a lane
            let wlo: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
            let mut whi: Vec<u8> = (0..n).map(|_| (rng.next() as u8) & 0xbf).collect();
            if rng.next().is_multiple_of(3) {
                let k = (rng.next() % n as u64) as usize;
                whi[k] = 0x7c | (rng.next() as u8 & 0x83);
            }
            let xlo: Vec<u8> =
                (0..n).map(|_| if rng.next().is_multiple_of(5) { rng.next() as u8 } else { 0 }).collect();
            let xhi: Vec<u8> =
                (0..n).map(|_| if rng.next().is_multiple_of(5) { (rng.next() as u8) & 0xbf } else { 0 }).collect();
            let act = F16Act::new(&xlo, &xhi);
            let flag = has_inf_or_nan(&whi);
            let want = f16_dot_prepared(&wlo, &whi, &act);
            assert_eq!(f16_dot_prepared_with(&wlo, &whi, &act, flag), want, "n {n}");
            assert_eq!(want, f16_dot_reference(&wlo, &whi, &xlo, &xhi), "n {n}");
            // a clean row may be given as clean, a special one as special (a clean row as special only takes the slow path)
            if !flag {
                assert_eq!(f16_dot_prepared_with(&wlo, &whi, &act, true), want, "n {n}");
            }
        }
    }
}
