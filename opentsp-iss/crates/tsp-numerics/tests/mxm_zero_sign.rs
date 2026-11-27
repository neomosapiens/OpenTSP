//! The sign of a zero dot product (`finish` of `f16_dot_prepared`): -0 only when every product is a zero of the negative
//! sign, in each lane a zero operand against an operand of the other sign. Compared with the literal reference over
//! every lane pattern of a small value set, for the lengths of a whole chunk, a tail and both.
use tsp_numerics::mxm::{f16_dot_prepared, f16_dot_prepared_scalar, f16_dot_reference, F16Act};

const VALUES: [u16; 6] = [0x0000, 0x8000, 0x3c00, 0xbc00, 0x3800, 0xb800]; // +0 -0 1 -1 0.5 -0.5

fn planes(v: &[u16]) -> (Vec<u8>, Vec<u8>) {
    (v.iter().map(|&x| x as u8).collect(), v.iter().map(|&x| (x >> 8) as u8).collect())
}

fn check(w: &[u16], x: &[u16]) -> u32 {
    let (wlo, whi) = planes(w);
    let (xlo, xhi) = planes(x);
    let act = F16Act::new(&xlo, &xhi);
    let want = f16_dot_reference(&wlo, &whi, &xlo, &xhi);
    assert_eq!(f16_dot_prepared(&wlo, &whi, &act), want, "w {w:x?} x {x:x?}");
    assert_eq!(f16_dot_prepared_scalar(&wlo, &whi, &act), want, "scalar: w {w:x?} x {x:x?}");
    want
}

#[test]
fn every_pattern_of_two_and_three_lanes() {
    let (mut neg, mut pos) = (0, 0);
    for n in [1usize, 2, 3] {
        let total = VALUES.len().pow(2 * n as u32);
        for k in 0..total {
            let mut k2 = k;
            let (mut w, mut x) = (vec![0u16; n], vec![0u16; n]);
            for l in 0..n {
                w[l] = VALUES[k2 % 6];
                k2 /= 6;
                x[l] = VALUES[k2 % 6];
                k2 /= 6;
            }
            match check(&w, &x) {
                0x8000_0000 => neg += 1,
                0 => pos += 1,
                _ => {}
            }
        }
    }
    assert!(neg > 0 && pos > 0, "both signs of zero must occur: {neg} {pos}");
}

#[test]
fn long_rows_whole_chunks_and_tails() {
    let mut state = 0x9e3779b97f4a7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let (mut neg, mut pos) = (0, 0);
    for n in [8usize, 9, 16, 17, 64, 320] {
        for trial in 0..2000 {
            // a value of the given sign: a zero or a number of the set
            let signed = |r: u64, neg: bool| -> u16 {
                let mag = [0x0000u16, 0x0000, 0x3c00, 0x3800, 0x4000, 0x0001][(r % 6) as usize];
                mag | if neg { 0x8000 } else { 0 }
            };
            let (mut w, mut x) = (vec![0u16; n], vec![0u16; n]);
            for l in 0..n {
                let r = next();
                let s = r & 1 == 1;
                // a lane whose product is a zero of the negative sign: a zero operand against the other sign
                let (zero, other) = (if s { 0x8000u16 } else { 0 }, signed(r >> 8, !s));
                (w[l], x[l]) = if r & 2 == 0 { (zero, other) } else { (other, zero) };
                match trial % 4 {
                    // a few lanes of any kind (the row is then rarely a negative zero)
                    1 if r >> 20 & 31 == 0 => (w[l], x[l]) = (signed(r >> 24, s), signed(r >> 30, r >> 40 & 1 == 1)),
                    // a lane with a plain +0 against a number of either sign
                    2 if r >> 20 & 7 == 0 => (w[l], x[l]) = (0, signed(r >> 24, s)),
                    // one lane that cancels another: the sum is a zero of the positive sign
                    3 if l == 1 => (w[l], x[l]) = (0x3c00, 0x3c00),
                    3 if l == 2 => (w[l], x[l]) = (0x3c00, 0xbc00),
                    _ => {}
                }
            }
            match check(&w, &x) {
                0x8000_0000 => neg += 1,
                0 => pos += 1,
                _ => {}
            }
        }
    }
    assert!(neg > 50 && pos > 50, "both signs of zero must occur often: {neg} {pos}");
}
