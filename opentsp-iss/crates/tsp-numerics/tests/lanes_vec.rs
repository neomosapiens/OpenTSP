//! `lanes::apply_vec` (a vector of lanes through one operation, the f32 arithmetic and the f16 conversions as plain
//! loops) and the f16 conversions it uses, against the per-lane `lanes::apply` and `fp::fp_convert` of the model's
//! port, which stay in the tree as the reference.
mod common;

use common::{Rng, F16_EDGES, F32_EDGES};
use tsp_numerics::fp::{self, FMT16, FMT32};
use tsp_numerics::lanes::{self, LaneOp};
use tsp_numerics::EType;

/// A random element of type `t`, with a share of edge values.
fn element(rng: &mut Rng, t: EType) -> u32 {
    let r = rng.next();
    let bits = t.nbits();
    let mask = ((1u64 << bits) - 1) as u32;
    match (t, r % 8) {
        (EType::F16, 0..=2) => F16_EDGES[(r >> 8) as usize % F16_EDGES.len()],
        (EType::F32, 0..=2) => F32_EDGES[(r >> 8) as usize % F32_EDGES.len()],
        (_, 3) => 0,
        (_, 4) => mask,
        (_, 5) => (r >> 8) as u32 & mask & 0x8000_00ff,
        _ => (r >> 16) as u32 & mask,
    }
}

fn lanes_of(rng: &mut Rng, t: EType, n: usize) -> Vec<u32> {
    (0..n).map(|_| element(rng, t)).collect()
}

fn check(op: LaneOp, t: EType, rng: &mut Rng, n: usize) {
    let (a, b, m) = (lanes_of(rng, t, n), lanes_of(rng, t, n), lanes_of(rng, t, n));
    let mut got = vec![0xdead_beefu32; n];
    lanes::apply_vec(op, &a, &b, &m, &mut got);
    for l in 0..n {
        let want = lanes::apply(op, a[l], b[l], m[l]);
        assert_eq!(got[l], want, "{op:?} lane {l}: a={:#x} b={:#x} m={:#x}", a[l], b[l], m[l]);
    }
}

#[test]
fn f16_to_f32_matches_the_reference_for_every_pattern() {
    for h in 0..=0xffffu32 {
        assert_eq!(lanes::f16_to_f32_bits(h), fp::fp_convert(FMT16, FMT32, h), "{h:#06x}");
    }
    // bits above the format width are ignored, as `fdecode` ignores them
    for h in [0x1_3c00u32, 0xffff_7c01, 0x8000_0001, 0xdead_fc00] {
        assert_eq!(lanes::f16_to_f32_bits(h), fp::fp_convert(FMT16, FMT32, h), "{h:#x}");
    }
}

#[test]
fn f32_to_f16_matches_the_reference_at_the_rounding_boundaries() {
    // every f16 value, the midpoint to its successor, and the f32 neighbours of each, both signs
    for h in 0..0x7c00u32 {
        let lo = f32::from_bits(fp::fp_convert(FMT16, FMT32, h));
        let hi = f32::from_bits(fp::fp_convert(FMT16, FMT32, h + 1));
        let mid = ((lo as f64 + hi as f64) / 2.0) as f32;
        for base in [lo.to_bits(), mid.to_bits(), hi.to_bits()] {
            for d in [-2i64, -1, 0, 1, 2] {
                let x = (base as i64 + d) as u32;
                for s in [x, x | 0x8000_0000] {
                    assert_eq!(lanes::f32_to_f16_bits(s), fp::fp_convert(FMT32, FMT16, s), "{s:#010x}");
                }
            }
        }
    }
}

#[test]
fn f32_to_f16_matches_the_reference_on_edges_and_random_patterns() {
    for &x in &F32_EDGES {
        for s in [x, x ^ 0x8000_0000] {
            assert_eq!(lanes::f32_to_f16_bits(s), fp::fp_convert(FMT32, FMT16, s), "{s:#010x}");
        }
    }
    // every exponent with a few mantissa shapes, then random
    for e in 0..=255u32 {
        for m in [0u32, 1, 0x1000, 0x1001, 0x0fff, 0x1fff, 0x2000, 0x7f_ffff, 0x40_0000, 0x7f_e000, 0x7f_f000] {
            for sgn in [0u32, 0x8000_0000] {
                let x = sgn | (e << 23) | m;
                assert_eq!(lanes::f32_to_f16_bits(x), fp::fp_convert(FMT32, FMT16, x), "{x:#010x}");
            }
        }
    }
    let mut rng = Rng(0x5eed_f16f);
    for _ in 0..3_000_000 {
        let x = rng.next() as u32;
        assert_eq!(lanes::f32_to_f16_bits(x), fp::fp_convert(FMT32, FMT16, x), "{x:#010x}");
    }
    // the subnormal and overflow range of f16 with random low bits
    for _ in 0..2_000_000 {
        let r = rng.next();
        let e = 100 + (r % 50) as u32;
        let x = ((r >> 8) as u32 & 0x8000_0000) | (e << 23) | ((r >> 16) as u32 & 0x7f_ffff);
        assert_eq!(lanes::f32_to_f16_bits(x), fp::fp_convert(FMT32, FMT16, x), "{x:#010x}");
    }
}

#[test]
fn apply_vec_equals_apply_for_every_operation_and_type() {
    let mut rng = Rng(0xa991_7ec5);
    for t in EType::ALL {
        for op in 0..=13 {
            check(LaneOp::Bin(op, t), t, &mut rng, 320);
        }
        for op in [14, 15, 16, 20] {
            check(LaneOp::Un(op, t), t, &mut rng, 320);
        }
        check(LaneOp::Mask(t), t, &mut rng, 320);
        check(LaneOp::Select(t), t, &mut rng, 320);
        for mode in 0..6 {
            check(LaneOp::Cmp(mode, t), t, &mut rng, 320);
        }
    }
    for op in 25..=32 {
        let st = lanes::conv_src(op);
        check(LaneOp::Conv(op), st, &mut rng, 320);
    }
    for st in EType::ALL {
        for dt in EType::ALL {
            check(LaneOp::Cvt(st, dt), st, &mut rng, 320);
        }
    }
}

#[test]
fn apply_vec_equals_apply_for_transcendentals() {
    let mut rng = Rng(0x7a9a_55e5);
    for t in [EType::I8, EType::F16, EType::F32] {
        for op in 21..=24 {
            check(LaneOp::Un(op, t), t, &mut rng, 64);
        }
    }
}

#[test]
fn apply_vec_f16_transcendentals_on_every_pattern() {
    // the vector loop reads the table once; a word with bits above the format width takes the per-lane path
    let mut a: Vec<u32> = (0..=0xffffu32).collect();
    a.extend([0x1_3c00u32, 0xffff_7c01, 0x8000_0001, 0xdead_fc00]);
    let z = vec![0u32; a.len()];
    for op in 21..=24 {
        let mut got = vec![0xdead_beefu32; a.len()];
        lanes::apply_vec(LaneOp::Un(op, EType::F16), &a, &z, &z, &mut got);
        for (l, &x) in a.iter().enumerate() {
            assert_eq!(got[l], lanes::apply(LaneOp::Un(op, EType::F16), x, 0, 0), "op {op} a={x:#x}");
        }
    }
}

#[test]
fn apply_vec_f32_arithmetic_on_a_dense_edge_grid() {
    // every pair of the edge values, so each ordering and each NaN, zero and infinity combination runs
    let vals: Vec<u32> = F32_EDGES.iter().flat_map(|&x| [x, x ^ 0x8000_0000]).collect();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for &x in &vals {
        for &y in &vals {
            a.push(x);
            b.push(y);
        }
    }
    let m = vec![0u32; a.len()];
    for op in [0u32, 3, 6, 9, 10] {
        let mut got = vec![0u32; a.len()];
        lanes::apply_vec(LaneOp::Bin(op, EType::F32), &a, &b, &m, &mut got);
        for l in 0..a.len() {
            assert_eq!(
                got[l],
                lanes::apply(LaneOp::Bin(op, EType::F32), a[l], b[l], 0),
                "op {op} {:#x} {:#x}",
                a[l],
                b[l]
            );
        }
    }
}

#[test]
fn apply_vec_handles_lengths_that_are_not_a_vector_multiple() {
    let mut rng = Rng(0x1e47_0a11);
    for n in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 100, 319, 320] {
        check(LaneOp::Bin(0, EType::F32), EType::F32, &mut rng, n);
        check(LaneOp::Bin(9, EType::F32), EType::F32, &mut rng, n);
        check(LaneOp::Conv(32), EType::F32, &mut rng, n);
        check(LaneOp::Conv(31), EType::F16, &mut rng, n);
    }
}
