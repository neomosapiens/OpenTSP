//! `planes::apply_planes` (the VXM kernels on byte planes, NEON on AArch64) against the per-lane `lanes::apply` of the
//! model's port, which stays in the tree as the reference: every operation and type on random and edge operands, the
//! f32 operations on a dense grid of edge values, both f16 conversions on every input pattern, and lengths that are
//! not a multiple of 16.
mod common;

use common::{Rng, F16_EDGES, F32_EDGES};
use tsp_numerics::lanes::{self, LaneOp};
use tsp_numerics::planes;
use tsp_numerics::EType;

fn element(rng: &mut Rng, t: EType) -> u32 {
    let r = rng.next();
    let bits = t.nbits();
    let mask = ((1u64 << bits) - 1) as u32;
    match (t, r % 8) {
        (EType::F16, 0..=2) => F16_EDGES[(r >> 8) as usize % F16_EDGES.len()],
        (EType::F32, 0..=2) => F32_EDGES[(r >> 8) as usize % F32_EDGES.len()] ^ (((r >> 40) as u32 & 1) << 31),
        (_, 3) => 0,
        (_, 4) => mask,
        (_, 5) => (r >> 8) as u32 & mask & 0x8000_00ff,
        _ => (r >> 16) as u32 & mask,
    }
}

/// Words to the four planes of `n` lanes (the first `ws` bytes of each word).
fn to_planes(w: &[u32], ws: usize) -> [Vec<u8>; 4] {
    let mut p: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; w.len()]);
    for (l, &x) in w.iter().enumerate() {
        for (b, plane) in p.iter_mut().enumerate().take(ws) {
            plane[l] = (x >> (8 * b)) as u8;
        }
    }
    p
}

fn refs(p: &[Vec<u8>; 4]) -> [&[u8]; 4] {
    [&p[0], &p[1], &p[2], &p[3]]
}

/// Run `apply_planes` and `apply` on `n` random lanes (or the given words) and compare every result plane.
fn check_words(op: LaneOp, ws: usize, wd: usize, a: &[u32], b: &[u32], m: &[u32]) {
    let n = a.len();
    let (pa, pb, pm) = (to_planes(a, ws), to_planes(b, ws), to_planes(m, ws));
    let mut out: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0xa5u8; n]);
    {
        let [o0, o1, o2, o3] = &mut out;
        planes::apply_planes(op, ws, wd, &refs(&pa), &refs(&pb), &refs(&pm), &mut [o0, o1, o2, o3]);
    }
    for l in 0..n {
        let want = lanes::apply(op, a[l], b[l], m[l]);
        let got = (0..wd).fold(0u32, |x, k| x | (out[k][l] as u32) << (8 * k));
        let mask = if wd == 4 { u32::MAX } else { (1u32 << (8 * wd)) - 1 };
        assert_eq!(got, want & mask, "{op:?} lane {l}/{n}: a={:#x} b={:#x} m={:#x}", a[l], b[l], m[l]);
    }
}

fn check(op: LaneOp, t: EType, wd: usize, rng: &mut Rng, n: usize) {
    let ws = t.width() as usize;
    let gen = |rng: &mut Rng| (0..n).map(|_| element(rng, t)).collect::<Vec<u32>>();
    let (a, b, m) = (gen(rng), gen(rng), gen(rng));
    check_words(op, ws, wd, &a, &b, &m);
}

#[test]
fn every_operation_and_type_equals_apply() {
    let mut rng = Rng(0x91a7_ee5d);
    for t in EType::ALL {
        let w = t.width() as usize;
        for op in 0..=13 {
            check(LaneOp::Bin(op, t), t, w, &mut rng, 320);
        }
        for op in [14, 15, 16, 20] {
            check(LaneOp::Un(op, t), t, w, &mut rng, 320);
        }
        check(LaneOp::Mask(t), t, w, &mut rng, 320);
        check(LaneOp::Select(t), t, w, &mut rng, 320);
        for mode in 0..6 {
            check(LaneOp::Cmp(mode, t), t, w, &mut rng, 320);
        }
    }
    for op in 25..=32 {
        check(LaneOp::Conv(op), lanes::conv_src(op), lanes::conv_dst(op).width() as usize, &mut rng, 320);
    }
    for st in EType::ALL {
        for dt in EType::ALL {
            check(LaneOp::Cvt(st, dt), st, dt.width() as usize, &mut rng, 320);
        }
    }
    for t in [EType::I8, EType::F16, EType::F32] {
        for op in 21..=24 {
            check(LaneOp::Un(op, t), t, t.width() as usize, &mut rng, 64);
        }
    }
}

#[test]
fn f32_arithmetic_on_a_dense_edge_grid() {
    // every pair of the edge values with both signs: each ordering and each NaN, zero and infinity combination
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
        check_words(LaneOp::Bin(op, EType::F32), 4, 4, &a, &b, &m);
    }
}

#[test]
fn f32_arithmetic_on_a_million_random_patterns() {
    let mut rng = Rng(0x5eed_0a11);
    let n = 1 << 20;
    // raw patterns: NaN payloads, subnormals and every exponent come up at their natural rate
    let a: Vec<u32> = (0..n).map(|_| rng.next() as u32).collect();
    let b: Vec<u32> = (0..n).map(|_| rng.next() as u32).collect();
    let m = vec![0u32; n];
    for op in [0u32, 3, 6, 9, 10] {
        check_words(LaneOp::Bin(op, EType::F32), 4, 4, &a, &b, &m);
    }
    // operands of the same magnitude, where rounding ties and cancellation happen
    let c: Vec<u32> = a.iter().zip(&b).map(|(&x, &y)| (x & 0xff80_0000) | (y & 0x7f_ffff)).collect();
    let d: Vec<u32> = a.iter().zip(&b).map(|(&x, &y)| (x & 0xffc0_0000) | (y & 0x3f_ffff) | 0x1000).collect();
    for op in [0u32, 3, 6, 9, 10] {
        check_words(LaneOp::Bin(op, EType::F32), 4, 4, &c, &d, &m);
    }
}

#[test]
fn f16_to_f32_on_every_pattern() {
    let a: Vec<u32> = (0..=0xffffu32).collect();
    let z = vec![0u32; a.len()];
    check_words(LaneOp::Conv(31), 2, 4, &a, &z, &z);
}

/// The sweep of all 2^32 patterns of `f32_to_f16_on_every_pattern` is `cargo test --release -- --ignored`; this one
/// covers every exponent and sign with a mantissa stride, the rounding neighbourhoods of every f16 value, and random
/// patterns.
#[test]
fn f32_to_f16_on_edges_neighbourhoods_and_random_patterns() {
    let mut v: Vec<u32> = Vec::new();
    for x in F32_EDGES {
        v.push(x);
        v.push(x ^ 0x8000_0000);
    }
    // every f16 value, its midpoint to the successor, and the f32 neighbours of each, both signs
    for h in 0..0x7c00u32 {
        let lo = f32::from_bits(lanes::f16_to_f32_bits(h));
        let hi = f32::from_bits(lanes::f16_to_f32_bits(h + 1));
        let mid = ((lo as f64 + hi as f64) / 2.0) as f32;
        for base in [lo.to_bits(), mid.to_bits(), hi.to_bits()] {
            for d in [-2i64, -1, 0, 1, 2] {
                let x = (base as i64 + d) as u32;
                v.push(x);
                v.push(x | 0x8000_0000);
            }
        }
    }
    for e in 0..=255u32 {
        for s in [0u32, 0x8000_0000] {
            for mant in (0..0x80_0000u32).step_by(0x1001) {
                v.push(s | e << 23 | mant);
            }
        }
    }
    let mut rng = Rng(0xf32_f16);
    v.extend((0..1_000_000).map(|_| rng.next() as u32));
    let pad = v.len() % 16;
    v.extend(std::iter::repeat_n(0, if pad == 0 { 0 } else { 16 - pad }));
    let z = vec![0u32; v.len()];
    check_words(LaneOp::Conv(32), 4, 2, &v, &z, &z);
}

#[test]
#[ignore = "sweeps 2^32 patterns (about a minute in release); run with --ignored"]
fn f32_to_f16_on_every_pattern() {
    const CHUNK: usize = 1 << 22;
    let z = vec![0u32; CHUNK];
    std::thread::scope(|s| {
        let threads = 8u64;
        let per = (1u64 << 32) / threads;
        for t in 0..threads {
            let z = &z;
            s.spawn(move || {
                let (lo, hi) = (t * per, (t + 1) * per);
                let mut start = lo;
                while start < hi {
                    let a: Vec<u32> = (start..start + CHUNK as u64).map(|x| x as u32).collect();
                    check_words(LaneOp::Conv(32), 4, 2, &a, z, z);
                    start += CHUNK as u64;
                }
            });
        }
    });
}

#[test]
fn lengths_that_are_not_a_multiple_of_16() {
    let mut rng = Rng(0x1e47_0a11);
    for n in [0usize, 1, 3, 4, 5, 15, 16, 17, 31, 33, 100, 319, 320] {
        for op in [0u32, 3, 6, 9, 10] {
            check(LaneOp::Bin(op, EType::F32), EType::F32, 4, &mut rng, n);
        }
        check(LaneOp::Conv(32), EType::F32, 2, &mut rng, n);
        check(LaneOp::Conv(31), EType::F16, 4, &mut rng, n);
        check(LaneOp::Bin(6, EType::I32), EType::I32, 4, &mut rng, n);
        check(LaneOp::Bin(0, EType::I8), EType::I8, 1, &mut rng, n);
    }
}

#[test]
fn words_and_planes_round_trip() {
    let mut rng = Rng(0xabcd_0123);
    for n in [0usize, 1, 15, 16, 17, 64, 320] {
        let w: Vec<u32> = (0..n).map(|_| rng.next() as u32).collect();
        for ws in 1..=4usize {
            let p = to_planes(&w, ws);
            let mut got = vec![0xffff_ffffu32; n];
            planes::planes_to_words(&refs(&p), ws, n, &mut got);
            for l in 0..n {
                let mask = if ws == 4 { u32::MAX } else { (1u32 << (8 * ws)) - 1 };
                assert_eq!(got[l], w[l] & mask, "ws {ws} n {n} lane {l}");
            }
            let mut out: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; n]);
            {
                let [o0, o1, o2, o3] = &mut out;
                planes::words_to_planes(&w, ws, n, &mut [o0, o1, o2, o3]);
            }
            for b in 0..ws {
                assert_eq!(out[b], p[b], "plane {b} ws {ws} n {n}");
            }
        }
    }
}
