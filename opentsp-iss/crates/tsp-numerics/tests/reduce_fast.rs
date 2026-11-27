//! `reduce_fold` for f32 Add (host adds, one canonical NaN at the end) against the step-by-step `k_binary` fold.
use tsp_numerics::lanes::{self, ReduceOp};
use tsp_numerics::types::EType;

fn reference(op: ReduceOp, t: EType, e: &[u32]) -> u32 {
    let mut acc = e[0];
    for &x in &e[1..] {
        acc = lanes::k_binary(op.kop(), t, acc, x);
    }
    acc
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

const EDGES: [u32; 16] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x0080_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x7f7f_ffff,
    0xff7f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0xffc0_0001,
    0x7f80_0001,
    0x4b80_0000,
];

#[test]
fn f32_add_equals_the_stepwise_fold_on_edges_and_random_groups() {
    let mut r = Rng(0x2545_f491_4f6c_dd1d);
    let mut n = 0;
    // every pair of edge values in lanes 0 and k, zero elsewhere or filled with another edge
    for a in EDGES {
        for b in EDGES {
            for k in [1usize, 7, 15] {
                for fill in [0u32, 0x8000_0000, 0x3f80_0000] {
                    let mut g = [fill; 16];
                    g[0] = a;
                    g[k] = b;
                    assert_eq!(
                        lanes::reduce_fold(ReduceOp::Add, EType::F32, &g),
                        reference(ReduceOp::Add, EType::F32, &g),
                        "{g:x?}"
                    );
                    n += 1;
                }
            }
        }
    }
    // random groups: any bit pattern, a mix of edge values and patterns, and lengths other than 16
    for i in 0..400_000 {
        let len = [16usize, 16, 16, 2, 3, 5, 32][i % 7];
        let g: Vec<u32> = (0..len)
            .map(|_| match r.next() % 4 {
                0 => EDGES[(r.next() % 16) as usize],
                1 => (r.next() as u32 & 0x807f_ffff) | 0x3f00_0000, // same binade: exact and tie cases
                _ => r.next() as u32,
            })
            .collect();
        assert_eq!(
            lanes::reduce_fold(ReduceOp::Add, EType::F32, &g),
            reference(ReduceOp::Add, EType::F32, &g),
            "{g:x?}"
        );
        n += 1;
    }
    println!("{n} groups");
}

#[test]
fn other_ops_and_types_still_take_the_generic_fold() {
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    for op in [ReduceOp::Add, ReduceOp::Max, ReduceOp::Min] {
        for t in [EType::I8, EType::U8, EType::F16, EType::F32] {
            for _ in 0..2000 {
                let g: Vec<u32> =
                    (0..16).map(|_| (r.next() as u32) & if t == EType::F32 { u32::MAX } else { 0xffff }).collect();
                let g: Vec<u32> =
                    if matches!(t, EType::I8 | EType::U8) { g.iter().map(|x| x & 0xff).collect() } else { g };
                assert_eq!(lanes::reduce_fold(op, t, &g), reference(op, t, &g), "{op:?} {t:?} {g:x?}");
            }
        }
    }
}
