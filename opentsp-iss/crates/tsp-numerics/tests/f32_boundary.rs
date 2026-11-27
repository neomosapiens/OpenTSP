//! f32 TanH, Exp, RSqrt and Log on the inputs whose value lies closest to a rounding boundary
//! (`data/f32_boundary.bin`, made by `tools/gen_f32_boundary.py`: up to 50 000 closest per function with
//! the mpmath oracle's results), in the normal test run. The fast path takes the host libm's f64
//! value when it is more than 2^-40 (relative) from a boundary: this checks that margin on the libm
//! of the machine that runs the tests, as the exhaustive i8 and f16 tests do for their formats, and
//! that the Ziv port, which the fast path declines to, gives the same results.
use tsp_numerics::fp::FMT32;
use tsp_numerics::transcendental::{self as tr, Pre};
use tsp_numerics::ziv;

const DATA: &[u8] = include_bytes!("data/f32_boundary.bin");
const MAGIC: u32 = 0x4232_3346;

struct Sample {
    op: u32,
    x: u32,
    want: u32,
}

fn word(at: usize) -> u32 {
    u32::from_le_bytes(DATA[at..at + 4].try_into().unwrap())
}

fn samples() -> Vec<Sample> {
    assert_eq!(word(0), MAGIC, "data/f32_boundary.bin: bad magic");
    let mut at = 20;
    let mut v = Vec::new();
    for (k, op) in [21u32, 22, 23, 24].into_iter().enumerate() {
        for _ in 0..word(4 + 4 * k) {
            v.push(Sample { op, x: word(at), want: word(at + 4) });
            at += 8;
        }
    }
    assert_eq!(at, DATA.len(), "data/f32_boundary.bin: trailing bytes");
    v
}

/// Release builds check every input with the Ziv port; debug builds (the big-integer port is slow
/// there) every 16th.
fn ziv_stride() -> usize {
    if std::env::var_os("TSP_NUMERICS_FULL").is_some() || !cfg!(debug_assertions) {
        1
    } else {
        16
    }
}

#[test]
fn the_sample_covers_every_function() {
    let s = samples();
    for op in [21u32, 22, 23, 24] {
        let n = s.iter().filter(|e| e.op == op).count();
        assert!(n >= 10_000, "op {op}: only {n} inputs");
    }
}

#[test]
fn the_fast_path_and_the_fallback_give_the_correct_result() {
    let (mut fast, mut declined) = (0usize, 0usize);
    for e in samples() {
        assert_eq!(tr::fp_tr_value(e.op, FMT32, e.x), Some(e.want), "op {} x={:08x}", e.op, e.x);
        // the fast path alone: whenever it answers it is right
        let Pre::Eval(a) = tr::fp_tr_pre(e.op, FMT32, e.x) else {
            panic!("op {} x={:08x}: not an Eval input", e.op, e.x)
        };
        match tr::tr_fast(e.op, FMT32, &a) {
            Some(r) => {
                assert_eq!(r, e.want, "fast path, op {} x={:08x}", e.op, e.x);
                fast += 1;
            }
            None => declined += 1,
        }
    }
    // both paths are exercised: the closest inputs are declined, those beyond the 2^-40 filter are answered
    assert!(fast > 10_000 && declined > 1_000, "fast {fast}, declined {declined}");
}

#[test]
fn the_ziv_port_gives_the_correct_result() {
    let stride = ziv_stride();
    for e in samples().into_iter().step_by(stride) {
        assert_eq!(ziv::fp_tr_value(e.op, FMT32, e.x), Some(e.want), "ziv, op {} x={:08x}", e.op, e.x);
    }
}
