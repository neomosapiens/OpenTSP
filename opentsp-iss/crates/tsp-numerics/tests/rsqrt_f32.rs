//! `rsqrt_f32_fast` against the generic evaluator (`fp_tr_value`: the f64 fast path with its 2^-40 filter, then the exact
//! port) on positive normal f32 inputs.
use tsp_numerics::fp::FMT32;
use tsp_numerics::transcendental::{self as tr, RSQRT};

fn check(x: u32, accepted: &mut u64, declined: &mut u64) {
    match tr::rsqrt_f32_fast(x) {
        Some(r) => {
            *accepted += 1;
            let want = tr::fp_tr_value(RSQRT, FMT32, x).unwrap();
            assert_eq!(r, want, "rsqrt({x:#010x}) fast {r:#010x} exact {want:#010x}");
        }
        None => *declined += 1,
    }
}

#[test]
fn declines_what_it_does_not_cover() {
    for x in [
        0u32,
        0x8000_0000,
        0x8000_0001,
        0x0000_0001,
        0x007f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0x3f80_0001 | 1 << 31,
    ] {
        assert_eq!(tr::rsqrt_f32_fast(x), None, "{x:#010x}");
        // the public entry still answers them
        let _ = tr::fp_transcendental(RSQRT, FMT32, x);
    }
}

#[test]
fn matches_exact_on_a_stride_sweep_and_every_binade_edge() {
    let (mut acc, mut dec) = (0u64, 0u64);
    // a prime stride over the whole positive range (about 4.2M inputs)
    let mut x = 0x0080_0000u32;
    while x < 0x7f80_0000 {
        check(x, &mut acc, &mut dec);
        x += 509;
    }
    // the first and last 2048 mantissas of every binade, and the powers of two with their neighbours
    for e in 1u32..255 {
        for m in (0..2048u32).chain((1 << 23) - 2048..1 << 23) {
            check(e << 23 | m, &mut acc, &mut dec);
        }
    }
    assert!(acc > 4_000_000, "accepted {acc}");
    assert!((dec as f64) < acc as f64 * 0.002, "declined {dec} of {}", acc + dec);
    println!("accepted {acc}, declined {dec}");
}

#[test]
fn public_entry_equals_generic_everywhere_sampled() {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..300_000 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let x = s as u32;
        let got = tr::fp_transcendental(RSQRT, FMT32, x);
        let want = tr::fp_tr_value(RSQRT, FMT32, x).unwrap();
        assert_eq!(got, want, "{x:#010x}");
    }
}

/// Every positive normal f32 (2^31 inputs, minutes): `cargo test --release -p tsp-numerics --test rsqrt_f32 -- --ignored`.
#[test]
#[ignore]
fn every_positive_normal_f32() {
    let threads = 4u32;
    let total = 0x7f80_0000u32 - 0x0080_0000;
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            std::thread::spawn(move || {
                let (mut acc, mut dec) = (0u64, 0u64);
                let mut x = 0x0080_0000 + t;
                while x < 0x7f80_0000 {
                    check(x, &mut acc, &mut dec);
                    x += threads;
                }
                (acc, dec)
            })
        })
        .collect();
    let (mut a, mut d) = (0, 0);
    for h in handles {
        let (x, y) = h.join().unwrap();
        a += x;
        d += y;
    }
    assert_eq!(a + d, total as u64);
    println!("accepted {a}, declined {d}");
}
