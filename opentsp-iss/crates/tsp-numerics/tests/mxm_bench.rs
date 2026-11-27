//! Microbenchmark of the f16 dot kernels on the shape of the step programs (160 rows of 320 lanes, 64 activation
//! lanes not zero). `cargo test --release -p tsp-numerics --test mxm_bench -- --ignored --nocapture`.
use std::time::Instant;
use tsp_numerics::mxm::{f16_dot_prepared, f16_dot_prepared_scalar, f16_dot_reference, F16Act};

fn rows(seed: u64) -> (Vec<Vec<u8>>, Vec<u8>, Vec<u8>) {
    let mut s = seed;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut lane = |zero: bool| -> u16 {
        if zero {
            0
        } else {
            (next() >> 16) as u16 & 0xbfff // finite, |v| < 2
        }
    };
    let mut w: Vec<Vec<u8>> = Vec::new();
    for _ in 0..160 {
        let row: Vec<u16> = (0..320).map(|_| lane(false)).collect();
        w.push(row.iter().map(|v| *v as u8).collect());
        w.push(row.iter().map(|v| (v >> 8) as u8).collect());
    }
    let x: Vec<u16> = (0..320).map(|k| lane(k >= 64)).collect();
    (w, x.iter().map(|v| *v as u8).collect(), x.iter().map(|v| (v >> 8) as u8).collect())
}

#[test]
#[ignore]
fn f16_dot_ns_per_row() {
    let (w, xlo, xhi) = rows(0x1234_5678_9abc_def1);
    let reps = 2_000;
    for (name, kind) in [("reference", 0), ("prepared scalar", 1), ("prepared (NEON on aarch64)", 2)] {
        let t = Instant::now();
        let mut acc = 0u32;
        for _ in 0..reps {
            let act = F16Act::new(&xlo, &xhi);
            for r in 0..160 {
                acc = acc.wrapping_add(match kind {
                    0 => f16_dot_reference(&w[2 * r], &w[2 * r + 1], &xlo, &xhi),
                    1 => f16_dot_prepared_scalar(&w[2 * r], &w[2 * r + 1], &act),
                    _ => f16_dot_prepared(&w[2 * r], &w[2 * r + 1], &act),
                });
            }
        }
        let ns = t.elapsed().as_nanos() as f64 / (reps * 160) as f64;
        println!("{name}: {ns:.1} ns per row, {:.1} us per ABC (checksum {acc:08x})", ns * 160.0 / 1000.0);
    }
}

/// The usual ABC of the step programs: 64 rows, the activation non-zero in one chunk of eight lanes (a padded vector).
#[test]
#[ignore]
fn f16_dot_ns_per_row_one_chunk() {
    let (w, mut xlo, mut xhi) = rows(0x1234_5678_9abc_def1);
    for k in 8..320 {
        xlo[k] = 0;
        xhi[k] = 0;
    }
    let reps = 20_000;
    let t = Instant::now();
    let mut acc = 0u32;
    for _ in 0..reps {
        let act = F16Act::new(&xlo, &xhi);
        acc = acc.wrapping_add(act.chunk_count() as u32);
    }
    println!("F16Act::new: {:.0} ns per activation (checksum {acc})", t.elapsed().as_nanos() as f64 / reps as f64);
    let act = F16Act::new(&xlo, &xhi);
    let t = Instant::now();
    for _ in 0..reps {
        for r in 0..64 {
            acc = acc.wrapping_add(f16_dot_prepared(&w[2 * r], &w[2 * r + 1], &act));
        }
    }
    println!(
        "f16_dot_prepared: {:.1} ns per row, {:.2} us per 64-row ABC (checksum {acc:08x})",
        t.elapsed().as_nanos() as f64 / (reps * 64) as f64,
        t.elapsed().as_nanos() as f64 / reps as f64 / 1000.0
    );
}
