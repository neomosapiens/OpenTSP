//! The FP8 element types e4m3 and e5m2 (; `microarchitecture.md` sec. 5 and 6) against the digests that the Sail
//! model's unit test holds. `opentsp-arch-spec/model/unit_tests/test_fp8.sail` is generated from the independent oracle
//! (`test/oracle/numerics.py`, exact fractions, no Sail), and each of its 56 assertions compares the digest
//! h = (h * 257 + r + 1) mod 1000000007 of a kernel's results over a whole operand set (every pair of 8-bit values, every
//! binary16 value, binary32 patterns with four low halves, ...). This test parses those assertions and recomputes every digest
//! with the ISS kernels, so the ISS, the model and the oracle agree on every FP8 result without a second oracle.

use tsp_numerics::fp::{canon_nan, fdecode, fp_add, fround, FClass, FMT32, FMT_E4M3, FMT_E5M2};
use tsp_numerics::lanes::{k_binary, k_cmp, k_convert_any, k_unary};
use tsp_numerics::mxm::fp8_dot;
use tsp_numerics::EType;

const SAIL: &str = include_str!("../../../../opentsp-arch-spec/model/unit_tests/test_fp8.sail");

fn ty(name: &str) -> EType {
    match name {
        "E4M3" => EType::E4M3,
        "E5M2" => EType::E5M2,
        "F16" => EType::F16,
        "F32" => EType::F32,
        "I8" => EType::I8,
        "U8" => EType::U8,
        "I16" => EType::I16,
        "I32" => EType::I32,
        "B8" => EType::B8,
        _ => panic!("type {name}"),
    }
}

fn step(h: u64, r: u32) -> u64 {
    (h * 257 + r as u64 + 1) % 1_000_000_007
}

/// The digest the assertion `kind(z, args...)` asks for.
fn digest(kind: &str, args: &[&str]) -> u64 {
    let mut h = 0u64;
    match kind {
        "dg_bin" | "dg_cmp" => {
            let (op, t) = (args[0].parse::<u32>().unwrap(), ty(args[1]));
            for a in 0..256u32 {
                for b in 0..256u32 {
                    h = step(h, if kind == "dg_bin" { k_binary(op, t, a, b) } else { k_cmp(op, t, a, b) });
                }
            }
        }
        "dg_un" => {
            let (op, t) = (args[0].parse::<u32>().unwrap(), ty(args[1]));
            for a in 0..256u32 {
                h = step(h, k_unary(op, t, a));
            }
        }
        "dg_from16" => {
            let t = ty(args[0]);
            for a in 0..65536u32 {
                h = step(h, k_convert_any(EType::F16, t, a));
            }
        }
        "dg_from32" => {
            let t = ty(args[0]);
            for a in 0..65536u32 {
                for lo in [0u32, 32768, 32769, 32767] {
                    h = step(h, k_convert_any(EType::F32, t, a * 65536 + lo));
                }
            }
        }
        "dg_to" => {
            let (s, d) = (ty(args[0]), ty(args[1]));
            for a in 0..256u32 {
                h = step(h, k_convert_any(s, d, a));
            }
        }
        "dg_fromint" => {
            let (s, d) = (ty(args[0]), ty(args[1]));
            for a in 0..256u32 {
                let iv = match s {
                    EType::I16 => a * 257,
                    EType::I32 => a * 16843009,
                    _ => a,
                };
                h = step(h, k_convert_any(s, d, iv));
            }
        }
        _ => panic!("digest function {kind}"),
    }
    h
}

// model: unit_tests/test_fp8.sail test_fp8_exhaustive
#[test]
fn every_digest_of_the_model_unit_test_is_reproduced() {
    let mut n = 0;
    for line in SAIL.lines().filter(|l| l.contains("foreach (z") && l.contains("assert(dg_")) {
        // `  foreach (z from 0 to 0) { assert(dg_bin(z, 0, E4M3) == 330113675, "bin e4m3 0") };`
        let call = &line[line.find("assert(").unwrap() + 7..];
        let open = call.find('(').unwrap();
        let kind = &call[..open];
        let close = call.find(')').unwrap();
        let args: Vec<&str> = call[open + 1..close].split(',').map(|a| a.trim()).skip(1).collect(); // skip z
        let want: u64 = call[close + 1..]
            .trim_start()
            .trim_start_matches("==")
            .trim_start()
            .split(',')
            .next()
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert_eq!(digest(kind, &args), want, "{line}");
        n += 1;
    }
    assert_eq!(n, 56, "the model unit test holds 56 digests");
}

// model: unit_tests/test_fp8.sail test_fp8_landmarks
#[test]
fn landmarks() {
    // e4m3: no infinity, 0x7F is the only NaN (either sign), 448 = 14 * 2^5, saturation
    assert_eq!(fdecode(FMT_E4M3, 0x7E).cls, FClass::Finite);
    assert_eq!((fdecode(FMT_E4M3, 0x7E).mag, fdecode(FMT_E4M3, 0x7E).e), (14, 5));
    assert_eq!(fdecode(FMT_E4M3, 0x7F).cls, FClass::NaN);
    assert_eq!(fdecode(FMT_E4M3, 0xFF).cls, FClass::NaN);
    assert_eq!(canon_nan(FMT_E4M3), 0x7F);
    assert_eq!(k_convert_any(EType::F32, EType::E4M3, 0x43E8_0000), 0x7E, "464, the tie above 448, saturates");
    assert_eq!(k_convert_any(EType::F32, EType::E4M3, 0x7F80_0000), 0x7E, "+infinity saturates");
    assert_eq!(k_convert_any(EType::F32, EType::E4M3, 0xFF80_0000), 0xFE);
    assert_eq!(k_binary(0, EType::E4M3, 0x7E, 0x7E), 0x7E, "448 + 448 saturates");
    // e5m2: IEEE style, 57344 = 7 * 2^13, the tie above it is infinity
    assert_eq!(fdecode(FMT_E5M2, 0x7C).cls, FClass::Inf);
    assert_eq!(fdecode(FMT_E5M2, 0x7D).cls, FClass::NaN);
    assert_eq!((fdecode(FMT_E5M2, 0x7B).mag, fdecode(FMT_E5M2, 0x7B).e), (7, 13));
    assert_eq!(canon_nan(FMT_E5M2), 0x7E);
    assert_eq!(k_convert_any(EType::F32, EType::E5M2, 0x4770_0000), 0x7C, "61440 rounds to even: infinity");
    assert_eq!(k_binary(6, EType::E5M2, 0x7B, 0x7B), 0x7C);
    // the type table
    assert_eq!((EType::E4M3.code(), EType::E5M2.code()), (11, 12));
    assert_eq!((EType::E4M3.width(), EType::E5M2.width()), (1, 1));
    assert!(EType::E4M3.is_float() && EType::E5M2.is_float());
    assert_eq!(EType::from_code(13), None);
    assert_eq!(fround(FMT_E4M3, false, 1, -9), 0x01);
    assert_eq!(fp_add(FMT32, 0x3F80_0000, 0x3F80_0000), 0x4000_0000);
}

/// The cases of the model's `test_mxm_fp8_abc` (`unit_tests/test_vxm_mxm.sail`), with 16 lanes.
#[test]
fn fp8_dot_cases() {
    let mut w = vec![0u8; 16];
    let mut x = vec![0u8; 16];
    // e4m3: [1, 2] . [0.5, 3] = 6.5 = 0x40D00000
    (w[0], w[1], x[0], x[1]) = (0x38, 0x40, 0x30, 0x44);
    assert_eq!(fp8_dot(FMT_E4M3, &w, &x), 0x40D0_0000);
    // e5m2: 1 = 0x3C, 2 = 0x40, 0.5 = 0x38, 3 = 0x42
    (w[0], w[1], x[0], x[1]) = (0x3C, 0x40, 0x38, 0x42);
    assert_eq!(fp8_dot(FMT_E5M2, &w, &x), 0x40D0_0000);
    // exactness: the smallest subnormals squared are 2^-18 (e4m3) and 2^-32 (e5m2); 448^2 - 448^2 + 2^-18 survives
    let (mut w, mut x) = (vec![0u8; 16], vec![0u8; 16]);
    (w[0], w[1], w[2], x[0], x[1], x[2]) = (0x7E, 0x7E, 0x01, 0x7E, 0xFE, 0x01);
    assert_eq!(fp8_dot(FMT_E4M3, &w, &x), 0x3680_0000, "2^-18, the exact sum, not a float32 accumulation");
    // zero signs: the sum is -0 only if every product is -0
    assert_eq!(fp8_dot(FMT_E4M3, &[0x80, 0x00], &[0x38, 0x38]), 0x0000_0000);
    assert_eq!(fp8_dot(FMT_E4M3, &[0x80, 0x80], &[0x38, 0x38]), 0x8000_0000);
    // specials: a NaN, infinity times zero (e5m2) and opposite infinities are the canonical NaN; one infinity is that infinity
    assert_eq!(fp8_dot(FMT_E4M3, &[0x7F], &[0x38]), 0x7FC0_0000);
    assert_eq!(fp8_dot(FMT_E5M2, &[0x7C], &[0x00]), 0x7FC0_0000);
    assert_eq!(fp8_dot(FMT_E5M2, &[0x7C, 0x7C], &[0x3C, 0xBC]), 0x7FC0_0000);
    assert_eq!(fp8_dot(FMT_E5M2, &[0x7C], &[0xBC]), 0xFF80_0000);
}

#[test]
fn every_fp8_pair_and_dense_signed_rows_match_the_exact_grid() {
    use tsp_numerics::fp::{FMT_E4M3, FMT_E5M2};
    use tsp_numerics::mxm::{dsum_round, fp8_dot, fp8_exact};
    for fmt in [FMT_E4M3, FMT_E5M2] {
        for w in 0..=255u8 {
            for x in 0..=255u8 {
                let (w, x) = ([w], [x]);
                assert_eq!(fp8_dot(fmt, &w, &x), dsum_round(&fp8_exact(fmt, &w, &x)));
            }
        }
        let mut seed = 781237331u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 24) as u8) & 0xbf
        };
        for n in [0, 1, 16, 160, 320, 512] {
            for _ in 0..30 {
                let w: Vec<_> = (0..n).map(|_| next()).collect();
                let x: Vec<_> = (0..n).map(|_| next()).collect();
                assert_eq!(fp8_dot(fmt, &w, &x), dsum_round(&fp8_exact(fmt, &w, &x)));
            }
        }
    }
}

#[test]
fn prepared_fp8_operands_match_the_exact_grid() {
    use tsp_numerics::fp::{FMT_E4M3, FMT_E5M2};
    use tsp_numerics::mxm::{dsum_round, fp8_exact, Fp8Act};
    for fmt in [FMT_E4M3, FMT_E5M2] {
        for w in 0..=255u8 {
            for x in 0..=255u8 {
                let (w, x) = ([w], [x]);
                assert_eq!(Fp8Act::new(fmt, &x).dot(&w), dsum_round(&fp8_exact(fmt, &w, &x)));
            }
        }
        let mut seed = 781237331u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 24) as u8) & 0xbf
        };
        for n in [0, 1, 16, 160, 320, 512] {
            for _ in 0..30 {
                let w: Vec<_> = (0..n).map(|_| next()).collect();
                let x: Vec<_> = (0..n).map(|_| next()).collect();
                assert_eq!(Fp8Act::new(fmt, &x).dot(&w), dsum_round(&fp8_exact(fmt, &w, &x)));
            }
        }
    }
}

#[test]
fn e4m3_vector_grid_covers_every_operand_pair_and_lane() {
    use tsp_numerics::fp::FMT_E4M3;
    use tsp_numerics::mxm::{dsum_round, fp8_exact, Fp8Act};

    for w in 0..=255u8 {
        for x in 0..=255u8 {
            let mut weights = [0; 17];
            let mut activation = [0; 17];
            let lane = usize::from(w ^ x) % 17;
            weights[lane] = w;
            activation[lane] = x;
            assert_eq!(
                Fp8Act::new(FMT_E4M3, &activation).dot(&weights),
                dsum_round(&fp8_exact(FMT_E4M3, &weights, &activation)),
                "weight {w:#04x}, activation {x:#04x}, lane {lane}",
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn e4m3_vector_grid_preserves_host_saturation_flag() {
    use std::{arch::asm, hint::black_box};
    use tsp_numerics::{fp::FMT_E4M3, mxm::Fp8Act};

    let original: u64;
    unsafe { asm!("mrs {value}, FPSR", value = out(reg) original, options(nomem, nostack)) };
    let qc = 1u64 << 27;
    unsafe { asm!("msr FPSR, {value}", value = in(reg) original & !qc, options(nomem, nostack)) };
    let weight = black_box([0x01; 24]);
    let activation = black_box([0x38; 24]);
    let result = black_box(Fp8Act::new(FMT_E4M3, &activation).dot(&weight));
    let after: u64;
    unsafe {
        asm!("mrs {value}, FPSR", value = out(reg) after, options(nomem, nostack));
        asm!("msr FPSR, {value}", value = in(reg) original, options(nomem, nostack));
    }
    assert_ne!(result, 0);
    assert_eq!(after & qc, 0, "the host FPSR.QC flag was changed");
}

#[test]
fn e5m2_vector_grid_checks_all_byte_pairs_and_high_exponent_fallback() {
    use tsp_numerics::fp::FMT_E5M2;
    use tsp_numerics::mxm::{dsum_round, fp8_exact, Fp8Act};

    for w in 0..=255u8 {
        for x in 0..=255u8 {
            let mut weights = [0; 17];
            let mut activation = [0; 17];
            let lane = usize::from(w ^ x) % 17;
            weights[lane] = w;
            activation[lane] = x;
            assert_eq!(
                Fp8Act::new(FMT_E5M2, &activation).dot(&weights),
                dsum_round(&fp8_exact(FMT_E5M2, &weights, &activation)),
                "weight {w:#04x}, activation {x:#04x}, lane {lane}",
            );
        }
    }
}

#[test]
fn fp8_vector_grid_preserves_sign_in_each_vectorized_lane() {
    use tsp_numerics::fp::{FMT_E4M3, FMT_E5M2};
    use tsp_numerics::mxm::{dsum_round, fp8_exact, Fp8Act};

    for fmt in [FMT_E4M3, FMT_E5M2] {
        for lane in 0..32 {
            let mut weights = [0u8; 32];
            weights[lane] = 0x81;
            let activation = [0x06; 32];
            let result = Fp8Act::new(fmt, &activation).dot(&weights);
            assert_eq!(result, dsum_round(&fp8_exact(fmt, &weights, &activation)));
            assert_eq!(result >> 31, 1);
        }
    }
}

#[test]
fn e5m2_wide_coefficient_buckets_agree_with_the_exact_fraction() {
    use tsp_numerics::fp::FMT_E5M2;
    use tsp_numerics::mxm::{dsum_round, fp8_exact, Fp8Act};

    for width in [1, 7, 16, 17, 96, 319, 320, 511, 512, 513] {
        for w in [0x40, 0x54, 0x7b, 0xf0, 0xfb] {
            for x in [0, 1, 0x38, 0x49, 0x75, 0xf9] {
                let weights = vec![w; width];
                let activation: Vec<_> = (0..width).map(|i| if i % 2 != 0 { x ^ 0x80 } else { x }).collect();
                let expected = dsum_round(&fp8_exact(FMT_E5M2, &weights, &activation));
                assert_eq!(Fp8Act::new(FMT_E5M2, &activation).dot(&weights), expected, "{width} {w} {x}");
            }
        }
    }
}

#[test]
fn e5m2_wide_vector_selects_the_high_bucket_at_exact_boundary() {
    use tsp_numerics::fp::FMT_E5M2;
    use tsp_numerics::mxm::{dsum_round, fp8_exact, Fp8Act};

    for n in [16, 17, 255, 320, 511, 512] {
        for weights in [vec![0x40; n], vec![0xc0; n], vec![0x7b; n]] {
            for value in [0x45, 0x49, 0x4d, 0x7b, 0xfb] {
                let activation = vec![value; n];
                assert_eq!(
                    Fp8Act::new(FMT_E5M2, &activation).dot(&weights),
                    dsum_round(&fp8_exact(FMT_E5M2, &weights, &activation)),
                    "width {n}, activation {value:#04x}, weight {:#04x}",
                    weights[0],
                );
            }
        }
    }
}
