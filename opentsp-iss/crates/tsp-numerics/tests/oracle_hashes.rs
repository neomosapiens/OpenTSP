//! The mpmath oracle's result streams (reference/oracle_hashes.txt, written by
//! tools/gen_reference.py) against this crate: every set is recomputed and its SHA-256 compared.
//! The sets cover all 8-bit operand pairs of every binary operation and comparison (i8, u8, b8),
//! every unary input of the 8- and 16-bit types, all 8- and 16-bit conversion sources against
//! every destination type, all f16 inputs of the four transcendentals, and random plus edge
//! inputs for the 16- and 32-bit types, Reduce groups and f16 dot products.
mod common;

use common::*;
use sha2::{Digest, Sha256};
use tsp_numerics::lanes::{self, LaneOp};
use tsp_numerics::{fp, transcendental, EType};

fn parse_inp(toks: &[&str]) -> Inp {
    match toks[0] {
        "all8" => Inp::All8,
        "all16" => Inp::All16,
        "rand" => Inp::Rand(toks[1].parse().unwrap(), toks[2].parse().unwrap()),
        t => panic!("input set {t}"),
    }
}

fn push(h: &mut Sha256, v: u32, w: u32) {
    h.update(&v.to_le_bytes()[..w as usize]);
}

/// Recompute a set; returns (count, sha256 hex).
fn run_set(kind: &str, args: &[u32], inp: Inp) -> (usize, String) {
    let mut h = Sha256::new();
    let mut count = 0usize;
    match kind {
        "bin" | "cmp" => {
            let t = ty(args[1]);
            for (x, y) in binary_inputs(t, inp) {
                let r = if kind == "bin" { lanes::k_binary(args[0], t, x, y) } else { lanes::k_cmp(args[0], t, x, y) };
                push(&mut h, r, t.width());
                count += 1;
            }
        }
        "un" => {
            let t = ty(args[1]);
            for x in unary_inputs(t, inp) {
                push(&mut h, lanes::apply(LaneOp::Un(args[0], t), x, 0, 0), t.width());
                count += 1;
            }
        }
        "cvt" => {
            let (s, d) = (ty(args[0]), ty(args[1]));
            for x in unary_inputs(s, inp) {
                push(&mut h, lanes::k_convert_any(s, d, x), d.width());
                count += 1;
            }
        }
        "conv" => {
            let (s, d) = conv_types(args[0]);
            for x in unary_inputs(s, inp) {
                push(&mut h, lanes::k_convert(args[0], x), d.width());
                count += 1;
            }
        }
        "i8" => {
            for x in 0..256 {
                push(&mut h, transcendental::i8_transcendental(args[0], x), 1);
                count += 1;
            }
        }
        "tr" => {
            for x in 0..65536 {
                push(&mut h, transcendental::fp_transcendental(args[0], fp::FMT16, x), 2);
                count += 1;
            }
        }
        "reduce" => {
            let t = ty(args[1]);
            let Inp::Rand(seed, n) = inp else { panic!() };
            let mut rng = Rng(seed);
            for _ in 0..n {
                let g: Vec<u32> = (0..16).map(|_| rand_val(&mut rng, t)).collect();
                push(&mut h, lanes::reduce_fold(reduce_of(args[0]), t, &g), t.width());
                count += 1;
            }
        }
        "dot16" => {
            let Inp::Rand(seed, n) = inp else { panic!() };
            let mut rng = Rng(seed);
            for i in 0..n {
                let (ws, xs) = dot_case(&mut rng, i, args[0] as usize);
                push(&mut h, dot_f16_of(&ws, &xs), 4);
                count += 1;
            }
        }
        k => panic!("kind {k}"),
    }
    let digest = h.finalize();
    (count, digest.iter().map(|b| format!("{b:02x}")).collect())
}

#[test]
fn oracle_hashes_match() {
    let text = std::fs::read_to_string(reference_dir().join("oracle_hashes.txt")).unwrap();
    let mut total = 0usize;
    let mut sets = 0usize;
    let mut failures = Vec::new();
    for line in text.lines().filter(|l| l.starts_with("H ")) {
        let f: Vec<&str> = line.split_whitespace().collect();
        // H name count sha kind args inputs...
        let (name, count, sha, kind) = (f[1], f[2].parse::<usize>().unwrap(), f[3], f[4]);
        let args: Vec<u32> = f[5].split(',').map(|a| a.parse().unwrap()).collect();
        let (n, got) = run_set(kind, &args, if f.len() > 6 { parse_inp(&f[6..]) } else { Inp::All8 });
        total += n;
        sets += 1;
        if n != count || got != sha {
            failures.push(format!("{name}: count {n} (want {count}), sha {got} (want {sha})"));
        }
    }
    eprintln!("oracle_hashes: {sets} sets, {total} results compared by SHA-256");
    assert!(sets > 300, "reference/oracle_hashes.txt is missing or truncated ({sets} sets)");
    assert!(failures.is_empty(), "{} sets differ from the mpmath oracle:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn types_are_the_thirteen_of_the_model() {
    assert_eq!(EType::ALL.len(), 13);
    for (i, t) in EType::ALL.iter().enumerate() {
        assert_eq!(t.code() as usize, i);
        assert_eq!(EType::from_code(i as u8), Some(*t));
    }
    assert_eq!(EType::from_code(13), None);
}
