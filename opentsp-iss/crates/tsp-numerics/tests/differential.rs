//! Differential tests against the Sail model and the mpmath oracle.
//!
//! * `sample_matches_reference` (always on): reference/sample.txt, results of the mpmath oracle
//!   that equal the Sail model's (checked by tools/gen_reference.py sample --sail).
//! * `sail_dumps_match` (ignored; `just iss-numerics-sail` runs it with --ignored after generating the dumps): every `*.txt` in `$TSP_NUMERICS_SAIL_DUMP`
//!   (default `<workspace>/target/tsp-numerics/sail`), written by tools/run_differential.sh from
//!   the Sail model's own functions through tools/sail_dump.sh. Fails when the directory has no dump.
//! * `f32_matches_screening_rechecks` (ignored, long; `just iss-transcendental`): the jobs of the exhaustive f32
//!   check (opentsp-arch-spec/tools/check_transcendental_f32.sh --work DIR) in `$TSP_F32_WORK`.
mod common;

use common::*;
use std::path::PathBuf;

#[test]
fn sample_matches_reference() {
    let (n, bad, shown) = compare_file(&reference_dir().join("sample.txt"));
    eprintln!("sample: {n} jobs, {bad} mismatches");
    assert!(n >= 2500, "reference/sample.txt is missing or truncated ({n} jobs)");
    assert_eq!(bad, 0, "mismatches:\n{}", shown.join("\n"));
}

/// The inputs on which the fast path declines are the ones the Ziv fallback decides; their
/// results are mpmath's, confirmed by the Sail model in the exhaustive f32 check (f32 screening).
#[test]
fn f32_hard_inputs_match_reference() {
    use tsp_numerics::{fp, transcendental as tr};
    let path = reference_dir().join("f32_hard.txt");
    let (n, bad, shown) = compare_file(&path);
    eprintln!("f32 hard: {n} inputs, {bad} mismatches");
    assert!(n >= 600, "reference/f32_hard.txt is missing or truncated ({n} inputs)");
    assert_eq!(bad, 0, "mismatches:\n{}", shown.join("\n"));
    // every one of them really takes the fallback
    let text = std::fs::read_to_string(&path).unwrap();
    let mut declined = 0;
    for line in text.lines().filter(|l| l.starts_with("tr ")) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let op: u32 = f[1].parse().unwrap();
        let x = u32::from_str_radix(f[3], 16).unwrap();
        let tr::Pre::Eval(a) = tr::fp_tr_pre(op, fp::FMT32, x) else { panic!("{line}: not an Eval input") };
        assert!(tr::tr_fast(op, fp::FMT32, &a).is_none(), "{line}: the fast path accepted a hard input");
        declined += 1;
    }
    assert_eq!(declined, n);
}

fn dump_dir() -> PathBuf {
    match std::env::var_os("TSP_NUMERICS_SAIL_DUMP") {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../target/tsp-numerics/sail"),
    }
}

#[test]
#[ignore = "needs the Sail dumps of tools/run_differential.sh (`just iss-numerics-sail`), which runs it with --ignored"]
fn sail_dumps_match() {
    let dir = dump_dir();
    let mut files: Vec<PathBuf> = match std::fs::read_dir(&dir) {
        Ok(rd) => {
            rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "txt")).collect()
        }
        Err(_) => Vec::new(),
    };
    files.sort();
    assert!(
        !files.is_empty(),
        "no Sail dumps in {} (generate them with tools/run_differential.sh, or set TSP_NUMERICS_SAIL_DUMP)",
        dir.display()
    );
    let mut failures = Vec::new();
    for f in &files {
        let (n, bad, shown) = compare_file(f);
        eprintln!("sail dump {}: {n} jobs, {bad} mismatches", f.file_name().unwrap().to_string_lossy());
        if bad > 0 {
            failures.push(format!("{}: {bad} of {n}:\n{}", f.display(), shown.join("\n")));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
#[ignore = "long: needs the work directory of check_transcendental_f32.sh (TSP_F32_WORK)"]
fn f32_matches_screening_rechecks() {
    let dir = PathBuf::from(
        std::env::var_os("TSP_F32_WORK").expect("set TSP_F32_WORK to the --work DIR of check_transcendental_f32.sh"),
    );
    let text = std::fs::read_to_string(dir.join("jobs_eval.txt"))
        .or_else(|_| std::fs::read_to_string(dir.join("jobs.txt")))
        .unwrap();
    let mut counts = [0usize; 4];
    let mut bad = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() != 4 || f[0] != "E" {
            continue;
        }
        let op: u32 = f[1].parse().unwrap();
        let x = u32::from_str_radix(f[2], 16).unwrap();
        let want = u32::from_str_radix(f[3], 16).unwrap();
        counts[(op - 21) as usize] += 1;
        let got = tsp_numerics::transcendental::fp_tr_value(op, tsp_numerics::fp::FMT32, x);
        if got != Some(want) {
            bad.push(format!("op {op} x={x:08x}: rust {got:08x?}, f32 screening {want:08x}"));
        }
    }
    eprintln!(
        "f32 vs f32 screening: TanH {} Exp {} RSqrt {} Log {} inputs, {} mismatches",
        counts[0],
        counts[1],
        counts[2],
        counts[3],
        bad.len()
    );
    assert!(counts.iter().all(|&c| c > 0), "no jobs read from {}", dir.display());
    assert!(bad.is_empty(), "{}", bad.iter().take(20).cloned().collect::<Vec<_>>().join("\n"));
}

/// The fast path against the Ziv port alone on a stride of f32 inputs (all four functions).
/// `TSP_F32_STRIDE` (default 16411, a prime: about 262 thousand inputs per function).
#[test]
#[ignore = "long: the Ziv port runs on every input (about a minute at the default stride)"]
fn f32_fast_path_equals_ziv_on_a_stride() {
    use tsp_numerics::{fp::FMT32, transcendental as tr, ziv};
    let stride: u64 = std::env::var("TSP_F32_STRIDE").ok().map(|s| s.parse().unwrap()).unwrap_or(16411);
    let mut n = 0u64;
    let mut declined = 0u64;
    for op in [tr::TANH, tr::EXP, tr::RSQRT, tr::LOG] {
        let mut x = 12345u64;
        while x < (1u64 << 32) {
            let x32 = x as u32;
            let fast = tr::fp_tr_value(op, FMT32, x32);
            let slow = ziv::fp_tr_value(op, FMT32, x32);
            assert_eq!(fast, slow, "op {op} x={x32:08x}");
            if let tr::Pre::Eval(a) = tr::fp_tr_pre(op, FMT32, x32) {
                declined += tr::tr_fast(op, FMT32, &a).is_none() as u64;
            }
            n += 1;
            x += stride;
        }
    }
    eprintln!("f32 stride {stride}: {n} inputs, fast path = Ziv on all, {declined} declined");
}
