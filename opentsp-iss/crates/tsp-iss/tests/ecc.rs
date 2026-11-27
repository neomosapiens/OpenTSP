//! SECDED ECC on SRAM words, stream registers and data paths (`core/ecc.sail`, `microarchitecture.md` §15).
//! The code tables are checked against the properties of the code; the programs and the `F` lines run on the reference
//! `tsp_sim` and on the ISS and must give the same bytes.
mod common;
use common::*;
use tsp_iss::ecc::{classify, data_index, flip_delta, pos, Class, POISON_SL};

// model: unit_tests/test_ecc.sail test_ecc_positions, test_ecc_single_flips, test_ecc_double_flips
// model: core/ecc.sail ecc_pos, ecc_flip_delta, ecc_classify (the equivalence test_ecc.sail proves against the encoder)
#[test]
fn every_single_flip_is_corrected_and_every_double_flip_is_uncorrectable() {
    // data bit j: position P(j) is not a power of two and names j back
    for j in 0..128 {
        let p = pos(j);
        assert!(p >= 3 && !p.is_power_of_two() && p <= 136, "bit {j} at {p}");
        assert_eq!(data_index(p), j);
        assert_eq!(classify(flip_delta(j)), (Class::Corrected, p), "data bit {j}");
    }
    // check bits c[0..7] name their own position, c[8] position 0; the spare bit is no event
    for i in 0..8 {
        assert_eq!(classify(flip_delta(128 + i)), (Class::Corrected, 1 << i), "check bit {i}");
    }
    assert_eq!(classify(flip_delta(136)), (Class::Corrected, 0));
    assert_eq!(classify(flip_delta(137)).0, Class::Ok);
    // two flips of the 137 code bits: never corrected as a single, never silent
    for a in 0..137 {
        for b in a + 1..137 {
            let e = flip_delta(a) ^ flip_delta(b);
            assert_eq!(classify(e).0, Class::Uncorrectable, "bits {a} and {b}");
        }
    }
}

// model: unit_tests/test_ecc.sail test_ecc_poison
#[test]
fn the_poison_overlay_is_uncorrectable_and_sets_the_spare_bit() {
    assert_eq!(classify(POISON_SL).0, Class::Uncorrectable);
    assert_eq!(POISON_SL >> 9 & 1, 0, "poison leaves the spare bit to the error report");
}

fn s_line(slice: u32, addr: u32, hex: &str) -> String {
    let bytes: Vec<String> = (0..hex.len() / 2).map(|i| hex[2 * i..2 * i + 2].to_string()).collect();
    format!("S {slice:x} {addr:x} {}\n", bytes.join(" "))
}

/// The conformance vectors' program `ecc-002` on the mini profile: Read slice 0 address 1 onto E:0, Read slice 1 address
/// 2 onto E:4, a VXM Add (E:0, E:4 into E:8), and a Write of E:8 to slice 44 address 9; `faults` are the F lines.
fn add_program(faults: &str) -> String {
    let mut s = String::new();
    s += "Q 0 00000036 10000001 00000000\nQ 1 00000036 10000002 10000000\nQ 2c 00000044 10400009 20000000\n";
    s += "Q 58 0000003c 20020010 00000000\n";
    s += &s_line(0, 1, "05121f2c394653606d7a8794a1aebbc8");
    s += &s_line(1, 2, "5b626970777e858c939aa1a8afb6bdc4");
    s += faults;
    s += "D 2c 9\n";
    s
}

fn both(args: &[&str], input: &str) -> String {
    let ours = run_iss(args, input);
    let theirs = run_sim(args, input);
    assert_eq!(ours.status.code(), theirs.status.code(), "exit code, input {input:?}\nours: {}", text(&ours.stderr));
    assert_eq!(text(&ours.stdout), text(&theirs.stdout), "stdout, input {input:?}");
    assert_eq!(text(&ours.stderr), norm_stderr(&text(&theirs.stderr)), "stderr, input {input:?}");
    text(&ours.stdout)
}

fn mini() -> String {
    config("mini").to_str().unwrap().to_string()
}

// model: unit_tests/test_ecc.sail test_ecc_check_value
// model: core/ecc.sail ecc_check_value (the corrected operand is what the Add sees)
#[test]
fn a_single_flip_in_an_sram_word_is_corrected_for_the_reader() {
    let c = mini();
    let clean = both(&[&c, "--trace"], &add_program(""));
    let out = both(&[&c, "--trace"], &add_program("F sram 0 0 1 0 5\n"));
    assert!(out.contains("T ecc 62 queue 88 corr E:0 sl 0 pos 10\n"), "{out}");
    assert!(out.contains("E - -1 0\n"), "no diagnostic: {out}");
    let word = |s: &str| s.lines().find(|l| l.starts_with("W 2c 9 ")).unwrap().to_string();
    assert_eq!(word(&out), word(&clean), "the corrected value reaches the result");
}

// model: core/ecc.sail ecc_poison_sl; postlude/tick.sail execute_element (OTS-E-ECC-001)
#[test]
fn a_double_flip_is_reported_once_and_poisons_the_result() {
    let c = mini();
    let clean = both(&[&c, "--trace"], &add_program(""));
    let out = both(&[&c, "--trace"], &add_program("F sram 0 0 1 0 5\nF sram 0 0 1 0 4d\n"));
    assert!(out.contains("T ecc 62 queue 88 unc E:0 sl 0 pos 95\n"), "{out}");
    assert!(out.contains("T diag 62 queue 88 OTS-E-ECC-001\n") && out.contains("E OTS-E-ECC-001 62 1\n"), "{out}");
    assert_ne!(out.lines().find(|l| l.starts_with("W 2c 9 ")), clean.lines().find(|l| l.starts_with("W 2c 9 ")));
}

// model: c_emulator/tsp_sim.c read_fault_line
#[test]
fn the_f_lines_are_checked_like_the_reference_checks_them() {
    let c = mini();
    for bad in [
        "F sram 0 0 1 0 8a\n",
        "F sram 0 0 1 1 5\n",
        "F sram 0 0 2000 0 5\n",
        "F sram 0 58 1 0 5\n",
        "F srf 5 x 0 0 0 5\n",
        "F srf 5 e 20 0 0 5\n",
        "F srf 5 e 0 2c 0 5\n",
        "F sram 0 0 1 0 5 q\n",
        "F sram 0 0 1 0 5 t extra\n",
        "F sram 0 0 1 0 5\nF sram 0 0 1 0 5 p\n",
        "F foo 1\n",
        "F sram 1 0 1 0\n",
        "F iq 0 91 0 5\n",
        "F iq 0 0 1000000 5\n",
        "F iq 0 0 0 8a\n",
        "F iq 0 0 0\n",
        "F iq 0 0 0 5 q\n",
        "F iq 0 0 0 5\nF iq 0 0 0 5 p\n",
    ] {
        let ours = run_iss(&[&c], bad);
        let theirs = run_sim(&[&c], bad);
        assert_eq!(ours.status.code(), theirs.status.code(), "{bad:?}");
        assert_eq!(text(&ours.stderr), norm_stderr(&text(&theirs.stderr)), "{bad:?}");
        assert_ne!(ours.status.code(), Some(0), "{bad:?} is an input error");
    }
    // the same bit in different cycles, and an srf flip of an empty register, are fine
    both(&[&c, "--trace"], &add_program("F sram 0 0 1 0 5\nF sram 1 0 1 0 5\nF srf 3 e 0 5 0 5 t\n"));
}
