//! Parity of the ISS with `tsp_sim` on two rare paths.
//!
//! Event order: when one ABC has an operand ECC event and an MXM sum-check event, the model records the
//! operand event first. Stop: `--on-irq stop` ends the run before the end-of-run check of unused
//! `F link` lines, so a stopped run exits 5 with its results.
mod common;
use common::*;

/// `mini` with a corrected-ECC weight plane read (`F sram`) and a sum-check fault (`F mxm`) on the same ABC.
const ECC_MATRIX_FAULT_PROGRAM: &str = r#"S 2c 3 00 00 00 00 00 00 00 00 00 00 80 80 00 00 00 00
S 30 4 3c bc 47 3c 3c 00 bc 3e 42 47 c6 c6 b4 00 c0 47
S 34 5 00 00 00 00 00 00 00 80 00 00 00 00 00 80 00 80
S 38 6 bc 42 c0 c0 c0 38 c0 c6 38 bc bc 3e 00 c6 3e c6
S 3c 7 00 00 00 00 00 00 00 00 00 00 00 00 00 00 80 00
S 40 8 3e c0 b4 bc 38 00 00 b4 3e 30 c0 b4 42 bc c6 47
S 44 9 00 00 00 00 80 00 00 00 80 00 00 00 80 00 80 00
S 48 a 3c bc 3e 00 c6 3e b4 42 c6 3e 42 00 c6 00 c6 38
S 4c b 00 00 00 80 00 00 00 00 00 00 00 00 00 00 00 00
S 50 c 30 00 38 c6 3e bc c0 42 38 3e b4 42 38 42 42 30
S 54 d 80 00 00 00 80 00 00 00 00 80 00 00 00 00 00 00
S 2d e c6 30 3e 30 c6 47 b4 38 bc c6 c0 bc 30 00 42 3e
S 31 f 00 00 80 00 00 00 00 00 00 00 00 00 00 00 00 00
S 35 10 c0 00 c6 bc 3c b4 42 3e b4 bc 3c 00 3e 38 47 b4
S 39 11 00 00 00 00 00 00 80 80 00 00 00 00 00 00 00 00
S 3d 12 47 00 c0 c0 3e 3e c6 c6 bc 3e 3c 3c bc c0 42 3e
S 3c 5 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
S 3d 6 40 40 00 38 bc 40 40 40 c2 bc 34 38 00 3f 34 c2
Q 0 00000102 10400009 20000000
Q 4 00000101 10400009 24000000
Q 8 00000100 10400009 28000000
Q c 000000ff 10400009 2c000000
Q 2c 000000aa 10000003 80000000
Q 2d 000000aa 1000000e ac000000
Q 30 000000a9 10000004 84000000
Q 31 000000a9 1000000f b0000000
Q 34 000000a8 10000005 88000000
Q 35 000000a8 10000010 b4000000
Q 38 000000a7 10000006 8c000000
Q 39 000000a7 10000011 b8000000
Q 3c 000000a6 10000007 90000000 00000001 10000005 e0000000
Q 3d 000000a6 10000012 bc000000 00000001 10000006 e4000000
Q 40 000000a5 10000008 94000000
Q 44 000000a4 10000009 98000000
Q 48 000000a3 1000000a 9c000000
Q 4c 000000a2 1000000b a0000000
Q 50 000000a1 1000000c a4000000
Q 54 000000a0 1000000d a8000000
Q 68 000000c8 30582000
Q 69 000000ca 309e0000
Q 6a 000000de 30f21000
D 0 9
D 4 9
D 8 9
D c 9
F mxm ca 0 5 28
R 10
R 11
R 12
R 13
R 14
R 20
R 21
R 22
R f0
R f1
R 30
R 32
F sram 4d 3c 5 0 88 p
"#;

/// The expected result words come from `tsp_sim` (the log at 0x20 and `ERR_FIRST` at 0xf0).
const D2_EXPECT: [&str; 5] =
    ["X 020 6910000000000ca", "X 021 af1", "X 022 6950000000000ca", "X 0f0 6910000000000ca", "X 0f1 af1"];

fn x_lines(out: &str, wanted: &[&str]) -> Vec<String> {
    let addrs: Vec<&str> = wanted.iter().map(|w| w.split(' ').nth(1).unwrap()).collect();
    out.lines()
        .filter(|l| l.starts_with("X ") && addrs.contains(&l.split(' ').nth(1).unwrap_or("")))
        .map(String::from)
        .collect()
}

#[test]
fn operand_matrix_operand_event_precedes_sum_check_event() {
    let cfg = config("mini");
    let cfg = cfg.to_str().unwrap();
    let o = run_iss(&[cfg], ECC_MATRIX_FAULT_PROGRAM);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o.stderr));
    assert_eq!(x_lines(&text(&o.stdout), &D2_EXPECT), D2_EXPECT, "ERR_FIRST and log words");
    // and exactly what the reference prints
    let s = run_sim(&[cfg], ECC_MATRIX_FAULT_PROGRAM);
    assert_eq!(s.status.code(), o.status.code());
    assert_eq!(text(&s.stdout), text(&o.stdout));
}

#[test]
fn operand_matrix_trace_lists_operand_event_first() {
    let cfg = config("mini");
    let cfg = cfg.to_str().unwrap();
    let o = run_iss(&[cfg, "--trace"], ECC_MATRIX_FAULT_PROGRAM);
    let s = run_sim(&[cfg, "--trace"], ECC_MATRIX_FAULT_PROGRAM);
    let ecc = |b: &[u8]| text(b).lines().filter(|l| l.starts_with("T ecc")).map(String::from).collect::<Vec<_>>();
    assert!(ecc(&o.stdout).len() >= 2, "{:?}", ecc(&o.stdout));
    assert_eq!(ecc(&o.stdout), ecc(&s.stdout));
}

/// Operand faults x sum-check faults on one ABC: the table of fault lines replaces the
/// two `F` lines of `ECC_MATRIX_FAULT_PROGRAM`; the ISS must equal `tsp_sim` in exit code and stdout in every cell.
#[test]
fn operand_matrix_operand_by_sum_check_table_matches_model() {
    const OPERAND: [(&str, &str); 3] = [
        ("corr", "F sram 4d 3c 5 0 88 p\n"),
        ("unc", "F sram 4d 3c 5 0 88 p\nF sram 4d 3c 5 0 3 p\n"),
        ("spare", "F sram 4d 3c 5 0 89 p\n"),
    ];
    const SUM: [(&str, &str); 2] = [("corr", "F mxm ca 0 5 28\n"), ("unc", "F mxm ca 0 5 28\nF mxm ca 0 5 29\n")];
    let base: String = ECC_MATRIX_FAULT_PROGRAM
        .lines()
        .filter(|l| !l.starts_with("F sram 4d") && !l.starts_with("F mxm ca"))
        .map(|l| format!("{l}\n"))
        .collect();
    let cfg = config("mini");
    let cfg = cfg.to_str().unwrap();
    let mut bad = Vec::new();
    for (on, of) in OPERAND {
        for (sn, sf) in SUM {
            let input = format!("{base}{sf}{of}");
            let o = run_iss(&[cfg, "--trace"], &input);
            let s = run_sim(&[cfg, "--trace"], &input);
            if (o.status.code(), text(&o.stdout)) != (s.status.code(), text(&s.stdout)) {
                bad.push(format!(
                    "operand {on} x sum {sn}: iss exit {:?}, tsp_sim exit {:?}",
                    o.status.code(),
                    s.status.code()
                ));
            }
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

/// A run stopped by `--on-irq stop` never reaches the end-of-run check of `F link` lines.
const IRQ_STOP_PROGRAM: &str = r#"Q 4 00000021 10000010 00000000
Q 5 00000021 10000000 10000000
Q 2c 00000030 10400009 20000000
Q 58 00000028 20020010 00000000
S 4 10 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
S 5 0 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
D 2c 9
F sram 0 4 10 0 3
F sram 0 4 10 0 4
F link 64 0 1 corr
"#;

fn run_irq_policy(policy: &str) -> (std::process::Output, std::process::Output) {
    let cfg = config("ref");
    let cfg = cfg.to_str().unwrap();
    (run_iss(&[cfg, "--on-irq", policy], IRQ_STOP_PROGRAM), run_sim(&[cfg, "--on-irq", policy], IRQ_STOP_PROGRAM))
}

#[test]
fn irq_stop_stop_ends_before_unused_link_fault_check() {
    let (o, s) = run_irq_policy("stop");
    assert_eq!(s.status.code(), Some(5), "reference");
    assert_eq!(o.status.code(), Some(5), "{}", text(&o.stderr));
    assert_eq!(text(&o.stdout), text(&s.stdout));
    assert!(text(&o.stdout).contains("E OTS-E-ECC-001"), "{}", text(&o.stdout));
}

#[test]
fn irq_stop_continue_and_replay_still_reject_unused_link_fault() {
    for p in ["continue", "replay"] {
        let (o, s) = run_irq_policy(p);
        assert_eq!(s.status.code(), Some(2), "{p} reference");
        assert_eq!(o.status.code(), Some(2), "{p}");
        assert!(
            text(&o.stderr).contains("F link: no vector is sent on link 0 in cycle 64"),
            "{p}: {}",
            text(&o.stderr)
        );
        assert!(o.stdout.is_empty(), "{p}");
    }
}

// ---- cycles over the whole unsigned 64-bit range ---------------------------------------------------------------------

/// A one-unit add of two preloaded words (the program of the reproducers), without its `H start` line.
const D1_BODY: &str = r#"Q 4 00000021 10000010 00000000
Q 5 00000021 10000000 10000000
Q 2c 00000030 10400009 20000000
Q 58 00000028 20020010 00000000
S 4 10 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
S 5 0 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
D 2c 9
"#;

fn from_cycle(start: u64) -> String {
    format!("H start {start:x}\n{D1_BODY}")
}

/// The model computes with unbounded integers: a start or a deadline at or above 2^62, 2^63 or near 2^64 gives the result
/// `tsp_sim` gives (exit code, standard output, trace), where the ISS once wrapped its cycle counter to a negative number
/// at 2^63 and ran forever at 2^63 and beyond.
#[test]
fn cycles_at_and_above_two_to_the_62_and_63_match_the_model() {
    let cfg = config("ref");
    let cfg = cfg.to_str().unwrap();
    let starts: [u64; 9] = [
        1 << 62,
        (1 << 62) + 7,
        (1 << 63) - 48,
        (1 << 63) - 1,
        1 << 63,
        (1 << 63) + 1,
        (1 << 63) + (1 << 62),
        u64::MAX - 99,
        u64::MAX - (1 << 27),
    ];
    let mut bad = Vec::new();
    for start in starts {
        let input = from_cycle(start);
        for max in [u64::MAX, 1 << 63, 1 << 62] {
            let max = max.to_string();
            let args = [cfg, "--max-cycles", max.as_str(), "--trace"];
            let o = run_iss(&args, &input);
            let s = run_sim(&args, &input);
            if (o.status.code(), text(&o.stdout)) != (s.status.code(), text(&s.stdout)) {
                bad.push(format!(
                    "start {start:#x} --max-cycles {max}: iss exit {:?} {}, tsp_sim exit {:?}",
                    o.status.code(),
                    text(&o.stderr).trim(),
                    s.status.code()
                ));
            }
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

/// The three reproducers of the audit, by name: a start 48 cycles below 2^63, at 2^63, and 100 below 2^64.
#[test]
fn the_cycle_range_reproducers_print_the_reference_results() {
    let cfg = config("ref");
    let cfg = cfg.to_str().unwrap();
    for (name, start, last_srf) in [
        ("D1_min", 0x7fff_ffff_ffff_ffd0u64, "T sram 9223372036854775808 slice 44 addr 9"),
        ("D1b", 0x8000_0000_0000_0000, "T sram 9223372036854775856 slice 44 addr 9"),
        ("D1c", 0xffff_ffff_ffff_ff9c, "T sram 18446744073709551564 slice 44 addr 9"),
    ] {
        let o = run_iss(&[cfg, "--max-cycles", "18446744073709551615", "--trace"], &from_cycle(start));
        let out = text(&o.stdout);
        assert_eq!(o.status.code(), Some(0), "{name}: {}", text(&o.stderr));
        assert!(out.contains(last_srf), "{name}: {out}");
        assert!(out.contains("E - -1 0\nW 2c 9 07 07"), "{name}: the add of 3 and 4 is 7: {out}");
    }
}
