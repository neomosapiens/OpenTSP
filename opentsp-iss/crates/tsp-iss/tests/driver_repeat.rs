//! Tests of `driver/repeat`, moved from the module's in-source test block (they use the crate's public API).
use ots_isa::Insn;
use tsp_iss::driver::repeat::*;

#[test]
fn what_a_repeat_can_repeat() {
    assert!(
        !repeatable(&Insn::NOP { n: 1 })
            && !repeatable(&Insn::Repeat { n: 1, d: 1 })
            && !repeatable(&Insn::Step { a: 1 })
    );
    assert!(
        repeatable(&Insn::SetVL { n: 1 })
            && repeatable(&Insn::Sync)
            && repeatable(&Insn::Read { addr: 0, dst: 0, thread: 0 })
    );
    assert!(
        clears_repeat(&Insn::Sync)
            && clears_repeat(&Insn::Notify)
            && clears_repeat(&Insn::Deskew)
            && clears_repeat(&Insn::RuntimeDeskew { t: 1 })
    );
    assert!(!clears_repeat(&Insn::SetVL { n: 1 }));
}

mod common;

/// The harness input of queue 0: `IfetchM 10; NOP 1; <second>; <tail>` with the fetched words 10 and 11 holding only padding.
fn fetch_input(second: &[&str], tail: &[&str]) -> String {
    let pad = "ff ".repeat(320);
    let mut units = vec!["01c0000a", "00000001"];
    units.extend_from_slice(second);
    units.extend_from_slice(tail);
    format!("S 0 a {pad}\nS 0 b {pad}\nQ 0 {}\n", units.join(" "))
}

fn run(input: &str) -> Vec<String> {
    let out = common::run_bin(common::iss(), &[common::config("ref").to_str().unwrap(), "--trace"], input.as_bytes());
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .filter(|l| l.starts_with("T srf") || l.starts_with("T diag"))
        .map(str::to_string)
        .collect()
}

// model: postlude/tick.sail reserve_local_fetch, repeat_issue: `ifetchm a; nop 1; repeat 1, 1` is `ifetchm a; nop 1; ifetchm a`.
#[test]
fn repeated_ifetchm_reserves_its_second_beat_like_an_explicit_one() {
    let read: [&str; 2] = ["10000020", "1c000000"];
    let write: [&str; 2] = ["1040000b", "14000000"];
    let plain_read = run(&fetch_input(&["01c0000a"], &read));
    let repeat_read = run(&fetch_input(&["00c00011"], &read));
    // the Read dispatches at 4 (the R path is held through the second beat at 3) and commits at 4 + d_func 5
    assert_eq!(plain_read, ["T srf 9 E:7 gap 18 ".to_string() + &"00".repeat(320)]);
    assert_eq!(repeat_read, plain_read);
    // a Write of the bank of word a + 1 in the second beat's cycle is OTS-E-MEM-001, not a port overlap
    let plain_write = run(&fetch_input(&["01c0000a"], &write));
    let repeat_write = run(&fetch_input(&["00c00011"], &write));
    assert_eq!(plain_write, ["T diag 3 queue 0 OTS-E-MEM-001"]);
    assert_eq!(repeat_write, plain_write);
}
