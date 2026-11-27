//! Tests of `harness`, moved from the module's in-source test block (they use the crate's public API).
use std::ffi::OsString;
use tsp_iss::harness::*;

fn run(args: &[&str], input: &str) -> (i32, String, String) {
    let a: Vec<OsString> = args.iter().map(OsString::from).collect();
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let code = main_with(&a, &mut input.as_bytes(), &mut out, &mut err);
    (code, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
}

fn cfg(name: &str) -> String {
    format!("{}/../../../opentsp-arch-spec/config/{name}.json", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn empty_run_in_process() {
    let (code, out, err) = run(&[&cfg("mini")], "S 1 1 7\nD 1 1\n");
    assert_eq!((code, err.as_str()), (0, ""));
    assert!(out.starts_with("E - -1 0\nW 1 1 07 00"), "{out}");
}

#[test]
fn guarded_turns_a_panic_into_exit_4() {
    std::panic::set_hook(Box::new(|_| {}));
    assert_eq!(guarded(|| 0), 0);
    assert_eq!(guarded(|| panic!("model fault")), 4);
    let _ = std::panic::take_hook();
}

#[test]
fn order_of_checks() {
    // a bad configuration is reported before a bad link option combination and before bad input
    let (code, _, err) = run(&["/nonexistent.json", "--link-latency", "1", "--link-latency-seed", "1"], "zzz\n");
    assert_eq!(code, 2);
    assert!(err.contains("config /nonexistent.json: cannot open: No such file or directory"), "{err}");
    // then the link options, before the input
    let (code, _, err) = run(&[&cfg("mini"), "--link-latency", "1", "--link-latency-seed", "1"], "zzz\n");
    assert!(code == 2 && err.contains("usage"), "{err}");
    // then the input, before the range of --link-latency
    let (code, _, err) = run(&[&cfg("mini"), "--link-latency", "1"], "zzz\n");
    assert!(code == 2 && err.contains("input line 1"), "{err}");
    let (code, _, err) = run(&[&cfg("mini"), "--link-latency", "1"], "");
    assert!(code == 2 && err.contains("--link-latency must be in [209, 228]"), "{err}");
}
