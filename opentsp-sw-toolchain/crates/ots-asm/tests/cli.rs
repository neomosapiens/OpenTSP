//! The `ots-asm` binary: `insn`, `word`, exit statuses and diagnostics as `tsp_as.py` prints them.
use std::process::{Command, Output};

/// Run the binary. Another `cargo build` or `cargo test` in the shared target directory (the
/// workspace is built by several agents at once) relinks `target/debug/ots-asm`, and between the
/// unlink and the link the path does not exist: that surfaced once as a failed `cargo test
/// --workspace` ("No such file or directory", a flaky test). A missing binary is therefore retried
/// for a few seconds before it is an error.
fn run(args: &[&str]) -> Output {
    let mut waited = 0;
    loop {
        match Command::new(env!("CARGO_BIN_EXE_ots-asm")).args(args).output() {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && waited < 100 => {
                waited += 1;
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            r => return r.unwrap(),
        }
    }
}

fn text(o: &Output) -> (i32, String, String) {
    (
        o.status.code().unwrap(),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

#[test]
fn insn_prints_width_and_lower_case_hex() {
    let (rc, out, err) = text(&run(&["insn", "nop 0xfcb3"]));
    assert_eq!((rc, out.as_str(), err.as_str()), (0, "32 fcb3\n", ""));
    let (rc, out, _) = text(&run(&["insn", "--profile", "mini", "read 0x0, E:0, 0x0"]));
    assert_eq!(rc, 0);
    assert!(out.starts_with("64 "), "{out}");
}

#[test]
fn insn_joins_several_arguments_with_spaces() {
    let a = text(&run(&["insn", "add.i8 E:8, E:0, E:4"]));
    let b = text(&run(&["insn", "add.i8", "E:8,", "E:0,", "E:4"]));
    assert_eq!(a.0, 0);
    assert_eq!(a, b);
    // no argument: the empty text, no template matches
    let (rc, _, err) = text(&run(&["insn"]));
    assert_eq!(rc, 1);
    assert!(err.starts_with("insn: OTS-E-DEC-003: "), "{err}");
}

#[test]
fn a_rejected_text_is_exit_1_with_where_and_code_on_stderr() {
    let (rc, out, err) = text(&run(&["insn", "nop 0x0"]));
    assert_eq!((rc, out.as_str()), (1, ""));
    assert!(err.starts_with("insn: OTS-E-OPR-001: "), "{err}");
    let (rc, _, err) = text(&run(&["insn", "nop\t0x1"]));
    assert_eq!(rc, 1);
    assert!(err.starts_with("insn: OTS-E-ASM-001: "), "{err}");
}

#[test]
fn an_unknown_profile_is_asm_003_before_the_text_is_read() {
    for p in ["nosuch", "REF", "re f", "../ref"] {
        let (rc, _, err) = text(&run(&["insn", "--profile", p, "bogus"]));
        assert_eq!(rc, 1, "{p}");
        assert!(err.starts_with("insn: OTS-E-ASM-003: "), "{p}: {err}");
        // `word` too, before its operands are read
        let (rc, out, err) = text(&run(&["word", "--profile", p, "x", "y"]));
        assert_eq!((rc, out.as_str()), (1, ""), "{p}");
        assert!(err.starts_with("word: OTS-E-ASM-003: "), "{p}: {err}");
    }
}

#[test]
fn config_overrides_the_profile_file() {
    let cfg = format!("{}/../../../opentsp-arch-spec/config/mini.json", env!("CARGO_MANIFEST_DIR"));
    // the shift amount bound is 0xf with the mini configuration, whatever name is given
    let (rc, _, _) = text(&run(&["insn", "--profile", "ref", "--config", &cfg, "shift E:0, up, 0x10, E:1"]));
    assert_eq!(rc, 1);
    let (rc, _, _) = text(&run(&["insn", "--profile", "ref", "shift E:0, up, 0x10, E:1"]));
    assert_eq!(rc, 0);
    let (rc, _, err) = text(&run(&["insn", "--profile", "ref", "--config", "/nonexistent.json", "sync"]));
    assert_eq!(rc, 1);
    assert!(err.starts_with("insn: OTS-E-ASM-003: "), "{err}");
}

#[test]
fn word_prints_text_or_illegal() {
    assert_eq!(text(&run(&["word", "32", "40000"])), (0, "illegal32 0x40000\n".into(), String::new())); // opcode 0 with n = 4 << 16: reserved bits
    assert_eq!(text(&run(&["word", "32", "fcb3"])).1, "nop 0xfcb3\n");
    assert_eq!(text(&run(&["word", "32", "0x400000"])).1, "sync\n");
    assert_eq!(text(&run(&["word", "32", "ffffffff"])).1, "illegal32 0xffffffff\n");
    assert_eq!(text(&run(&["word", "64", "ffffffffffffffff"])).1, "illegal64 0xffffffffffffffff\n");
    assert_eq!(text(&run(&["word", "032", "1_0"])).1, "nop 0x10\n");
}

#[test]
fn a_word_of_another_width_a_wider_value_or_a_negative_value_is_asm_001_exit_1() {
    // as `tsp_as.py word`: no such width, a value above the width and a negative value are refused, not printed as illegal
    for args in [
        &["word", "16", "1"][..],
        &["word", "0", "0"],
        &["word", "48", "40"],
        &["word", "32", "100400000"],
        &["word", "32", "-1"],
        &["word", "64", "-1"],
        &["word", "64", "10000000000000000"],
        &["word", "64", "fffffffffffffffffffff"],
    ] {
        let (rc, out, err) = text(&run(args));
        assert_eq!((rc, out.as_str()), (1, ""), "{args:?}");
        assert!(err.starts_with("word: OTS-E-ASM-001: "), "{args:?}: {err}");
    }
    // the widest values of each width are still words
    assert_eq!(text(&run(&["word", "32", "ffffffff"])).0, 0);
    assert_eq!(text(&run(&["word", "64", "ffffffffffffffff"])).0, 0);
}

#[test]
fn word_of_what_insn_prints_round_trips() {
    for t in [
        "read 0x10, E:0, 0x0",
        "add.i8 E:8, E:0, E:4",
        "lw.i8 E:16",
        "acc.i32 E:0, replace, 1",
        "sync",
        "shift E:0, up, 0x1, E:1",
    ] {
        let (rc, out, _) = text(&run(&["insn", t]));
        assert_eq!(rc, 0, "{t}");
        let mut it = out.split_whitespace();
        let (w, h) = (it.next().unwrap(), it.next().unwrap());
        assert_eq!(text(&run(&["word", w, h])).1, format!("{t}\n"), "{t}");
    }
}

#[test]
fn usage_and_unknown_commands_are_exit_2() {
    for args in [&[][..], &["bogus"], &["insn", "--profile"]] {
        let (rc, out, err) = text(&run(args));
        assert_eq!((rc, out.as_str()), (2, ""), "{args:?}");
        assert!(!err.is_empty(), "{args:?}");
    }
}

#[test]
fn malformed_word_arguments_are_asm_001_exit_1() {
    // as `tsp_as.py word`: a coded diagnostic and exit 1, not a usage message
    for args in [&["word"][..], &["word", "32"], &["word", "x", "y"], &["word", "64", "zz"]] {
        let (rc, out, err) = text(&run(args));
        assert_eq!((rc, out.as_str()), (1, ""), "{args:?}");
        assert!(err.starts_with("word: OTS-E-ASM-001: "), "{args:?}: {err}");
    }
}

#[test]
fn flags_and_options_are_parsed_as_tsp_as_does() {
    // an unknown --flag is ignored; -o takes a value; a single-dash word is positional text
    assert_eq!(text(&run(&["--verbose", "insn", "-o", "x", "sync"])).1, "32 400000\n");
    let (rc, _, err) = text(&run(&["insn", "nop", "-0x1"]));
    assert_eq!(rc, 1);
    assert!(err.starts_with("insn: OTS-E-DEC-003: "), "{err}");
}

fn tmp_program(name: &str, body: &str) -> String {
    let p = std::env::temp_dir().join(format!("ots-asm-cli-{}-{name}.ots", std::process::id()));
    std::fs::write(&p, body).unwrap();
    p.to_str().unwrap().to_string()
}

#[test]
fn asm_input_prints_the_q_lines_and_warnings_go_to_stderr() {
    let ex = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/example.ots");
    let want = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/example.q")).unwrap();
    assert_eq!(text(&run(&["asm", ex, "--input"])), (0, want.clone(), String::new()));
    // without --input: nothing on stdout
    assert_eq!(text(&run(&["asm", ex])), (0, String::new(), String::new()));
    // a warning: exit 0, the Q lines, the warning on stderr as <where>: <code> (warning): <message>
    let p = tmp_program(
        "warn",
        ".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=0 {\n  MEM[4]: ifetchm 0x1fff\n}\n",
    );
    let (rc, out, err) = text(&run(&["asm", &p, "--input"]));
    assert_eq!((rc, out.as_str()), (0, "Q 4 01c01fff\n"));
    assert_eq!(err.lines().count(), 1, "{err}");
    assert!(err.starts_with(&format!("{p}:5: OTS-E-OPR-001 (warning): ")), "{err}");
    // an error: exit 1, nothing on stdout, <where>: <code>: <message>
    let p = tmp_program(
        "err",
        ".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=0 {\n  VXM[0]: sync\n  VXM[0]: sync\n}\n",
    );
    let (rc, out, err) = text(&run(&["asm", &p, "--input"]));
    assert_eq!((rc, out.as_str()), (1, ""));
    assert!(err.starts_with(&format!("{p}:6: OTS-E-QUE-001: ")), "{err}");
    // an object that cannot be written is exit 1 after the Q lines, the path on stderr
    let (rc, out, err) = text(&run(&["asm", ex, "--input", "-o", "/nonexistent-dir/x.elf"]));
    assert_eq!((rc, out), (1, want));
    assert!(err.starts_with("asm: OTS-E-ASM-003: /nonexistent-dir/x.elf: "), "{err}");
    // a missing file is exit 1; a missing operand is usage, exit 2
    assert_eq!(text(&run(&["asm", "/nonexistent.ots"])).0, 1);
    assert_eq!(text(&run(&["asm"])).0, 2);
}

#[test]
fn asm_config_overrides_the_profile_file_and_a_parcel_keeps_source_order() {
    let p = tmp_program("asm1", ".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=0 {\n  MEM[4]: write 0x1000, E:1, 0x0\n  MEM[4]: read 0x0, E:0, 0x0\n}\n");
    // source order: the write first, though its word is the larger
    assert_eq!(text(&run(&["asm", &p, "--input"])).1, "Q 4 90401000 04000000 10000000 00000000\n");
    assert!(!ots_asm::cli::USAGE.contains("word-order"), "the ASM-1 switch is gone");
    let cfg = format!("{}/../../../opentsp-arch-spec/config/mini.json", env!("CARGO_MANIFEST_DIR"));
    let q = tmp_program(
        "mini",
        ".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=0 {\n  MEM[4]: read 0x100, E:0, 0x0\n}\n",
    );
    assert!(text(&run(&["asm", &q, "--config", &cfg])).2.contains("OTS-E-OPR-001 (warning)"), "mini has 256 words");
    assert_eq!(text(&run(&["asm", &q])).2, "");
}

#[test]
fn help_prints_usage_on_stdout_and_succeeds() {
    for flag in ["--help", "help"] {
        let (code, out, err) = text(&run(&[flag]));
        assert_eq!(code, 0, "{err}");
        assert!(out.starts_with("usage:"));
        assert!(err.is_empty());
    }
}
