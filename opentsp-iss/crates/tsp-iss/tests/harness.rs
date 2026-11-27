//! The `tsp_sim`-compatible harness (input, options, configuration, output, exit codes).
//! Every test that has a reference runs `tsp_sim` on the same arguments and input and compares
//! exit code, stdout and (program name aside) stderr.
mod common;
use common::*;
use std::process::Output;

fn cfg(name: &str) -> String {
    config(name).to_str().unwrap().to_string()
}

fn mutated(base: &str, f: impl FnOnce(&mut serde_json::Value)) -> (tempfile::TempDir, String) {
    let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(config(base)).unwrap()).unwrap();
    f(&mut v);
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("cfg.json");
    std::fs::write(&p, serde_json::to_vec(&v).unwrap()).unwrap();
    let s = p.to_str().unwrap().to_string();
    (d, s)
}

/// Run both harnesses; the exit code and stdout must agree, and so must stderr with the program
/// name normalised. Returns our output.
fn both(args: &[&str], input: &str) -> Output {
    let ours = run_iss(args, input);
    let theirs = run_sim(args, input);
    assert_eq!(
        ours.status.code(),
        theirs.status.code(),
        "exit code, args {args:?}, input {input:?}\nours: {}\ntsp_sim: {}",
        text(&ours.stderr),
        text(&theirs.stderr)
    );
    assert_eq!(text(&ours.stdout), text(&theirs.stdout), "stdout, args {args:?}, input {input:?}");
    let (a, b) = (text(&ours.stderr), norm_stderr(&text(&theirs.stderr)));
    if a.contains("usage") || b.contains("usage") {
        assert!(a.contains("usage") && b.contains("usage"), "usage, args {args:?}: {a:?} vs {b:?}");
    } else {
        assert_eq!(a, b, "stderr, args {args:?}, input {input:?}");
    }
    ours
}

#[test]
fn input_errors_exit_2() {
    // (name, input, the line the error is on)
    let long_image = format!("Q 0 {}\n", vec!["1"; 321].join(" ")); // iq_depth 1280 bytes = 320 units
    let cases: Vec<(&str, String, usize)> = vec![
        ("bad queue", "# c\nQ 90 1\n".into(), 2),
        ("bad queue 10000", "Q 10000 1\n".into(), 1),
        ("slice", "\nS 58 0 1\n".into(), 2),
        ("slice in D", "D 58 0\n".into(), 1),
        ("address", "S 0 2000 1\n".into(), 1),
        ("byte", "S 0 0 100\n".into(), 1),
        ("too many bytes", format!("S 0 0 {}\n", vec!["1"; 321].join(" ")), 1),
        ("unit", "Q 0 100000000\n".into(), 1),
        ("P line", "S 0 0 1\nQ 0 0\nP 0 0 0 0\n".into(), 3),
        ("long image", long_image, 1),
        ("long image across lines", format!("Q 0 {}\nQ 0 {}\n", vec!["1"; 300].join(" "), vec!["1"; 21].join(" ")), 2),
        ("bad C", "C 1\n".into(), 1),
        ("C with extra", "C 0 0\n".into(), 1),
        ("D with extra", "D 0 0 0\n".into(), 1),
        ("missing address", "S 1\n".into(), 1),
        ("missing queue", "Q\n".into(), 1),
        ("tag without separator", "Q0 1\n".into(), 1),
        ("lower-case tag", "q 0 1\n".into(), 1),
        ("sign", "S 0 0 +1\n".into(), 1),
        ("not hex", "S 0 0 zz\n".into(), 1),
        ("number out of range", "S 0 0 10000000000000000\n".into(), 1),
        ("0x without digits", "S 0 0 0x\n".into(), 1),
        ("lone tag", "S\n".into(), 1),
    ];
    for (name, input, line) in cases {
        for c in ["ref", "mini"] {
            let o = both(&[&cfg(c)], &input);
            let err = text(&o.stderr);
            assert_eq!(o.status.code(), Some(2), "{name}: {err}");
            assert!(o.stdout.is_empty(), "{name}: stdout must be empty");
            assert!(err.starts_with("tsp-iss: "), "{name}: {err}");
            assert!(err.contains(&format!("line {line}:")), "{name}: wanted line {line}: {err}");
            assert_eq!(err.matches('\n').count(), 1, "{name}: {err}");
        }
    }
}

#[test]
fn input_errors_hold_with_chips_and_trace() {
    both(&[&cfg("ref"), "--chips", "2"], "C 2\n");
    both(&[&cfg("ref"), "--chips", "8"], "C 7\nQ 90 0\n");
    both(&[&cfg("ref"), "--trace"], "Q 0 100000000\n");
}

#[test]
fn config_rejected_one_line() {
    type Mutation = Box<dyn Fn(&mut serde_json::Value)>;
    let cases: Vec<(&str, &str, Mutation)> = vec![
        ("links 0", "c2c.links", Box::new(|v| v["c2c"]["links"] = 0.into())),
        ("mem d_func -3", "timing.mem.d_func", Box::new(|v| v["timing"]["mem"]["d_func"] = (-3).into())),
        ("link_latency_max 300", "c2c.link_latency_max", Box::new(|v| v["c2c"]["link_latency_max"] = 300.into())),
        (
            "unknown override entry",
            "timing.overrides",
            Box::new(|v| {
                v["timing"]["overrides"] = serde_json::json!([{"entry": "Frobnicate", "d_func": 5, "d_skew": 0}])
            }),
        ),
        ("superlanes 40", "geometry.superlanes", Box::new(|v| v["geometry"]["superlanes"] = 40.into())),
        ("spec_version", "spec_version", Box::new(|v| v["spec_version"] = "9.9".into())),
        ("unknown key", "extra", Box::new(|v| v["extra"] = 1.into())),
        ("words_per_slice 100", "sram.words_per_slice", Box::new(|v| v["sram"]["words_per_slice"] = 100.into())),
        (
            "geometry missing",
            "geometry",
            Box::new(|v| {
                v.as_object_mut().unwrap().remove("geometry");
            }),
        ),
    ];
    for (name, key, f) in cases {
        let (_d, path) = mutated("ref", |v| f(v));
        let o = both(&[&path], "");
        let err = text(&o.stderr);
        assert_eq!(o.status.code(), Some(2), "{name}: {err}");
        assert!(o.stdout.is_empty(), "{name}");
        assert_eq!(err.matches('\n').count(), 1, "{name}: {err}");
        assert!(err.starts_with(&format!("tsp-iss: config {path}: {key}")), "{name}: {err}");
    }
}

#[test]
fn config_raw_files_rejected() {
    let d = tempfile::tempdir().unwrap();
    let write = |name: &str, bytes: &[u8]| {
        let p = d.path().join(name);
        std::fs::write(&p, bytes).unwrap();
        p.to_str().unwrap().to_string()
    };
    let base = std::fs::read_to_string(config("ref")).unwrap();
    let dup = format!("{}, \"spec_version\": \"0.1\"}}", base.trim_end().trim_end_matches('}'));
    let mut nul = base.clone().into_bytes();
    nul.push(0);
    let paths = [
        write("array.json", b"[]"),
        write("brace.json", b"{"),
        write("empty.json", b""),
        write("dup.json", dup.as_bytes()),
        write("nul.json", &nul),
        d.path().join("missing.json").to_str().unwrap().to_string(),
        d.path().to_str().unwrap().to_string(), // a directory
    ];
    for p in paths {
        let o = both(&[&p], "");
        let err = text(&o.stderr);
        assert_eq!(o.status.code(), Some(2), "{p}: {err}");
        assert!(err.starts_with(&format!("tsp-iss: config {p}: ")), "{err}");
        assert_eq!(err.matches('\n').count(), 1, "{err}");
    }
}

#[test]
fn usage_errors_exit_2() {
    let c = cfg("ref");
    let arg_sets: Vec<Vec<&str>> = vec![
        vec![],
        vec![&c, "--bogus"],
        vec![&c, "--max-cycles"],
        vec![&c, "--max-cycles", "-1"],
        vec![&c, "--max-cycles", "+5"],
        vec![&c, "--max-cycles", " 5"],
        vec![&c, "--max-cycles", ""],
        vec![&c, "--max-cycles", "18446744073709551616"],
        vec![&c, "--max-cycles", "5x"],
        vec![&c, "--max-cycles", "0x"],
        vec![&c, "--max-cycles", "08"],
        vec![&c, "--link-latency", "-1"],
        vec![&c, "--link-latency", "1x"],
        vec![&c, "--link-latency-seed", "-1"],
        vec![&c, "--link-jitter-seed", "-3"],
        vec![&c, "--link-latency", "216", "--link-latency-seed", "1"],
        vec![&c, "--link-latency", "216", "--link-jitter-seed", "1"],
        vec![&c, "--link-latency-seed", "1", "--link-jitter-seed", "1"],
        vec![&c, "--trace", "extra"],
    ];
    for args in arg_sets {
        let o = both(&args, "");
        assert_eq!(o.status.code(), Some(2), "{args:?}");
        assert!(o.stdout.is_empty());
        assert!(text(&o.stderr).contains("usage"), "{args:?}: {}", text(&o.stderr));
    }
}

#[test]
fn chips_and_link_options_checked() {
    let c = cfg("ref");
    for args in [
        vec![&c[..], "--chips", "0"],
        vec![&c, "--chips", "265"],
        vec![&c, "--chips", "x"],
        vec![&c, "--chips", ""],
        vec![&c, "--link-latency", "1"],
        vec![&c, "--link-latency", "229"],
        vec![&c, "--link-latency", ""],
    ] {
        let o = both(&args, "");
        assert_eq!(o.status.code(), Some(2), "{args:?}");
        assert!(o.stdout.is_empty());
    }
}

#[test]
fn numeric_forms_accepted() {
    let c = cfg("mini");
    for args in [
        vec![&c[..], "--max-cycles", "0x10"],
        vec![&c, "--max-cycles", "010"],
        vec![&c, "--max-cycles", "0"],
        vec![&c, "--chips", "0x2"],
        vec![&c, "--chips", " 3"],
        vec![&c, "--chips", "+2"],
        vec![&c, "--chips", "16"],
        vec![&c, "--chips", "9"],
        vec![&c, "--chips", "12"],
        vec![&c, "--link-latency", "209"],
        vec![&c, "--link-latency", "228"],
        vec![&c, "--link-latency-seed", "0"],
        vec![&c, "--link-jitter-seed", "7"],
        vec![&c, "--link-jitter-seed", "0x7"],
        vec![&c, "--trace", "--trace"],
        vec![&c, "--max-cycles", "5", "--max-cycles", "6"],
    ] {
        let o = both(&args, "");
        assert_eq!(o.status.code(), Some(0), "{args:?}: {}", text(&o.stderr));
    }
}

/// `tsp_sim` tests `errno` after `strtol`: an empty value converts no digit, which glibc leaves
/// alone (accepted as 0) and the BSD libc reports as `EINVAL` (usage error). The ISS follows the
/// libc of the platform it runs on, as `tsp_sim` does.
#[test]
fn empty_link_option_value_follows_the_platform_libc() {
    let c = cfg("mini");
    let want = if cfg!(target_vendor = "apple") { 2 } else { 0 };
    for opt in ["--link-latency-seed", "--link-jitter-seed"] {
        let o = both(&[&c, opt, ""], "");
        assert_eq!(o.status.code(), Some(want), "{opt}: {}", text(&o.stderr));
    }
}

#[test]
fn output_lines_match_tsp_sim() {
    let hex320 =
        |prefix: &[u8]| (0..320).map(|i| format!(" {:02x}", prefix.get(i).copied().unwrap_or(0))).collect::<String>();
    let input = "# preload then dump\n S 3 5 01 02 0x03\t04\r\nS\t3\t5 aa bb\n\nD 3 5\nD 0 0\nD 57 1\n";
    for c in ["ref", "mini"] {
        let o = both(&[&cfg(c)], input);
        let out = text(&o.stdout);
        let vl = if c == "ref" { 320 } else { 16 };
        // the second S replaces the first; lanes beyond vlanes read 00
        let mut want = String::from("E - -1 0\n");
        want += &format!("W 3 5{}\n", hex320(&[0xaa, 0xbb]));
        want += &format!("W 0 0{}\n", hex320(&[]));
        want += &format!("W 57 1{}\n", hex320(&[]));
        assert_eq!(out, want, "{c} (vlanes {vl})");
    }
    // : an S line longer than the lane count is an input error naming the line and the lane count (it used to
    // lose its tail silently); a line of exactly the lane count is whole, a shorter one is zero-filled
    let full: String = (0..320).map(|i| format!(" {:x}", i as u8)).collect();
    let o = both(&[&cfg("mini")], &format!("# c\nS 1 2{full}\nD 1 2\n"));
    assert_eq!(o.status.code(), Some(2));
    let err = text(&o.stderr);
    assert!(err.contains("input line 2") && err.contains("16 lanes"), "{err}");
    assert!(text(&o.stdout).is_empty());
    let exact: String = (0..16).map(|i| format!(" {:x}", i as u8)).collect();
    let o = both(&[&cfg("mini")], &format!("S 1 2{exact}\nD 1 2\n"));
    let line = text(&o.stdout).lines().nth(1).unwrap().to_string();
    let bytes: Vec<&str> = line.split(' ').skip(3).collect();
    assert_eq!(bytes.len(), 320, "a dump line is 320 bytes, lanes beyond vlanes read 00");
    assert!(bytes[..16].iter().enumerate().all(|(i, b)| *b == format!("{i:02x}")));
    assert!(bytes[16..].iter().all(|b| *b == "00"));
}

#[test]
fn queue_images_accepted_and_padding_dropped() {
    // an image of exactly iq_depth/4 units, padding anywhere, several lines per queue, ffff
    let full = vec!["0"; 320].join(" ");
    for input in [
        format!("Q 0 {full}\n"),
        format!("Q 0 {full} ffffffff ffffffff\n"),
        format!("Q 0 ffffffff {full}\nQ 0 ffffffff\n"),
        "Q 8f 1 2 3\nQ ffff 4\nQ 0 0x5\n".to_string(),
    ] {
        let o = both(&[&cfg("ref")], &input);
        assert_eq!(o.status.code(), Some(0), "{input:.40}: {}", text(&o.stderr));
    }
}

#[test]
fn chips_blocks_in_chip_order() {
    let input = "S 0 0 11\nD 0 0\nC 1\nS 0 0 22\nD 0 0\nC 0\nD 0 1\nC 1\nD 0 0\n";
    for k in ["2", "8", "16"] {
        let o = both(&[&cfg("mini"), "--chips", k], input);
        let out = text(&o.stdout);
        assert!(out.starts_with("C 0\nE - -1 0\n"), "{out:.80}");
        let blocks = out.matches("\nC ").count() + 1;
        assert_eq!(blocks, k.parse::<usize>().unwrap());
    }
}

#[test]
fn trace_flag_adds_nothing_without_events() {
    both(&[&cfg("ref"), "--trace"], "S 0 0 1\nD 0 0\n");
}

#[test]
fn absent_slices_match_tsp_sim() {
    // With 8 MEM slices per hemisphere, slices 8..43 and 52..87 do not exist: `S` and `D`
    // lines for them are input errors, a `Q` line for their queue is accepted
    let (_d, path) = mutated("ref", |v| v["geometry"]["mem_slices_per_hemisphere"] = 8.into());
    for (input, ok) in [
        ("S 7 0 1\nD 7 0\nS 33 0 1\nD 33 1\n", true),
        ("S 8 0 1\n", false),
        ("D 2b 0\n", false),
        ("D 34 0\n", false),
        ("D 57 0\n", false),
        ("Q 8 1\nQ 3c 2\n", true),
    ] {
        let o = both(&[&path], input);
        assert_eq!(o.status.code(), Some(if ok { 0 } else { 2 }), "{input:?}: {}", text(&o.stderr));
    }
}
