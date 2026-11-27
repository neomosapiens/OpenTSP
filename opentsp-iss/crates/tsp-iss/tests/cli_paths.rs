//! The rarely taken paths of the command line: the compatible mode's I/O failures (a configuration
//! too large to read, an unreadable standard input, a standard output that cannot be written),
//! each run on `tsp_sim` as well, and the error paths of `tsp-iss run` and `tsp-iss vectors`
//!: bad options, unreadable or inconsistent inputs, failing vectors, `--stats`.
mod common;
use common::*;
use ots_format::weights::TensorIn;
use ots_format::{write_object, write_weights, Extern};
use ots_profile::Profile;
use serde_json::{json, Value};
use std::fs::File;
use std::process::{Command, Output, Stdio};

fn cfg(name: &str) -> String {
    config(name).to_str().unwrap().to_string()
}

struct Dir(tempfile::TempDir);

impl Dir {
    fn new() -> Dir {
        Dir(tempfile::tempdir().unwrap())
    }
    fn put(&self, name: &str, data: &[u8]) -> String {
        let p = self.0.path().join(name);
        std::fs::write(&p, data).unwrap();
        p.to_str().unwrap().to_string()
    }
    fn path(&self, name: &str) -> String {
        self.0.path().join(name).to_str().unwrap().to_string()
    }
}

/// Run `bin` with the given stdin and stdout (files, directories and devices included).
fn run_io(bin: &str, args: &[&str], stdin: Stdio, stdout: Stdio) -> Output {
    Command::new(bin)
        .args(args)
        .stdin(stdin)
        .stdout(stdout)
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
        .wait_with_output()
        .unwrap()
}

/// The same run on the ISS and on `tsp_sim`: the exit code and (program name aside) stderr agree.
fn both_io(args: &[&str], stdin: impl Fn() -> Stdio, stdout: impl Fn() -> Stdio) -> Output {
    let ours = run_io(iss(), args, stdin(), stdout());
    let theirs = run_io(tsp_sim().to_str().unwrap(), args, stdin(), stdout());
    assert_eq!(
        ours.status.code(),
        theirs.status.code(),
        "exit code, args {args:?}\nours: {}\ntsp_sim: {}",
        text(&ours.stderr),
        text(&theirs.stderr)
    );
    assert_eq!(text(&ours.stderr), norm_stderr(&text(&theirs.stderr)), "stderr, args {args:?}");
    ours
}

fn devnull() -> Stdio {
    Stdio::from(File::open("/dev/null").unwrap())
}

#[cfg(target_os = "linux")]
fn full() -> Stdio {
    Stdio::from(File::options().write(true).open("/dev/full").unwrap())
}

// ---- the compatible mode ---------------------------------------------------------------------

#[test]
fn configuration_of_two_to_the_25_bytes_is_too_large() {
    // `tsp_config_load` reads at most 2^25 - 1 bytes; a file of exactly 2^25 is refused
    let d = Dir::new();
    let big = d.put("big.json", &vec![b' '; 1 << 25]);
    let o = both_io(&[&big], devnull, Stdio::piped);
    assert_eq!(o.status.code(), Some(2));
    assert_eq!(text(&o.stderr), format!("tsp-iss: config {big}: file too large\n"));
    // one byte less is read, and is not a configuration
    let small = d.put("small.json", &vec![b' '; (1 << 25) - 1]);
    let o = both_io(&[&small], devnull, Stdio::piped);
    assert_eq!(o.status.code(), Some(2));
    assert!(!text(&o.stderr).contains("too large"), "{}", text(&o.stderr));
}

/// An unreadable standard input (a directory: reading fails with EISDIR) exits 2 with the same
/// message on the ISS and on `tsp_sim` (before it, the reference ran an empty program).
/// The unwritable standard output stays an INTENDED DIFFERENCE (below).
#[test]
fn unreadable_standard_input_exits_2_on_both() {
    let dir = || Stdio::from(File::open("/").unwrap());
    let ours = both_io(&[&cfg("mini")], dir, Stdio::piped);
    assert_eq!(ours.status.code(), Some(2));
    assert!(ours.stdout.is_empty());
    assert_eq!(text(&ours.stderr), "tsp-iss: cannot read standard input: Is a directory\n");
}

#[cfg(target_os = "linux")] // needs /dev/full; Rust's stdout ignores EBADF, so no macOS stand-in fails the write
#[test]
fn unwritable_standard_output_exits_4_where_tsp_sim_exits_0() {
    let d = Dir::new();
    // more than a stdio buffer of `W` lines, so that the write fails while the run still prints
    let dumps: String = (0..40).map(|a| format!("D 0 {a:x}\n")).collect();
    let prog = d.put("p.txt", format!("S 0 0 1\n{dumps}").as_bytes());
    let two = d.put("two.txt", format!("S 0 0 1\n{dumps}C 1\n{dumps}").as_bytes());
    let cfg = cfg("mini");
    let cases: Vec<(&str, Vec<&str>)> = vec![("p.txt", vec![&cfg]), ("p.txt", vec![&cfg, "--trace"])];
    for (file, args) in cases {
        let from = || Stdio::from(File::open(if file == "p.txt" { &prog } else { &two }).unwrap());
        let ours = run_io(iss(), &args, from(), full());
        assert_eq!(ours.status.code(), Some(4), "{args:?}");
        assert_eq!(text(&ours.stderr), "tsp-iss: cannot write standard output: No space left on device\n", "{args:?}");
        let theirs = run_io(tsp_sim().to_str().unwrap(), &args, from(), full());
        assert_eq!(theirs.status.code(), Some(0), "{args:?}: {}", text(&theirs.stderr));
    }
    // several chips: each chip's output is kept in memory and written at the end, and neither checks that write
    let from = || Stdio::from(File::open(&two).unwrap());
    let o = both_io(&[&cfg, "--chips", "2"], from, full);
    assert_eq!(o.status.code(), Some(0));
    // a small output that fits the buffer: neither notices (the final flush is not checked)
    let small = d.put("s.txt", b"S 0 0 1\nD 0 0\n");
    let from = || Stdio::from(File::open(&small).unwrap());
    let o = both_io(&[&cfg], from, full);
    assert_eq!(o.status.code(), Some(0));
}

// ---- tsp-iss run -----------------------------------------------------------------------------

fn parts(profile: &Profile) -> (Vec<u8>, Vec<u8>, Vec<Extern>) {
    let mut images = ots_format::text::Images::new();
    images.extend(
        0,
        &[
            ots_isa::encode(&ots_isa::Insn::SetVL { n: 2 }) as u32,
            ots_isa::encode(&ots_isa::Insn::SetVL { n: 3 }) as u32,
        ],
    );
    let externs = vec![Extern { name: "w".into(), slice: 6, addr: 5, size: 1 }];
    let obj = write_object(profile, &images.drop_empty(), &externs);
    let tens = TensorIn { name: "w".into(), dtype: None, words: vec![vec![7; profile.vlanes() as usize]] };
    (obj, write_weights(profile, &[tens]).unwrap(), externs)
}

fn run_args(args: &[&str]) -> (i32, String, String) {
    let o = run_iss(args, "");
    (o.status.code().unwrap(), text(&o.stdout), text(&o.stderr))
}

/// `args` fail with exit 2 and a stderr that starts with `tsp-iss: run: ` and contains `want`.
fn refused(args: &[&str], want: &str) {
    let (code, out, err) = run_args(args);
    assert_eq!(code, 2, "{args:?}: {err}");
    assert_eq!(out, "", "{args:?}");
    assert!(err.starts_with("tsp-iss: run: ") && err.contains(want), "{args:?}: wanted {want:?} in {err:?}");
    assert_eq!(err.matches('\n').count(), 1, "{err}");
}

#[test]
fn run_refuses_a_manifest_of_more_than_264_chips_without_a_topology() {
    // : tsp_sim --chips refuses 265; the built-in wiring has no global link for them, so the ISS refuses too and names the option
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/verify");
    let (m, topo) = (dir.join("m265.otsm"), dir.join("topo265.json"));
    refused(&["run", m.to_str().unwrap()], "265 chips: more than 264 chips need --topology FILE");
    let (code, _, err) = run_args(&["run", "--topology", topo.to_str().unwrap(), m.to_str().unwrap()]);
    assert_eq!(code, 0, "{err}");
}

#[test]
fn run_refuses_bad_options() {
    let d = Dir::new();
    let prog = d.put("p.txt", b"Q 0 1\n");
    refused(&["run", "--profile"], "usage");
    refused(&["run", "--weights"], "usage");
    refused(&["run", "--dump"], "usage");
    refused(&["run", "--bogus", &prog], "usage");
    refused(&["run", &prog, &prog], "usage"); // two programs
    refused(&["run", "--profile", "mini", "--max-cycles", "x", &prog], "usage");
    // a message of its own, from the option shared with the compatible mode
    refused(&["run", "--profile", "mini", "--chips", "0", &prog], "--chips must be 1..264");
    // the three link options exclude one another; checked after the profile is read
    refused(&["run", "--profile", "mini", "--link-latency", "216", "--link-latency-seed", "1", &prog], "usage");
    for bad in ["zz", "1:2:3:4", ":1", "1:", "1", "1:-2", "0x:1"] {
        refused(
            &["run", "--profile", "mini", "--dump", bad, &prog],
            &format!("--dump {bad:?}: expected [CHIP:]SLICE:ADDR in hex"),
        );
    }
    // a dump of another chip, a slice or an address that does not exist
    refused(&["run", "--profile", "mini", "--dump", "1:0:0", &prog], "--dump: chip 1 out of range (--chips)");
    refused(&["run", "--profile", "mini", "--dump", "58:0", &prog], "--dump 58:0: ");
    refused(&["run", "--profile", "mini", "--dump", "0:fffff", &prog], "--dump 0:fffff: ");
    // the hex forms are accepted
    let (code, out, err) = run_args(&["run", "--profile", "mini", "--dump", "0x3:0X2", "--dump", "0:0:1", &prog]);
    assert_eq!((code, err.as_str()), (0, ""));
    assert!(out.contains("\nW 3 2 ") && out.contains("\nW 0 1 "), "{out}");
}

#[test]
fn run_refuses_what_it_cannot_load() {
    let d = Dir::new();
    let mini = Profile::load("mini").unwrap();
    let (obj, weights, _) = parts(&mini);
    let objf = d.put("p.elf", &obj);
    let wf = d.put("p.safetensors", &weights);
    let text_ok = d.put("p.txt", b"Q 0 1\n");
    refused(&["run", "--profile", "nosuch", &text_ok], "nosuch");
    refused(&["run", "--config", &d.path("missing.json"), &text_ok], "missing.json");
    refused(&["run", "--config", &d.put("c.json", b"{}"), &text_ok], "config ");
    refused(&["run", "--profile", "mini", &d.path("missing.txt")], "missing.txt: No such file or directory");
    // a text program has no profile of its own
    refused(&["run", &text_ok], "a text program has no profile of its own");
    // text that is not ASCII, or does not parse
    refused(&["run", "--profile", "mini", &d.put("na.txt", "Q 0 1 \u{e9}\n".as_bytes())], "not ASCII");
    refused(&["run", "--profile", "mini", &d.put("bad.txt", b"Q 0 zz\n")], "bad.txt");
    // an object of another profile than the one named, and a package of one
    refused(&["run", "--profile", "ref", &objf], "OTS-E-ASM-003");
    refused(&["run", "--profile", "mini", "--weights", &d.put("junk.safetensors", b"junk"), &objf], "junk.safetensors");
    refused(
        &["run", "--profile", "mini", "--weights", &d.path("none.safetensors"), &objf],
        "none.safetensors: No such file or directory",
    );
    // Multiple packages may supply distinct externs; repeating a tensor is still refused.
    refused(
        &["run", "--profile", "mini", "--weights", &wf, "--weights", &wf, &objf],
        "OTS-E-ASM-004: duplicate tensor w across weight packages",
    );
    // the program of another chip count is not a manifest: a manifest reader is still a hook
    let mut man = b"OTSM".to_vec();
    man.extend([0u8; 60]);
    refused(&["run", "--profile", "mini", &d.put("m.otsm", &man)], ".otsm");
    // an image longer than the queue, an extern at a slice that is absent
    let long = format!("Q 0 {}\n", vec!["1"; 321].join(" "));
    refused(&["run", "--profile", "ref", &d.put("long.txt", long.as_bytes())], "chip 0: ");
    let mut v: Value = serde_json::from_slice(&std::fs::read(config("mini")).unwrap()).unwrap();
    v["geometry"]["mem_slices_per_hemisphere"] = 4.into();
    let few_bytes = serde_json::to_vec(&v).unwrap();
    let absent = d.put("few.json", &few_bytes);
    let few = Profile::from_bytes("config", &few_bytes).unwrap();
    let (obj, weights, _) = parts(&few);
    let (objf, wf) = (d.put("few.elf", &obj), d.put("few.safetensors", &weights));
    // (the object reader refuses it before `run` could)
    refused(&["run", "--config", &absent, "--weights", &wf, &objf], "slice 6 does not exist");
}

#[test]
fn run_stats_failures_and_stdout_errors() {
    let d = Dir::new();
    let prog = d.put("p.txt", b"Q 0 1\n");
    // --stats=FILE into a place that cannot be written: exit 4 after a complete run
    let bad = d.path("no/such/dir/stats.json");
    let o = run_iss(&["run", "--profile", "mini", &format!("--stats={bad}"), &prog], "");
    assert_eq!(o.status.code(), Some(4), "{}", text(&o.stderr));
    assert!(text(&o.stdout).starts_with("E "), "the run printed its result first");
    assert!(
        text(&o.stderr).starts_with(&format!("tsp-iss: run: cannot write {bad}: No such file or directory")),
        "{}",
        text(&o.stderr)
    );
    // several chips are counted in the summary
    let (code, _, err) = run_args(&["run", "--profile", "mini", "--chips", "2", "--stats", &prog]);
    assert_eq!(code, 0, "{err}");
    let v: Value = serde_json::from_str(err.trim()).unwrap();
    assert_eq!(v["chips"], 2);
    assert_eq!(v["per_chip"].as_array().unwrap().len(), 2);
    assert!(v["per_chip"][0]["first_diagnostic"].is_object() || v["per_chip"][0]["first_diagnostic"].is_null());
    // a failed run has no summary
    let late = d.put(
        "late.txt",
        format!(
            "Q 0 {:x} {:x}\n",
            ots_isa::encode(&ots_isa::Insn::NOP { n: 5 }),
            ots_isa::encode(&ots_isa::Insn::SetVL { n: 2 })
        )
        .as_bytes(),
    );
    let (code, _, err) = run_args(&["run", "--profile", "mini", "--max-cycles", "1", "--stats", &late]);
    assert_eq!(code, 3, "{err}");
    assert!(!err.contains("\"ticks\""), "{err}");
}

// ---- tsp-iss vectors -------------------------------------------------------------------------

fn vectors_doc(profile: &str) -> Value {
    let path = spec().join("conformance/model").join(format!("{profile}.json"));
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn vectors_refused(args: &[&str], want: &str) {
    let o = run_iss(args, "");
    let err = text(&o.stderr);
    assert_eq!(o.status.code(), Some(2), "{args:?}: {err}");
    assert!(o.stdout.is_empty(), "{args:?}");
    assert!(err.starts_with("tsp-iss: vectors: ") && err.contains(want), "{args:?}: wanted {want:?} in {err:?}");
}

#[test]
fn vectors_refuse_inputs_that_are_not_conformance_files() {
    let d = Dir::new();
    let c = cfg("mini");
    let mut doc = vectors_doc("mini");
    doc["vectors"].as_array_mut().unwrap().truncate(3);
    let good = d.put("good.json", &serde_json::to_vec(&doc).unwrap());
    // a vector without a name
    let mut nameless = doc.clone();
    nameless["vectors"][1].as_object_mut().unwrap().remove("name");
    let nameless = d.put("nameless.json", &serde_json::to_vec(&nameless).unwrap());
    vectors_refused(&["vectors", &nameless, "--config", &c], "a vector has no name");
    // a name nobody has
    vectors_refused(&["vectors", &good, "--config", &c, "--only", "nope"], "no vector named nope");
    // a configuration that is not there, or not one
    vectors_refused(&["vectors", &good, "--config", &d.path("missing.json")], "missing.json");
    vectors_refused(&["vectors", &good, "--config", &d.put("c.json", b"[]")], "config ");
    // the profile the file names does not exist
    let mut other = doc.clone();
    other["profile"] = "nosuch".into();
    vectors_refused(&["vectors", &d.put("other.json", &serde_json::to_vec(&other).unwrap())], "nosuch");
    // options
    vectors_refused(&["vectors", &good, "--only"], "usage");
    vectors_refused(&["vectors", &good, "--config"], "usage");
    vectors_refused(&["vectors", &good, &good], "usage");
    vectors_refused(&["vectors", &good, "--jobs", "1025"], "--jobs needs a number from 1 to 1024");
    vectors_refused(&["vectors", &good, "--jobs", "x"], "--jobs needs a number from 1 to 1024");
    vectors_refused(&["vectors", &good, "--jobs"], "--jobs needs a number from 1 to 1024");
    // everything is fine with the right arguments, several --only names and many jobs
    let names: Vec<String> =
        doc["vectors"].as_array().unwrap().iter().map(|v| v["name"].as_str().unwrap().to_string()).collect();
    let o = run_iss(&["vectors", &good, "--config", &c, "--only", &names[2], "--only", &names[0], "--jobs", "64"], "");
    assert_eq!(o.status.code(), Some(0), "{}", text(&o.stderr));
    assert_eq!(text(&o.stdout), format!("pass {}\npass {}\n2 vectors: 2 pass, 0 FAIL\n", names[0], names[2]));
}

#[test]
fn vectors_report_malformed_vectors_as_failures() {
    let d = Dir::new();
    let c = cfg("mini");
    let base = vectors_doc("mini");
    let one = &base["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| !v["expect"]["words"].as_object().unwrap().is_empty())
        .unwrap()
        .clone();
    let word_key = one["expect"]["words"].as_object().unwrap().keys().next().unwrap().clone();
    type Mutation = Box<dyn Fn(&mut Value)>;
    let cases: Vec<(&str, Mutation, &str)> = vec![
        ("no text", Box::new(|v| v["text"] = Value::Null), "malformed vector: text"),
        ("no sram", Box::new(|v| v["sram"] = json!(3)), "malformed vector: sram"),
        ("no expect", Box::new(|v| v["expect"] = json!([])), "malformed vector: expect"),
        (
            "no diagnostics",
            Box::new(|v| v["expect"]["diagnostics"] = json!("x")),
            "malformed vector: expect.diagnostics",
        ),
        ("no words", Box::new(|v| v["expect"]["words"] = json!(null)), "malformed vector: expect.words"),
        ("queue", Box::new(|v| v["text"] = json!({"zz": "1"})), "malformed vector: queue number"),
        ("units not text", Box::new(|v| v["text"] = json!({"0": 1})), "malformed vector: text unit list"),
        ("unit not hex", Box::new(|v| v["text"] = json!({"0": "1 xyz"})), "malformed vector: text unit"),
        ("queue out of range", Box::new(|v| v["text"] = json!({"200": "1"})), "load: "),
        ("sram key", Box::new(|v| v["sram"] = json!({"3": "00"})), "malformed vector: sram key"),
        ("sram word odd", Box::new(|v| v["sram"] = json!({"3:1": "0"})), "malformed vector: sram word"),
        ("sram word not hex", Box::new(|v| v["sram"] = json!({"3:1": "zz"})), "malformed vector: sram word"),
        ("sram word non-ASCII", Box::new(|v| v["sram"] = json!({"3:1": "\u{e9}"})), "malformed vector: sram word"),
        ("sram absent slice", Box::new(|v| v["sram"] = json!({"99:1": "00"})), "preload: "),
        (
            "diagnostic entry",
            Box::new(|v| v["expect"]["diagnostics"] = json!([{"cycle": 1}])),
            "malformed vector: expect.diagnostics entry",
        ),
        ("word key", Box::new(|v| v["expect"]["words"] = json!({"x": "00"})), "malformed vector: word key"),
        ("word not text", Box::new(|v| v["expect"]["words"] = json!({"0:0": 5})), "malformed vector: word"),
        ("dump absent slice", Box::new(|v| v["expect"]["words"] = json!({"99:0": "00"})), "dump 99:0: "),
        ("wrong word", Box::new(move |v| v["expect"]["words"] = json!({word_key.clone(): "ff".repeat(16)})), "word "),
        (
            "wrong diagnostics",
            Box::new(|v| v["expect"]["diagnostics"] = json!([{"cycle": 7, "queue": 1, "code": "OTS-E-QUE-002"}])),
            "diagnostics: got ",
        ),
    ];
    for (name, f, want) in cases {
        let mut v = one.clone();
        f(&mut v);
        let doc = json!({"profile": "mini", "config_sha256": base["config_sha256"], "vectors": [v]});
        let p = d.put("v.json", &serde_json::to_vec(&doc).unwrap());
        let o = run_iss(&["vectors", &p, "--config", &c], "");
        let out = text(&o.stdout);
        assert_eq!(o.status.code(), Some(1), "{name}: {out} {}", text(&o.stderr));
        let first = out.lines().next().unwrap();
        assert!(first.starts_with("FAIL ") && first.contains(want), "{name}: wanted {want:?} in {first:?}");
        assert!(out.ends_with("1 vectors: 0 pass, 1 FAIL\n"), "{name}: {out}");
    }
}

#[cfg(target_os = "linux")] // needs /dev/full, see unwritable_standard_output_exits_4_where_tsp_sim_exits_0
#[test]
fn vectors_unwritable_standard_output_exits_4() {
    let d = Dir::new();
    let mut doc = vectors_doc("mini");
    doc["vectors"].as_array_mut().unwrap().truncate(2);
    let p = d.put("v.json", &serde_json::to_vec(&doc).unwrap());
    let o = run_io(iss(), &["vectors", &p, "--config", &cfg("mini")], devnull(), full());
    assert_eq!(o.status.code(), Some(4), "{}", text(&o.stderr));
    assert_eq!(text(&o.stderr), "tsp-iss: vectors: cannot write standard output\n");
}

#[test]
fn info_refuses_what_it_cannot_describe() {
    for args in
        [&["info"][..], &["info", "--config"], &["info", "--bogus", "x"], &["info", "--profile", "ref", "extra"]]
    {
        let o = run_iss(args, "");
        assert_eq!(o.status.code(), Some(2), "{args:?}");
        assert!(text(&o.stderr).starts_with("tsp-iss: info: usage: "), "{}", text(&o.stderr));
    }
    let o = run_iss(&["info", "--profile", "nosuch"], "");
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o.stderr).starts_with("tsp-iss: info: "), "{}", text(&o.stderr));
}

// ---- usage ---------------------------------------------------------------------

#[test]
fn help_and_help_word_list_the_subcommands_after_the_reference_line() {
    let sim = run_io(tsp_sim().to_str().unwrap(), &["--help"], devnull(), Stdio::piped());
    let line = text(&sim.stdout).replace(tsp_sim().to_str().unwrap(), "tsp-iss");
    for args in [&["--help"][..], &["help"]] {
        let o = run_io(iss(), args, devnull(), Stdio::piped());
        assert_eq!(o.status.code(), Some(0), "{args:?}");
        let out = text(&o.stdout);
        assert!(out.starts_with(&line), "{args:?}: {out}");
        for sub in ["run", "bench", "vectors", "info", "verify"] {
            assert!(out.lines().any(|l| l.trim_start().starts_with(&format!("{sub} "))), "{args:?} lacks {sub}: {out}");
        }
        assert!(o.stderr.is_empty());
    }
}

#[test]
fn usage_line_on_argument_errors_is_the_frozen_line() {
    // no argument, an unknown option, a missing option value: the line `tsp_sim` prints, byte for byte (program name aside)
    for args in [&[][..], &["x.json", "--bogus"], &["x.json", "--max-cycles"]] {
        let o = run_io(iss(), args, devnull(), Stdio::piped());
        let s = run_io(tsp_sim().to_str().unwrap(), args, devnull(), Stdio::piped());
        assert_eq!(o.status.code(), s.status.code(), "{args:?}");
        // the ISS prefixes its program name; the rest of the line is the reference's, with its path read as `tsp-iss`
        let theirs = text(&s.stderr).replace(tsp_sim().to_str().unwrap(), "tsp-iss");
        assert_eq!(text(&o.stderr), format!("tsp-iss: {theirs}"), "{args:?}");
        assert!(!text(&o.stderr).contains("subcommands"));
    }
}

/// The configurations the model refuses (33 superlanes, 16384 words per slice on a 13-bit address, 131072 words per slice, 640 MXM rows) get the messages that `ots-asm --config` prints
/// (`opentsp-sw-toolchain/crates/ots-asm/tests/config_validate.rs` pins the same strings).
#[test]
fn configurations_the_assemblers_also_refuse_have_the_same_messages() {
    let r = std::fs::read_to_string(config("ref")).unwrap();
    let cases = [
        (
            r.replace("\"superlanes\": 20", "\"superlanes\": 33"),
            "geometry.superlanes: 33 is above the maximum 32 (must be between 1 and 32)",
        ),
        (
            r.replace(
                "\"words_per_slice\": 8192,\n    \"bank_bit\": 12",
                "\"words_per_slice\": 16384,\n    \"bank_bit\": 13",
            ),
            "sram.bank_bit: must be < sram.addr_bits (sram.bank_bit = 13, sram.addr_bits = 13): a slice has at most 2^addr_bits words, so the bank bit (log2(words_per_slice) - 1) is below addr_bits",
        ),
        (
            r.replace(
                "\"words_per_slice\": 8192,\n    \"bank_bit\": 12,\n    \"addr_bits\": 13",
                "\"words_per_slice\": 131072,\n    \"bank_bit\": 16,\n    \"addr_bits\": 16",
            ),
            "sram.words_per_slice: must be one of 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536",
        ),
        (
            r.replace("\"rows\": 320", "\"rows\": 640"),
            "mxm.rows: 640 is above the maximum 320 (must be between 16 and 320)",
        ),
    ];
    let d = Dir::new();
    for (i, (body, message)) in cases.iter().enumerate() {
        let path = d.put(&format!("c{i}.json"), body.as_bytes());
        let o = both_io(&[&path], devnull, Stdio::piped);
        assert_eq!(o.status.code(), Some(2));
        assert_eq!(text(&o.stderr), format!("tsp-iss: config {path}: {message}\n"));
    }
}
