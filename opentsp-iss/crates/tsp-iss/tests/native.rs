//! The native subcommands `run` and `vectors`.
//!
//! `run` reads an `.elf` (with its `.safetensors`), a Q-line text file or an `.otsm`, through
//! `ots-format`, and must print exactly what the compatible mode prints for the harness text that
//! `ots-asm objinput` makes of the same files, and what `tsp_sim` prints for that text. The objects
//! and weight packages are made from the golden vectors of `conformance/model/`: one extern per
//! preloaded word, one tensor per extern.
mod common;
use common::*;
use ots_format::text::{input_text, Images};
use ots_format::weights::TensorIn;
use ots_format::{
    bind_weights, preload_text, read_object, read_weights, write_manifest, write_object, write_weights, Extern,
};
use ots_profile::Profile;
use serde_json::Value;
use std::path::PathBuf;

fn vectors_of(profile: &str) -> Vec<Value> {
    let path = spec().join("conformance/model").join(format!("{profile}.json"));
    let doc: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    doc["vectors"].as_array().unwrap().clone()
}

/// A vector with an `F` (fault injection) line in its host lines: `run` has no way to inject a fault, so
/// these vectors are checked by `tsp-iss vectors` and the fault tests of tests/ecc.rs instead.
fn injects_faults(v: &Value) -> bool {
    v["host"].as_array().is_some_and(|h| h.iter().any(|l| l.as_str().is_some_and(|l| l.starts_with("F "))))
}

fn hex_bytes(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}

fn key_pair(k: &str) -> (u64, u64) {
    let (a, b) = k.split_once(':').unwrap();
    (a.parse().unwrap(), b.parse().unwrap())
}

/// The images, externs and tensors of a vector.
fn parts(v: &Value) -> (Images, Vec<Extern>, Vec<TensorIn>) {
    let mut images = Images::new();
    for (q, units) in v["text"].as_object().unwrap() {
        let units: Vec<u32> =
            units.as_str().unwrap().split_whitespace().map(|u| u32::from_str_radix(u, 16).unwrap()).collect();
        images.extend(q.parse().unwrap(), &units);
    }
    let (mut externs, mut tensors) = (Vec::new(), Vec::new());
    for (i, (k, w)) in v["sram"].as_object().unwrap().iter().enumerate() {
        let (slice, addr) = key_pair(k);
        let name = format!("w{i}");
        externs.push(Extern { name: name.clone(), slice, addr, size: 1 });
        tensors.push(TensorIn { name, dtype: None, words: vec![hex_bytes(w.as_str().unwrap())] });
    }
    (images.drop_empty(), externs, tensors)
}

struct Files {
    dir: tempfile::TempDir,
}

impl Files {
    fn new() -> Files {
        Files { dir: tempfile::tempdir().unwrap() }
    }
    fn put(&self, name: &str, data: &[u8]) -> String {
        let p = self.dir.path().join(name);
        std::fs::write(&p, data).unwrap();
        p.to_str().unwrap().to_string()
    }
    fn path(&self, name: &str) -> String {
        self.dir.path().join(name).to_str().unwrap().to_string()
    }
}

/// `--dump` arguments (hex `slice:addr`) and `D` lines for the words a vector expects.
fn dumps(v: &Value) -> (Vec<String>, String) {
    let (mut args, mut lines) = (Vec::new(), String::new());
    for k in v["expect"]["words"].as_object().unwrap().keys() {
        let (s, a) = key_pair(k);
        args.push("--dump".to_string());
        args.push(format!("{s:x}:{a:x}"));
        lines.push_str(&format!("D {s:x} {a:x}\n"));
    }
    (args, lines)
}

fn strs(v: &[String]) -> Vec<&str> {
    v.iter().map(String::as_str).collect()
}

/// The words a vector expects, as the `W` lines of the output.
fn expected_w(v: &Value, vl: usize) -> Vec<String> {
    v["expect"]["words"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, w)| {
            let (s, a) = key_pair(k);
            let bytes = hex_bytes(w.as_str().unwrap());
            let mut line = format!("W {s:x} {a:x}");
            for b in (0..320).map(|i| if i < vl { bytes[i] } else { 0 }) {
                line.push_str(&format!(" {b:02x}"));
            }
            line
        })
        .collect()
}

fn config_arg(profile: &str) -> String {
    config(profile).to_str().unwrap().to_string()
}

/// What one vector did: through an object, or only as text (the object reader refused the text).
#[derive(Debug, PartialEq, Eq)]
enum Way {
    Object,
    TextOnly,
}

/// One vector through `run` (object + package), the compatible mode (the `objinput` text) and `tsp_sim`.
/// A few vectors are deliberately malformed text that an object cannot hold (a parcel that does not fit
/// its queue's slots): `ots-format` refuses them with OTS-E-ASM-003 and so must `run`; those go through
/// `run` as a text file instead.
fn check_vector(profile: &str, p: &Profile, v: &Value, extra: &[&str]) -> Way {
    let name = v["name"].as_str().unwrap();
    let (images, externs, tensors) = parts(v);
    let f = Files::new();
    let bytes = write_object(p, &images, &externs);
    let w = f.put("p.safetensors", &write_weights(p, &tensors).unwrap());
    let (dargs, dlines) = dumps(v);
    let cfg = config_arg(profile);

    let (prog, how) = match read_object(&bytes, Some(profile), None) {
        Ok(_) => (f.put("p.elf", &bytes), Way::Object),
        Err(e) => {
            assert_eq!(e.code.as_str(), "OTS-E-ASM-003", "{profile}/{name}: {e}");
            let obj = f.put("p.elf", &bytes);
            let o = run_iss(&["run", "--config", &cfg, &obj], "");
            assert_eq!(o.status.code(), Some(2), "{profile}/{name}");
            assert!(o.stdout.is_empty());
            assert!(text(&o.stderr).contains("OTS-E-ASM-003"), "{}", text(&o.stderr));
            (f.put("p.txt", input_text(&images).as_bytes()), Way::TextOnly)
        }
    };

    let mut args = vec!["run", "--config", &cfg];
    if how == Way::Object {
        args.extend(["--weights", &w]);
    }
    args.extend(extra);
    args.extend(strs(&dargs));
    args.push(&prog);
    let ours = run_iss(&args, "");
    assert_eq!(ours.status.code(), Some(0), "{profile}/{name}: {}", text(&ours.stderr));

    // what `ots-asm objinput prog --weights w` prints, and the dump lines
    let pre = if how == Way::Object {
        let pk = read_weights(&std::fs::read(&w).unwrap(), Some(profile), None).unwrap();
        preload_text(&bind_weights(&externs, &pk).unwrap())
    } else {
        String::new()
    };
    let stdin = format!("{pre}{}{dlines}", input_text(&images));
    let mut cargs = vec![cfg.as_str()];
    cargs.extend(extra);
    let compat = run_iss(&cargs, &stdin);
    assert_eq!(text(&ours.stdout), text(&compat.stdout), "{profile}/{name}: run vs compatible mode");
    assert_eq!(ours.status.code(), compat.status.code());
    let sim = run_sim(&cargs, &stdin);
    assert_eq!(text(&ours.stdout), text(&sim.stdout), "{profile}/{name}: run vs tsp_sim");

    // with the weights, the words are the golden ones
    if how == Way::Object {
        let out = text(&ours.stdout);
        let got: Vec<&str> = out.lines().filter(|l| l.starts_with("W ")).collect();
        let want = expected_w(v, p.vlanes() as usize);
        assert_eq!(got, want.iter().map(String::as_str).collect::<Vec<_>>(), "{profile}/{name}: words");
    }
    how
}

#[test]
fn run_object_equals_objinput_on_compat_mode() {
    for profile in ["ref", "mini"] {
        let p = Profile::load(profile).unwrap();
        let vs = vectors_of(profile);
        assert!(vs.len() >= 500);
        let mut counts = [0usize; 2];
        for (i, v) in vs.iter().enumerate() {
            if i % 11 == 0 && !injects_faults(v) {
                counts[(check_vector(profile, &p, v, &[]) == Way::TextOnly) as usize] += 1;
            }
        }
        // most vectors are objects; the few that are not were refused by name
        assert!(counts[0] >= 40 && counts[0] > 3 * counts[1], "{profile}: {counts:?}");
    }
}

#[test]
fn run_with_trace_and_max_cycles_equals_compat_mode() {
    let p = Profile::load("ref").unwrap();
    let vs = vectors_of("ref");
    for v in vs.iter().step_by(37).filter(|v| !injects_faults(v)) {
        check_vector("ref", &p, v, &["--trace"]);
        check_vector("ref", &p, v, &["--max-cycles", "0x10000000"]);
    }
}

/// Without `--config` or `--profile` an object names its own profile; `--profile` must agree.
#[test]
fn run_takes_the_profile_of_the_object() {
    let p = Profile::load("mini").unwrap();
    let vs = vectors_of("mini");
    let v = &vs[3];
    let (images, externs, tensors) = parts(v);
    let f = Files::new();
    let obj = f.put("p.elf", &write_object(&p, &images, &externs));
    let w = f.put("p.safetensors", &write_weights(&p, &tensors).unwrap());
    let (dargs, _) = dumps(v);
    let base: Vec<&str> = ["run", "--weights", &w].into_iter().chain(strs(&dargs)).collect();
    let a = run_iss(&[&base[..], &[&obj]].concat(), "");
    assert_eq!(a.status.code(), Some(0), "{}", text(&a.stderr));
    let b = run_iss(&[&["run", "--profile", "mini"], &base[1..], &[&obj]].concat(), "");
    assert_eq!(text(&a.stdout), text(&b.stdout));
    let c = run_iss(&[&["run", "--profile", "ref"], &base[1..], &[&obj]].concat(), "");
    assert_eq!(c.status.code(), Some(2));
    assert!(c.stdout.is_empty());
    assert!(text(&c.stderr).starts_with("tsp-iss: run: "), "{}", text(&c.stderr));
    assert!(text(&c.stderr).contains("mini"), "{}", text(&c.stderr));
}

/// A Q-line text file needs a profile; its output is the compatible mode's on the same lines.
#[test]
fn run_text_program() {
    let p = Profile::load("ref").unwrap();
    let vs = vectors_of("ref");
    let v = &vs[5];
    let (images, _, _) = parts(v);
    let f = Files::new();
    let q = input_text(&images);
    let prog = f.put("p.txt", q.as_bytes());
    let cfg = config_arg("ref");
    let ours = run_iss(&["run", "--config", &cfg, &prog], "");
    assert_eq!(ours.status.code(), Some(0), "{}", text(&ours.stderr));
    let compat = run_iss(&[&cfg], &q);
    assert_eq!(text(&ours.stdout), text(&compat.stdout));
    let by_name = run_iss(&["run", "--profile", "ref", &prog], "");
    assert_eq!(text(&by_name.stdout), text(&compat.stdout));
    // the text has no profile of its own
    let none = run_iss(&["run", &prog], "");
    assert_eq!(none.status.code(), Some(2));
    assert!(text(&none.stderr).contains("--profile"), "{}", text(&none.stderr));
    // the file is checked like `objinput` checks it
    let bad = f.put("bad.txt", b"Q 90 1\n");
    let o = run_iss(&["run", "--config", &cfg, &bad], "");
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o.stderr).contains("OTS-E-ASM-003"), "{}", text(&o.stderr));
    let _ = p;
}

#[test]
fn run_several_chips_equals_compat_mode() {
    let vs = vectors_of("ref");
    let v = &vs[8];
    let (images, _, _) = parts(v);
    let f = Files::new();
    let q = input_text(&images);
    let prog = f.put("p.txt", q.as_bytes());
    let cfg = config_arg("ref");
    for extra in [
        &["--chips", "2"][..],
        &["--chips", "3", "--link-latency", "210"],
        &["--chips", "2", "--link-jitter-seed", "5", "--trace"],
    ] {
        let mut a = vec!["run", "--config", &cfg];
        a.extend(extra);
        a.push(&prog);
        let ours = run_iss(&a, "");
        let mut c = vec![cfg.as_str()];
        c.extend(extra);
        let compat = run_iss(&c, &q);
        assert_eq!(ours.status.code(), compat.status.code(), "{extra:?}: {}", text(&ours.stderr));
        assert_eq!(text(&ours.stdout), text(&compat.stdout), "{extra:?}");
        assert!(text(&ours.stdout).starts_with("C 0\n"), "{extra:?}");
    }
}

#[test]
fn run_exit_codes() {
    let vs = vectors_of("ref");
    let p = Profile::load("ref").unwrap();
    let v = vs.iter().find(|v| !v["sram"].as_object().unwrap().is_empty()).unwrap();
    let (images, externs, tensors) = parts(v);
    assert!(!externs.is_empty());
    let f = Files::new();
    let obj = f.put("p.elf", &write_object(&p, &images, &externs));
    let cfg = config_arg("ref");
    // no program, unknown option, a missing file: usage and input errors are exit 2
    for args in [
        &["run"][..],
        &["run", "--bogus", &obj],
        &["run", "--config"],
        &["run", &f.path("missing.elf")],
        &["run", "--max-cycles", "x", &obj],
    ] {
        let o = run_iss(args, "");
        assert_eq!(o.status.code(), Some(2), "{args:?}: {}", text(&o.stderr));
        assert!(o.stdout.is_empty(), "{args:?}");
        assert!(text(&o.stderr).starts_with("tsp-iss: run: "), "{args:?}: {}", text(&o.stderr));
    }
    // an object whose externs need a package
    let o = run_iss(&["run", "--config", &cfg, &obj], "");
    // no package: the externs stay unfilled (zero words), as `tsp_sim` has them with no `S` lines
    assert_eq!(o.status.code(), Some(0), "{}", text(&o.stderr));
    // a package without the tensors: OTS-E-ASM-004, nothing printed
    let other = f.put(
        "other.safetensors",
        &write_weights(&p, &[TensorIn { name: "zz".into(), dtype: None, words: vec![vec![0; p.vlanes() as usize]] }])
            .unwrap(),
    );
    let o = run_iss(&["run", "--config", &cfg, "--weights", &other, &obj], "");
    assert_eq!(o.status.code(), Some(2));
    assert!(o.stdout.is_empty());
    assert!(text(&o.stderr).contains("OTS-E-ASM-004"), "{}", text(&o.stderr));
    // a package for the wrong profile
    let m = Profile::load("mini").unwrap();
    let wm = f.put(
        "m.safetensors",
        &write_weights(
            &m,
            &[TensorIn { name: externs[0].name.clone(), dtype: None, words: vec![vec![0; m.vlanes() as usize]] }],
        )
        .unwrap(),
    );
    let o = run_iss(&["run", "--config", &cfg, "--weights", &wm, &obj], "");
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o.stderr).contains("OTS-E-ASM-003"), "{}", text(&o.stderr));
    // garbage that is neither text nor an object
    let junk = f.put("junk.elf", b"\x00\x01\x02garbage");
    let o = run_iss(&["run", "--config", &cfg, &junk], "");
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o.stderr).contains("OTS-E-ASM-003"), "{}", text(&o.stderr));
    // exhausting the cycle budget: exit 3 naming max-cycles, as in the compatible mode
    let w2 = f.put("b.safetensors", &write_weights(&p, &tensors).unwrap());
    let o = run_iss(&["run", "--config", &cfg, "--weights", &w2, "--max-cycles", "1", &obj], "");
    let pk = read_weights(&std::fs::read(&w2).unwrap(), Some("ref"), None).unwrap();
    let stdin = format!("{}{}", preload_text(&bind_weights(&externs, &pk).unwrap()), input_text(&images));
    let c = run_iss(&[&cfg, "--max-cycles", "1"], &stdin);
    assert_eq!(o.status.code(), c.status.code(), "{}", text(&o.stderr));
    assert_eq!(text(&o.stderr).replace("tsp-iss: run: ", "tsp-iss: "), text(&c.stderr));
    if o.status.code() == Some(3) {
        assert!(text(&o.stderr).contains("max-cycles"), "{}", text(&o.stderr));
    }
}

#[test]
fn run_stats_go_to_stderr_or_a_file() {
    let vs = vectors_of("ref");
    let p = Profile::load("ref").unwrap();
    let (images, externs, tensors) = parts(&vs[4]);
    let f = Files::new();
    let obj = f.put("p.elf", &write_object(&p, &images, &externs));
    let w = f.put("p.safetensors", &write_weights(&p, &tensors).unwrap());
    let cfg = config_arg("ref");
    let plain = run_iss(&["run", "--config", &cfg, "--weights", &w, &obj], "");
    let stat = run_iss(&["run", "--config", &cfg, "--weights", &w, "--stats", &obj], "");
    assert_eq!(text(&plain.stdout), text(&stat.stdout), "--stats never changes stdout");
    let v: Value =
        serde_json::from_str(text(&stat.stderr).trim()).unwrap_or_else(|e| panic!("{e}: {}", text(&stat.stderr)));
    assert_eq!(v["profile"], "ref");
    assert_eq!(v["chips"], 1);
    assert!(v["ticks"].is_u64() && v["diagnostics"].is_u64());
    let file = f.path("stats.json");
    let stat2 = run_iss(&["run", "--config", &cfg, "--weights", &w, &format!("--stats={file}"), &obj], "");
    assert_eq!(text(&plain.stdout), text(&stat2.stdout));
    assert!(stat2.stderr.is_empty(), "{}", text(&stat2.stderr));
    let v2: Value = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    assert_eq!(v2["ticks"], v["ticks"]);
}

// ---- the manifest -------------------------------------------------------------------

/// A two-chip manifest with one weight package per chip equals the compatible mode on the
/// `mobjinput` text (`C <i>`, the `S` lines of chip i, its `Q` lines), and `tsp_sim`.
#[test]
fn run_otsm_equals_mobjinput_on_compat_mode() {
    let p = Profile::load("ref").unwrap();
    let vs = vectors_of("ref");
    let (v0, v1) = (&vs[6], &vs[9]);
    let f = Files::new();
    let mut objs = Vec::new();
    let mut wpaths = Vec::new();
    let mut stdin = String::new();
    for (i, v) in [v0, v1].iter().enumerate() {
        let (images, externs, tensors) = parts(v);
        objs.push(write_object(&p, &images, &externs));
        let w = f.put(&format!("c{i}.safetensors"), &write_weights(&p, &tensors).unwrap());
        let pk = read_weights(&std::fs::read(&w).unwrap(), Some("ref"), None).unwrap();
        stdin.push_str(&format!(
            "C {i:x}\n{}{}",
            preload_text(&bind_weights(&externs, &pk).unwrap()),
            input_text(&images)
        ));
        wpaths.push(w);
    }
    let m = f.put("m.otsm", &write_manifest("ref", &objs, None).unwrap());
    let cfg = config_arg("ref");
    let ours = run_iss(&["run", "--config", &cfg, "--weights", &wpaths[0], "--weights", &wpaths[1], &m], "");
    assert_eq!(ours.status.code(), Some(0), "{}", text(&ours.stderr));
    let compat = run_iss(&[&cfg, "--chips", "2"], &stdin);
    assert_eq!(text(&ours.stdout), text(&compat.stdout));
    let sim = run_sim(&[&cfg, "--chips", "2"], &stdin);
    assert_eq!(text(&ours.stdout), text(&sim.stdout));
}

// ---- vectors ----------------------------------------------------------------------------------

fn vectors_file(profile: &str) -> PathBuf {
    spec().join("conformance/model").join(format!("{profile}.json"))
}

fn cfg_path(profile: &str) -> String {
    config(profile).to_str().unwrap().to_string()
}

#[test]
fn vectors_pass_on_every_golden_vector() {
    for profile in ["ref", "mini"] {
        let file = vectors_file(profile);
        let n = vectors_of(profile).len();
        let o = run_iss(&["vectors", file.to_str().unwrap(), "--config", &cfg_path(profile)], "");
        let out = text(&o.stdout);
        assert_eq!(o.status.code(), Some(0), "{profile}: {out}\n{}", text(&o.stderr));
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.iter().filter(|l| l.starts_with("pass ")).count(), n, "{profile}");
        assert!(!lines.iter().any(|l| l.starts_with("FAIL")), "{profile}");
        assert_eq!(*lines.last().unwrap(), format!("{n} vectors: {n} pass, 0 FAIL"), "{profile}");
        // in file order
        let names: Vec<String> =
            vectors_of(profile).iter().map(|v| format!("pass {}", v["name"].as_str().unwrap())).collect();
        let got: Vec<String> = lines[..n].iter().map(|l| l.to_string()).collect();
        assert_eq!(got, names, "{profile}");
    }
}

#[test]
fn vectors_default_to_the_profile_of_the_file() {
    let file = vectors_file("mini");
    let only = only_args("mini", 6);
    let o: Vec<&str> = only.iter().map(String::as_str).collect();
    let a = run_iss(&[&["vectors", file.to_str().unwrap()], &o[..]].concat(), "");
    let b = run_iss(&[&["vectors", file.to_str().unwrap(), "--config", &cfg_path("mini")], &o[..]].concat(), "");
    assert_eq!(a.status.code(), Some(0), "{}", text(&a.stderr));
    assert_eq!(text(&a.stdout), text(&b.stdout));
    assert_eq!(text(&a.stdout).lines().count(), 7);
}

/// `--only` arguments for the first `n` vectors of a profile.
fn only_args(profile: &str, n: usize) -> Vec<String> {
    vectors_of(profile)
        .iter()
        .take(n)
        .flat_map(|v| ["--only".to_string(), v["name"].as_str().unwrap().to_string()])
        .collect()
}

#[test]
fn vectors_only_and_jobs() {
    let file = vectors_file("ref");
    let f = file.to_str().unwrap();
    let c = cfg_path("ref");
    let name = vectors_of("ref")[7]["name"].as_str().unwrap().to_string();
    let o = run_iss(&["vectors", f, "--config", &c, "--only", &name], "");
    assert_eq!(text(&o.stdout), format!("pass {name}\n1 vectors: 1 pass, 0 FAIL\n"));
    let o = run_iss(&["vectors", f, "--config", &c, "--only", "no-such-vector"], "");
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o.stderr).contains("no-such-vector"), "{}", text(&o.stderr));
    let only = only_args("ref", 40);
    let o: Vec<&str> = only.iter().map(String::as_str).collect();
    let j1 = run_iss(&[&["vectors", f, "--config", &c, "--jobs", "1"], &o[..]].concat(), "");
    let j4 = run_iss(&[&["vectors", f, "--config", &c, "--jobs", "4"], &o[..]].concat(), "");
    assert_eq!(j1.status.code(), Some(0));
    assert_eq!(text(&j1.stdout), text(&j4.stdout), "output does not depend on --jobs");
    assert_eq!(text(&j1.stdout).lines().count(), 41);
    let o = run_iss(&["vectors", f, "--config", &c, "--jobs", "0"], "");
    assert_eq!(o.status.code(), Some(2));
}

fn mutate_expect(profile: &str, f: impl Fn(usize, &mut Value)) -> (tempfile::TempDir, String) {
    let mut doc: Value = serde_json::from_slice(&std::fs::read(vectors_file(profile)).unwrap()).unwrap();
    for (i, v) in doc["vectors"].as_array_mut().unwrap().iter_mut().enumerate() {
        f(i, v);
    }
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("v.json");
    std::fs::write(&p, serde_json::to_vec(&doc).unwrap()).unwrap();
    let s = p.to_str().unwrap().to_string();
    (d, s)
}

#[test]
fn vectors_report_each_failure_and_exit_1() {
    let (_d, path) = mutate_expect("ref", |i, v| {
        if i == 3 {
            // a wrong word
            let w = v["expect"]["words"].as_object_mut().unwrap();
            if let Some((_, x)) = w.iter_mut().next() {
                let s = x.as_str().unwrap().to_string();
                let flipped = match s.strip_prefix('0') {
                    Some(rest) => format!("1{rest}"),
                    None => format!("0{}", &s[1..]),
                };
                *x = Value::String(flipped);
            } else {
                v["expect"]["diagnostics"] = serde_json::json!([{"cycle": 1, "queue": 0, "code": "OTS-E-IQ-002"}]);
            }
        }
        if i == 10 {
            // an unexpected diagnostic
            v["expect"]["diagnostics"] = serde_json::json!([{"cycle": 99999, "queue": 7, "code": "OTS-E-QUE-002"}]);
        }
    });
    let only = only_args("ref", 12);
    let o: Vec<&str> = only.iter().map(String::as_str).collect();
    let o = run_iss(&[&["vectors", &path, "--config", &cfg_path("ref")], &o[..]].concat(), "");
    let out = text(&o.stdout);
    assert_eq!(o.status.code(), Some(1), "{}", text(&o.stderr));
    let n = 12;
    let fails: Vec<&str> = out.lines().filter(|l| l.starts_with("FAIL ")).collect();
    let names = vectors_of("ref");
    assert_eq!(fails.len(), 2, "{out}");
    assert!(fails[0].starts_with(&format!("FAIL {}: ", names[3]["name"].as_str().unwrap())), "{}", fails[0]);
    assert!(fails[1].starts_with(&format!("FAIL {}: ", names[10]["name"].as_str().unwrap())), "{}", fails[1]);
    assert_eq!(out.lines().last().unwrap(), format!("{n} vectors: {} pass, 2 FAIL", n - 2));
}

#[test]
fn vectors_input_errors_exit_2() {
    let f = Files::new();
    let junk = f.put("j.json", b"{not json");
    let bad = f.put("b.json", b"{\"profile\":\"ref\"}");
    let c = cfg_path("ref");
    for args in [
        vec!["vectors"],
        vec!["vectors", &junk, "--config", &c],
        vec!["vectors", &bad, "--config", &c],
        vec!["vectors", "/nonexistent/v.json"],
        vec!["vectors", "--bogus"],
    ] {
        let o = run_iss(&args, "");
        assert_eq!(o.status.code(), Some(2), "{args:?}: {}", text(&o.stderr));
        assert!(o.stdout.is_empty(), "{args:?}");
        assert!(text(&o.stderr).starts_with("tsp-iss: vectors: "), "{args:?}: {}", text(&o.stderr));
    }
}

#[test]
fn separate_program_data_and_weight_packages_form_one_single_chip_binding() {
    for profile in ["mini", "ref"] {
        let p = Profile::load(profile).unwrap();
        let v = vectors_of(profile)
            .into_iter()
            .find(|v| {
                !injects_faults(v)
                    && v["sram"].as_object().unwrap().len() >= 2
                    && !v["expect"]["words"].as_object().unwrap().is_empty()
            })
            .unwrap();
        let (images, externs, tensors) = parts(&v);
        let f = Files::new();
        let obj = f.put("program.elf", &write_object(&p, &images, &externs));
        let all = f.put("all.safetensors", &write_weights(&p, &tensors).unwrap());
        let data = f.put("program-data.safetensors", &write_weights(&p, &tensors[..1]).unwrap());
        let weights = f.put("weights.safetensors", &write_weights(&p, &tensors[1..]).unwrap());
        let empty = f.put("empty.safetensors", &write_weights(&p, &[]).unwrap());
        let cfg = config_arg(profile);
        let (dargs, dlines) = dumps(&v);
        let mut args = vec!["run", "--config", &cfg, "--trace", "--lint", "--weights", &all];
        args.extend(strs(&dargs));
        args.push(&obj);
        let combined = run_iss(&args, "");
        assert_eq!(combined.status.code(), Some(0), "{}", text(&combined.stderr));
        for packages in [vec![&data, &weights], vec![&weights, &data], vec![&empty, &data, &weights]] {
            let mut args = vec!["run", "--config", &cfg, "--trace", "--lint"];
            for path in packages {
                args.extend(["--weights", path]);
            }
            args.extend(strs(&dargs));
            args.push(&obj);
            let split = run_iss(&args, "");
            assert_eq!(split.status.code(), Some(0), "{profile}: {}", text(&split.stderr));
            assert_eq!(split.stdout, combined.stdout, "{profile}: package order/partition changes results or trace");
        }
        // The independent executable oracle receives the same fully bound words
        // before cycle zero, with no change to a normative format or C behavior.
        let pk = read_weights(&std::fs::read(&all).unwrap(), Some(profile), None).unwrap();
        let stdin = format!("{}{}{dlines}", preload_text(&bind_weights(&externs, &pk).unwrap()), input_text(&images));
        let sim = run_sim(&[&cfg, "--trace", "--lint"], &stdin);
        assert_eq!(sim.status.code(), Some(0));
        assert_eq!(combined.stdout, sim.stdout, "{profile}: split-package binding versus C preload semantics");
        let got: Vec<_> = text(&combined.stdout).lines().filter(|l| l.starts_with("W ")).map(str::to_string).collect();
        assert_eq!(got, expected_w(&v, p.vlanes() as usize));
    }
}

#[test]
fn multiple_package_errors_are_transactional_and_manifest_mapping_stays_per_chip() {
    let p = Profile::load("mini").unwrap();
    let images = Images::new();
    let externs = [
        Extern { name: "x".into(), slice: 0, addr: 0, size: 1 },
        Extern { name: "y".into(), slice: 0, addr: 1, size: 1 },
    ];
    let tensor =
        |name: &str, n| TensorIn { name: name.into(), dtype: None, words: vec![vec![7; p.vlanes() as usize]; n] };
    let f = Files::new();
    let bytes = write_object(&p, &images, &externs);
    let obj = f.put("program.elf", &bytes);
    let x = f.put("x.safetensors", &write_weights(&p, &[tensor("x", 1)]).unwrap());
    let y = f.put("y.safetensors", &write_weights(&p, &[tensor("y", 1)]).unwrap());
    let short = f.put("short.safetensors", &write_weights(&p, &[tensor("y", 2)]).unwrap());
    let missing = f.put("missing.safetensors", &write_weights(&p, &[tensor("z", 1)]).unwrap());
    let corrupt = f.put("corrupt.safetensors", b"OTSW");
    for (second, message) in [
        (&x, "duplicate tensor x"),
        (&short, "extern y: tensor has 2 words"),
        (&missing, "extern y: no tensor"),
        (&corrupt, "OTS-E-ASM-003"),
    ] {
        let result = run_iss(&["run", "--profile", "mini", "--weights", &x, "--weights", second, &obj], "");
        assert_eq!(result.status.code(), Some(2));
        assert!(result.stdout.is_empty(), "no partial SRAM/result publication");
        assert!(text(&result.stderr).contains(message), "{}", text(&result.stderr));
    }
    let manifest = f.put("single.otsm", &write_manifest("mini", &[bytes], None).unwrap());
    let result = run_iss(&["run", "--weights", &x, "--weights", &y, &manifest], "");
    assert_eq!(result.status.code(), Some(2));
    assert!(text(&result.stderr).contains("manifest has 1 chips"), "{}", text(&result.stderr));
}

#[test]
fn a_multichip_manifest_without_packages_keeps_the_zero_filled_loader_path() {
    let p = Profile::load("mini").unwrap();
    let f = Files::new();
    let object = write_object(&p, &Images::new(), &[]);
    let m = f.put("empty.otsm", &write_manifest("mini", &[object.clone(), object], None).unwrap());
    let actual = run_iss(&["run", &m], "");
    assert_eq!(actual.status.code(), Some(0), "{}", text(&actual.stderr));
    let expected = run_sim(&[&config_arg("mini"), "--chips", "2"], "");
    assert_eq!(actual.stdout, expected.stdout);
}

#[test]
fn a_separate_program_data_package_can_supply_text_fetched_after_cycle_zero() {
    let p = Profile::load("mini").unwrap();
    let f = Files::new();
    let mut images = Images::new();
    images.extend(0, &[0x01c0_0010]); // IfetchM of the two SRAM text words at 0x10.
    let externs = [
        Extern { name: "text".into(), slice: 0, addr: 0x10, size: 2 },
        Extern { name: "data".into(), slice: 1, addr: 0x20, size: 1 },
    ];
    let obj = f.put("paged.elf", &write_object(&p, &images, &externs));
    let mut last = vec![0xff; 16];
    last[..8].copy_from_slice(&[0x12, 0, 0, 0x10, 0, 0, 0, 0x1c]);
    let text_tensor = TensorIn { name: "text".into(), dtype: None, words: vec![vec![0xff; 16], last] };
    let data_tensor = TensorIn { name: "data".into(), dtype: None, words: vec![vec![42; 16]] };
    let text_package =
        f.put("program-data.safetensors", &write_weights(&p, std::slice::from_ref(&text_tensor)).unwrap());
    let data_package = f.put("weights.safetensors", &write_weights(&p, std::slice::from_ref(&data_tensor)).unwrap());
    let combined = write_weights(&p, &[text_tensor, data_tensor]).unwrap();
    let tensors = read_weights(&combined, Some("mini"), None).unwrap();
    let stdin =
        format!("{}{}D 0 11\nD 1 20\n", preload_text(&bind_weights(&externs, &tensors).unwrap()), input_text(&images));
    let actual = run_iss(
        &[
            "run",
            "--weights",
            &text_package,
            "--weights",
            &data_package,
            "--trace",
            "--observe",
            "--dump",
            "0:11",
            "--dump",
            "1:20",
            &obj,
        ],
        "",
    );
    let reference = run_sim(&[&config_arg("mini"), "--trace", "--observe"], &stdin);
    assert_eq!(actual.status.code(), Some(0), "{}", text(&actual.stderr));
    assert_eq!(actual.stdout, reference.stdout);
    assert!(
        text(&actual.stdout).lines().any(|l| l.starts_with("O 0 1 0 fetch ") && l.ends_with(" 8")),
        "the text package must actually feed instruction fetch (eight 32-bit units): {}",
        text(&actual.stdout)
    );
}
