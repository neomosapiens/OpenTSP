//! `tsp-iss run` of an `.otsm` manifest (`assembly.md` §4) equals the compatible mode on the
//! `ots-asm mobjinput` text: for every manifest of `ots-asm/tests/data` (stdout and exit code, with and
//! without the weight packages, with `--trace`), and one of them on `tsp_sim`. A bad manifest gives
//! the reason `ots-asm mobjinput` gives.
mod common;
use common::*;
use std::path::{Path, PathBuf};

/// `(manifest, weight packages in chip order)` of the fixtures (the objects of `ots-asm`'s `SETS`).
const SETS: [(&str, &[&str]); 4] = [
    ("ref1", &["example.safetensors"]),
    ("ref2", &["example.safetensors", "rich.safetensors"]),
    ("mini3", &["example_mini.safetensors", "rich_mini.safetensors", "example_mini.safetensors"]),
    (
        "mini16",
        &[
            "example_mini.safetensors",
            "rich_mini.safetensors",
            "example_mini.safetensors",
            "rich_mini.safetensors",
            "example_mini.safetensors",
            "rich_mini.safetensors",
            "example_mini.safetensors",
            "rich_mini.safetensors",
            "example_mini.safetensors",
            "rich_mini.safetensors",
            "example_mini.safetensors",
            "rich_mini.safetensors",
            "example_mini.safetensors",
            "rich_mini.safetensors",
            "example_mini.safetensors",
            "rich_mini.safetensors",
        ],
    ),
];

fn data(name: &str) -> PathBuf {
    root().join("opentsp-sw-toolchain/crates/ots-asm/tests/data").join(name)
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn profile_of(manifest: &[u8]) -> String {
    String::from_utf8(manifest[8..24].iter().copied().take_while(|&b| b != 0).collect()).unwrap()
}

/// What `ots-asm mobjinput M [--weights W ...]` prints: the checked-in text, which `ots-asm`'s own
/// tests pin to its output, and `ots_format::manifest_input`, the function it calls.
fn mobjinput(name: &str, weights: &[&str]) -> String {
    let m = std::fs::read(data(&format!("{name}.otsm"))).unwrap();
    let pk: Vec<Vec<u8>> = weights.iter().map(|w| std::fs::read(data(w)).unwrap()).collect();
    let text = ots_format::manifest_input(&m, &pk, None).unwrap();
    let file = if weights.is_empty() { format!("{name}.mobjinput") } else { format!("{name}.weights.mobjinput") };
    assert_eq!(text.as_bytes(), std::fs::read(data(&file)).unwrap().as_slice(), "{name}");
    text
}

fn chips(text: &str) -> String {
    text.lines().filter(|l| l.starts_with("C ")).count().to_string()
}

fn run_manifest(name: &str, weights: &[&str], extra: &[&str]) -> std::process::Output {
    let m = data(&format!("{name}.otsm"));
    let w: Vec<PathBuf> = weights.iter().map(|w| data(w)).collect();
    let mut args = vec!["run"];
    args.extend(extra);
    for p in &w {
        args.extend(["--weights", s(p)]);
    }
    args.push(s(&m));
    run_iss(&args, "")
}

#[test]
fn run_otsm_equals_the_compat_mode_on_the_mobjinput_text() {
    for (name, weights) in SETS {
        let m = std::fs::read(data(&format!("{name}.otsm"))).unwrap();
        let cfg = config(&profile_of(&m));
        let cfg = s(&cfg);
        for (w, trace) in [(&[][..], false), (weights, false), (weights, true)] {
            let stdin = mobjinput(name, w);
            let k = chips(&stdin);
            let mut compat_args = vec![cfg, "--chips", &k];
            let mut run_args = vec!["--config", cfg];
            if trace {
                compat_args.push("--trace");
                run_args.push("--trace");
            }
            let compat = run_iss(&compat_args, &stdin);
            let ours = run_manifest(name, w, &run_args);
            let what = format!("{name} weights={} trace={trace}", w.len());
            assert_eq!(ours.status.code(), compat.status.code(), "{what}: {}", text(&ours.stderr));
            assert_eq!(text(&ours.stdout), text(&compat.stdout), "{what}");
            assert!(text(&ours.stdout).contains("\nE ") || text(&ours.stdout).starts_with("E "), "{what}: no E line");
        }
    }
}

#[test]
fn run_otsm_takes_its_profile_from_the_manifest() {
    let ours = run_manifest("ref2", &["example.safetensors", "rich.safetensors"], &[]);
    assert_eq!(ours.status.code(), Some(0), "{}", text(&ours.stderr));
    let cfg = config("ref");
    let compat = run_iss(&[s(&cfg), "--chips", "2"], &mobjinput("ref2", &["example.safetensors", "rich.safetensors"]));
    assert_eq!(text(&ours.stdout), text(&compat.stdout));
}

#[test]
fn run_otsm_agrees_with_tsp_sim() {
    let stdin = mobjinput("ref2", &["example.safetensors", "rich.safetensors"]);
    let cfg = config("ref");
    let ours = run_manifest("ref2", &["example.safetensors", "rich.safetensors"], &["--config", s(&cfg), "--trace"]);
    let sim = run_sim(&[s(&cfg), "--chips", "2", "--trace"], &stdin);
    assert_eq!(ours.status.code(), sim.status.code(), "{}", text(&ours.stderr));
    assert_eq!(text(&ours.stdout), text(&sim.stdout));
}

#[test]
fn run_otsm_chips_must_match_the_manifest() {
    let o = run_manifest("ref2", &[], &["--chips", "2"]);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o.stderr));
    let o = run_manifest("ref2", &[], &["--chips", "3"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(o.stdout.is_empty());
    let e = text(&o.stderr);
    assert!(e.starts_with("tsp-iss: run: ") && e.contains("--chips 3") && e.contains("2 chips"), "{e}");
}

#[test]
fn run_otsm_weight_packages_are_one_per_chip_in_chip_order() {
    let manifest = std::fs::read(data("ref2.otsm")).unwrap();
    for weights in [
        &["example.safetensors"][..],
        &["rich.safetensors", "example.safetensors"],
        &["example.safetensors", "rich_mini.safetensors"],
    ] {
        let pk: Vec<Vec<u8>> = weights.iter().map(|w| std::fs::read(data(w)).unwrap()).collect();
        let e = ots_format::manifest_input(&manifest, &pk, None).unwrap_err();
        let o = run_manifest("ref2", weights, &[]);
        assert_eq!(o.status.code(), Some(2), "{weights:?}");
        assert!(o.stdout.is_empty(), "{weights:?}");
        let got = text(&o.stderr);
        assert!(got.starts_with("tsp-iss: run: ") && got.contains(e.code.as_str()), "{weights:?}: {got}");
    }
}

#[test]
fn run_otsm_of_a_mutated_manifest_gives_the_reason_of_mobjinput() {
    let good = std::fs::read(data("ref2.otsm")).unwrap();
    let f = Files::new();
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("topology", {
            let mut b = good.clone();
            b[28] = 1;
            b
        }),
        ("encoding digest", {
            let mut b = good.clone();
            b[40] ^= 1;
            b
        }),
        ("chip count", {
            let mut b = good.clone();
            b[24] = 0;
            b
        }),
        ("truncated", good[..good.len() - 1].to_vec()),
        ("trailing byte", [good.as_slice(), &[0]].concat()),
        ("entry offset", {
            let mut b = good.clone();
            b[64] ^= 1;
            b
        }),
    ];
    for (what, bytes) in cases {
        let want = ots_format::manifest_input(&bytes, &[], None).unwrap_err();
        assert_eq!(want.code.as_str(), "OTS-E-ASM-003", "{what}");
        let p = f.put("bad.otsm", &bytes);
        let cfg = config("ref");
        let o = run_iss(&["run", "--config", s(&cfg), &p], "");
        assert_eq!(o.status.code(), Some(2), "{what}");
        assert!(o.stdout.is_empty(), "{what}");
        let e = text(&o.stderr);
        assert!(
            e.starts_with("tsp-iss: run: ") && e.trim_end().ends_with(&want.to_string()),
            "{what}: got {e:?}, want ...{want}"
        );
    }
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
}
