//! `tsp-iss vectors` checks what `tests/conformance.rs` checks: the diagnostics, the words,
//! the `events`, the `trace_sha256` of every vector and the `config_sha256` of the file. A copy of
//! `ref.json` with one value changed fails exactly the vector it changed, or is refused as a file.
mod common;
use common::*;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const CHECKED: usize = 8;

fn golden(profile: &str) -> Value {
    let path = spec().join("conformance/model").join(format!("{profile}.json"));
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn name(doc: &Value, i: usize) -> String {
    doc["vectors"][i]["name"].as_str().unwrap().to_string()
}

/// `vectors` on `doc` restricted to its first `CHECKED` vectors: `(exit code, stdout, stderr)`.
fn vectors(doc: &Value, profile: &str) -> (Option<i32>, String, String) {
    vectors_with(doc, &config(profile))
}

fn vectors_with(doc: &Value, cfg: &Path) -> (Option<i32>, String, String) {
    let dir = tempfile::tempdir().unwrap();
    let file: PathBuf = dir.path().join("v.json");
    std::fs::write(&file, serde_json::to_vec(doc).unwrap()).unwrap();
    let mut args = vec!["vectors", file.to_str().unwrap(), "--config", cfg.to_str().unwrap()];
    let names: Vec<String> = (0..CHECKED).map(|i| name(doc, i)).collect();
    for n in &names {
        args.extend(["--only", n]);
    }
    let o = run_iss(&args, "");
    (o.status.code(), text(&o.stdout), text(&o.stderr))
}

fn fails(stdout: &str) -> Vec<&str> {
    stdout.lines().filter(|l| l.starts_with("FAIL ")).collect()
}

#[test]
fn the_untouched_file_passes() {
    for profile in ["ref", "mini"] {
        let doc = golden(profile);
        let (code, out, err) = vectors(&doc, profile);
        assert_eq!((code, err.as_str()), (Some(0), ""), "{profile}: {out}");
        assert_eq!(out.lines().last().unwrap(), format!("{CHECKED} vectors: {CHECKED} pass, 0 FAIL"));
    }
}

#[test]
fn a_changed_trace_sha256_fails_that_vector() {
    let mut doc = golden("ref");
    doc["vectors"][3]["expect"]["trace_sha256"] = json!("0".repeat(64));
    let (code, out, _) = vectors(&doc, "ref");
    assert_eq!(code, Some(1), "{out}");
    let f = fails(&out);
    assert_eq!(f.len(), 1, "{out}");
    assert!(f[0].starts_with(&format!("FAIL {}: trace_sha256", name(&doc, 3))), "{}", f[0]);
    assert_eq!(out.lines().last().unwrap(), format!("{CHECKED} vectors: {} pass, 1 FAIL", CHECKED - 1));
}

#[test]
fn a_missing_trace_sha256_fails_that_vector() {
    let mut doc = golden("ref");
    doc["vectors"][2]["expect"].as_object_mut().unwrap().remove("trace_sha256");
    let (code, out, _) = vectors(&doc, "ref");
    assert_eq!(code, Some(1), "{out}");
    let f = fails(&out);
    assert!(f.len() == 1 && f[0].starts_with(&format!("FAIL {}: trace_sha256", name(&doc, 2))), "{out}");
}

#[test]
fn a_changed_event_fails_that_vector() {
    let mut doc = golden("ref");
    let events = doc["vectors"][5]["expect"]["events"].as_array_mut().unwrap();
    assert!(!events.is_empty());
    events[0] = json!("srf 0 E:0 gap 0 00");
    let (code, out, _) = vectors(&doc, "ref");
    assert_eq!(code, Some(1), "{out}");
    let f = fails(&out);
    assert_eq!(f.len(), 1, "{out}");
    assert!(f[0].starts_with(&format!("FAIL {}: events", name(&doc, 5))), "{}", f[0]);
}

#[test]
fn all_three_changes_at_once_fail_the_vector_and_refuse_the_file() {
    let mut doc = golden("ref");
    doc["vectors"][0]["expect"]["trace_sha256"] = json!("0".repeat(64));
    doc["vectors"][0]["expect"]["events"][0] = json!("srf 0 E:0 gap 0 00");
    doc["config_sha256"] = json!("f".repeat(64));
    let (code, out, err) = vectors(&doc, "ref");
    assert_eq!(code, Some(2), "{out}");
    assert!(out.is_empty(), "{out}");
    assert!(err.starts_with("tsp-iss: vectors: ") && err.contains("config_sha256"), "{err}");
}

#[test]
fn a_file_of_another_configuration_is_refused() {
    for bad in [json!("f".repeat(64)), json!(null)] {
        let mut doc = golden("ref");
        doc["config_sha256"] = bad;
        let (code, out, err) = vectors(&doc, "ref");
        assert_eq!(code, Some(2), "{out}");
        assert!(out.is_empty() && err.contains("config_sha256"), "{out}{err}");
    }
    // the digest of `mini` is not the digest of `ref`
    let mut doc = golden("ref");
    doc["config_sha256"] = golden("mini")["config_sha256"].clone();
    let (code, _, err) = vectors(&doc, "ref");
    assert_eq!(code, Some(2), "{err}");
}

#[test]
fn the_digest_is_of_the_canonical_json_of_the_configuration_given() {
    let doc = golden("ref");
    let cfg: Value = serde_json::from_slice(&std::fs::read(config("ref")).unwrap()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    // the same configuration, written otherwise: accepted
    let reformatted = dir.path().join("ref.json");
    std::fs::write(&reformatted, serde_json::to_vec(&cfg).unwrap()).unwrap();
    let (code, out, err) = vectors_with(&doc, &reformatted);
    assert_eq!((code, err.as_str()), (Some(0), ""), "{out}");
    // another configuration: the file is refused before any vector runs
    let (code, out, err) = vectors_with(&doc, &config("mini"));
    assert_eq!(code, Some(2), "{out}");
    assert!(out.is_empty() && err.contains("config_sha256"), "{out}{err}");
}
