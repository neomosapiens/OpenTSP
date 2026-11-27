//! The shared negative corpus of `tests/spec/config_corpus/`: `tsp-iss` rejects every file
//! with exit 2 and a configuration message, and still accepts the shipped profiles. The same corpus
//! drives the C harness in `tests/spec/test_harness_contract.py`.
mod common;
use common::*;
use std::process::{Command, Stdio};

#[test]
fn every_corpus_configuration_is_rejected() {
    let dir = root().join("tests/spec/config_corpus");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    assert!(files.len() >= 12, "corpus shrank: {files:?}");
    for f in files {
        let out = Command::new(iss()).arg(&f).stdin(Stdio::null()).output().unwrap();
        let err = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{} accepted: {err}", f.display());
        assert!(err.contains("config"), "{}: {err}", f.display());
    }
}

#[test]
fn the_shipped_profiles_are_accepted() {
    for name in ["ref", "mini"] {
        let out = Command::new(iss()).arg(config(name)).stdin(Stdio::null()).output().unwrap();
        assert_eq!(out.status.code(), Some(0), "{name}: {}", String::from_utf8_lossy(&out.stderr));
    }
}

#[test]
fn custom_causal_boundaries_agree_across_loaders() {
    use ots_profile::Profile;
    use serde_json::json;
    use tsp_iss::{platform::Platform, Machine, NoLinks};
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("custom.json");
    for field in ["Shift", "Permute", "ABC", "ABC8", "map", "fetch"] {
        for valid in [false, true] {
            let mut p = Profile::load("ref").unwrap().value().clone();
            match field {
                "fetch" => {
                    p["icu"]["fetch_bytes_per_cycle"] = json!(64);
                    p["icu"]["fetch_latency"] = json!(14 + u8::from(valid));
                }
                "map" => p["timing"]["sxm_map_ready"] = json!(19 + u8::from(valid)),
                name => p["timing"]["overrides"] = json!([{"entry":name,"d_skew":15,"d_func":34 + u8::from(valid)}]),
            }
            std::fs::write(&cfg, p.to_string()).unwrap();
            let p = Profile::load_file("ref", &cfg).unwrap();
            let arg = cfg.to_str().unwrap();
            assert_eq!(run_sim(&[arg], "").status.success(), valid, "Sail {field}");
            assert_eq!(run_iss(&[arg], "").status.success(), valid, "ISS CLI {field}");
            assert_eq!(ots_format::load_profile("ref", Some(arg)).is_ok(), valid, "assembler {field}");
            assert_eq!(Machine::new(&p, 0, Box::new(NoLinks)).is_ok(), valid, "machine {field}");
            assert_eq!(Platform::new(&p, 0).is_ok(), valid, "platform {field}");
        }
    }
}
