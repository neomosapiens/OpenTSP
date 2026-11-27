//! The golden vectors of `opentsp-arch-spec/conformance/model/{ref,mini}.json`, read live from
//! the specification tree and compared exactly by the library code `tsp-iss vectors` runs
//! (`tsp_iss::native::check_vector`: diagnostics in order, dumped words, and the trace:
//! `trace_sha256` of the canonical `T srf`/`T sram` lines with their values and `events` when
//! present; `check_config_digest`: each file's `config_sha256` against the profile): every vector
//! of both profiles.
//!
//! Each vector is one chip with every link looped back at the nominal latency, as `tsp_sim` ran it.
mod common;
use common::*;
use ots_profile::Profile;
use serde_json::Value;
use tsp_iss::native::{check_config_digest, check_vector};
use tsp_iss::Geom;

/// The minimum corpus size guards against accidental truncation.
const MIN_VECTORS: usize = 500;

fn check_profile(name: &str) {
    let path = spec().join("conformance/model").join(format!("{name}.json"));
    let doc: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(doc["profile"], name);
    let profile = Profile::load(name).unwrap();
    check_config_digest(&doc, &profile).unwrap_or_else(|e| panic!("{name}: {e}"));
    let geom = Geom::new(&profile);
    let (mut ran, mut failed) = (0, Vec::new());
    for v in doc["vectors"].as_array().unwrap() {
        ran += 1;
        let id = format!("{name}/{}", v["name"].as_str().unwrap());
        if let Err(why) = check_vector(&profile, &geom, v) {
            failed.push(format!("{id}: {why}"));
        }
    }
    eprintln!("conformance {name}: {ran} vectors, {} failed", failed.len());
    assert!(ran >= MIN_VECTORS, "{name}: only {ran} vectors");
    assert!(failed.is_empty(), "{} vectors of {name} fail:\n{}", failed.len(), failed.join("\n"));
}

#[test]
fn conformance_ref() {
    check_profile("ref");
}

#[test]
fn conformance_mini() {
    check_profile("mini");
}
