//! Completeness of the ports of the Sail unit tests: every
//! `function test_*` of `opentsp-arch-spec/model/unit_tests/` is ported by a Rust file of
//! `opentsp-iss/` or `opentsp-sw-toolchain/`: a `fn` of the same name, or a `// model:` tag naming it, or sits on
//! the allow-list below with the reason it cannot be ported.
mod common;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Sail tests with no Rust port, and why.
const NOT_PORTABLE: &[(&str, &str)] = &[
    (
        "test_ecc_linear",
        "the ISS keeps no encoder: a word's check bits are the 10-bit overlay delta from the valid code, so the linearity of \
         the encoder is what the overlay representation assumes; the single and double flip tests of tests/ecc.rs and the \
         ecc vectors would fail if it did not hold",
    ),
];

fn sail_tests() -> BTreeSet<String> {
    let dir = common::spec().join("model/unit_tests");
    let mut names = BTreeSet::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "sail") {
            continue;
        }
        for line in std::fs::read_to_string(&path).unwrap().lines() {
            if let Some(rest) = line.strip_prefix("function test_") {
                names.insert(format!(
                    "test_{}",
                    rest.split(|c: char| !(c.is_alphanumeric() || c == '_')).next().unwrap()
                ));
            }
        }
    }
    names
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            if path.file_name().unwrap() != "target" {
                rust_files(&path, out);
            }
        } else if path.extension().is_some_and(|e| e == "rs") && !path.ends_with("tests/unit_test_ports.rs") {
            out.push(path);
        }
    }
}

/// The Sail test names a file ports: a `fn <name>` outside `//` comments, or a model tag naming
/// a unit test (older ports keep their own Rust names).
fn ported_names(text: &str) -> impl Iterator<Item = &str> {
    text.lines().flat_map(|line| {
        if let Some(tag) = line.trim_start().strip_prefix(concat!("//", " model: unit_tests/")) {
            let names = tag.split_once(' ').map_or("", |(_, n)| n);
            let names = names.split('(').next().unwrap_or("");
            return names.split(',').map(str::trim).collect::<Vec<_>>();
        }
        let code = line.split("//").next().unwrap();
        let rest = code.trim_start().strip_prefix("fn ").or_else(|| code.split(" fn ").nth(1));
        rest.and_then(|r| r.split(|c: char| !(c.is_alphanumeric() || c == '_')).next()).into_iter().collect()
    })
}

#[test]
fn every_sail_unit_test_is_ported_or_allow_listed() {
    let sail = sail_tests();
    assert!(sail.len() > 60, "found only {} Sail tests: the scan is broken", sail.len());
    let mut files = Vec::new();
    rust_files(&common::root().join("opentsp-iss"), &mut files);
    rust_files(&common::root().join("opentsp-sw-toolchain"), &mut files);
    let mut named = BTreeSet::new();
    for file in &files {
        for name in ported_names(&std::fs::read_to_string(file).unwrap()) {
            if sail.contains(name) {
                named.insert(name.to_string());
            }
        }
    }
    let allowed: BTreeSet<&str> = NOT_PORTABLE.iter().map(|(name, _)| *name).collect();
    let missing: Vec<&String> = sail.iter().filter(|t| !named.contains(*t) && !allowed.contains(t.as_str())).collect();
    assert!(
        missing.is_empty(),
        "{} Sail unit tests have no Rust port or allow-list entry:\n{}",
        missing.len(),
        missing.iter().map(|m| m.as_str()).collect::<Vec<_>>().join("\n")
    );
    let stale: Vec<&str> = allowed.iter().copied().filter(|a| !sail.contains(*a) || named.contains(*a)).collect();
    assert!(stale.is_empty(), "allow-list entries that are ported or no longer exist: {stale:?}");
    for (name, reason) in NOT_PORTABLE {
        assert!(!reason.is_empty(), "{name}: the allow-list needs a reason");
    }
}
