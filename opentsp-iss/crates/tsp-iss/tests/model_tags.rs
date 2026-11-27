//! Traceability check: every `// model: <path> <name>` tag in
//! `opentsp-iss/` and `opentsp-sw-toolchain/` names a file that exists and a name that file defines.
//! `<path>` is relative to `opentsp-arch-spec/model/`; a path starting with `c_emulator/` is
//! relative to `opentsp-arch-spec/` and names the C harness instead.
//! A tag may list several names separated by `, `; the keywords `struct`, `enum`, `val`, ... before
//! the name are skipped, and anything after the name (a clause name, a note) is not checked.
mod common;

use std::path::{Path, PathBuf};

/// A name counts as defined when it occurs in the file outside `//` comments, as a whole word.
fn defines(source: &str, name: &str) -> bool {
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    source.lines().any(|line| {
        let code = line.split("//").next().unwrap_or("");
        code.match_indices(name).any(|(at, _)| {
            let before = code[..at].chars().next_back().is_some_and(is_word);
            let after = code[at + name.len()..].chars().next().is_some_and(is_word);
            !before && !after
        })
    })
}

const KEYWORDS: [&str; 11] =
    ["struct", "enum", "union", "val", "function", "clause", "mapping", "type", "let", "register", "overload"];

/// The `(path, names)` of the tag on `line`, if it carries one: `model: ` after `// `, or followed
/// by a `.sail` or `.c` path (a tag inside a parenthesis of a longer comment).
fn parse_tag(line: &str) -> Option<(String, Vec<String>)> {
    let at = line.find("model: ")?;
    let mut tokens = line[at + "model: ".len()..].split_whitespace();
    let path = tokens.next()?.trim_end_matches([',', ')']).to_string();
    if !line[..at].ends_with("// ") && !path.ends_with(".sail") && !path.ends_with(".c") {
        return None;
    }
    let tokens = tokens.skip_while(|t| KEYWORDS.contains(t));
    let mut names = Vec::new();
    for token in tokens {
        let name: String = token.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
        names.push(name);
        if !token.ends_with(',') {
            break;
        }
    }
    Some((path, names))
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            if path.file_name().unwrap() != "target" {
                rust_files(&path, out);
            }
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// What is wrong with the tag `<path> <names>`, one message per problem.
fn problems(path: &str, names: &[String]) -> Vec<String> {
    let spec = common::spec();
    let target = if path.starts_with("c_emulator/") { spec.join(path) } else { spec.join("model").join(path) };
    let Ok(source) = std::fs::read_to_string(&target) else {
        return vec![format!("`{path}` is not a file under opentsp-arch-spec")];
    };
    let mut bad = Vec::new();
    if names.is_empty() {
        bad.push(format!("the tag names no function after `{path}`"));
    }
    bad.extend(
        names.iter().filter(|name| !defines(&source, name)).map(|name| format!("`{path}` does not define `{name}`")),
    );
    bad
}

fn check_tags(files: &[PathBuf]) -> (usize, Vec<String>) {
    let mut tags = 0;
    let mut bad = Vec::new();
    for file in files {
        if file.ends_with("tests/model_tags.rs") {
            continue;
        }
        let text = std::fs::read_to_string(file).unwrap();
        for (n, line) in text.lines().enumerate() {
            let Some((path, names)) = parse_tag(line) else { continue };
            tags += 1;
            let at = format!("{}:{}", file.strip_prefix(common::root()).unwrap().display(), n + 1);
            bad.extend(problems(&path, &names).into_iter().map(|p| format!("{at}: {p}")));
        }
    }
    (tags, bad)
}

fn all_rust_files() -> Vec<PathBuf> {
    let mut files = Vec::new();
    rust_files(&common::root().join("opentsp-iss"), &mut files);
    rust_files(&common::root().join("opentsp-sw-toolchain"), &mut files);
    files.sort();
    files
}

#[test]
fn every_model_tag_names_a_defined_function() {
    let (tags, bad) = check_tags(&all_rust_files());
    assert!(tags > 300, "found only {tags} tags: the scan is broken");
    assert!(bad.is_empty(), "{} of {tags} tags do not resolve:\n{}", bad.len(), bad.join("\n"));
}

#[test]
fn the_checker_rejects_unresolved_tags() {
    let verdicts: Vec<bool> = [
        "// model: fp.sail fp_add",
        "// a note (model: fp.sail fp_add)",
        "// model: core/numerics/fp.sail no_such_function_anywhere",
        "// model: core/numerics/fp.sail fp_add, fp_nope",
        "// model: c_emulator/tsp_sim.c no_such_c_function",
        "// model: core/numerics/fp.sail",
        "// model: c_emulator/tsp_sim.c main",
        "// model: core/numerics/fp.sail struct fval",
        "// model: core/numerics/fp.sail fp_add, fp_sub (two names)",
    ]
    .iter()
    .map(|line| {
        let (path, names) = parse_tag(line).unwrap();
        problems(&path, &names).is_empty()
    })
    .collect();
    assert_eq!(verdicts, [false, false, false, false, false, false, true, true, true]);
}
