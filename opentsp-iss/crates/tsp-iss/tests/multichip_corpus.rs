//! The multi-chip regression corpus (`tests/data/multichip/`, ISS side): inputs with
//! K = 2 to 16 chips that send vectors over the links, the link options, and `test/link_tight`,
//! each with the stdout and exit code `tsp_sim` gave. The ISS must give both again, byte for byte,
//! without `tsp_sim`; `data/multichip/gen_corpus.py` regenerates the expectations from it.
mod common;
use common::*;
use std::path::PathBuf;

struct Case {
    name: String,
    profile: String,
    args: Vec<String>,
    exit: i32,
    input: String,
    want: Vec<u8>,
}

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/multichip")
}

fn cases() -> Vec<Case> {
    let index = std::fs::read_to_string(dir().join("cases.tsv")).unwrap();
    index
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f.len(), 4, "cases.tsv: {l:?}");
            Case {
                name: f[0].to_string(),
                profile: f[1].to_string(),
                args: f[2].split_whitespace().map(str::to_string).collect(),
                exit: f[3].parse().unwrap(),
                input: std::fs::read_to_string(dir().join(format!("{}.in", f[0]))).unwrap(),
                want: std::fs::read(dir().join(format!("{}.out", f[0]))).unwrap(),
            }
        })
        .collect()
}

fn chips(c: &Case) -> usize {
    c.args.iter().position(|a| a == "--chips").map_or(1, |p| c.args[p + 1].parse().unwrap())
}

/// The first line on which two outputs differ.
fn first_difference(got: &str, want: &str) -> String {
    let (g, w): (Vec<&str>, Vec<&str>) = (got.lines().collect(), want.lines().collect());
    match (0..g.len().max(w.len())).find(|&i| g.get(i) != w.get(i)) {
        Some(i) => format!(
            "line {}:\n  iss:      {:.200}\n  expected: {:.200}",
            i + 1,
            g.get(i).unwrap_or(&"<end>"),
            w.get(i).unwrap_or(&"<end>")
        ),
        None => "identical lines".into(),
    }
}

#[test]
fn the_iss_reproduces_every_recorded_multi_chip_run() {
    let cases = cases();
    let bad = parallel(&cases, |c| {
        let cfg = config(&c.profile);
        let mut args: Vec<&str> = vec![cfg.to_str().unwrap(), "--trace"];
        args.extend(c.args.iter().map(String::as_str));
        let out = run_iss(&args, &c.input);
        if out.status.code() != Some(c.exit) {
            return Err(format!(
                "{}: exit {:?}, expected {}\n{}",
                c.name,
                out.status.code(),
                c.exit,
                text(&out.stderr)
            ));
        }
        if out.stdout != c.want {
            return Err(format!(
                "{}: stdout differs at {}",
                c.name,
                first_difference(&text(&out.stdout), &text(&c.want))
            ));
        }
        Ok(())
    });
    assert!(bad.is_empty(), "{} of {} cases differ:\n{}", bad.len(), cases.len(), bad.join("\n"));
}

/// The corpus must keep exercising links: every run that ends normally has stream writes after the
/// shortest link latency (a vector arrived and was received), and the set covers the chip counts and
/// options of the design.
#[test]
fn the_corpus_covers_the_chip_counts_and_link_options_with_real_traffic() {
    let cases = cases();
    assert!(cases.len() >= 20, "{} cases", cases.len());
    for k in [2, 8, 16] {
        assert!(cases.iter().filter(|c| chips(c) == k).count() >= 3, "K = {k}");
    }
    for option in ["--link-latency", "--link-latency-seed", "--link-jitter-seed", "--max-cycles"] {
        assert!(cases.iter().any(|c| c.args.iter().any(|a| a == option)), "{option} is not exercised");
    }
    assert!(cases.iter().any(|c| c.profile == "test/link_tight"), "link_tight is not exercised");
    for c in cases.iter().filter(|c| c.exit == 0) {
        let late_writes = text(&c.want)
            .lines()
            .filter(|l| {
                l.starts_with("T srf ")
                    && l.split(' ').nth(2).and_then(|n| n.parse::<u64>().ok()).is_some_and(|n| n >= 209)
            })
            .count();
        assert!(late_writes >= 2, "{}: {late_writes} stream writes after cycle 209: no link traffic", c.name);
        let has_tx = c.input.lines().any(|l| ["Q 7e ", "Q 7f ", "Q 8c ", "Q 8d "].iter().any(|q| l.starts_with(q)));
        assert!(has_tx, "{}: no C2C TX queue", c.name);
    }
    let all = cases.iter().map(|c| text(&c.want)).collect::<String>();
    for code in ["OTS-E-LNK-001", "OTS-E-LNK-002", "OTS-E-TIM-001"] {
        assert!(all.contains(code), "no case reports {code}");
    }
    let bytes: usize = cases.iter().map(|c| c.input.len() + c.want.len()).sum();
    assert!(bytes < 400_000, "the corpus has grown to {bytes} bytes");
}
