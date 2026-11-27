//! Operand ECC events at cycles at or above 2^63: the cycle of `T ecc`, `T irq`, the chip log and `ERR_FIRST` is the unsigned 64-bit
//! cycle of the read, not a wrapped one.
//!
//! The golden vectors of the `ref` profile with `H start` at a high cycle and every host line's cycle shifted by it, as the
//! differential `iss_vecmut.py --starts` makes them. The first test pins the shipped vector `ecc-009` against the lines `tsp_sim`
//! prints for it; the second compares every `ecc-*` vector with `tsp_sim` at three start cycles.
mod common;
use common::*;

const M64: u128 = (1 << 64) - 1;

/// The harness input of a golden vector with `H start offset` and every `H`, `F` and `P` host line moved by `offset`.
fn shifted(v: &serde_json::Value, offset: u64) -> String {
    let mut out = Vec::new();
    for (k, w) in v["sram"].as_object().unwrap() {
        let (s, a) = k.split_once(':').unwrap();
        let w = w.as_str().unwrap();
        let bytes: Vec<&str> = (0..w.len() / 2).map(|i| &w[2 * i..2 * i + 2]).collect();
        out.push(format!("S {:x} {:x} {}", s.parse::<u32>().unwrap(), a.parse::<u32>().unwrap(), bytes.join(" ")));
    }
    for (q, u) in v["text"].as_object().unwrap() {
        if !u.as_str().unwrap().trim().is_empty() {
            out.push(format!("Q {:x} {}", q.parse::<u32>().unwrap(), u.as_str().unwrap()));
        }
    }
    for k in v["dump"].as_array().unwrap() {
        let (s, a) = k.as_str().unwrap().split_once(':').unwrap();
        out.push(format!("D {:x} {:x}", s.parse::<u32>().unwrap(), a.parse::<u32>().unwrap()));
    }
    let mut host = Vec::new();
    let mut has_start = false;
    for h in v.get("host").and_then(|h| h.as_array()).into_iter().flatten() {
        let mut p: Vec<String> = h.as_str().unwrap().split_whitespace().map(String::from).collect();
        if matches!(p[0].as_str(), "H" | "F" | "P") && p.len() > 2 {
            if p[0] == "H" && p[1] == "start" {
                has_start = true;
            }
            let c = u128::from_str_radix(&p[2], 16).unwrap();
            p[2] = format!("{:x}", (c + u128::from(offset)).min(M64));
        }
        host.push(p.join(" "));
    }
    if !has_start {
        host.insert(0, format!("H start {offset:x}"));
    }
    out.extend(host);
    out.join("\n") + "\n"
}

fn vectors(profile: &str) -> Vec<serde_json::Value> {
    let path = spec().join("conformance/model").join(format!("{profile}.json"));
    let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    j["vectors"].as_array().unwrap().clone()
}

const ARGS: [&str; 5] = ["--max-cycles", "18446744073709551615", "--trace", "--observe", "--lint"];

/// What `tsp_sim` prints for the shipped vector `ref/ecc-009` started at 2^63: the error event and its interrupt carry the cycle of
/// the read, 2^63 + 62.
#[test]
fn an_operand_ecc_event_at_cycle_2_pow_63_is_reported_at_that_cycle() {
    let v = vectors("ref").into_iter().find(|v| v["name"] == "ecc-009").expect("the vector ecc-009");
    let cfg = config("ref");
    let mut args = vec![cfg.to_str().unwrap()];
    args.extend(ARGS);
    let o = run_iss(&args, &shifted(&v, 1 << 63));
    assert_eq!(o.status.code(), Some(0), "{}", text(&o.stderr));
    let lines: Vec<String> = text(&o.stdout)
        .lines()
        .filter(|l| l.starts_with("T ecc") || l.starts_with("T irq"))
        .map(String::from)
        .collect();
    assert_eq!(
        lines,
        ["T ecc 9223372036854775870 queue 88 unc E:0 sl 0 pos 95", "T irq 9223372036854775870 msix 0"],
        "the lines of tsp_sim"
    );
}

#[test]
fn every_ecc_vector_matches_the_model_from_high_start_cycles() {
    let cfg = config("ref");
    let cfg = cfg.to_str().unwrap();
    let mut jobs = Vec::new();
    for v in vectors("ref").into_iter().filter(|v| v["name"].as_str().unwrap().starts_with("ecc")) {
        for off in [1u64 << 62, 1 << 63, u64::MAX - (1 << 20) + 1] {
            jobs.push((v.clone(), off));
        }
    }
    assert!(jobs.len() >= 400, "{} runs", jobs.len());
    let norm = |b: &[u8]| text(b).replace("tsp-iss", "P").replace("tsp_sim", "P");
    let bad = parallel(&jobs, |(v, off)| {
        let input = shifted(v, *off);
        let mut args = vec![cfg];
        args.extend(ARGS);
        let (o, s) = (run_iss(&args, &input), run_sim(&args, &input));
        if (o.status.code(), norm(&o.stdout), norm(&o.stderr)) != (s.status.code(), norm(&s.stdout), norm(&s.stderr)) {
            return Err(format!("{} at {off:#x}: exit {:?} against {:?}", v["name"], o.status.code(), s.status.code()));
        }
        Ok(())
    });
    assert!(
        bad.is_empty(),
        "{} of {} runs differ from tsp_sim:\n{:#?}",
        bad.len(),
        jobs.len(),
        &bad[..bad.len().min(10)]
    );
}
