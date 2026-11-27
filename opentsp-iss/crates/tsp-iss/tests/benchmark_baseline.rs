//! The performance targets of the design (§4.10) on the reference host, release builds.
//!
//! A timing test with little headroom (1.1 to 1.7 x): it needs the **reference host** (the `tsp_sim` seconds below were
//! measured on it) and a load that does not move the figures, so it is `#[ignore]`d and is not part of `cargo test` or
//! `just test`. It runs as the second half of `just iss-perf-gate`, which prints the load average before it, and alone with
//! `just iss-bench` (`-- --ignored`).
//!
//! B4 is not held to an absolute rate (a cycles-per-second figure moves 2x with the load of the host and is not comparable
//! between program versions): it is held to the **stored baseline** (`benches/baseline.json`, written by
//! `just iss-perf-baseline` on a quiet host with its commit and load), whose `ns_per_element` and `ns_per_visited_tick` do not depend
//! on the idle cycles of a program (`ns_per_dispatch` says little for a looped program: B4's dispatches drive millions of elements). A real slowdown shows in the interleaved A/B of `just iss-perf-gate`; this test is the
//! absolute floor for a host like the one of the baseline. The file itself is checked by a test that always runs.
mod common;
use common::*;
use std::collections::BTreeMap;

/// `name key=value ...` lines of `bench` as name -> key -> value.
fn parse(out: &str) -> BTreeMap<String, BTreeMap<String, String>> {
    out.lines()
        .map(|l| {
            let mut f = l.split_whitespace();
            let name = f.next().unwrap().to_string();
            let kv = f.map(|p| p.split_once('=').unwrap_or_else(|| panic!("{l}"))).map(|(k, v)| (k.into(), v.into()));
            (name, kv.collect())
        })
        .collect()
}

fn num(r: &BTreeMap<String, String>, k: &str) -> f64 {
    r[k].parse().unwrap_or_else(|_| panic!("{k} in {r:?}"))
}

/// The targets on the reference host, measured against `tsp_sim`. Seconds of
/// `tsp_sim` on the reference host (an Apple Silicon Mac) divided by 1000 are the time limits of
/// B1..B3; B4 needs 50 000 simulated cycles per second and 10^8 idle cycles take under a second.
/// A debug build only checks that the programs run.
#[test]
#[ignore = "timing: run on a quiet reference host with just iss-bench"]
fn reference_host_benchmark_targets() {
    let o = run_iss(&["bench", "--repeat", "3", "--elements"], "");
    assert_eq!(o.status.code(), Some(0), "{}", text(&o.stderr));
    let t = parse(&text(&o.stdout));
    if cfg!(debug_assertions) {
        eprintln!("reference_host_benchmark_targets: debug build, the targets are checked in release builds only");
        return;
    }
    let mut missed = Vec::new();
    for (name, sim_seconds) in SIM_SECONDS {
        let limit = sim_seconds / 1000.0;
        let s = num(&t[name], "seconds");
        if s > limit {
            missed.push(format!("{name}: {s:.4} s, limit {limit:.4} s ({} x tsp_sim)", sim_seconds / s));
        }
    }
    let base = baseline();
    for key in ["ns_per_element", "ns_per_visited_tick"] {
        let (now, then) = (num(&t["b4"], key), base["programs"]["b4"][key].as_f64().unwrap());
        if now > then * B4_SLACK {
            missed.push(format!("b4: {key} {now:.1} ns, the baseline {then:.1} ns (limit {B4_SLACK} x)"));
        }
    }
    let idle = num(&t["idle"], "seconds");
    if idle >= 1.0 {
        missed.push(format!("idle: {idle:.3} s, target < 1 s"));
    }
    assert!(missed.is_empty(), "ISS-7 targets missed: {missed:#?}");
}

/// How much slower than the stored baseline B4 may be before the test fails: the noise of a quiet shared host.
const B4_SLACK: f64 = 1.5;

fn baseline() -> serde_json::Value {
    let p = root().join("opentsp-iss/crates/tsp-iss/benches/baseline.json");
    serde_json::from_slice(&std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())))
        .expect("baseline.json")
}

/// The stored baseline is there and says where it came from: a commit, the load of the host, and per program the rates that
/// do not depend on the idle cycles (it runs in every `cargo test`; the timing test that uses it is `#[ignore]`d).
#[test]
fn the_stored_baseline_is_complete() {
    let b = baseline();
    assert!(
        b["commit"].as_str().is_some_and(|c| c.len() == 40 && c.bytes().all(|x| x.is_ascii_hexdigit())),
        "{}",
        b["commit"]
    );
    assert!(b["load_average"].as_array().is_some_and(|l| l.len() == 3 && l.iter().all(|x| x.is_f64())));
    assert!(b["host"].is_string() && b["command"].is_string());
    for name in ["b1", "b2", "b3", "b4", "dense_f16", "dense_i8", "vxm_valid", "transc", "syncnotify"] {
        let p = &b["programs"][name];
        for key in ["seconds", "ticks", "dispatches", "ns_per_dispatch", "ns_per_visited_tick"] {
            assert!(p[key].as_f64().is_some_and(|x| x > 0.0), "{name}.{key} in the baseline: {p}");
        }
        // B2 raises a diagnostic at every cycle and issues nothing: it has no executed elements to divide by
        if name != "b2" {
            assert!(
                p["ns_per_element"].as_f64().is_some_and(|x| x > 0.0),
                "{name}.ns_per_element in the baseline: {p}"
            );
        }
    }
}

/// Seconds of `tsp_sim` on the reference host, release build, `ref` profile, no trace.
const SIM_SECONDS: [(&str, f64); 3] = [("b1", 3.41), ("b2", 16.48), ("b3", 5.89)];
