//! The run profile and the observability artifacts of `tsp-iss run` (`harness::obs`): opt-in, counted from the same
//! events as `--stats`, and never a change of the trace on stdout.
mod common;
use common::*;
use serde_json::Value;
use std::path::PathBuf;

fn program(name: &str) -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/programs").join(name).to_str().unwrap().to_string()
}

fn json(path: &std::path::Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap()
}

/// `run --profile ref` of a benchmark program with `extra` arguments.
fn run(prog: &str, extra: &[&str]) -> std::process::Output {
    let mut args = vec!["run", "--profile", "ref"];
    args.extend_from_slice(extra);
    args.push(prog);
    run_iss(&args, "")
}

#[test]
fn the_run_profile_counts_what_stats_counts() {
    let d = tempfile::tempdir().unwrap();
    let (stats, prof) = (d.path().join("stats.json"), d.path().join("profile.json"));
    let o = run(
        &program("b3.txt"),
        &[&format!("--stats={}", stats.display()), "--run-profile-json", prof.to_str().unwrap()],
    );
    assert_eq!(o.status.code(), Some(0), "{}", text(&o.stderr));
    let (s, p) = (json(&stats), json(&prof));
    assert_eq!(p["schema"], 1);
    assert_eq!(p["profile"], "ref");
    assert!(p["wall_ns"].as_u64().unwrap() > 0);
    assert!(p["clock"]["step_ns"].as_u64().unwrap() > 0 && p["clock"]["read_ns"].as_f64().unwrap() > 0.0);
    for k in ["read_program_ns", "decode_program_ns", "read_weights_file_ns", "verify_weights_ns", "bind_weights_ns"] {
        assert!(p["load"][k].is_u64(), "load.{k}");
    }
    let c = &p["chips"][0];
    assert_eq!(p["chips"].as_array().unwrap().len(), 1);
    assert_eq!(c["ticks"], s["ticks"]);
    assert_eq!(c["active_cycles"], s["ticks"]);
    assert_eq!(c["dispatches"], s["dispatches"]);
    assert_eq!(c["last_cycle"], s["last_cycle"]);
    let cycles = s["last_cycle"].as_u64().unwrap() + 1;
    assert_eq!(c["active_cycles"].as_u64().unwrap() + c["idle_cycles"].as_u64().unwrap(), cycles);
    // every phase of the tick is there, and they add up to the profiled time, which is the run time
    let phases = c["phases_ns"].as_object().unwrap();
    let names: Vec<&str> = phases.keys().map(String::as_str).collect();
    for n in ["calendar", "commit", "control", "dispatch", "driver", "event", "execute", "flush"] {
        assert!(names.contains(&n), "phase {n} in {names:?}");
    }
    let sum: u64 = phases.values().map(|v| v.as_u64().unwrap()).sum();
    assert_eq!(sum, c["profiled_ns"].as_u64().unwrap());
    assert!(c["profiled_ns"].as_u64().unwrap() <= c["run_ns"].as_u64().unwrap());
    assert!(phases["execute"].as_u64().unwrap() > 0 && phases["dispatch"].as_u64().unwrap() > 0);
    // the opcodes add up to the executions, by slice and in all; b3 reads memory and adds in the VXM
    let ops = c["opcodes"].as_object().unwrap();
    let n: u64 = ops.values().map(|v| v["count"].as_u64().unwrap()).sum();
    assert_eq!(n, c["executions"].as_u64().unwrap());
    assert!(c["executions"].as_u64().unwrap() >= c["dispatches"].as_u64().unwrap() / 2);
    assert!(ops["Read"]["count"].as_u64().unwrap() > 0 && ops["Read"]["slice"] == "MEM", "{ops:?}");
    let slices = c["slices"].as_object().unwrap();
    let ns: u64 = slices.values().map(|v| v["count"].as_u64().unwrap()).sum();
    assert_eq!(ns, n);
    assert!(slices["MEM"]["count"].as_u64().unwrap() > 0 && slices["VXM"]["count"].as_u64().unwrap() > 0);
}

#[test]
fn the_run_profile_never_changes_the_trace() {
    let d = tempfile::tempdir().unwrap();
    let prof = d.path().join("profile.json");
    for prog in ["b1.txt", "b2.txt", "b3.txt"] {
        let plain = run(&program(prog), &["--trace", "--lint"]);
        let timed = run(&program(prog), &["--trace", "--lint", "--run-profile-json", prof.to_str().unwrap()]);
        assert_eq!(plain.status.code(), Some(0));
        assert_eq!(timed.status.code(), Some(0), "{}", text(&timed.stderr));
        assert!(plain.stdout.len() > 1000, "{prog}: a trace to compare");
        assert!(plain.stdout == timed.stdout, "{prog}: the trace differs with --run-profile-json");
        assert!(timed.stderr.is_empty(), "{}", text(&timed.stderr));
    }
    let off = run(&program("b3.txt"), &["--trace"]);
    assert!(!d.path().join("nothing.json").exists() && off.stderr.is_empty());
}

#[test]
fn the_run_profile_flag_refuses_bad_input() {
    // no file name
    let o = run_iss(&["run", "--profile", "ref", &program("b1.txt"), "--run-profile-json"], "");
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o.stderr).contains("usage: tsp-iss run"), "{}", text(&o.stderr));
    // a file that cannot be written: exit 4, one line naming it, the trace already complete on stdout
    let o = run(&program("b1.txt"), &["--run-profile-json", "/nonexistent-dir/p.json"]);
    assert_eq!(o.status.code(), Some(4), "{}", text(&o.stderr));
    assert!(text(&o.stderr).contains("tsp-iss: run: cannot write /nonexistent-dir/p.json"), "{}", text(&o.stderr));
    assert!(text(&o.stdout).starts_with("E "), "{}", text(&o.stdout));
}

/// The files of `--obs-out` for `prog` with `extra` flags, in a fresh directory.
fn obs_run(prog: &str, extra: &[&str]) -> (tempfile::TempDir, std::process::Output) {
    let d = tempfile::tempdir().unwrap();
    let dir = d.path().to_str().unwrap().to_string();
    let mut args = vec!["--obs-out", dir.as_str()];
    args.extend_from_slice(extra);
    let o = run(prog, &args);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o.stderr));
    (d, o)
}

fn lines(path: &std::path::Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{l}: {e}")))
        .collect()
}

#[test]
fn diagnostics_json_lists_what_the_trace_reports() {
    // b3 reports OTS-E-TIM-001 many times
    let (d, o) = obs_run(&program("b3.txt"), &["--obs-diagnostics", "--trace"]);
    let doc = json(&d.path().join("diagnostics.json"));
    let c = &doc["chips"][0];
    let trace = text(&o.stdout);
    let tdiags: Vec<(u64, u64, String)> = trace
        .lines()
        .filter_map(|l| l.strip_prefix("T diag "))
        .map(|l| {
            let f: Vec<&str> = l.split(' ').collect();
            (f[0].parse().unwrap(), f[2].parse().unwrap(), f[3].to_string())
        })
        .collect();
    assert!(tdiags.len() > 100);
    let ours: Vec<(u64, u64, String)> = c["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| (d["cycle"].as_u64().unwrap(), d["queue"].as_u64().unwrap(), d["code"].as_str().unwrap().to_string()))
        .collect();
    assert_eq!(ours, tdiags, "the same diagnostics in the same order as `T diag`");
    assert_eq!(c["total"].as_u64().unwrap(), tdiags.len() as u64);
    assert_eq!(c["truncated"], false);
    assert_eq!(c["by_code"]["OTS-E-TIM-001"].as_u64().unwrap(), tdiags.len() as u64);
    assert!(c["diagnostics"][0]["queue_name"].is_string());
}

#[test]
fn utilization_json_counts_executions_by_unit_and_queue() {
    let (d, _) = obs_run(&program("b3.txt"), &["--obs-utilization", "--obs-bin", "50"]);
    let prof = d.path().join("prof.json");
    let o = run(&program("b3.txt"), &["--run-profile-json", prof.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(0));
    let (u, p) = (&json(&d.path().join("utilization.json"))["chips"][0].clone(), &json(&prof)["chips"][0].clone());
    assert_eq!(u["bin_cycles"], 50);
    let cycles = u["cycles"].as_u64().unwrap();
    let units = u["units"].as_object().unwrap();
    for name in ["MEM", "VXM"] {
        assert_eq!(units[name]["executions"], p["slices"][name]["count"], "{name}");
        assert!(units[name]["active_cycles"].as_u64().unwrap() <= cycles);
    }
    assert_eq!(units["MXM"]["executions"], 0);
    let q: u64 = u["queues"].as_array().unwrap().iter().map(|q| q["executions"].as_u64().unwrap()).sum();
    assert_eq!(q, p["executions"].as_u64().unwrap());
    let t: u64 = u["timeline"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|b| b["executions"].as_object().unwrap().values().map(|v| v.as_u64().unwrap()))
        .sum();
    assert_eq!(t, q, "the timeline holds every execution");
    assert_eq!(u["timeline"][1]["start"], 50);
    // MEM queue 0 reads in each of 200 cycles
    let q0 = u["queues"].as_array().unwrap().iter().find(|q| q["queue"] == 0).unwrap();
    assert_eq!((q0["executions"].as_u64(), q0["active_cycles"].as_u64()), (Some(200), Some(200)));
    assert_eq!(q0["name"], "MEM[0]");
}

#[test]
fn events_and_probes_agree_with_each_other_and_the_trace() {
    let (d, o) = obs_run(
        &program("b3.txt"),
        &[
            "--trace",
            "--obs-events",
            "--obs-probe",
            "event_window@20-22",
            "--obs-probe",
            "srf_snapshot@30",
            "--obs-summary",
        ],
    );
    let ev = lines(&d.path().join("events.jsonl"));
    let sum = json(&d.path().join("summary.json"));
    let kind = |k: &str| ev.iter().filter(|e| e["kind"] == k).count() as u64;
    let c = &sum["chips"][0];
    assert_eq!(kind("dispatch"), c["dispatches"].as_u64().unwrap());
    assert_eq!(kind("exec"), c["executions"].as_u64().unwrap());
    assert_eq!(kind("diag"), c["diagnostics"].as_u64().unwrap());
    let trace = text(&o.stdout);
    assert_eq!(kind("srf") as usize, trace.lines().filter(|l| l.starts_with("T srf ")).count());
    assert_eq!(kind("sram") as usize, trace.lines().filter(|l| l.starts_with("T sram ")).count());
    assert!(ev.iter().all(|e| e["chip"] == 0));
    // the event window is the events of the stream inside it
    let probes = json(&d.path().join("probes.json"));
    let w = &probes["chips"][0]["probes"][0];
    assert_eq!(
        (w["type"].as_str(), w["begin"].as_u64(), w["end"].as_u64()),
        (Some("event_window"), Some(20), Some(22))
    );
    let inside: Vec<&Value> = ev.iter().filter(|e| (20..=22).contains(&e["c"].as_u64().unwrap())).collect();
    let got = w["events"].as_array().unwrap();
    assert_eq!(got.len(), inside.len());
    assert!(!got.is_empty());
    for (g, e) in got.iter().zip(&inside) {
        let mut e = (*e).clone();
        e.as_object_mut().unwrap().insert("chip".into(), 0.into());
        assert_eq!(g["c"], e["c"]);
        assert_eq!(g["kind"], e["kind"]);
    }
    // the snapshot is the `T srf` writes of the trace that are still in flight: written before the cycle, and not
    // past register 44
    let mut expect: Vec<(String, u64, u64, u64, u64, String)> = Vec::new();
    for l in trace.lines().filter_map(|l| l.strip_prefix("T srf ")) {
        let f: Vec<&str> = l.split(' ').collect();
        let (cyc, rs, gap) = (f[0].parse::<u64>().unwrap(), f[1], f[3].parse::<u64>().unwrap());
        let (row, stream) = rs.split_once(':').unwrap();
        if cyc < 30 && gap + (30 - cyc) <= 44 {
            expect.push((
                row.into(),
                stream.parse().unwrap(),
                cyc,
                gap,
                gap + 30 - cyc,
                f.get(4).copied().unwrap_or("").into(),
            ));
        }
    }
    let snap = &probes["chips"][0]["probes"][1];
    assert_eq!(snap["type"], "srf_snapshot");
    let mut got: Vec<(String, u64, u64, u64, u64, String)> = snap["writes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| {
            let v = w["value"].as_str().unwrap();
            // the trace keeps only the lanes of the profile; the snapshot has the same bytes
            (
                w["row"].as_str().unwrap().into(),
                w["stream"].as_u64().unwrap(),
                w["written_cycle"].as_u64().unwrap(),
                w["written_gap"].as_u64().unwrap(),
                w["register"].as_u64().unwrap(),
                v.into(),
            )
        })
        .collect();
    expect.sort();
    got.sort();
    assert!(!expect.is_empty());
    assert_eq!(got, expect);
}

#[test]
fn utilization_probe_reports_the_cycle_and_what_ran_in_it() {
    let (d, _) = obs_run(&program("b3.txt"), &["--obs-probe", "utilization@25"]);
    let p = &json(&d.path().join("probes.json"))["chips"][0]["probes"][0];
    assert_eq!(
        (p["type"].as_str(), p["cycle"].as_u64(), p["beyond_run"].as_bool()),
        (Some("utilization"), Some(25), Some(false))
    );
    // cycles 0..=25 each issued a read on 64 MEM queues
    assert_eq!(p["up_to_cycle"]["MEM"]["active_cycles"], 26);
    assert_eq!(p["up_to_cycle"]["MEM"]["executions"], 26 * 64);
    let ex = p["executing"].as_array().unwrap();
    assert_eq!(ex.iter().filter(|e| e["op"] == "Read").count(), 64);
    // a cycle after the run ends is flagged, and holds nothing in flight
    let (d, _) = obs_run(&program("b3.txt"), &["--obs-probe", "srf_snapshot@100000"]);
    let p = &json(&d.path().join("probes.json"))["chips"][0]["probes"][0];
    assert_eq!((p["beyond_run"].as_bool(), p["writes"].as_array().map(Vec::len)), (Some(true), Some(0)));
}

#[test]
fn the_report_is_text_on_stderr_and_the_same_every_time() {
    let a = run(&program("b3.txt"), &["--obs-report", "medium"]);
    let b = run(&program("b3.txt"), &["--obs-report", "medium"]);
    assert_eq!(a.status.code(), Some(0), "{}", text(&a.stderr));
    assert_eq!(a.stderr, b.stderr, "no wall time in the report");
    let t = text(&a.stderr);
    assert!(
        t.starts_with(
            "chip 0 (ref): 205 cycles, 205 ticks (0 idle), 160 dispatches, 15856 executions, 184 diagnostics\n"
        ),
        "{t}"
    );
    assert!(t.contains("unit MEM: 12800 executions") && t.contains("queue 0 (MEM[0])"), "{t}");
    assert!(!t.contains("opcode "), "opcodes are verbose only: {t}");
    let v = text(&run(&program("b3.txt"), &["--obs-report", "verbose"]).stderr);
    assert!(v.contains("opcode Read (MEM): 12800") && v.contains("diagnostic OTS-E-TIM-001: 184"), "{v}");
    let c = text(&run(&program("b3.txt"), &["--obs-report", "compact"]).stderr);
    assert_eq!(c.lines().count(), 2, "{c}");
    assert!(a.stdout.starts_with(b"E "), "stdout is the harness output");
}

#[test]
fn no_observability_flag_changes_the_trace() {
    let d = tempfile::tempdir().unwrap();
    let dir = d.path().join("out");
    let prof = d.path().join("p.json");
    for prog in ["b1.txt", "b3.txt"] {
        let plain = run(&program(prog), &["--trace", "--lint"]);
        let all = run(
            &program(prog),
            &[
                "--trace",
                "--lint",
                "--run-profile-json",
                prof.to_str().unwrap(),
                "--obs-out",
                dir.to_str().unwrap(),
                "--obs-summary",
                "--obs-diagnostics",
                "--obs-utilization",
                "--obs-events",
                "--obs-probe",
                "srf_snapshot@10",
                "--obs-probe",
                "utilization@10",
                "--obs-probe",
                "event_window@5-9",
                "--obs-report",
                "verbose",
            ],
        );
        assert_eq!(all.status.code(), Some(0), "{}", text(&all.stderr));
        assert!(plain.stdout == all.stdout, "{prog}: the trace differs with every flag on");
    }
}

#[test]
fn recording_works_for_several_chips() {
    let d = tempfile::tempdir().unwrap();
    let (dir, prof) = (d.path().join("o"), d.path().join("p.json"));
    let plain = run(&program("b3.txt"), &["--chips", "2", "--trace"]);
    let both = run(
        &program("b3.txt"),
        &[
            "--chips",
            "2",
            "--trace",
            "--run-profile-json",
            prof.to_str().unwrap(),
            "--obs-out",
            dir.to_str().unwrap(),
            "--obs-utilization",
            "--obs-events",
        ],
    );
    assert_eq!(both.status.code(), Some(0), "{}", text(&both.stderr));
    assert!(plain.stdout == both.stdout);
    let p = json(&prof);
    assert_eq!(p["chips"].as_array().unwrap().len(), 2);
    assert!(p["chips"][0]["executions"].as_u64().unwrap() > 0);
    let u = json(&dir.join("utilization.json"));
    assert_eq!(u["chips"].as_array().unwrap().len(), 2);
    assert_eq!(u["chips"][1]["chip"], 1);
    // a text program is chip 0's: chip 1 ticks once and executes nothing
    assert_eq!((p["chips"][1]["ticks"].as_u64(), p["chips"][1]["executions"].as_u64()), (Some(1), Some(0)));
    let ev = lines(&dir.join("events.jsonl"));
    assert!(ev.len() > 1000 && ev.iter().all(|e| e["chip"] == 0));
}

#[test]
fn the_obs_flags_refuse_bad_input() {
    let bad = |args: &[&str], msg: &str| {
        let o = run(&program("b1.txt"), args);
        assert_eq!(o.status.code(), Some(2), "{args:?}: {}", text(&o.stderr));
        assert!(text(&o.stderr).contains(msg), "{args:?}: {}", text(&o.stderr));
        assert!(o.stdout.is_empty(), "{args:?}: nothing runs");
    };
    bad(&["--obs-diagnostics"], "give --obs-out DIR");
    bad(&["--obs-summary"], "give --obs-out DIR");
    bad(&["--obs-probe", "srf_snapshot@3"], "give --obs-out DIR");
    bad(&["--obs-out", "/tmp/x", "--obs-probe", "bogus"], "expected srf_snapshot@CYCLE");
    bad(&["--obs-out", "/tmp/x", "--obs-probe", "event_window@9-3"], "ends before it begins");
    bad(&["--obs-out", "/tmp/x", "--obs-probe", "srf_snapshot@-1"], "expected srf_snapshot@CYCLE");
    bad(&["--obs-report", "huge"], "expected compact, medium or verbose");
    bad(&["--obs-bin", "0"], "expected a number of cycles");
    // a directory that cannot be made: exit 4, after the run
    let f = tempfile::NamedTempFile::new().unwrap();
    let under = f.path().join("sub");
    let o = run(&program("b1.txt"), &["--obs-out", under.to_str().unwrap(), "--obs-diagnostics"]);
    assert_eq!(o.status.code(), Some(4), "{}", text(&o.stderr));
    assert!(text(&o.stderr).contains("tsp-iss: run: cannot write"), "{}", text(&o.stderr));
}

#[test]
fn a_failed_observer_publication_invalidates_the_previous_manifest() {
    let (d, _) = obs_run(&program("b1.txt"), &["--obs-summary"]);
    assert!(d.path().join("manifest.json").is_file());
    std::fs::create_dir(d.path().join("diagnostics.json")).unwrap();

    let result = run(&program("b1.txt"), &["--obs-out", d.path().to_str().unwrap(), "--obs-diagnostics"]);
    assert_eq!(result.status.code(), Some(4), "{}", text(&result.stderr));
    assert!(!d.path().join("manifest.json").exists());
}

#[test]
fn a_second_observer_publisher_receives_an_explicit_output_error() {
    let (d, _) = obs_run(&program("b1.txt"), &["--obs-summary"]);
    let directory = std::fs::File::open(d.path()).unwrap();
    directory.try_lock().unwrap();
    let manifest = d.path().join("manifest.json");
    let before = std::fs::read(&manifest).unwrap();

    let result = run(&program("b1.txt"), &["--obs-out", d.path().to_str().unwrap(), "--obs-diagnostics"]);
    assert_eq!(result.status.code(), Some(4), "{}", text(&result.stderr));
    assert_eq!(std::fs::read(manifest).unwrap(), before);
    assert!(!d.path().join("diagnostics.json").exists());
}

#[test]
fn link_diagnostics_cover_the_full_supported_link_range() {
    use tsp_iss::harness::obs::{diagnostics_json, ObsConfig, ObsState, Recorder};
    use tsp_iss::{NoObserver, Observer};

    let mut state = ObsState::new(&ObsConfig { diagnostics: true, ..ObsConfig::default() });
    let mut observer = Recorder { inner: &mut NoObserver, st: &mut state };
    for link in [0, 31, 32, 127, 128] {
        observer.link_op(link, true);
    }
    observer.link_op(127, false);

    let out = diagnostics_json(0, &state);
    assert_eq!(out["links_sent"], serde_json::json!([0, 31, 32, 127]));
    assert_eq!(out["links_received"], serde_json::json!([127]));
}

#[test]
fn summary_alone_activates_exact_execution_counts_without_extra_artifacts() {
    for cycle in [100u64, u64::MAX] {
        let d = tempfile::tempdir().unwrap();
        let program = d.path().join("one.txt");
        let host = d.path().join("host.txt");
        std::fs::write(&program, "Q 0 01800001\n").unwrap();
        std::fs::write(&host, format!("H start {cycle:x}\n")).unwrap();
        let out = d.path().join("out");
        let r = run(
            program.to_str().unwrap(),
            &[
                "--host",
                host.to_str().unwrap(),
                "--max-cycles",
                &u64::MAX.to_string(),
                "--obs-out",
                out.to_str().unwrap(),
                "--obs-summary",
            ],
        );
        assert_eq!(r.status.code(), Some(0), "{}", text(&r.stderr));
        let summary = json(&out.join("summary.json"));
        assert_eq!(summary["chips"][0]["cycles"].to_string(), (u128::from(cycle) + 1).to_string());
        assert_eq!(summary["chips"][0]["dispatches"], 1);
        assert_eq!(summary["chips"][0]["executions"], 1);
        assert!(text(&r.stderr).contains("1 dispatches, 1 executions"));
        assert!(out.join("manifest.json").is_file());
        for name in ["events.jsonl", "diagnostics.json", "utilization.json", "profile.json"] {
            assert!(!out.join(name).exists(), "unrequested {name}");
        }
    }
}

#[test]
fn mixed_chip_utilization_uses_one_explicit_encoding_and_exact_prefixes() {
    for width in [1u64, 7, u64::MAX] {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("one.txt");
        let h = d.path().join("host.txt");
        std::fs::write(
            &p,
            "Q 0 01800001
",
        )
        .unwrap();
        std::fs::write(
            &h,
            "C 0
H start ffffffffffffffff
C 1
H inject 64 0 01800001
",
        )
        .unwrap();
        let out = d.path().join("obs");
        let r = run(
            p.to_str().unwrap(),
            &[
                "--chips",
                "3",
                "--jobs",
                "2",
                "--host",
                h.to_str().unwrap(),
                "--max-cycles",
                &u64::MAX.to_string(),
                "--obs-out",
                out.to_str().unwrap(),
                "--obs-utilization",
                "--obs-bin",
                &width.to_string(),
            ],
        );
        assert_eq!(r.status.code(), Some(0), "{}", text(&r.stderr));
        let doc = json(&out.join("utilization.json"));
        let chips = doc["chips"].as_array().unwrap();
        assert_eq!(chips.len(), 3);
        assert_eq!(chips[0]["cycles"].to_string(), "18446744073709551616");
        assert_eq!(chips[1]["cycles"], 101);
        assert_eq!(chips[2]["cycles"], 1);
        if width == u64::MAX {
            assert_eq!(doc["schema"], 1);
            assert_eq!(chips[0]["timeline"].as_array().unwrap().len(), 2);
            assert_eq!(chips[0]["timeline"][0]["executions"], serde_json::json!({}));
            assert_eq!(chips[0]["timeline"][1]["executions"]["CU"], 1);
            assert_eq!(chips[1]["timeline"].as_array().unwrap().len(), 1);
            assert!(chips[2]["timeline"].as_array().unwrap().is_empty());
        } else {
            assert_eq!(doc["schema"], "opentsp.utilization.sparse.v1");
            for (i, chip) in chips.iter().enumerate() {
                assert!(chip.get("timeline").is_none());
                assert_eq!(chip["timeline_encoding"], "absolute-nonzero-v1");
                assert_eq!(chip["timeline_omitted_bins_are_zero"], true);
                let rows = chip["timeline_nonzero"].as_array().unwrap();
                if i == 2 {
                    assert_eq!(chip["timeline_bins"], 0);
                    assert!(rows.is_empty());
                } else {
                    let cycle = if i == 0 { u64::MAX } else { 100 };
                    let index = cycle / width;
                    assert_eq!(chip["timeline_bins"].to_string(), (u128::from(index) + 1).to_string());
                    assert_eq!(rows.len(), 1);
                    assert_eq!(rows[0]["index"].as_u64(), Some(index));
                    assert_eq!(rows[0]["start"].as_u64(), Some(index * width));
                    assert_eq!(rows[0]["executions"], serde_json::json!({"CU": 1}));
                }
            }
        }
    }
}
