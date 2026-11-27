//! `tsp-iss bench`: the checked-in programs B0..B4 and `idle` run natively and report
//! simulated cycles per second; the programs are the output of `benches/programs/gen_bench.py`; the
//! performance targets hold on release builds.
mod common;
use common::*;
use std::collections::BTreeMap;

/// `b0`..`b4`, `idle`, then the benchmark coverage programs.
const COVERAGE_ORDER: [&str; 6] = ["b0", "b1", "b2", "b3", "b4", "idle"];
const COVERAGE: [&str; 5] = ["dense_f16", "dense_i8", "vxm_valid", "transc", "syncnotify"];

fn programs_dir() -> std::path::PathBuf {
    root().join("opentsp-iss/crates/tsp-iss/benches/programs")
}

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

#[test]
fn bench_runs_every_checked_in_program() {
    let o = run_iss(&["bench"], "");
    assert_eq!(o.status.code(), Some(0), "{}", text(&o.stderr));
    let t = parse(&text(&o.stdout));
    let mut want: Vec<&str> = COVERAGE_ORDER.iter().chain(&COVERAGE).copied().collect();
    want.sort_unstable();
    assert_eq!(t.keys().map(String::as_str).collect::<Vec<_>>(), want);
    for (name, r) in &t {
        assert!(num(r, "seconds") >= 0.0, "{name}");
    }
    // the programs' own lengths: B0 has no text; B1 is 64 loops of 200 cycles; B4 is about 10^6 cycles
    assert_eq!(num(&t["b0"], "ticks"), 1.0, "the machine ticks cycle 0");
    assert!((200.0..400.0).contains(&num(&t["b1"], "cycles")), "{:?}", t["b1"]);
    assert!((1.0e6..1.2e6).contains(&num(&t["b4"], "cycles")), "{:?}", t["b4"]);
    assert!(num(&t["idle"], "cycles") >= 1.0e8);
    assert!(num(&t["idle"], "ticks") < 1000.0, "idle cycles are skipped");
}

#[test]
fn bench_takes_a_program_and_repeats() {
    let p = programs_dir().join("b1.txt");
    let o = run_iss(&["bench", "--program", p.to_str().unwrap(), "--repeat", "3"], "");
    assert_eq!(o.status.code(), Some(0), "{}", text(&o.stderr));
    let t = parse(&text(&o.stdout));
    assert_eq!(t.keys().map(String::as_str).collect::<Vec<_>>(), ["b1"]);
    assert_eq!(t["b1"]["repeat"], "3");
}

#[test]
fn bench_refuses_bad_arguments() {
    for args in [&["--repeat"][..], &["--repeat", "0"], &["--repeat", "x"], &["--program"], &["--bogus"], &["stray"]] {
        let mut a = vec!["bench"];
        a.extend_from_slice(args);
        let o = run_iss(&a, "");
        assert_eq!(o.status.code(), Some(2), "{args:?}");
        assert!(o.stdout.is_empty());
        assert!(text(&o.stderr).starts_with("tsp-iss: bench: usage: tsp-iss bench"), "{}", text(&o.stderr));
    }
    let o = run_iss(&["bench", "--program", "/nonexistent/x.txt"], "");
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o.stderr).starts_with("tsp-iss: bench: /nonexistent/x.txt: "), "{}", text(&o.stderr));
    let d = tempfile::tempdir().unwrap();
    let bad = d.path().join("bad.txt");
    std::fs::write(&bad, "Q zz 1\n").unwrap();
    let o = run_iss(&["bench", "--program", bad.to_str().unwrap()], "");
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o.stderr).contains("input line 1"), "{}", text(&o.stderr));
}

/// A program's output through `bench` agrees with the compatible mode on the program's last cycle.
#[test]
fn bench_programs_match_the_compatible_mode() {
    for name in ["b1", "b2", "b3"] {
        let src = std::fs::read_to_string(programs_dir().join(format!("{name}.txt"))).unwrap();
        let cfg = config("ref");
        let o = run_iss(&[cfg.to_str().unwrap(), "--trace"], &src);
        assert_eq!(o.status.code(), Some(0), "{name}");
        let last = text(&o.stdout)
            .lines()
            .filter_map(|l| l.strip_prefix("T srf ").or_else(|| l.strip_prefix("T diag ")))
            .filter_map(|l| l.split(' ').next()?.parse::<u64>().ok())
            .max()
            .unwrap_or(0);
        let p = programs_dir().join(format!("{name}.txt"));
        let b = run_iss(&["bench", "--program", p.to_str().unwrap()], "");
        let t = parse(&text(&b.stdout));
        let cycles = num(&t[name], "cycles") as u64;
        // `cycles` counts through the last tick, which may follow the last visible event
        assert!(cycles > last && cycles <= last + 64, "{name}: bench says {cycles} cycles, the trace ends at {last}");
    }
}

/// The checked-in text is what the generator writes.
#[test]
fn programs_are_generated() {
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".into());
    let o = std::process::Command::new(python)
        .arg(programs_dir().join("gen_bench.py"))
        .arg("--check")
        .output()
        .expect("python3 (source the toolchain env.sh)");
    assert!(o.status.success(), "{}", text(&o.stderr));
}

/// The programs against the reference: B0, B1 and the real 10^8-cycle `idle` (a read at about cycle 10^8,
/// `--max-cycles` raised past the default) give `tsp_sim`'s exit code, stdout and stderr, with and without
/// `--trace`. (B2 to B4 take the reference 16 s and more; the differential nets cover their instructions.)
#[test]
fn bench_programs_match_tsp_sim() {
    let cfg = config("ref").to_str().unwrap().to_string();
    for name in ["b0", "b1", "idle"] {
        let prog = std::fs::read_to_string(programs_dir().join(format!("{name}.txt"))).unwrap();
        for trace in [false, true] {
            let mut args = vec![cfg.as_str(), "--max-cycles", "4000000000"];
            if trace {
                args.push("--trace");
            }
            let (ours, theirs) = (run_iss(&args, &prog), run_sim(&args, &prog));
            assert_eq!(ours.status.code(), theirs.status.code(), "{name} trace={trace}: exit code");
            assert_eq!(text(&ours.stdout), text(&theirs.stdout), "{name} trace={trace}: stdout");
            assert_eq!(text(&ours.stderr), norm_stderr(&text(&theirs.stderr)), "{name} trace={trace}: stderr");
            assert_eq!(ours.status.code(), Some(0), "{name}: {}", text(&ours.stderr));
            if name == "idle" {
                // the read really is at cycle 10^8 + 14: the trace says so
                assert!(!trace || text(&ours.stdout).contains("T srf 100000014 "), "idle: no read at 10^8");
            }
        }
    }
}

/// The benchmark coverage programs run with no diagnostic at all, as the benchmark of a workload that is not the
/// diagnostic path (B2 raises 3056 TIM-001 in 193 cycles): the full programs on the ISS, their short variants on `tsp_sim`
/// too, with its exit code, stdout and stderr.
#[test]
fn coverage_programs_have_no_diagnostics_and_match_tsp_sim() {
    let cfg = config("ref").to_str().unwrap().to_string();
    for name in COVERAGE {
        let src = std::fs::read_to_string(programs_dir().join(format!("{name}.txt"))).unwrap();
        let o = run_iss(&[cfg.as_str()], &src);
        assert_eq!(text(&o.stdout).trim_end(), "E - -1 0", "{name}: the program raises a diagnostic");
        assert_eq!(o.status.code(), Some(0), "{name}");
    }
    let d = tempfile::tempdir().unwrap();
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".into());
    let g = std::process::Command::new(python)
        .arg(programs_dir().join("gen_bench.py"))
        .args(["--small", d.path().to_str().unwrap()])
        .output()
        .expect("python3 (source the toolchain env.sh)");
    assert!(g.status.success(), "{}", text(&g.stderr));
    for name in COVERAGE {
        let src = std::fs::read_to_string(d.path().join(format!("{name}.txt"))).unwrap();
        for trace in [false, true] {
            let mut args = vec![cfg.as_str()];
            if trace {
                args.push("--trace");
            }
            let (a, b) = (run_sim(&args, &src), run_iss(&args, &src));
            assert_eq!(a.status.code(), b.status.code(), "{name} trace={trace}");
            assert_eq!(text(&a.stdout), text(&b.stdout), "{name} trace={trace}: stdout");
            assert_eq!(norm_stderr(&text(&a.stderr)), norm_stderr(&text(&b.stderr)), "{name} trace={trace}: stderr");
            if !trace {
                assert_eq!(
                    text(&a.stdout).trim_end(),
                    "E - -1 0",
                    "{name}: the short variant raises a diagnostic on tsp_sim"
                );
            }
        }
    }
}

/// `bench` reports the two rates that do not depend on a program's idle cycles.
#[test]
fn bench_reports_the_cost_per_dispatch_and_per_visited_tick() {
    let p = programs_dir().join("vxm_valid.txt");
    let o = run_iss(&["bench", "--program", p.to_str().unwrap()], "");
    assert_eq!(o.status.code(), Some(0), "{}", text(&o.stderr));
    let r = &parse(&text(&o.stdout))["vxm_valid"];
    let (secs, ticks, disp) = (num(r, "seconds"), num(r, "ticks"), num(r, "dispatches"));
    assert!(disp > 0.0 && ticks > 0.0);
    // the printed rates are the seconds over the counts (`seconds` is printed to the microsecond)
    for (key, n) in [("ns_per_dispatch", disp), ("ns_per_visited_tick", ticks)] {
        let want = secs * 1e9 / n;
        assert!((num(r, key) - want).abs() <= 0.06 + 1e-3 * want, "{key}: {} against {want}", num(r, key));
    }
}
