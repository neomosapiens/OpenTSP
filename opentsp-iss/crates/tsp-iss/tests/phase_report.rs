//! Exercise the actual cumulative stderr report in an isolated process. Global
//! counters from other tests must not influence its calibration or call counts.
#![cfg(feature = "phases")]

use std::process::Command;
use tsp_iss::driver::phase;

#[test]
fn repeated_reports_child() {
    if std::env::var_os("TSP_PHASE_REPORT_CHILD").is_none() {
        return;
    }
    for n in 1..=12 {
        for _ in 0..10_000 {
            let mark = phase::begin();
            phase::end(0, mark);
        }
        phase::report(n * 10_000);
    }
}

#[test]
fn repeated_reports_calibrate_the_cumulative_counters_without_multiplying_cost() {
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "repeated_reports_child", "--nocapture"])
        .env("TSP_PHASE_REPORT_CHILD", "1")
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let stderr = String::from_utf8(output.stderr).unwrap();
    let mut recorded = Vec::new();
    let mut calls = Vec::new();
    for line in stderr.lines() {
        if let Some((_, tail)) = line.split_once("of which recorded in a phase ") {
            recorded.push(tail.split_whitespace().next().unwrap().parse::<f64>().unwrap());
        }
        if line.starts_with("phase  0 ") {
            let fields: Vec<_> = line.split_whitespace().collect();
            let at = fields.iter().position(|s| *s == "calls").unwrap();
            calls.push(fields[at - 1].parse::<u64>().unwrap());
        }
    }
    assert_eq!(recorded.len(), 12, "{stderr}");
    assert_eq!(calls, (1..=12).map(|n| n * 10_000).collect::<Vec<_>>(), "{stderr}");
    // An average pair cost should remain comparable across reports. The old
    // current-batch denominator makes the last cost twelve times the first.
    // A factor of four tolerates substantial host timing noise without allowing
    // that deterministic accumulation error.
    let first = recorded[0];
    let last = recorded[11];
    assert!(first > 0.0 && last < first * 4.0, "calibration grew from {first} to {last} ns\n{stderr}");
}

#[test]
fn machine_reports_child() {
    if std::env::var_os("TSP_PHASE_MACHINE_CHILD").is_none() {
        return;
    }
    use ots_profile::Profile;
    use tsp_iss::{
        links::NoLinks,
        machine::{Machine, RunError},
        observe::NoObserver,
    };
    let new = || Machine::new(&Profile::load("mini").unwrap(), 0, Box::new(NoLinks)).unwrap();
    let expect = |ticks| println!("expected cumulative ticks: {ticks}");
    let mut total = 0;
    phase::report(0);
    expect(0);
    let mut m = new();
    total += m.run(100, &mut NoObserver).unwrap().ticks;
    expect(total);
    assert_eq!(m.run(100, &mut NoObserver).unwrap().ticks, 0);
    expect(total);
    let delayed = |idle| {
        let mut m = new();
        let image = [ots_isa::Insn::NOP { n: 5 }, ots_isa::Insn::SetVL { n: 2 }].map(|i| ots_isa::encode(&i) as u32);
        m.load_image(100, &image).unwrap();
        m.set_tick_every_cycle(idle);
        m
    };
    for idle in [false, true] {
        let mut m = delayed(idle);
        let partial = m.run_until(2, 100, &mut NoObserver).unwrap();
        assert_eq!(partial.ticks, if idle { 2 } else { 1 });
        total += partial.ticks;
        total += m.run(100, &mut NoObserver).unwrap().ticks;
        expect(total);
    }
    let mut m = delayed(false);
    assert_eq!(m.run(4, &mut NoObserver), Err(RunError::MaxCycles { cycle: 5, limit: 4 }));
    // Tick zero completed before the limit error. Failed calls emit no automatic
    // report, but their completed work belongs to the next cumulative snapshot.
    total += 1;
    phase::report(total);
    expect(total);
    // Reading a snapshot twice must not add another run's ticks.
    phase::report(total);
    expect(total);
}

#[test]
fn machine_reports_match_all_completed_ticks_including_partial_and_idle_runs() {
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "machine_reports_child", "--nocapture"])
        .env("TSP_PHASE_MACHINE_CHILD", "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(output.status.success(), "{stdout}\n{stderr}");
    let expected: Vec<u64> = stdout
        .lines()
        .filter_map(|line| line.strip_prefix("expected cumulative ticks: "))
        .map(|n| n.parse().unwrap())
        .collect();
    let actual: Vec<u64> = stderr
        .lines()
        .filter(|line| line.starts_with("phases: counter "))
        .map(|line| line.split(", ").nth(1).unwrap().split_whitespace().next().unwrap().parse().unwrap())
        .collect();
    assert_eq!(expected.len(), 7, "{stdout}\n{stderr}");
    assert_eq!(actual, expected, "{stdout}\n{stderr}");
    assert!(!stderr.contains("NaN") && !stderr.contains("inf"), "zero-tick snapshot\n{stderr}");
}

#[test]
fn concurrent_semantics_child() {
    if std::env::var_os("TSP_PHASE_THREADS_CHILD").is_none() {
        return;
    }
    // The timed outer interval belongs to one thread; another thread performs
    // semantics entirely inside it. That work must not be subtracted from the
    // first thread's phase. Channels establish ordering without a timing race.
    let (start_tx, start_rx) = std::sync::mpsc::channel();
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        start_rx.recv().unwrap();
        let semantic = phase::begin();
        std::thread::sleep(std::time::Duration::from_millis(80));
        phase::sem_end(phase::SEM, semantic);
        done_tx.send(()).unwrap();
    });
    let clock = std::time::Instant::now();
    let outer = phase::begin();
    start_tx.send(()).unwrap();
    done_rx.recv().unwrap();
    phase::end(0, outer);
    let elapsed = clock.elapsed().as_secs_f64() * 1e3;
    worker.join().unwrap();
    // Conversely, semantics nested in this same thread must still be removed.
    let local_outer = phase::begin();
    let local_semantics = phase::begin();
    std::thread::sleep(std::time::Duration::from_millis(80));
    phase::sem_end(phase::SEM, local_semantics);
    phase::end(1, local_outer);
    println!("independent elapsed ms: {elapsed}");
    phase::report(1);
}

#[test]
fn another_threads_semantics_does_not_erase_the_measured_interval() {
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "concurrent_semantics_child", "--nocapture"])
        .env("TSP_PHASE_THREADS_CHILD", "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(output.status.success(), "{stdout}\n{stderr}");
    let elapsed =
        stdout.lines().find_map(|l| l.strip_prefix("independent elapsed ms: ")).unwrap().parse::<f64>().unwrap();
    let phase_ms = |n| {
        let prefix = format!("phase {n:>2} ");
        let fields = stderr.lines().find(|l| l.starts_with(&prefix)).unwrap().split_whitespace().collect::<Vec<_>>();
        let at = fields.iter().position(|s| *s == "ms").unwrap();
        fields[at - 1].parse::<f64>().unwrap()
    };
    assert!(phase_ms(15) >= 120.0, "the other thread did measurable semantic work\n{stderr}");
    assert!(phase_ms(1) < elapsed * 0.25, "same-thread semantics must remain excluded\n{stderr}");
    assert!(phase_ms(0) >= elapsed * 0.5, "another thread erased the outer phase ({elapsed}ms)\n{stderr}");
}

#[test]
fn concurrent_machine_totals_child() {
    if std::env::var_os("TSP_PHASE_MACHINE_THREADS_CHILD").is_none() {
        return;
    }
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
    let workers: Vec<_> = (0..2)
        .map(|_| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                use ots_profile::Profile;
                use tsp_iss::{links::NoLinks, machine::Machine, observe::NoObserver};
                let mut m = Machine::new(&Profile::load("mini").unwrap(), 0, Box::new(NoLinks)).unwrap();
                let image =
                    [ots_isa::Insn::NOP { n: 10 }, ots_isa::Insn::SetVL { n: 2 }].map(|i| ots_isa::encode(&i) as u32);
                m.load_image(100, &image).unwrap();
                m.set_tick_every_cycle(true);
                barrier.wait();
                m.run_until(u64::MAX, 100, &mut NoObserver).unwrap().ticks
            })
        })
        .collect();
    barrier.wait();
    let expected: u64 = workers.into_iter().map(|w| w.join().unwrap()).sum();
    assert_eq!(phase::completed_ticks(), expected);
    println!("thread summary ticks: {expected}");
    phase::report(expected);
}

#[test]
fn threaded_machine_counters_aggregate_all_completed_work() {
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "concurrent_machine_totals_child", "--nocapture"])
        .env("TSP_PHASE_MACHINE_THREADS_CHILD", "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(output.status.success(), "{stdout}\n{stderr}");
    let expected: u64 = stdout.lines().find_map(|l| l.strip_prefix("thread summary ticks: ")).unwrap().parse().unwrap();
    assert_eq!(expected, 22, "both threads tick every cycle 0..10");
    assert!(stderr.contains(&format!(", {expected} ticks,")), "{stderr}");
    let fields = stderr.lines().find(|l| l.starts_with("phase 14 ")).unwrap().split_whitespace().collect::<Vec<_>>();
    let at = fields.iter().position(|s| *s == "calls").unwrap();
    assert_eq!(fields[at - 1].parse::<u64>().unwrap(), expected, "{stderr}");
}

#[test]
fn instrumented_phase_marks_cannot_be_moved_to_another_thread() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("main.rs");
    let phase_source = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/driver/phase.rs");
    std::fs::write(&source, format!(
        "#[path = {:?}] mod phase;\nfn main() {{ let mark = phase::begin(); std::thread::spawn(move || phase::end(0, mark)).join().unwrap(); }}\n",
        phase_source)).unwrap();
    let result = Command::new("rustc")
        .args(["--edition", "2021", "--cfg", "feature=\"phases\""])
        .arg(&source)
        .arg("-o")
        .arg(directory.path().join("cross-thread-mark"))
        .output()
        .unwrap();
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(!result.status.success(), "a phase Mark was allowed to move across threads");
    assert!(
        stderr.contains("cannot be sent between threads safely"),
        "expected thread-affinity diagnostic, not another compiler error:\n{stderr}"
    );
}
