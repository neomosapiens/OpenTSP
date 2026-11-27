//! `tsp-iss run --stats`: the JSON summary of a run: dispatches per queue, wait cycles
//! per hold reason, IQ level min/max per queue, fetch bytes, diagnostics per code, wall time. The
//! statistics never change the simulated results or the trace.
mod common;
use common::*;
use ots_isa::PADDING;
use serde_json::Value;
use std::path::PathBuf;
use tsp_iss::harness::stats::{Counters, StatsObserver};
use tsp_iss::{Machine, NoLinks, NoObserver};

const REASONS: [&str; 7] = ["parked", "busy", "path", "loop", "repeat", "configuration", "incomplete_text"];

/// A VXM queue (88) that adds, then two Repeats: the second waits for the first.
const REPEAT_CHAIN: &str = "Q 58 20020010 00000000 00c00051 00c00031\n";
/// MEM queue 4 parks on a Sync until the control channel's Notify at cycle 10.
const PARKED: &str = "Q 4 00400000 10000010 00000000\nQ ffff 0000000a 00800000\n";
/// A read, a loop of four iterations, then another read that waits for the loop.
const LOOP_THEN_READ: &str = "Q 4 10000010 00000000 11c00000 00040000 10000011 04000000\n";

fn bench_program(name: &str) -> PathBuf {
    root().join("opentsp-iss/crates/tsp-iss/benches/programs").join(format!("{name}.txt"))
}

struct Run {
    out: Output,
    stats: Value,
}
use std::process::Output;

fn run_stats(prog: &std::path::Path, extra: &[&str]) -> Run {
    let d = tempfile::tempdir().unwrap();
    let file = d.path().join("stats.json");
    let st = format!("--stats={}", file.display());
    let mut args = vec!["run", "--profile", "ref", &st];
    args.extend_from_slice(extra);
    let p = prog.to_str().unwrap();
    args.push(p);
    let out = run_iss(&args, "");
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    let stats = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    Run { out, stats }
}

fn text_program(d: &tempfile::TempDir, name: &str, src: &str) -> PathBuf {
    let p = d.path().join(name);
    std::fs::write(&p, src).unwrap();
    p
}

fn queue(chip: &Value, q: u64) -> Option<&Value> {
    chip["queues"].as_array().unwrap().iter().find(|e| e["queue"] == q)
}

#[test]
fn stats_schema() {
    let d = tempfile::tempdir().unwrap();
    let p = text_program(&d, "parked.txt", PARKED);
    let s = run_stats(&p, &[]).stats;
    assert_eq!(s["schema"], 1, "the version of the document");
    assert_eq!(s["profile"], "ref");
    assert_eq!(s["chips"], 1);
    for k in ["ticks", "diagnostics", "dispatches", "fetch_bytes"] {
        assert!(s[k].is_u64(), "{k}: {s}");
    }
    assert!(s["last_cycle"].is_u64());
    assert!(s["wall_seconds"].is_f64() && s["wall_seconds"].as_f64().unwrap() >= 0.0);
    let w = s["wait_cycles"].as_object().unwrap();
    assert_eq!(w.keys().map(String::as_str).collect::<Vec<_>>(), {
        let mut k = REASONS.to_vec();
        k.sort();
        k
    });
    assert!(w.values().all(Value::is_u64));
    assert!(s["diagnostics_by_code"].is_object());
    // the schedule cache: the counters are in every document, zero when no run used a cache
    let r = s["replay"].as_object().unwrap();
    assert_eq!(r["enabled"], cfg!(feature = "replay"));
    for k in [
        "hits",
        "misses",
        "divergences",
        "evictions",
        "recordings",
        "oversize",
        "refused",
        "shadow_checks",
        "shadow_mismatches",
        "schedules",
        "retained_bytes",
        "budget_bytes",
        "max_schedules",
    ] {
        assert!(r[k].is_u64(), "replay.{k}: {s}");
    }
    assert_eq!(
        (r["hits"].as_u64(), r["schedules"].as_u64(), r["retained_bytes"].as_u64()),
        (Some(0), Some(0), Some(0))
    );
    assert!(r["budget_bytes"].as_u64().unwrap() > 0 && r["max_schedules"].as_u64().unwrap() > 0);
    let chips = s["per_chip"].as_array().unwrap();
    assert_eq!(chips.len(), 1);
    let c = &chips[0];
    for k in ["ticks", "diagnostics", "dispatches", "fetch_bytes"] {
        assert_eq!(c[k], s[k], "one chip: the totals are the chip's");
    }
    assert_eq!(c["wait_cycles"], s["wait_cycles"]);
    assert!(c["first_diagnostic"].is_null());
    for e in c["queues"].as_array().unwrap() {
        assert!(e["queue"].is_u64() && e["name"].is_string() && e["dispatches"].is_u64());
        assert!(e["iq_min"].is_u64() && e["iq_max"].is_u64() && e["wait_cycles"].is_object());
        assert!(e["iq_min"].as_u64() <= e["iq_max"].as_u64());
    }
}

#[test]
fn dispatches_and_iq_levels_per_queue() {
    let d = tempfile::tempdir().unwrap();
    let p = text_program(&d, "parked.txt", PARKED);
    let s = run_stats(&p, &[]).stats;
    let c = &s["per_chip"][0];
    // Sync and Read on queue 4, Notify on the control channel
    assert_eq!(s["dispatches"], 3);
    let q4 = queue(c, 4).unwrap();
    assert_eq!((q4["name"].as_str(), q4["dispatches"].as_u64()), (Some("MEM[4]"), Some(2)));
    assert_eq!((q4["iq_min"].as_u64(), q4["iq_max"].as_u64()), (Some(0), Some(3)), "three units loaded, all popped");
    let cu = queue(c, 0xffff).unwrap();
    assert_eq!((cu["name"].as_str(), cu["dispatches"].as_u64(), cu["iq_max"].as_u64()), (Some("CU"), Some(1), Some(2)));
    // an idle queue is not listed
    assert!(queue(c, 5).is_none());
}

#[test]
fn wait_cycles_per_hold_reason() {
    let d = tempfile::tempdir().unwrap();
    let waits = |src: &str| -> Value {
        let p = text_program(&d, "w.txt", src);
        run_stats(&p, &[]).stats["wait_cycles"].clone()
    };
    let only = |w: &Value, reason: &str| -> u64 {
        for r in REASONS {
            if r != reason {
                assert_eq!(w[r], 0, "{reason}: unexpected {r} in {w}");
            }
        }
        w[reason].as_u64().unwrap()
    };
    // the Sync dispatches at cycle 0 and the Read is due at 1; the control channel's NOP of ten cycles puts
    // the Notify at cycle 10, which releases the Sync at 10 + `barrier.release_latency` (35) = 45: the Read
    // waits in cycles 1 to 44
    let w = waits(PARKED);
    assert_eq!(only(&w, "parked"), 44, "{w}");
    // the second Repeat waits for the first one's iterations (a lower bound: its exact count depends on the
    // Repeat's operand encoding and is not derived by hand here; the run gives 3)
    let w = waits(REPEAT_CHAIN);
    assert!(w["repeat"].as_u64().unwrap() >= 3, "{w}");
    // plus the cycle after its last iteration, in which the queue is still busy
    assert_eq!(w["busy"], 1, "{w}");
    for r in ["parked", "path", "loop", "configuration", "incomplete_text"] {
        assert_eq!(w[r], 0, "{w}");
    }
    // the Read behind a loop of four references waits for the loop
    let w = waits(LOOP_THEN_READ);
    assert_eq!(w["loop"], 2, "the loop's references at 0..3, the Read due at 2 dispatches at 4: {w}");
    for r in ["parked", "busy", "path", "repeat", "configuration", "incomplete_text"] {
        assert_eq!(w[r], 0, "{w}");
    }
}

/// `IfetchM` of two words brings 2 x 320 bytes (the `fetch` hook), and a parcel whose last unit has
/// not arrived yet waits as incomplete text (the `wait` hook through the counters).
#[test]
fn fetch_bytes_and_incomplete_text() {
    let profile = ots_profile::Profile::load("ref").unwrap();
    let mut m = Machine::new(&profile, 0, Box::new(NoLinks)).unwrap();
    // queue 4: IfetchM of words 0x10 and 0x11 of slice 4; the fetched text is 78 padding units, a Read
    // that continues the parcel (bundle bit) in units 78..79 and the NOP that ends it as unit 0 of the
    // second word: the fill rate is 4 units a cycle, so the NOP arrives a cycle after the Read
    m.load_image(4, &[0x01c0_0010]).unwrap();
    let mut units = vec![PADDING; 78];
    units.extend([0x9000_0010, 0]);
    let word = |us: &[u32]| us.iter().flat_map(|u| u.to_le_bytes()).collect::<Vec<u8>>();
    m.preload(4, 0x10, &word(&units)).unwrap();
    let mut second = vec![0x0000_0001];
    second.resize(80, PADDING);
    m.preload(4, 0x11, &word(&second)).unwrap();
    let mut counts = Counters::default();
    let mut sink = NoObserver;
    let sum = m.run(1000, &mut StatsObserver { inner: &mut sink, counts: &mut counts }).unwrap();
    assert!(sum.ticks >= 2);
    assert_eq!(counts.fetch_bytes, 640);
    assert_eq!(counts.dispatches[4], 2, "IfetchM, then the fetched parcel");
    assert_eq!(counts.wait_totals()[6], 1, "{:?}", counts.waits[4]);
}

#[test]
fn diagnostics_by_code() {
    let s = run_stats(&bench_program("b2"), &[]).stats;
    assert_eq!(s["diagnostics"], 3056);
    assert_eq!(s["diagnostics_by_code"]["OTS-E-TIM-001"], 3056);
    assert_eq!(s["diagnostics_by_code"].as_object().unwrap().len(), 1);
    assert_eq!(s["per_chip"][0]["first_diagnostic"]["code"], "OTS-E-TIM-001");
}

/// `--stats` changes neither the output nor the trace, and the counts do not depend on ticking the
/// idle cycles too.
#[test]
fn stats_do_not_change_the_run() {
    let d = tempfile::tempdir().unwrap();
    let progs = [
        text_program(&d, "parked.txt", PARKED),
        text_program(&d, "repeat.txt", REPEAT_CHAIN),
        text_program(&d, "loop.txt", LOOP_THEN_READ),
        bench_program("b1"),
        bench_program("b2"),
        bench_program("b3"),
    ];
    for p in &progs {
        let ps = p.to_str().unwrap();
        let plain = run_iss(&["run", "--profile", "ref", "--trace", "--lint", ps], "");
        assert_eq!(plain.status.code(), Some(0));
        let with = run_stats(p, &["--trace", "--lint"]);
        assert_eq!(plain.stdout, with.out.stdout, "{ps}: stdout");
        assert_eq!(plain.stderr, with.out.stderr, "{ps}: stderr");
        let every = run_stats(p, &["--trace", "--lint", "--tick-every-cycle"]);
        assert_eq!(plain.stdout, every.out.stdout, "{ps}: tick-every-cycle stdout");
        let strip = |v: &Value| {
            let mut v = v.clone();
            v.as_object_mut().unwrap().remove("wall_seconds");
            v.as_object_mut().unwrap().remove("ticks");
            for c in v["per_chip"].as_array_mut().unwrap() {
                c.as_object_mut().unwrap().remove("ticks");
            }
            v
        };
        assert_eq!(strip(&with.stats), strip(&every.stats), "{ps}: the counts do not depend on the idle ticks");
    }
}

/// One entry per chip, in chip order; the totals are the sums.
#[test]
fn stats_of_several_chips() {
    let d = tempfile::tempdir().unwrap();
    let p = text_program(&d, "parked.txt", PARKED);
    let r = run_stats(&p, &["--chips", "2", "--link-latency", "210"]);
    let s = r.stats;
    assert_eq!(s["chips"], 2);
    let chips = s["per_chip"].as_array().unwrap();
    assert_eq!(chips.len(), 2);
    assert_eq!(chips[0]["dispatches"], 3);
    assert_eq!(chips[1]["dispatches"], 0);
    assert_eq!(s["dispatches"], 3);
}

/// `--replay-runs 3` in a replay build: the first run records, the next two are served from the cache, and the
/// `--stats` document says so.
#[cfg(feature = "replay")]
#[test]
fn the_stats_count_the_runs_the_schedule_cache_served() {
    let d = tempfile::tempdir().unwrap();
    let p = text_program(&d, "loop.txt", LOOP_THEN_READ);
    let r = &run_stats(&p, &["--replay-runs", "3"]).stats["replay"];
    assert_eq!(r["enabled"], true);
    assert_eq!(
        (r["misses"].as_u64(), r["hits"].as_u64(), r["recordings"].as_u64(), r["schedules"].as_u64()),
        (Some(1), Some(2), Some(1), Some(1)),
        "{r}"
    );
    assert!(r["retained_bytes"].as_u64().unwrap() > 0);
    assert_eq!((r["divergences"].as_u64(), r["evictions"].as_u64()), (Some(0), Some(0)));
}

#[test]
fn wait_intervals_and_totals_retain_the_complete_timestamp_domain() {
    use tsp_iss::observe::{Hold, Observer};
    let mut counts = Counters::default();
    let mut inner = NoObserver;
    let mut obs = StatsObserver { inner: &mut inner, counts: &mut counts };
    for queue in 0..144 {
        obs.wait(0, queue, Hold::IncompleteText, 0);
        obs.wait(u64::MAX, queue, Hold::IncompleteText, 0);
    }
    obs.wait(u64::MAX, 0xffff, Hold::IncompleteText, u64::MAX);
    assert!(counts.waits[..144].iter().all(|waits| waits[6] == 1u128 << 64));
    assert_eq!(counts.waits[144][6], 1);
    assert_eq!(counts.wait_totals()[6], 144 * (1u128 << 64) + 1);
}
