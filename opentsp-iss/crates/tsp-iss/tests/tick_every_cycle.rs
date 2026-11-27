//! `--tick-every-cycle`: the machine ticks every cycle instead of jumping to
//! `model_next_event`. The model claims that ticking an idle cycle changes nothing; these tests
//! check the claim: the exit code, stdout and stderr of a run are identical with and without the
//! option, for
//!
//! - every golden vector of `ref` and `mini` (the vector's text, preloads and dumps as a program);
//! - programs with long NOP gaps, Repeat, fetch latency, `--max-cycles` stops and link arrivals
//!   (several chips, every latency option, the `link_tight` profile);
//! - the library: `Machine::run_until` with `set_tick_every_cycle`.
mod common;
use common::*;
use ots_isa::{encode, Insn};
use ots_profile::Profile;
use serde_json::Value;
use std::fmt::Write as _;
use tsp_iss::harness::output::TraceObserver;
use tsp_iss::links::NoLinks;
use tsp_iss::machine::{Machine, RunError};

const EVERY: &str = "--tick-every-cycle";

/// `args` with and without the option: both runs' exit code, stdout and stderr must be equal.
fn same_both_ways(args: &[&str], stdin: &str) -> Result<(), String> {
    let plain = run_iss(args, stdin);
    let mut with: Vec<&str> = args.to_vec();
    with.push(EVERY);
    let every = run_iss(&with, stdin);
    if plain.status.code() == every.status.code() && plain.stdout == every.stdout && plain.stderr == every.stderr {
        return Ok(());
    }
    Err(format!(
        "{args:?}\n{stdin}\nplain (exit {:?}):\n{}\n{}\nevery cycle (exit {:?}):\n{}\n{}",
        plain.status.code(),
        text(&plain.stdout),
        text(&plain.stderr),
        every.status.code(),
        text(&every.stdout),
        text(&every.stderr)
    ))
}

fn units_of(i: &Insn) -> Vec<u32> {
    let w = encode(i);
    if i.entry().width == 64 {
        vec![(w >> 32) as u32, w as u32]
    } else {
        vec![w as u32]
    }
}

fn q_line(q: u16, insns: &[Insn]) -> String {
    let mut s = format!("Q {q:x}");
    for i in insns {
        for u in units_of(i) {
            write!(s, " {u:x}").unwrap();
        }
    }
    s.push('\n');
    s
}

fn nop(n: u32) -> Insn {
    Insn::NOP { n }
}

/// A vector of a conformance file as a program: its `Q`, `S` and `D` lines.
fn vector_program(v: &Value) -> String {
    let mut p = String::new();
    for (q, units) in v["text"].as_object().unwrap() {
        writeln!(p, "Q {:x} {}", q.parse::<u16>().unwrap(), units.as_str().unwrap()).unwrap();
    }
    for (k, w) in v["sram"].as_object().unwrap() {
        let (s, a) = k.split_once(':').unwrap();
        let hex = w.as_str().unwrap();
        let bytes: Vec<&str> = (0..hex.len() / 2).map(|i| &hex[2 * i..2 * i + 2]).collect();
        writeln!(p, "S {:x} {:x} {}", s.parse::<u32>().unwrap(), a.parse::<u32>().unwrap(), bytes.join(" ")).unwrap();
    }
    for d in v["dump"].as_array().unwrap() {
        let (s, a) = d.as_str().unwrap().split_once(':').unwrap();
        writeln!(p, "D {:x} {:x}", s.parse::<u32>().unwrap(), a.parse::<u32>().unwrap()).unwrap();
    }
    p
}

fn golden_vectors(name: &str) {
    let path = spec().join("conformance/model").join(format!("{name}.json"));
    let doc: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let progs: Vec<String> = doc["vectors"].as_array().unwrap().iter().map(vector_program).collect();
    assert!(progs.len() >= 500, "{name}: only {} vectors", progs.len());
    let cfg = config(name);
    let bad = parallel(&progs, |p| same_both_ways(&[cfg.to_str().unwrap(), "--trace"], p));
    assert!(bad.is_empty(), "{} of {} vectors of {name} differ; first:\n{}", bad.len(), progs.len(), bad[0]);
}

#[test]
fn every_golden_vector_of_ref_is_the_same_ticking_every_cycle() {
    golden_vectors("ref");
}

#[test]
fn every_golden_vector_of_mini_is_the_same_ticking_every_cycle() {
    golden_vectors("mini");
}

fn single_chip(cfg: &str, extra: &[&str], progs: &[String]) {
    let mut args = vec![cfg, "--trace"];
    args.extend_from_slice(extra);
    let bad = parallel(progs, |p| same_both_ways(&args, p));
    assert!(bad.is_empty(), "{} of {} programs differ with {extra:?}; first:\n{}", bad.len(), progs.len(), bad[0]);
}

fn gap_programs() -> Vec<String> {
    let read = |a, d| Insn::Read { addr: a, dst: d, thread: 0 };
    let write = |a, s| Insn::Write { addr: a, src: s, thread: 0 };
    vec![
        // long NOP gaps: nothing happens for hundreds of cycles, twice, on two queues
        q_line(0, &[nop(900), read(3, 1), nop(700), write(5, 1)]) + &q_line(44, &[nop(400), read(2, 2)]),
        // Repeat: iterations separated by a delay, and a Repeat of a NOP
        q_line(0, &[read(1, 1), Insn::Repeat { n: 5, d: 3 }, nop(50), write(2, 1), Insn::Repeat { n: 2, d: 0 }]),
        q_line(88, &[nop(30), Insn::Repeat { n: 4, d: 7 }, nop(1), Insn::Repeat { n: 3, d: 2 }]),
        // fetch latency: the text arrives after the fetch latency; Sync and Notify park the queues
        q_line(0, &[Insn::IfetchM { addr: 0 }, nop(300), Insn::Ifetch { stream: 4 }])
            + &q_line(104, &[Insn::Sync, nop(5)])
            + &q_line(0xffff, &[nop(1000), Insn::Notify]),
        // diagnostics in an idle stretch and after it
        q_line(100, &[Insn::Add { r#type: 0, dst: 3, lhs: 1, rhs: 2, sg4: 0, lhs_chain: 0, rhs_chain: 0 }, nop(600)])
            + &q_line(130, &[nop(250), Insn::Deskew, nop(10)]),
        // a text of padding only, and an empty program
        "Q 0 ffffffff ffffffff\n".to_string(),
        String::new(),
    ]
}

#[test]
fn long_gaps_repeat_and_fetch_latency_are_the_same_ticking_every_cycle() {
    for name in ["ref", "mini", "test/ref_slow", "test/link_tight"] {
        single_chip(config(name).to_str().unwrap(), &[], &gap_programs());
    }
}

#[test]
fn a_max_cycles_stop_is_the_same_ticking_every_cycle() {
    let cfg = config("ref");
    for limit in ["0", "1", "100", "899", "900", "901", "1600", "100000"] {
        single_chip(cfg.to_str().unwrap(), &["--max-cycles", limit], &gap_programs());
    }
}

/// The links: a Send on one chip, a Receive near the arrival on its peer, and bursts that overrun
/// the receive FIFO of `link_tight`.
fn link_programs(lat: u64, burst: bool) -> Vec<String> {
    let send = |peer, link| Insn::Send { chip_id: peer, link, stream: 3 };
    let recv = |peer, link| Insn::Receive { chip_id: peer, link, stream: 5 };
    let mut v = Vec::new();
    for delta in [-2i64, -1, 0, 1, 2] {
        let wait = (lat as i64 + delta).max(1) as u32;
        let mut p = String::from("C 0\n");
        p += &q_line(126, &[nop(4), send(1, 0)]);
        p += "C 1\n";
        p += &q_line(128, &[nop(wait + 4), recv(0, 0)]);
        v.push(p);
    }
    if burst {
        let mut p = String::from("C 0\n");
        p += &q_line(126, &[nop(2), send(1, 0), send(1, 0), send(1, 0), send(1, 0), nop(1), send(1, 0)]);
        p += "C 1\n";
        p += &q_line(128, &[nop(lat as u32 + 400), recv(0, 0)]);
        v.push(p);
    }
    v
}

#[test]
fn link_arrivals_are_the_same_ticking_every_cycle() {
    for (name, lat, burst) in [("ref", 216, false), ("mini", 208, false), ("test/link_tight", 216, true)] {
        let cfg = config(name);
        let cfg = cfg.to_str().unwrap();
        let progs = link_programs(lat, burst);
        for extra in [
            vec!["--chips", "2"],
            vec!["--chips", "2", "--link-latency", "212"],
            vec!["--chips", "2", "--link-latency-seed", "7"],
            vec!["--chips", "2", "--link-jitter-seed", "9"],
            vec!["--chips", "2", "--max-cycles", "300"],
        ] {
            single_chip(cfg, &extra, &progs);
        }
    }
}

#[test]
fn run_until_with_the_option_ticks_every_cycle_and_reports_the_same() {
    let profile = Profile::load("ref").unwrap();
    let mk = |every: bool| {
        let mut m = Machine::new(&profile, 0, Box::new(NoLinks)).unwrap();
        let prog =
            [nop(40), Insn::Read { addr: 1, dst: 2, thread: 0 }, nop(25), Insn::Write { addr: 3, src: 2, thread: 0 }];
        let units: Vec<u32> = prog.iter().flat_map(units_of).collect();
        m.load_image(0, &units).unwrap();
        m.set_tick_every_cycle(every);
        let mut out = Vec::new();
        let sum = {
            let mut obs = TraceObserver::new(&mut out, true);
            m.run(1_000, &mut obs).unwrap()
        };
        (sum, out)
    };
    let ((plain, out_plain), (every, out_every)) = (mk(false), mk(true));
    assert_eq!(out_plain, out_every);
    assert!(plain.ticks < 20, "the plain run ticks only the events: {}", plain.ticks);
    let last = every.last_cycle.unwrap();
    assert_eq!((every.ticks, Some(last)), (last + 1, plain.last_cycle), "every cycle from 0 to the last event");
    // a window ends the run early and the next call carries on from where it stopped
    let mut m = Machine::new(&profile, 0, Box::new(NoLinks)).unwrap();
    let prog: Vec<u32> = [nop(30), Insn::Read { addr: 1, dst: 2, thread: 0 }].iter().flat_map(units_of).collect();
    m.load_image(0, &prog).unwrap();
    m.set_tick_every_cycle(true);
    let mut sink = Vec::new();
    let mut obs = TraceObserver::new(&mut sink, false);
    assert_eq!(m.run_until(10, 1_000, &mut obs).unwrap().ticks, 10);
    assert_eq!(m.next_tick(), Some(30), "the model's next event is unchanged");
    assert_eq!(m.run_until(20, 1_000, &mut obs).unwrap().ticks, 10);
    // the limit is reported at the cycle of the next event, as without the option
    match m.run(25, &mut obs) {
        Err(RunError::MaxCycles { cycle: 30, limit: 25 }) => {}
        r => panic!("{r:?}"),
    }
}

#[test]
fn the_option_reaches_every_chip_of_a_run() {
    let dir = tempfile::tempdir().unwrap();
    let prog = dir.path().join("p.txt");
    // the program is chip 0's; the other chips of a run have nothing to do
    std::fs::write(&prog, q_line(0, &[nop(60), Insn::Read { addr: 1, dst: 2, thread: 0 }])).unwrap();
    let ticks = |extra: &[&str], chips: &str| -> Vec<u64> {
        let stats = dir.path().join("stats.json");
        let st = format!("--stats={}", stats.display());
        let mut args = vec!["run", "--profile", "ref", "--chips", chips, st.as_str(), prog.to_str().unwrap()];
        args.extend_from_slice(extra);
        let out = run_iss(&args, "");
        assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
        let doc: Value = serde_json::from_slice(&std::fs::read(&stats).unwrap()).unwrap();
        doc["per_chip"].as_array().unwrap().iter().map(|c| c["ticks"].as_u64().unwrap()).collect()
    };
    for chips in ["1", "2"] {
        let (plain, every) = (ticks(&[], chips), ticks(&[EVERY], chips));
        assert_eq!(plain.len(), chips.parse::<usize>().unwrap());
        assert!(plain.iter().all(|t| *t < 10), "{plain:?}");
        assert!(every[0] > 60, "{every:?}");
        assert!(plain.iter().zip(&every).all(|(p, e)| e >= p), "{plain:?} {every:?}");
    }
}
