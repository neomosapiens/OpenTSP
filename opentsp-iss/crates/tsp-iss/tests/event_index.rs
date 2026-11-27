//! The driver's event index (`Driver::wake`, `Driver::ev`) agrees with the full scan of the queues at every tick.
//!
//! A tick runs the body of a data queue only when the index says it can act, and `model_next_event` reuses each
//! queue's last answer until the queue or the barrier changes. A missed invalidation is a queue that sleeps through
//! its dispatch, or a skipped cycle. With `Driver::index_check` set, every tick asserts that each skipped queue
//! would have done nothing, and every `model_next_event` asserts equality with `model_next_event_scan`; these tests
//! run programs to the end that way, under an observer that tracks waits and one that does not.
mod common;
use common::*;
use ots_isa::{encode, Insn};
use ots_profile::Profile;
use serde_json::Value;
use tsp_iss::harness::output::TraceObserver;
use tsp_iss::harness::stats::{Counters, StatsObserver};
use tsp_iss::links::NoLinks;
use tsp_iss::machine::Machine;

fn units_of(i: &Insn) -> Vec<u32> {
    let w = encode(i);
    if i.entry().width == 64 {
        vec![(w >> 32) as u32, w as u32]
    } else {
        vec![w as u32]
    }
}

fn load(m: &mut Machine, queue: u16, insns: &[Insn]) {
    let units: Vec<u32> = insns.iter().flat_map(units_of).collect();
    m.load_image(queue, &units).unwrap();
}

/// Run `fresh()` to the end twice with the index checked, once under a trace observer and once under the
/// statistics observer (which makes the driver report waits); both runs must also give the same trace and
/// statistics as each other's unchecked run. Returns the ticks.
fn checked_run(fresh: &dyn Fn() -> Machine, what: &str) -> u64 {
    let run = |check: bool, stats: bool| {
        let mut m = fresh();
        m.edit_state(|state| state.driver.index_check = check).unwrap();
        let mut sink = Vec::new();
        let mut inner = TraceObserver::new(&mut sink, true);
        let mut counts = Counters::default();
        let r = if stats {
            let mut obs = StatsObserver { inner: &mut inner, counts: &mut counts };
            m.run(2_000_000, &mut obs)
        } else {
            m.run(2_000_000, &mut inner)
        };
        let r = r.map(|s| s.ticks).map_err(|e| e.to_string());
        drop(inner);
        (r, sink, counts.wait_totals(), counts.dispatches)
    };
    let mut ticks = 0;
    for stats in [false, true] {
        let checked = run(true, stats);
        let plain = run(false, stats);
        assert_eq!(checked, plain, "{what} (stats {stats}): the checked run differs");
        ticks = checked.0.clone().unwrap_or(0);
    }
    ticks
}

/// Programs that exercise each cause of "the queue can act": a fetch filling an IQ under a held head, Repeat,
/// a long timing prefix, Syncs released late by the control channel and by another queue, AGT loops.
#[test]
fn the_index_agrees_with_the_scan_on_the_fetch_repeat_barrier_and_loop_shapes() {
    let profile = Profile::load("ref").unwrap();
    let read = |a, d| Insn::Read { addr: a, dst: d, thread: 0 };
    let write = |a, s| Insn::Write { addr: a, src: s, thread: 0 };
    let nop = |n| Insn::NOP { n };
    type Images = Vec<(u16, Vec<Insn>)>;
    let progs: Vec<(&str, Images)> = vec![
        ("fetch under a Repeat", vec![(0, vec![Insn::IfetchM { addr: 0 }, Insn::Repeat { n: 40, d: 5 }, read(3, 1)])]),
        ("fetch under a timing prefix", vec![(0, vec![Insn::IfetchM { addr: 0 }, nop(120), read(3, 1), write(5, 1)])]),
        (
            "Sync released late by the control channel",
            vec![(0, vec![Insn::IfetchM { addr: 0 }, Insn::Sync, read(3, 1)]), (0xffff, vec![nop(200), Insn::Notify])],
        ),
        (
            "Syncs on several queues, Notify from a data queue",
            vec![
                (0, vec![Insn::Sync, read(3, 1), Insn::Repeat { n: 6, d: 3 }]),
                (1, vec![nop(5), Insn::Sync, write(4, 2)]),
                (2, vec![nop(90), Insn::Notify, read(7, 3)]),
            ],
        ),
        ("control channel Sync parks every queue", vec![(0, vec![nop(3), read(1, 1)]), (0xffff, vec![Insn::Sync])]),
        (
            "two fetches on one queue",
            vec![(0, vec![Insn::IfetchM { addr: 0 }, Insn::Ifetch { stream: 4 }, nop(60), read(1, 1)])],
        ),
        ("a bundle of references", vec![(0, vec![nop(30), read(1, 1), write(2, 1), Insn::Repeat { n: 8, d: 4 }])]),
    ];
    for (what, prog) in &progs {
        let fresh = || {
            let mut m = Machine::new(&profile, 0, Box::new(NoLinks)).unwrap();
            for (q, insns) in prog {
                load(&mut m, *q, insns);
            }
            m
        };
        assert!(checked_run(&fresh, what) > 0, "{what}: nothing ran");
    }
}

/// Every golden vector of a profile, run to the end with the index checked.
fn golden_vectors(name: &str) {
    let path = spec().join("conformance/model").join(format!("{name}.json"));
    let doc: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let profile = Profile::load(name).unwrap();
    let vectors = doc["vectors"].as_array().unwrap();
    assert!(vectors.len() >= 250, "{name}: only {} vectors", vectors.len());
    for v in vectors {
        let what = format!("{name} {}", v["name"].as_str().unwrap_or("?"));
        let fresh = || {
            let mut m = Machine::new(&profile, 0, Box::new(NoLinks)).unwrap();
            for (q, units) in v["text"].as_object().unwrap() {
                let hex = units.as_str().unwrap();
                let words: Vec<u32> = hex.split_whitespace().map(|u| u32::from_str_radix(u, 16).unwrap()).collect();
                m.load_image(q.parse().unwrap(), &words).unwrap();
            }
            for (k, w) in v["sram"].as_object().unwrap() {
                let (s, a) = k.split_once(':').unwrap();
                let hex = w.as_str().unwrap();
                let bytes: Vec<u8> =
                    (0..hex.len() / 2).map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap()).collect();
                m.preload(s.parse().unwrap(), a.parse().unwrap(), &bytes).unwrap();
            }
            m
        };
        checked_run(&fresh, &what);
    }
}

#[test]
fn the_index_agrees_with_the_scan_on_every_golden_vector_of_ref() {
    golden_vectors("ref");
}

#[test]
fn the_index_agrees_with_the_scan_on_every_golden_vector_of_mini() {
    golden_vectors("mini");
}
