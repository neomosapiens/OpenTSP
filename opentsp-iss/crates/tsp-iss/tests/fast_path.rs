//! The common-case driver paths (`Driver::fast_on`) give the trace, the statistics and the tick count of the general
//! code, and the assumptions they make hold (`Driver::fast_check`).
//!
//! A tick takes three shortcuts when the parcel it dispatches is the common one (no Sync, Notify, Repeat, loop
//! clash, configuration record or fetch, and an idle AGT): the dispatch skips the checks that the common case
//! cannot fail, the hold test is the short one (`held_common`), and an AGT configuration that nothing can continue
//! is closed at the end of the tick that opened it, with the next tick made all the same (`cfg_tick`). With
//! `fast_check` set the shortcuts also compute what they leave out and assert that it is what they assumed: the
//! checked run panics on a wrong guard. These tests run shapes and every golden vector to the end in three modes
//! (the shortcuts off, on, on and checked) and require identical output, and that the shortcuts were taken.
mod common;
use common::*;
use ots_isa::{encode, Insn};
use ots_profile::Profile;
use serde_json::Value;
use tsp_iss::driver::{bits_from, pop_bit};
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

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Off,
    On,
    Checked,
}

type Out = (Result<u64, String>, Vec<u8>, [u128; 7], Vec<u64>);

/// Run `fresh()` to the end in `mode`, with a statistics observer (the driver reports waits) or without; returns the
/// ticks (or the error), the trace, the wait totals and dispatches, and the two counters of the shortcuts taken.
fn run_in(fresh: &dyn Fn() -> Machine, mode: Mode, stats: bool) -> (Out, (u64, u64)) {
    let mut m = fresh();
    m.edit_state(|state| state.driver.fast_on = mode != Mode::Off).unwrap();
    m.edit_state(|state| state.driver.fast_check = mode == Mode::Checked).unwrap();
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
    ((r, sink, counts.wait_totals(), counts.dispatches.clone()), (m.driver().fast_dispatches, m.driver().early_closes))
}

/// The three modes agree, with and without statistics. Returns the shortcuts taken in the checked run.
fn agree(fresh: &dyn Fn() -> Machine, what: &str) -> (u64, u64) {
    let mut taken = (0, 0);
    for stats in [false, true] {
        let (off, none) = run_in(fresh, Mode::Off, stats);
        let (on, _) = run_in(fresh, Mode::On, stats);
        let (checked, t) = run_in(fresh, Mode::Checked, stats);
        assert_eq!(none, (0, 0), "{what}: the shortcuts are off");
        assert_eq!(off, on, "{what} (stats {stats}): shortcuts on differ from off");
        assert_eq!(off, checked, "{what} (stats {stats}): checked differs from off");
        taken = t;
    }
    taken
}

type Images = Vec<(u16, Vec<Insn>)>;

fn shapes() -> Vec<(&'static str, Images)> {
    let read = |a, d| Insn::Read { addr: a, dst: d, thread: 0 };
    let write = |a, s| Insn::Write { addr: a, src: s, thread: 0 };
    let nop = |n| Insn::NOP { n };
    vec![
        ("a bundle of references", vec![(0, vec![nop(30), read(1, 1), write(2, 1), Insn::Repeat { n: 8, d: 4 }])]),
        ("reads one after another", vec![(0, vec![read(1, 1), read(2, 2), read(3, 3), nop(5), read(4, 1)])]),
        ("a read, a gap, a read", vec![(0, vec![read(1, 1), nop(1), read(2, 2), nop(7), write(5, 1)])]),
        (
            "a read with a configuration record",
            vec![(0, vec![read(1, 1), Insn::Iterations { n: 3 }, Insn::Step { a: 2 }, nop(40), read(2, 2)])],
        ),
        (
            "a read and a loop record in the next cycle",
            vec![(0, vec![read(1, 1), nop(0), Insn::Iterations { n: 2 }, read(2, 2), nop(30)])],
        ),
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
        (
            "many queues reading together",
            (0..40).map(|q| (q, vec![nop(q as u32 % 5), read(1, 1), read(2, 2), nop(3), write(3, 1)])).collect(),
        ),
    ]
}

#[test]
fn the_shortcuts_agree_with_the_general_code_on_the_shapes() {
    let profile = Profile::load("ref").unwrap();
    let mut common = (0, 0);
    for (what, prog) in &shapes() {
        let fresh = || {
            let mut m = Machine::new(&profile, 0, Box::new(NoLinks)).unwrap();
            for (q, insns) in prog {
                load(&mut m, *q, insns);
            }
            m
        };
        let t = agree(&fresh, what);
        common = (common.0 + t.0, common.1 + t.1);
    }
    assert!(common.0 > 0, "no parcel took the common dispatch");
    assert!(common.1 > 0, "no configuration was closed early");
}

/// A read that nothing continues is closed early, and one a configuration record follows is not.
#[test]
fn a_configuration_is_closed_early_only_when_nothing_can_continue_it() {
    let profile = Profile::load("ref").unwrap();
    let read = |a, d| Insn::Read { addr: a, dst: d, thread: 0 };
    let nop = |n| Insn::NOP { n };
    let closes = |prog: &[Insn]| {
        let fresh = || {
            let mut m = Machine::new(&profile, 0, Box::new(NoLinks)).unwrap();
            load(&mut m, 0, prog);
            m
        };
        let ((r, ..), (_, early)) = run_in(&fresh, Mode::Checked, false);
        r.expect("run");
        early
    };
    assert_eq!(closes(&[read(1, 1), nop(9), read(2, 2)]), 2, "two reads, each followed by a gap");
    assert_eq!(closes(&[read(1, 1), Insn::Iterations { n: 3 }, nop(9)]), 0, "a configuration record follows");
}

/// Text that a fetch delivers while a read is dispatched: the read sits at the end of one fetched word, and the
/// configuration record that continues it in the next, which arrives one cycle later. The queue's IQ is empty
/// after the read is popped, so the early end of its configuration must see the arrival.
#[test]
fn text_that_arrives_in_the_next_cycle_continues_a_configuration() {
    let profile = Profile::load("ref").unwrap();
    let read = Insn::Read { addr: 1, dst: 1, thread: 0 };
    let (mut closed, mut kept) = (0, 0);
    for k in 0..8usize {
        for gap in [0usize, 1, 3] {
            let nop = encode(&Insn::NOP { n: 1 }) as u32;
            let mut text: Vec<u32> = vec![nop; k];
            text.extend(units_of(&read));
            text.extend(vec![nop; gap]);
            text.extend(units_of(&Insn::Iterations { n: 3 }));
            text.extend(units_of(&Insn::Step { a: 2 }));
            text.resize(80, 0xFFFF_FFFF); // the padding of a fetched word
            let fresh = || {
                let mut m = Machine::new(&profile, 0, Box::new(NoLinks)).unwrap();
                load(&mut m, 0, &[Insn::IfetchM { addr: 1 }]);
                let bytes: Vec<u8> = text.iter().flat_map(|u| u.to_le_bytes()).collect();
                m.preload(0, 1, &bytes).unwrap();
                m
            };
            let (_, early) = agree(&fresh, &format!("fetched read, {k} units in, gap {gap}"));
            closed += (early > 0) as u64;
            kept += (early == 0) as u64;
        }
    }
    assert!(
        closed > 0 && kept > 0,
        "the sweep must have shapes that close early ({closed}) and shapes that do not ({kept})"
    );
}

fn golden_vectors(name: &str) {
    let path = spec().join("conformance/model").join(format!("{name}.json"));
    let doc: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let profile = Profile::load(name).unwrap();
    let vectors = doc["vectors"].as_array().unwrap();
    assert!(vectors.len() >= 250, "{name}: only {} vectors", vectors.len());
    let mut taken = (0, 0);
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
        let t = agree(&fresh, &what);
        taken = (taken.0 + t.0, taken.1 + t.1);
    }
    assert!(taken.0 > 0 && taken.1 > 0, "{name}: the shortcuts were not taken: {taken:?}");
}

#[test]
fn the_shortcuts_agree_with_the_general_code_on_every_golden_vector_of_ref() {
    golden_vectors("ref");
}

#[test]
fn the_shortcuts_agree_with_the_general_code_on_every_golden_vector_of_mini() {
    golden_vectors("mini");
}

/// The barrier's parked counter (`Barrier::quiet`) is what a scan of the parked queues finds, after every tick.
#[test]
fn the_barrier_counter_equals_the_scan_after_every_tick() {
    let profile = Profile::load("ref").unwrap();
    let read = |a, d| Insn::Read { addr: a, dst: d, thread: 0 };
    let nop = |n| Insn::NOP { n };
    let progs: Vec<Vec<(u16, Vec<Insn>)>> = vec![
        vec![(0, vec![Insn::Sync, read(3, 1)]), (0xffff, vec![nop(60), Insn::Notify])],
        vec![(0, vec![nop(3), read(1, 1)]), (0xffff, vec![Insn::Sync])],
        vec![
            (0, vec![Insn::Sync, read(3, 1)]),
            (1, vec![nop(5), Insn::Sync, read(4, 2)]),
            (2, vec![nop(90), Insn::Notify, read(7, 3)]),
        ],
    ];
    let mut parked_seen = false;
    for prog in &progs {
        let mut m = Machine::new(&profile, 0, Box::new(NoLinks)).unwrap();
        for (q, insns) in prog {
            load(&mut m, *q, insns);
        }
        let mut sink = Vec::new();
        let mut obs = TraceObserver::new(&mut sink, false);
        while let Some(t) = m.next_tick() {
            m.run_until((t + 1).try_into().unwrap(), u64::MAX, &mut obs).expect("run");
            let b = &m.driver().barrier;
            assert_eq!(b.parked_counter(), b.parked_count_scan(), "after cycle {t}");
            parked_seen |= !b.quiet();
        }
    }
    assert!(parked_seen, "no queue was ever parked");
}

#[test]
fn the_bit_set_helpers_walk_and_cut_the_data_queues() {
    let mut set = [0u64; 3];
    for q in [0usize, 5, 63, 64, 100, 127, 128, 143] {
        set[q >> 6] |= 1 << (q & 63);
    }
    let cut = bits_from(&set, 64);
    assert_eq!(cut, [0, set[1], set[2]], "from the second word");
    let cut = bits_from(&set, 6);
    assert_eq!(cut[0], set[0] & !0x3f, "within a word");
    assert_eq!(bits_from(&set, 1)[0], set[0] & !1, "the queue before the cut is dropped");
    assert_eq!(bits_from(&set, 5)[0], set[0] & !0x1f, "the queue at the cut is kept");
    assert_eq!(bits_from(&set, 65)[1], set[1] & !1, "the same in the second word");
    let cut = bits_from(&set, 144);
    assert_eq!(cut, [0, 0, 0]);
    let mut walk = set;
    let mut got = Vec::new();
    while let Some(q) = pop_bit(&mut walk) {
        got.push(q);
    }
    assert_eq!(got, [0, 5, 63, 64, 100, 127, 128, 143]);
    assert_eq!(walk, [0, 0, 0]);
    assert_eq!(pop_bit(&mut walk), None);
}
