//! The parse of the IQs ([`Iq::head`]) holds what a fresh scan finds, at every cycle of a run.
//!
//! The driver asks each queue for its head parcel and the parcel's slots several times a cycle (the tick and every
//! candidate cycle of `model_next_event`), so the IQ decodes and classifies the text once, when it enters
//! ([`Iq::parse`]), and moves on to the next parcel when one is popped. A parse that misses a unit or a pop is a
//! stale parcel: the machine would dispatch text the queue no longer holds. These tests run programs cycle by
//! cycle and compare the parsed head of every queue with a fresh scan after every tick, on the fetch, Repeat and
//! barrier shapes (the IQ grows under the head while the head waits) and on the golden vectors of both profiles.
mod common;
use common::*;
use ots_isa::{encode, Insn};
use ots_profile::Profile;
use serde_json::Value;
use tsp_iss::harness::output::TraceObserver;
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

fn nop(n: u32) -> Insn {
    Insn::NOP { n }
}

/// Run `m` one cycle at a time (at most `limit` ticks) and check after every tick that every queue's parsed head is
/// the head a fresh scan finds. Returns the ticks run.
fn parse_is_fresh_every_cycle(m: &mut Machine, limit: u64, what: &str) -> u64 {
    let mut sink = Vec::new();
    let mut obs = TraceObserver::new(&mut sink, false);
    let mut ticks = 0;
    while let Some(t) = m.next_tick() {
        if ticks >= limit {
            break;
        }
        ticks += m.run_until((t + 1).try_into().unwrap(), u64::MAX, &mut obs).expect("run").ticks;
        for q in 0..=144 {
            assert!(
                m.iq().head_matches_scan(m.isa(), q),
                "{what}: queue {q} after cycle {t}: the parsed head is not the queue's head"
            );
        }
    }
    ticks
}

/// One queue's image.
struct Image {
    queue: u16,
    units: Vec<u32>,
}

fn image(queue: u16, insns: &[Insn]) -> Image {
    Image { queue, units: insns.iter().flat_map(units_of).collect() }
}

fn machine(profile: &Profile, images: &[Image]) -> Machine {
    let mut m = Machine::new(profile, 0, Box::new(NoLinks)).unwrap();
    for i in images {
        m.load_image(i.queue, &i.units).unwrap();
    }
    m
}

/// Programs whose IQs grow while the head parcel waits: a fetch delivers units cycle by cycle under a head held by
/// a Repeat, by a long timing prefix, or by a Sync the control channel releases late.
#[test]
fn the_parsed_head_is_the_queue_s_head_while_a_fetch_fills_the_iq() {
    let profile = Profile::load("ref").unwrap();
    let read = |a, d| Insn::Read { addr: a, dst: d, thread: 0 };
    let write = |a, s| Insn::Write { addr: a, src: s, thread: 0 };
    let progs: Vec<(&str, Vec<Image>)> = vec![
        (
            "fetch under a Repeat",
            vec![image(0, &[Insn::IfetchM { addr: 0 }, Insn::Repeat { n: 40, d: 5 }, read(3, 1)])],
        ),
        (
            "fetch under a timing prefix",
            vec![image(0, &[Insn::IfetchM { addr: 0 }, nop(120), read(3, 1), write(5, 1)])],
        ),
        (
            "fetch under a Sync released late",
            vec![
                image(0, &[Insn::IfetchM { addr: 0 }, Insn::Sync, read(3, 1)]),
                image(0xffff, &[nop(200), Insn::Notify]),
            ],
        ),
        (
            "two fetches on one queue",
            vec![image(0, &[Insn::IfetchM { addr: 0 }, Insn::Ifetch { stream: 4 }, nop(60), read(1, 1)])],
        ),
        ("a bundle of references", vec![image(0, &[nop(30), read(1, 1), write(2, 1), Insn::Repeat { n: 8, d: 4 }])]),
    ];
    for (what, prog) in &progs {
        let mut m = machine(&profile, prog);
        let ticks = parse_is_fresh_every_cycle(&mut m, 4_000, what);
        assert!(ticks > 0, "{what}: nothing ran");
    }
}

/// Every golden vector of a profile, cycle by cycle: the vector's text images and SRAM preloads, the first 400 ticks.
fn golden_vectors(name: &str) {
    let path = spec().join("conformance/model").join(format!("{name}.json"));
    let doc: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let profile = Profile::load(name).unwrap();
    let vectors = doc["vectors"].as_array().unwrap();
    assert!(vectors.len() >= 500, "{name}: only {} vectors", vectors.len());
    for v in vectors {
        let what = v["name"].as_str().unwrap_or("?");
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
        parse_is_fresh_every_cycle(&mut m, 400, &format!("{name} {what}"));
    }
}

#[test]
fn the_parsed_head_is_the_queue_s_head_on_every_golden_vector_of_ref() {
    golden_vectors("ref");
}

#[test]
fn the_parsed_head_is_the_queue_s_head_on_every_golden_vector_of_mini() {
    golden_vectors("mini");
}
