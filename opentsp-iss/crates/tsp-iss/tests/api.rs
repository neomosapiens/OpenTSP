//! The library surface of `tsp-iss` that the command-line tests do not reach, run in-process:
//! the geometry's own checks (the harness rejects the same values earlier, with the configuration
//! schema), the link plan's range checks, the run of several chips whose links have no latency,
//! the machine's accessors and error texts, and the defaults of `LinkPort` and `Observer`.
//! Each test states the case of `tsp_sim.c` or of the model it stands for.
mod common;
use common::*;
use ots_profile::Profile;
use tsp_iss::geom::MAX_LANES;
use tsp_iss::harness::cli::{LinkMode, Options};
use tsp_iss::harness::input::{ChipInput, Preload, Program};
use tsp_iss::harness::links::LinkPlan;
use tsp_iss::harness::{execute, load_config};
use tsp_iss::layout::{inward_row, outward_row};
use tsp_iss::observe::{Hold, NoObserver, Observer, Row};
use tsp_iss::{Diag, Geom, LinkPort, LoadError, Machine, NoLinks, RunError};

fn profile(name: &str) -> Profile {
    Profile::load(name).unwrap()
}

fn machine(name: &str) -> Machine {
    Machine::new(&profile(name), 3, Box::new(NoLinks)).unwrap()
}

/// A profile the schema would refuse, built without the check (the library does not repeat it).
fn mutated(name: &str, f: impl FnOnce(&mut serde_json::Value)) -> Profile {
    let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(config(name)).unwrap()).unwrap();
    f(&mut v);
    Profile::from_bytes("mutated", &serde_json::to_vec(&v).unwrap()).unwrap()
}

#[test]
fn geometry_checks_lanes_and_link_latency() {
    let g = Geom::new(&profile("ref"));
    assert_eq!(g.check_lanes(), Ok(()));
    assert_eq!(g.check_link_latency(), Ok(()));
    for vlanes in [0, MAX_LANES as u64 + 1] {
        let bad = Geom { vlanes, ..g.clone() };
        let e = bad.check_lanes().unwrap_err();
        assert!(e.starts_with("geometry.superlanes: ") && e.contains("at most 512"), "{e}");
    }
    assert_eq!(Geom { vlanes: 1, ..g.clone() }.check_lanes(), Ok(()));
    assert_eq!(Geom { vlanes: 320, ..g.clone() }.check_lanes(), Ok(()));
    for (min, nom, max) in [(209, 208, 228), (209, 229, 228), (230, 216, 228)] {
        let bad = Geom { link_latency_min: min, link_latency: nom, link_latency_max: max, ..g.clone() };
        let e = bad.check_link_latency().unwrap_err();
        assert!(e.starts_with("c2c.link_latency: must be in"), "{e}");
    }
    // the bounds themselves are inside
    for nom in [209, 228] {
        assert_eq!(Geom { link_latency: nom, ..g.clone() }.check_link_latency(), Ok(()));
    }
}

#[test]
fn link_plan_refuses_an_empty_latency_range() {
    let g = Geom::new(&profile("mini"));
    for (min, nom, max) in [(209, 208, 228), (209, 229, 228)] {
        let bad = Geom { link_latency_min: min, link_latency: nom, link_latency_max: max, ..g.clone() };
        let e = LinkPlan::new(LinkMode::Nominal, &bad, 1).unwrap_err();
        assert_eq!(e, "need c2c.link_latency_min <= c2c.link_latency <= c2c.link_latency_max");
    }
    assert_eq!(LinkPlan::new(LinkMode::Fixed(208), &g, 2).unwrap_err(), "--link-latency must be in [209, 228]");
    assert_eq!(LinkPlan::new(LinkMode::Fixed(229), &g, 2).unwrap_err(), "--link-latency must be in [209, 228]");
    assert!(LinkPlan::new(LinkMode::Fixed(209), &g, 2).is_ok());
}

fn opts(chips: u32) -> Options {
    let mut o = Options::new("cfg".into());
    o.chips = chips;
    o
}

fn run_execute(p: &Profile, g: &Geom, prog: &Program, o: &Options, mode: LinkMode) -> (i32, String, String) {
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let code = execute(p, g, prog, o, mode, None, &mut out, &mut err);
    (code, text(&out), text(&err))
}

#[test]
fn several_chips_need_a_link_latency_of_at_least_one() {
    // `tsp_sim.c`: "--chips needs link latencies >= 1". The configuration schema keeps the latency
    // from being 0, so only a library caller (here a geometry with an empty lower bound) gets there.
    let p = profile("mini");
    let g = Geom { link_latency_min: 0, ..Geom::new(&p) };
    let prog = Program { chips: vec![ChipInput::default(); 2], ..Program::default() };
    let (code, out, err) = run_execute(&p, &g, &prog, &opts(2), LinkMode::Fixed(0));
    assert_eq!((code, out.as_str()), (2, ""));
    assert_eq!(err, "tsp-iss: --chips needs link latencies >= 1\n");
    // the same chips with a latency of 1 or more run
    let (code, out, err) = run_execute(&p, &g, &prog, &opts(2), LinkMode::Fixed(1));
    assert_eq!((code, err.as_str()), (0, ""));
    assert!(out.starts_with("C 0\nE - -1 0\n") && out.contains("\nC 1\nE - -1 0\n"), "{out}");
    // a single chip does not need links
    let one = Program { chips: vec![ChipInput::default()], ..Program::default() };
    assert_eq!(run_execute(&p, &g, &one, &opts(1), LinkMode::Fixed(0)).0, 0);
}

#[test]
fn execute_reports_a_chip_that_cannot_be_built() {
    let g = Geom::new(&profile("mini"));
    // a profile wider than the harness holds: `Machine::new` refuses it
    let wide = mutated("mini", |v| v["geometry"]["superlanes"] = 33.into());
    let prog = Program { chips: vec![ChipInput::default()], ..Program::default() };
    let (code, out, err) = run_execute(&wide, &g, &prog, &opts(1), LinkMode::Nominal);
    assert_eq!((code, out.as_str()), (2, ""));
    assert!(err.starts_with("tsp-iss: geometry.superlanes: ") && err.contains("at most 512"), "{err}");
    // an input the parser would have refused: a preload for a slice that does not exist
    let p = profile("mini");
    let mut chip = ChipInput::default();
    chip.preloads.push(Preload { slice: 90, addr: 0, bytes: vec![1] });
    let prog = Program { chips: vec![chip], ..Program::default() };
    let (code, _, err) = run_execute(&p, &g, &prog, &opts(1), LinkMode::Nominal);
    assert_eq!(code, 2);
    assert!(err.starts_with("tsp-iss: slice 0x5a out of range"), "{err}");
}

#[test]
fn load_config_reads_the_shipped_profiles_and_refuses_the_rest() {
    assert!(load_config(&config("ref")).is_ok());
    let e = load_config(&spec().join("config").join("no-such.json")).unwrap_err();
    assert_eq!(e, "cannot open: No such file or directory");
    let e = load_config(&spec().join("config")).unwrap_err();
    assert!(e == "read error" || e.starts_with("cannot open"), "{e}");
}

#[test]
fn load_errors_have_the_harness_texts() {
    let m = |e: LoadError| e.to_string();
    assert_eq!(m(LoadError::Queue(0x90)), "queue 0x90 out of range (0..8f or ffff)");
    assert_eq!(m(LoadError::SliceRange(0x58)), "slice 0x58 out of range (0..57 hex)");
    assert_eq!(
        m(LoadError::SliceAbsent(0x20)),
        "slice 0x20 does not exist (slice mod 44 >= geometry.mem_slices_per_hemisphere)"
    );
    assert_eq!(m(LoadError::Address(0x800)), "address 0x800 out of range (>= words_per_slice)");
    assert_eq!(
        m(LoadError::TooManyBytes { bytes: 321, lanes: 320 }),
        "321 bytes: an S line has more bytes than the 320 lanes of this profile"
    );
    assert_eq!(
        m(LoadError::ImageTooLong { queue: 3, units: 321, limit: 320 }),
        "initial IQ image of queue 0x3 has 321 units, more than icu.iq_depth allows (320)"
    );
    assert_eq!(RunError::Internal("model fault".into()).to_string(), "model fault");
    let e: Box<dyn std::error::Error> = Box::new(LoadError::Queue(1));
    assert!(e.source().is_none());
    let e: Box<dyn std::error::Error> = Box::new(RunError::Internal("x".into()));
    assert!(e.source().is_none());
}

#[test]
fn machine_accessors_and_load_checks() {
    let mut m = machine("mini");
    assert_eq!(m.chip_id(), 3);
    assert_eq!(m.geom().vlanes, 16);
    assert_eq!(m.links().peer(0), None, "no links are wired");
    // queues: 0..143 and the control channel have a state, nothing else does
    assert_eq!(m.queue_state(5).map(|s| s.iq_units), Some(0));
    assert_eq!(m.queue_state(0xffff).map(|s| s.iq_units), Some(0));
    assert_eq!(m.queue_state(144), None);
    assert_eq!(m.load_image(144, &[1]), Err(LoadError::Queue(144)));
    m.load_image(5, &[0xffff_ffff, 7, 8]).unwrap();
    assert_eq!(m.queue_state(5).map(|s| s.iq_units), Some(2), "padding is dropped");
    let limit = m.geom().iq_units;
    let e = m.load_image(6, &vec![1; limit + 1]).unwrap_err();
    assert_eq!(e, LoadError::ImageTooLong { queue: 6, units: limit + 1, limit });
    // words
    assert_eq!(m.preload(88, 0, &[1]), Err(LoadError::SliceRange(88)));
    assert_eq!(m.preload(0, 1 << 20, &[1]), Err(LoadError::Address(1 << 20)));
    let lanes = m.geom().vlanes as usize;
    assert_eq!(m.preload(0, 0, &vec![0; lanes + 1]), Err(LoadError::TooManyBytes { bytes: lanes + 1, lanes }));
    m.preload(0, 3, &[9, 8, 7]).unwrap();
    assert_eq!(&m.sram_word(0, 3).unwrap()[..4], &[9, 8, 7, 0]);
    assert_eq!(m.sram_word(88, 0), Err(LoadError::SliceRange(88)));
    // nothing was written to a stream register
    assert_eq!(m.srf_value(Row::East, 0, 0, 0), None);
    assert_eq!(m.srf_value(Row::West, 31, 44, 100), None);
    // the run: one cycle at the least, then quiescent
    assert!(!m.quiescent() || m.next_event(0).is_none());
    let sum = m.run(1000, &mut NoObserver).unwrap();
    assert!(m.quiescent());
    assert_eq!(m.next_tick(), None);
    assert_eq!(m.next_time(), None);
    assert!(sum.ticks >= 1);
    // a reset puts the image back and the machine can run again
    m.reset();
    assert_eq!(m.queue_state(5).map(|s| s.iq_units), Some(2));
    assert_eq!(m.next_tick(), Some(0));
    assert_eq!(m.run(1000, &mut NoObserver).unwrap(), sum);
}

#[test]
fn absent_slices_are_refused_by_the_machine() {
    // With 8 MEM slices per hemisphere, slices 8..43 do not exist
    let p = mutated("mini", |v| v["geometry"]["mem_slices_per_hemisphere"] = 8.into());
    let mut m = Machine::new(&p, 0, Box::new(NoLinks)).unwrap();
    assert!(
        m.geom().slice_present(7)
            && !m.geom().slice_present(8)
            && m.geom().slice_present(44)
            && !m.geom().slice_present(88)
    );
    assert_eq!(m.preload(8, 0, &[1]), Err(LoadError::SliceAbsent(8)));
    assert_eq!(m.sram_word(52, 0), Err(LoadError::SliceAbsent(52)));
    assert!(m.preload(7, 0, &[1]).is_ok());
}

#[test]
fn no_links_is_a_chip_without_peers() {
    let mut l = NoLinks;
    assert_eq!(l.peer(0), None);
    l.send(0, 0, &[1; MAX_LANES]);
    assert!(!l.recv_ready(0, 0));
    assert_eq!(l.recv(0, 0), [0; MAX_LANES]);
}

#[derive(Default)]
struct Bare;
impl Observer for Bare {}

#[test]
fn observer_defaults_ignore_everything() {
    for o in [&mut Bare as &mut dyn Observer, &mut NoObserver] {
        o.srf(1, Row::East, 2, 3, &[]);
        o.sram(1, 2, 3, &[]);
        o.diag(1, 2, Diag::Dec003);
        o.lint(1, 2, "x");
        o.dispatch(1, 2);
        o.wait(1, 2, Hold::Parked, 0);
        o.fetch(1, 2, 640);
        assert!(!o.tracks_waits());
    }
    assert_eq!((Row::East.letter(), Row::West.letter()), ('E', 'W'));
    for h in [Hold::Parked, Hold::Busy, Hold::Path, Hold::Loop, Hold::Repeat, Hold::Configuration, Hold::IncompleteText]
    {
        assert_eq!(h, h);
    }
}

#[test]
fn layout_rows_are_opposite() {
    use tsp_iss::layout::RowId;
    assert_eq!(inward_row(true), RowId::W);
    assert_eq!(inward_row(false), RowId::E);
    assert_eq!(outward_row(true), RowId::E);
    assert_eq!(outward_row(false), RowId::W);
}
