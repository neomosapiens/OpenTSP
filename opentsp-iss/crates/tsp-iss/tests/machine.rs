//! Tests of `machine`, moved from the module's in-source test block (they use the crate's public API).
use ots_profile::Profile;
use tsp_iss::links::NoLinks;
use tsp_iss::machine::*;
use tsp_iss::observe::NoObserver;
use tsp_iss::observe::Row;

fn machine(name: &str) -> Machine {
    Machine::new(&Profile::load(name).unwrap(), 0, Box::new(NoLinks)).unwrap()
}

#[test]
fn images_drop_padding_append_and_are_bounded() {
    let mut m = machine("ref");
    m.load_image(3, &[1, 0xffff_ffff, 2]).unwrap();
    m.load_image(3, &[3]).unwrap();
    assert_eq!(m.queue_state(3), Some(QueueState { iq_units: 3, iq_max: 3, iq_min: 3 }));
    // 320 units fit exactly, padding does not count
    m.load_image(0xffff, &vec![7; 320]).unwrap();
    m.load_image(0xffff, &[0xffff_ffff]).unwrap();
    assert_eq!(m.queue_state(0xffff).unwrap().iq_units, 320);
    assert!(matches!(m.load_image(0xffff, &[1]), Err(LoadError::ImageTooLong { limit: 320, .. })));
    assert_eq!(m.load_image(144, &[1]), Err(LoadError::Queue(144)));
    assert_eq!(m.queue_state(144), None);
    assert_eq!(m.queue_state(143), Some(QueueState::default()));
}

#[test]
fn preload_checks_and_rejects_long_lines() {
    let mut m = machine("mini");
    // : a line of exactly the lane count is whole, a longer one is an error and writes nothing
    m.preload(87, 255, &[5; 16]).unwrap();
    let w = m.sram_word(87, 255).unwrap();
    assert!(w[..16].iter().all(|b| *b == 5) && w[16..].iter().all(|b| *b == 0));
    assert_eq!(m.preload(86, 255, &[5; 320]), Err(LoadError::TooManyBytes { bytes: 320, lanes: 16 }));
    assert!(m.sram_word(86, 255).unwrap().iter().all(|b| *b == 0));
    assert_eq!(m.preload(88, 0, &[]), Err(LoadError::SliceRange(88)));
    assert_eq!(m.preload(0, 256, &[]), Err(LoadError::Address(256)));
    assert_eq!(m.preload(0, 0, &[0; 17]), Err(LoadError::TooManyBytes { bytes: 17, lanes: 16 }));
    assert_eq!(m.sram_word(0, 256), Err(LoadError::Address(256)));
}

#[test]
fn an_empty_machine_ticks_cycle_zero_only_and_reports_nothing() {
    let mut m = machine("ref");
    assert_eq!(m.next_tick(), Some(0));
    let s = m.run(0, &mut NoObserver).unwrap();
    assert_eq!(s, RunSummary { ticks: 1, last_cycle: Some(0) });
    assert_eq!(m.next_tick(), None);
    assert_eq!(m.srf_value(Row::East, 0, 0, 0), None);
    // a second run has nothing to do
    assert_eq!(m.run(0, &mut NoObserver).unwrap().ticks, 0);
}

#[test]
fn a_window_stops_the_run_and_the_limit_stops_it_with_an_error() {
    let mut m = machine("ref");
    assert_eq!(m.run_until(0, 0, &mut NoObserver).unwrap().ticks, 0);
    assert_eq!(m.next_tick(), Some(0));
    assert_eq!(m.run_until(1, 0, &mut NoObserver).unwrap().ticks, 1);
    // a machine whose next tick is beyond the limit reports exit-3 information
    let mut m = machine("ref");
    // a NOP prefix of 5 cycles: the head parcel cannot dispatch before cycle 5, beyond the limit
    let units: Vec<u32> = [ots_isa::Insn::NOP { n: 5 }, ots_isa::Insn::SetVL { n: 2 }]
        .iter()
        .map(|i| ots_isa::encode(i) as u32)
        .collect();
    m.load_image(100, &units).unwrap();
    assert_eq!(m.run(4, &mut NoObserver), Err(RunError::MaxCycles { cycle: 5, limit: 4 }));
}

#[test]
fn wide_profiles_are_refused() {
    let mut c: serde_json::Value =
        serde_json::from_slice(include_bytes!("../../../../opentsp-arch-spec/config/ref.json")).unwrap();
    c["geometry"]["superlanes"] = 33.into();
    let p = Profile::from_bytes("wide", &serde_json::to_vec(&c).unwrap()).unwrap();
    let e = Machine::new(&p, 0, Box::new(NoLinks)).err().unwrap();
    assert!(e.contains("528 lanes"), "{e}");
}
