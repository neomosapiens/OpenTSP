//! The patrol scrub's schedule on the machine itself: how many scrub cycles a run to cycle `upto` makes, and which cycles the
//! patrol slot of a MEM queue is given up to software traffic. The ports of the model's pure helpers of the same decisions
//! (`unit_tests/test_pure_helpers.sail`); the ISS decides them inline in `scrub.rs`, so each row is observed as the advance of the
//! patrol pointer of slice 0 (one step per scrub cycle that the slice's patrol slot is free for).
mod common;
use ots_isa::Insn;
use ots_profile::Profile;
use tsp_iss::ecc::flip_delta;
use tsp_iss::observe::NoObserver;
use tsp_iss::{Machine, NoLinks};

/// `ref` with `scrub.period` and the d_skew shared by Read, Write, Gather, Scatter and IfetchM changed.
fn machine(period: u64, skew: u64) -> Machine {
    machine_of("ref", period, skew)
}

fn machine_of(profile: &str, period: u64, skew: u64) -> Machine {
    let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(common::config(profile)).unwrap()).unwrap();
    v["scrub"]["period"] = period.into();
    v["timing"]["mem"]["d_skew"] = skew.into();
    v["timing"]["cu"]["d_skew"] = skew.into(); // IfetchM shares the SRAM ports' d_skew
    let p = Profile::from_bytes(profile.rsplit('/').next().unwrap(), &serde_json::to_vec(&v).unwrap()).unwrap();
    Machine::new(&p, 0, Box::new(NoLinks)).unwrap()
}

// model: unit_tests/test_pure_helpers.sail test_scrub_due_count
#[test]
fn scrub_cycles_due_up_to_a_cycle() {
    // (next, period, upto, cycles run); the model's row (0, 4, -1, 0) has no cycle -1 to run to
    for (next, period, upto, want) in
        [(0u64, 4u64, 0u64, 1u32), (0, 4, 3, 1), (0, 4, 4, 2), (8, 4, 7, 0), (8, 4, 8, 1), (8, 4, 20, 4), (8, 0, 20, 0)]
    {
        let mut m = machine(period, 0);
        m.edit_state(|state| state.driver.scrub_next = next as u128).unwrap();
        m.tick(upto, &mut NoObserver);
        assert_eq!(m.driver().scrub_ptr[0], want, "next {next}, period {period}, upto {upto}");
    }
}

// model: unit_tests/test_pure_helpers.sail test_scrub_slot_taken
#[test]
fn patrol_slot_is_taken_by_a_held_path_or_the_last_issue() {
    let read = Insn::Read { addr: 0, dst: 0, thread: 0 };
    // (m, busy0, busy1, last issue cycle, taken); the d_skew of Read is 3
    let rows: [(u64, u64, u64, Option<u64>, bool); 7] = [
        (2, 100, 100, Some(0), false), // before the first dispatch cycle
        (13, 11, 0, None, true),       // d = 10 < 11: path 0 held through it
        (14, 11, 0, None, false),
        (13, 0, 11, None, true),
        (13, 0, 0, Some(10), true),
        (13, 0, 0, Some(9), false),
        (13, 0, 0, None, false),
    ];
    for (cycle, busy0, busy1, last, taken) in rows {
        let mut m = machine(4, 3);
        m.edit_state(|state| state.driver.scrub_next = cycle as u128).unwrap();
        m.edit_state(|state| state.driver.path_busy_until[0] = busy0.into()).unwrap();
        m.edit_state(|state| state.driver.path_busy_until[1] = busy1.into()).unwrap();
        if let Some(at) = last {
            m.edit_state(|state| state.driver.queue_last[0] = Some(read)).unwrap();
            m.edit_state(|state| state.driver.queue_last_at[0] = at).unwrap();
        }
        m.tick(cycle, &mut NoObserver);
        assert_eq!(
            m.driver().scrub_ptr[0],
            if taken { 0 } else { 1 },
            "m {cycle}, busy {busy0} {busy1}, last {last:?}"
        );
    }
}

/// A correctable error (one flipped check bit of superlane 0) in the stored word at slice 0, address 0, where the patrol pointer starts.
fn corrupt_word_under_the_pointer(m: &mut Machine) {
    m.edit_state(|state| state.ecc.live = true).unwrap();
    let id = m.ecc().sram_ov(0, 0);
    let e = m.ecc().get(id, 0) ^ flip_delta(0);
    let nid = m.edit_state(|state| state.ecc.set(id, 0, e)).unwrap();
    m.edit_state(|state| state.ecc.set_sram_ov(0, 0, nid)).unwrap();
}

/// A machine whose patrol has read the damaged word at cycle 0 and holds the repair, not yet written.
fn with_a_saved_repair() -> Machine {
    let mut m = machine(4, 0);
    corrupt_word_under_the_pointer(&mut m);
    m.edit_state(|state| state.driver.scrub_next = 0).unwrap();
    m.tick(0, &mut NoObserver);
    m
}

// model: unit_tests/test_pure_helpers.sail test_scrub_arbitrate
#[test]
fn what_a_patrol_slot_does_for_a_slice() {
    use tsp_iss::ecc::OV0;
    // SCRUB_PATROL: present, free, nothing saved. The patrol reads the word, saves the repair, and moves on.
    let m = with_a_saved_repair();
    assert_eq!(m.driver().scrub_ptr[0], 1);
    assert!(m.driver().scrub_pending[0].is_some(), "the repair is saved, not yet written");
    assert_ne!(m.ecc().sram_ov(0, 0), OV0, "the stored overlay is still the damaged one");

    // SCRUB_SKIP: a saved repair stays while software holds the slot (the path is busy through the dispatch cycle)
    let mut held = with_a_saved_repair();
    held.edit_state(|edit| edit.driver.path_busy_until[0] = 100).unwrap();
    held.tick(4, &mut NoObserver);
    assert!(held.driver().scrub_pending[0].is_some());
    assert_ne!(held.ecc().sram_ov(0, 0), OV0);
    assert_eq!(held.driver().scrub_ptr[0], 1);

    // SCRUB_DROP: the slice was written since the repair was saved, so the saved repair is stale and is thrown away
    let mut stale = with_a_saved_repair();
    stale.edit_state(|state| state.ecc.set_sram_ov(0, 9, OV0)).unwrap();
    stale.tick(4, &mut NoObserver);
    assert!(stale.driver().scrub_pending[0].is_none());
    assert_ne!(stale.ecc().sram_ov(0, 0), OV0, "nothing was written back");
    assert_eq!(stale.driver().scrub_ptr[0], 1, "the drop takes the slot: no patrol read in it");

    // SCRUB_WRITEBACK: fresh and free: the repair is written, the overlay cleared
    let mut fresh = with_a_saved_repair();
    fresh.tick(4, &mut NoObserver);
    assert!(fresh.driver().scrub_pending[0].is_none());
    assert_eq!(fresh.ecc().sram_ov(0, 0), OV0);
    assert_eq!(fresh.driver().scrub_ptr[0], 1, "the writeback takes the slot: no patrol read in it");

    // SCRUB_SKIP: a slice that is not there (mid_min has four MEM slices per hemisphere) is never scrubbed
    let mut small = machine_of("test/mid_min", 4, 0);
    small.edit_state(|edit| edit.driver.scrub_next = 0).unwrap();
    small.tick(0, &mut NoObserver);
    assert_eq!((small.driver().scrub_ptr[0], small.driver().scrub_ptr[4]), (1, 0));
}
