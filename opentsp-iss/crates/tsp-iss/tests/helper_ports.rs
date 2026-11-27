//! Ports of the model's pure helpers of port reservation, Repeat planning and tick predicates
//! (`unit_tests/test_pure_helpers.sail`): the ISS keeps the same arithmetic in `hold_path`,
//! `reserve_local_fetch`, Repeat dispatch and `wait_site` / `polls_at` / configuration end.
mod common;

use ots_isa::Insn;
use tsp_iss::driver::agt::AgtState;
use tsp_iss::driver::repeat::{repeat_after, repeat_plan, RepeatState};
use tsp_iss::driver::{config_ends, fetchm_bank_note, fetchm_hold_until, path_hold_until, poll_epoch, wait_reason};

// model: unit_tests/test_pure_helpers.sail test_port_reservation
#[test]
fn test_port_reservation() {
    assert_eq!(path_hold_until(10), 11);
    assert_eq!(fetchm_hold_until(10), 12);
    let i = Insn::IfetchM { addr: 5 };
    assert_eq!(fetchm_bank_note(&i, 10, 100, 8192), Some((11, 6)));
    // the word after the last of the segment (or of the slice) is not read
    assert_eq!(fetchm_bank_note(&i, 10, 6, 8192), None);
    assert_eq!(fetchm_bank_note(&i, 10, 0, 8192), None);
    assert_eq!(fetchm_bank_note(&Insn::Ifetch { stream: 0 }, 10, 100, 8192), None);
}

// model: unit_tests/test_pure_helpers.sail test_repeat_plan
#[test]
fn test_repeat_plan() {
    let i = Insn::NOP { n: 1 };
    // issued at 10, d = 2: iterations at 13, 16, ...; a Repeat dispatched at 13 issues now
    let (now, rest) = repeat_plan(i, 3, 2, 10, 13);
    assert!(now);
    let r = rest.expect("rest");
    assert_eq!((r.left, r.gap, r.next), (2, 3, 16));
    // dispatched at 11: the first iteration is not before the Repeat's own cycle, so at 13
    let (now2, rest2) = repeat_plan(i, 3, 2, 10, 11);
    assert!(!now2);
    let r2 = rest2.expect("rest2");
    assert_eq!((r2.left, r2.next), (3, 13));
    let (now3, rest3) = repeat_plan(i, 1, 0, 10, 20);
    assert!(now3 && rest3.is_none());
    let r = RepeatState { instr: i, left: 3, gap: 3, next: 13 };
    let after = repeat_after(r, 13).expect("after");
    assert_eq!((after.left, after.next), (2, 16));
    assert!(repeat_after(RepeatState { instr: i, left: 1, gap: 3, next: 13 }, 13).is_none());
}

// model: unit_tests/test_pure_helpers.sail test_tick_helpers
#[test]
fn test_tick_helpers() {
    assert_eq!(wait_reason(true, true, true, true), "wait-host");
    assert_eq!(wait_reason(false, true, true, true), "wait-barrier");
    assert_eq!(wait_reason(false, false, true, true), "wait-poll");
    assert_eq!(wait_reason(false, false, false, true), "wait-repeat");
    assert_eq!(wait_reason(false, false, false, false), "wait-resource");
    assert!(config_ends(AgtState::Config, false, true));
    assert!(config_ends(AgtState::Config, true, false));
    assert!(!config_ends(AgtState::Config, true, true));
    assert!(!config_ends(AgtState::Run, false, false));
    assert!(!config_ends(AgtState::Idle, false, false));
    assert!(poll_epoch(10, 20, 10, false));
    assert!(!poll_epoch(20, 20, 10, false));
    assert!(!poll_epoch(10, 21, 10, false));
    assert!(!poll_epoch(10, 20, 10, true));
}
