//! Tests of `driver/classify`, moved from the module's in-source test block (they use the crate's public API).
use ots_isa::Insn;
use tsp_iss::driver::classify::*;
use tsp_iss::Diag;

const READ: Insn = Insn::Read { addr: 1, dst: 0, thread: 0 };
const WRITE: Insn = Insn::Write { addr: 1, src: 0, thread: 0 };

#[test]
fn a_lone_instruction_is_single_or_slotted() {
    // non-MEM queue: always single
    let r = classify_items(100, &[Ok(Insn::SetVL { n: 1 })]);
    assert!(r.ok && r.n == 1 && r.single == Some(Insn::SetVL { n: 1 }));
    // MEM queue: a CU instruction needs the whole queue, a reference takes its slot
    let r = classify_items(3, &[Ok(Insn::SetVL { n: 1 })]);
    assert!(r.ok && r.single.is_some());
    let r = classify_items(3, &[Ok(READ)]);
    assert!(r.ok && r.single.is_none() && r.refs[0] == Some(READ));
    // an instruction that does not decode
    let r = classify_items(3, &[Err(Diag::Dec003)]);
    assert!(!r.ok && r.err == Some(Diag::Dec003));
}

// two records of one parcel
#[test]
fn parcels_fit_their_slots_or_break_the_slot_rule() {
    // a Read and a Write: the two paths
    let r = classify_items(3, &[Ok(READ), Ok(WRITE)]);
    assert!(r.ok && r.refs[0].is_some() && r.refs[2].is_some() && r.single.is_none());
    // two Reads of one thread share a slot
    assert!(!classify_items(3, &[Ok(READ), Ok(READ)]).ok);
    // ... but not of two threads
    let r1 = Insn::Read { addr: 1, dst: 0, thread: 1 };
    assert!(classify_items(3, &[Ok(READ), Ok(r1)]).ok);
    // a whole-queue instruction with another, an undecodable one among several, anything on a non-MEM queue
    assert!(!classify_items(3, &[Ok(READ), Ok(Insn::SetVL { n: 1 })]).ok);
    assert!(!classify_items(3, &[Ok(READ), Err(Diag::Dec003)]).ok);
    let r = classify_items(100, &[Ok(Insn::Sync), Ok(Insn::Sync)]);
    assert!(!r.ok && r.err.is_none() && r.n == 2);
    // IfetchM does not fit with an R-type reference, Ifetch does
    let ifm = Insn::IfetchM { addr: 4 };
    let ifs = Insn::Ifetch { stream: 4 };
    assert!(!classify_items(3, &[Ok(ifm), Ok(READ)]).ok);
    assert!(classify_items(3, &[Ok(ifm), Ok(WRITE)]).ok);
    assert!(classify_items(3, &[Ok(ifs), Ok(READ)]).ok);
    // one configuration record per cycle
    assert!(!classify_items(3, &[Ok(Insn::Step { a: 1 }), Ok(Insn::Countdown { d: 1 })]).ok);
    assert!(classify_items(3, &[Ok(Insn::Step { a: 1 }), Ok(READ)]).ok);
}

#[test]
fn ref_paths_and_threads() {
    assert_eq!(ref_path(&READ), Some(0));
    assert_eq!(ref_path(&Insn::Gather { dst: 0, map: 0, k: 0, thread: 0 }), Some(0));
    assert_eq!(ref_path(&WRITE), Some(1));
    assert_eq!(ref_path(&Insn::Scatter { src: 0, map: 0, k: 0, thread: 1 }), Some(1));
    assert_eq!(ref_path(&Insn::Countdown { d: 1 }), None);
    assert_eq!(ref_path(&Insn::Ifetch { stream: 0 }), None);
    assert_eq!(ref_thread(&Insn::Scatter { src: 0, map: 0, k: 0, thread: 1 }), 1);
    assert_eq!(ref_thread(&READ), 0);
    assert_eq!(ref_thread(&Insn::Sync), 0);
}
