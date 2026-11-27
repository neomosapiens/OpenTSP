//! Ports of the model's `bank_rule` unit tests (`unit_tests/test_pure_helpers.sail`): the ISS's
//! `slices/mem.rs` `bank_rule` returns the same composed verdict (`vbank`, `vlblock`/`vlsrc`,
//! `vfm_clash`/`vfm_loop`) as Sail for the same (rt, go, loop, fm) inputs.
mod common;

use tsp_iss::slices::mem::{bank_rule, BankUse, RefTag};

const T0: RefTag = RefTag { thread: 0, bank: BankUse::Of(0) };
const T1: RefTag = RefTag { thread: 0, bank: BankUse::Of(1) };
const TA: RefTag = RefTag { thread: 0, bank: BankUse::Any };
const U0: RefTag = RefTag { thread: 1, bank: BankUse::Of(0) };

// model: unit_tests/test_pure_helpers.sail test_bank_rule_new_references
#[test]
fn test_bank_rule_new_references() {
    let mut go = [false; 4];
    go[0] = true;
    go[2] = true;
    let mut rt = [None; 4];
    rt[0] = Some(T0);
    rt[2] = Some(T0);
    let u1 = bank_rule(rt, go, [None; 2], false, BankUse::None);
    assert!(u1.vbank[0] && u1.vbank[2] && !u1.vbank[1] && !u1.vbank[3]);
    rt[2] = Some(T1);
    let w = bank_rule(rt, go, [None; 2], false, BankUse::None);
    assert!(!w.vbank[0] && !w.vbank[2]);
    rt[2] = Some(U0);
    assert!(!bank_rule(rt, go, [None; 2], false, BankUse::None).vbank[0]);
    rt[2] = Some(TA);
    assert!(bank_rule(rt, go, [None; 2], false, BankUse::None).vbank[0]);
    // a reference that failed its checks neither loses nor makes the other lose
    let mut go2 = [false; 4];
    go2[0] = true;
    rt[2] = Some(T0);
    let x = bank_rule(rt, go2, [None; 2], false, BankUse::None);
    assert!(!x.vbank[0] && !x.vbank[2]);
}

// model: unit_tests/test_pure_helpers.sail test_bank_rule_loops
#[test]
fn test_bank_rule_loops() {
    let mut go = [false; 4];
    go[2] = true;
    let mut rt = [None; 4];
    rt[2] = Some(T0);
    let mut lp = [None; 2];
    lp[0] = Some(T0);
    let u1 = bank_rule(rt, go, lp, false, BankUse::None);
    assert!(u1.vbank[2] && u1.vlblock[0] && u1.vlsrc[0] == 2 && !u1.vlblock[1]);
    let mut lp2 = [None; 2];
    lp2[0] = Some(T0);
    lp2[1] = Some(T0);
    let w = bank_rule([None; 4], [false; 4], lp2, false, BankUse::None);
    assert!(w.vlblock[0] && w.vlblock[1] && w.vlsrc[0] == 4 && w.vlsrc[1] == 4);
    let mut lp3 = [None; 2];
    lp3[0] = Some(T0);
    lp3[1] = Some(T1);
    let x = bank_rule([None; 4], [false; 4], lp3, false, BankUse::None);
    assert!(!x.vlblock[0] && !x.vlblock[1] && x.vlsrc[0] == -1);
}

// model: unit_tests/test_pure_helpers.sail test_bank_rule_ifetchm
#[test]
fn test_bank_rule_ifetchm() {
    let mut go = [false; 4];
    go[2] = true;
    let mut rt = [None; 4];
    rt[2] = Some(T0);
    let u1 = bank_rule(rt, go, [None; 2], true, BankUse::Of(0));
    assert!(u1.vfm_clash && u1.vbank[2] && !u1.vfm_loop);
    let w = bank_rule(rt, go, [None; 2], true, BankUse::Of(1));
    assert!(!w.vfm_clash && !w.vbank[2]);
    let x = bank_rule(rt, go, [None; 2], false, BankUse::Of(0));
    assert!(!x.vfm_clash && !x.vbank[2]);
    let mut lp = [None; 2];
    lp[1] = Some(T0);
    let y = bank_rule([None; 4], [false; 4], lp, true, BankUse::Of(0));
    assert!(y.vfm_clash && y.vfm_loop && y.vlblock[1] && !y.vlblock[0]);
    let mut lp0 = [None; 2];
    lp0[0] = Some(T0);
    let z = bank_rule([None; 4], [false; 4], lp0, true, BankUse::Of(0));
    assert!(!z.vfm_clash && !z.vlblock[0]);
}
