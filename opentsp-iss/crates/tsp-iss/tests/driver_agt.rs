//! Tests of `driver/agt`, moved from the module's in-source test block (they use the crate's public API).
use ots_isa::Insn;
use tsp_iss::driver::agt::*;
use tsp_iss::driver::Driver;

fn configuring() -> Agt {
    Agt { state: AgtState::Config, ..Agt::idle() }
}

// test_agt_advance_two_levels
#[test]
fn test_agt_advance_two_levels() {
    let mut a = configuring();
    assert!(agt_configure(&mut a, 0, 1));
    assert!(agt_configure(&mut a, 1, 2));
    assert!(agt_configure(&mut a, 3, 3)); // inner: 3 references
    assert!(agt_configure(&mut a, 0, 4)); // starts level 1
    assert!(agt_configure(&mut a, 1, 10));
    assert!(agt_configure(&mut a, 3, 2)); // outer: 2 references
                                          // references: offsets 0, 2, 4 | 14, 16, 18; delays 2, 2, then 1 + 1 + 4 = 6 at the level boundary, then 2, 2
    let offs = [2, 4, 14, 16, 18];
    let dels = [2, 2, 6, 2, 2];
    let mut ls = a.levels;
    let mut off = 0;
    for k in 0..5 {
        let (l2, d, st, _) = agt_advance(&ls, 2).expect("loop ended early");
        off += st;
        assert_eq!((off, d), (offs[k], dels[k]), "reference {}", k + 1);
        ls = l2;
    }
    assert_eq!(agt_advance(&ls, 2), None);
}

// test_agt_defaults
#[test]
fn test_agt_defaults() {
    let mut a = configuring();
    assert!(agt_configure(&mut a, 1, 2));
    let l0 = a.levels[0];
    assert!(l0.countdown == 0 && l0.iters == 2 && l0.step == 2);
    let (ls, d, st, _) = agt_advance(&a.levels, 1).expect("default numIter is 2");
    assert!(d == 1 && st == 2);
    assert_eq!(agt_advance(&ls, 1), None);
    assert_eq!(agt_advance(&a.levels, 0), None, "no level received");
    let mut a = configuring();
    assert!(agt_configure(&mut a, 3, 0));
    assert_eq!(a.levels[0].iters, 1);
    assert!(agt_configure(&mut a, 0, 0)); // level 1; explicit 0 counts as 1
    assert_eq!(a.levels[1].countdown, 1);
}

// test_agt_four_levels_max
#[test]
fn test_agt_four_levels_max() {
    let mut a = configuring();
    for _ in 0..4 {
        assert!(agt_configure(&mut a, 0, 1));
    }
    assert!(!agt_configure(&mut a, 0, 1));
}

// test_agt_streamstep
#[test]
fn test_agt_streamstep() {
    let mut a = configuring();
    assert!(agt_configure(&mut a, 1, 1));
    assert!(agt_configure(&mut a, 2, 2));
    assert!(agt_configure(&mut a, 3, 2));
    assert_eq!(a.level, 0);
    let (_, _, st, ss) = agt_advance(&a.levels, 1).unwrap();
    assert_eq!((st, ss), (1, 2));
    // Step is signed
    let mut b = configuring();
    agt_configure(&mut b, 1, 0xFFFF);
    assert_eq!(b.levels[0].step, -1);
}

// model: unit_tests/test_dispatch.sail test_dispatch_config_fields
#[test]
fn config_fields_of_the_catalogue() {
    assert_eq!(agt_config(&Insn::Step { a: 0x18 }), Some((1, 0x18)));
    assert_eq!(agt_config(&Insn::Iterations { n: 0x6F }), Some((3, 0x6F)));
    assert_eq!(agt_config(&Insn::Countdown { d: 1 }), Some((0, 1)));
    assert_eq!(agt_config(&Insn::StreamStep { s: 2 }), Some((2, 2)));
    assert_eq!(agt_config(&Insn::Read { addr: 0, dst: 0, thread: 0 }), None);
    assert_eq!(agt_config(&Insn::Sync), None);
}

// test_dispatch_agt_tail: a finished AGT loop holds its path through its trailing waits
#[test]
fn test_dispatch_agt_tail() {
    let mut d = Driver::new(35);
    let mut l0 = AGT_DEFAULT_LEVEL;
    l0.countdown = 3;
    l0.idx = 1;
    let mut lv = [AGT_DEFAULT_LEVEL; 4];
    lv[0] = l0;
    let a = Agt { state: AgtState::Run, levels: lv, level: 0, pos: 0, last: 40, ..Agt::idle() };
    d.agt_set(5, 1, a);
    d.agt_refresh(5, 1);
    assert_eq!(d.agt_get(5, 1).state, AgtState::Idle);
    assert!(d.path_tail(5, 1, 41) && d.path_tail(5, 1, 43));
    assert!(!d.path_tail(5, 1, 44));
    assert!(!d.path_tail(5, 0, 41));
}

#[test]
fn cfg_path_alternates_and_agt_is_per_path() {
    let mut d = Driver::new(35);
    assert_eq!(d.cfg_path(5), 0);
    d.agts_w[5].state = AgtState::Config;
    assert_eq!(d.cfg_path(5), 1);
    d.agts[5].state = AgtState::Config;
    assert_eq!(d.cfg_path(5), 0);
    d.agt_records[2 * 5] = 1;
    assert_eq!(d.cfg_path(5), 1);
    assert_eq!(d.agt_get(5, 0).state, AgtState::Config);
    // the end of a configuration with the defaults: two references, then idle with a tail
    d.agts = [Agt::idle(); 144];
    d.agts_w = [Agt::idle(); 144];
    d.agts[3] = Agt { state: AgtState::Config, pos: 1, instr: Some(Insn::Sync), last: 20, ..Agt::idle() };
    d.agt_finish_config(3, 0);
    assert_eq!(d.agt_get(3, 0).state, AgtState::Run);
    assert_eq!(d.agt_get(3, 0).next, 21, "default level: 2 references, 1 cycle apart (countdown 0)");
}

// agt_next_wait is agt_advance's delay without the copy of the levels: the same answer for every configuration
#[test]
fn agt_next_wait_is_the_delay_of_agt_advance() {
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = move |m: u64| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x % m
    };
    for _ in 0..2000 {
        let mut ls = [AGT_DEFAULT_LEVEL; 4];
        for l in &mut ls {
            l.countdown = next(5);
            l.iters = 1 + next(3);
            l.idx = next(l.iters);
        }
        for n in 0..=4 {
            let want = agt_advance(&ls, n).map(|(_, d, _, _)| d);
            assert_eq!(agt_next_wait(&ls, n), want, "levels {ls:?}, n {n}");
        }
    }
}

// the in-place accessor marks the queue changed, as agt_set does
#[test]
fn agt_mut_edits_in_place_and_agt_refresh_matches_the_copying_form() {
    let mut d = Driver::new(1);
    let mut a = Agt { state: AgtState::Run, instr: None, t0: 5, last: 5, ..Agt::idle() };
    assert!(agt_configure(&mut a, 0, 2));
    assert!(agt_configure(&mut a, 3, 4));
    a.state = AgtState::Run;
    d.agt_set(3, 1, a);
    d.agt_refresh(3, 1);
    let want_next = a.last + agt_advance(&a.levels, agt_nlevels(&a)).unwrap().1;
    assert_eq!(d.agt_ref(3, 1).next, want_next.into());
    d.agt_mut(3, 1).last = 99;
    assert_eq!(d.agt_get(3, 1).last, 99);
    assert_eq!(d.agt_get(3, 0), Agt::idle(), "the other path is untouched");
}

// agt_due_lean and agt_apply_advance are agt_due and agt_advance without the copies of the levels: the same due
// reference with the same delay, step and stream step, and the same counters afterwards, for every configuration and cycle
#[test]
fn the_lean_due_check_and_the_in_place_advance_equal_the_copying_forms() {
    let mut x: u64 = 0x1234_5678_9abc_def1;
    let mut next = move |m: u64| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x % m
    };
    for _ in 0..3000 {
        let mut a = Agt { state: [AgtState::Idle, AgtState::Config, AgtState::Run][next(3) as usize], ..Agt::idle() };
        a.last = next(50);
        a.pos = next(5) as i64 - 1;
        a.level = next(4) as usize;
        for l in &mut a.levels {
            l.countdown = next(4);
            l.step = next(9) as i64 - 4;
            l.sstep = next(5) as i64 - 2;
            l.iters = 1 + next(3);
            l.idx = next(l.iters);
        }
        for (c, parked) in [(a.last + next(8), false), (a.last + next(8), true)] {
            let old = agt_due(&a, c, parked);
            let lean = agt_due_lean(&a, c, parked);
            assert_eq!(lean, old.map(|(_, d, st, ss)| (d, st, ss)), "{a:?} cycle {c} parked {parked}");
            if let Some((ls, ..)) = old {
                let mut m = a.levels;
                agt_apply_advance(&mut m, agt_nlevels(&a));
                assert_eq!(m, ls, "{a:?}");
            }
        }
    }
}
