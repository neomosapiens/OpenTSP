//! Ports of the Sail unit tests of the pure helpers (`unit_tests/test_pure_helpers.sail`) that the ISS has a counterpart
//! for: the stream register file (`Srf`), the hold decision (`Machine::hold_check`), the address generation table
//! (`driver::agt`) and the commit arbitration (`Machine::commit_due`). Each test states the facts of its Sail twin through
//! the ISS's own functions; the Sail slot functions take a gap and a diagonal `d = gap - cycle`, which these tests turn
//! into the cycle of a write or a read.
use ots_isa::{Code, Insn};
use ots_profile::Profile;
use tsp_iss::driver::agt::{agt_configure, agt_due, agt_due_lean, Agt, AgtLevel, AgtState};
use tsp_iss::driver::repeat::{PollState, RepeatState};
use tsp_iss::layout::RowId;
use tsp_iss::links::NoLinks;
use tsp_iss::observe::{Observer, Row};
use tsp_iss::operands::Commit;
use tsp_iss::srf::{Srf, SrfLookup, Vec320};
use tsp_iss::Machine;

const VL: usize = 320;

fn machine() -> Machine {
    Machine::new(&Profile::load("ref").unwrap(), 0, Box::new(NoLinks)).unwrap()
}

/// `ph_v`: every lane `b`.
fn v(b: u8) -> Vec320 {
    let mut x = [0u8; tsp_iss::geom::MAX_LANES];
    x[..VL].fill(b);
    x
}

// ---- the stream register file ------------------------------------------------------------------------------------

/// The cycle of a write at `gap` on the diagonal `d`.
fn at(gap: u8, d: i64) -> u64 {
    (i64::from(gap) - d) as u64
}

/// An empty file holding the writes `(gap, diagonal, byte)` on E:0.
fn slot(writes: &[(u8, i64, u8)]) -> Srf {
    let mut s = Srf::new(VL);
    for &(g, d, b) in writes {
        assert!(!s.write(RowId::E, 0, g, at(g, d), &v(b)), "write {g} {d}: no collision");
    }
    s
}

/// What a flow read at `gap` on diagonal `d` finds: its first lane, when it finds a value.
fn read(s: &Srf, gap: u8, d: i64) -> Option<u8> {
    match s.read(RowId::E, 0, gap, i64::from(gap) - d) {
        (SrfLookup::Found, x) => {
            assert!(x[..VL].iter().all(|b| *b == x[0]), "a uniform value");
            Some(x[0])
        }
        _ => None,
    }
}

/// What a chained read at `gap` of the write on diagonal `d` finds (the write at the reader's own gap, a cycle earlier).
fn read_chain(s: &Srf, gap: u8, d: i64) -> Option<u8> {
    match s.read_chain(RowId::E, 0, gap, i64::from(gap) - d + 1) {
        (SrfLookup::Found, x) => Some(x[0]),
        _ => None,
    }
}

// model: unit_tests/test_pure_helpers.sail test_slot_write_read
#[test]
fn slot_write_read() {
    // a write replaces its gap only: a second write at the same gap and diagonal collides, another gap or diagonal does not
    let mut s = slot(&[(3, -10, 0x01)]);
    assert!(s.write(RowId::E, 0, 3, at(3, -10), &v(0x01)), "the same gap and diagonal collide");
    assert!(!s.write(RowId::E, 0, 3, at(3, -11), &v(0x01)), "another diagonal");
    assert!(!s.write(RowId::E, 0, 4, at(4, -10), &v(0x01)), "another gap");
    // the reader takes the nearest upstream writer of its diagonal, strictly below its own gap
    let s2 = slot(&[(3, -10, 0x01), (7, -10, 0x02)]);
    assert_eq!(read(&s2, 5, -10), Some(0x01));
    assert_eq!(read(&s2, 9, -10), Some(0x02));
    assert_eq!(read(&s2, 3, -10), None, "a reader at the gap of a write does not see it");
    assert_eq!(read(&s2, 7, -10), Some(0x01));
    assert_eq!(read(&s2, 9, -9), None, "nor one on another diagonal");
    // the nearest writer below each reader (the Sail `srf_slot_best`)
    for (reader, best) in [(9, Some(0x02)), (8, Some(0x02)), (7, Some(0x01)), (4, Some(0x01)), (3, None)] {
        assert_eq!(read(&s2, reader, -10), best, "reader at gap {reader}");
    }
}

// model: unit_tests/test_pure_helpers.sail test_slot_chain
#[test]
fn slot_chain() {
    let s = slot(&[(12, -3, 0x33)]);
    assert_eq!(read_chain(&s, 12, -3), Some(0x33));
    assert_eq!(read_chain(&s, 13, -3), None);
    assert_eq!(read_chain(&s, 12, -4), None);
}

// model: unit_tests/test_pure_helpers.sail test_slot_flip
#[test]
fn slot_flip() {
    // a data bit flip changes the value the reader sees and nothing else
    let mut s = slot(&[(4, -2, 0x00)]);
    let c = 8u64;
    s.flip(c, (RowId::E, 0, 6), 0, 5, |old| old);
    let x = s.read(RowId::E, 0, 6, c as i64).1;
    assert_eq!(x[0], 1 << 5);
    assert!(x[1..VL].iter().all(|b| *b == 0), "only the flipped bit changed");
    // a reader at the writer's gap sees no value: nothing flips
    let mut s = slot(&[(4, -2, 0x00)]);
    s.flip(6, (RowId::E, 0, 4), 0, 5, |old| old);
    assert!(s.read(RowId::E, 0, 6, 8).1[..VL].iter().all(|b| *b == 0));
    // a check bit flips the overlay, not the data
    let mut s = slot(&[(4, -2, 0x00)]);
    assert_eq!(s.ov_at(RowId::E, 0, 6, 8, false), 0);
    s.flip(8, (RowId::E, 0, 6), 0, 130, |old| old ^ 1);
    assert_ne!(s.ov_at(RowId::E, 0, 6, 8, false), 0);
    assert!(s.read(RowId::E, 0, 6, 8).1[..VL].iter().all(|b| *b == 0));
}

// model: unit_tests/test_pure_helpers.sail test_srf_wrappers_agree_with_slots
#[test]
fn srf_wrappers_agree_with_slots() {
    let mut s = Srf::new(VL);
    assert!(!s.write(RowId::E, 5, 8, 15, &v(0x0a)));
    assert!(s.write(RowId::E, 5, 8, 15, &v(0x0b)), "the second write of one gap and cycle collides");
    assert!(!s.write(RowId::E, 5, 13, 20, &v(0x0c)));
    let found = |r: (SrfLookup, Vec320)| (r.0, r.1[0]);
    assert_eq!(found(s.read(RowId::E, 5, 11, 18)), (SrfLookup::Found, 0x0b));
    assert_eq!(found(s.read(RowId::E, 5, 16, 23)), (SrfLookup::Found, 0x0c));
    assert_ne!(s.read(RowId::E, 5, 8, 15).0, SrfLookup::Found);
    assert_eq!(found(s.read_chain(RowId::E, 5, 13, 21)), (SrfLookup::Found, 0x0c));
    assert_eq!(s.read_chain(RowId::E, 5, 13, 22).0, SrfLookup::None);
}

// model: unit_tests/test_pure_helpers.sail test_stream_only_downstream
#[test]
fn stream_only_downstream() {
    let mut s = Srf::new(VL);
    s.write(RowId::W, 9, 21, 25, &v(0x05));
    s.write(RowId::W, 9, 30, 25, &v(0x06));
    assert_eq!(s.read(RowId::W, 9, 18, 26).0, SrfLookup::Upstream, "every live write is downstream of the reader");
    assert_eq!(s.read(RowId::W, 9, 25, 40).0, SrfLookup::None, "another diagonal, a live write upstream of the reader");
}

// model: unit_tests/test_pure_helpers.sail test_rep_merge
#[test]
fn rep_merge() {
    // the report is written at gap 5 on diagonal 0; a reader above it sees the merged spare bits
    let spare = |s: &Srf, n: u8| s.spare(RowId::E, n, 6, 6) & 1;
    let streams = |hot: &[usize]| {
        let mut a = [0u32; 30];
        for &h in hot {
            a[h] = 1;
        }
        a
    };
    let mut s = Srf::new(VL);
    s.report(RowId::E, 5, 5, 2, 1, &streams(&[0, 5]));
    assert_eq!((spare(&s, 0), spare(&s, 5)), (1, 1));
    assert_eq!(spare(&s, 3), 1, "one error bit set: odd parity");
    // lower priority: S[1] merges, stream 6 does not
    s.report(RowId::E, 5, 5, 1, 1, &streams(&[1, 6]));
    assert_eq!((spare(&s, 1), spare(&s, 6), spare(&s, 5)), (1, 0, 1));
    assert_eq!(spare(&s, 3), 0, "two error bits: even parity");
    // higher priority: the auxiliary streams are replaced
    s.report(RowId::E, 5, 5, 3, 1, &streams(&[1, 6]));
    assert_eq!((spare(&s, 6), spare(&s, 5)), (1, 0));
}

// ---- the hold decision ---------------------------------------------------------------------------------------------

const Q: usize = 2;
const CYCLE: u64 = 20;

/// What the instruction is to the hold decision.
#[derive(Clone, Copy)]
enum Kind {
    Plain,
    Barrier,
    WholeFree,
    ConfigRecord,
}

fn insn(kind: Kind) -> Insn {
    match kind {
        Kind::Plain => Insn::NOP { n: 1 },
        Kind::Barrier => Insn::Sync,
        Kind::WholeFree => Insn::IfetchM { addr: 0 },
        Kind::ConfigRecord => Insn::Countdown { d: 1 },
    }
}

/// One dispatch path: busy, in a tail, and the state of its AGT (0 idle, 1 configuring, 2 running).
type PathView = (bool, bool, u8);

/// The state of a queue and its two dispatch paths, as the Sail `hold_view`.
#[derive(Clone, Copy, Default)]
struct View {
    parked: bool,
    host: bool,
    busy: bool,
    repeat: bool,
    poll: bool,
    paths: [PathView; 2],
}

fn set_view(m: &mut Machine, v: View) {
    m.edit_state(|state| state.driver.barrier.reset()).unwrap();
    if v.parked {
        m.edit_state(|state| state.driver.barrier.sync(Q, 5)).unwrap();
    }
    if v.host {
        m.edit_state(|state| state.driver.barrier.hw_park(Q, 5, 0)).unwrap();
    }
    m.edit_state(|state| state.driver.queue_busy_until[Q] = if v.busy { (CYCLE + 10).into() } else { 0 }).unwrap();
    m.edit_state(|state| {
        state.driver.queue_repeat[Q] =
            v.repeat.then_some(RepeatState { instr: Insn::NOP { n: 1 }, left: 1, gap: 1, next: u128::from(CYCLE) + 5 })
    })
    .unwrap();
    m.edit_state(|state| {
        state.driver.queue_poll[Q] = v.poll.then_some(PollState { chip: 0, link: 0, stream: 0, since: 5 })
    })
    .unwrap();
    m.edit_state(|state| state.driver.poll_n = u32::from(v.poll)).unwrap();
    for (p, &(busy, tail, state)) in v.paths.iter().enumerate() {
        m.edit_state(|state| state.driver.path_busy_until[2 * Q + p] = if busy { (CYCLE + 10).into() } else { 0 })
            .unwrap();
        m.edit_state(|state| state.driver.path_tail_until[2 * Q + p] = if tail { (CYCLE + 10).into() } else { 0 })
            .unwrap();
        let state = [AgtState::Idle, AgtState::Config, AgtState::Run][state as usize];
        m.edit_state(|edit| edit.driver.agt_set(Q, p, Agt { state, ..Agt::idle() })).unwrap();
    }
}

/// `hold_decide`: is the instruction `kind` on dispatch path `path` held, in the state `view`?
fn held(m: &mut Machine, view: View, kind: Kind, path: Option<usize>) -> bool {
    set_view(m, view);
    m.hold_check(Q, CYCLE, Some(&insn(kind)), path, false)
}

/// `view_paths_busy`: a path busy, in a tail, or with an AGT that is not idle.
fn paths_busy(v: &View) -> bool {
    v.paths.iter().any(|&(busy, tail, state)| busy || tail || state != 0)
}

/// `view_free`: no queue flag and both paths idle.
fn view_free(v: &View) -> bool {
    !(v.parked || v.host || v.busy || v.repeat || v.poll || paths_busy(v))
}

// model: unit_tests/test_pure_helpers.sail test_hold_decide_queue_flags
#[test]
fn hold_decide_queue_flags() {
    let mut m = machine();
    let free = View::default();
    assert!(view_free(&free));
    assert!(!held(&mut m, free, Kind::Plain, None));
    assert!(!held(&mut m, free, Kind::Plain, Some(0)));
    // a parked queue holds every instruction except its own Sync or Notify
    let parked = View { parked: true, ..free };
    assert!(held(&mut m, parked, Kind::Plain, None));
    assert!(!held(&mut m, parked, Kind::Barrier, None));
    assert!(!view_free(&parked));
    // a host park holds the barrier instructions too
    assert!(held(&mut m, View { host: true, ..free }, Kind::Barrier, None));
    assert!(held(&mut m, View { busy: true, ..free }, Kind::ConfigRecord, Some(1)));
    assert!(held(&mut m, View { repeat: true, ..free }, Kind::Plain, Some(0)));
    assert!(held(&mut m, View { poll: true, ..free }, Kind::Barrier, None));
}

// model: unit_tests/test_pure_helpers.sail test_hold_decide_paths
#[test]
fn hold_decide_paths() {
    let mut m = machine();
    let free = View::default();
    let with = |p0: PathView, p1: PathView| View { paths: [p0, p1], ..free };
    let idle = (false, false, 0);
    let busy1 = with(idle, (true, false, 0));
    assert!(!held(&mut m, busy1, Kind::Plain, Some(0)));
    assert!(held(&mut m, busy1, Kind::Plain, Some(1)));
    // a whole-queue instruction needs both paths free; an Ifetch on a MEM queue only its own path
    assert!(held(&mut m, busy1, Kind::Plain, None));
    assert!(!held(&mut m, busy1, Kind::WholeFree, None));
    let tail0 = with((false, true, 0), idle);
    assert!(held(&mut m, tail0, Kind::Plain, Some(0)));
    assert!(!held(&mut m, tail0, Kind::Plain, Some(1)));
    assert!(held(&mut m, tail0, Kind::Plain, None));
    // a loop runs: it holds its path in both states; a configuration open on a path holds it against all but its records
    let run0 = with((false, false, 2), idle);
    assert!(held(&mut m, run0, Kind::ConfigRecord, Some(0)));
    assert!(held(&mut m, run0, Kind::Plain, None));
    let cfg1 = with(idle, (false, false, 1));
    assert!(held(&mut m, cfg1, Kind::Plain, Some(1)));
    assert!(!held(&mut m, cfg1, Kind::ConfigRecord, Some(1)));
    assert!(held(&mut m, cfg1, Kind::Plain, None));
    assert!(!view_free(&cfg1));
}

// model: unit_tests/test_pure_helpers.sail test_hold_decide_exhaustive
#[test]
fn hold_decide_exhaustive() {
    let mut m = machine();
    let path_view = |n: u32| -> PathView { (n & 1 == 1, n >> 1 & 1 == 1, (n / 4).min(2) as u8) };
    for f in 0..32u32 {
        for a in 0..12 {
            for b in 0..12 {
                let view = View {
                    parked: f & 1 == 1,
                    host: f >> 1 & 1 == 1,
                    busy: f >> 2 & 1 == 1,
                    repeat: f >> 3 & 1 == 1,
                    poll: f >> 4 & 1 == 1,
                    paths: [path_view(a), path_view(b)],
                };
                if view_free(&view) {
                    assert!(!held(&mut m, view, Kind::Plain, None), "free view {f} {a} {b}");
                    assert!(!held(&mut m, view, Kind::Plain, Some(0)), "free view {f} {a} {b}");
                    assert!(!held(&mut m, view, Kind::ConfigRecord, Some(1)), "free view {f} {a} {b}");
                } else if f == 0 {
                    assert_eq!(held(&mut m, view, Kind::Plain, None), paths_busy(&view), "paths {a} {b}");
                    assert!(!held(&mut m, view, Kind::WholeFree, None), "paths {a} {b}");
                }
            }
        }
    }
}

// ---- the address generation table ----------------------------------------------------------------------------------

/// `agt_opened`: a configuration opened by a reference dispatched at `t0`.
fn agt_opened(t0: u64) -> Agt {
    Agt { state: AgtState::Config, t0, last: t0, ..Agt::idle() }
}

const COUNTDOWN: u8 = 0;
const STEP: u8 = 1;
const ITERATIONS: u8 = 3;

// model: unit_tests/test_pure_helpers.sail test_agt_configure_pure
#[test]
fn agt_configure_pure() {
    let a0 = agt_opened(5);
    assert!(a0.state == AgtState::Config && a0.t0 == 5 && a0.last == 5 && a0.pos == -1);
    let mut a = a0;
    assert!(agt_configure(&mut a, COUNTDOWN, 0x0000));
    assert!(a.levels[0].countdown == 1 && a.pos == 0 && a.level == 0, "below 1 counts as 1");
    assert!(agt_configure(&mut a, STEP, 0xfffe));
    assert!(a.levels[0].step == -2 && a.pos == 1);
    assert!(agt_configure(&mut a, COUNTDOWN, 0x0004), "a field that does not advance in order starts the next level");
    assert!(a.level == 1 && a.pos == 0 && a.levels[1].countdown == 4 && a.levels[0].step == -2);
    assert!(agt_configure(&mut a, COUNTDOWN, 0x0001));
    assert!(agt_configure(&mut a, COUNTDOWN, 0x0001));
    assert_eq!(a.level, 3);
    let before = a;
    assert!(!agt_configure(&mut a, COUNTDOWN, 0x0001), "a fifth level");
    assert_eq!(a, before, "a refused record changes nothing");
    let mut a = a0;
    assert!(agt_configure(&mut a, ITERATIONS, 0x0003) && a.levels[0].iters == 3);
}

const LEVEL: AgtLevel = AgtLevel { countdown: 3, step: 0, sstep: 0, iters: 2, idx: 1 };

fn running(level: AgtLevel, last: u64) -> Agt {
    let mut a = Agt { state: AgtState::Run, level: 0, pos: 0, last, ..Agt::idle() };
    a.levels[0] = level;
    a
}

// model: unit_tests/test_pure_helpers.sail test_agt_refresh_pure
#[test]
fn agt_refresh_pure() {
    let (q, p) = (3, 0);
    let tail = |m: &Machine| m.driver().path_tail_until[2 * q + p];
    // no level can advance: idle, held through the trailing waits
    let mut m = machine();
    m.edit_state(|edit| edit.driver.agt_set(q, p, running(LEVEL, 40))).unwrap();
    m.edit_state(|state| state.driver.agt_refresh(q, p)).unwrap();
    assert_eq!((m.driver().agt_get(q, p).state, tail(&m)), (AgtState::Idle, 44));
    // a level that can advance: the next reference after the countdown
    let mut m = machine();
    m.edit_state(|edit| edit.driver.agt_set(q, p, running(AgtLevel { idx: 0, ..LEVEL }, 40))).unwrap();
    m.edit_state(|state| state.driver.agt_refresh(q, p)).unwrap();
    let b = m.driver().agt_get(q, p);
    assert_eq!((b.state, b.next, tail(&m)), (AgtState::Run, 44, 0));
    // an AGT that is not running is unchanged
    let mut m = machine();
    let config = Agt { state: AgtState::Config, ..running(LEVEL, 40) };
    m.edit_state(|edit| edit.driver.agt_set(q, p, config)).unwrap();
    m.edit_state(|state| state.driver.agt_refresh(q, p)).unwrap();
    assert_eq!((m.driver().agt_get(q, p), tail(&m)), (config, 0));
    // the end of a configuration runs the loop
    m.edit_state(|edit| {
        edit.driver.agt_set(q, p, Agt { state: AgtState::Config, ..running(AgtLevel { idx: 0, ..LEVEL }, 40) })
    })
    .unwrap();
    m.edit_state(|state| state.driver.agt_finish_config(q, p)).unwrap();
    let d = m.driver().agt_get(q, p);
    assert_eq!((d.state, d.next), (AgtState::Run, 44));
}

// model: unit_tests/test_pure_helpers.sail test_agt_due_and_advanced
#[test]
fn agt_due_and_advanced() {
    let l0 = AgtLevel { countdown: 2, step: 8, sstep: 1, iters: 3, idx: 0 };
    let run = running(l0, 10);
    assert!(agt_due(&run, 12, false).is_none(), "the reference after 1 + countdown = 3 cycles");
    assert!(agt_due(&run, 13, true).is_none(), "parked");
    let (ls, d, s, ss) = agt_due(&run, 13, false).expect("due at 13");
    assert_eq!((d, s, ss, ls[0].idx), (3, 8, 1, 1));
    assert!(agt_due(&Agt::idle(), 100, false).is_none());
    // the allocation-free form agrees: the same delay and steps at the same cycle
    assert_eq!(agt_due_lean(&run, 13, false), Some((3, 8, 1)));
    assert!(agt_due_lean(&run, 12, false).is_none() && agt_due_lean(&run, 13, true).is_none());
}

// model: unit_tests/test_pure_helpers.sail test_cfg_path_pick
#[test]
fn cfg_path_pick() {
    let mut m = machine();
    let q = 4;
    let mut pick = |r: bool, w: bool, rr: u64, wr: u64| {
        let state = |on: bool| Agt { state: if on { AgtState::Config } else { AgtState::Idle }, ..Agt::idle() };
        m.edit_state(|edit| edit.driver.agt_set(q, 0, state(r))).unwrap();
        m.edit_state(|edit| edit.driver.agt_set(q, 1, state(w))).unwrap();
        m.edit_state(|state| state.driver.agt_records[2 * q] = rr).unwrap();
        m.edit_state(|state| state.driver.agt_records[2 * q + 1] = wr).unwrap();
        m.driver().cfg_path(q)
    };
    assert_eq!(pick(true, false, 0, 0), 0);
    assert_eq!(pick(false, true, 0, 0), 1);
    assert_eq!(pick(false, false, 0, 0), 0);
    assert_eq!(pick(true, true, 0, 0), 0, "a tie: R");
    assert_eq!(pick(true, true, 1, 0), 1, "W has fewer records so far");
    assert_eq!(pick(true, true, 1, 1), 0);
}

// ---- commit arbitration ---------------------------------------------------------------------------------------------

fn commit(q: usize, b: u8, x: u8) -> Commit {
    Commit { r: RowId::E, id: 4, g: 20, value: v(x), q, b, ov: 0 }
}

#[derive(Default)]
struct Hits(Vec<usize>);
impl Observer for Hits {
    fn diag(&mut self, _c: u64, q: u16, code: Code) {
        assert_eq!(code, Code::Srf002);
        self.0.push(q as usize);
    }
}

/// Commit `list` at cycle 7: the queues that collide, and the first lane of the register the target holds afterwards.
fn commit_list(list: Vec<Commit>) -> (Vec<usize>, Option<u8>) {
    let mut m = machine();
    m.edit_state(|state| state.driver.set_pending_list(7, list)).unwrap();
    let mut hits = Hits::default();
    m.commit_due(7, &mut hits);
    let held = m.srf_value(Row::East, 4, 21, 8).map(|x| x[0]);
    hits.0.sort_unstable();
    (hits.0, held)
}

// model: unit_tests/test_pure_helpers.sail test_commit_wins
#[test]
fn commit_wins() {
    let (a, b) = (commit(2, 0, 0x01), commit(5, 0, 0x02));
    // one writer: it wins with its own value
    assert_eq!(commit_list(vec![a.clone()]), (vec![], Some(0x01)));
    // two writers on one target that are not an OR pair collide; the lowest (queue, plane) writer keeps its value
    assert_eq!(commit_list(vec![a.clone(), b.clone()]), (vec![2, 5], Some(0x01)));
    assert_eq!(commit_list(vec![b.clone(), a.clone()]), (vec![2, 5], Some(0x01)));
    // the writes of one (queue, plane) are one writer: their OR is stored once
    let a2 = commit(2, 0, 0x10);
    assert_eq!(commit_list(vec![a.clone(), a2.clone()]), (vec![], Some(0x11)));
    assert_eq!(commit_list(vec![a2.clone(), a.clone()]), (vec![], Some(0x11)));
    // writes to another gap are another target
    let d = Commit { g: 21, ..commit(9, 0, 0x04) };
    assert_eq!(commit_list(vec![a.clone(), d]), (vec![], Some(0x01)));
    // a SEL queue and a TSP queue of one hemisphere merge: no collision, the register takes the OR
    let (tsp, sel) = (commit(120, 0, 0x01), commit(125, 0, 0x02));
    assert_eq!(commit_list(vec![tsp.clone(), sel.clone()]), (vec![], Some(0x03)));
    assert_eq!(commit_list(vec![sel.clone(), tsp.clone()]), (vec![], Some(0x03)));
    // two TSP queues are not an OR pair
    let tsp2 = commit(121, 0, 0x04);
    assert_eq!(commit_list(vec![tsp.clone(), tsp2]), (vec![120, 121], Some(0x01)));
    // three writers collide
    let (hits, _) = commit_list(vec![tsp, sel, commit(3, 0, 0x08)]);
    assert_eq!(hits, vec![3, 120, 125]);
}

// model: unit_tests/test_pure_helpers.sail test_commit_due_applies
#[test]
fn commit_due_applies() {
    let mut m = machine();
    m.edit_state(|state| state.driver.set_pending_list(7, vec![commit(5, 0, 0x02), commit(2, 0, 0x01)])).unwrap();
    m.commit_due(7, &mut tsp_iss::NoObserver);
    assert_eq!(
        m.srf_value(Row::East, 4, 21, 8).map(|x| x[0]),
        Some(0x01),
        "a collision keeps the lowest writer's value"
    );
    assert!(m.driver().pending[7].is_empty(), "the ring slot is cleared");
}
