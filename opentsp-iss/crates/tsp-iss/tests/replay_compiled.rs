//! The compiled replay (`driver::replay`, level 2): a run of a recorded schedule executes the recorded operations of
//! each tick (the elements that execute, the dispatches and the driver's reports) and the commit ring, and nothing of
//! the IQs, the hold checks or the calendar. It must be the normal run: every event in order and what `--stats`
//! reads after the run. The check mode (`TSP_ISS_REPLAY_CHECK`) compares the operations of the normal driver with
//! the recording per tick, so an altered recording is caught there.
//!
//! The programs are those of `tests/data/replay/` (the generated mixed programs: every unit, Repeat, AGT loops,
//! Sync and Notify, diagnostics of every kind).
#![cfg(feature = "replay")]
use ots_profile::Profile;
use std::path::PathBuf;
use std::sync::Arc;
use tsp_iss::driver::replay::{Mode, Op, Schedule};
use tsp_iss::links::NoLinks;
use tsp_iss::machine::{Machine, RunError};
use tsp_iss::observe::{Hold, Observer, Row};

const MAX_CYCLES: u64 = 6000;

#[test]
fn local_fetch_second_read_is_recorded_replayed_and_hashed() {
    let prog = "Q 0 01c00010\nS 0 10 ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff\nS 0 11 12 00 00 10 00 00 00 1c ff ff ff ff ff ff ff ff\n";
    let (normal, schedule) = record("mini", prog, false);
    let schedule = schedule.expect("local fetch is recordable");
    assert!(normal.0.as_ref().unwrap().iter().any(|e| e == "fetch 1 0 32"));
    let (_, again) = replay("mini", prog, &schedule, false, false);
    assert_eq!(normal, again);
    let changed = prog.replace("S 0 11 12", "S 0 11 13");
    let (_, mismatch) = replay("mini", &changed, &schedule, false, false);
    assert!(mismatch.0.is_err(), "changed second word escaped the replay hash");
}

#[test]
fn changed_fetched_bytes_are_refused_even_when_legacy_text_digest_is_equal() {
    use tsp_iss::driver::replay::TextHash;
    // One local fetch completes at cycle1. Change its first payload while
    // preserving the old polynomial digest; data equality must decide safety.
    let factor = 0x0000_0100_0000_01b3u64;
    let prefix = (1u64 << 8).wrapping_mul(factor);
    let old_chunk = u64::MAX;
    let new_chunk = old_chunk ^ 1;
    let correction = (prefix ^ old_chunk).wrapping_mul(factor) ^ (prefix ^ new_chunk).wrapping_mul(factor) ^ old_chunk;
    let original = [0xffu8; 16];
    let mut changed = original;
    changed[..8].copy_from_slice(&new_chunk.to_le_bytes());
    changed[8..].copy_from_slice(&correction.to_le_bytes());
    let second = [0x12, 0, 0, 0x10, 0, 0, 0, 0x1c, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff];
    let digest = |a: &[u8]| {
        let mut text = TextHash::default();
        text.on = true;
        text.note(0, 1, a, &second);
        text.h
    };
    assert_ne!(changed, original);
    assert_eq!(digest(&changed), digest(&original), "the witness must preserve even the full legacy digest");
    let program = |a: &[u8]| {
        format!(
            "Q 0 01c00010\nS 0 10 {}\nS 0 11 12 00 00 10 00 00 00 1c ff ff ff ff ff ff ff ff\n",
            a.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ")
        )
    };
    let old = program(&original);
    let new = program(&changed);
    let (normal, schedule) = record("mini", &old, false);
    let schedule = schedule.expect("local fetch must be recordable");
    assert!(normal.0.is_ok());
    let changed_normal = run(machine("mini", &new), false);
    assert_ne!(normal.0, changed_normal.0, "changed payload must change observable normal behavior");
    let (_, changed_replay) = replay("mini", &new, &schedule, false, false);
    assert!(
        matches!(changed_replay.0, Err(RunError::Internal(ref e)) if e.contains("replay divergence")),
        "changed instruction bytes were silently accepted by compiled replay: {:?}",
        changed_replay.0
    );
}

#[derive(Default)]
struct Log {
    waits: bool,
    ev: Vec<String>,
}

impl Observer for Log {
    fn srf(&mut self, c: u64, row: Row, stream: u8, gap: u8, v: &[u8]) {
        self.ev.push(format!("srf {c} {row:?} {stream} {gap} {v:02x?}"));
    }
    fn sram(&mut self, c: u64, slice: u8, addr: u32, v: &[u8]) {
        self.ev.push(format!("sram {c} {slice} {addr} {v:02x?}"));
    }
    fn diag(&mut self, c: u64, queue: u16, code: ots_isa::Code) {
        self.ev.push(format!("diag {c} {queue} {}", code.as_str()));
    }
    fn lint(&mut self, c: u64, queue: u16, code: &'static str) {
        self.ev.push(format!("lint {c} {queue} {code}"));
    }
    fn dispatch(&mut self, c: u64, queue: u16) {
        self.ev.push(format!("dispatch {c} {queue}"));
    }
    fn wait(&mut self, c: u64, queue: u16, why: Hold, since: u64) {
        self.ev.push(format!("wait {c} {queue} {why:?} {since}"));
    }
    fn fetch(&mut self, c: u64, queue: u16, bytes: u32) {
        self.ev.push(format!("fetch {c} {queue} {bytes}"));
    }
    fn tracks_waits(&self) -> bool {
        self.waits
    }
}

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/replay")
}

fn data(name: &str) -> String {
    let p = data_dir().join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn names() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(data_dir())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n.ends_with(".txt"))
        .collect();
    v.sort();
    v
}

fn profile_of(name: &str) -> &'static str {
    if name.starts_with("ref") {
        "ref"
    } else {
        "mini"
    }
}

fn machine(profile: &str, prog: &str) -> Machine {
    let mut m = Machine::new(&Profile::load(profile).unwrap(), 0, Box::new(NoLinks)).unwrap();
    for l in prog.lines() {
        let f: Vec<&str> = l.split(' ').collect();
        let hex = |s: &str| u64::from_str_radix(s, 16).unwrap();
        match f[0] {
            "S" => {
                let bytes: Vec<u8> = f[3..].iter().map(|b| hex(b) as u8).collect();
                m.preload(hex(f[1]) as u8, hex(f[2]) as u32, &bytes).unwrap();
            }
            "Q" => {
                let units: Vec<u32> = f[2..].iter().map(|u| hex(u) as u32).collect();
                m.load_image(hex(f[1]) as u16, &units).unwrap();
            }
            _ => {}
        }
    }
    m
}

/// What a run leaves that `--stats` and the platform read: the IQ levels of every queue and quiescence.
fn levels(m: &Machine) -> String {
    let q = (0..144u16).chain([0xffff]).map(|q| format!("{:?}", m.queue_state(q).unwrap())).collect::<Vec<_>>();
    format!("{};quiescent={}", q.join(";"), m.quiescent())
}

type Outcome = (Result<Vec<String>, RunError>, String);

fn run(mut m: Machine, waits: bool) -> Outcome {
    let mut log = Log { waits, ev: Vec::new() };
    let r = m.run(MAX_CYCLES, &mut log);
    (r.map(|_| log.ev), levels(&m))
}

fn record(profile: &str, prog: &str, waits: bool) -> (Outcome, Option<Schedule>) {
    let mut m = machine(profile, prog);
    m.record_schedule().unwrap();
    let mut log = Log { waits, ev: Vec::new() };
    let r = m.run(MAX_CYCLES, &mut log);
    let ok = r.is_ok();
    let s = ok.then(|| m.take_schedule()).flatten();
    ((r.map(|_| log.ev), levels(&m)), s)
}

/// A replay; the machine is returned with the outcome to see which path it took.
fn replay(profile: &str, prog: &str, s: &Schedule, check: bool, waits: bool) -> (Machine, Outcome) {
    let mut m = machine(profile, prog);
    m.replay_schedule_with(Arc::new(s.clone()), check).unwrap();
    let compiled = m.replay_state().compiled;
    let mut log = Log { waits, ev: Vec::new() };
    let r = m.run(MAX_CYCLES, &mut log);
    assert_eq!(compiled, !check && s.compiled(), "the path of the replay");
    let lv = levels(&m);
    (m, (r.map(|_| log.ev), lv))
}

fn recordable(waits: bool) -> Vec<(String, Outcome, Schedule)> {
    let mut out = Vec::new();
    for n in names() {
        let ((r, d), s) = record(profile_of(&n), &data(&n), waits);
        if let (Ok(ev), Some(s)) = (r, s) {
            out.push((n, (Ok(ev), d), s));
        }
    }
    out
}

fn flip(prog: &str) -> String {
    prog.lines()
        .map(|l| {
            if l.starts_with("S ") {
                let mut f: Vec<String> = l.split(' ').map(String::from).collect();
                for b in f.iter_mut().skip(3) {
                    *b = format!("{:x}", u8::from_str_radix(b, 16).unwrap() ^ 0x5a);
                }
                f.join(" ")
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn a_compiled_replay_is_the_normal_run_with_and_without_wait_tracking() {
    let mut n_programs = 0;
    for waits in [false, true] {
        let progs = recordable(waits);
        assert!(progs.len() >= 12, "only {} recordable programs", progs.len());
        for (n, normal, s) in &progs {
            assert!(s.compiled(), "{n}: the recording is not compiled");
            let (m, again) = replay(profile_of(n), &data(n), s, false, waits);
            assert_eq!(m.replay_mode(), Mode::Replay);
            assert!(m.replay_state().compiled_done, "{n}: the compiled replay did not reach its last tick");
            assert_eq!(normal.0.as_ref().unwrap(), again.0.as_ref().unwrap(), "{n} (waits {waits}): events differ");
            assert_eq!(normal.1, again.1, "{n} (waits {waits}): the state after the run differs");
            n_programs += 1;
        }
    }
    assert!(n_programs >= 24);
}

#[test]
fn a_compiled_replay_serves_other_data_of_the_same_program() {
    let mut served = 0;
    for (n, _, s) in recordable(false) {
        let flipped = flip(&data(&n));
        let normal = run(machine(profile_of(&n), &flipped), false);
        let (_, again) = replay(profile_of(&n), &flipped, &s, false, false);
        match (&normal.0, &again.0) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a, b, "{n}: events differ on other data");
                assert_eq!(normal.1, again.1, "{n}: state differs on other data");
                served += 1;
            }
            (_, Err(RunError::Internal(m))) if m.contains("replay divergence") => {}
            (a, b) => panic!("{n}: normal {:?}, replay {:?}", a.as_ref().err(), b.as_ref().err()),
        }
    }
    assert!(served >= 6, "only {served} programs served other data");
}

/// A commit that other data removes (a data-dependent diagnostic drops a write) is a tick of the recording and not of
/// the other run, and the reverse: the compiled replay follows the commit ring there too.
#[test]
fn a_compiled_replay_follows_a_commit_that_other_data_removes_or_adds() {
    let prog = data("mini-xtick.txt");
    let other = flip(&prog);
    for (from, to) in [(&prog, &other), (&other, &prog)] {
        let s = record("mini", from, false).1.expect("recordable");
        let normal = run(machine("mini", to), false);
        let (_, again) = replay("mini", to, &s, false, false);
        assert_eq!(normal.0.as_ref().unwrap(), again.0.as_ref().unwrap());
        assert_eq!(normal.1, again.1);
    }
}

/// The alterations of the operation log of one program; each must be seen by the check mode.
fn op_mutants(s: &Schedule) -> Vec<(&'static str, Schedule)> {
    let mut out = Vec::new();
    let mid = s.ticks() / 2;
    let tick_with = |p: &dyn Fn(&Op) -> bool| {
        (mid..s.ticks()).find(|&i| s.ops[s.ops_off[i] as usize..s.ops_off[i + 1] as usize].iter().any(p))
    };
    let at = |i: usize, p: &dyn Fn(&Op) -> bool| {
        (s.ops_off[i] as usize..s.ops_off[i + 1] as usize).find(|&k| p(&s.ops[k])).unwrap()
    };
    let remove_op = |m: &mut Schedule, i: usize, k: usize| {
        m.ops.remove(k);
        for o in m.ops_off.iter_mut().skip(i + 1) {
            *o -= 1;
        }
    };
    if let Some(i) = tick_with(&|o| matches!(o, Op::Dispatch(_))) {
        let k = at(i, &|o| matches!(o, Op::Dispatch(_)));
        let mut m = s.clone();
        remove_op(&mut m, i, k);
        out.push(("a dispatch is missing", m));
        let mut m = s.clone();
        if let Op::Dispatch(q) = m.ops[k] {
            m.ops[k] = Op::Dispatch(q ^ 1);
        }
        out.push(("a dispatch is on another queue", m));
    }
    if let Some(i) = tick_with(&|o| matches!(o, Op::Exec)) {
        let k = at(i, &|o| matches!(o, Op::Exec));
        let mut m = s.clone();
        remove_op(&mut m, i, k);
        // the element stays in `execs` of the tick: the lists of the tick differ
        out.push(("an execution is missing from the operations", m));
        let e0 = s.execs_off[i] as usize;
        let mut m = s.clone();
        m.execs[e0].q ^= 1;
        out.push(("an element runs for another queue", m));
        let mut m = s.clone();
        m.execs[e0].c0 += 1;
        out.push(("an element was issued one cycle late", m));
        let mut m = s.clone();
        m.execs[e0].off += 1;
        out.push(("an element has another address offset", m));
    }
    if let Some(i) = tick_with(&|o| matches!(o, Op::Diag(..))) {
        let k = at(i, &|o| matches!(o, Op::Diag(..)));
        let mut m = s.clone();
        remove_op(&mut m, i, k);
        out.push(("a driver diagnostic is missing", m));
    }
    // two operations of one tick swapped
    if let Some(i) = (mid..s.ticks()).find(|&i| {
        let o = &s.ops[s.ops_off[i] as usize..s.ops_off[i + 1] as usize];
        o.len() >= 2 && o[0] != o[1]
    }) {
        let mut m = s.clone();
        m.ops.swap(s.ops_off[i] as usize, s.ops_off[i] as usize + 1);
        out.push(("two operations are in the other order", m));
    }
    // an operation moved to the next tick
    if let Some(i) = tick_with(&|o| matches!(o, Op::Dispatch(_))).filter(|&i| i + 1 < s.ticks()) {
        let k = at(i, &|o| matches!(o, Op::Dispatch(_)));
        let mut m = s.clone();
        let op = m.ops.remove(k);
        let ins = m.ops_off[i + 1] as usize - 1; // the next tick starts one earlier now
        m.ops.insert(ins, op);
        m.ops_off[i + 1] -= 1;
        out.push(("a dispatch is in the next tick", m));
    }
    out
}

#[test]
fn an_altered_operation_log_is_caught_by_the_check_and_never_replayed_silently_wrong_by_the_key() {
    let mut total = 0;
    let mut kinds = std::collections::BTreeSet::new();
    for waits in [false, true] {
        for (n, _, s) in recordable(waits).into_iter().filter(|(_, _, s)| s.ticks() > 40).take(8) {
            for (what, m) in op_mutants(&s) {
                total += 1;
                kinds.insert(what);
                let (_, c) = replay(profile_of(&n), &data(&n), &m, true, waits);
                match &c.0 {
                    Err(RunError::Internal(e)) if e.contains("replay divergence") => {}
                    other => panic!("{n}: {what}: the check did not catch it: {:?}", other.as_ref().map(|e| e.len())),
                }
            }
        }
    }
    eprintln!("altered operation logs: {total}, kinds {}", kinds.len());
    assert!(total >= 40, "only {total} alterations were tried");
    assert!(kinds.len() >= 8, "only {} kinds of alteration: {kinds:?}", kinds.len());
}

#[test]
fn a_wrong_text_hash_stops_the_compiled_replay() {
    // a program that fetches text: the hash changes in the tick of an Ifetch
    let (name, _, s) = recordable(false)
        .into_iter()
        .find(|(_, _, s)| s.ticks() > 40 && (1..s.ticks()).any(|i| s.text[i] != s.text[i - 1]))
        .expect("a program that fetches text");
    let i = (1..s.ticks()).find(|&i| s.text[i] != s.text[i - 1]).unwrap();
    let mut m = s.clone();
    m.text[i] ^= 1;
    let (_, r) = replay(profile_of(&name), &data(&name), &m, false, false);
    match r.0 {
        Err(RunError::Internal(e)) if e.contains("replay divergence") => {}
        other => panic!("expected a divergence, got {:?}", other.as_ref().map(|e| e.len())),
    }
}

/// A schedule without its operations (the first form of the replay, which replayed the queues visited) cannot be
/// replayed: it is refused, and the run takes the normal driver.
#[test]
fn a_schedule_without_operations_is_refused() {
    use tsp_iss::driver::replay::Refused;
    let (n, normal, mut s) = recordable(false).into_iter().next().unwrap();
    s.ops.clear();
    s.ops_off.clear();
    s.execs.clear();
    s.execs_off.clear();
    assert!(!s.compiled());
    let mut m = machine(profile_of(&n), &data(&n));
    assert!(matches!(m.replay_schedule_with(Arc::new(s), false), Err(Refused::NotApplicable(_))));
    assert_eq!(m.replay_mode(), Mode::Off);
    let again = run(m, false);
    assert_eq!(normal.0.as_ref().unwrap(), again.0.as_ref().unwrap());
}

/// Ticking every cycle (`--tick-every-cycle`) ticks the idle cycles too, which no schedule describes: the run takes the
/// normal driver and is the normal run with that option.
#[test]
fn a_run_that_ticks_every_cycle_takes_the_normal_driver() {
    let (n, _, s) = recordable(false).into_iter().find(|(_, _, s)| s.ticks() > 40).unwrap();
    let go = |replay: bool| {
        let mut m = machine(profile_of(&n), &data(&n));
        if replay {
            m.replay_schedule_with(Arc::new(s.clone()), false).unwrap();
        }
        m.set_tick_every_cycle(true);
        let mut log = Log::default();
        let r = m.run(MAX_CYCLES, &mut log).map(|s| (s.ticks, s.last_cycle));
        (r.unwrap(), log.ev, m.replay_mode())
    };
    let (normal, replayed) = (go(false), go(true));
    assert_eq!(normal.0, replayed.0);
    assert_eq!(normal.1, replayed.1);
    assert_eq!(replayed.2, Mode::Off, "the replay was given up");
}
