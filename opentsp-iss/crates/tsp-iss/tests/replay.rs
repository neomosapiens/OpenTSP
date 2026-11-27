//! Schedule record and replay (`driver::replay`): a replayed run is the normal run (every event, in order, and the
//! machine's state at the end), and the key refuses what the recording does not cover. Altered recordings are in
//! `replay_compiled.rs`.
//!
//! The programs are the generated mixed programs of `mixed_diff` (every unit, Repeat, AGT loops, Sync and Notify,
//! diagnostics of every kind) saved to `tests/data/replay/`; a program that uses a chip-to-chip link is not
//! recordable and the tests say so.
#![cfg(feature = "replay")]
use ots_profile::Profile;
use std::path::PathBuf;
use std::sync::Arc;
use tsp_iss::driver::replay::{Began, Mode, Refused, Schedule};
use tsp_iss::links::NoLinks;
use tsp_iss::machine::{Machine, RunError};
use tsp_iss::observe::{Hold, Observer, Row};

const MAX_CYCLES: u64 = 6000;

/// Every event of a run, in order; `waits` is `tracks_waits`.
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

fn data(name: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/replay").join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn names() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/replay"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n.ends_with(".txt"))
        .collect();
    v.sort();
    v
}

/// A machine for the program text of the harness protocol (`S` preload lines and `Q` image lines, hex).
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

fn profile_of(name: &str) -> &'static str {
    if name.starts_with("ref") {
        "ref"
    } else {
        "mini"
    }
}

/// What a run leaves that a user can see besides its events: the IQ levels of every queue (the state digest holds
/// the event index of the driver too, which a replay does not keep).
fn levels(m: &Machine) -> String {
    (0..144u16).chain([0xffff]).map(|q| format!("{:?}", m.queue_state(q).unwrap())).collect::<Vec<_>>().join(";")
}

/// The result of a run: its events, or the error; and the state it left.
type Outcome = (Result<Vec<String>, RunError>, String);

fn run(mut m: Machine, waits: bool) -> Outcome {
    let mut log = Log { waits, ev: Vec::new() };
    let r = m.run(MAX_CYCLES, &mut log);
    (r.map(|_| log.ev), levels(&m))
}

/// A run that records: its outcome and the schedule (`None`: not recordable or the run failed).
fn record(profile: &str, prog: &str, waits: bool) -> (Outcome, Option<Schedule>) {
    let mut m = machine(profile, prog);
    m.record_schedule().unwrap();
    let mut log = Log { waits, ev: Vec::new() };
    let r = m.run(MAX_CYCLES, &mut log);
    let ok = r.is_ok();
    let s = ok.then(|| m.take_schedule()).flatten();
    ((r.map(|_| log.ev), levels(&m)), s)
}

fn replay(profile: &str, prog: &str, s: &Schedule, check: bool, waits: bool) -> Outcome {
    let mut m = machine(profile, prog);
    m.replay_schedule_with(Arc::new(s.clone()), check).unwrap();
    run(m, waits)
}

/// The programs a schedule could be recorded for, with the schedule and the normal run's outcome.
fn recordable(waits: bool) -> Vec<(String, Outcome, Schedule)> {
    let mut out = Vec::new();
    for n in names() {
        let prog = data(&n);
        let ((r, d), s) = record(profile_of(&n), &prog, waits);
        if let (Ok(ev), Some(s)) = (r, s) {
            out.push((n, (Ok(ev), d), s));
        }
    }
    out
}

#[test]
fn the_corpus_has_enough_recordable_programs() {
    let r = recordable(false);
    let total = names().len();
    eprintln!("replay corpus: {} of {total} programs recordable", r.len());
    eprintln!("RECORDABLE {:?}", r.iter().map(|(n, _, s)| format!("{n}:{}", s.ticks())).collect::<Vec<_>>());
    assert!(r.len() >= 8, "only {} of {total} programs are recordable", r.len());
    // the recording is the normal run: ticks and the schedule shape agree
    for (n, _, s) in &r {
        assert!(s.ticks() > 0 && s.compiled(), "{n}");
    }
}

#[test]
fn a_replay_is_the_normal_run_with_and_without_wait_tracking() {
    for waits in [false, true] {
        let progs = recordable(waits);
        assert!(!progs.is_empty());
        for (n, normal, s) in &progs {
            let again = replay(profile_of(n), &data(n), s, false, waits);
            assert!(normal.0.is_ok() && again.0.is_ok(), "{n}: {:?}", again.0.as_ref().err());
            assert_eq!(normal.0.as_ref().unwrap(), again.0.as_ref().unwrap(), "{n} (waits {waits}): events differ");
            assert_eq!(normal.1, again.1, "{n} (waits {waits}): the final state differs");
            let checked = replay(profile_of(n), &data(n), s, true, waits);
            assert_eq!(
                normal.0.as_ref().unwrap(),
                checked.0.as_ref().unwrap(),
                "{n} (waits {waits}): check mode differs"
            );
        }
    }
}

/// The same program with every preload byte changed (the data of another request; the text is the same).
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
fn a_replay_serves_other_data_of_the_same_program() {
    // the data (S lines) is not part of the key: change every preload, the schedule still fits and the replay is
    // the normal run on the new data
    let mut served = 0;
    for (n, _, s) in recordable(false) {
        let prog = data(&n);
        let flipped = flip(&prog);
        let normal = run(machine(profile_of(&n), &flipped), false);
        let again = replay(profile_of(&n), &flipped, &s, false, false);
        match (&normal.0, &again.0) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a, b, "{n}: events differ on other data");
                assert_eq!(normal.1, again.1, "{n}: state differs on other data");
                served += 1;
            }
            // other data may change which text an Ifetch brings: then the replay must stop, not differ
            (_, Err(RunError::Internal(m))) if m.contains("replay divergence") => {}
            (a, b) => panic!("{n}: normal {:?}, replay {:?}", a.as_ref().err(), b.as_ref().err()),
        }
    }
    assert!(served >= 6, "only {served} programs served other data");
}

/// `mini-xtick` on other data drops a write with a diagnostic (a data-dependent one): the commit of that write is
/// not a tick of the other run, though it is one of the recording. The replay must follow the data there too, with
/// the same events and the same state as the normal run.
#[test]
fn a_commit_that_other_data_removes_or_adds_is_followed_by_the_replay() {
    let prog = data("mini-xtick.txt");
    let (_, s) = record("mini", &prog, false);
    let s = s.expect("recordable");
    let other = flip(&prog);
    let (normal, state) = run(machine("mini", &other), false);
    let (again, again_state) = replay("mini", &other, &s, false, false);
    assert_eq!(normal.as_ref().unwrap(), again.as_ref().unwrap());
    assert_eq!(state, again_state);
    let (first, _) = run(machine("mini", &prog), false);
    assert_ne!(first.unwrap(), normal.unwrap(), "the other data must change the run, or the test shows nothing");
}

/// The text of the program is checked as it is fetched: a recording whose text hash of one tick is wrong is refused.
#[test]
fn a_wrong_text_hash_is_refused_by_the_replay() {
    let (n, _, s) = recordable(false).into_iter().find(|(_, _, s)| s.ticks() > 40).unwrap();
    let mut m = s.clone();
    let mid = s.ticks() / 2;
    m.text[mid] ^= 1;
    match replay(profile_of(&n), &data(&n), &m, false, false).0 {
        Err(RunError::Internal(e)) if e.contains("instruction text") => {}
        other => panic!("{n}: {:?}", other.map(|e| e.len())),
    }
}

#[test]
fn the_key_refuses_another_program_profile_or_starting_state() {
    let (n, _, s) = recordable(false).into_iter().find(|(n, ..)| n.starts_with("mini")).unwrap();
    let prog = data(&n);
    let s = Arc::new(s);
    // the same program
    let mut m = machine("mini", &prog);
    assert_eq!(m.replay_schedule_with(s.clone(), false), Ok(()));
    // another IQ image
    let other = format!("{prog}Q 1 0\n");
    let mut m = machine("mini", &other);
    assert_eq!(m.replay_schedule_with(s.clone(), false), Err(Refused::Key));
    // another profile
    let mut m = machine("ref", &prog);
    assert_eq!(m.replay_schedule_with(s.clone(), false), Err(Refused::Key));
    // a machine that has run
    let mut m = machine("mini", &prog);
    m.run(MAX_CYCLES, &mut Log::default()).unwrap();
    assert!(matches!(m.replay_schedule_with(s.clone(), false), Err(Refused::NotApplicable(_))));
    // a schedule of a run that was not under way is not a key either
    let mut bad = (*s).clone();
    bad.key[0] ^= 1;
    let mut m = machine("mini", &prog);
    assert_eq!(m.replay_schedule_with(Arc::new(bad), false), Err(Refused::Key));
}

#[test]
fn a_run_that_used_a_link_is_not_recordable() {
    let mut found = 0;
    for n in names() {
        let mut m = machine(profile_of(&n), &data(&n));
        m.record_schedule().unwrap();
        let ok = m.run(MAX_CYCLES, &mut Log::default()).is_ok();
        let uses = m.replay_state().used_links;
        let s = m.take_schedule();
        if uses {
            assert!(s.is_none(), "{n}: a run that used a link produced a schedule");
            found += 1;
        } else if ok {
            assert!(s.is_some(), "{n}");
        }
    }
    assert!(found > 0, "no program of the corpus uses a link");
}

#[test]
fn a_replay_that_does_not_fit_the_observer_takes_the_normal_driver() {
    // recorded without wait tracking, run with it: the schedule has no wait entries, so the machine must not replay
    let (n, normal, s) = recordable(false).into_iter().next().unwrap();
    let mut m = machine(profile_of(&n), &data(&n));
    m.replay_schedule_with(Arc::new(s), false).unwrap();
    let mut log = Log { waits: true, ev: Vec::new() };
    m.run(MAX_CYCLES, &mut log).unwrap();
    assert_eq!(m.replay_mode(), Mode::Off);
    let waits_only = run(machine(profile_of(&n), &data(&n)), true);
    assert_eq!(waits_only.0.unwrap(), log.ev);
    assert!(normal.0.is_ok());
}

#[test]
fn the_session_cache_records_the_first_run_and_replays_the_next() {
    let name = names()
        .into_iter()
        .find(|n| {
            let mut m = machine(profile_of(n), &data(n));
            m.record_schedule().unwrap();
            m.run(MAX_CYCLES, &mut Log::default()).is_ok() && m.take_schedule().is_some()
        })
        .expect("a recordable program");
    let prog = data(&name);
    let profile = profile_of(&name);
    let mut first = machine(profile, &prog);
    assert_eq!(first.replay_session_begin(), Began::Recording);
    let a = {
        let mut log = Log::default();
        first.run(MAX_CYCLES, &mut log).unwrap();
        first.replay_session_end();
        log.ev
    };
    let mut second = machine(profile, &prog);
    assert_eq!(second.replay_session_begin(), Began::Replaying);
    let mut log = Log::default();
    second.run(MAX_CYCLES, &mut log).unwrap();
    assert_eq!(a, log.ev, "{name}: the second run of a session differs from the first");
    // reset and run again on the same machine: the schedule is replayed from its start
    second.reset();
    let mut again = Log::default();
    second.run(MAX_CYCLES, &mut again).unwrap();
    assert_eq!(a, again.ev, "{name}: a reset machine differs");
}
