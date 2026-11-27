//! Schedule replay is opt-in (cargo feature `replay`, then a switch): a run without the switch never takes the replay
//! path. The recording hooks (`rec_on`, the operation and element logs, the fetched-text hash) stay untouched, the
//! chip stays in `Mode::Off`, nothing enters the process's schedule cache, and a build without the feature refuses
//! every request to record or replay. The programs are those of `tests/data/replay/`.
use ots_profile::Profile;
use std::path::PathBuf;
use tsp_iss::driver::replay::{session_cache_len, Mode};
use tsp_iss::links::NoLinks;
use tsp_iss::machine::Machine;
use tsp_iss::observe::NoObserver;

fn programs() -> Vec<(String, String)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/replay");
    let mut v: Vec<(String, String)> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n.ends_with(".txt"))
        .map(|n| {
            let text = std::fs::read_to_string(dir.join(&n)).unwrap();
            (n, text)
        })
        .collect();
    v.sort();
    assert!(v.len() >= 8, "the corpus is there");
    v
}

fn machine(name: &str, prog: &str) -> Machine {
    let profile = if name.starts_with("ref") { "ref" } else { "mini" };
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

#[test]
fn a_run_without_the_switch_never_touches_the_replay_path() {
    for (name, prog) in programs() {
        let mut m = machine(&name, &prog);
        assert_eq!(
            (m.replay_mode(), m.replay_state().rec_on, m.replay_state().compiled),
            (Mode::Off, false, false),
            "{name}: before"
        );
        let _ = m.run(6000, &mut NoObserver);
        assert_eq!(
            (m.replay_mode(), m.replay_state().rec_on, m.replay_state().compiled),
            (Mode::Off, false, false),
            "{name}: after"
        );
        let rec = &m.replay_state().rec;
        let logs =
            (rec.cycles.len(), rec.ops.len(), rec.execs.len(), rec.text.len(), rec.ops_off.len(), rec.execs_off.len());
        assert_eq!(logs, (0, 0, 0, 0, 0, 0), "{name}: the recording hooks ran without the switch");
    }
    assert_eq!(session_cache_len(), 0, "nothing was stored for a later run");
}

#[test]
fn a_platform_without_the_switch_stays_off() {
    if std::env::var_os("TSP_ISS_REPLAY_SESSION").is_some_and(|v| !v.is_empty() && v != "0") {
        return; // the environment of this process switched it on
    }
    let mut p = tsp_iss::platform::Platform::new(&Profile::load("mini").unwrap(), 0).unwrap();
    for k in 0..3u8 {
        p.reset().unwrap();
        p.select_iq(4).unwrap();
        p.write_iq_window(&[0x14, 0x1000_0001, 0]).unwrap();
        p.write_sram_word(4, 1, &[9 + k; 16]).unwrap();
        p.start().unwrap();
        p.run_to_done(1000).unwrap();
        assert_eq!(p.replay_mode(), Mode::Off, "run {k}");
    }
    assert_eq!(session_cache_len(), 0);
}

/// A build without the feature has no replay path in its tick loop, so it refuses to record or replay, with sessions
/// asked for or not.
#[cfg(not(feature = "replay"))]
#[test]
fn a_build_without_the_feature_refuses_to_record_or_replay() {
    use tsp_iss::driver::replay::Refused;
    assert!(!std::hint::black_box(tsp_iss::driver::replay::ENABLED));
    let (name, prog) = programs().remove(0);
    let mut m = machine(&name, &prog);
    assert!(matches!(m.record_schedule(), Err(Refused::NotApplicable(_))));
    assert!(m.take_schedule().is_none());
    let mut p = tsp_iss::platform::Platform::new(&Profile::load("mini").unwrap(), 0).unwrap();
    p.set_replay_sessions(true);
    for _ in 0..3 {
        p.reset().unwrap();
        p.select_iq(4).unwrap();
        p.write_iq_window(&[0x14, 0x1000_0001, 0]).unwrap();
        p.start().unwrap();
        p.run_to_done(1000).unwrap();
        assert_eq!(p.replay_mode(), Mode::Off);
    }
    assert_eq!(session_cache_len(), 0);
}
