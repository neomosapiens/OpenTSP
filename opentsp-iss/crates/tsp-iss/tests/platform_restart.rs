//! `Platform::restart` (the optional platform restart in `platform.md` §1.1): the state after `reset` with the SRAM contents kept.
//! It must be observationally the same as a replay: `reset`, then write every word the SRAM held back through the
//! aperture (design `opentsp-vllm/README.md` §4.5).
mod common;
use common::*;
use ots_profile::Profile;
use serde_json::Value;
use tsp_iss::platform::*;

fn platform() -> Platform {
    Platform::new(&Profile::load("mini").unwrap(), 0).unwrap()
}

fn units_of(s: &str) -> Vec<u32> {
    s.split_whitespace().map(|u| u32::from_str_radix(u, 16).unwrap()).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}

fn key_pair(k: &str) -> (u8, u32) {
    let (a, b) = k.split_once(':').unwrap();
    (a.parse().unwrap(), b.parse().unwrap())
}

fn vector(name: &str) -> Value {
    let path = spec().join("conformance/model/mini.json");
    let doc: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    doc["vectors"].as_array().unwrap().iter().find(|v| v["name"] == name).unwrap().clone()
}

/// The programs: an MXM one (weights installed and multiplied), one with SXM permute maps, and one whose run raises
/// data-dependent diagnostics (two OTS-E-OPR-001 of a MEM queue: the error FIFO and `ERR_FIRST` fill).
const MXM: &str = "directed-013";
const PERMUTE: &str = "directed-232";
const DIAGNOSTIC: &str = "directed-028";

/// Load a vector's program and its data words (on top of whatever the SRAM holds).
fn load(p: &mut Platform, v: &Value) {
    for (q, units) in v["text"].as_object().unwrap() {
        p.select_iq(q.parse().unwrap()).unwrap();
        p.write_iq_window(&units_of(units.as_str().unwrap())).unwrap();
    }
    for (k, w) in v["sram"].as_object().unwrap() {
        let (s, a) = key_pair(k);
        p.write_sram_word(s, a, &unhex(w.as_str().unwrap())).unwrap();
    }
}

/// Everything the host can see of a finished run.
#[derive(Debug, PartialEq, Eq)]
struct Seen {
    cycle: u64,
    status: u32,
    diagnostics: Vec<(u64, u16, String)>,
    err_first: Option<(u64, u16, String)>,
    fifo: Vec<(u64, u16, String)>,
    dump: Vec<Vec<u8>>,
    sram: [u8; 32],
}

fn rec(r: &Record) -> (u64, u16, String) {
    (r.cycle, r.queue, r.code.as_str().to_string())
}

fn run(p: &mut Platform, v: &Value) -> Seen {
    load(p, v);
    p.start().unwrap();
    p.run_to_done(100_000).unwrap();
    let dump = v["dump"].as_array().unwrap().iter().map(|k| key_pair(k.as_str().unwrap())).collect::<Vec<_>>();
    Seen {
        cycle: p.cycle(),
        status: p.status().bits(),
        diagnostics: p.diagnostics().iter().map(rec).collect(),
        err_first: p.err_first().as_ref().map(rec),
        fifo: std::iter::from_fn(|| p.err_pop()).map(|r| rec(&r)).collect(),
        dump: dump.iter().map(|&(s, a)| p.read_sram_word(s, a).unwrap()).collect(),
        sram: p.sram_digest(),
    }
}

/// The replay of `design` §4.5: reset, write the words back through the aperture.
fn replay(p: &mut Platform) {
    let words = p.sram_words();
    p.reset().unwrap();
    for (slice, addr, bytes) in &words {
        p.write_sram_word(*slice, *addr, bytes).unwrap();
    }
}

#[test]
fn restart_keeps_the_sram_and_clears_the_rest() {
    let mut p = platform();
    let empty = p.sram_digest();
    p.set_c2c_peer(0, Some(7)).unwrap();
    p.set_irq_enable(IrqEnable { quiescent: true, error: true });
    // the SRAM holds a run's results, then a second run leaves error state pending
    run(&mut p, &vector(MXM));
    p.restart().unwrap();
    // a refused IQ window sets `STATUS.load_error`, and a run with twenty undecodable units overflows the FIFO of 16
    let depth = (p.icu_iq_depth() / 4) as usize;
    p.select_iq(3).unwrap();
    p.write_iq_window(&vec![1; depth]).unwrap();
    assert!(p.write_iq_window(&[1]).is_err());
    assert!(p.status().load_error);
    p.select_iq(5).unwrap();
    let mut undecodable = vec![10];
    undecodable.extend(vec![0x7000_0000; 20]);
    p.write_iq_window(&undecodable).unwrap();
    let diag = vector(DIAGNOSTIC);
    load(&mut p, &diag);
    p.start().unwrap();
    p.run_to_done(100_000).unwrap();
    assert!(p.err_count() > 0 && p.err_first().is_some() && p.status().error && p.irq_pending());
    assert!(p.status().load_error && p.status().fifo_overflow, "both sticky bits are raised before the restart");
    assert!(!p.diagnostics().is_empty() && p.cycle() > 0);
    let before = (p.sram_digest(), p.sram_words());
    assert_ne!(before.0, empty, "the runs wrote something");

    p.restart().unwrap();
    assert_eq!((p.sram_digest(), p.sram_words()), before, "the SRAM is kept");
    assert_eq!(p.phase(), Phase::Reset);
    assert_eq!(p.cycle(), 0);
    assert_eq!(p.status(), Status::default());
    assert!(!p.status().load_error && !p.status().fifo_overflow);
    assert!(p.diagnostics().is_empty());
    assert_eq!((p.err_count(), p.err_first(), p.err_pop()), (0, None, None));
    assert_eq!(p.selected_iq(), 0);
    assert_eq!(p.sram_addr(), SramAddr::default());
    assert!(!p.irq_pending());
    // the configuration survives a restart as it survives a reset
    assert_eq!(p.c2c_peer(0), Ok(Some(7)));
    assert_eq!(p.irq_enable(), IrqEnable { quiescent: true, error: true });
    // restarting twice, and restarting a chip that is running, are the same as one restart
    p.restart().unwrap();
    assert_eq!(p.sram_digest(), before.0);
    load(&mut p, &diag);
    p.start().unwrap();
    p.restart().unwrap();
    assert_eq!((p.phase(), p.status()), (Phase::Reset, Status::default()));
    assert_eq!(p.sram_digest(), before.0);
}

#[test]
fn the_programs_leave_something_to_keep() {
    // the equivalence tests below are not vacuous: MXM and permute runs write SRAM, the diagnostic run fills the FIFO
    for name in [MXM, PERMUTE] {
        let mut p = platform();
        let seen = run(&mut p, &vector(name));
        assert!(seen.diagnostics.is_empty(), "{name}");
        assert!(!p.sram_words().is_empty(), "{name}");
    }
    let seen = run(&mut platform(), &vector(DIAGNOSTIC));
    assert!(!seen.diagnostics.is_empty() && !seen.fifo.is_empty() && seen.err_first.is_some());
}

#[test]
fn reset_still_clears_the_sram() {
    let mut p = platform();
    let empty = p.sram_digest();
    run(&mut p, &vector(MXM));
    assert_ne!(p.sram_digest(), empty);
    p.reset().unwrap();
    assert_eq!(p.sram_digest(), empty);
    assert!(p.sram_words().is_empty());
}

#[test]
fn a_restarted_chip_is_a_fresh_chip_with_the_old_sram() {
    // the same program on a restarted chip and on a fresh chip holding the same SRAM words: the same run, cycle for cycle
    for first in [MXM, PERMUTE, DIAGNOSTIC] {
        for second in [MXM, PERMUTE, DIAGNOSTIC] {
            let mut warm = platform();
            run(&mut warm, &vector(first));
            let mut fresh = platform();
            for (s, a, w) in warm.sram_words() {
                fresh.write_sram_word(s, a, &w).unwrap();
            }
            warm.restart().unwrap();
            let b = vector(second);
            assert_eq!(run(&mut warm, &b), run(&mut fresh, &b), "{first} then {second}");
        }
    }
}

#[test]
fn restart_equals_replay_on_programs_of_different_kinds() {
    let kinds = [MXM, PERMUTE, DIAGNOSTIC];
    for first in kinds {
        for second in kinds {
            let (a, b) = (vector(first), vector(second));
            let (mut warm, mut cold) = (platform(), platform());
            assert_eq!(run(&mut warm, &a), run(&mut cold, &a));
            warm.restart().unwrap();
            replay(&mut cold);
            assert_eq!(run(&mut warm, &b), run(&mut cold, &b), "{first} then {second}");
        }
    }
}

#[test]
fn restart_equals_replay_over_a_chain_of_runs() {
    let chain = [MXM, PERMUTE, DIAGNOSTIC, MXM, DIAGNOSTIC, PERMUTE, PERMUTE, MXM];
    let (mut warm, mut cold) = (platform(), platform());
    for (i, name) in chain.iter().enumerate() {
        let v = vector(name);
        if i > 0 {
            warm.restart().unwrap();
            replay(&mut cold);
        }
        assert_eq!(run(&mut warm, &v), run(&mut cold, &v), "step {i}: {name}");
    }
}

/// `program` with the text of `queue` cut to its first `keep` units: the setup that queue did is missing.
fn without(program: &Value, queue: &str, keep: usize) -> Value {
    let mut b = program.clone();
    let units = b["text"][queue].as_str().unwrap().split_whitespace().take(keep).collect::<Vec<_>>().join(" ");
    b["text"][queue] = Value::String(units);
    b
}

#[test]
fn a_second_program_without_the_setup_sees_no_state_of_the_first() {
    // The first program installs MXM weights or SXM maps; the second lacks the install (or compute, or map load).
    // A restart that kept that state would run the second program differently from a replay.
    let mxm = vector(MXM);
    let permute = vector(PERMUTE);
    let mut probes: Vec<(String, Value, Value)> = vec![];
    for q in mxm["text"].as_object().unwrap().keys() {
        for keep in [0, 1, 2] {
            probes.push((format!("mxm queue {q} keep {keep}"), mxm.clone(), without(&mxm, q, keep)));
        }
    }
    for q in permute["text"].as_object().unwrap().keys() {
        for keep in [0, 1, 4] {
            probes.push((format!("sxm queue {q} keep {keep}"), permute.clone(), without(&permute, q, keep)));
        }
    }
    // and the other way round: one kind of state left behind, a different kind of program next
    probes.push(("mxm then permute".into(), mxm.clone(), permute.clone()));
    probes.push(("permute then mxm".into(), permute.clone(), mxm.clone()));
    assert!(probes.len() > 10);
    for (what, a, b) in &probes {
        let (mut warm, mut cold) = (platform(), platform());
        assert_eq!(run(&mut warm, a), run(&mut cold, a), "{what}");
        warm.restart().unwrap();
        replay(&mut cold);
        assert_eq!(run(&mut warm, b), run(&mut cold, b), "{what}");
    }
}

fn vectors_of(profile: &str) -> Vec<Value> {
    let doc: Value =
        serde_json::from_slice(&std::fs::read(spec().join(format!("conformance/model/{profile}.json"))).unwrap())
            .unwrap();
    doc["vectors"].as_array().unwrap().clone()
}

/// A chip in the state of the host's configuration: C2C peers set, as a serving host sets them once. The peers are
/// those the golden vectors' Send and Transmit instructions name, so that a link accepts a vector.
fn configured(profile: &str) -> Platform {
    let mut p = Platform::new(&Profile::load(profile).unwrap(), 0).unwrap();
    for (link, peer) in [(0, 7), (6, 0), (8, 47), (11, 0), (15, 0)] {
        p.set_c2c_peer(link, Some(peer)).unwrap();
    }
    p
}

fn find(profile: &str, name: &str) -> Value {
    vectors_of(profile).into_iter().find(|v| v["name"] == name).unwrap()
}

#[test]
fn restart_returns_every_register_but_the_sram_to_the_state_of_a_new_chip() {
    // `state_digest` covers all registers but the SRAM (IQs, SRF, MXM, SXM, masks, segments, link acceptance, driver,
    // clock, diagnostics, IQ select, SRAM address, load error), so any state `restart` forgot to clear shows here,
    // for every program of the golden vectors of both profiles, whichever unit it leaves state in. (`mini` has one
    // superlane, so the SetVL and Config masks can only change on `ref`.)
    for profile in ["mini", "ref"] {
        let fresh = configured(profile).state_digest();
        let (mut dirty, mut kept, mut running) = (0, 0, 0);
        for v in vectors_of(profile) {
            let mut p = configured(profile);
            let name = v["name"].as_str().unwrap();
            // no panic is swallowed: every vector loads and starts, and a diagnostic is a result, not an error. A few
            // programs wait longer than the cycle limit (a long countdown): the chip is still running, and the restart
            // must clear that too.
            load(&mut p, &v);
            p.start().unwrap();
            match p.run_to_done(100_000) {
                Ok(_) => {}
                Err(PlatformError::NotDone { .. }) => running += 1,
                Err(e) => panic!("{profile} {name}: the run failed: {e:?}"),
            }
            if p.state_digest() != fresh {
                dirty += 1;
            }
            let words = p.sram_words();
            p.restart().unwrap();
            assert_eq!(p.state_digest(), fresh, "{profile} {name}: the state after a restart is a new chip's");
            assert_eq!(p.sram_words(), words, "{profile} {name}: the SRAM is kept");
            kept += usize::from(!words.is_empty());
            // a reset also returns to the state of a new chip, and empties the SRAM
            p.reset().unwrap();
            assert_eq!(p.state_digest(), fresh, "{profile} {name}: reset");
        }
        assert!(running < 10, "{profile}: only a few vectors outlast the limit ({running})");
        assert!(dirty > 400, "{profile}: the runs leave state behind ({dirty} of the vectors)");
        assert!(kept > 100, "{profile}: and write SRAM words that survive ({kept})");
    }
}

#[test]
fn a_load_error_and_a_selected_queue_are_state_that_a_restart_clears() {
    // the golden vectors never raise `STATUS.load_error`: a refused IQ window does, and the digest sees it
    let fresh = configured("mini").state_digest();
    let mut p = configured("mini");
    let depth = (p.icu_iq_depth() / 4) as usize;
    p.select_iq(9).unwrap();
    p.write_iq_window(&vec![1; depth]).unwrap();
    assert!(p.write_iq_window(&[1]).is_err());
    assert!(p.status().load_error && p.selected_iq() == 9);
    assert_ne!(p.state_digest(), fresh);
    p.restart().unwrap();
    assert!(!p.status().load_error && p.selected_iq() == 0);
    assert_eq!(p.state_digest(), fresh);
    // and the refusal alone, with the queue selected again and nothing loaded
    p.select_iq(9).unwrap();
    assert_ne!(p.state_digest(), fresh, "the selected queue is state");
    p.reset().unwrap();
    assert_eq!(p.state_digest(), fresh);
}

#[test]
fn every_kind_of_state_is_left_behind_by_some_program() {
    // the test above is only as strong as the programs: each unit the digest covers is dirtied by one of these runs,
    // and the digest sees it (it differs from a new chip's after the run, and equals it after the restart)
    for (what, profile, name) in [
        ("MXM weights", "mini", MXM),
        ("SXM maps", "mini", PERMUTE),
        ("masks: SetVL", "ref", "directed-050"),
        ("masks: Config", "ref", "directed-051"),
        ("segments: SetSegment", "mini", "directed-004"),
        ("link acceptance: Send", "mini", "generated-298"),
        ("link acceptance: Transmit", "mini", "generated-454"),
        ("diagnostics", "mini", DIAGNOSTIC),
    ] {
        let fresh = configured(profile).state_digest();
        let mut p = configured(profile);
        load(&mut p, &find(profile, name));
        p.start().unwrap();
        let _ = p.run_to_done(100_000);
        assert_ne!(p.state_digest(), fresh, "{what} ({name}) leaves state");
        p.restart().unwrap();
        assert_eq!(p.state_digest(), fresh, "{what} ({name})");
    }
}
