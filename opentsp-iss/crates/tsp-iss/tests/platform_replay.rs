//! `Platform::set_replay_sessions`: the first run of a program in a process runs the normal driver, the second records,
//! the later runs replay (the compiled replay), and a run is the same as without the sessions: cycles, diagnostics and
//! the SRAM. The bounded cache, the counters and the divergence fallback are in `replay_cache.rs`.
#![cfg(feature = "replay")]
use ots_profile::Profile;
use tsp_iss::driver::replay::Mode;
use tsp_iss::platform::Platform;

/// Reset, load the program (a MEM Read of slice 4 address 1 onto E:0, a MEM Write of slice 0 address 3, and a Read
/// of an unwritten word) and the data `fill`, run to done: (cycle, diagnostics, SRAM digest, replay mode).
fn serve(p: &mut Platform, fill: u8) -> (u64, String, [u8; 32], Mode) {
    p.reset().unwrap();
    p.select_iq(4).unwrap();
    p.write_iq_window(&[0x14, 0x1000_0001, 0]).unwrap();
    p.select_iq(0).unwrap();
    p.write_iq_window(&[0x1a, 0x1040_0003, 0]).unwrap();
    p.write_sram_word(4, 1, &[fill; 16]).unwrap();
    p.start().unwrap();
    let cycles = p.run_to_done(1000).unwrap();
    assert!(p.status().quiescent && !p.status().error);
    assert_eq!(p.read_sram_word(0, 3).unwrap()[..16], [fill; 16]);
    (cycles, format!("{:?}", p.diagnostics()), p.sram_digest(), p.replay_mode())
}

/// One test: the schedule cache is the process's, so two tests of one program would see each other's recordings.
#[test]
fn later_runs_of_a_program_replay_and_equal_the_run_without_sessions() {
    use tsp_iss::driver::replay::{session_cache_clear, session_cache_len};
    session_cache_clear();
    let profile = Profile::load("mini").unwrap();
    let mut plain = Platform::new(&profile, 0).unwrap();
    plain.set_replay_sessions(false);
    let mut sess = Platform::new(&profile, 0).unwrap();
    sess.set_replay_sessions(true);
    let mut modes = Vec::new();
    for k in 0..4u8 {
        let fill = 9 + k; // other data every time
        let a = serve(&mut plain, fill);
        let b = serve(&mut sess, fill);
        assert_eq!((a.0, &a.1, a.2), (b.0, &b.1, b.2), "run {k}: the run differs with sessions");
        assert_eq!(a.3, Mode::Off);
        modes.push((b.3, session_cache_len()));
    }
    // the first run is only seen (nothing is stored: a program that never recurs costs nothing), the second records (the
    // schedule is stored, the chip is back to the normal driver), the later ones replay
    assert_eq!(modes, [(Mode::Off, 0), (Mode::Off, 1), (Mode::Replay, 1), (Mode::Replay, 1)]);
    // another program (another address in the Read) has its own schedule: its first run is seen and does not replay the first
    for k in 0..2 {
        sess.reset().unwrap();
        sess.select_iq(4).unwrap();
        sess.write_iq_window(&[0x14, 0x1000_0002, 0]).unwrap();
        sess.select_iq(0).unwrap();
        sess.write_iq_window(&[0x1a, 0x1040_0003, 0]).unwrap();
        sess.start().unwrap();
        sess.run_to_done(1000).unwrap();
        assert_eq!((sess.replay_mode(), session_cache_len()), (Mode::Off, 1 + k), "run {k} of the other program");
    }
}
