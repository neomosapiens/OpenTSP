//! Tests of `driver`, moved from the module's in-source test block (they use the crate's public API).
//! Ports of the machine-level unit tests of the Sail model (`unit_tests/test_frontend.sail`,
//! `test_c2c.sail`): `test_frontend_dispatch`, `test_frontend_events`, `test_c2c_deskew_occupancy`,
//! `test_harness_next_event`; and the ISS's own: exit 3 / idle skip, `reset`, the control channel.
use ots_isa::{encode, Code, Insn};
use ots_profile::Profile;
use tsp_iss::driver::agt::AgtState;
use tsp_iss::driver::repeat::RepeatState;
use tsp_iss::observe::{NoObserver, Observer};
use tsp_iss::{Machine, RunError};

fn machine(name: &str) -> Machine {
    Machine::new(&Profile::load(name).unwrap(), 0, Box::new(tsp_iss::links::NoLinks)).unwrap()
}

fn units(i: &Insn) -> Vec<u32> {
    let w = encode(i);
    if i.entry().width == 64 {
        vec![(w >> 32) as u32, w as u32]
    } else {
        vec![w as u32]
    }
}

fn load(m: &mut Machine, q: u16, prog: &[Insn]) {
    let u: Vec<u32> = prog.iter().flat_map(units).collect();
    m.load_image(q, &u).unwrap();
}

fn nop(n: u32) -> Insn {
    Insn::NOP { n }
}
fn setvl(n: u32) -> Insn {
    Insn::SetVL { n }
}

#[derive(Default)]
struct Diags(Vec<(u64, u16, Code)>);
impl Observer for Diags {
    fn diag(&mut self, cycle: u64, queue: u16, code: Code) {
        self.0.push((cycle, queue, code));
    }
}

// model: unit_tests/test_frontend.sail test_frontend_dispatch
#[test]
fn dispatch_in_order_from_the_text_with_the_nop_prefix() {
    let mut m = machine("ref");
    let sl = m.geom().superlanes as u32;
    load(&mut m, 100, &[nop(3), setvl(5), setvl(6)]);
    for c in 0..3 {
        m.tick(c, &mut NoObserver);
    }
    assert_eq!(m.iq().len[100], 3);
    m.tick(3, &mut NoObserver);
    assert_eq!((m.masks().queue_vl[100] as u32, m.iq().len[100], m.iq().last[100]), (5.min(sl), 1, 3));
    m.tick(4, &mut NoObserver);
    assert_eq!((m.masks().queue_vl[100] as u32, m.iq().len[100], m.iq().last[100]), (6.min(sl), 0, 4));
    assert!(m.quiescent());
}

// model: unit_tests/test_frontend.sail test_frontend_dispatch (second half)
#[test]
fn a_synced_queue_is_released_barrier_latency_cycles_after_the_notify() {
    let mut m = machine("ref");
    let sl = m.geom().superlanes as u32;
    load(&mut m, 100, &[Insn::Sync, setvl(4)]);
    load(&mut m, 101, &[nop(5), Insn::Notify]);
    m.tick(0, &mut NoObserver);
    assert!(m.driver().barrier.queue_parked(100, 1) && m.iq().len[100] == 1);
    m.tick(1, &mut NoObserver);
    assert_eq!(m.iq().len[100], 1);
    assert_eq!(m.next_event(1), Some(5), "the Notify at 5; the Sync's head parcel is held");
    m.tick(5, &mut NoObserver);
    assert_eq!((m.iq().len[101], m.iq().len[100]), (0, 1));
    let rel = 5 + m.geom().barrier_release_latency;
    assert_eq!(m.next_event(5), Some(u128::from(rel)));
    m.tick(rel, &mut NoObserver);
    assert_eq!((m.masks().queue_vl[100] as u32, m.iq().len[100]), (4.min(sl), 0));
    assert!(m.quiescent());
}

// model: unit_tests/test_frontend.sail test_frontend_events
#[test]
fn a_parcel_is_an_event_only_when_it_can_dispatch() {
    let mut m = machine("ref");
    assert!(m.quiescent() && m.next_event(0).is_none());
    load(&mut m, 100, &[nop(40), setvl(2)]);
    assert!(!m.quiescent());
    assert_eq!(m.next_event(0), Some(40));
    // an unreleased Sync holds what is behind it: nothing to wait for
    let mut m = machine("ref");
    load(&mut m, 100, &[Insn::Sync, setvl(2)]);
    m.tick(0, &mut NoObserver);
    assert!(m.quiescent() && m.next_event(0).is_none());
    // fetched text that is on its way is an event
    let mut m = machine("ref");
    let v = vec![0u8; m.geom().vlanes as usize];
    m.edit_state(|state| state.iq.fetch(100, 20, &v, &v)).unwrap();
    assert!(!m.quiescent());
    assert_eq!(m.next_event(20), Some(20 + u128::from(m.geom().fetch_latency)));
}

// model: unit_tests/test_frontend.sail test_frontend_whole_queue_open_config (a
// whole-queue instruction due in the cycle the configuration of a path ends waits for the loop and
// its tail; it must not dispatch next to a reference of its own cycle)
#[test]
fn a_whole_queue_instruction_waits_for_a_path_with_an_open_configuration() {
    for cd in [1u64, 3] {
        // Read at 0, Countdown at 1, SetVL due at 2: the references are at 0 and 1 + cd (two default
        // iterations); the path is held until last + 1 + cd = 2 + 2 cd (design 7.6).
        let mut m = machine("ref");
        load(&mut m, 0, &[Insn::Read { addr: 1, dst: 1, thread: 0 }, Insn::Countdown { d: cd as u32 }, setvl(4)]);
        m.tick(0, &mut NoObserver);
        m.tick(1, &mut NoObserver);
        assert_eq!(m.driver().agt_get(0, 0).state, AgtState::Config);
        assert_eq!(m.iq().len[0], 1);
        m.tick(2, &mut NoObserver);
        assert_eq!(m.iq().len[0], 1, "the SetVL waits, with no fault");
        let free = 2 + 2 * cd;
        assert_eq!(m.next_event(2), Some(4), "the reference of cycle 4 (cd 3), or the end of the tail (cd 1)");
        for t in 3..free {
            m.tick(t, &mut NoObserver);
            assert_eq!(m.iq().len[0], 1);
        }
        m.tick(free, &mut NoObserver);
        assert_eq!((m.iq().len[0], m.iq().last[0]), (0, i128::from(free)));
    }
}

// model: unit_tests/test_c2c.sail test_c2c_deskew_occupancy
#[test]
fn deskew_occupancy_and_issuing_queues() {
    use ots_isa::placement;
    let m = machine("ref");
    let p = m.geom().hac_period;
    assert_eq!(m.occupancy(&Insn::Deskew, 0), p);
    assert_eq!(m.occupancy(&Insn::Deskew, 1), p - 1);
    assert_eq!(m.occupancy(&Insn::Deskew, p - 1), 1);
    assert_eq!(m.occupancy(&Insn::Deskew, 3 * p + 5), p - 5);
    assert_eq!(m.occupancy(&Insn::RuntimeDeskew { t: 0x10 }, 7), 16);
    assert_eq!(m.occupancy(&Insn::RuntimeDeskew { t: 0 }, 7), 1);
    assert_eq!(m.occupancy(&nop(9), 7), 1);
    // issued on any data queue that has a station (the MEM, VXM, MXM, SXM, C2C queues)
    for q in [0usize, 87, 88, 100, 120, 130, 143] {
        assert!(m.issued_by(&Insn::Deskew, q), "queue {q}");
        assert!(m.issued_by(&Insn::RuntimeDeskew { t: 1 }, q), "queue {q}");
        assert!(placement(Insn::Deskew.id(), q as u16).is_none());
    }
    // a MEM Read is not issued by a VXM queue
    {
        assert!(!m.issued_by(&Insn::Read { addr: 1, dst: 0, thread: 0 }, 100));
    }
}

// model: unit_tests/test_c2c.sail test_harness_next_event
#[test]
fn next_event_candidates() {
    let mut m = machine("ref");
    assert!(m.quiescent());
    assert_eq!(m.next_event(5), None);
    // an AGT reference due at 50 on queue 3; a Repeat iteration due at 45 on queue 4
    let mut a = m.driver().agt_get(3, 0);
    a.state = AgtState::Run;
    a.next = 50;
    m.edit_state(|state| state.driver.agt_set(3, 0, a)).unwrap();
    assert_eq!(m.next_event(10), Some(50));
    m.edit_state(|state| state.driver.queue_repeat[4] = Some(RepeatState { instr: nop(1), left: 2, gap: 5, next: 45 }))
        .unwrap();
    m.edit_state(|state| state.driver.touch(4)).unwrap(); // a write past the driver's own methods tells the event index
    assert_eq!(m.next_event(10), Some(45));
    // a due-in-the-past event (parked earlier) is due on the next cycle
    assert_eq!(m.next_event(60), Some(61));
    // queue 4 parked from 20, the Notify at 35: its iteration waits for the release at 35 + latency
    let rl = m.geom().barrier_release_latency;
    m.edit_state(|state| state.driver.barrier.sync(4, 20)).unwrap();
    m.edit_state(|state| state.driver.barrier.notify(35)).unwrap();
    assert_eq!(m.next_event(40), Some(50));
    a.state = AgtState::Idle;
    m.edit_state(|state| state.driver.agt_set(3, 0, a)).unwrap();
    assert_eq!(m.next_event(40), Some(35 + u128::from(rl)));
    // a control-channel Sync with no Notify holds every queue: nothing to wait for
    let mut m = machine("ref");
    m.edit_state(|state| state.driver.queue_repeat[4] = Some(RepeatState { instr: nop(1), left: 2, gap: 5, next: 45 }))
        .unwrap();
    m.edit_state(|state| state.driver.barrier.sync(144, 20)).unwrap();
    assert_eq!(m.next_event(30), None);
    assert!(m.quiescent());
    // an open AGT configuration ends at the first cycle without a configuration record
    let mut m = machine("ref");
    let mut a = m.driver().agt_get(7, 0);
    a.state = AgtState::Config;
    m.edit_state(|state| state.driver.agt_set(7, 0, a)).unwrap();
    assert_eq!(m.next_event(12), Some(13));
}

#[test]
fn idle_skip_reaches_a_late_nop_without_ticking_every_cycle() {
    let mut m = machine("ref");
    load(&mut m, 100, &[nop(60_000), setvl(2)]);
    let s = m.run(1_000_000, &mut NoObserver).unwrap();
    assert!(s.ticks < 20, "{s:?}");
    assert_eq!(s.last_cycle, Some(60_000));
    assert!(m.quiescent());
}

#[test]
fn a_next_event_beyond_the_limit_is_exit_three_information() {
    let mut m = machine("ref");
    load(&mut m, 100, &[nop(500), setvl(2)]);
    assert_eq!(m.run(100, &mut NoObserver), Err(RunError::MaxCycles { cycle: 500, limit: 100 }));
}

#[test]
fn a_sync_with_no_notify_ends_the_run_quiescent() {
    let mut m = machine("mini");
    load(&mut m, 0xffff, &[Insn::Sync]);
    load(&mut m, 100, &[setvl(2)]);
    let s = m.run(1_000, &mut NoObserver).unwrap();
    assert_eq!(s.last_cycle, Some(0));
    assert_eq!(m.iq().len[100], 1, "the data queue stays parked");
}

#[test]
fn reset_restores_the_loaded_images() {
    let mut m = machine("ref");
    load(&mut m, 100, &[setvl(3)]);
    m.run(10, &mut NoObserver).unwrap();
    assert_eq!(m.iq().len[100], 0);
    m.reset();
    assert_eq!(m.iq().len[100], 1);
    assert_eq!(m.next_tick(), Some(0));
    assert_eq!(m.masks().queue_vl[100] as u32, m.geom().superlanes as u32);
}

#[test]
fn an_instruction_on_a_queue_that_does_not_issue_it_is_que_002() {
    let mut m = machine("ref");
    load(&mut m, 100, &[Insn::Read { addr: 1, dst: 0, thread: 0 }]);
    let mut d = Diags::default();
    m.run(100, &mut d).unwrap();
    assert_eq!(d.0, vec![(0, 100, Code::Que002)]);
}

// model: unit_tests/test_streams.sail test_reset_state (microarchitecture.md §11): the
// architectural state after reset(). The MEM segment, the MXM planes and the SXM maps are
// not covered here.
#[test]
fn reset_state() {
    use tsp_iss::layout::RowId;
    use tsp_iss::srf::SrfLookup;
    let mut m = machine("ref");
    let sl = m.geom().superlanes as u8;
    // dirty the state this test covers, then reset
    m.edit_state(|state| state.srf.write(RowId::E, 0, 21, 5, &[1; 320])).unwrap();
    m.edit_state(|state| state.masks.queue_vl[3] = 2).unwrap();
    m.edit_state(|state| state.masks.mem_thread_split[4] = 1).unwrap();
    m.edit_state(|state| state.driver.queue_busy_until[7] = 99).unwrap();
    m.edit_state(|state| state.driver.barrier.sync(5, 3)).unwrap();
    m.edit_state(|state| state.driver.barrier.notify(4)).unwrap();
    m.edit_state(|state| {
        state.driver.set_pending_list(
            9,
            vec![tsp_iss::operands::Commit {
                r: RowId::W,
                id: 1,
                g: 2,
                value: [0; tsp_iss::geom::MAX_LANES],
                q: 3,
                b: 0,
                ov: 0,
            }],
        )
    })
    .unwrap();
    m.reset();
    // stream register file empty: any read finds no value
    assert_eq!(m.srf().read(RowId::E, 0, 21, 5).0, SrfLookup::None);
    for q in 0..144 {
        assert_eq!(m.masks().queue_vl[q], sl, "SetVL: every superlane (queue {q})");
        assert_eq!(m.driver().queue_busy_until[q], 0, "no queue busy");
        assert_eq!(m.driver().agts[q].state, AgtState::Idle, "no AGT configuration or loop");
        assert_eq!(m.driver().agts_w[q].state, AgtState::Idle);
        assert!(m.driver().queue_last[q].is_none(), "nothing to Repeat");
        assert!(m.driver().queue_repeat[q].is_none());
    }
    for s in 0..88 {
        assert_eq!(m.masks().mem_thread_split[s], sl, "every tile thread 0 (slice {s})");
    }
    for k in 0..145 {
        assert!(
            m.driver().barrier.parked_since[k].is_none() && m.driver().barrier.release_at[k].is_none(),
            "no barrier"
        );
    }
    assert!(m.driver().barrier.last_notify.is_none());
    assert!(m.driver().pending.iter().all(Vec::is_empty));
    assert!(m.quiescent());
}

// model: postlude/tick.sail free_at, parcel_held, classify_items, config_for, model_idle: the corners of the
// control channel and of a parcel with no items
#[test]
fn the_control_channel_is_always_free_and_an_empty_parcel_is_never_held() {
    use tsp_iss::driver::classify::{classify_items, config_for};
    let mut m = machine("ref");
    assert!(m.free_at(144, 0) && m.free_at(144, 10_000), "the control channel is always free");
    assert!(m.free_at(5, 0));
    let none = classify_items(5, &[]);
    assert_eq!(none.n, 0);
    assert!(!m.parcel_held(5, 0, &none, &[]));
    assert!(!config_for(&none));
    let one = classify_items(5, &[Ok(setvl(2))]);
    assert!(one.ok && !config_for(&one), "a SetVL is no configuration record");
    let cfg = classify_items(0, &[Ok(Insn::Countdown { d: 1 })]);
    assert!(cfg.ok && config_for(&cfg), "a Countdown on a MEM queue is a configuration record");
    // a control-channel parcel that has not been dispatched keeps the machine from being idle
    assert!(m.quiescent());
    load(&mut m, 0xffff, &[nop(2), Insn::Notify]);
    assert!(!m.quiescent(), "a head parcel waits on the control channel");
    assert!(m.control_event(0).is_some());
    m.run(100, &mut NoObserver).unwrap();
    assert!(m.quiescent() && m.control_event(10).is_none());
    // an instruction held on the control channel's parked queue is reported against the data queue
    let mut m = machine("ref");
    load(&mut m, 0xffff, &[Insn::Sync]);
    load(&mut m, 3, &[setvl(2)]);
    m.run(100, &mut NoObserver).unwrap();
    assert!(m.quiescent(), "a Sync no Notify releases parks every data queue for good, and is not work");
}
