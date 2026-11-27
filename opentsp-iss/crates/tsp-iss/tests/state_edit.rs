use ots_isa::{encode, Insn};
use ots_profile::Profile;
use std::panic::{catch_unwind, AssertUnwindSafe};
use tsp_iss::driver::agt::AgtState;
use tsp_iss::driver::execute::ExecElement;
use tsp_iss::driver::repeat::RepeatState;
use tsp_iss::layout::RowId;
use tsp_iss::operands::Commit;
use tsp_iss::{Machine, NoLinks, NoObserver};

fn machine() -> Machine {
    Machine::new(&Profile::load("mini").unwrap(), 0, Box::new(NoLinks)).unwrap()
}
fn repeated() -> RepeatState {
    RepeatState { instr: Insn::SetVL { n: 1 }, left: 2, gap: 5, next: 45 }
}

#[test]
fn editing_repeat_rebuilds_a_warmed_event_index() {
    let mut m = machine();
    assert_eq!(m.next_event(0), None);
    m.edit_state(|s| s.driver.queue_repeat[4] = Some(repeated())).unwrap();
    assert_eq!(m.next_event(0), Some(45));
    assert_eq!(m.next_event(0), m.model_next_event_scan(0));
    assert!(m.driver().index_on, "editing must preserve the event-index optimization");
}

#[test]
fn editing_agt_and_barrier_rebuilds_derived_state() {
    let mut m = machine();
    assert_eq!(m.next_event(0), None);
    m.edit_state(|s| s.driver.agts[4].state = AgtState::Config).unwrap();
    assert_eq!(m.next_event(0), Some(1));
    m.edit_state(|s| {
        s.driver.agts[4].state = AgtState::Idle;
        s.driver.queue_repeat[4] = Some(repeated());
        s.driver.barrier.parked_since[4] = Some(0);
    })
    .unwrap();
    assert!(m.driver().barrier.queue_parked(4, 1));
    assert_eq!(m.next_event(0), None);
    assert_eq!(m.next_event(0), m.model_next_event_scan(0));
}

#[test]
fn editing_pending_and_execute_records_rebuilds_both_occupancy_sets() {
    let mut m = machine();
    assert_eq!(m.next_event(0), None);
    m.edit_state(|s| {
        s.driver.pending[50].push(Commit { r: RowId::E, id: 0, g: 4, value: [0; 512], q: 4, b: 0, ov: 0 });
        s.driver.execute_ring[13].push(ExecElement {
            q: 100,
            instr: Insn::SetVL { n: 1 },
            c0: 13,
            k: 0,
            off: 0,
            soff: 0,
        });
    })
    .unwrap();
    assert_eq!(m.next_event(0), Some(13));
    assert_eq!(m.next_event(0), m.model_next_event_scan(0));
    m.edit_state(|s| s.driver.execute_ring[13].clear()).unwrap();
    assert_eq!(m.next_event(0), Some(50));
}

#[test]
fn editing_iq_reparses_text_and_editing_masks_remains_explicit() {
    let mut m = machine();
    assert_eq!(m.next_event(0), None);
    m.edit_state(|s| {
        assert!(s.iq.push(100, encode(&Insn::SetVL { n: 1 }) as u32));
        s.masks.queue_vl[100] = 0;
    })
    .unwrap();
    assert!(m.iq().head(100).complete);
    assert_eq!(m.masks().queue_vl[100], 0);
    assert_eq!(m.next_event(0), Some(1));
    assert_eq!(m.next_event(0), m.model_next_event_scan(0));
}

#[test]
fn unwinding_an_edit_still_rebuilds_the_index() {
    let mut m = machine();
    assert_eq!(m.next_event(0), None);
    let failed = catch_unwind(AssertUnwindSafe(|| {
        let _ = m.edit_state(|s| {
            s.driver.queue_repeat[4] = Some(repeated());
            panic!("interrupted debugger edit");
        });
    }));
    assert!(failed.is_err());
    assert_eq!(m.next_event(0), Some(45));
}

#[test]
fn editing_after_quiescence_refreshes_the_next_run_frontier() {
    let mut m = machine();
    m.run(100, &mut NoObserver).unwrap();
    assert_eq!(m.next_tick(), None);
    m.edit_state(|s| s.driver.queue_repeat[4] = Some(repeated())).unwrap();
    assert_eq!(m.next_tick(), Some(45));
    assert!(m.run(100, &mut NoObserver).unwrap().ticks > 0);
}

#[test]
fn invalid_shape_is_reported_and_reset_recovers() {
    let mut m = machine();
    assert!(m.edit_state(|s| s.iq.len[4] = usize::MAX).is_err());
    assert!(m.run(100, &mut NoObserver).is_err());
    m.reset();
    assert!(m.run(100, &mut NoObserver).is_ok());
}

fn load(m: &mut Machine, q: u16, insns: &[Insn]) {
    let units: Vec<u32> = insns
        .iter()
        .flat_map(|i| {
            let word = encode(i);
            if i.entry().width == 64 {
                vec![(word >> 32) as u32, word as u32]
            } else {
                vec![word as u32]
            }
        })
        .collect();
    m.load_image(q, &units).unwrap();
}

fn continue_scenario(scenario: usize, edit: bool) -> (Vec<u8>, Vec<u64>, [u128; 7], Option<u128>) {
    use tsp_iss::harness::output::TraceObserver;
    use tsp_iss::harness::stats::{Counters, StatsObserver};
    let mut m = machine();
    let mut horizon = 1;
    let read = Insn::Read { addr: 1, dst: 1, thread: 0 };
    match scenario {
        0 => load(&mut m, 0, &[read]),
        1 => {
            load(&mut m, 0, &[read, Insn::Iterations { n: 3 }, Insn::Step { a: 1 }, Insn::NOP { n: 5 }]);
            horizon = 2;
        }
        2 => load(&mut m, 0, &[read, Insn::Repeat { n: 3, d: 4 }]),
        3 => {
            load(&mut m, 0, &[Insn::Sync, Insn::SetVL { n: 1 }]);
            load(&mut m, 1, &[Insn::NOP { n: 5 }, Insn::Notify]);
        }
        4 => {
            load(&mut m, 0, &[read]);
            horizon = 2;
        }
        5 => {
            let data = vec![0; m.geom().vlanes as usize];
            m.edit_state(|s| assert!(s.iq.fetch(100, 8, &data, &data))).unwrap();
        }
        6 => {
            let wide = encode(&read);
            m.load_image(0, &[(wide >> 32) as u32]).unwrap();
        }
        _ => unreachable!(),
    }
    let mut sink = Vec::new();
    let mut counts = Counters::default();
    {
        let mut inner = TraceObserver::new(&mut sink, true);
        let mut observer = StatsObserver { inner: &mut inner, counts: &mut counts };
        m.run_until(horizon, 1000, &mut observer).unwrap();
        let before = m.next_tick();
        if scenario == 0 {
            assert!(m.driver().early_closes > 0, "exercise early-close shadow event");
        }
        if edit {
            m.edit_state(|_| ()).unwrap();
            assert_eq!(m.next_tick(), before, "no-op edit changed frontier in scenario {scenario}");
        }
        m.run(1000, &mut observer).unwrap();
    }
    let waits = counts.wait_totals();
    (sink, counts.dispatches, waits, m.next_tick())
}

#[test]
fn no_op_edit_preserves_future_trace_and_frontier_across_partial_histories() {
    for scenario in 0..7 {
        assert_eq!(continue_scenario(scenario, false), continue_scenario(scenario, true), "scenario {scenario}");
    }
}

#[test]
fn speculative_future_query_does_not_poison_an_earlier_query_or_run() {
    let mut m = machine();
    m.edit_state(|s| s.driver.queue_repeat[4] = Some(repeated())).unwrap();
    assert_eq!(m.next_event(0), Some(45));
    assert_eq!(m.next_event(1000), Some(1001));
    assert_eq!(m.next_event(0), Some(45), "event cache omitted the query-cycle input");
    assert_eq!(m.next_event(0), m.model_next_event_scan(0));
    assert!(m.run(100, &mut NoObserver).is_ok());
}
