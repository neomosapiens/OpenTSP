//! Tests of `driver/commit`, moved from the module's in-source test block (they use the crate's public API).
//! Ports of `unit_tests/test_streams.sail` that go through `tick` (`test_sxm_or_merge`,
//! `test_mem_thread_merge`) and the ISS's own tests of `commit_due`: SRF-002, the report order, the
//! callbacks, the ring slots.
use ots_isa::Code;
use ots_profile::Profile;
use tsp_iss::layout::{write_gap, RowId, UnitKind};
use tsp_iss::links::NoLinks;
use tsp_iss::observe::{NoObserver, Observer, Row};
use tsp_iss::operands::Commit;
use tsp_iss::srf::Vec320;
use tsp_iss::Machine;

fn machine(name: &str) -> Machine {
    Machine::new(&Profile::load(name).unwrap(), 0, Box::new(NoLinks)).unwrap()
}

/// `v(b)`: `vlanes` copies of byte `b`.
fn v(m: &Machine, b: u8) -> Vec320 {
    let mut x = [0u8; tsp_iss::geom::MAX_LANES];
    x[..m.geom().vlanes as usize].fill(b);
    x
}

fn commit(r: RowId, id: u8, g: u8, value: Vec320, q: usize, b: u8) -> Commit {
    Commit { r, id, g, value, q, b, ov: 0 }
}

#[derive(Default)]
struct Log {
    diags: Vec<(u64, u16, Code)>,
    srf: Vec<(u64, Row, u8, u8)>,
}
impl Observer for Log {
    fn diag(&mut self, c: u64, q: u16, code: Code) {
        self.diags.push((c, q, code));
    }
    fn srf(&mut self, c: u64, r: Row, s: u8, g: u8, _v: &[u8]) {
        self.srf.push((c, r, s, g));
    }
}

/// `or_case`: two commits to E:7 at the SXM west select's write gap, in cycle 50, in list order.
// model: unit_tests/test_streams.sail or_case
fn or_case(qa: usize, va: u8, qb: usize, vb: u8) -> Vec320 {
    let mut m = machine("ref");
    let g = write_gap(m.geom(), UnitKind::SxmWSel, 0, RowId::E).unwrap_or(0);
    let c = 50;
    let (a, b) = (v(&m, va), v(&m, vb));
    m.edit_state(|state| {
        state.driver.set_pending_list(c % 128, vec![commit(RowId::E, 7, g, a, qa, 0), commit(RowId::E, 7, g, b, qb, 0)])
    })
    .unwrap();
    m.tick(c as u64, &mut NoObserver);
    let g1 = if g < 43 { g + 1 } else { g }; // the turn-out write is in register 5 next cycle
    m.srf_value(Row::East, 7, g1, c as u64 + 1).expect("found")
}

// model: unit_tests/test_streams.sail test_sxm_or_merge
#[test]
fn sxm_or_merge_of_sel_and_tsp() {
    let m = machine("ref");
    assert_eq!(or_case(125, 0x0c, 120, 0x30), v(&m, 0x3c)); // SEL + TSP: OR
    assert_eq!(or_case(120, 0x30, 125, 0x0c), v(&m, 0x3c)); // record order does not matter
    assert_eq!(or_case(121, 0x01, 125, 0x80), v(&m, 0x81)); // the other transposer
    assert_eq!(or_case(122, 0x0c, 125, 0x30), v(&m, 0x0c)); // SHF + SEL: collision, 122 kept
    assert_eq!(or_case(124, 0x0c, 125, 0x30), v(&m, 0x0c)); // PRM + SEL: collision
    assert_eq!(or_case(121, 0x30, 120, 0x0c), v(&m, 0x0c)); // two TSPs: collision, 120 kept
}

// model: unit_tests/test_streams.sail test_sxm_or_pair_counts_distinct_writers
#[test]
fn duplicate_tsp_records_are_one_writer_in_a_sel_tsp_pair() {
    let mut m = machine("ref");
    let g = write_gap(m.geom(), UnitKind::SxmWSel, 0, RowId::E).unwrap();
    // An admitted custom timing profile can align Rotate and Transpose from
    // the same TSP queue at one commit cycle. They remain one (queue, plane) writer.
    let values = [v(&m, 1), v(&m, 2), v(&m, 4)];
    m.edit_state(|state| {
        state.driver.set_pending_list(
            50,
            vec![
                commit(RowId::E, 7, g, values[0], 125, 0),
                commit(RowId::E, 7, g, values[1], 120, 0),
                commit(RowId::E, 7, g, values[2], 120, 0),
            ],
        )
    })
    .unwrap();
    let mut log = Log::default();
    m.tick(50, &mut log);
    assert_eq!(m.srf_value(Row::East, 7, g + 1, 51), Some(v(&m, 7)));
    assert!(log.diags.is_empty(), "one SEL writer and one TSP writer must not collide: {:?}", log.diags);
    assert_eq!(log.srf.len(), 1, "the representative stores the merged value once");
}

// model: unit_tests/test_streams.sail test_mem_thread_merge
#[test]
fn mem_thread_merge() {
    let m = machine("ref");
    assert_eq!(or_case(4, 0x0c, 4, 0x30), v(&m, 0x3c));
    assert_eq!(or_case(4, 0x30, 4, 0x0c), v(&m, 0x3c));
    let mut m = machine("ref");
    let g = write_gap(m.geom(), UnitKind::SxmWSel, 0, RowId::E).unwrap_or(0);
    let c = 50u64;
    let list = vec![
        commit(RowId::E, 7, g, v(&m, 0x30), 4, 0),
        commit(RowId::E, 7, g, v(&m, 0x40), 9, 0),
        commit(RowId::E, 7, g, v(&m, 0x0c), 4, 0),
    ];
    m.edit_state(|state| state.driver.set_pending_list(50, list)).unwrap();
    let mut log = Log::default();
    m.tick(c, &mut log);
    let g1 = if g < 43 { g + 1 } else { g };
    // with a third writer the slice still collides and keeps the merged value as the lowest writer
    assert_eq!(m.srf_value(Row::East, 7, g1, c + 1), Some(v(&m, 0x3c)));
    assert_eq!(log.diags, vec![(50, 4, Code::Srf002), (50, 9, Code::Srf002)]);
    assert_eq!(log.srf, vec![(50, Row::East, 7, g)], "the winner writes once");
}

#[test]
fn collisions_report_on_every_writer_in_queue_order_and_the_lowest_keeps_its_value() {
    let mut m = machine("ref");
    let g = 18;
    // list order: 101, 100, 100 on another plane; all three target E:2 at gap 18
    let list = vec![
        commit(RowId::E, 2, g, v(&m, 1), 101, 0),
        commit(RowId::E, 2, g, v(&m, 2), 100, 1),
        commit(RowId::E, 2, g, v(&m, 3), 100, 0),
        commit(RowId::E, 5, g, v(&m, 9), 120, 0), // another stream: no collision
    ];
    m.edit_state(|state| state.driver.set_pending_list(7, list)).unwrap();
    let mut log = Log::default();
    m.tick(7, &mut log);
    assert_eq!(log.diags, vec![(7, 100, Code::Srf002), (7, 100, Code::Srf002), (7, 101, Code::Srf002)]);
    assert_eq!(m.srf_value(Row::East, 2, g + 1, 8), Some(v(&m, 3)), "(queue 100, plane 0) wins");
    assert_eq!(m.srf_value(Row::East, 5, g + 1, 8), Some(v(&m, 9)));
    // the callbacks are in list order: stream 2's winner is third in the list, stream 5 fourth
    assert_eq!(log.srf, vec![(7, Row::East, 2, g), (7, Row::East, 5, g)]);
}

#[test]
fn writes_to_different_gaps_rows_or_streams_do_not_collide() {
    let mut m = machine("ref");
    let list = vec![
        commit(RowId::E, 2, 18, v(&m, 1), 100, 0),
        commit(RowId::E, 2, 19, v(&m, 2), 101, 0),
        commit(RowId::W, 2, 18, v(&m, 3), 102, 0),
        commit(RowId::E, 3, 18, v(&m, 4), 103, 0),
    ];
    m.edit_state(|state| state.driver.set_pending_list(3, list)).unwrap();
    let mut log = Log::default();
    m.tick(3, &mut log);
    assert!(log.diags.is_empty());
    assert_eq!(log.srf.len(), 4);
    assert_eq!(
        log.srf.iter().map(|x| (x.1, x.2, x.3)).collect::<Vec<_>>(),
        vec![(Row::East, 2, 18), (Row::East, 2, 19), (Row::West, 2, 18), (Row::East, 3, 18)]
    );
}

#[test]
fn a_ring_slot_is_applied_in_its_cycle_and_cleared() {
    let mut m = machine("ref");
    let value = v(&m, 1);
    m.edit_state(|state| state.driver.set_pending_list(5, vec![commit(RowId::E, 0, 10, value, 100, 0)])).unwrap();
    let value = v(&m, 2);
    m.edit_state(|state| state.driver.set_pending_list(6, vec![commit(RowId::E, 1, 10, value, 100, 0)])).unwrap();
    m.tick(5, &mut NoObserver);
    assert!(m.driver().pending[5].is_empty() && m.driver().pending[6].len() == 1);
    assert_eq!(m.srf_value(Row::East, 0, 11, 6), Some(v(&m, 1)));
    assert_eq!(m.srf_value(Row::East, 1, 11, 7), None);
    assert_eq!(m.next_event(5), Some(6));
    m.tick(6, &mut NoObserver);
    assert_eq!(m.srf_value(Row::East, 1, 11, 7), Some(v(&m, 2)));
    assert!(m.quiescent());
    // cycle 5 + 128 uses the same slot
    let value = v(&m, 3);
    m.edit_state(|state| state.driver.set_pending_list(5, vec![commit(RowId::W, 4, 10, value, 100, 0)])).unwrap();
    assert_eq!(m.next_event(6), Some(5 + 128));
    m.tick(5 + 128, &mut NoObserver);
    assert_eq!(m.srf_value(Row::West, 4, 11, 5 + 129), Some(v(&m, 3)));
}

#[test]
fn a_commit_to_the_same_target_in_a_later_cycle_is_not_a_collision() {
    let mut m = machine("ref");
    let value = v(&m, 1);
    m.edit_state(|state| state.driver.set_pending_list(10, vec![commit(RowId::E, 2, 18, value, 100, 0)])).unwrap();
    let value = v(&m, 2);
    m.edit_state(|state| state.driver.set_pending_list(11, vec![commit(RowId::E, 2, 18, value, 101, 0)])).unwrap();
    let mut log = Log::default();
    m.tick(10, &mut log);
    m.tick(11, &mut log);
    assert!(log.diags.is_empty());
    assert_eq!(m.srf_value(Row::East, 2, 20, 12), Some(v(&m, 1)));
    assert_eq!(m.srf_value(Row::East, 2, 20, 13), Some(v(&m, 2)));
}

// model: postlude/tick.sail commit_wins: the writes of one (queue, plane) are one writer
#[test]
fn two_writes_of_one_writer_or_together_and_the_first_in_the_list_stands_for_it() {
    let mut m = machine("ref");
    let list = vec![commit(RowId::E, 2, 18, v(&m, 0x03), 4, 0), commit(RowId::E, 2, 18, v(&m, 0x0c), 4, 0)];
    m.edit_state(|state| state.driver.set_pending_list(9, list)).unwrap();
    let mut log = Log::default();
    m.tick(9, &mut log);
    assert!(log.diags.is_empty());
    assert_eq!(log.srf.len(), 1);
    assert_eq!(m.srf_value(Row::East, 2, 19, 10), Some(v(&m, 0x0f)));
}

// The trace of a commit cycle: the `T srf` lines of the cycle's writes, then the cycle's
// SRF-002 reports (the model's order: callbacks during `commit_due`, reports after the loop).
#[test]
fn the_trace_lines_of_a_commit_cycle() {
    use tsp_iss::harness::output::TraceObserver;
    let mut m = machine("mini");
    let list = vec![
        commit(RowId::E, 2, 18, v(&m, 1), 101, 0),
        commit(RowId::E, 2, 18, v(&m, 2), 100, 0),
        commit(RowId::W, 31, 0, v(&m, 3), 100, 0),
    ];
    m.edit_state(|state| state.driver.set_pending_list(7, list)).unwrap();
    let mut out = Vec::new();
    {
        let mut obs = TraceObserver::new(&mut out, true);
        m.tick(7, &mut obs);
        assert_eq!(obs.diags, 2);
    }
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "T srf 7 E:2 gap 18 02020202020202020202020202020202\nT srf 7 W:31 gap 0 03030303030303030303030303030303\nT diag 7 queue 100 OTS-E-SRF-002\nT diag 7 queue 101 OTS-E-SRF-002\n"
    );
}

/// A cycle with many commits (the duplicate check is a bit set, not the pairwise scan of a small list): without a
/// shared target every commit is written and nothing is reported; with any two commits (at any positions) on one
/// target, both writers are reported (OTS-E-SRF-002) and one value is written less; the detector is clean afterwards.
#[test]
fn many_commits_in_one_cycle_find_every_shared_target() {
    let c = 50u64;
    let run = |n: usize, dup: Option<(usize, usize)>, two_rows: bool| {
        let mut m = machine("ref");
        let list: Vec<Commit> = (0..n)
            .map(|i| {
                let target = match dup {
                    Some((a, b)) if i == b => a,
                    _ => i,
                };
                // `two_rows`: the second half of the list repeats the first half's streams on the other row (distinct targets)
                let (row, target) = if two_rows {
                    (if i < n / 2 { RowId::E } else { RowId::W }, target % (n / 2))
                } else {
                    (RowId::E, target)
                };
                commit(row, target as u8, 10, v(&m, i as u8 + 1), 20 + i, 0)
            })
            .collect();
        m.edit_state(|state| state.driver.set_pending_list(c as usize % 128, list)).unwrap();
        let mut log = Log::default();
        m.tick(c, &mut log);
        assert!(m.driver().commit_seen.iter().all(|w| *w == 0), "the detector is clean after a cycle");
        if dup.is_none() {
            // the cycle took the fast path: the sorting scratch of the slow path was never used
            assert_eq!(m.driver().commit_order.capacity(), 0, "{n} distinct targets took the slow path");
        }
        (log.diags.iter().filter(|d| d.2 == Code::Srf002).count(), log.srf.len())
    };
    for n in [9usize, 12, 31] {
        assert_eq!(run(n, None, false), (0, n), "{n} distinct targets");
        if n % 2 == 0 {
            assert_eq!(run(n, None, true), (0, n), "{n} targets on two rows");
        }
        for a in 0..n {
            for b in a + 1..n {
                assert_eq!(run(n, Some((a, b)), false), (2, n - 1), "{n} commits, {a} and {b} share a target");
            }
        }
    }
}
