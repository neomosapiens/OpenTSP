//! Full admitted timestamp boundaries: absence is separate from a valid maximum cycle.
mod common;
use common::*;
use ots_profile::Profile;
use tsp_iss::links::NoLinks;
use tsp_iss::observe::NoObserver;
use tsp_iss::Machine;

#[test]
fn run_keeps_the_maximum_tick_distinct_from_an_exclusive_horizon() {
    let mut m = Machine::new(&Profile::load("mini").unwrap(), 0, Box::new(NoLinks)).unwrap();
    m.set_host_script(Vec::new(), u64::MAX);
    let before = m.run_until(u64::MAX, u64::MAX, &mut NoObserver).unwrap();
    assert_eq!(before.ticks, 0);
    assert_eq!(m.next_tick(), Some(u128::from(u64::MAX)));
    let last = m.run(u64::MAX, &mut NoObserver).unwrap();
    assert_eq!(last.ticks, 1);
    assert_eq!(last.last_cycle, Some(u64::MAX));
    assert!(m.quiescent());
}

#[test]
fn generated_commits_beyond_maximum_are_budget_failures_with_exact_future_time() {
    let profile = config("mini").to_string_lossy().to_string();
    let maximum = u64::MAX.to_string();
    for start in [u64::MAX - 5, u64::MAX - 1, u64::MAX] {
        let input = format!("S 0 0 {}\nH start {start:x}\nQ 0 10000000 00000000\n", vec!["01"; 16].join(" "));
        let r = run_iss(&[&profile, "--trace", "--max-cycles", &maximum], &input);
        if start == u64::MAX - 5 {
            assert_eq!(r.status.code(), Some(0), "{}", text(&r.stderr));
            assert!(text(&r.stdout).contains(&format!("T srf {} ", u64::MAX)));
        } else {
            assert_eq!(r.status.code(), Some(3), "{}", text(&r.stderr));
            let next = if start == u64::MAX { u128::from(start) + 1 } else { u128::from(start) + 5 };
            assert!(text(&r.stderr).contains(&format!("needs cycle {next}")), "{}", text(&r.stderr));
        }
    }
}

#[cfg(feature = "replay")]
#[test]
fn replay_pending_commit_at_maximum_has_a_presence_bit_and_no_clock_wrap() {
    use std::sync::Arc;
    use tsp_iss::layout::RowId;
    use tsp_iss::observe::{Observer, Row};
    use tsp_iss::operands::Commit;
    #[derive(Default)]
    struct Writes(Vec<u64>);
    impl Observer for Writes {
        fn srf(&mut self, c: u64, _: Row, _: u8, _: u8, _: &[u8]) {
            self.0.push(c);
        }
    }
    let create = || Machine::new(&Profile::load("mini").unwrap(), 0, Box::new(NoLinks)).unwrap();
    let mut recorded = create();
    recorded.record_schedule().unwrap();
    recorded.run(u64::MAX, &mut NoObserver).unwrap();
    let mut schedule = recorded.take_schedule().unwrap();
    assert_eq!(schedule.cycles, vec![0]);
    // A controlled complete empty schedule shifted to the last two cycles exercises
    // the extra-commit path without changing the ordinary recording admission policy.
    schedule.cycles[0] = u64::MAX - 1;
    let schedule = Arc::new(schedule);
    let mut replay = create();
    // Set the architectural state before arming replay, whose admission checks this state.
    replay
        .edit_state(|state| {
            state.driver.set_pending_list(
                127,
                vec![Commit { r: RowId::E, id: 0, g: 0, value: [1; tsp_iss::geom::MAX_LANES], q: 0, b: 0, ov: 0 }],
            );
        })
        .unwrap();
    replay.replay_schedule_with(schedule, false).unwrap();
    replay.set_host_script(Vec::new(), u64::MAX - 1);
    let mut writes = Writes::default();
    let summary = replay.run(u64::MAX, &mut writes).unwrap();
    assert_eq!(summary.last_cycle, Some(u64::MAX));
    assert_eq!(writes.0, vec![u64::MAX]);
}

#[test]
fn maximum_timestamp_events_reach_the_budget_gate() {
    let profile = config("mini").to_string_lossy().to_string();
    let zero = vec!["00"; 16].join(" ");
    for cycle in ["fffffffffffffffe", "ffffffffffffffff"] {
        let forms = [
            format!("H csr {cycle} w 300 1\nR 300\n"),
            format!("H csr {cycle} r 300\nR 300\n"),
            format!("H db {cycle} 0\nR 300\n"),
            format!("H inject {cycle} 0 00000001\nR 300\n"),
            format!("H dma {cycle} {zero}\nR 300\n"),
            format!("H start {cycle}\nR 300\n"),
            format!("P ctrl {cycle} 0\nR 300\n"),
            format!("P csr {cycle} 0 w 300 1\nR 300\n"),
            format!("P csr {cycle} 0 r 300\nR 300\n"),
            format!("F sram {cycle} 0 0 0 0\nR 300\n"),
            format!("F srf {cycle} e 0 0 0 0\nR 300\n"),
            format!("F iq {cycle} 0 0 0\nR 300\n"),
            format!("F mxm {cycle} 0 0 0\nR 300\n"),
        ];
        for input in forms {
            let r = run_iss(&[&profile, "--trace"], &input);
            assert_eq!(r.status.code(), Some(3), "{input}: {}", text(&r.stderr));
            assert!(r.stdout.is_empty());
            assert!(text(&r.stderr).contains(&format!(
                "needs cycle {}, beyond --max-cycles 100000000",
                u64::from_str_radix(cycle, 16).unwrap()
            )));
        }
    }
}

#[test]
fn expired_srf_values_do_not_become_live_across_the_signed_boundary() {
    let profile = config("mini").to_string_lossy().to_string();
    for cycle in [100u64, (1u64 << 63) + 100] {
        let input = format!(
            "S 2c 0 {}\nQ 2c 10000000 00000000\nH inject {cycle:x} 58 23c10000 00000000\n",
            vec!["01"; 16].join(" ")
        );
        let limit = (cycle + 1000).to_string();
        let r = run_iss(&[&profile, "--trace", "--max-cycles", &limit], &input);
        assert_eq!(r.status.code(), Some(0), "{}", text(&r.stderr));
        let output = text(&r.stdout);
        assert!(output.contains(&format!("T diag {} queue 88 OTS-E-TIM-001", cycle + 2)), "{output}");
        assert!(output.contains(&format!("E OTS-E-TIM-001 {} 1", cycle + 2)), "{output}");
    }
}
