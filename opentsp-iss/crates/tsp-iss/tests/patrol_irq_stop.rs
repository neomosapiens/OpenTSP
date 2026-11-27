//! Error-interrupt stopping must not execute host events after the patrol IRQ cycle.
mod common;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Config(PathBuf);
impl Drop for Config {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn config() -> Config {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let mut json: serde_json::Value = serde_json::from_slice(&std::fs::read(common::config("mini")).unwrap()).unwrap();
    json["scrub"]["period"] = 8.into();
    let path = std::env::temp_dir().join(format!(
        "opentsp-patrol-irq-stop-{}-{}.json",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
    Config(path)
}

fn stop(input: &str) -> std::process::Output {
    let cfg = config();
    common::run_iss(&[cfg.0.to_str().unwrap(), "--trace", "--on-irq", "stop", "--max-cycles", "100"], input)
}

const FAULTS: &str = "F sram 1 0 1 0 5\nF sram 1 0 1 0 9\n";

#[test]
fn patrol_irq_stops_before_a_future_host_mask_write() {
    let input = format!("{FAULTS}H csr 28 w 16 0\nR 16\n");
    let out = stop(&input);
    let text = String::from_utf8(out.stdout).unwrap();
    assert_eq!(out.status.code(), Some(5), "{text}");
    assert!(text.contains("T irq 8 msix 0\n"), "first error IRQ must be cycle 8: {text}");
    assert!(text.contains("X 016 aa\n"), "future mask write must not occur: {text}");
    assert!(!text.contains("T csr 40 "), "must stop before future cycle 40: {text}");
}

#[test]
fn sparse_and_dense_event_schedules_stop_with_the_same_ras_state() {
    let future = "H csr 28 w 16 0\nR 16\nR 13\nR 14\n";
    let sparse = stop(&format!("{FAULTS}{future}"));
    let mut dense = FAULTS.to_string();
    for cycle in 0..40 {
        dense.push_str(&format!("H csr {cycle:x} w 7f0 0\n"));
    }
    dense.push_str(future);
    let dense = stop(&dense);
    let a = String::from_utf8(sparse.stdout).unwrap();
    let b = String::from_utf8(dense.stdout).unwrap();
    assert_eq!(sparse.status.code(), Some(5), "{a}");
    assert_eq!(dense.status.code(), Some(5), "{b}");
    let state =
        |s: &str| s.lines().filter(|l| l.starts_with("X ") || l.starts_with("E ")).collect::<Vec<_>>().join("\n");
    assert_eq!(state(&a), state(&b), "sparse:\n{a}\ndense:\n{b}");
    assert!(a.contains("X 016 aa\n"), "{a}");
    assert!(!a.contains("T irq 16 "), "no later patrol cycle may execute: {a}");
}

#[test]
fn stop_mode_does_not_keep_a_quiescent_machine_alive_for_patrol() {
    let out = stop(&format!("{FAULTS}R 16\n"));
    let text = String::from_utf8(out.stdout).unwrap();
    assert_eq!(out.status.code(), Some(0), "{text}");
    assert!(!text.contains("T irq"), "no remaining work reaches the next patrol slot: {text}");
}

#[test]
fn an_irq_from_a_link_arrival_stops_even_without_a_scheduled_model_tick() {
    use tsp_iss::geom::MAX_LANES;
    use tsp_iss::links::{LinkEvent, LinkPort};
    struct Arrival(bool);
    impl LinkPort for Arrival {
        fn peer(&self, _: u8) -> Option<u16> {
            None
        }
        fn send(&mut self, _: u64, _: u8, _: &[u8; MAX_LANES]) {}
        fn recv_ready(&mut self, _: u64, _: u8) -> bool {
            false
        }
        fn recv(&mut self, _: u64, _: u8) -> [u8; MAX_LANES] {
            [0; MAX_LANES]
        }
        fn next_event(&self) -> Option<u128> {
            (!self.0).then_some(5)
        }
        fn link_events(&mut self, upto: u64) -> Vec<LinkEvent> {
            if !self.0 && upto >= 5 {
                self.0 = true;
                vec![LinkEvent::Fec { cycle: 5, link: 0, unc: true, flit: 0 }]
            } else {
                vec![]
            }
        }
    }
    let profile = ots_profile::Profile::load("mini").unwrap();
    let mut machine = tsp_iss::Machine::new(&profile, 0, Box::new(Arrival(false))).unwrap();
    machine.set_stop_on_irq(true);
    let result = machine.run(100, &mut tsp_iss::NoObserver).unwrap();
    assert!(machine.error_irq_seen(), "the arrival must exercise the real RAS/IRQ path");
    assert!(machine.stopped_at_irq(), "arrival-only IRQ must not bypass the stop check");
    assert_eq!(result.last_cycle, Some(5), "the IRQ cycle must complete before stopping");
}
