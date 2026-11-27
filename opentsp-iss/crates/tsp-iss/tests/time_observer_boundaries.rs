//! Observer probes preserve the complete admitted timestamp domain.
use tsp_iss::harness::obs::{ObsConfig, ObsState, Probe, Recorder};
use tsp_iss::observe::{NoObserver, Observer, Row};

fn state(at: u64) -> ObsState {
    ObsState::new(&ObsConfig { probes: vec![Probe::SrfSnapshot(at)], ..ObsConfig::default() })
}

#[test]
fn final_snapshot_includes_the_maximum_cycle() {
    let mut st = state(u64::MAX);
    Recorder { inner: &mut NoObserver, st: &mut st }.srf(u64::MAX - 1, Row::East, 0, 0, &[7]);
    st.finish();
    let snap = st.snapshots[0].1.as_ref().expect("the maximum-cycle probe is finalized");
    assert_eq!(snap["cycle"].as_u64(), Some(u64::MAX));
    assert_eq!(snap["writes"][0]["register"].as_u64(), Some(1));
    assert_eq!(snap["writes"][0]["value"], "07");
}

#[test]
fn final_writes_do_not_overflow_the_snapshot_retention_age() {
    let mut st = state(u64::MAX);
    let mut observer = NoObserver;
    let mut r = Recorder { inner: &mut observer, st: &mut st };
    r.srf(u64::MAX - 2, Row::East, 0, 40, &[7]);
    r.srf(u64::MAX, Row::East, 1, 0, &[9]);
    st.finish();
    let snap = st.snapshots[0].1.as_ref().unwrap();
    assert_eq!(snap["writes"].as_array().unwrap().len(), 1);
    assert_eq!(snap["writes"][0]["register"].as_u64(), Some(42));
}

#[test]
fn ancient_writes_do_not_wrap_into_final_snapshot_liveness() {
    let mut st = state(u64::MAX - 1);
    Recorder { inner: &mut NoObserver, st: &mut st }.srf(0, Row::East, 0, 43, &[7]);
    st.finish();
    assert!(st.snapshots[0].1.as_ref().unwrap()["writes"].as_array().unwrap().is_empty());
}

#[test]
fn utilization_records_maximum_execution_without_a_sentinel_overflow() {
    use tsp_iss::harness::obs::utilization_json;
    let cfg = ObsConfig { utilization: true, bin: u64::MAX, ..ObsConfig::default() };
    let mut st = ObsState::new(&cfg);
    let mut inner = NoObserver;
    let mut r = Recorder { inner: &mut inner, st: &mut st };
    r.exec_begin(0, 0, u64::MAX);
    r.exec_end();
    let d = utilization_json(0, &st, 1u128 << 64);
    assert_eq!(d["cycles"].to_string(), "18446744073709551616");
    assert_eq!(d["queues"][0]["active_cycles"], 1);
    assert_eq!(d["queues"][0]["last_cycle"].as_u64(), Some(u64::MAX));
}

#[test]
fn utilization_huge_empty_ranges_keep_every_execution_with_implicit_zero_bins() {
    use tsp_iss::harness::obs::utilization_json;
    let cfg = ObsConfig { utilization: true, bin: 1, ..ObsConfig::default() };
    let mut st = ObsState::new(&cfg);
    let mut inner = NoObserver;
    let mut r = Recorder { inner: &mut inner, st: &mut st };
    for c in [0, 2, u64::MAX - 1] {
        r.exec_begin(0, 0, c);
        r.exec_end();
    }
    let d = utilization_json(0, &st, 1u128 << 64);
    assert_eq!(d["timeline_sparse"], true);
    assert_eq!(d["timeline_bins"].to_string(), "18446744073709551615");
    assert_eq!(d["timeline_nonzero"].as_array().unwrap().len(), 3);
    assert_eq!(d["timeline_nonzero"][2]["start"].as_u64(), Some(u64::MAX - 1));
    assert_eq!(d["queues"][0]["executions"], 3);
    assert_eq!(d["queues"][0]["active_cycles"], 3);
}

#[test]
fn summary_requests_hooks_without_timers_events_or_utilization() {
    let cfg = ObsConfig { summary: true, ..ObsConfig::default() };
    assert!(cfg.any());
    let mut st = ObsState::new(&cfg);
    assert!(st.prof.is_none() && st.util.is_none() && st.events.is_empty());
    let mut inner = NoObserver;
    let r = Recorder { inner: &mut inner, st: &mut st };
    assert!(r.profiling(), "the driver must deliver execution counter hooks");
}

#[test]
fn timeline_threshold_and_nondividing_width_preserve_exact_prefix_coordinates() {
    use tsp_iss::harness::obs::{utilization_json, MAX_EVENTS};
    let cfg = ObsConfig { utilization: true, bin: 1, ..ObsConfig::default() };
    let mut st = ObsState::new(&cfg);
    let mut inner = NoObserver;
    {
        let mut r = Recorder { inner: &mut inner, st: &mut st };
        r.exec_begin(0, 0, MAX_EVENTS - 1);
        r.exec_end();
    }
    let u = st.util.as_ref().unwrap();
    assert!(!u.sparse_timeline());
    assert_eq!(u.timeline_bins(), u128::from(MAX_EVENTS));
    assert_eq!(u.bins.len(), usize::try_from(MAX_EVENTS).unwrap());
    {
        let mut r = Recorder { inner: &mut inner, st: &mut st };
        r.exec_begin(0, 0, MAX_EVENTS);
        r.exec_end();
    }
    let d = utilization_json(0, &st, u128::from(MAX_EVENTS) + 1);
    assert!(d.get("timeline").is_none());
    assert_eq!(d["timeline_encoding"], "absolute-nonzero-v1");
    assert_eq!(d["timeline_bins"], MAX_EVENTS + 1);
    let rows = d["timeline_nonzero"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["index"], MAX_EVENTS - 1);
    assert_eq!(rows[1]["index"], MAX_EVENTS);
    drop(st);

    let cfg = ObsConfig { utilization: true, bin: 7, ..ObsConfig::default() };
    let mut st = ObsState::new(&cfg);
    let mut r = Recorder { inner: &mut inner, st: &mut st };
    for c in [0, u64::MAX - 1, u64::MAX - 1] {
        r.exec_begin(0, 0, c);
        r.exec_end();
    }
    let d = utilization_json(0, &st, 1u128 << 64);
    let rows = d["timeline_nonzero"].as_array().unwrap();
    let index = (u64::MAX - 1) / 7;
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[1]["index"].as_u64(), Some(index));
    assert_eq!(rows[1]["start"].as_u64(), Some(index * 7));
    assert_eq!(d["timeline_bins"].to_string(), (u128::from(index) + 1).to_string());
    assert_eq!(d["queues"][0]["executions"], 3);
    assert_eq!(d["queues"][0]["active_cycles"], 2);
    assert!(d["timeline_omitted_bins_are_zero"].as_bool().unwrap());
}
