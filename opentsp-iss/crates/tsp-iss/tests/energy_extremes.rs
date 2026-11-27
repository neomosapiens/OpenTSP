// Range correctness of post-hoc estimates; no simulation timing or trace semantics change.
use serde_json::{json, Value};
use tsp_iss::slices::activity::{Activity, Structure};
use tsp_iss::slices::energy::{estimate, estimate_run, Table};

fn reference_activity() -> Activity {
    Activity::with_structure(
        320,
        20,
        Structure { vlanes: 320, mem_slices: 88, mxm_planes: 4, mxm_rows: 320, c2c_links: 16 },
    )
}
fn document() -> Value {
    let mut v: Value = serde_json::from_str(include_str!("fixtures/energy-table-schema1.json")).unwrap();
    fn zero(v: &mut Value) {
        match v {
            Value::Object(map) => map.values_mut().for_each(zero),
            Value::Number(_) => *v = json!(0),
            _ => {}
        }
    }
    zero(&mut v["events"]);
    zero(&mut v["cycles"]);
    v["static_w"] = json!(0);
    v
}
fn table(v: &Value) -> Table {
    Table::parse(&v.to_string()).unwrap()
}
fn close(value: &Value, expected: f64) {
    let actual = value.as_f64().unwrap_or_else(|| panic!("expected finite {expected:e}, got {value}"));
    assert!(actual.is_finite());
    let relative = ((actual - expected) / expected).abs();
    assert!(relative < 2e-14, "{actual:e} vs {expected:e}: relative {relative:e}");
}
#[test]
fn finite_joules_survive_overflowing_picojoules() {
    let mut v = document();
    v["cycles"]["CU"]["idle_pj"] = json!(1e308);
    let mut a = reference_activity();
    a.active_cycles[5] = 1;
    let run = estimate_run(&table(&v), &[(a, 0, 101)], 900.0);
    close(&run["per_chip"][0]["joules"]["idle_cycles"], 1e298);
    close(&run["joules"], 1e298);
    close(&run["watts"], 1e298 / (101.0 / 900e6));
}
#[test]
fn event_sums_and_multiple_chips_are_priced_before_serialization() {
    let mut v = document();
    v["events"]["mem_read_pj_per_byte"] = json!(1e308);
    v["events"]["host_pj_per_byte"] = json!(1e308);
    let mut a = reference_activity();
    a.mem[0].read_bytes = 100;
    a.host_in = 1;
    let run = estimate_run(&table(&v), &[(a.clone(), 0, 101), (a, 0, 101)], 900.0);
    close(&run["joules"], 2.0 * (100.0 + 320.0) * 1e296);
}
#[test]
fn truly_unrepresentable_joules_are_unavailable_while_watts_can_be_finite() {
    let mut v = document();
    v["cycles"]["CU"]["idle_pj"] = json!(1e300);
    let a = reference_activity();
    let cycles = 100_000_000_000_000_000_000u128;
    let run = estimate_run(&table(&v), &[(a.clone(), 0, cycles), (a, 0, cycles)], 900.0);
    assert!(run["joules"].is_null(), "actual2e308J cannot be represented: {run}");
    close(&run["watts"], 2e300 * 1e-12 * 900e6);
}
#[test]
fn zero_work_absorbs_large_factors_and_zero_duration_has_no_watts() {
    let mut v = document();
    v["events"]["mem_read_pj_per_byte"] = json!(1e308);
    v["events"]["vxm_op_pj"]["add_sub"] = json!(1e308);
    v["events"]["vxm_width_scale"]["w1"] = json!(1e308);
    let a = reference_activity();
    let run = estimate_run(&table(&v), &[(a, 0, 0)], 900.0);
    assert_eq!(run["joules"], json!(0.0));
    assert!(run["watts"].is_null());
}
#[test]
fn tiny_coefficients_with_large_counts_remain_nonzero() {
    let mut v = document();
    v["events"]["mem_read_pj_per_byte"] = json!(1e-320);
    let mut a = reference_activity();
    a.mem[0].read_bytes = u64::MAX;
    let run = estimate(&table(&v), &a, 0, 1, 900.0);
    let actual = run["joules"]["mem"].as_f64().unwrap();
    let expected = (u64::MAX as f64 * 1e-320) * 1e-12;
    assert!(actual > 0.0);
    assert!((actual - expected).abs() <= f64::from_bits(2), "{actual:e} vs {expected:e}");
}
#[test]
fn integer_counter_sums_do_not_wrap_before_pricing() {
    let mut v = document();
    v["events"]["srf_write_pj_per_byte"] = json!(1);
    let mut a = reference_activity();
    a.srf_writes = [u64::MAX, u64::MAX];
    let run = estimate(&table(&v), &a, 0, 1, 900.0);
    close(&run["joules"]["srf"], (2.0 * u64::MAX as f64) * 320.0 * 1e-12);
}

// Precision regression at the upper bound of the reference profile.
#[test]
fn reference_values_are_not_silently_truncated_to_integer_counts() {
    let mut v = document();
    v["reference"]["vlanes"] = json!(320.5);
    let run = estimate(&table(&v), &reference_activity(), 0, 1, 900.0);
    close(&run["class_scale"]["VXM"], 320.0 / 320.5);
}
#[test]
fn reference_products_can_be_wide_while_final_energy_is_finite() {
    let mut v = document();
    v["reference"]["vlanes"] = json!(1e308);
    v["reference"]["mem_slices"] = json!(1e308);
    v["cycles"]["MEM"]["idle_pj"] = json!(1e308);
    let run = estimate(&table(&v), &reference_activity(), 0, 100_000_000_000_000_000_000u128, 900.0);
    close(&run["joules"]["idle_cycles"], 2.816e-296);
}

#[test]
fn compound_factors_recover_from_overflow_and_underflow() {
    let mut v = document();
    v["events"]["vxm_op_pj"]["add_sub"] = json!(1e308);
    v["events"]["vxm_width_scale"]["w4"] = json!(1e308);
    v["events"]["vxm_float_scale"] = json!(1e-308);
    let mut a = reference_activity();
    a.vxm[ots_isa::EntryId::Add.index()][3] = 1; // f32
    close(&estimate(&table(&v), &a, 0, 1, 900.0)["joules"]["vxm"], 1e296);
    v["events"]["vxm_op_pj"]["add_sub"] = json!(1e-308);
    v["events"]["vxm_width_scale"]["w4"] = json!(1e-308);
    v["events"]["vxm_float_scale"] = json!(1e308);
    let tiny = estimate(&table(&v), &a, 0, 1, 900.0)["joules"]["vxm"].as_f64().unwrap();
    assert!(tiny > 0.0 && (tiny - 1e-320).abs() <= f64::from_bits(2), "{tiny:e}");
}
#[test]
fn zero_factor_after_large_products_is_mathematical_zero() {
    let mut v = document();
    v["events"]["vxm_op_pj"]["add_sub"] = json!(1e308);
    v["events"]["vxm_width_scale"]["w4"] = json!(1e308);
    v["events"]["vxm_float_scale"] = json!(0);
    let mut a = reference_activity();
    a.vxm[ots_isa::EntryId::Add.index()][3] = 1;
    assert_eq!(estimate(&table(&v), &a, 0, 1, 900.0)["joules"]["vxm"], json!(0.0));
}
#[test]
fn unavailable_chip_joules_are_not_aggregated_as_zero() {
    let mut v = document();
    v["cycles"]["CU"]["idle_pj"] = json!(1e308);
    let cycles = 100_000_000_000_000_000_000u128;
    let run = estimate_run(&table(&v), &[(reference_activity(), 0, cycles)], 900.0);
    assert!(run["per_chip"][0]["joules"]["total"].is_null());
    assert!(run["joules"].is_null());
    close(&run["watts"], 9e304);
    assert!(tsp_iss::slices::energy::summary(&run).contains("unavailable values are not zero"));
}
#[test]
fn unrepresentable_static_watts_can_still_have_finite_joules() {
    let mut v = document();
    v["static_w"] = json!(1e308);
    for value in v["static_share"].as_object_mut().unwrap().values_mut() {
        *value = json!(0);
    }
    v["static_share"]["MEM"] = json!(1);
    v["reference"]["vlanes"] = json!(1);
    v["reference"]["mem_slices"] = json!(1);
    let run = estimate_run(&table(&v), &[(reference_activity(), 0, 1)], 100000.0);
    close(&run["joules"], 2.816e301);
    assert!(run["watts"].is_null());
}

#[test]
fn finite_output_at_binary64_upper_boundary_is_not_spurious_null() {
    let mut v = document();
    v["cycles"]["CU"]["idle_pj"] = json!(1.7976931348622493e305);
    let run = estimate(&table(&v), &reference_activity(), 0, 1_000_000_000_000_037, 900.0);
    assert_eq!(run["joules"]["idle_cycles"].as_f64(), Some(f64::MAX));
}

#[test]
fn picojoule_unit_conversion_rounds_subnormal_ties_once() {
    let mut v = document();
    v["cycles"]["CU"]["idle_pj"] = json!(f64::from_bits(1));
    for (cycles, expected_bits) in [(500_000_000_000u128, 0u64), (1_500_000_000_000, 2), (2_500_000_000_000, 2)] {
        let run = estimate(&table(&v), &reference_activity(), 0, cycles, 900.0);
        assert_eq!(run["joules"]["idle_cycles"].as_f64().unwrap().to_bits(), expected_bits);
    }
}
