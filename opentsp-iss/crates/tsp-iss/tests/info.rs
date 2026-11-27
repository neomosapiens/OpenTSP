//! `tsp-iss info --config F` prints the derived profile as JSON.
mod common;
use common::*;
use serde_json::Value;

fn info(args: &[&str]) -> (Option<i32>, Value, String) {
    let o = run_iss(args, "");
    let v = serde_json::from_slice(&o.stdout).unwrap_or(Value::Null);
    (o.status.code(), v, text(&o.stderr))
}

#[test]
fn info_derives_the_profile() {
    for (name, vlanes, wps) in [("ref", 320, 8192), ("mini", 16, 256)] {
        let path = config(name);
        let (rc, v, err) = info(&["info", "--config", path.to_str().unwrap()]);
        assert_eq!(rc, Some(0), "{err}");
        assert_eq!(v["spec_version"], "0.1");
        assert_eq!(v["vlanes"], vlanes);
        assert_eq!(v["words_per_slice"], wps);
        assert_eq!(v["iq_units"], 320);
        assert_eq!(v["mem_slices_per_hemisphere"], 44);
        assert_eq!(v["quads"], 11);
        // 144 data queues and the control channel
        let q = v["queues"].as_array().unwrap();
        assert_eq!(q.len(), 145);
        assert_eq!(q[0]["unit"], "MEM_W");
        assert_eq!(q[0]["name"], "MEM[0]");
        assert_eq!(q[0]["present"], true);
        assert_eq!(q[144]["queue"], 0xffff);
        assert_eq!(q[144]["unit"], "CU");
        // the gaps follow topology: MEM_W reads row E at mem_up_gap_base - quad
        let c: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let base = c["layout"]["mem_up_gap_base"].as_i64().unwrap();
        assert_eq!(q[0]["stations"]["read"]["E"], base);
        assert_eq!(q[4]["stations"]["read"]["E"], base - 1);
        assert_eq!(q[4]["stations"]["read"]["W"], c["layout"]["mem_down_gap_base"].as_i64().unwrap() + 1);
        assert_eq!(q[88]["unit"], "VXM");
        assert_eq!(q[88]["stations"]["write"]["E"], c["layout"]["vxm_gap"]);
        // the timing table has one row per catalogue entry, with the effective latencies
        let t = v["timing"].as_array().unwrap();
        let enc: Value =
            serde_json::from_slice(&std::fs::read(spec().join("isa-db/generated/encoding.json")).unwrap()).unwrap();
        assert_eq!(t.len(), enc["instructions"].as_array().unwrap().len());
        let add = t.iter().find(|r| r["entry"] == "Add").unwrap();
        assert_eq!(add["d_func"], c["timing"]["vxm"]["d_func"]);
        assert_eq!(add["d_skew"], c["timing"]["vxm"]["d_skew"]);
    }
}

#[test]
fn info_reports_the_extensions_and_the_mxm_keys() {
    // : a compiler asks the ISS what the target supports; the extensions and the MXM keys are in the answer
    for (name, exts, bits, rows, f16) in
        [("ref", 0, 32, 320, 160), ("test/mini_ext", 2, 128, 16, 8), ("test/ext_bf16", 1, 32, 320, 160)]
    {
        let path = config(name);
        let (rc, v, err) = info(&["info", "--config", path.to_str().unwrap()]);
        assert_eq!(rc, Some(0), "{name}: {err}");
        let c: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(v["extensions"], c["extensions"], "{name}");
        assert_eq!(v["extensions"].as_array().unwrap().len(), exts, "{name}");
        assert_eq!(v["mxm_acc_bits"], bits, "{name}");
        assert_eq!((&v["mxm_rows"], &v["mxm_f16_rows"]), (&Value::from(rows), &Value::from(f16)), "{name}");
        for (key, mxm) in [
            ("mxm_acc_depth", "acc_depth"),
            ("mxm_planes_per_hemisphere", "planes_per_hemisphere"),
            ("mxm_fp8_rate", "fp8_rate"),
        ] {
            assert_eq!(v[key], c["mxm"][mxm], "{name} {key}");
        }
    }
}

#[test]
fn info_by_profile_name_and_absent_slices() {
    let (rc, v, err) = info(&["info", "--profile", "ref"]);
    assert_eq!(rc, Some(0), "{err}");
    assert_eq!(v["vlanes"], 320);
    // a profile with 8 MEM slices per hemisphere: slices 8..43 of each hemisphere are absent
    let mut c: Value = serde_json::from_slice(&std::fs::read(config("ref")).unwrap()).unwrap();
    c["geometry"]["mem_slices_per_hemisphere"] = 8.into();
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("c.json");
    std::fs::write(&p, serde_json::to_vec(&c).unwrap()).unwrap();
    let (rc, v, err) = info(&["info", "--config", p.to_str().unwrap()]);
    assert_eq!(rc, Some(0), "{err}");
    assert_eq!(v["quads"], 2);
    let q = v["queues"].as_array().unwrap();
    assert_eq!(q[7]["present"], true);
    assert_eq!(q[8]["present"], false);
    assert_eq!(q[44 + 8]["present"], false);
    assert_eq!(q[44 + 7]["present"], true);
    assert_eq!(q[8]["stations"], Value::Null);
    assert_eq!(q[7]["stations"]["read"]["E"], c["layout"]["mem_up_gap_base"].as_i64().unwrap() - 1);
}

#[test]
fn info_errors_exit_2() {
    let ref_path = config("ref");
    let r = ref_path.to_str().unwrap();
    for args in [
        vec!["info"],
        vec!["info", "--config"],
        vec!["info", "--config", r, "extra"],
        vec!["info", "--profile", "nope"],
        vec!["info", "--config", "/nonexistent/x.json"],
    ] {
        let (rc, _, err) = info(&args);
        assert_eq!(rc, Some(2), "{args:?}");
        assert!(err.starts_with("tsp-iss: "), "{err}");
    }
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("bad.json");
    std::fs::write(&p, "{").unwrap();
    let (rc, v, err) = info(&["info", "--config", p.to_str().unwrap()]);
    assert_eq!((rc, v), (Some(2), Value::Null));
    assert_eq!(err.matches('\n').count(), 1);
    assert!(err.contains("config"), "{err}");
}

#[test]
fn run_and_vectors_need_a_program() {
    // without a program they are usage errors (`bench` is tests/bench.rs)
    for s in ["run", "vectors"] {
        let o = run_iss(&[s], "");
        assert_eq!(o.status.code(), Some(2), "{s}");
        assert!(o.stdout.is_empty());
        assert!(text(&o.stderr).starts_with(&format!("tsp-iss: {s}: usage: tsp-iss {s}")), "{}", text(&o.stderr));
    }
}

#[test]
fn info_lists_the_rows_keyed_by_an_element_type() {
    // a profile with rows keyed `<Entry>.<type>`: one extra object per typed row, with its type; none for a profile without
    let mut c: Value = serde_json::from_slice(&std::fs::read(config("ref")).unwrap()).unwrap();
    let rows = c["timing"]["overrides"].as_array_mut().unwrap();
    rows.push(serde_json::json!({"entry": "Max.i8", "d_func": 3, "d_skew": 2}));
    rows.push(serde_json::json!({"entry": "Max.f32", "d_func": 9, "d_skew": 4}));
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("typed.json");
    std::fs::write(&p, serde_json::to_vec(&c).unwrap()).unwrap();
    let (rc, v, err) = info(&["info", "--config", p.to_str().unwrap()]);
    assert_eq!(rc, Some(0), "{err}");
    let t = v["timing"].as_array().unwrap();
    let typed: Vec<&Value> = t.iter().filter(|r| r.get("type").is_some()).collect();
    assert_eq!(typed.len(), 2, "{typed:?}");
    let row = |ty: &str| typed.iter().find(|r| r["type"] == ty).unwrap();
    assert_eq!(
        (row("i8")["entry"].as_str(), row("i8")["d_func"].as_u64(), row("i8")["d_skew"].as_u64()),
        (Some("Max"), Some(3), Some(2))
    );
    assert_eq!((row("f32")["d_func"].as_u64(), row("f32")["d_skew"].as_u64()), (Some(9), Some(4)));
    let max = t.iter().find(|r| r["entry"] == "Max" && r.get("type").is_none()).unwrap();
    assert_eq!(max["d_func"], c["timing"]["vxm"]["d_func"], "the entry's own row is the class row");
    let (_, plain, _) = info(&["info", "--profile", "ref"]);
    assert!(plain["timing"].as_array().unwrap().iter().all(|r| r.get("type").is_none()));
}
