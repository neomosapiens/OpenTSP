//! Ports of `opentsp-arch-spec/test/config/config_negative.py`: every mutation of a shipped
//! profile that breaks one invariant must be refused, with a message that names the key. The
//! shipped profiles must validate. Run on both shipped profiles.
use ots_profile::{validate_bytes, Profile, SCHEMA_JSON};
use serde_json::{json, Value};

fn profile_value(name: &str) -> Value {
    let p = Profile::load(name).unwrap();
    serde_json::from_slice(p.bytes()).unwrap()
}

fn setk(cfg: &mut Value, key: &str, value: Value) {
    let mut node = cfg;
    let parts: Vec<&str> = key.split('.').collect();
    for k in &parts[..parts.len() - 1] {
        node = &mut node[*k];
    }
    node[parts[parts.len() - 1]] = value;
}

fn delk(cfg: &mut Value, key: &str) {
    let mut node = cfg;
    let parts: Vec<&str> = key.split('.').collect();
    for k in &parts[..parts.len() - 1] {
        node = &mut node[*k];
    }
    node.as_object_mut().unwrap().remove(parts[parts.len() - 1]);
}

fn getk<'a>(cfg: &'a Value, key: &str) -> &'a Value {
    key.split('.').fold(cfg, |n, k| &n[k])
}

fn int(cfg: &Value, key: &str) -> i64 {
    getk(cfg, key).as_i64().unwrap()
}

/// Run the validator on a value; replace the placeholder string `"@@2^70@@"` by the literal 2^70.
fn check(cfg: &Value) -> Result<(), String> {
    let text = serde_json::to_string(cfg).unwrap().replace("\"@@2^70@@\"", "1180591620717411303424");
    validate_bytes(text.as_bytes()).map_err(|e| e.to_string())
}

fn expect_reject(name: &str, key: &str, result: Result<(), String>) {
    match result {
        Ok(()) => panic!("{name}: accepted, expected a rejection naming {key}"),
        Err(m) => {
            assert!(m.contains(key), "{name}: message {m:?} does not name {key:?}");
            assert!(!m.contains('\n'), "{name}: message has several lines: {m:?}");
        }
    }
}

type Mutation = Box<dyn Fn(&mut Value)>;

fn set(key: &'static str, v: Value) -> Mutation {
    Box::new(move |c| setk(c, key, v.clone()))
}

fn override_row(entry: &'static str, d_func: i64, d_skew: i64) -> Mutation {
    Box::new(move |c| {
        c["timing"]["overrides"]
            .as_array_mut()
            .unwrap()
            .push(json!({"entry": entry, "d_func": d_func, "d_skew": d_skew}))
    })
}

fn cases(base: &Value) -> Vec<(String, String, Mutation)> {
    let wps = int(base, "sram.words_per_slice");
    let other_wps = if wps == 256 { 8192 } else { 256 };
    let other_bank = if wps == 256 { 12 } else { 7 };
    let sl = int(base, "geometry.superlanes");
    let c = |n: &str, k: &str, m: Mutation| (n.to_string(), k.to_string(), m);
    let mut v: Vec<(String, String, Mutation)> = vec![
        c("d_func negative", "timing.mem.d_func", set("timing.mem.d_func", json!(-3))),
        c("d_func 64", "timing.vxm.d_func", set("timing.vxm.d_func", json!(64))),
        c("d_skew 40", "timing.vxm.d_skew", set("timing.vxm.d_skew", json!(40))),
        c("superlanes 0", "geometry.superlanes", set("geometry.superlanes", json!(0))),
        c("superlanes 33", "geometry.superlanes", set("geometry.superlanes", json!(33))),
        c("superlanes 40", "geometry.superlanes", set("geometry.superlanes", json!(40))),
        c("release_latency negative", "barrier.release_latency", set("barrier.release_latency", json!(-3))),
        c("release_latency 2^70", "barrier.release_latency", set("barrier.release_latency", json!("@@2^70@@"))),
        c("bank_bit 20", "sram.bank_bit", set("sram.bank_bit", json!(20))),
        c("bank_bit of the other profile", "sram.bank_bit", set("sram.bank_bit", json!(other_bank))),
        c("words_per_slice of the other profile", "sram.bank_bit", set("sram.words_per_slice", json!(other_wps))),
        c("words_per_slice 100", "sram.words_per_slice", set("sram.words_per_slice", json!(100))),
        c("words_per_slice 0", "sram.words_per_slice", set("sram.words_per_slice", json!(0))),
        c("vxm_gap 99", "layout.vxm_gap", set("layout.vxm_gap", json!(99))),
        c("mem_up_gap_base 0 (quads unreachable)", "layout.c2c_rx_gap", set("layout.mem_up_gap_base", json!(0))),
        c("vxm_gap outside the MEM gaps", "layout.mem_up_gap_base", set("layout.vxm_gap", json!(10))),
        c("link_latency 0", "c2c.link_latency", set("c2c.link_latency", json!(0))),
        c("link_latency negative", "c2c.link_latency", set("c2c.link_latency", json!(-100))),
        c("link_latency a string", "c2c.link_latency", set("c2c.link_latency", json!("5"))),
        c("link_latency a fraction", "c2c.link_latency", set("c2c.link_latency", json!(1.5))),
        c("link_latency missing", "c2c.link_latency", Box::new(|c| delk(c, "c2c.link_latency"))),
        c("hac_period 0", "c2c.hac_period", set("c2c.hac_period", json!(0))),
        c("hac_period negative", "c2c.hac_period", set("c2c.hac_period", json!(-5))),
        c("links 0", "c2c.links", set("c2c.links", json!(0))),
        c("links 129", "c2c.links", set("c2c.links", json!(129))),
        c("links -1", "c2c.links", set("c2c.links", json!(-1))),
        c("mem d_func 0", "timing.mem.d_func", set("timing.mem.d_func", json!(0))),
        c("vxm d_func 0", "timing.vxm.d_func", set("timing.vxm.d_func", json!(0))),
        c("c2c d_func 0", "timing.c2c.d_func", set("timing.c2c.d_func", json!(0))),
        c("mxm_acc d_func 0", "timing.mxm_acc.d_func", set("timing.mxm_acc.d_func", json!(0))),
        c("sxm_cross d_func 0", "timing.sxm_cross.d_func", set("timing.sxm_cross.d_func", json!(0))),
        c("link_latency_min 0", "c2c.link_latency_min", set("c2c.link_latency_min", json!(0))),
        c("link_latency_max 0", "c2c.link_latency_max", set("c2c.link_latency_max", json!(0))),
        c("link_latency_max 2^16", "c2c.link_latency_max", set("c2c.link_latency_max", json!(65536))),
        c("link_latency_min above link_latency", "c2c.link_latency_min", set("c2c.link_latency_min", json!(500))),
        c("link_latency above link_latency_max", "c2c.link_latency", set("c2c.link_latency", json!(229))),
        c(
            "link_latency_max not below hac_period",
            "c2c.link_latency_max",
            Box::new(|c| {
                setk(c, "c2c.link_latency", json!(230));
                setk(c, "c2c.link_latency_max", json!(300));
            }),
        ),
        c("hac_period 100 (below the link latency)", "c2c.link_latency_max", set("c2c.hac_period", json!(100))),
        c("spec_version 9.9", "spec_version", set("spec_version", json!("9.9"))),
        c("unknown top-level key", "extra", set("extra", json!(1))),
        c("unknown nested key", "timing.vxm.extra", set("timing.vxm.extra", json!(1))),
        c("geometry missing", "geometry", Box::new(|c| delk(c, "geometry"))),
        // timing.overrides
        c(
            "override of an unknown entry",
            "timing.overrides: Frobnicate is not a catalogue entry",
            override_row("Frobnicate", 5, 0),
        ),
        c(
            "override listed twice",
            "timing.overrides: Add is listed more than once",
            Box::new(|c| {
                override_row("Add", 5, 2)(c);
                override_row("Add", 6, 2)(c);
            }),
        ),
        c("override d_func 64", "timing.overrides", override_row("Add", 64, 0)),
        c("override d_skew 16", "timing.overrides", override_row("Add", 5, 16)),
        c(
            "override with an unknown key",
            "timing.overrides",
            Box::new(|c| {
                c["timing"]["overrides"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"entry": "Add", "d_func": 5, "d_skew": 0, "x": 1}))
            }),
        ),
        c(
            "override d_skew equal to d_func",
            "timing.overrides.Add: Add: d_skew 3 must be below d_func 3",
            override_row("Add", 3, 3),
        ),
        c(
            "override of ACC with d_skew above d_func",
            "timing.overrides.ACC: ACC: d_skew 9 must be below d_func 4",
            override_row("ACC", 4, 9),
        ),
        c(
            "Write d_skew differs from Read's",
            "timing.overrides.Write: Write: d_skew 1 must equal Read's",
            override_row("Write", 5, 1),
        ),
        c(
            "mem d_skew without IfetchM's",
            "timing.cu: IfetchM: d_skew 0 must equal Read's 1",
            set("timing.mem.d_skew", json!(1)),
        ),
        // the timing and front-end keys added to the profile: each is required and bounded
        c("fetch_staging_depth 0", "icu.fetch_staging_depth", set("icu.fetch_staging_depth", json!(0))),
        c("fetch_staging_depth 65", "icu.fetch_staging_depth", set("icu.fetch_staging_depth", json!(65))),
        c("fetch_staging_depth missing", "icu.fetch_staging_depth", Box::new(|c| delk(c, "icu.fetch_staging_depth"))),
        c("sxm_map_ready 0", "timing.sxm_map_ready", set("timing.sxm_map_ready", json!(0))),
        c("sxm_map_ready 64", "timing.sxm_map_ready", set("timing.sxm_map_ready", json!(64))),
        c("sxm_map_ready missing", "timing.sxm_map_ready", Box::new(|c| delk(c, "timing.sxm_map_ready"))),
        c("occupancy_f16 0", "timing.mxm_abc.occupancy_f16", set("timing.mxm_abc.occupancy_f16", json!(0))),
        c("occupancy_f16 64", "timing.mxm_abc.occupancy_f16", set("timing.mxm_abc.occupancy_f16", json!(64))),
        c(
            "occupancy_f16 missing",
            "timing.mxm_abc.occupancy_f16",
            Box::new(|c| delk(c, "timing.mxm_abc.occupancy_f16")),
        ),
        c(
            "occupancy_f16 above the ABC class d_func",
            "timing.mxm_abc.occupancy_f16",
            Box::new(|c| {
                let d = int(c, "timing.mxm_abc.d_func");
                setk(c, "timing.mxm_abc.occupancy_f16", json!(d + 1));
            }),
        ),
        c("rtdeskew_floor 0", "timing.c2c.rtdeskew_floor", set("timing.c2c.rtdeskew_floor", json!(0))),
        c("rtdeskew_floor missing", "timing.c2c.rtdeskew_floor", Box::new(|c| delk(c, "timing.c2c.rtdeskew_floor"))),
        c("cycles_per_vector 0", "c2c.cycles_per_vector", set("c2c.cycles_per_vector", json!(0))),
        // the host object: required, bounded, and `remote_csr` is a JSON boolean
        c("host missing", "host", Box::new(|c| delk(c, "host"))),
        c("host.cycles_per_vector 0", "host.cycles_per_vector", set("host.cycles_per_vector", json!(0))),
        c("host.cycles_per_vector 65", "host.cycles_per_vector", set("host.cycles_per_vector", json!(65))),
        c("host.ingress_depth 0", "host.ingress_depth", set("host.ingress_depth", json!(0))),
        c("host.ingress_depth 257", "host.ingress_depth", set("host.ingress_depth", json!(257))),
        c("host.doorbells 0", "host.doorbells", set("host.doorbells", json!(0))),
        c("host.doorbells 65", "host.doorbells", set("host.doorbells", json!(65))),
        // `host.chip` is retired (every chip has its own host interface): a profile that carries it is rejected as an unknown key
        c("host.chip retired", "host.chip", set("host.chip", json!(0))),
        c("host.chip retired, any value", "host.chip", set("host.chip", json!(64))),
        c("host.remote_csr an integer", "host.remote_csr", set("host.remote_csr", json!(1))),
        c("host.remote_csr a string", "host.remote_csr", set("host.remote_csr", json!("true"))),
        c("host.remote_csr missing", "host.remote_csr", Box::new(|c| delk(c, "host.remote_csr"))),
        c("host.extra", "host.extra", set("host.extra", json!(1))),
    ];
    if sl > 1 {
        v.push(c(
            "Shift override below the superlane count",
            "timing.overrides.Shift: Shift: d_func",
            override_row("Shift", sl - 1, 0),
        ));
        v.push(c(
            "ABC override below the superlane count",
            "timing.overrides.ABC: ABC: d_func",
            override_row("ABC", sl - 1, 0),
        ));
        v.push(c(
            "mxm_abc d_func below the superlane count",
            "timing.mxm_abc.d_func",
            set("timing.mxm_abc.d_func", json!(sl - 1)),
        ));
        v.push(c(
            "sxm_cross d_func below the superlane count",
            "timing.sxm_cross.d_func",
            set("timing.sxm_cross.d_func", json!(sl - 1)),
        ));
    }
    v
}

fn each_profile(mut f: impl FnMut(&str, &Value)) {
    for name in ["ref", "mini"] {
        f(name, &profile_value(name));
    }
}

#[test]
fn shipped_profiles_validate() {
    for name in ["ref", "mini"] {
        let p = Profile::load(name).unwrap();
        p.validate().unwrap_or_else(|e| panic!("{name}: {e}"));
        validate_bytes(p.bytes()).unwrap();
    }
}

#[test]
fn each_config_negative_case_is_refused_naming_the_key() {
    let mut n = 0;
    each_profile(|profile, base| {
        for (name, key, mutate) in cases(base) {
            let mut cfg = base.clone();
            mutate(&mut cfg);
            expect_reject(&format!("{profile}: {name}"), &key, check(&cfg));
            n += 1;
        }
    });
    assert!(n >= 100, "only {n} cases ran");
}

#[test]
fn raw_texts_that_are_not_configurations() {
    let cases: [(&str, &str, &[u8]); 4] = [
        ("not an object", "(top level)", b"[]"),
        ("not JSON", "not valid JSON", b"{"),
        ("empty file", "not valid JSON", b""),
        ("a NUL byte", "NUL byte", b"{\0}"),
    ];
    for (name, key, text) in cases {
        expect_reject(name, key, validate_bytes(text).map_err(|e| e.to_string()));
    }
    each_profile(|_, base| {
        let body = serde_json::to_string(base).unwrap();
        let dup = format!("{}, \"spec_version\": \"0.1\"}}", &body[..body.len() - 1]);
        expect_reject("duplicate key", "spec_version", validate_bytes(dup.as_bytes()).map_err(|e| e.to_string()));
        let trailing = format!("{body} x");
        expect_reject(
            "trailing text",
            "not valid JSON",
            validate_bytes(trailing.as_bytes()).map_err(|e| e.to_string()),
        );
    });
}

#[test]
fn legal_timing_overrides_load() {
    each_profile(|_, base| {
        let mut legal = base.clone();
        legal["timing"]["mem"]["d_skew"] = json!(1);
        let sl = int(base, "geometry.superlanes").max(1);
        legal["timing"]["overrides"] = json!([
            {"entry": "Add", "d_func": 9, "d_skew": 3},
            {"entry": "IfetchM", "d_func": 0, "d_skew": 1},
            {"entry": "Shift", "d_func": sl, "d_skew": 0}
        ]);
        check(&legal).unwrap();
    });
}

// ---- h6: the boundary of every x-opentsp-relations entry of the schema -------------------

fn in_bounds(cfg: &Value, node: &Value) -> bool {
    match node.get("type").and_then(Value::as_str) {
        Some("object") => node["properties"]
            .as_object()
            .is_none_or(|props| props.iter().all(|(k, v)| cfg.get(k).is_none_or(|c| in_bounds(c, v)))),
        Some("integer") => {
            let x = cfg.as_i64().unwrap();
            node.get("minimum").and_then(Value::as_i64).is_none_or(|m| x >= m)
                && node.get("maximum").and_then(Value::as_i64).is_none_or(|m| x <= m)
        }
        _ => true,
    }
}

/// A relation's offset: `offset`, plus `quads` times the MEM quads per hemisphere.
fn eff(cfg: &Value, r: &Value) -> i64 {
    let quads = r.get("quads").and_then(Value::as_i64).unwrap_or(0);
    r["offset"].as_i64().unwrap()
        + if quads != 0 { quads * (int(cfg, "geometry.mem_slices_per_hemisphere") / 4) } else { 0 }
}

fn holds(cfg: &Value, r: &Value) -> bool {
    let (a, b, o) = (int(cfg, r["a"].as_str().unwrap()), int(cfg, r["b"].as_str().unwrap()), eff(cfg, r));
    match r["op"].as_str().unwrap() {
        "<" => a < b + o,
        "<=" => a <= b + o,
        "==" => a == b + o,
        _ => a != b + o,
    }
}

/// IfetchM shares the SRAM ports' d_skew with the MEM class: give it an override when the MEM
/// class's d_skew moves.
fn sram_skew(cfg: &mut Value) {
    let s = int(cfg, "timing.mem.d_skew");
    if s != int(cfg, "timing.cu.d_skew") {
        cfg["timing"]["overrides"] = json!([{"entry": "IfetchM", "d_func": 0, "d_skew": s}]);
    }
}

fn at_bound(base: &Value, schema: &Value, r: &Value, delta: i64) -> Option<(Value, String)> {
    let edge: i64 = if r["op"] == "<=" { 0 } else { -1 };
    let (a, b, off) = (r["a"].as_str().unwrap(), r["b"].as_str().unwrap(), eff(base, r));
    let moves: Vec<Option<i64>> = vec![None, None].into_iter().chain((0..64).map(Some)).collect();
    for (i, mv) in moves.iter().enumerate() {
        let mut cfg = base.clone();
        let label = match (i, mv) {
            (0, _) => {
                let v = int(&cfg, b) + off + edge + delta;
                setk(&mut cfg, a, json!(v));
                "a".to_string()
            }
            (1, _) => {
                let v = int(&cfg, a) - off - edge - delta;
                setk(&mut cfg, b, json!(v));
                "b".to_string()
            }
            (_, Some(v)) => {
                setk(&mut cfg, b, json!(v));
                setk(&mut cfg, a, json!(v + off + edge + delta));
                "a and b".to_string()
            }
            _ => unreachable!(),
        };
        sram_skew(&mut cfg);
        if a == "sram.bank_bit" {
            // the bank bit is tied to the words by an alternative of the schema: keep them consistent, so that only the relation is tested
            let b = int(&cfg, a);
            if (7..=15).contains(&b) {
                setk(&mut cfg, "sram.words_per_slice", json!(1i64 << (b + 1)));
            }
        }
        let others_hold =
            schema["x-opentsp-relations"].as_array().unwrap().iter().filter(|x| *x != r).all(|x| holds(&cfg, x));
        if in_bounds(&cfg, schema) && others_hold {
            return Some((cfg, label));
        }
    }
    None
}

#[test]
fn every_schema_relation_is_checked_at_its_boundary() {
    let schema: Value = serde_json::from_str(SCHEMA_JSON).unwrap();
    // Isolate each schema relation. Effective timing has additional causality
    // constraints, tested through validate_bytes in the directed timing tests.
    let check_relation = |cfg: &Value| {
        ots_profile::schema::check_text(&serde_json::to_vec(cfg).unwrap(), SCHEMA_JSON).map_err(|e| e.to_string())
    };
    each_profile(|profile, base| {
        for r in schema["x-opentsp-relations"].as_array().unwrap() {
            let name = format!("{profile}: {} {} {} {:+}", r["a"], r["op"], r["b"], eff(base, r));
            let at = at_bound(base, &schema, r, 0);
            let a = r["a"].as_str().unwrap();
            let (cfg, mv) = match at {
                Some(x) => x,
                None if a.ends_with(".d_skew") && at_bound(base, &schema, r, 1).is_none() => continue, // implied by d_skew's maximum
                None => panic!("{name}: no config at the bound"),
            };
            check_relation(&cfg).unwrap_or_else(|e| panic!("{name} at its bound (moved {mv}): {e}"));
            let (cfg, mv) =
                at_bound(base, &schema, r, 1).unwrap_or_else(|| panic!("{name}: no config one past the bound"));
            expect_reject(&format!("{name} one past its bound (moved {mv})"), a, check_relation(&cfg));
        }
    });
}

#[test]
fn an_unsupported_schema_keyword_is_a_schema_error() {
    let schema = r#"{"type": "object", "pattern": "x"}"#;
    let e = ots_profile::schema::check_text(b"{}", schema).unwrap_err();
    assert!(e.to_string().starts_with("schema: unsupported keyword \"pattern\""), "{e}");
}

#[test]
fn an_abc_override_below_occupancy_f16_is_refused_naming_the_override_row() {
    // the schema relation holds the class row only; `timing_config_error` holds the effective ABC row
    // (`model/postlude/timing_check.sail`), so an override that commits before the f16 plane frees fails there
    let mut base = profile_value("mini");
    setk(&mut base, "timing.mxm_abc.occupancy_f16", json!(2));
    let sl = int(&base, "geometry.superlanes");
    let occ = int(&base, "timing.mxm_abc.occupancy_f16");
    assert!(sl < occ, "the mini profile must have fewer superlanes than the f16 occupancy for this case");
    let mut cfg = base.clone();
    override_row("ABC", sl, 0)(&mut cfg);
    expect_reject(
        "ABC override d_func below occupancy_f16",
        "timing.mxm_abc.occupancy_f16: 2 must be at most ABC's d_func 1",
        check(&cfg),
    );
    let mut ok = base.clone();
    override_row("ABC", occ, 0)(&mut ok);
    check(&ok).unwrap();
}

// ---- per-type timing rows: an override row's entry is `<Entry>` or `<Entry>.<element type>` -----------

#[test]
fn per_type_timing_rows_load() {
    each_profile(|_, base| {
        let mut cfg = base.clone();
        for entry in ["Max.i8", "Max.f32", "Max", "Add.e4m3", "Exp.f16", "Reduce.u32", "Conv_F16_I8.f16", "Convert.i8"]
        {
            override_row(entry, 9, 3)(&mut cfg);
        }
        check(&cfg).unwrap();
    });
}

#[test]
fn a_wrong_per_type_timing_row_is_refused_naming_the_row() {
    each_profile(|_, base| {
        let cases: [(&str, i64, i64, &str); 7] = [
            ("Read.i8", 5, 0, "timing.overrides: Read.i8: Read does not take element type i8"),
            ("Max.b8", 5, 2, "timing.overrides: Max.b8: Max does not take element type b8"),
            ("Exp.i32", 5, 2, "timing.overrides: Exp.i32: Exp does not take element type i32"),
            ("Conv_F16_I8.i8", 5, 2, "timing.overrides: Conv_F16_I8.i8: Conv_F16_I8 does not take element type i8"),
            ("Max.q9", 5, 2, "timing.overrides: Max.q9 is not a catalogue entry"),
            ("Max.", 5, 2, "timing.overrides: Max. is not a catalogue entry"),
            ("Max.i8", 2, 2, "timing.overrides.Max.i8: Max.i8: d_skew 2 must be below d_func 2"),
        ];
        for (entry, f, k, msg) in cases {
            let mut cfg = base.clone();
            override_row(entry, f, k)(&mut cfg);
            expect_reject(entry, msg, check(&cfg));
        }
        let mut cfg = base.clone();
        override_row("Max.i8", 3, 2)(&mut cfg);
        override_row("Max.i8", 4, 2)(&mut cfg);
        expect_reject("duplicate typed row", "timing.overrides: Max.i8 is listed more than once", check(&cfg));
    });
}
