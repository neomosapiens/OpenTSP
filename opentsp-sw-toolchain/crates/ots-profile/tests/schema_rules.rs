//! Each rule of the schema subset `tsp_config_check.c` supports, on small schemas: what a
//! configuration error says (`<key>: <reason>`), and which faults are the schema's
//! (`schema_error`, message `schema: ...`), so no constraint is ever skipped silently.
use ots_profile::schema::check_text;

fn check(schema: &str, cfg: &str) -> Result<(), (bool, String)> {
    check_text(cfg.as_bytes(), schema).map_err(|e| (e.schema_error, e.message))
}

fn cfg_err(schema: &str, cfg: &str) -> String {
    match check(schema, cfg) {
        Err((false, m)) => m,
        r => panic!("{cfg} against {schema}: {r:?}"),
    }
}

fn schema_err(schema: &str, cfg: &str) -> String {
    match check(schema, cfg) {
        Err((true, m)) => m,
        r => panic!("{cfg} against {schema}: {r:?}"),
    }
}

const OBJ: &str = r#"{"type": "object", "additionalProperties": false, "required": ["a"],
  "properties": {"a": {"type": "integer"}, "s": {"type": "string"}, "l": {"type": "array", "items": {"type": "integer"}},
                 "o": {"type": "object", "properties": {"x": {"type": "integer"}}}}}"#;

#[test]
fn objects_arrays_and_strings() {
    assert_eq!(check(OBJ, r#"{"a": 1, "s": "t", "l": [1, 2], "o": {"x": 3, "free": true}}"#), Ok(()));
    assert_eq!(cfg_err(OBJ, "[]"), "(top level): expected an object");
    assert_eq!(cfg_err(OBJ, r#"{"o": 1, "a": 1}"#), "o: expected an object");
    assert_eq!(cfg_err(OBJ, "{}"), "a: missing");
    assert_eq!(cfg_err(OBJ, r#"{"a": 1, "b": 2}"#), "b: unknown key");
    assert_eq!(cfg_err(OBJ, r#"{"a": 1, "a": 1}"#), "a: duplicate key");
    assert_eq!(cfg_err(OBJ, r#"{"a": 1, "o": {"x": 1, "x": 2}}"#), "o.x: duplicate key");
    assert_eq!(cfg_err(OBJ, r#"{"a": 1, "l": 3}"#), "l: expected an array");
    assert_eq!(cfg_err(OBJ, r#"{"a": 1, "l": [1, "2"]}"#), "l[1]: expected an integer (decimal digits, at most 18)");
    assert_eq!(cfg_err(OBJ, r#"{"a": 1, "s": 5}"#), "s: expected a string");
    assert_eq!(cfg_err(OBJ, "{\"a\": 1,"), "not valid JSON (at byte 8)");
    // an array without an items schema takes any element
    assert_eq!(check(r#"{"type": "array"}"#, r#"[1, "x", null]"#), Ok(()));
}

#[test]
fn integers_are_decimal_literals_of_at_most_18_digits() {
    let int = r#"{"type": "integer"}"#;
    for good in ["0", "-5", "123456789012345678", "-123456789012345678"] {
        assert_eq!(check(int, good), Ok(()), "{good}");
    }
    for bad in ["1.0", "1e3", "1234567890123456789", "\"1\"", "true", "-0.5"] {
        assert_eq!(cfg_err(int, bad), "(top level): expected an integer (decimal digits, at most 18)", "{bad}");
    }
}

#[test]
fn integer_bounds_const_and_multiple() {
    let sch = r#"{"type": "integer", "minimum": -2, "maximum": 40, "multipleOf": 4}"#;
    assert_eq!(check(sch, "-0"), Ok(()));
    assert_eq!(check(sch, "40"), Ok(()));
    assert_eq!(cfg_err(sch, "-4"), "(top level): -4 is below the minimum -2 (must be between -2 and 40)");
    assert_eq!(cfg_err(sch, "44"), "(top level): 44 is above the maximum 40 (must be between -2 and 40)");
    assert_eq!(cfg_err(sch, "6"), "(top level): 6 is not a multiple of 4");
    let c = r#"{"type": "integer", "const": 7}"#;
    assert_eq!(check(c, "7"), Ok(()));
    assert_eq!(cfg_err(c, "8"), "(top level): must be 7");
}

#[test]
fn malformed_integer_keywords_are_schema_errors() {
    for (kw, v) in [
        ("minimum", "1.5"),
        ("maximum", "\"9\""),
        ("const", "true"),
        ("multipleOf", "0"),
        ("multipleOf", "-3"),
        ("multipleOf", "2.0"),
    ] {
        let sch = format!(r#"{{"type": "integer", "{kw}": {v}}}"#);
        assert_eq!(schema_err(&sch, "4"), format!("schema: bad {kw} at (top level)"), "{kw} {v}");
    }
}

#[test]
fn string_const() {
    let sch = r#"{"type": "object", "properties": {"v": {"type": "string", "const": "0.1"}}}"#;
    assert_eq!(check(sch, r#"{"v": "0.1"}"#), Ok(()));
    assert_eq!(cfg_err(sch, r#"{"v": "0.2"}"#), "v: must be \"0.1\"");
    // a non-string const can never be met by a string
    assert_eq!(cfg_err(r#"{"type": "string", "const": 1}"#, r#""1""#), "(top level): must be \"?\"");
}

#[test]
fn booleans_are_true_or_false_only() {
    // `host.remote_csr` is the first boolean of a configuration (`tsp_config_check.c` `boolean`)
    let sch = r#"{"type": "object", "properties": {"b": {"type": "boolean"}}}"#;
    assert_eq!(check(sch, r#"{"b": true}"#), Ok(()));
    assert_eq!(check(sch, r#"{"b": false}"#), Ok(()));
    for bad in ["1", "0", "\"true\"", "null", "[]", "{}"] {
        assert_eq!(cfg_err(sch, &format!("{{\"b\": {bad}}}")), "b: expected true or false", "{bad}");
    }
    // a boolean is not an integer, and an integer is not a boolean
    assert_eq!(
        cfg_err(r#"{"type": "object", "properties": {"n": {"type": "integer"}}}"#, r#"{"n": false}"#),
        "n: expected an integer (decimal digits, at most 18)"
    );
}

#[test]
fn malformed_schemas_are_schema_errors() {
    assert_eq!(schema_err("{", "{}"), "schema: not valid JSON");
    assert_eq!(schema_err("[]", "{}"), "schema: not an object at ");
    assert_eq!(
        schema_err(r#"{"type": "object", "properties": {"a": 3}}"#, r#"{"a": 1}"#),
        "schema: not an object at a"
    );
    assert_eq!(schema_err(r#"{"type": 3}"#, "{}"), "schema: bad type at (top level)");
    assert_eq!(schema_err(r#"{"type": "number"}"#, "1"), "schema: unsupported type number at (top level)");
    assert_eq!(schema_err(r#"{"pattern": "x"}"#, "1"), "schema: unsupported keyword \"pattern\" at ");
    assert_eq!(schema_err(r#"{"type": "object", "required": [1]}"#, "{}"), "schema: bad required list at ");
    // the configuration is not checked against a schema that is not JSON, even when it is not JSON either
    assert_eq!(schema_err("{", "{"), "schema: not valid JSON");
}

#[test]
fn all_of_and_untyped_properties() {
    let sch = r#"{"allOf": [{"type": "integer", "minimum": 0}, {"type": "integer", "maximum": 3}]}"#;
    assert_eq!(check(sch, "2"), Ok(()));
    assert_eq!(cfg_err(sch, "-1"), "(top level): -1 is below the minimum 0");
    assert_eq!(cfg_err(sch, "4"), "(top level): 4 is above the maximum 3");
    // an untyped schema with properties checks the fields of an object (an anyOf alternative)
    let untyped = r#"{"properties": {"m": {"type": "integer", "const": 1}}, "required": ["m"]}"#;
    assert_eq!(check(untyped, r#"{"m": 1}"#), Ok(()));
    assert_eq!(cfg_err(untyped, r#"{"m": 2}"#), "m: must be 1");
    assert_eq!(cfg_err(untyped, "{}"), "m: missing");
}

#[test]
fn any_of_reports_the_first_failure_or_the_reason() {
    let alts = r#"[{"type": "integer", "const": 1}, {"type": "integer", "const": 2}]"#;
    let plain = format!(r#"{{"anyOf": {alts}}}"#);
    assert_eq!(check(&plain, "1"), Ok(()));
    assert_eq!(check(&plain, "2"), Ok(()));
    // alternatives that are all integer constants: the set is named
    assert_eq!(cfg_err(&plain, "3"), "(top level): must be one of 1, 2");
    // a range names both of its ends
    let range = r#"{"type": "integer", "minimum": 1, "maximum": 32}"#;
    assert_eq!(cfg_err(range, "33"), "(top level): 33 is above the maximum 32 (must be between 1 and 32)");
    assert_eq!(cfg_err(range, "0"), "(top level): 0 is below the minimum 1 (must be between 1 and 32)");
    // a bound on one side only is named alone
    assert_eq!(cfg_err(r#"{"type": "integer", "minimum": 1}"#, "0"), "(top level): 0 is below the minimum 1");
    let why = format!(r#"{{"anyOf": {alts}, "x-opentsp-why": "one or two"}}"#);
    assert_eq!(cfg_err(&why, "3"), "(top level): one or two");
    // a schema fault inside an alternative is not hidden by a later alternative that holds
    let broken = r#"{"anyOf": [{"type": "integer", "minimum": "x"}, {"type": "integer"}]}"#;
    assert_eq!(schema_err(broken, "1"), "schema: bad minimum at (top level)");
    // once an alternative holds, the later ones (even broken ones) are not tried
    let early = r#"{"anyOf": [{"type": "integer"}, {"type": "integer", "minimum": "x"}]}"#;
    assert_eq!(check(early, "1"), Ok(()));
}

fn rel(entry: &str) -> String {
    format!(r#"{{"type": "object", "x-opentsp-relations": [{entry}]}}"#)
}

const CFG: &str = r#"{"p": {"a": 3, "b": 5}, "geometry": {"mem_slices_per_hemisphere": 8}, "t": "x"}"#;

#[test]
fn relations_hold_or_name_the_left_key() {
    let r = |op: &str, extra: &str| rel(&format!(r#"{{"a": "p.a", "op": "{op}", "b": "p.b", "why": "w"{extra}}}"#));
    for (op, extra) in [
        ("<", ""),
        ("<=", ""),
        ("!=", ""),
        ("==", r#", "offset": -2"#),
        ("<", r#", "offset": -1"#),
        ("==", r#", "quads": -1, "offset": 0"#),
    ] {
        assert_eq!(check(&r(op, extra), CFG), Ok(()), "{op}{extra}");
    }
    assert_eq!(cfg_err(&r("==", ""), CFG), "p.a: must be == p.b (p.a = 3, p.b = 5): w");
    assert_eq!(cfg_err(&r("<", r#", "offset": -2"#), CFG), "p.a: must be < p.b -2 (p.a = 3, p.b = 5): w");
    assert_eq!(cfg_err(&r("<=", r#", "offset": -3"#), CFG), "p.a: must be <= p.b -3 (p.a = 3, p.b = 5): w");
    assert_eq!(cfg_err(&r("!=", r#", "offset": -2"#), CFG), "p.a: must be != p.b -2 (p.a = 3, p.b = 5): w");
    // quads: k adds k * (mem_slices_per_hemisphere / 4) = 2 k
    assert_eq!(cfg_err(&r("<", r#", "quads": -1"#), CFG), "p.a: must be < p.b -2 (p.a = 3, p.b = 5): w");
    assert_eq!(cfg_err(&r("!=", r#", "quads": 1, "offset": -4"#), CFG), "p.a: must be != p.b -2 (p.a = 3, p.b = 5): w");
    // no relations at all
    assert_eq!(check(r#"{"type": "object"}"#, CFG), Ok(()));
}

#[test]
fn malformed_relations_are_schema_errors() {
    let bad = "schema: bad x-opentsp-relations entry";
    assert_eq!(schema_err(&rel(r#"{"a": "p.a", "op": "<", "b": "p.b"}"#), CFG), bad);
    assert_eq!(schema_err(&rel(r#"{"a": "p.a", "op": "<", "b": 1, "why": "w"}"#), CFG), bad);
    assert_eq!(schema_err(&rel(r#"{"a": "p.a", "op": "<", "b": "p.b", "why": "w", "offset": "1"}"#), CFG), bad);
    assert_eq!(schema_err(&rel(r#"{"a": "p.a", "op": "<", "b": "p.b", "why": "w", "quads": 0.5}"#), CFG), bad);
    assert_eq!(
        schema_err(
            &rel(r#"{"a": "p.a", "op": "<", "b": "p.b", "why": "w", "quads": 1}"#),
            r#"{"p": {"a": 3, "b": 5}}"#
        ),
        "schema: a quads relation needs geometry.mem_slices_per_hemisphere"
    );
    assert_eq!(
        schema_err(&rel(r#"{"a": "p.a", "op": ">", "b": "p.b", "why": "w"}"#), CFG),
        "schema: bad relation operator >"
    );
    for (a, b) in [("p.c", "p.b"), ("t", "p.b"), ("p.a", "t.z"), ("p", "p.b")] {
        let r = rel(&format!(r#"{{"a": "{a}", "op": "<", "b": "{b}", "why": "w"}}"#));
        assert_eq!(schema_err(&r, CFG), format!("schema: relation names a non-integer key {a} / {b}"), "{a} {b}");
    }
}
