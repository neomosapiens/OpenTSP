//! Configuration validation against `config/schema.json`: a port of
//! `opentsp-arch-spec/c_emulator/tsp_config_check.c`. It supports the JSON Schema subset the
//! schema generator writes (type object, array, integer, string, boolean; items, properties, required,
//! additionalProperties false, const, minimum, maximum, multipleOf, anyOf, allOf, default) and two
//! opentsp keywords: `x-opentsp-relations` and `x-opentsp-why`. An unknown keyword in the
//! schema is an error ("schema: ..."), so no constraint is ever ignored silently.
//!
//! Messages have the form `<key>: <reason>`, as the C checker's, which the tests (and the
//! emulator's negative tests) match on the key.
use crate::json::{self, Json};
use std::fmt;

/// A rejected configuration (or schema): the message `<key>: <reason>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError {
    pub message: String,
    /// The schema, not the configuration, is at fault.
    pub schema_error: bool,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ConfigError {}

fn fail<T>(message: String) -> Result<T, ConfigError> {
    Err(ConfigError { message, schema_error: false })
}

fn schema_fail<T>(message: String) -> Result<T, ConfigError> {
    Err(ConfigError { message, schema_error: true })
}

/// An integer as Sail reads it: an optional '-' and decimal digits only, at most 18 of them.
fn int_value(v: Option<&Json>) -> Option<i64> {
    let Some(Json::Num(s)) = v else { return None };
    let d = s.strip_prefix('-').unwrap_or(s);
    if d.is_empty() || d.len() > 18 || !d.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

const KNOWN: &[&str] = &[
    "$schema",
    "type",
    "properties",
    "items",
    "required",
    "additionalProperties",
    "const",
    "minimum",
    "maximum",
    "multipleOf",
    "anyOf",
    "allOf",
    "description",
    "default",
    "x-opentsp-relations",
    "x-opentsp-why",
];

type R = Result<(), ConfigError>;

fn dotted(key: &str, name: &str) -> String {
    if key.is_empty() {
        name.to_string()
    } else {
        format!("{key}.{name}")
    }
}

fn top(key: &str) -> &str {
    if key.is_empty() {
        "(top level)"
    } else {
        key
    }
}

fn check_keywords(sch: &Json, key: &str) -> R {
    if let Json::Obj(m) = sch {
        for (k, _) in m {
            if !KNOWN.contains(&k.as_str()) {
                return schema_fail(format!("schema: unsupported keyword \"{k}\" at {key}"));
            }
        }
    }
    Ok(())
}

fn check_object(sch: &Json, v: &Json, key: &str) -> R {
    let Json::Obj(fields) = v else {
        return fail(format!("{}: expected an object", top(key)));
    };
    let props = sch.get("properties");
    let req = sch.get("required");
    let addl = sch.get("additionalProperties");
    for (n, (name, f)) in fields.iter().enumerate() {
        if fields[..n].iter().any(|(g, _)| g == name) {
            return fail(format!("{}: duplicate key", dotted(key, name)));
        }
        let sub = dotted(key, name);
        match props.and_then(|p| p.get(name)) {
            None => {
                if addl == Some(&Json::Bool(false)) {
                    return fail(format!("{sub}: unknown key"));
                }
            }
            Some(ps) => check(ps, f, &sub)?,
        }
    }
    if let Some(Json::Arr(req)) = req {
        for r in req {
            let Json::Str(r) = r else {
                return schema_fail(format!("schema: bad required list at {key}"));
            };
            if v.get(r).is_none() {
                return fail(format!("{}: missing", dotted(key, r)));
            }
        }
    }
    Ok(())
}

fn check(sch: &Json, v: &Json, key: &str) -> R {
    if !matches!(sch, Json::Obj(_)) {
        return schema_fail(format!("schema: not an object at {key}"));
    }
    check_keywords(sch, key)?;
    let k = top(key);
    if let Some(ty) = sch.get("type") {
        let Some(t) = ty.as_str() else {
            return schema_fail(format!("schema: bad type at {k}"));
        };
        match t {
            "object" => check_object(sch, v, key)?,
            "array" => {
                let Json::Arr(items) = v else {
                    return fail(format!("{k}: expected an array"));
                };
                if let Some(it) = sch.get("items") {
                    for (n, e) in items.iter().enumerate() {
                        check(it, e, &format!("{key}[{n}]"))?;
                    }
                }
            }
            "integer" => {
                let Some(x) = int_value(Some(v)) else {
                    return fail(format!("{k}: expected an integer (decimal digits, at most 18)"));
                };
                if let Some(mn) = sch.get("minimum") {
                    let Some(b) = int_value(Some(mn)) else {
                        return schema_fail(format!("schema: bad minimum at {k}"));
                    };
                    if x < b {
                        if let Some(hi) = sch.get("maximum").and_then(|m| int_value(Some(m))) {
                            return fail(format!("{k}: {x} is below the minimum {b} (must be between {b} and {hi})"));
                        }
                        return fail(format!("{k}: {x} is below the minimum {b}"));
                    }
                }
                if let Some(mx) = sch.get("maximum") {
                    let Some(b) = int_value(Some(mx)) else {
                        return schema_fail(format!("schema: bad maximum at {k}"));
                    };
                    if x > b {
                        if let Some(lo) = sch.get("minimum").and_then(|m| int_value(Some(m))) {
                            return fail(format!("{k}: {x} is above the maximum {b} (must be between {lo} and {b})"));
                        }
                        return fail(format!("{k}: {x} is above the maximum {b}"));
                    }
                }
                if let Some(cs) = sch.get("const") {
                    let Some(b) = int_value(Some(cs)) else { return schema_fail(format!("schema: bad const at {k}")) };
                    if x != b {
                        return fail(format!("{k}: must be {b}"));
                    }
                }
                if let Some(mo) = sch.get("multipleOf") {
                    match int_value(Some(mo)) {
                        Some(b) if b > 0 => {
                            if x % b != 0 {
                                return fail(format!("{k}: {x} is not a multiple of {b}"));
                            }
                        }
                        _ => return schema_fail(format!("schema: bad multipleOf at {k}")),
                    }
                }
            }
            "string" => {
                let Json::Str(s) = v else {
                    return fail(format!("{k}: expected a string"));
                };
                if let Some(cs) = sch.get("const") {
                    match cs {
                        Json::Str(c) if c == s => {}
                        Json::Str(c) => return fail(format!("{k}: must be \"{c}\"")),
                        _ => return fail(format!("{k}: must be \"?\"")),
                    }
                }
            }
            "boolean" => {
                if !matches!(v, Json::Bool(_)) {
                    return fail(format!("{k}: expected true or false"));
                }
            }
            other => return schema_fail(format!("schema: unsupported type {other} at {k}")),
        }
    }
    // An untyped schema with properties (an anyOf alternative) constrains an object's fields.
    if sch.get("type").is_none() && sch.get("properties").is_some() {
        check_object(sch, v, key)?;
    }
    if let Some(Json::Arr(all)) = sch.get("allOf") {
        for s in all {
            check(s, v, key)?;
        }
    }
    if let Some(Json::Arr(any)) = sch.get("anyOf") {
        let mut first = String::new();
        let mut ok = false;
        for s in any {
            if ok {
                break;
            }
            match check(s, v, key) {
                Ok(()) => ok = true,
                Err(e) if e.schema_error => return Err(e),
                Err(e) => {
                    if first.is_empty() {
                        first = e.message;
                    }
                }
            }
        }
        if !ok {
            if let Some(Json::Str(why)) = sch.get("x-opentsp-why") {
                return fail(format!("{k}: {why}"));
            }
            // Alternatives that are all integer constants: name the set.
            let consts: Option<Vec<i64>> =
                any.iter().map(|s| s.get("const").and_then(|c| int_value(Some(c)))).collect();
            if let Some(list) = consts.filter(|l| !l.is_empty()) {
                let names: Vec<String> = list.iter().map(|b| b.to_string()).collect();
                return fail(format!("{k}: must be one of {}", names.join(", ")));
            }
            return fail(format!("{k}: none of the allowed alternatives ({first})"));
        }
    }
    Ok(())
}

fn lookup<'a>(root: &'a Json, dotted: &str) -> Option<&'a Json> {
    let mut v = root;
    for p in dotted.split('.').filter(|p| !p.is_empty()) {
        if !matches!(v, Json::Obj(_)) {
            return None;
        }
        v = v.get(p)?;
    }
    Some(v)
}

/// `x-opentsp-relations`: `[{"a": key, "op": "<"|"<="|"=="|"!=", "b": key, "offset": n, "why": text}]`:
/// value(a) op value(b) + offset (+ `quads` * MEM quads per hemisphere).
fn check_relations(sch: &Json, cfg: &Json) -> R {
    let Some(Json::Arr(rels)) = sch.get("x-opentsp-relations") else { return Ok(()) };
    for r in rels {
        let s = |n: &str| r.get(n).and_then(Json::as_str);
        let (Some(a), Some(b), Some(op), Some(why)) = (s("a"), s("b"), s("op"), s("why")) else {
            return schema_fail("schema: bad x-opentsp-relations entry".into());
        };
        let o = match r.get("offset") {
            None => 0,
            Some(off) => match int_value(Some(off)) {
                Some(o) => o,
                None => return schema_fail("schema: bad x-opentsp-relations entry".into()),
            },
        };
        // "quads": k adds k * (geometry.mem_slices_per_hemisphere / 4), the MEM quads per hemisphere
        let k = match r.get("quads") {
            None => 0,
            Some(q) => match int_value(Some(q)) {
                Some(k) => k,
                None => return schema_fail("schema: bad x-opentsp-relations entry".into()),
            },
        };
        let o = if k != 0 {
            match int_value(lookup(cfg, "geometry.mem_slices_per_hemisphere")) {
                Some(ms) => o + k * (ms / 4),
                None => return schema_fail("schema: a quads relation needs geometry.mem_slices_per_hemisphere".into()),
            }
        } else {
            o
        };
        let (Some(x), Some(y)) = (int_value(lookup(cfg, a)), int_value(lookup(cfg, b))) else {
            return schema_fail(format!("schema: relation names a non-integer key {a} / {b}"));
        };
        let ok = match op {
            "<" => x < y + o,
            "<=" => x <= y + o,
            "==" => x == y + o,
            "!=" => x != y + o,
            _ => return schema_fail(format!("schema: bad relation operator {op}")),
        };
        if !ok {
            if o != 0 {
                return fail(format!("{a}: must be {op} {b} {o:+} ({a} = {x}, {b} = {y}): {why}"));
            }
            return fail(format!("{a}: must be {op} {b} ({a} = {x}, {b} = {y}): {why}"));
        }
    }
    Ok(())
}

/// Check a configuration text against a schema text (`tsp_config_check`). The configuration is
/// parsed first: invalid JSON is `not valid JSON (at byte N)`.
pub fn check_text(config: &[u8], schema: &str) -> Result<(), ConfigError> {
    let Ok(sch) = json::parse(schema.as_bytes()) else {
        return schema_fail("schema: not valid JSON".into());
    };
    let cfg = match json::parse(config) {
        Ok(c) => c,
        Err(at) => return fail(format!("not valid JSON (at byte {at})")),
    };
    check(&sch, &cfg, "")?;
    check_relations(&sch, &cfg)
}
