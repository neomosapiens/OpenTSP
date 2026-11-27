//! The post-hoc energy model: the activity counters of a run (`activity`) priced with a
//! table of picojoules per event and per active or idle cycle by unit class (`opentsp-iss/perf/energy.yaml`), in joules, and
//! in watts at a clock the caller names (the clock is an argument, not a schema key).
//!
//! The per-cycle energies (`cycles`) and the static power (`static_w`, split by `static_share`) are calibrated for the structure
//! the table names under `reference` (the `ref` profile) and scale with the chip they price: each unit class by its own size
//! ([`Structure::class_sizes`](super::activity::Structure::class_sizes): MEM slices times lanes, VXM lanes, matrix planes times
//! rows times lanes, SXM lanes, C2C links, one control unit), so a 16-lane chip does not pay for 320 lanes. The per-event
//! energies need no scaling: they are counted per byte, per operation and per multiply-accumulate.
//!
//! It is an estimate end to end: every value in the shipped table is labelled `estimate`, the calibration targets are stated
//! and unresolved, and PWR-001 semantics (per-bank power-down, wake cycles) are not implemented. Nothing in
//! the model reads it; it runs after the run on the counters.
//!
//! The table is YAML in the subset this module reads (nested maps by indentation, `key: value`, `#` comments, numbers,
//! booleans, quoted or bare strings) or JSON.
use super::activity::{vxm_class, Activity, CLASSES, MXM_TYPES};
use ots_isa::ENTRIES;
use serde_json::{json, Map, Value};
use tsp_numerics::EType;

/// The version of the table format.
pub const SCHEMA: u64 = 1;

/// A parsed energy table.
#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    doc: Value,
}

fn scalar(s: &str) -> Value {
    let s = s.trim();
    if let Some(q) = s.strip_prefix('"').and_then(|r| r.strip_suffix('"')) {
        return Value::String(q.to_string());
    }
    if let Some(q) = s.strip_prefix('\'').and_then(|r| r.strip_suffix('\'')) {
        return Value::String(q.to_string());
    }
    match s {
        "true" => return Value::Bool(true),
        "false" => return Value::Bool(false),
        _ => {}
    }
    if let Ok(i) = s.parse::<i64>() {
        return json!(i);
    }
    if let Ok(f) = s.parse::<f64>() {
        if f.is_finite() {
            return json!(f);
        }
    }
    Value::String(s.to_string())
}

/// `line` without its comment: a `#` at the start or after a space, outside quotes.
fn strip_comment(line: &str) -> &str {
    let mut quote: Option<char> = None;
    let mut prev = ' ';
    for (i, c) in line.char_indices() {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None if c == '"' || c == '\'' => quote = Some(c),
            None if c == '#' && prev.is_whitespace() => return &line[..i],
            None => {}
        }
        prev = c;
    }
    line
}

/// One significant line: its number (for messages), indent, key and the text after the colon.
struct Line {
    no: usize,
    indent: usize,
    key: String,
    rest: String,
}

/// The maps whose keys are at `indent`, from line `*i` on (the lines of deeper maps are consumed with them).
fn block(lines: &[Line], i: &mut usize, indent: usize) -> Result<Value, String> {
    let mut m = Map::new();
    while *i < lines.len() {
        let l = &lines[*i];
        if l.indent < indent {
            break;
        }
        if l.indent > indent {
            return Err(format!("line {}: indentation does not match an enclosing map", l.no));
        }
        if m.contains_key(&l.key) {
            return Err(format!("line {}: duplicate key `{}`", l.no, l.key));
        }
        *i += 1;
        if l.rest.is_empty() {
            match lines.get(*i) {
                Some(n) if n.indent > indent => {
                    let child = n.indent;
                    m.insert(l.key.clone(), block(lines, i, child)?);
                }
                _ => return Err(format!("line {}: `{}:` has no value and no keys under it", l.no, l.key)),
            }
        } else {
            m.insert(l.key.clone(), scalar(&l.rest));
        }
    }
    Ok(Value::Object(m))
}

/// The YAML subset: nested maps by indentation; no lists, no flow style, no multi-line scalars.
fn parse_yaml(text: &str) -> Result<Value, String> {
    let mut lines = Vec::new();
    for (n, raw) in text.lines().enumerate() {
        let line = strip_comment(raw).trim_end();
        if line.trim().is_empty() {
            continue;
        }
        let at = |m: &str| format!("line {}: {m}", n + 1);
        if line.contains('\t') {
            return Err(at("a tab (indent with spaces)"));
        }
        let body = line.trim_start();
        if body.starts_with('-') || body.starts_with('{') || body.starts_with('[') {
            return Err(at("lists and flow style are not part of the subset"));
        }
        let (key, rest) = body.split_once(':').ok_or_else(|| at("expected `key: value` or `key:`"))?;
        let key = key.trim().trim_matches('"').to_string();
        if key.is_empty() {
            return Err(at("empty key"));
        }
        lines.push(Line { no: n + 1, indent: line.len() - body.len(), key, rest: rest.trim().to_string() });
    }
    let first = lines.first().map_or(0, |l| l.indent);
    let mut i = 0;
    let v = block(&lines, &mut i, first)?;
    if i < lines.len() {
        return Err(format!("line {}: indentation does not match an enclosing map", lines[i].no));
    }
    Ok(v)
}

impl Table {
    /// Parse a table (YAML subset, or JSON when the text starts with `{`) and check its shape.
    pub fn parse(text: &str) -> Result<Table, String> {
        let doc = if text.trim_start().starts_with('{') {
            serde_json::from_str(text).map_err(|e| format!("not JSON: {e}"))?
        } else {
            parse_yaml(text)?
        };
        let t = Table { doc };
        t.check()?;
        Ok(t)
    }

    /// Read and parse a file.
    pub fn load(path: &std::path::Path) -> Result<Table, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("cannot read: {}", crate::harness::strerror(&e)))?;
        Table::parse(&text)
    }

    fn get(&self, path: &str) -> Result<&Value, String> {
        let mut v = &self.doc;
        for k in path.split('.') {
            v = v.get(k).ok_or_else(|| format!("the table has no `{path}`"))?;
        }
        Ok(v)
    }

    /// The number at `path` (a non-negative finite number).
    fn num(&self, path: &str) -> Result<f64, String> {
        let f = self.get(path)?.as_f64().ok_or_else(|| format!("`{path}` is not a number"))?;
        if !f.is_finite() || f < 0.0 {
            return Err(format!("`{path}` must be a finite number, 0 or more (got {f})"));
        }
        Ok(f)
    }

    /// Every value the model reads, and the labels the model requires.
    fn check(&self) -> Result<(), String> {
        if self.get("schema")?.as_u64() != Some(SCHEMA) {
            return Err(format!("`schema` must be {SCHEMA}"));
        }
        if self.get("status")?.as_str() != Some("estimate") {
            return Err(
                "`status` must be `estimate`: every value of the table is an estimate until it is calibrated".into()
            );
        }
        for k in [
            "calibration.published_range_w.low",
            "calibration.published_range_w.high",
            "calibration.product_brief_w.average",
            "calibration.product_brief_w.maximum",
        ] {
            self.num(k)?;
        }
        for k in ["calibration.published_range_w.source", "calibration.product_brief_w.source", "calibration.status"] {
            self.get(k)?.as_str().ok_or_else(|| format!("`{k}` must be text"))?;
        }
        for k in EVENT_KEYS {
            self.num(&format!("events.{k}"))?;
        }
        for c in VXM_CLASSES {
            self.num(&format!("events.vxm_op_pj.{c}"))?;
        }
        for t in MXM_TYPES {
            self.num(&format!("events.mxm_mac_pj.{t}"))?;
        }
        for c in CLASSES {
            self.num(&format!("cycles.{c}.active_pj"))?;
            self.num(&format!("cycles.{c}.idle_pj"))?;
        }
        self.num("static_w")?;
        self.num("peak.vxm_alus_per_lane")?;
        for k in REFERENCE_KEYS {
            if self.num(&format!("reference.{k}"))? < 1.0 {
                return Err(format!("`reference.{k}` must be at least 1"));
            }
        }
        let mut share = 0.0;
        for c in CLASSES {
            share += self.num(&format!("static_share.{c}"))?;
        }
        if (share - 1.0).abs() > 1e-9 {
            return Err(format!("`static_share` must add up to 1 (got {share})"));
        }
        Ok(())
    }

    fn ev(&self, k: &str) -> f64 {
        self.num(&format!("events.{k}")).expect("checked")
    }

    fn n(&self, p: &str) -> f64 {
        self.num(p).expect("checked")
    }
}

/// The structure the per-cycle energies and the static power are calibrated for.
const REFERENCE_KEYS: [&str; 5] = ["vlanes", "mem_slices", "mxm_planes", "mxm_rows", "c2c_links"];

const EVENT_KEYS: [&str; 14] = [
    "mem_read_pj_per_byte",
    "mem_write_pj_per_byte",
    "srf_write_pj_per_byte",
    "vxm_width_scale.w1",
    "vxm_width_scale.w2",
    "vxm_width_scale.w4",
    "vxm_float_scale",
    "mxm_load_pj_per_byte",
    "mxm_install_pj_per_byte",
    "mxm_acc_pj_per_lane",
    "sxm_pj_per_byte",
    "c2c_pj_per_byte",
    "host_pj_per_byte",
    "ifetch_pj_per_byte",
];
const VXM_CLASSES: [&str; 8] =
    ["add_sub", "mul", "minmax", "logic", "compare_select", "convert", "reduce", "transcendental"];

// A reduced nonnegative rational shadow retains every admitted binary64 input
// and integer counter exactly. It is used after any unsafe floating intermediate
// and at the final overflow frontier. This is post-hoc pricing, never execution.
#[derive(Clone, Debug)]
struct Exact {
    numerator: num_bigint::BigUint,
    denominator: num_bigint::BigUint,
}
impl Exact {
    fn ratio(mut numerator: num_bigint::BigUint, mut denominator: num_bigint::BigUint) -> Self {
        use num_bigint::BigUint;
        if numerator == BigUint::from(0u8) && denominator != BigUint::from(0u8) {
            return Self { numerator, denominator: BigUint::from(1u8) };
        }
        let (mut a, mut b) = (numerator.clone(), denominator.clone());
        while b != BigUint::from(0u8) {
            let r = a % &b;
            a = b;
            b = r;
        }
        if a != BigUint::from(0u8) {
            numerator /= &a;
            denominator /= a;
        }
        Self { numerator, denominator }
    }
    fn integer(value: u128) -> Self {
        Self { numerator: value.into(), denominator: 1u8.into() }
    }
    fn number(value: f64) -> Self {
        if !value.is_finite() || value < 0.0 {
            return Self::ratio(0u8.into(), 0u8.into());
        }
        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1u64 << 52) - 1);
        let (significand, shift) =
            if exponent == 0 { (fraction, -1074) } else { (fraction | (1u64 << 52), exponent - 1023 - 52) };
        let n = num_bigint::BigUint::from(significand);
        if shift >= 0 {
            Self::ratio(n << shift as usize, 1u8.into())
        } else {
            Self::ratio(n, num_bigint::BigUint::from(1u8) << (-shift) as usize)
        }
    }
    fn is_zero(&self) -> bool {
        self.numerator == num_bigint::BigUint::from(0u8) && self.denominator != num_bigint::BigUint::from(0u8)
    }
    fn add(&self, other: &Self) -> Self {
        Self::ratio(
            &self.numerator * &other.denominator + &other.numerator * &self.denominator,
            &self.denominator * &other.denominator,
        )
    }
    fn mul(&self, other: &Self) -> Self {
        Self::ratio(&self.numerator * &other.numerator, &self.denominator * &other.denominator)
    }
    fn div(&self, other: &Self) -> Self {
        Self::ratio(&self.numerator * &other.denominator, &self.denominator * &other.numerator)
    }
    fn value(&self) -> f64 {
        use num_bigint::BigUint;
        if self.denominator == BigUint::from(0u8) {
            return f64::NAN;
        }
        if self.is_zero() {
            return 0.0;
        }
        let mut exponent = self.numerator.bits() as i64 - self.denominator.bits() as i64;
        let below = if exponent >= 0 {
            self.numerator < (&self.denominator << exponent as usize)
        } else {
            (&self.numerator << (-exponent) as usize) < self.denominator
        };
        if below {
            exponent -= 1;
        }
        if exponent > 1023 {
            return f64::INFINITY;
        }
        if exponent < -1075 {
            return 0.0;
        }
        // One exact quotient/remainder rounding, ties to even, at either the
        // normal significand or the fixed subnormal quantum 2^-1074.
        let shift = if exponent < -1022 { 1074 } else { 52 - exponent };
        let (n, d) = if shift >= 0 {
            (&self.numerator << shift as usize, self.denominator.clone())
        } else {
            (self.numerator.clone(), &self.denominator << (-shift) as usize)
        };
        let mut q = &n / &d;
        let remainder = n % &d;
        let twice = remainder << 1usize;
        if twice > d || (twice == d && q.bit(0)) {
            q += 1u8;
        }
        let rounded = q.to_u64_digits().first().copied().unwrap_or(0);
        if exponent < -1022 {
            return f64::from_bits(rounded);
        }
        let (rounded, exponent) =
            if rounded == 1u64 << 53 { (rounded >> 1, exponent + 1) } else { (rounded, exponent) };
        if exponent > 1023 {
            f64::INFINITY
        } else {
            f64::from_bits((((exponent + 1023) as u64) << 52) | (rounded - (1u64 << 52)))
        }
    }
}

// The ordinary path keeps its historical evaluation order and result bits.
// A nonfinite, subnormal, or lost-positive intermediate selects the exact shadow.
#[derive(Clone, Debug)]
struct Amount {
    ordinary: f64,
    exact: std::rc::Rc<Exact>,
    ordinary_safe: bool,
}
impl Amount {
    fn zero() -> Self {
        Self::count(0)
    }
    fn number(value: f64) -> Self {
        Self {
            ordinary: value,
            exact: std::rc::Rc::new(Exact::number(value)),
            ordinary_safe: value.is_finite() && value >= 0.0,
        }
    }
    fn count(value: u128) -> Self {
        Self { ordinary: value as f64, exact: std::rc::Rc::new(Exact::integer(value)), ordinary_safe: true }
    }
    fn picojoules_to_joules() -> Self {
        Self {
            ordinary: 1e-12,
            exact: std::rc::Rc::new(Exact::ratio(1u8.into(), 1_000_000_000_000u64.into())),
            ordinary_safe: true,
        }
    }
    fn combined(ordinary: f64, exact: Exact, inputs_safe: bool) -> Self {
        let ordinary_safe =
            inputs_safe && ordinary.is_finite() && !ordinary.is_subnormal() && (ordinary != 0.0 || exact.is_zero());
        Self { ordinary, exact: std::rc::Rc::new(exact), ordinary_safe }
    }
    fn value(&self) -> f64 {
        // Even a finite rounded intermediate is insufficient at the overflow
        // frontier. Consult the complete exact expression there, without a clamp.
        if self.ordinary_safe && self.ordinary < f64::MAX / 2.0 {
            self.ordinary
        } else {
            self.exact.value()
        }
    }
    fn json(&self) -> Value {
        json!(self.value())
    }
}
impl std::ops::Add for Amount {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::combined(
            self.ordinary + rhs.ordinary,
            self.exact.add(&rhs.exact),
            self.ordinary_safe && rhs.ordinary_safe,
        )
    }
}
impl std::ops::AddAssign for Amount {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}
impl std::ops::Mul for Amount {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::combined(
            self.ordinary * rhs.ordinary,
            self.exact.mul(&rhs.exact),
            self.ordinary_safe && rhs.ordinary_safe,
        )
    }
}
impl std::ops::Div for Amount {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self::combined(
            self.ordinary / rhs.ordinary,
            self.exact.div(&rhs.exact),
            self.ordinary_safe && rhs.ordinary_safe,
        )
    }
}
impl std::iter::Sum for Amount {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let values: Vec<_> = iter.collect();
        let ordinary = values.iter().map(|v| v.ordinary).sum();
        let exact = values.iter().fold(Exact::integer(0), |sum, v| sum.add(&v.exact));
        Self::combined(ordinary, exact, values.iter().all(|v| v.ordinary_safe))
    }
}

/// Sizes are ratios of the actual admitted numeric reference values; converting
/// the reference to integer structure counts would truncate or saturate them.
fn class_scale(t: &Table, a: &Activity) -> [Amount; 6] {
    let n = |v: u64| Amount::count(v.into());
    let have = a.structure();
    let r = |k: &str| Amount::number(t.n(&format!("reference.{k}")));
    [
        n(have.mem_slices) * n(have.vlanes) / (r("mem_slices") * r("vlanes")),
        n(have.vlanes) / r("vlanes"),
        n(have.mxm_planes) * n(have.mxm_rows) * n(have.vlanes) / (r("mxm_planes") * r("mxm_rows") * r("vlanes")),
        n(have.vlanes) / r("vlanes"),
        n(have.c2c_links) / r("c2c_links"),
        Amount::number(1.0),
    ]
}

fn static_watts(t: &Table, scale: &[Amount; 6]) -> Amount {
    CLASSES
        .iter()
        .zip(scale)
        .map(|(c, s)| Amount::number(t.n(&format!("static_share.{c}"))) * s.clone())
        .sum::<Amount>()
        * Amount::number(t.n("static_w"))
}

/// Price counters in picojoules, retaining wide intermediate magnitudes.
fn price(t: &Table, a: &Activity, fetch_bytes: u64, cycles: u128) -> Vec<(&'static str, Amount)> {
    let vb = Amount::count(a.vector_bytes().into());
    let ev = |k: &str| Amount::number(t.ev(k));
    let n = |k: &str| Amount::number(t.n(k));
    let reads: u128 = a.mem.iter().map(|m| u128::from(m.read_bytes)).sum();
    let writes: u128 = a.mem.iter().map(|m| u128::from(m.write_bytes)).sum();
    let mem = Amount::count(reads) * ev("mem_read_pj_per_byte") + Amount::count(writes) * ev("mem_write_pj_per_byte");
    let srf = Amount::count(u128::from(a.srf_writes[0]) + u128::from(a.srf_writes[1]))
        * vb.clone()
        * ev("srf_write_pj_per_byte");
    let mut vxm = Amount::zero();
    for (e, entry) in ENTRIES.iter().enumerate() {
        if entry.slice != "VXM" {
            continue;
        }
        let class = n(&format!("events.vxm_op_pj.{}", vxm_class(entry.id)));
        for ty in EType::ALL.iter().chain(EType::EXTENSIONS.iter()) {
            let ops = a.vxm[e][ty.index()];
            if ops == 0 {
                continue;
            }
            let w = ev(match ty.width() {
                1 => "vxm_width_scale.w1",
                2 => "vxm_width_scale.w2",
                _ => "vxm_width_scale.w4",
            });
            let f = if ty.is_float() { ev("vxm_float_scale") } else { Amount::number(1.0) };
            vxm += Amount::count(ops.into()) * class.clone() * w * f;
        }
    }
    let mut mxm = Amount::zero();
    for (k, ty) in MXM_TYPES.iter().enumerate() {
        mxm += Amount::count(a.mxm.macs[k].into()) * n(&format!("events.mxm_mac_pj.{ty}"));
    }
    mxm +=
        Amount::count(u128::from(a.mxm.lw_rows) + u128::from(a.mxm.iws_rows)) * vb.clone() * ev("mxm_load_pj_per_byte");
    mxm += Amount::count(a.mxm.iw_rows.into()) * vb.clone() * ev("mxm_install_pj_per_byte");
    mxm += Amount::count(a.mxm.acc_lanes.into()) * ev("mxm_acc_pj_per_lane");
    let sxm = Amount::count(u128::from(a.sxm_in_vectors) + u128::from(a.sxm_out_vectors))
        * vb.clone()
        * ev("sxm_pj_per_byte");
    let c2c_vectors: u128 = a.c2c_sent.iter().chain(&a.c2c_received).chain(&a.c2c_sync).map(|n| u128::from(*n)).sum();
    let c2c = Amount::count(c2c_vectors) * vb.clone() * ev("c2c_pj_per_byte");
    let host = Amount::count(u128::from(a.host_in) + u128::from(a.host_out)) * vb.clone() * ev("host_pj_per_byte");
    let fetch = Amount::count(fetch_bytes.into()) * ev("ifetch_pj_per_byte");
    let (mut active, mut idle) = (Amount::zero(), Amount::zero());
    let scale = class_scale(t, a);
    for (k, c) in CLASSES.iter().enumerate() {
        let act = u128::from(a.active_cycles[k]).min(cycles);
        active += Amount::count(act) * n(&format!("cycles.{c}.active_pj")) * scale[k].clone();
        idle += Amount::count(cycles - act) * n(&format!("cycles.{c}.idle_pj")) * scale[k].clone();
    }
    vec![
        ("mem", mem),
        ("srf", srf),
        ("vxm", vxm),
        ("mxm", mxm),
        ("sxm", sxm),
        ("c2c", c2c),
        ("host", host),
        ("ifetch", fetch),
        ("active_cycles", active),
        ("idle_cycles", idle),
    ]
}

/// A canonical busy reference mix, not a universal maximum or a physical measurement.
fn full_activity_w(t: &Table, a: &Activity, clock_mhz: f64) -> Amount {
    let vb = Amount::count(a.vector_bytes().into());
    let rows = Amount::count(a.mxm_rows().into());
    let planes = Amount::count(a.structure().mxm_planes.into());
    let slices = Amount::count(a.structure().mem_slices.into());
    let scale = class_scale(t, a);
    let n = |k: &str| Amount::number(t.n(k));
    let ev = |k: &str| Amount::number(t.ev(k));
    let pj_per_cycle = planes * rows * vb.clone() * n("events.mxm_mac_pj.i8")
        + slices
            * vb.clone()
            * (ev("mem_read_pj_per_byte") + ev("mem_write_pj_per_byte") + ev("srf_write_pj_per_byte"))
        + n("peak.vxm_alus_per_lane") * vb.clone() * n("events.vxm_op_pj.add_sub") * ev("vxm_width_scale.w1")
        + CLASSES.iter().zip(&scale).map(|(c, s)| n(&format!("cycles.{c}.active_pj")) * s.clone()).sum::<Amount>();
    pj_per_cycle * Amount::picojoules_to_joules() * Amount::number(clock_mhz) * Amount::number(1e6)
        + static_watts(t, &scale)
}

struct PricedChip {
    json: Value,
    joules: Amount,
}
fn priced_chip(t: &Table, a: &Activity, fetch_bytes: u64, cycles: u128, clock_mhz: f64) -> PricedChip {
    let parts = price(t, a, fetch_bytes, cycles);
    let dynamic_pj: Amount = parts.iter().map(|p| p.1.clone()).sum();
    let seconds = Amount::count(cycles) / (Amount::number(clock_mhz) * Amount::number(1e6));
    let scale = class_scale(t, a);
    let static_w = static_watts(t, &scale);
    let static_j = static_w.clone() * seconds.clone();
    let class_scale_json: Map<String, Value> =
        CLASSES.iter().zip(scale).map(|(c, s)| ((*c).to_string(), s.json())).collect();
    let mut joules = Map::new();
    for (k, pj) in &parts {
        joules.insert((*k).to_string(), (pj.clone() * Amount::picojoules_to_joules()).json());
    }
    joules.insert("static".into(), static_j.json());
    let dynamic_j = dynamic_pj * Amount::picojoules_to_joules();
    let total = dynamic_j.clone() + static_j;
    joules.insert("total".into(), total.json());
    let watts = |j: Amount| if cycles > 0 { (j / seconds.clone()).json() } else { Value::Null };
    PricedChip {
        joules: total.clone(),
        json: json!({
            "status": "estimate", "clock_mhz": clock_mhz, "cycles": cycles, "seconds": seconds.json(), "joules": joules,
            "watts": {"average": watts(total), "dynamic": watts(dynamic_j), "static": static_w.json()},
            "structure": {
                "vlanes": a.structure().vlanes, "mem_slices": a.structure().mem_slices, "mxm_planes": a.structure().mxm_planes,
                "mxm_rows": a.structure().mxm_rows, "c2c_links": a.structure().c2c_links,
            },
            "class_scale": class_scale_json, "full_activity_watts": full_activity_w(t, a, clock_mhz).json(),
        }),
    }
}

/// The energy object for one chip. Finite coefficients remain admitted; a genuinely
/// unrepresentable final quantity retains the existing nullable numeric encoding.
pub fn estimate(t: &Table, a: &Activity, fetch_bytes: u64, cycles: u128, clock_mhz: f64) -> Value {
    priced_chip(t, a, fetch_bytes, cycles, clock_mhz).json
}

/// Chips' joules add; watts use the longest run. Aggregate typed magnitudes before
/// serialization so an unavailable per-chip number can never become zero.
pub fn estimate_run(t: &Table, chips: &[(Activity, u64, u128)], clock_mhz: f64) -> Value {
    let priced: Vec<_> = chips.iter().map(|(a, f, c)| priced_chip(t, a, *f, *c, clock_mhz)).collect();
    let cycles = chips.iter().map(|c| c.2).max().unwrap_or(0);
    let seconds = Amount::count(cycles) / (Amount::number(clock_mhz) * Amount::number(1e6));
    let total: Amount = priced.iter().map(|c| c.joules.clone()).sum();
    let per_chip: Vec<_> = priced.into_iter().map(|c| c.json).collect();
    json!({
        "status": "estimate", "table_schema": SCHEMA, "clock_mhz": clock_mhz, "cycles": cycles,
        "seconds": seconds.json(), "joules": total.json(),
        "watts": if cycles > 0 { (total / seconds).json() } else { Value::Null },
        "calibration": t.doc["calibration"],
        "pwr_001": "not implemented: no per-bank power-down and no wake cycles; idle cycles are charged at the table's idle energy",
        "per_chip": per_chip,
    })
}

/// A few lines for standard error.
pub fn summary(e: &Value) -> String {
    let f = |v: &Value| v.as_f64().map_or("n/a".to_string(), |x| format!("{x:.4e}"));
    let mut s = format!(
        "energy (estimate, {} MHz, {} cycles): {} J, {} W average\n",
        e["clock_mhz"],
        e["cycles"],
        f(&e["joules"]),
        f(&e["watts"])
    );
    for (i, c) in e["per_chip"].as_array().into_iter().flatten().enumerate() {
        s += &format!(
            "  chip {i}: {} J = {}\n",
            f(&c["joules"]["total"]),
            c["joules"]
                .as_object()
                .map(|m| m
                    .iter()
                    .filter(|(k, _)| *k != "total")
                    .map(|(k, v)| format!("{k} {}", f(v)))
                    .collect::<Vec<_>>()
                    .join(", "))
                .unwrap_or_default()
        );
        s += &format!("          canonical busy mix {} W\n", f(&c["full_activity_watts"]));
    }
    let unavailable = e["joules"].is_null()
        || e["seconds"].is_null()
        || (e["cycles"] != json!(0) && e["watts"].is_null())
        || e["per_chip"].as_array().into_iter().flatten().any(|c| {
            c["full_activity_watts"].is_null()
                || c["seconds"].is_null()
                || c["joules"].as_object().into_iter().flat_map(|m| m.values()).any(Value::is_null)
        });
    if unavailable {
        s += "  n/a: this estimate exceeds the finite numeric output range; unavailable values are not zero.\n";
    }
    s += "  calibration targets (unresolved, both published): 35-350 W range; 185 W average and 300 W maximum\n";
    s
}

#[cfg(test)]
#[path = "../../tests/private/energy_rounding.rs"]
mod range_rounding_tests;
