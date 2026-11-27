//! The timing table (microarchitecture.md section 1): one row per
//! catalogue entry, the entry's `timing.overrides` row or else its class's. Port of
//! `model/core/timing.sail` and `model/postlude/timing_check.sail`.
use crate::{Profile, CATALOGUE};
use serde_json::Value;

/// A timing class of the catalogue (`encoding.json` `instructions[].timing`). // model: core/timing.sail timing_class
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimingClass {
    Cu,
    Mem,
    Vxm,
    MxmLw,
    MxmIw,
    MxmAbc,
    MxmAcc,
    Sxm,
    SxmCross,
    C2c,
}

/// The configuration key of a class (`timing.<key>`). // model: postlude/timing_check.sail timing_class_key
pub fn timing_class_key(c: TimingClass) -> &'static str {
    match c {
        TimingClass::Cu => "cu",
        TimingClass::Mem => "mem",
        TimingClass::Vxm => "vxm",
        TimingClass::MxmLw => "mxm_lw",
        TimingClass::MxmIw => "mxm_iw",
        TimingClass::MxmAbc => "mxm_abc",
        TimingClass::MxmAcc => "mxm_acc",
        TimingClass::Sxm => "sxm",
        TimingClass::SxmCross => "sxm_cross",
        TimingClass::C2c => "c2c",
    }
}

/// A catalogue entry as the timing table sees it (generated from `encoding.json`).
#[derive(Debug, Clone, Copy)]
pub struct CatalogueEntry {
    pub name: &'static str,
    pub slice: &'static str,
    pub class: TimingClass,
    /// The codes of the element types the entry's `type` operand takes (empty: the entry has no element type).
    pub type_codes: &'static [u8],
}

/// The name of the element type with `code` (`encoding.json` `eltype_codes`).
pub fn eltype_name(code: u32) -> Option<&'static str> {
    crate::ELTYPES.iter().find(|(_, c)| u32::from(*c) == code).map(|(n, _)| *n)
}

/// `d_func`: cycles from dispatch to the commit of the stream results; `d_skew`: cycles from
/// dispatch to the execution of the first reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Latency {
    pub d_func: u32,
    pub d_skew: u32,
}

/// The effective latency of every catalogue entry, in catalogue order (index = the entry's
/// position in `encoding.json` `instructions`, which is `ots_isa::EntryId`).
#[derive(Debug, Clone)]
pub struct TimingTable {
    rows: Vec<Latency>,
    /// Per entry, the rows keyed by an element type of the entry, as (type code, row); empty for nearly every entry.
    typed: Vec<Vec<(u8, Latency)>>,
}

impl TimingTable {
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
    /// The row of the catalogue entry at `index`.
    pub fn row(&self, index: usize) -> Latency {
        self.rows[index]
    }
    /// The row of the entry named `name` (`isa-db` `name`, for example `Add`). // model: postlude/timing_check.sail entry_row
    pub fn entry(&self, name: &str) -> Option<Latency> {
        CATALOGUE.iter().position(|e| e.name == name).map(|i| self.rows[i])
    }
    /// Whether the entry at `index` has a row keyed by an element type (the lookup by type is needed only then).
    pub fn has_typed(&self, index: usize) -> bool {
        !self.typed[index].is_empty()
    }
    /// The row of the entry at `index` for an instruction whose `type` operand is `code`: the row keyed by that type,
    /// else the entry's row. // model: core/timing.sail entry_latency_typed
    pub fn row_typed(&self, index: usize, code: u32) -> Latency {
        match self.typed[index].iter().find(|(c, _)| u32::from(*c) == code) {
            Some((_, l)) => *l,
            None => self.rows[index],
        }
    }
    /// The row of entry `name` at the element type named `ty` (`i8`): `None` when `name` is not an entry or does not take `ty`.
    pub fn entry_typed(&self, name: &str, ty: &str) -> Option<Latency> {
        let i = CATALOGUE.iter().position(|e| e.name == name)?;
        let code = crate::ELTYPES.iter().find(|(n, _)| *n == ty)?.1;
        CATALOGUE[i].type_codes.contains(&code).then(|| self.row_typed(i, u32::from(code)))
    }
}

// model: core/timing.sail latency_of
fn class_row(p: &Profile, c: TimingClass) -> Latency {
    let key = timing_class_key(c);
    let n = |f: &str| p.uint(&format!("timing.{key}.{f}")).unwrap_or(0) as u32;
    Latency { d_func: n("d_func"), d_skew: n("d_skew") }
}

/// The `timing.overrides` rows, as (entry, row), in file order. // model: core/timing.sail timing_overrides
fn overrides(p: &Profile) -> Vec<(String, Latency)> {
    let Some(Value::Array(rows)) = p.get("timing.overrides") else { return Vec::new() };
    rows.iter()
        .map(|r| {
            let n = |f: &str| r.get(f).and_then(Value::as_u64).unwrap_or(0) as u32;
            (
                r.get("entry").and_then(Value::as_str).unwrap_or("").to_string(),
                Latency { d_func: n("d_func"), d_skew: n("d_skew") },
            )
        })
        .collect()
}

// model: core/timing.sail entry_latency, entry_latency_typed
pub(crate) fn table(p: &Profile) -> TimingTable {
    let ov = overrides(p);
    // A profile has one immutable row per class, shared by many catalogue entries.
    let mut class_rows = std::collections::HashMap::new();
    let rows = CATALOGUE
        .iter()
        .map(|e| match ov.iter().find(|(n, _)| n == e.name) {
            Some((_, l)) => *l,
            None => *class_rows.entry(e.class).or_insert_with(|| class_row(p, e.class)),
        })
        .collect();
    let typed = CATALOGUE
        .iter()
        .map(|e| {
            // Shipped profiles have only bare-entry overrides. Avoid building every possible
            // "Entry.type" key when no override could match one; keep the typed path unchanged.
            if !ov.iter().any(|(name, _)| name.contains('.')) {
                return Vec::new();
            }
            e.type_codes
                .iter()
                .filter_map(|&c| {
                    let key = format!("{}.{}", e.name, eltype_name(u32::from(c))?);
                    ov.iter().find(|(n, _)| *n == key).map(|(_, l)| (c, *l))
                })
                .collect()
        })
        .collect();
    TimingTable { rows, typed }
}

/// How a row's entry reads when it is not a catalogue entry: `<entry>.<type>` with `entry` a catalogue entry and `type`
/// an element type, as (entry, type, whether the entry takes the type). // model: postlude/timing_check.sail typed_parts
fn typed_parts(s: &str) -> Option<(&'static str, &'static str, bool)> {
    for e in CATALOGUE {
        for (tn, code) in crate::ELTYPES {
            if s.len() == e.name.len() + 1 + tn.len()
                && s.starts_with(e.name)
                && s[e.name.len()..].strip_prefix('.') == Some(*tn)
            {
                return Some((e.name, *tn, e.type_codes.contains(code)));
            }
        }
    }
    None
}

/// The order in which the model lists the catalogue (`catalogue_entries` of
/// `timing_check.sail`): by unit, then by name without regard to case. It decides which row is
/// named when several are wrong. // model: postlude/timing_check.sail catalogue_entries
fn model_order() -> Vec<usize> {
    let mut slices: Vec<&str> = Vec::new();
    for e in CATALOGUE {
        if !slices.contains(&e.slice) {
            slices.push(e.slice);
        }
    }
    let mut idx: Vec<usize> = (0..CATALOGUE.len()).collect();
    idx.sort_by_key(|&i| {
        (slices.iter().position(|s| *s == CATALOGUE[i].slice), CATALOGUE[i].name.to_ascii_lowercase())
    });
    idx
}

const CROSS_ROWS: [(&str, &str); 4] =
    [("Shift", "sxm_cross"), ("Permute", "sxm_cross"), ("ABC", "mxm_abc"), ("ABC8", "mxm_abc")];

fn cross_row_early(func: i128, skew: i128, tiles: i128) -> bool {
    func - skew < tiles
}

fn fetch_early(rate: i128, latency: i128, tiles: i128) -> bool {
    if rate <= 0 || tiles == 0 {
        return false;
    }
    // For a given beat the delivery-minus-capture time is monotone in tile:
    // the slope before flooring is 16/rate - 1. Its minimum is at an endpoint.
    (0..2).any(|beat| {
        [0, tiles - 1].into_iter().any(|tile| 1 + latency + 16 * (beat * tiles + tile) / rate <= beat + tile)
    })
}

impl Profile {
    /// The assembler's profile contract (tsp_as.load_profile). Report all causal
    /// errors in reference order. Full schema and implementation limits apply at execution.
    pub fn check_assembler_timing(&self) -> Result<(), String> {
        // The Python assembler applies these relations before any simulator schema
        // check. Preserve that contract for custom profiles with signed timing values.
        let integer = |v: &Value| v.as_i64().map(i128::from).or_else(|| v.as_u64().map(i128::from));
        let key = |k: &str| self.get(k).and_then(integer).ok_or_else(|| format!("{k}: expected integer"));
        let sl = i128::from(self.superlanes());
        let mut errors = Vec::new();
        if fetch_early(key("icu.fetch_bytes_per_cycle")?, key("icu.fetch_latency")?, sl) {
            errors
                .push("icu.fetch_bytes_per_cycle: instruction text delivery precedes source tile capture".to_string());
        }
        let overrides = self
            .get("timing.overrides")
            .and_then(Value::as_array)
            .ok_or_else(|| "timing.overrides: expected array".to_owned())?;
        for (name, class) in CROSS_ROWS {
            let row =
                overrides.iter().rev().find(|r| r["entry"] == name).or_else(|| self.get(&format!("timing.{class}")));
            let field =
                |k: &str| row.and_then(|r| integer(&r[k])).ok_or_else(|| format!("{name}.{k}: expected integer"));
            if cross_row_early(field("d_func")?, field("d_skew")?, sl) {
                errors.push(format!("{name}: d_func - d_skew must be at least geometry.superlanes ({sl})"));
            }
        }
        if key("timing.sxm_map_ready")? < sl {
            errors.push("timing.sxm_map_ready: must be at least geometry.superlanes".to_string());
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }
}

// model: postlude/timing_check.sail timing_config_error
pub(crate) fn config_error(p: &Profile) -> Option<String> {
    let ov = overrides(p);
    let t = table(p);
    let row_of = |name: &str| t.entry(name).expect("entry");
    let row_key = |name: &str| match ov.iter().any(|(n, _)| n == name) {
        true => format!("timing.overrides.{name}"),
        false => match CATALOGUE.iter().find(|e| e.name == name) {
            Some(e) => format!("timing.{}", timing_class_key(e.class)),
            None => format!("timing.overrides.{name}"),
        },
    };
    // `extensions` names only the extensions of this version, each once (the first rule, as in the model)
    let mut err = String::new();
    let names = p.extensions();
    for (i, x) in names.iter().enumerate() {
        if err.is_empty() {
            if *x != "bf16" && *x != "wide_acc" && *x != "block_fp8" {
                err = format!("extensions: {x} is not an extension of this version");
            } else if names[i + 1..].contains(x) {
                err = format!("extensions: {x} is listed more than once");
            }
        }
    }
    macro_rules! first {
        ($e:expr) => {{
            let e: String = $e;
            if err.is_empty() {
                err = e;
            }
        }};
    }
    for (entry, _) in &ov {
        let twice = ov.iter().filter(|(n, _)| n == entry).count() > 1;
        if CATALOGUE.iter().any(|e| e.name == entry) {
            if twice {
                first!(format!("timing.overrides: {entry} is listed more than once"));
            }
        } else if let Some((n, tn, takes)) = typed_parts(entry) {
            if !takes {
                first!(format!("timing.overrides: {entry}: {n} does not take element type {tn}"));
            } else if twice {
                first!(format!("timing.overrides: {entry} is listed more than once"));
            }
        } else {
            first!(format!("timing.overrides: {entry} is not a catalogue entry"));
        }
    }
    if err.is_empty() {
        for i in model_order() {
            let e = &CATALOGUE[i];
            let l = t.row(i);
            if e.class != TimingClass::Cu && l.d_skew >= l.d_func {
                first!(format!(
                    "{}: {}: d_skew {} must be below d_func {} (an instruction commits after it executes at dispatch + d_skew)",
                    row_key(e.name),
                    e.name,
                    l.d_skew,
                    l.d_func
                ));
            }
        }
        for (entry, l) in &ov {
            if !CATALOGUE.iter().any(|e| e.name == entry) && l.d_skew >= l.d_func {
                first!(format!(
                    "timing.overrides.{entry}: {entry}: d_skew {} must be below d_func {} (an instruction commits after it executes at dispatch + d_skew)",
                    l.d_skew, l.d_func
                ));
            }
        }
        let superlanes = p.superlanes() as u32;
        for (n, _) in CROSS_ROWS {
            let l = row_of(n);
            if cross_row_early(l.d_func.into(), l.d_skew.into(), superlanes.into()) {
                first!(format!(
                    "{}: {n}: d_func {} must be at least geometry.superlanes {superlanes} plus d_skew {} (later-superlane operands must arrive before output)",
                    row_key(n),
                    l.d_func, l.d_skew
                ));
            }
        }
        let rate = p.uint("icu.fetch_bytes_per_cycle").unwrap_or(0);
        let latency = p.uint("icu.fetch_latency").unwrap_or(0);
        let tiles = u64::from(superlanes);
        if fetch_early(rate.into(), latency.into(), tiles.into()) {
            first!("icu.fetch_bytes_per_cycle: instruction text delivery precedes source tile capture".to_string());
        }
        if p.uint("timing.sxm_map_ready").unwrap_or(0) < u64::from(superlanes) {
            first!("timing.sxm_map_ready: must be at least geometry.superlanes (map dissemination)".to_string());
        }
        let occupancy = p.uint("timing.mxm_abc.occupancy_f16").unwrap_or(0);
        let abc = row_of("ABC").d_func as u64;
        if occupancy > abc {
            first!(format!(
                "timing.mxm_abc.occupancy_f16: {occupancy} must be at most ABC's d_func {abc} (macc_busy finds the f16 entry that still holds the plane only while it has not committed)"
            ));
        }
        let acc_bits = p.uint("mxm.acc_bits").unwrap_or(32);
        if acc_bits != 32 && acc_bits != 128 {
            first!(format!("mxm.acc_bits: {acc_bits} must be 32 or 128"));
        }
        if p.has_extension("wide_acc") != (acc_bits == 128) {
            first!("extensions: wide_acc is declared exactly when mxm.acc_bits is 128".to_string());
        }
        let s0 = row_of("Read").d_skew;
        for n in ["Read", "Write", "Gather", "Scatter", "IfetchM"] {
            let l = row_of(n);
            if l.d_skew != s0 {
                first!(format!(
                    "{}: {n}: d_skew {} must equal Read's {s0} (Read, Write, Gather, Scatter and IfetchM share the SRAM ports' dispatch rule)",
                    row_key(n),
                    l.d_skew
                ));
            }
        }
        let rows = p.uint("mxm.rows").unwrap_or(0);
        let f16 = p.uint("mxm.f16_rows").unwrap_or(0);
        if !(2 * f16 >= rows && f16 <= rows) {
            first!(format!("mxm.f16_rows: {f16} must be between mxm.rows / 2 and mxm.rows ({} and {rows})", rows / 2));
        }
    }
    if err.is_empty() {
        None
    } else {
        Some(err)
    }
}
