//! The native subcommands: `info` here, `run`, `vectors` and `bench` in module `native`.
use crate::geom::{Geom, CONTROL_QUEUE, DATA_QUEUES};
use crate::harness::{load_config, PROG};
use ots_isa::{queue_name, Access, Isa, UNITS};
use ots_profile::{timing_class_key, Profile, CATALOGUE};
use serde_json::{json, Value};
use std::ffi::OsString;
use std::io::Write;
use std::path::Path;

const INFO_USAGE: &str = "info: usage: tsp-iss info (--config <config.json> | --profile <ref|mini|name>)";

/// Dispatch a native subcommand; `args[0]` is its name.
pub fn native(args: &[OsString], stdout: &mut dyn Write, stderr: &mut dyn Write) -> i32 {
    let name = args[0].to_string_lossy().into_owned();
    match name.as_str() {
        "info" => info(&args[1..], stdout, stderr),
        "run" => crate::native::run(&args[1..], stdout, stderr),
        "vectors" => crate::native::vectors(&args[1..], stdout, stderr),
        "bench" => crate::native::bench(&args[1..], stdout, stderr),
        "verify" => crate::verify::verify(&args[1..], stdout, stderr),
        _ => unreachable!("is_subcommand admits only the names above"),
    }
}

fn info(args: &[OsString], stdout: &mut dyn Write, stderr: &mut dyn Write) -> i32 {
    let usage = |stderr: &mut dyn Write| {
        let _ = writeln!(stderr, "{PROG}: {INFO_USAGE}");
        2
    };
    let (kind, value) = match args {
        [k, v] if k == "--config" || k == "--profile" => (k.to_string_lossy().into_owned(), v.clone()),
        _ => return usage(stderr),
    };
    let profile = if kind == "--config" {
        let path = Path::new(&value);
        match load_config(path) {
            Ok(p) => p,
            Err(m) => {
                let _ = writeln!(stderr, "{PROG}: config {}: {m}", path.display());
                return 2;
            }
        }
    } else {
        let name = value.to_string_lossy().into_owned();
        let p = match Profile::load(&name) {
            Ok(p) => p,
            Err(e) => {
                let _ = writeln!(stderr, "{PROG}: info: {e}");
                return 2;
            }
        };
        if let Err(e) = p.validate() {
            let _ = writeln!(stderr, "{PROG}: config {name}: {e}");
            return 2;
        }
        p
    };
    let doc = describe(&profile);
    let _ = writeln!(stdout, "{}", serde_json::to_string_pretty(&doc).expect("json"));
    0
}

/// The unit of a data queue (`encoding.json` `units`, not the `any` alias).
fn unit_of(q: u16) -> Option<&'static str> {
    UNITS.iter().find(|(n, qs)| *n != "any" && qs.contains(&q)).map(|(n, _)| *n)
}

/// The derived profile as JSON: sizes, the stations of every queue and the timing table.
pub fn describe(p: &Profile) -> Value {
    let g = Geom::new(p);
    let isa = Isa::new(p);
    let mut queues = Vec::new();
    let qs = (0..DATA_QUEUES as u16).chain(std::iter::once(CONTROL_QUEUE));
    for q in qs {
        let unit = unit_of(q).unwrap_or("?");
        let mem = unit.starts_with("MEM");
        let (present, quad) = if mem { (g.slice_present((q % 88) as u64), ((q % 44) / 4) as i64) } else { (true, 0) };
        let stations = if present {
            let gap = |a: Access, east: bool| isa.station(unit, a, east, quad).map_or(Value::Null, |x| json!(x));
            json!({
                "read": {"E": gap(Access::Read, true), "W": gap(Access::Read, false)},
                "write": {"E": gap(Access::Write, true), "W": gap(Access::Write, false)},
            })
        } else {
            Value::Null
        };
        queues.push(json!({
            "queue": q,
            "name": queue_name(q),
            "unit": unit,
            "present": present,
            "stations": stations,
        }));
    }
    let key = |k: &str| p.get(k).cloned().unwrap_or(Value::Null);
    let table = p.timing_table();
    let mut timing: Vec<Value> = CATALOGUE
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let l = table.row(i);
            json!({"entry": e.name, "unit": e.slice, "class": timing_class_key(e.class), "d_func": l.d_func, "d_skew": l.d_skew})
        })
        .collect();
    // the rows keyed by an element type (only a profile with `<Entry>.<type>` override rows has any)
    for (i, e) in CATALOGUE.iter().enumerate().filter(|(i, _)| table.has_typed(*i)) {
        for &c in e.type_codes {
            let (Some(ty), l) = (ots_profile::eltype_name(u32::from(c)), table.row_typed(i, u32::from(c))) else {
                continue;
            };
            if l != table.row(i) {
                timing.push(json!({"entry": e.name, "type": ty, "unit": e.slice, "class": timing_class_key(e.class), "d_func": l.d_func, "d_skew": l.d_skew}));
            }
        }
    }
    json!({
        "spec_version": ots_profile::SPEC_VERSION,
        "digest12": p.digest12().iter().map(|b| format!("{b:02x}")).collect::<String>(),
        "superlanes": g.superlanes,
        "vlanes": g.vlanes,
        "words_per_slice": g.words_per_slice,
        "bank_bit": g.bank_bit,
        "addr_bits": g.addr_bits,
        "iq_depth": g.iq_depth,
        "iq_units": g.iq_units,
        "mem_slices_per_hemisphere": g.mem_slices,
        "quads": g.quads,
        "mxm_rows": g.mxm_rows,
        // the profile's extensions and the MXM keys a compiler asks for
        "extensions": p.extensions(),
        "mxm_acc_bits": key("mxm.acc_bits"),
        "mxm_acc_depth": key("mxm.acc_depth"),
        "mxm_planes_per_hemisphere": key("mxm.planes_per_hemisphere"),
        "mxm_f16_rows": key("mxm.f16_rows"),
        "mxm_fp8_rate": key("mxm.fp8_rate"),
        "links": g.links,
        "link_latency": {"nominal": g.link_latency, "min": g.link_latency_min, "max": g.link_latency_max},
        "queues": queues,
        "timing": timing,
    })
}
