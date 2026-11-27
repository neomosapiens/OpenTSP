// Generates the instruction tables from isa-db/generated/encoding.json (nothing about the ISA is
// written twice): EntryId, Insn (one variant per catalogue entry), the entry and operand-kind
// tables, queue names and units, placement sets, AGT and slot data, stations, diagnostic codes.
// The path is relative to the crate; OTS_ENCODING_JSON overrides it. A renamed catalogue key
// breaks this script (a panic naming it), not a distant test.
use serde_json::Value;
use std::{env, fmt::Write as _, fs, path::PathBuf};

fn q(s: &str) -> String {
    format!("{s:?}")
}

fn span(loc: &str) -> (u32, u32) {
    let (a, b) = loc.split_once('-').unwrap_or((loc, loc));
    (a.parse().unwrap(), b.parse().unwrap())
}

fn arr_u32(v: &Value) -> String {
    let xs: Vec<String> = v.as_array().unwrap().iter().map(|x| format!("{}", x.as_i64().unwrap())).collect();
    format!("&[{}]", xs.join(", "))
}

fn ident(name: &str) -> String {
    const KW: &[&str] = &[
        "type", "mod", "match", "move", "ref", "self", "super", "use", "where", "loop", "in", "as", "fn", "let",
        "impl", "mut", "pub", "static", "struct", "trait", "unsafe", "async", "await", "dyn", "enum", "extern",
        "false", "true", "for", "if", "else", "while", "return", "const", "break", "continue", "crate", "box", "do",
        "final", "macro", "override", "priv", "try", "typeof", "unsized", "virtual", "yield",
    ];
    if KW.contains(&name) {
        format!("r#{name}")
    } else {
        name.to_string()
    }
}

fn camel(code: &str) -> String {
    // OTS-E-DEC-003 -> Dec003
    let parts: Vec<&str> = code.split('-').skip(2).collect();
    parts
        .iter()
        .map(|p| {
            let mut c = p.chars();
            let f = c.next().unwrap().to_ascii_uppercase();
            format!("{f}{}", c.as_str().to_ascii_lowercase())
        })
        .collect()
}

fn main() {
    let default = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("../../../opentsp-arch-spec/isa-db/generated/encoding.json");
    println!("cargo:rerun-if-env-changed=OTS_ENCODING_JSON");
    let path = env::var("OTS_ENCODING_JSON").map(PathBuf::from).unwrap_or(default);
    let path = fs::canonicalize(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    println!("cargo:rerun-if-changed={}", path.display());
    let text = fs::read_to_string(&path).unwrap();
    let db: Value = serde_json::from_str(&text).expect("encoding.json is not JSON");
    assert_eq!(db["spec_version"], "0.1", "encoding.json: spec version is not 0.1");
    let mut o = String::new();

    writeln!(o, "pub static ENCODING_JSON: &[u8] = include_bytes!({});", q(path.to_str().unwrap())).unwrap();

    // ---- scalars -------------------------------------------------------------------------
    let control = db["control_channel"]["queue"].as_u64().unwrap();
    writeln!(o, "pub const CONTROL: u16 = {control};").unwrap();
    let hex = |s: &str| u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap();
    writeln!(o, "pub const PADDING: u32 = {:#x};", hex(db["text_format"]["padding"].as_str().unwrap())).unwrap();
    writeln!(o, "pub const UNDECODABLE: u32 = {:#x};", hex(db["text_format"]["undecodable_unit"].as_str().unwrap()))
        .unwrap();
    writeln!(o, "pub const BUNDLE_BIT: u32 = {:#x};", 1u64 << db["text_format"]["bundle_bit"].as_u64().unwrap())
        .unwrap();
    writeln!(
        o,
        "pub static NARROW_TAGS: &[u8] = &{:?};",
        db["decode_rule"]["narrow_tags"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect::<Vec<_>>()
    )
    .unwrap();
    writeln!(
        o,
        "pub static WIDE_TAGS: &[u8] = &{:?};",
        db["decode_rule"]["wide_tags"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect::<Vec<_>>()
    )
    .unwrap();
    writeln!(
        o,
        "pub static BOOLEAN_CODES: &[u32] = &{:?};",
        db["boolean_codes"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect::<Vec<_>>()
    )
    .unwrap();
    writeln!(o, "pub const STREAM_ROW_BIT: u32 = {};", db["stream_encoding"]["row_bit"]).unwrap();
    let rows: Vec<String> = db["stream_encoding"]["row_codes"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| format!("({}, {})", q(k), v))
        .collect();
    writeln!(o, "pub static STREAM_ROW_CODES: &[(&str, u32)] = &[{}];", rows.join(", ")).unwrap();
    let mut el: Vec<(String, u64)> =
        db["eltype_codes"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap())).collect();
    el.sort_by_key(|(_, c)| *c);
    writeln!(
        o,
        "pub static ELTYPE_CODES: &[(&str, u32)] = &[{}];",
        el.iter().map(|(k, c)| format!("({}, {c})", q(k))).collect::<Vec<_>>().join(", ")
    )
    .unwrap();
    let mut st: Vec<(String, u64)> =
        db["slice_tags"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap())).collect();
    st.sort_by_key(|(_, c)| *c);
    writeln!(
        o,
        "pub static SLICE_TAGS: &[(&str, u32)] = &[{}];",
        st.iter().map(|(k, c)| format!("({}, {c})", q(k))).collect::<Vec<_>>().join(", ")
    )
    .unwrap();

    // ---- diagnostics -----------------------------------------------------------------------
    let diags = db["diagnostics"].as_object().unwrap();
    let mut codes: Vec<(String, String, String)> =
        diags.iter().map(|(k, v)| (camel(k), k.clone(), v.as_str().unwrap().to_string())).collect();
    for n in 1..=5 {
        let k = format!("OTS-E-ASM-00{n}");
        codes.push((camel(&k), k, String::new()));
    }
    writeln!(o, "#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]\npub enum Code {{").unwrap();
    for (c, _, _) in &codes {
        writeln!(o, "    {c},").unwrap();
    }
    writeln!(
        o,
        "}}\nimpl Code {{\n    pub const ALL: &'static [Code] = &[{}];",
        codes.iter().map(|(c, _, _)| format!("Code::{c}")).collect::<Vec<_>>().join(", ")
    )
    .unwrap();
    writeln!(o, "    pub fn as_str(self) -> &'static str {{ match self {{").unwrap();
    for (c, k, _) in &codes {
        writeln!(o, "        Code::{c} => {},", q(k)).unwrap();
    }
    writeln!(o, "    }} }}\n    /// The one-line meaning from encoding.json `diagnostics` (empty for the assembler codes).\n    pub fn meaning(self) -> &'static str {{ match self {{").unwrap();
    for (c, _, m) in &codes {
        writeln!(o, "        Code::{c} => {},", q(m)).unwrap();
    }
    writeln!(o, "    }} }}\n}}").unwrap();

    // ---- operand kinds -----------------------------------------------------------------------
    let kinds = db["operand_kinds"].as_object().unwrap();
    let kind_names: Vec<&String> = kinds.keys().collect();
    writeln!(o, "pub static KINDS: &[Kind] = &[").unwrap();
    for (name, k) in kinds {
        let syntax = match k["syntax"].as_str().unwrap() {
            "name" => "Name",
            "stream" => "Stream",
            "dec" => "Dec",
            "hex" => "Hex",
            other => panic!("operand kind {name}: unknown syntax {other}"),
        };
        let names: Vec<String> = k
            .get("names")
            .map(|n| n.as_object().unwrap().iter().map(|(a, b)| format!("({}, {})", q(a), b)).collect())
            .unwrap_or_default();
        writeln!(
            o,
            "    Kind {{ name: {}, syntax: Syntax::{syntax}, names: &[{}], signed: {}, symbol: {}, group: {} }},",
            q(name),
            names.join(", "),
            k.get("signed").is_some_and(|x| x.as_bool().unwrap()),
            k.get("symbol").is_some_and(|x| x.as_bool().unwrap()),
            k.get("group").map_or(1, |g| g.as_u64().unwrap())
        )
        .unwrap();
    }
    writeln!(o, "];").unwrap();

    // ---- queues and units ---------------------------------------------------------------------
    let mut names: Vec<(String, u64)> =
        db["queues"]["names"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap())).collect();
    names.sort_by_key(|(_, q)| if *q == control { u64::MAX } else { *q });
    writeln!(
        o,
        "pub static QUEUE_NAMES: &[(&str, u16)] = &[{}];",
        names.iter().map(|(k, v)| format!("({}, {v})", q(k))).collect::<Vec<_>>().join(", ")
    )
    .unwrap();
    let al: Vec<(String, u64)> =
        db["queues"]["aliases"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap())).collect();
    writeln!(
        o,
        "pub static QUEUE_ALIASES: &[(&str, u16)] = &[{}];",
        al.iter().map(|(k, v)| format!("({}, {v})", q(k))).collect::<Vec<_>>().join(", ")
    )
    .unwrap();
    let units: Vec<(&String, &Value)> = db["units"].as_object().unwrap().iter().filter(|(k, _)| *k != "text").collect();
    let qbit = |qv: u64| if qv == control { 144 } else { qv as usize };
    let set_of = |v: &Value| {
        let mut s = [0u64; 3];
        for x in v.as_array().unwrap() {
            let b = qbit(x.as_u64().unwrap());
            s[b / 64] |= 1 << (b % 64);
        }
        s
    };
    writeln!(o, "pub static UNITS: &[(&str, &[u16])] = &[").unwrap();
    for (k, v) in &units {
        writeln!(
            o,
            "    ({}, &{:?}),",
            q(k),
            v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect::<Vec<_>>()
        )
        .unwrap();
    }
    writeln!(o, "];").unwrap();
    writeln!(
        o,
        "pub static CONTROL_INSTRUCTIONS: &[&str] = &[{}];",
        db["control_channel"]["instructions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| q(x.as_str().unwrap()))
            .collect::<Vec<_>>()
            .join(", ")
    )
    .unwrap();

    // ---- memory slots, AGT -------------------------------------------------------------------
    let ms = &db["mem_slots"];
    let slot_names: Vec<String> = {
        let mut v: Vec<String> = Vec::new();
        for names in ms["slot_of"].as_object().unwrap().values() {
            for n in names.as_array().unwrap() {
                let n = n.as_str().unwrap().to_string();
                if !v.contains(&n) {
                    v.push(n);
                }
            }
        }
        v
    };
    assert!(slot_names.len() <= 8, "mem_slots: more than 8 slots");
    writeln!(
        o,
        "pub static MEM_SLOT_NAMES: &[&str] = &[{}];",
        slot_names.iter().map(|s| q(s)).collect::<Vec<_>>().join(", ")
    )
    .unwrap();
    writeln!(
        o,
        "pub static MEM_SLOT_PER_THREAD: &[bool] = &[{}];",
        slot_names
            .iter()
            .map(|s| ms["per_thread"].as_array().unwrap().iter().any(|x| x == s.as_str()).to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
    .unwrap();
    writeln!(
        o,
        "pub static MEM_SLOT_PATH: &[Option<u8>] = &[{}];",
        slot_names
            .iter()
            .map(|s| ms["paths"].get(s).map_or("None".to_string(), |p| format!("Some({p})")))
            .collect::<Vec<_>>()
            .join(", ")
    )
    .unwrap();
    writeln!(o, "pub const MEM_THREADS: u32 = {};", ms["threads"]).unwrap();
    let mut slot_q = [0u64; 3];
    for u in ms["units"].as_array().unwrap() {
        let s = set_of(&db["units"][u.as_str().unwrap()]);
        for i in 0..3 {
            slot_q[i] |= s[i];
        }
    }
    writeln!(o, "pub const MEM_SLOT_QUEUES: [u64; 3] = {slot_q:?};").unwrap();
    let agt = &db["agt"];
    writeln!(o, "pub const AGT_LEVELS: u32 = {};", agt["levels"]).unwrap();

    // ---- stations ------------------------------------------------------------------------------
    let stn = db["stations"].as_object().unwrap();
    let sx = |v: &Value| match v.as_str() {
        None => "None".to_string(),
        Some(e) => {
            let (key, quad) = if let Some((k, _)) = e.split_once(" - quad") {
                (k, -1)
            } else if let Some((k, _)) = e.split_once(" + quad") {
                (k, 1)
            } else {
                (e, 0)
            };
            format!("Some(StationExpr {{ key: {}, quad: {quad} }})", q(key.trim()))
        }
    };
    writeln!(o, "pub static STATIONS: &[Station] = &[").unwrap();
    for (k, v) in stn.iter().filter(|(k, _)| *k != "text") {
        writeln!(
            o,
            "    Station {{ unit: {}, read: [{}, {}], write: [{}, {}] }},",
            q(k),
            sx(&v["read"]["E"]),
            sx(&v["read"]["W"]),
            sx(&v["write"]["E"]),
            sx(&v["write"]["W"])
        )
        .unwrap();
    }
    writeln!(o, "];").unwrap();

    // ---- entries -------------------------------------------------------------------------------
    let ents = db["instructions"].as_array().unwrap();
    let max_vars = ents.iter().map(|e| e["variables"].as_array().unwrap().len()).max().unwrap();
    // the `max_expr` operands (every entry's, in catalogue order): the profile variable and the offset of each. An `Isa` evaluates each bound
    // once (`variable - offset`) and `MaxExpr::idx` indexes them, so a decode does no lookup by name
    let mut bounds: Vec<(String, u32)> = Vec::new();
    for e in ents {
        for v in e["variables"].as_array().unwrap() {
            if let Some(x) = v.get("max_expr") {
                let s = x.as_str().unwrap();
                let (n, off) = s.split_once('-').map_or((s, 0u32), |(n, o)| (n, o.trim().parse().unwrap()));
                bounds.push((n.trim().to_string(), off));
            }
        }
    }
    writeln!(
        o,
        "pub static MAX_EXPR_BOUNDS: &[(&str, u32)] = &[{}];",
        bounds.iter().map(|(n, off)| format!("({}, {off})", q(n))).collect::<Vec<_>>().join(", ")
    )
    .unwrap();
    let mut next_bound = 0usize;
    writeln!(o, "pub const MAX_VARS: usize = {max_vars};\npub const ENTRY_COUNT: usize = {};", ents.len()).unwrap();
    writeln!(o, "#[allow(non_camel_case_types)]\n#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]\n#[repr(u8)]\npub enum EntryId {{").unwrap();
    for e in ents {
        writeln!(o, "    {},", e["name"].as_str().unwrap()).unwrap();
    }
    writeln!(
        o,
        "}}\nimpl EntryId {{\n    pub const ALL: [EntryId; {}] = [{}];\n}}",
        ents.len(),
        ents.iter().map(|e| format!("EntryId::{}", e["name"].as_str().unwrap())).collect::<Vec<_>>().join(", ")
    )
    .unwrap();

    let slice_of = ms["slot_of"].as_object().unwrap();
    writeln!(o, "pub static ENTRIES: [Entry; {}] = [", ents.len()).unwrap();
    for e in ents {
        let name = e["name"].as_str().unwrap();
        let width = e["width"].as_u64().unwrap() as u32;
        let m = e["match"].as_str().unwrap();
        assert_eq!(m.len() as u32, width, "{name}: match pattern length");
        let fixed = u64::from_str_radix(&m.replace('-', "0"), 2).unwrap();
        let mask =
            u64::from_str_radix(&m.chars().map(|c| if c == '-' { '0' } else { '1' }).collect::<String>(), 2).unwrap();
        let units_of: Vec<&str> = e["units"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
        let mut qs = [0u64; 3];
        for u in &units_of {
            let s = set_of(&db["units"][*u]);
            for i in 0..3 {
                qs[i] |= s[i];
            }
        }
        qs[2] &= !(1u64 << (144 - 128)); // data queues only; control below
        if db["control_channel"]["instructions"].as_array().unwrap().iter().any(|x| x == name) {
            qs[2] |= 1u64 << (144 - 128);
        }
        let mut var_mask = 0u64;
        let mut vars = String::new();
        for v in e["variables"].as_array().unwrap() {
            let (hi, lo) = span(v["location"].as_str().unwrap());
            let n = hi - lo + 1;
            let (hl, hn) = v.get("hi_location").map_or((0, 0), |h| {
                let (a, b) = span(h.as_str().unwrap());
                (b, a - b + 1)
            });
            let (tl, tn) = v.get("top_location").map_or((0, 0), |h| {
                let (a, b) = span(h.as_str().unwrap());
                (b, a - b + 1)
            });
            assert!(n + hn + tn <= 32, "{name}: field wider than 32 bits");
            var_mask |= ((1u64 << n) - 1) << lo;
            if hn > 0 {
                var_mask |= ((1u64 << hn) - 1) << hl;
            }
            if tn > 0 {
                var_mask |= ((1u64 << tn) - 1) << tl;
            }
            let ext = v.get("extension_codes").map_or("&[]".to_string(), |m| {
                let items: Vec<String> =
                    m.as_object().unwrap().iter().map(|(e, cs)| format!("({}, {})", q(e), arr_u32(cs))).collect();
                format!("&[{}]", items.join(", "))
            });
            let kind = v["kind"].as_str().unwrap();
            let kidx = kind_names
                .iter()
                .position(|k| *k == kind)
                .unwrap_or_else(|| panic!("{name}: unknown operand kind {kind}"));
            let legal = v
                .get("values")
                .or_else(|| v.get("codes"))
                .map_or("None".to_string(), |l| format!("Some({})", arr_u32(l)));
            let opt_i = |k: &str| v.get(k).map_or("None".to_string(), |x| format!("Some({})", x.as_i64().unwrap()));
            let max_expr = v.get("max_expr").map_or("None".to_string(), |x| {
                let s = x.as_str().unwrap();
                let (n, off) = s.split_once('-').map_or((s, 0u32), |(n, o)| (n, o.trim().parse().unwrap()));
                let idx = next_bound;
                next_bound += 1;
                format!("Some(MaxExpr {{ var: {}, idx: {idx}, offset: {off} }})", q(n.trim()))
            });
            let opt_s = |k: &str| {
                v.get(k).map_or("None".to_string(), |x| format!("Some({})", q(x.to_string().trim_matches('"'))))
            };
            write!(vars, "Var {{ name: {}, kind: {kidx}, lo: {lo}, n: {n}, hi_lo: {hl}, hi_n: {hn}, top_lo: {tl}, top_n: {tn}, nbits: {}, ext_codes: {ext}, legal: {legal}, not_values: {}, min: {}, max: {}, max_expr: {max_expr}, bool_values: {}, streams: {}, align: {} }}, ",
                q(v["name"].as_str().unwrap()), n + hn + tn, v.get("not_values").map_or("&[]".to_string(), arr_u32), opt_i("min"), opt_i("max"),
                v.get("bool_values").map_or("None".to_string(), |l| format!("Some({})", arr_u32(l))), opt_s("streams"), opt_s("align")).unwrap();
        }
        let slots = slice_of.get(name).map_or("None".to_string(), |names| {
            let idx: Vec<String> = names
                .as_array()
                .unwrap()
                .iter()
                .map(|n| slot_names.iter().position(|s| s == n.as_str().unwrap()).unwrap().to_string())
                .collect();
            format!("Some(&[{}])", idx.join(", "))
        });
        let has = |list: &Value| list.as_array().unwrap().iter().any(|x| x == name);
        let agt_config = agt["config"]
            .as_array()
            .unwrap()
            .iter()
            .position(|x| x == name)
            .map_or("None".to_string(), |i| format!("Some({i})"));
        writeln!(o, "    Entry {{ id: EntryId::{name}, name: {}, mnemonic: {}, slice: {}, units: &[{}], timing: ots_profile::TimingClass::{}, assembly: {}, width: {width}, fixed: {fixed:#x}, mask: {mask:#x}, var_mask: {var_mask:#x}, vars: &[{vars}], queues: {qs:?}, sram_words: {}, slots: {slots}, all_threads: {}, indirect: {}, agt_loopable: {}, agt_config: {agt_config} }},",
            q(name), q(e["mnemonic"].as_str().unwrap()), q(e["slice"].as_str().unwrap()), units_of.iter().map(|u| q(u)).collect::<Vec<_>>().join(", "),
            match e["timing"].as_str().unwrap() { "CU" => "Cu", "MEM" => "Mem", "VXM" => "Vxm", "MXM_LW" => "MxmLw", "MXM_IW" => "MxmIw", "MXM_ABC" => "MxmAbc", "MXM_ACC" => "MxmAcc", "SXM" => "Sxm", "SXM_CROSS" => "SxmCross", "C2C" => "C2c", t => panic!("timing class {t}") },
            q(e["assembly"].as_str().unwrap()), db["sram_access"]["words"].get(name).map_or(1, |w| w.as_u64().unwrap()),
            has(&ms["all_threads"]), has(&ms["indirect"]), has(&agt["loopable"])).unwrap();
    }
    writeln!(o, "];").unwrap();

    // ---- Insn ----------------------------------------------------------------------------------
    writeln!(o, "#[allow(non_camel_case_types)]\n#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]\npub enum Insn {{")
        .unwrap();
    for e in ents {
        let vs = e["variables"].as_array().unwrap();
        if vs.is_empty() {
            writeln!(o, "    {},", e["name"].as_str().unwrap()).unwrap();
        } else {
            writeln!(
                o,
                "    {} {{ {} }},",
                e["name"].as_str().unwrap(),
                vs.iter()
                    .map(|v| format!("{}: u32", ident(v["name"].as_str().unwrap())))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
            .unwrap();
        }
    }
    writeln!(o, "}}\nimpl Insn {{\n    pub fn id(&self) -> EntryId {{ match self {{").unwrap();
    for e in ents {
        let n = e["name"].as_str().unwrap();
        let pat = if e["variables"].as_array().unwrap().is_empty() { String::new() } else { " { .. }".to_string() };
        writeln!(o, "        Insn::{n}{pat} => EntryId::{n},").unwrap();
    }
    writeln!(o, "    }} }}\n    /// The operand values in the order of the entry's variables (the rest zero).\n    pub fn fields(&self) -> [u32; MAX_VARS] {{ match *self {{").unwrap();
    for e in ents {
        let n = e["name"].as_str().unwrap();
        let vs = e["variables"].as_array().unwrap();
        if vs.is_empty() {
            writeln!(o, "        Insn::{n} => [0; MAX_VARS],").unwrap();
        } else {
            let binds: Vec<String> = vs.iter().map(|v| ident(v["name"].as_str().unwrap())).collect();
            let mut arr: Vec<String> = binds.clone();
            arr.resize(max_vars, "0".into());
            writeln!(o, "        Insn::{n} {{ {} }} => [{}],", binds.join(", "), arr.join(", ")).unwrap();
        }
    }
    writeln!(o, "    }} }}\n    /// The instruction of entry `id` with these operand values (extra values are ignored).\n    pub fn from_fields(id: EntryId, f: &[u32; MAX_VARS]) -> Insn {{ match id {{").unwrap();
    for e in ents {
        let n = e["name"].as_str().unwrap();
        let vs = e["variables"].as_array().unwrap();
        if vs.is_empty() {
            writeln!(o, "        EntryId::{n} => Insn::{n},").unwrap();
        } else {
            let init: Vec<String> =
                vs.iter().enumerate().map(|(i, v)| format!("{}: f[{i}]", ident(v["name"].as_str().unwrap()))).collect();
            writeln!(o, "        EntryId::{n} => Insn::{n} {{ {} }},", init.join(", ")).unwrap();
        }
    }
    writeln!(o, "    }} }}\n}}").unwrap();

    fs::write(PathBuf::from(env::var("OUT_DIR").unwrap()).join("tables.rs"), o).unwrap();
}
