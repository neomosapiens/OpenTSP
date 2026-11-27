// Generates the catalogue table (entry name, slice, timing class) and the profile variables from
// isa-db/generated/encoding.json, so that nothing about the catalogue is written twice.
// OTS_ENCODING_JSON overrides the path.
use std::{env, fs, path::PathBuf};

fn main() {
    let default = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("../../../opentsp-arch-spec/isa-db/generated/encoding.json");
    println!("cargo:rerun-if-env-changed=OTS_ENCODING_JSON");
    let path = env::var("OTS_ENCODING_JSON").map(PathBuf::from).unwrap_or(default);
    println!("cargo:rerun-if-changed={}", path.display());
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let db: serde_json::Value = serde_json::from_str(&text).expect("encoding.json is not JSON");
    assert_eq!(db["spec_version"], "0.1", "encoding.json: spec version is not 0.1");

    let mut out = String::from("pub static CATALOGUE: &[CatalogueEntry] = &[\n");
    for e in db["instructions"].as_array().expect("instructions") {
        let class = match e["timing"].as_str().unwrap() {
            "CU" => "Cu",
            "MEM" => "Mem",
            "VXM" => "Vxm",
            "MXM_LW" => "MxmLw",
            "MXM_IW" => "MxmIw",
            "MXM_ABC" => "MxmAbc",
            "MXM_ACC" => "MxmAcc",
            "SXM" => "Sxm",
            "SXM_CROSS" => "SxmCross",
            "C2C" => "C2c",
            other => panic!("encoding.json: unknown timing class {other}"),
        };
        // the element types the entry's `type` operand takes (a timing row may be keyed by one of them)
        let codes: Vec<String> = e["variables"]
            .as_array()
            .expect("variables")
            .iter()
            .filter(|v| v["name"] == "type" && v["kind"] == "eltype_code")
            .flat_map(|v| v["codes"].as_array().expect("codes").iter().map(|c| c.as_u64().unwrap().to_string()))
            .collect();
        out += &format!(
            "    CatalogueEntry {{ name: {:?}, slice: {:?}, class: TimingClass::{class}, type_codes: &[{}] }},\n",
            e["name"].as_str().unwrap(),
            e["slice"].as_str().unwrap(),
            codes.join(", ")
        );
    }
    out += "];\n\n/// The element types of the catalogue: the name the assembly writes and the code of the `type` operand.\npub static ELTYPES: &[(&str, u8)] = &[\n";
    for (name, code) in db["eltype_codes"].as_object().expect("eltype_codes") {
        out += &format!("    ({:?}, {}),\n", name, code.as_u64().unwrap());
    }
    out += "];\n\npub static PROFILE_VARIABLES: &[(&str, &str)] = &[\n";
    let mut vars: Vec<_> = db["profile_variables"].as_object().expect("profile_variables").iter().collect();
    vars.sort_by_key(|(k, _)| k.to_string());
    for (k, v) in vars {
        out += &format!("    ({:?}, {:?}),\n", k, v["expr"].as_str().unwrap());
    }
    out += "];\n";

    // the keys a profile may leave out: the schema's `default` of each property of a section, as (section, key, JSON text)
    let schema_path =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../../opentsp-arch-spec/config/schema.json");
    println!("cargo:rerun-if-changed={}", schema_path.display());
    let schema: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&schema_path).expect("schema.json")).expect("schema.json is not JSON");
    out += "\n/// The default of every schema property that has one, as (section, key, JSON text).\npub static SCHEMA_DEFAULTS: &[(&str, &str, &str)] = &[\n";
    for (section, sch) in schema["properties"].as_object().expect("schema properties") {
        let Some(keys) = sch.get("properties").and_then(|k| k.as_object()) else { continue };
        for (key, ks) in keys {
            if let Some(d) = ks.get("default") {
                out += &format!("    ({:?}, {:?}, {:?}),\n", section, key, d.to_string());
            }
        }
    }
    out += "];\n";
    fs::write(PathBuf::from(env::var("OUT_DIR").unwrap()).join("catalogue.rs"), out).unwrap();
}
