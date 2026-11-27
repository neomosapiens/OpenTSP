//! Profile loading, digests, profile variables and the timing table.
use ots_profile::{Latency, Profile, ProfileError, TimingClass, CATALOGUE};
use sha2::{Digest, Sha256};

#[test]
fn every_constructor_enforces_the_serialized_profile_name_bound() {
    let bytes = Profile::load("mini").unwrap().bytes().to_vec();
    assert!(Profile::from_bytes("abcdefghijklmnop", &bytes).is_ok());
    for name in ["abcdefghijklmnopq", "", "UPPER", "has/path", "caf\u{e9}"] {
        assert!(Profile::from_bytes(name, &bytes).is_err(), "accepted {name:?}");
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
fn digest_of_ref_json() {
    // the first 12 bytes of SHA-256 of config/ref.json byte for byte (assembly.md section 4).
    // The shipped files are still being edited by the specification work, so the expected value is
    // computed from the file of the tree; a known-answer test pins the function itself.
    let path = format!("{}/../../../opentsp-arch-spec/config/ref.json", env!("CARGO_MANIFEST_DIR"));
    let want: [u8; 32] = Sha256::digest(std::fs::read(path).unwrap()).into();
    assert_eq!(Profile::load("ref").unwrap().digest12()[..], want[..12]);
    // SHA-256("abc") = ba7816bf 8f01cfea 414140de 5dae2223 ...
    let tiny = br#"{"spec_version":"0.1","geometry":{"superlanes":1},"sram":{"words_per_slice":256,"bank_bit":7,"addr_bits":13},"icu":{"iq_depth":64}}"#;
    let p = Profile::from_bytes("ref", tiny).unwrap();
    let want: [u8; 32] = Sha256::digest(tiny).into();
    assert_eq!(p.digest12()[..], want[..12]);
    assert_eq!(hex(&Sha256::digest(b"abc")[..4]), "ba7816bf");
}

#[test]
fn embedded_profiles_are_the_files_of_the_tree() {
    for name in ["ref", "mini"] {
        let path = format!("{}/../../../opentsp-arch-spec/config/{name}.json", env!("CARGO_MANIFEST_DIR"));
        let disk = std::fs::read(&path).unwrap();
        assert_eq!(Profile::load(name).unwrap().bytes(), &disk[..], "{name}: embedded copy differs from {path}");
        assert_eq!(Profile::load_file(name, &path).unwrap().digest12(), Profile::load(name).unwrap().digest12());
        let d: [u8; 32] = Sha256::digest(&disk).into();
        assert_eq!(Profile::load(name).unwrap().digest12()[..], d[..12]);
    }
}

#[test]
fn unknown_profiles_and_versions_are_refused() {
    assert!(matches!(Profile::load("nope"), Err(ProfileError::Unknown(_))));
    assert!(matches!(Profile::load("Ref"), Err(ProfileError::Unknown(_))), "names are [a-z0-9_]+");
    assert!(matches!(Profile::load("../ref"), Err(ProfileError::Unknown(_))));
    let r = Profile::from_bytes("ref", br#"{"spec_version": "0.2"}"#);
    assert!(matches!(r, Err(ProfileError::SpecVersion { .. })), "{r:?}");
}

#[test]
fn each_load_error_names_the_profile_and_the_fault() {
    let ref_v = || -> serde_json::Value { serde_json::from_slice(Profile::load("ref").unwrap().bytes()).unwrap() };
    let load = |v: &serde_json::Value| {
        Profile::from_bytes("p", &serde_json::to_vec(v).unwrap()).map(|_| ()).map_err(|e| e.to_string())
    };
    let e = |r: Result<Profile, ProfileError>| r.map(|_| ()).unwrap_err().to_string();
    assert_eq!(e(Profile::load("")), "unknown profile \"\"");
    assert_eq!(e(Profile::load("no_such_profile")), "unknown profile \"no_such_profile\"");
    assert!(e(Profile::from_bytes("p", b"{")).starts_with("profile \"p\": not a JSON object: "));
    assert_eq!(e(Profile::from_bytes("p", b"[1]")), "profile \"p\": not a JSON object: not an object");
    assert_eq!(
        e(Profile::from_bytes("p", br#"{"spec_version": "0.2"}"#)),
        "profile \"p\" is spec version 0.2, not 0.1"
    );
    assert_eq!(e(Profile::from_bytes("p", br#"{"spec_version": 0.1}"#)), "profile \"p\" is spec version 0.1, not 0.1");
    assert_eq!(e(Profile::from_bytes("p", b"{}")), "profile \"p\" is spec version (none), not 0.1");
    for key in ["geometry.superlanes", "sram.words_per_slice", "sram.bank_bit", "sram.addr_bits", "icu.iq_depth"] {
        let mut v = ref_v();
        let (sec, k) = key.split_once('.').unwrap();
        v[sec][k] = serde_json::json!(-1);
        assert_eq!(load(&v), Err(format!("profile \"p\": {key} is missing or not an unsigned integer")));
        v[sec].as_object_mut().unwrap().remove(k);
        assert_eq!(load(&v), Err(format!("profile \"p\": {key} is missing or not an unsigned integer")));
    }
    // vlanes = 16 * geometry.superlanes must not overflow
    let mut v = ref_v();
    v["geometry"]["superlanes"] = serde_json::json!(u64::MAX / 2);
    assert_eq!(load(&v), Err("profile \"p\": vlanes is missing or not an unsigned integer".into()));
}

#[test]
fn load_file_needs_a_valid_name_and_a_readable_file() {
    let dir = std::env::temp_dir().join(format!("ots-profile-load-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let e = |r: Result<Profile, ProfileError>| r.map(|_| ()).unwrap_err();
    assert!(matches!(e(Profile::load_file("x", dir.join("absent.json"))), ProfileError::Unknown(n) if n == "x"));
    assert!(matches!(e(Profile::load_file("Bad", dir.join("absent.json"))), ProfileError::Unknown(_)));
    // the path exists but is a directory: an I/O error that names the path
    let io = e(Profile::load_file("x", &dir));
    assert!(matches!(io, ProfileError::Io(_)), "{io:?}");
    assert!(io.to_string().starts_with(&dir.display().to_string()), "{io}");
    let f = dir.join("p.json");
    std::fs::write(&f, Profile::load("mini").unwrap().bytes()).unwrap();
    let p = Profile::load_file("mine", &f).unwrap();
    assert_eq!((p.name(), p.vlanes()), ("mine", 16));
    assert_eq!(p.value()["spec_version"], "0.1");
    assert_eq!(p.get("geometry.superlanes").and_then(|v| v.as_u64()), Some(1));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn validate_runs_the_schema_then_the_timing_rules() {
    assert_eq!(Profile::load("ref").unwrap().validate(), Ok(()));
    let nul = ots_profile::validate_bytes(b"{\"spec_version\": \"0.1\"}\0").unwrap_err();
    assert_eq!((nul.message.as_str(), nul.schema_error), ("NUL byte in file", false));
    assert_eq!(nul.to_string(), "NUL byte in file");
}

#[test]
fn profile_variables_and_accessors() {
    let r = Profile::load("ref").unwrap();
    assert_eq!(r.variable("vlanes"), Some(320));
    assert_eq!((r.vlanes(), r.words_per_slice(), r.bank_bit(), r.iq_depth()), (320, 8192, 12, 1280));
    let m = Profile::load("mini").unwrap();
    assert_eq!(m.variable("vlanes"), Some(16));
    assert_eq!((m.vlanes(), m.words_per_slice(), m.bank_bit()), (16, 256, 7));
    assert_eq!(m.variable("nope"), None);
    assert_eq!(m.uint("c2c.link_latency"), Some(216));
    assert_eq!(m.uint("geometry.nope"), None);
}

#[test]
fn a_profile_with_unknown_extra_keys_still_loads_generically() {
    // the assembler never runs the schema check; extra keys the specification may add
    // (geometry.mem_slices_per_hemisphere, mxm.rows) are visible through the generic accessors
    let mut v: serde_json::Value = serde_json::from_slice(Profile::load("ref").unwrap().bytes()).unwrap();
    v["geometry"]["mem_slices_per_hemisphere"] = serde_json::json!(44);
    v["mxm"] = serde_json::json!({"rows": 320});
    v["sram"]["words_per_slice"] = serde_json::json!(4096);
    let p = Profile::from_bytes("ref", &serde_json::to_vec(&v).unwrap()).unwrap();
    assert_eq!(p.uint("geometry.mem_slices_per_hemisphere"), Some(44));
    assert_eq!(p.uint("mxm.rows"), Some(320));
    assert_eq!(p.words_per_slice(), 4096);
}

#[test]
fn the_shipped_profiles_carry_the_keys_and_override_rows_through_cr_22() {
    for name in ["ref", "mini"] {
        let p = Profile::load(name).unwrap();
        let sl = p.superlanes();
        assert_eq!(
            p.uint("timing.sxm_map_ready"),
            Some(sl),
            "{name}: LoadMap's map-ready delay is the superlane count"
        );
        assert_eq!(p.uint("timing.mxm_abc.occupancy_f16"), Some(1), "{name}");
        assert_eq!(p.uint("timing.c2c.rtdeskew_floor"), Some(1), "{name}");
        assert_eq!(p.uint("icu.fetch_staging_depth"), Some(8), "{name}");
        assert_eq!(p.uint("c2c.cycles_per_vector"), Some(24), "{name}");
        let t = p.timing_table();
        let row = |n: &str| t.entry(n).unwrap();
        for (n, f, k) in
            [("Transpose", 5, 0), ("Permute", 20, 0), ("Exp", 5, 2), ("Log", 5, 2), ("RSqrt", 5, 2), ("TanH", 5, 2)]
        {
            assert_eq!(row(n), Latency { d_func: f, d_skew: k }, "{name}: {n}");
        }
        let rows = p.get("timing.overrides").and_then(|v| v.as_array()).unwrap().len();
        assert_eq!(rows, 6, "{name}: the six explicit rows");
    }
}

#[test]
fn timing_table_per_catalogue_entry() {
    let p = Profile::load("ref").unwrap();
    let t = p.timing_table();
    assert_eq!((t.len(), t.is_empty()), (CATALOGUE.len(), false));
    let row = |n: &str| t.entry(n).unwrap();
    assert_eq!(row("Add"), Latency { d_func: 5, d_skew: 2 });
    assert_eq!(row("Read"), Latency { d_func: 5, d_skew: 0 });
    assert_eq!(row("LW"), Latency { d_func: 20, d_skew: 0 });
    assert_eq!(row("Shift"), Latency { d_func: 20, d_skew: 0 });
    assert!(t.entry("Frobnicate").is_none());
    let class_of = |n: &str| CATALOGUE.iter().find(|e| e.name == n).unwrap().class;
    assert_eq!(class_of("IWS"), TimingClass::MxmIw);
    assert_eq!(class_of("Permute"), TimingClass::SxmCross);
    // an override row replaces the class row
    let mut v: serde_json::Value = serde_json::from_slice(p.bytes()).unwrap();
    v["timing"]["overrides"] = serde_json::json!([{"entry": "Add", "d_func": 9, "d_skew": 3}]);
    let q = Profile::from_bytes("ref", &serde_json::to_vec(&v).unwrap()).unwrap();
    assert_eq!(q.timing_table().entry("Add").unwrap(), Latency { d_func: 9, d_skew: 3 });
    assert_eq!(q.timing_table().entry("Sub").unwrap(), Latency { d_func: 5, d_skew: 2 });
}

#[test]
fn a_duplicate_key_is_still_loadable_by_the_assembler_path_last_wins() {
    // tsp_as reads profiles with json.load (last duplicate wins, no schema); validate() refuses it
    let text = br#"{"spec_version": "0.1", "spec_version": "0.1", "geometry": {"superlanes": 1}, "sram": {"words_per_slice": 256, "bank_bit": 7, "addr_bits": 13}, "icu": {"iq_depth": 64}}"#;
    let p = Profile::from_bytes("ref", text).unwrap();
    assert_eq!(p.vlanes(), 16);
    assert!(p.validate().unwrap_err().to_string().contains("duplicate key"));
}

#[test]
fn a_profile_without_diag_depth_reads_the_default_and_keeps_its_own_digest() {
    let mini: serde_json::Value = serde_json::from_slice(Profile::load("mini").unwrap().bytes()).unwrap();
    assert_eq!(mini["ras"]["diag_depth"], 16, "the shipped profiles carry the default depth");
    let mut older = mini.clone();
    older["ras"].as_object_mut().unwrap().remove("diag_depth");
    let bytes = serde_json::to_vec(&older).unwrap();
    // loads and validates; every reader gets the default
    let p = Profile::from_validated_bytes(&bytes).expect("a profile written before the key existed loads");
    assert_eq!(p.uint("ras.diag_depth"), Some(16));
    assert_eq!(p.get("ras.diag_depth").and_then(|v| v.as_u64()), Some(16));
    // the digest stays the one of the bytes of the file, which is not the shipped file
    let want: [u8; 32] = Sha256::digest(&bytes).into();
    assert_eq!(p.digest12()[..], want[..12]);
    // a value that is given is kept, and the bounds still hold
    for (depth, ok) in [(1, true), (7, true), (64, true), (0, false), (65, false)] {
        let mut c = mini.clone();
        c["ras"]["diag_depth"] = depth.into();
        let r = Profile::from_validated_bytes(&serde_json::to_vec(&c).unwrap());
        assert_eq!(r.is_ok(), ok, "diag_depth {depth}");
        match r {
            Ok(p) => assert_eq!(p.uint("ras.diag_depth"), Some(depth)),
            Err(e) => assert!(e.message.contains("ras.diag_depth"), "{}", e.message),
        }
    }
}

#[test]
fn timing_table_per_element_type() {
    // a row named `<Entry>.<type>` is the row of that element type: it beats the entry's own row, which beats the class row
    let p = Profile::load("ref").unwrap();
    let mut v: serde_json::Value = serde_json::from_slice(p.bytes()).unwrap();
    v["timing"]["overrides"].as_array_mut().unwrap().extend([
        serde_json::json!({"entry": "Max.i8", "d_func": 3, "d_skew": 2}),
        serde_json::json!({"entry": "Min", "d_func": 7, "d_skew": 1}),
        serde_json::json!({"entry": "Min.u8", "d_func": 2, "d_skew": 1}),
        serde_json::json!({"entry": "Conv_F16_I8.f16", "d_func": 4, "d_skew": 1}),
    ]);
    let q = Profile::from_bytes("ref", &serde_json::to_vec(&v).unwrap()).unwrap();
    assert_eq!(q.timing_config_error(), None);
    let t = q.timing_table();
    let l = |f, k| Latency { d_func: f, d_skew: k };
    assert_eq!(t.entry_typed("Max", "i8").unwrap(), l(3, 2));
    assert_eq!(t.entry_typed("Max", "f32").unwrap(), l(5, 2), "no row for the type: the class row");
    assert_eq!(t.entry("Max").unwrap(), l(5, 2), "the untyped row is not moved by a typed one");
    assert_eq!(t.entry_typed("Min", "i8").unwrap(), l(7, 1), "the entry's row covers its other types");
    assert_eq!(t.entry_typed("Min", "u8").unwrap(), l(2, 1));
    assert_eq!(t.entry_typed("Conv_F16_I8", "f16").unwrap(), l(4, 1));
    assert!(t.entry_typed("Max", "q9").is_none(), "not an element type");
    assert!(t.entry_typed("Read", "i8").is_none(), "Read takes no element type");
    // the shipped profiles carry no typed row: every typed lookup is the untyped one
    for name in ["ref", "mini"] {
        let t = Profile::load(name).unwrap().timing_table();
        assert_eq!(t.entry_typed("Max", "i8"), t.entry("Max"), "{name}");
    }
}

#[test]
fn source_path_is_set_only_for_a_config_file() {
    let by_name = Profile::load("ref").unwrap();
    assert_eq!(by_name.source_path(), None);
    assert_eq!(by_name.describe(), "profile ref");
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../opentsp-arch-spec/config/ref.json");
    let by_file = Profile::load_file("ref", path).unwrap();
    assert_eq!(by_file.source_path(), Some(path));
    assert_eq!(by_file.describe(), format!("file {path}"));
}
