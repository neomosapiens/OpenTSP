//! The `extensions` list and `mxm.acc_bits` of a profile: the rules `timing_config_error` of the model states
//! (`postlude/timing_check.sail`), and what a profile that declares nothing looks like.
use ots_profile::{validate_bytes, Profile};

fn text(name: &str) -> String {
    String::from_utf8(Profile::load(name).unwrap().bytes().to_vec()).unwrap()
}

fn with_extensions(base: &str, list: &str) -> String {
    let old = "\"extensions\": [],";
    assert!(base.contains(old), "the shipped profile declares no extension");
    base.replacen(old, &format!("\"extensions\": {list},"), 1)
}

fn message(cfg: &str) -> String {
    validate_bytes(cfg.as_bytes()).unwrap_err().message
}

#[test]
fn the_shipped_profiles_declare_no_extension() {
    for n in ["ref", "mini"] {
        let p = Profile::load(n).unwrap();
        assert!(p.extensions().is_empty(), "{n}");
        assert_eq!(p.uint("mxm.acc_bits"), Some(32), "{n}");
    }
}

#[test]
fn a_declared_extension_is_accepted_with_its_accumulator_width() {
    let base = text("mini");
    assert_eq!(validate_bytes(with_extensions(&base, "[\"block_fp8\"]").as_bytes()), Ok(()));
    let wide =
        with_extensions(&base, "[\"wide_acc\", \"block_fp8\"]").replacen("\"acc_bits\": 32", "\"acc_bits\": 128", 1);
    assert_eq!(validate_bytes(wide.as_bytes()), Ok(()));
}

#[test]
fn an_unknown_or_repeated_extension_is_refused() {
    let base = text("mini");
    assert_eq!(
        message(&with_extensions(&base, "[\"block_fp9\"]")),
        "extensions: block_fp9 is not an extension of this version"
    );
    assert_eq!(
        message(&with_extensions(&base, "[\"block_fp8\", \"block_fp8\"]")),
        "extensions: block_fp8 is listed more than once"
    );
}

// model: unit_tests/test_ext_types.sail test_ext_config_relation
#[test]
fn wide_acc_and_the_accumulator_width_go_together() {
    let base = text("mini");
    let wide_only = with_extensions(&base, "[\"wide_acc\"]");
    assert_eq!(message(&wide_only), "extensions: wide_acc is declared exactly when mxm.acc_bits is 128");
    let bits_only = base.replacen("\"acc_bits\": 32", "\"acc_bits\": 128", 1);
    assert_eq!(message(&bits_only), "extensions: wide_acc is declared exactly when mxm.acc_bits is 128");
    let odd = base.replacen("\"acc_bits\": 32", "\"acc_bits\": 64", 1);
    // the schema's alternatives (32 and 128) are checked before the model's rule, which a direct call of the timing check still reaches
    assert_eq!(message(&odd), "mxm.acc_bits: must be one of 32, 128");
}
