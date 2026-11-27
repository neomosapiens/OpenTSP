//! The extension gate: a word that uses the extension `bf16` decodes in a profile that declares it and does not decode
//! in a profile that does not (OTS-E-DEC-003); the reserved type codes never decode. The model's twin is
//! `unit_tests/test_bf16.sail`.
// model: unit_tests/test_bf16.sail test_bf16_extension_names, test_bf16_mxm_decode, test_bf16_vxm_decode
use ots_isa::{encode, EntryId, Insn, Isa};
use ots_profile::Profile;

const BF16: u32 = 16;

fn isa(extensions: bool) -> Isa {
    let p = if extensions {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../opentsp-arch-spec/config/test/ext_bf16.json");
        Profile::load_file("ext_bf16", path).unwrap()
    } else {
        Profile::load("ref").unwrap()
    };
    assert_eq!(p.has_extension("bf16"), extensions);
    Isa::new(&p)
}

fn add(ty: u32) -> Insn {
    Insn::Add { r#type: ty, dst: 0b001000, lhs: 0b000000, rhs: 0b000100, sg4: 0, lhs_chain: 0, rhs_chain: 0 }
}

fn with(w: u64, hi: u32, lo: u32, v: u64) -> u64 {
    let m = ((1u64 << (hi - lo + 1)) - 1) << lo;
    (w & !m) | (v << lo)
}

// model: unit_tests/test_bf16.sail test_bf16_vxm_decode
#[test]
fn vxm_type_code_16_needs_the_extension() {
    let w = encode(&add(BF16));
    assert_eq!((w >> 17) & 1, 1, "the type-extension bit");
    assert_eq!(((w >> 52) & 3, (w >> 22) & 3), (0, 0));
    assert_eq!(isa(false).decode(64, w), None);
    assert_eq!(isa(true).decode(64, w), Some(add(BF16)));
    // the base words decode on both
    let base = encode(&add(1));
    assert_eq!(base >> 17 & 1, 0);
    for ext in [false, true] {
        assert_eq!(isa(ext).decode(64, base), Some(add(1)));
    }
    // codes 17..31 never decode
    for c in 17..32u64 {
        let w = with(with(with(base, 53, 52, c & 3), 23, 22, (c >> 2) & 3), 17, 17, c >> 4);
        assert_eq!(isa(true).decode(64, w), None, "code {c}");
    }
    // the operations bf16 does not have stay illegal with the extension
    let and =
        encode(&Insn::And { r#type: 0, dst: 0b001000, lhs: 0, rhs: 0b000100, sg4: 0, lhs_chain: 0, rhs_chain: 0 });
    assert_eq!(isa(true).decode(64, with(and, 17, 17, 1)), None);
}

// model: unit_tests/test_bf16.sail test_bf16_vxm_decode
#[test]
fn convert_destination_bit() {
    let cvt = |dtype: u32| encode(&Insn::Convert { r#type: 3, dst: 0b001000, lhs: 0, dtype, lhs_chain: 0 });
    let w = cvt(0);
    let to_bf = with(w, 16, 16, 1);
    assert_eq!(isa(false).decode(64, to_bf), None);
    assert_eq!(
        isa(true).decode(64, to_bf),
        Some(Insn::Convert { r#type: 3, dst: 0b001000, lhs: 0, dtype: BF16, lhs_chain: 0 })
    );
    assert_eq!(encode(&Insn::Convert { r#type: 3, dst: 0b001000, lhs: 0, dtype: BF16, lhs_chain: 0 }), to_bf);
    assert_eq!(isa(true).decode(64, with(w, 27, 24, 1) | 1 << 16), None, "destination code 17");
}

// model: unit_tests/test_bf16.sail test_bf16_mxm_decode
#[test]
fn mxm_type_4_needs_the_extension() {
    let lw = encode(&Insn::LW { r#type: 1, src: 0b11 });
    let bf = with(with(lw, 21, 20, 0), 12, 12, 1);
    assert_eq!(isa(false).decode(32, bf), None);
    assert_eq!(isa(true).decode(32, bf), Some(Insn::LW { r#type: 4, src: 0b11 }));
    assert_eq!(encode(&Insn::LW { r#type: 4, src: 0b11 }), bf);
    for t in [1u64, 2, 3] {
        assert_eq!(isa(true).decode(32, with(with(lw, 21, 20, t), 12, 12, 1)), None, "weight type {}", 4 + t);
    }
    let abc = encode(&Insn::ABC { r#type: 0, src: 0b100100 });
    assert_eq!(isa(false).decode(32, with(abc, 12, 12, 1)), None);
    assert_eq!(isa(true).decode(32, with(abc, 12, 12, 1)), Some(Insn::ABC { r#type: 4, src: 0b100100 }));
    // fp8 has no type-extension bit, and bit 12 of ACC is its accumulator select
    let abc8 = encode(&Insn::ABC8 { fmt: 0, src: 0b100100 });
    assert_eq!(isa(true).decode(32, with(abc8, 12, 12, 1)), None);
    let acc = encode(&Insn::ACC { r#type: 3, dst: 0b001100, mode: 1, acc: 0 });
    assert!(isa(false).decode(32, with(acc, 12, 12, 1)).is_some());
    assert_eq!(Insn::ABC { r#type: 4, src: 0 }.id(), EntryId::ABC);
}

// model: unit_tests/test_bf16.sail test_bf16_extension_names
#[test]
fn profile_validation_of_extensions() {
    let text = |ext: &str| {
        let ref_json =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../opentsp-arch-spec/config/ref.json"))
                .unwrap();
        ref_json.replacen("\"extensions\": []", &format!("\"extensions\": {ext}"), 1)
    };
    assert!(ots_profile::validate_bytes(text("[]").as_bytes()).is_ok());
    assert!(ots_profile::validate_bytes(text("[\"bf16\"]").as_bytes()).is_ok());
    let e = ots_profile::validate_bytes(text("[\"fp4\"]").as_bytes()).unwrap_err();
    assert!(e.message.contains("extensions: fp4 is not an extension of this version"), "{}", e.message);
    let e = ots_profile::validate_bytes(text("[\"bf16\", \"bf16\"]").as_bytes()).unwrap_err();
    assert!(e.message.contains("extensions: bf16 is listed more than once"), "{}", e.message);
}
