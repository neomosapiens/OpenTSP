//! The optional extensions wide_acc (ACC type x128) and block_fp8 (MXM type bfp8): the decoder accepts their encodings only on a
//! profile that declares them (`slices/mxm/mxm_insts.sail` `wtype_declared`, `acctype_declared`; test
//! `test_ext_decode_gate` of `unit_tests/test_ext_types.sail`).
use ots_isa::{encode, Insn, Isa};
use ots_profile::Profile;

fn ref_isa() -> Isa {
    Isa::new(&Profile::load("ref").unwrap())
}

fn ext_isa() -> Isa {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../opentsp-arch-spec/config/test/mini_ext.json");
    Isa::new(&Profile::load_file("mini_ext", path).unwrap())
}

/// An MXM word: slice 3, the opcode, the type (bits 21..20), the stream field, the mode bit 13 and the bit 12.
fn word(op: u64, ty: u64, s: u64, mode: u64, b12: u64) -> u64 {
    (3 << 28) | (op << 22) | (ty << 20) | (s << 14) | (mode << 13) | (b12 << 12)
}

// model: unit_tests/test_ext_types.sail test_ext_decode_gate
#[test]
fn acc_x128_needs_the_extension() {
    let w = word(3, 0, 8, 0, 0);
    assert!(ref_isa().decode(32, w).is_none());
    let i = ext_isa().decode(32, w).expect("acc.x128 decodes on a profile that declares wide_acc");
    assert_eq!(i, Insn::ACC { r#type: 0, dst: 8, mode: 0, acc: 0 });
    assert_eq!(encode(&i), w);
    // the other unassigned type code stays illegal
    assert!(ext_isa().decode(32, word(3, 1, 8, 0, 0)).is_none());
}

#[test]
fn bfp8_needs_the_extension() {
    let words = [word(0, 1, 32, 0, 1), word(1, 1, 0, 0, 1), word(1, 1, 32, 1, 1), word(2, 1, 32 | 24, 0, 1)];
    for w in words {
        assert!(ref_isa().decode(32, w).is_none(), "{w:#x} must not decode without block_fp8");
        let i = ext_isa().decode(32, w).unwrap_or_else(|| panic!("{w:#x} decodes with block_fp8"));
        assert_eq!(encode(&i), w);
    }
    assert_eq!(ext_isa().decode(32, words[3]), Some(Insn::ABC { r#type: 5, src: 32 | 24 }));
    // the extension bit with any other type, and on ABC8, stays illegal
    for ty in [0, 2, 3] {
        assert!(ext_isa().decode(32, word(2, ty, 32 | 24, 0, 1)).is_none());
    }
    assert!(ext_isa().decode(32, word(2, 3, 32 | 24, 1, 1)).is_none());
}

#[test]
fn the_declaration_is_part_of_the_profile() {
    assert!(!Profile::load("ref").unwrap().has_extension("wide_acc"));
    assert!(!Profile::load("mini").unwrap().has_extension("block_fp8"));
}
