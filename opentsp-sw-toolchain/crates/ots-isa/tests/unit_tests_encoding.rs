//! Ports of the encoding unit tests of the Sail model (`unit_tests/test_encdec.sail`,
//! `test_slices.sail`, `test_c2c.sail`): the bit positions of `encoding.md` against the generated
//! encoder and decoder. `encode` stands for `encdec32` / `encdec64`; an `ILLEGAL` word is a word
//! that `Isa::decode` refuses. Operand values are the field values of the catalogue (a `stream`
//! operand is the 6-bit reference, a `stream_group16` operand the reference over 16).
use ots_isa::{encode, placement, unit_queues, EntryId, Insn, Isa};
use ots_profile::Profile;

fn isa() -> Isa {
    Isa::new(&Profile::load("ref").unwrap())
}

/// Bits `hi..=lo` of `w`.
fn bits(w: u64, hi: u32, lo: u32) -> u64 {
    (w >> lo) & ((1u64 << (hi - lo + 1)) - 1)
}

/// `w` with bits `hi..=lo` replaced by `v` (Sail `[w with hi .. lo = v]`).
fn with(w: u64, hi: u32, lo: u32, v: u64) -> u64 {
    let m = ((1u64 << (hi - lo + 1)) - 1) << lo;
    (w & !m) | (v << lo)
}

/// Concatenate `(value, width)` fields, first field most significant (Sail `@`).
fn cat(fields: &[(u64, u32)]) -> u64 {
    fields.iter().fold(0, |w, &(v, n)| {
        assert!(v < 1 << n, "{v:#x} does not fit {n} bits");
        (w << n) | v
    })
}

fn ill32(w: u64) -> bool {
    isa().decode(32, w).is_none()
}

fn ill64(w: u64) -> bool {
    isa().decode(64, w).is_none()
}

fn round_trips32(i: Insn) -> bool {
    isa().decode(32, encode(&i)) == Some(i)
}

fn round_trips64(i: Insn) -> bool {
    isa().decode(64, encode(&i)) == Some(i)
}

/// `issued_by(instruction, unit)`: the entry is placed on every queue of the unit, or on none.
fn issued_by(id: EntryId, unit: &str) -> bool {
    let placed: Vec<bool> = unit_queues(unit).unwrap().iter().map(|&q| placement(id, q).is_none()).collect();
    assert!(placed.iter().all(|&p| p == placed[0]), "{id:?} on {unit}: {placed:?}");
    placed[0]
}

const I8: u32 = 0;
const F16: u32 = 1;
/// The MXM weight type u8 (`mxm_wtype`), not the element type of the same name.
const W_U8: u32 = 2;
const W_FP8: u32 = 3;
const I32: u32 = 2;
const F32: u32 = 3;
const U8: u32 = 4;
const B8: u32 = 8;
const B16: u32 = 9;

fn bin(make: fn(u32, u32, u32, u32) -> Insn, ty: u32) -> Insn {
    make(ty, 0b001000, 0b000000, 0b000100)
}

fn add(ty: u32, dst: u32, lhs: u32, rhs: u32) -> Insn {
    Insn::Add { r#type: ty, dst, lhs, rhs, sg4: 0, lhs_chain: 0, rhs_chain: 0 }
}

fn and(ty: u32, dst: u32, lhs: u32, rhs: u32) -> Insn {
    Insn::And { r#type: ty, dst, lhs, rhs, sg4: 0, lhs_chain: 0, rhs_chain: 0 }
}

// model: unit_tests/test_encdec.sail test_mem_read_layout
#[test]
fn test_mem_read_layout() {
    let w = encode(&Insn::Read { addr: 0b1_0000_0000_0011, dst: 0b100101, thread: 0 });
    assert_eq!(bits(w, 63, 60), 0x1);
    assert_eq!(bits(w, 59, 54), 0b000000);
    assert_eq!(bits(w, 44, 32), 0b1_0000_0000_0011);
    assert_eq!(bits(w, 31, 26), 0b100101);
    assert!(bits(w, 53, 53) == 0 && bits(w, 52, 45) == 0 && bits(w, 25, 0) == 0);
}

// model: unit_tests/test_encdec.sail test_mem_roundtrip_and_reserved
#[test]
fn test_mem_roundtrip_and_reserved() {
    let i = Insn::Write { addr: 0b1_1111_1111_1111, src: 0b111111, thread: 0 };
    let w = encode(&i);
    assert!(round_trips64(i));
    // A reserved bit set does not decode to a MEM instruction.
    assert!(ill64(w | 0x0000_0000_0000_0001));
}

// model: unit_tests/test_encdec.sail test_cu_nop_zero_is_illegal
#[test]
fn test_cu_nop_zero_is_illegal() {
    assert!(ill32(0x0000_0000), "NOP 0 accepted");
    assert_eq!(isa().decode(32, 0x0000_0005), Some(Insn::NOP { n: 0x0005 }));
}

// model: unit_tests/test_encdec.sail test_vxm_illegal_type_and_mode
#[test]
fn test_vxm_illegal_type_and_mode() {
    // Add.i8 E:8 <- E:0, E:4 decodes.
    let ok = encode(&bin(add, I8));
    assert!(!ill64(ok));
    // Add with a non-zero mode field does not decode.
    assert!(ill64(ok | 0x0000_0000_0300_0000));
    // Add_sat.f16 does not decode (saturating ops are integer-only).
    assert!(ill64(cat(&[
        (0x2, 4),
        (0b000001, 6),
        (0b01, 2),
        (0b001000, 6),
        (0b000000, 6),
        (0b000100, 6),
        (0, 6),
        (0, 4),
        (0, 24)
    ])));
    // Cmp mode 6..15 does not decode; mode 5 does.
    let cmp = encode(&Insn::Cmp {
        r#type: I32,
        dst: 0b001000,
        lhs: 0b000000,
        rhs: 0b000100,
        mode: 5,
        sg4: 0,
        lhs_chain: 0,
        rhs_chain: 0,
    });
    assert!(!ill64(cmp));
    assert!(ill64((cmp & !0x0000_0000_0F00_0000) | 0x0000_0000_0600_0000));
    // Reduce op 3 does not decode.
    let red = encode(&Insn::Reduce { r#type: I8, dst: 0b001000, lhs: 0b000000, mode: 2, lhs_chain: 0 });
    assert!(!ill64(red));
    assert!(ill64((red & !0x0000_0000_0F00_0000) | 0x0000_0000_0300_0000));
    // A conversion with the wrong source type does not decode.
    let cv = encode(&Insn::Conv_I8_F32 { r#type: I8, dst: 0b000000, lhs: 0b000100, lhs_chain: 0 });
    assert!(!ill64(cv));
    assert!(ill64(cv | 0x0010_0000_0000_0000));
}

// model: unit_tests/test_slices.sail test_c2c_layout
#[test]
fn test_c2c_layout() {
    let send = Insn::Send { chip_id: 0b101010, stream: 0b100011, link: 0 };
    let w = encode(&send);
    assert_eq!(w, cat(&[(0x5, 4), (0b000000, 6), (0b101010, 6), (0b100011, 6), (0, 10)]));
    assert!(round_trips32(send));
    let r = encode(&Insn::Receive { chip_id: 0b000001, stream: 0b000010, link: 0 });
    assert_eq!(bits(r, 27, 22), 0b000001);
    // the upper bits of the chip (2..0) and of the link (5..3) are fields since the widening
    assert_eq!(
        isa().decode(32, w | 0x0000_0001),
        Some(Insn::Send { chip_id: 0b101010 | 1 << 6, stream: 0b100011, link: 0 })
    ); // chip bit 6
    assert_eq!(
        isa().decode(32, w | 0x0000_0008),
        Some(Insn::Send { chip_id: 0b101010, stream: 0b100011, link: 0b1 << 4 })
    ); // link bit 4
       // opcode 10 (2..4 are Deskew, Transmit, RuntimeDeskew; 5..8 the host instructions; 9 DeskewRecv)
    assert!(ill32(cat(&[(0x5, 4), (0b001010, 6), (0, 16), (0, 6)])));
}

// model: unit_tests/test_c2c.sail test_c2c_deskewrecv (the layout: Receive's fields with opcode 9)
#[test]
fn test_c2c_deskewrecv_layout() {
    let d = Insn::DeskewRecv { chip_id: 0b000001, stream: 0b000010, link: 0x3 };
    assert_eq!(encode(&d), cat(&[(0x5, 4), (0b001001, 6), (0b000001, 6), (0b000010, 6), (0x3, 4), (0, 6)]));
    assert!(round_trips32(d));
    assert!(round_trips32(Insn::DeskewRecv { chip_id: 0x1a5, stream: 0b000010, link: 0x5d }));
}

// model: unit_tests/test_slices.sail test_host_layout
#[test]
fn test_host_layout() {
    // PciReceive takes S24..S31 only, HostNotify n = 1..15, HostWait any doorbell
    let pr = encode(&Insn::PciReceive { stream: 0b111001 });
    assert_eq!(pr, cat(&[(0x5, 4), (0b000101, 6), (0b000000, 6), (0b111001, 6), (0, 4), (0, 6)]));
    assert!(round_trips32(Insn::PciReceive { stream: 0b111001 }));
    assert!(ill32(cat(&[(0x5, 4), (0b000101, 6), (0b000000, 6), (0b100010, 6), (0, 4), (0, 6)]))); // S2
    assert!(ill32(cat(&[(0x5, 4), (0b000101, 6), (0b000000, 6), (0b010111, 6), (0, 4), (0, 6)]))); // S23
    for s in (24..32).chain(56..64) {
        assert!(round_trips32(Insn::PciReceive { stream: s }), "S{s}");
    }
    assert!(round_trips32(Insn::PciSend { stream: 0b000101 }));
    let hw = encode(&Insn::HostWait { i: 0b000011 });
    assert_eq!(hw, cat(&[(0x5, 4), (0b000111, 6), (0, 16), (0b000011, 6)]));
    assert!(round_trips32(Insn::HostWait { i: 0 }) && round_trips32(Insn::HostWait { i: 63 }));
    assert!(round_trips32(Insn::HostNotify { n: 0x9 }) && round_trips32(Insn::HostNotify { n: 0xf }));
    assert!(ill32(cat(&[(0x5, 4), (0b001000, 6), (0, 16), (0, 2), (0, 4)]))); // HostNotify 0
}

// model: unit_tests/test_slices.sail test_sxm_rotate_transpose_encoding
#[test]
fn test_sxm_rotate_transpose_encoding() {
    let rot = encode(&Insn::Rotate { dst: 0b000000, src: 0b100000, n: 0b0000000011 });
    assert!(bits(rot, 59, 54) == 0b000011 && bits(rot, 39, 30) == 0b0000000011);
    assert!(ill64(with(rot, 39, 30, 0b0000000101)));
    // Transpose: src W:16 and dst E:16 are stream groups (field = reference / 16).
    let tr = Insn::Transpose { dst: 0b010000 / 16, src: 0b110000 / 16 };
    assert!(round_trips64(tr));
    assert!(ill64(with(encode(&tr), 41, 40, 0b01))); // src 0b110001: not 16-aligned
    assert!(issued_by(EntryId::Rotate, "SXM_W_TSP"));
    assert!(!issued_by(EntryId::Transpose, "SXM_W_PRM"));
}

// model: unit_tests/test_slices.sail test_mxm_type_legality
#[test]
fn test_mxm_type_legality() {
    let lw = encode(&Insn::LW { r#type: F16, src: 0b110000 / 16 });
    assert_eq!(bits(lw, 21, 20), 0b01);
    assert!(!ill32(with(lw, 21, 20, 0b10))); // u8 (HC34 p.26)
    assert!(!ill32(with(lw, 21, 20, 0b11))); // fp8
    assert_eq!(isa().decode(32, with(lw, 21, 20, 0b11)), Some(Insn::LW { r#type: W_FP8, src: 0b110000 / 16 }));
    let abc = encode(&Insn::ABC { r#type: 0, src: 0b100100 / 16 });
    assert!(ill32(with(abc, 13, 13, 1))); // mode bit 13 is zero for i8, f16 and u8
    assert_eq!(isa().decode(32, with(abc, 21, 20, 0b11)), Some(Insn::ABC8 { fmt: 0, src: 0b100100 / 16 })); // abc.e4m3
    assert_eq!(
        isa().decode(32, with(with(abc, 21, 20, 0b11), 13, 13, 1)),
        Some(Insn::ABC8 { fmt: 1, src: 0b100100 / 16 })
    );
    assert_eq!(encode(&Insn::ABC8 { fmt: 1, src: 0b100100 / 16 }), with(with(abc, 21, 20, 0b11), 13, 13, 1));
    assert!(ill32(with(lw, 17, 14, 0x1))); // LW source must be 16-aligned
    let acc = encode(&Insn::ACC { r#type: F32, dst: 0b001100, mode: 1, acc: 0 });
    assert_eq!(bits(acc, 13, 13), 1);
    assert!(ill32(with(acc, 21, 20, 0b00)));
}

// model: unit_tests/test_slices.sail test_mxm_u8
#[test]
fn test_mxm_u8() {
    let i = Insn::LW { r#type: W_U8, src: 0b110000 / 16 };
    assert_eq!(bits(encode(&i), 21, 20), 0b10);
    assert!(round_trips32(i));
}

// model: unit_tests/test_slices.sail test_vxm_type_codes
#[test]
fn test_vxm_type_codes() {
    // u8 add: type bits 53..52 = 00, extension 23..22 = 01.
    let i = bin(add, U8);
    let w = encode(&i);
    assert!(bits(w, 53, 52) == 0b00 && bits(w, 23, 22) == 0b01);
    assert!(round_trips64(i));
    // Codes 13..15 (extension 11 with type 01..11) stay reserved; extension 11 with type 00 is code 12, e5m2.
    assert!(ill64(with(with(w, 53, 52, 0b01), 23, 22, 0b11)));
    assert!(!ill64(with(w, 23, 22, 0b11)));
    // bool add does not decode (extension 10 type 00 is b8); bool and does.
    assert!(ill64(with(w, 23, 22, 0b10)));
    assert!(!ill64(encode(&bin(and, B16))));
}

// model: unit_tests/test_slices.sail test_undefined_type_codes
#[test]
fn test_undefined_type_codes() {
    // Undefined type codes (13, 14, 15: extension 11 with type 01..11; the codes 11 and 12 are the FP8 types) do not decode.
    let w = encode(&Insn::Convert { r#type: U8, dst: 0b001000, lhs: 0b000000, dtype: F32, lhs_chain: 0 });
    assert!(!ill64(w));
    assert!(ill64(with(w, 27, 24, 0b1101)));
    assert!(ill64(with(w, 27, 24, 0b1111)));
    assert!(!ill64(with(w, 27, 24, 0b1011)) && !ill64(with(w, 27, 24, 0b1100))); // e4m3, e5m2
    let b = encode(&bin(add, I8));
    assert!(!ill64(with(with(b, 53, 52, 0b11), 23, 22, 0b10))); // code 11: add.e4m3
    assert!(!ill64(with(with(b, 53, 52, 0b00), 23, 22, 0b11))); // code 12: add.e5m2
    for t in [0b01, 0b10, 0b11] {
        assert!(ill64(with(with(b, 53, 52, t), 23, 22, 0b11))); // codes 13..15
    }
    let a = encode(&bin(and, B8));
    assert!(!ill64(a));
    assert!(ill64(with(a, 53, 52, 0b11))); // and.e4m3 is not defined
}

// model: unit_tests/test_slices.sail test_c2c_deskew
#[test]
fn test_c2c_deskew() {
    let w = encode(&Insn::Deskew);
    assert_eq!(w, cat(&[(0x5, 4), (0b000010, 6), (0, 16), (0, 6)]));
}

// model: unit_tests/test_slices.sail test_icu_on_data_queues
#[test]
fn test_icu_on_data_queues() {
    // NOP, Sync, Notify, Repeat, Ifetch, Config are common to all functional slices (HC34 p.27).
    assert!(issued_by(EntryId::Sync, "VXM") && issued_by(EntryId::Notify, "MEM_E") && issued_by(EntryId::NOP, "MXM_W"));
    assert!(issued_by(EntryId::Repeat, "VXM"));
    let r = encode(&Insn::Repeat { n: 0x036, d: 0x1 });
    assert!(bits(r, 27, 22) == 0b000011 && bits(r, 15, 4) == 0x036 && bits(r, 3, 0) == 0x1);
    assert!(ill32(cat(&[(0x0, 4), (0b000011, 6), (0b000000, 6), (0x000, 12), (0x1, 4)]))); // Repeat 0 times does not decode
                                                                                           // d = 0 is legal (the spacing is d + 1, so d = 0 re-issues in consecutive cycles)
    assert!(round_trips32(Insn::Repeat { n: 0x001, d: 0x0 }));
    assert!(round_trips32(Insn::Repeat { n: 0xfff, d: 0xf }));
    assert!(ill32(cat(&[(0x0, 4), (0b000011, 6), (0b000000, 6), (0x000, 12), (0x0, 4)]))); // n = 0 stays illegal
    assert!(round_trips32(Insn::Ifetch { stream: 0b100011 }));
    // Ifetch has one stream operand of any parity: an odd stream is no longer OTS-E-OPR-002
    for s in [0b000100, 0b000101, 0b100100, 0b100101, 0b011111] {
        assert!(round_trips32(Insn::Ifetch { stream: s }), "ifetch stream {s:#b}");
    }
    assert!(round_trips32(Insn::Config { v: 0x0006 }));
}

// model: unit_tests/test_slices.sail test_power_and_sync_instructions
#[test]
fn test_power_and_sync_instructions() {
    // Power management and multi-chip synchronization instructions decode (ISCA 2022).
    assert!(round_trips32(Insn::SetVL { n: 0b01010 }));
    assert!(ill32(cat(&[(0x0, 4), (0b000110, 6), (0b000000, 6), (0, 10), (0b000000, 6)]))); // SetVL 0
    assert!(!ill32(cat(&[(0x0, 4), (0b000110, 6), (0b000000, 6), (0, 10), (0b010101, 6)]))); // SetVL 21: legal on every profile
    assert!(round_trips32(Insn::SetVL { n: 0b100000 })); // SetVL 32
    assert!(ill32(cat(&[(0x0, 4), (0b000110, 6), (0b000000, 6), (0, 10), (0b100001, 6)]))); // SetVL 33
    assert!(ill32(cat(&[(0x0, 4), (0b000110, 6), (0b000000, 6), (0, 10), (0b111111, 6)]))); // SetVL 63
    assert!(round_trips64(Insn::PowerConfig { v: 0x0003 }));
    assert!(round_trips64(Insn::DeepSleep));
    assert!(round_trips32(Insn::Transmit { chip_id: 0b000111, link: 0x2 }));
    assert!(round_trips32(Insn::RuntimeDeskew { t: 0x00FC }));
}

// model: unit_tests/test_c2c.sail test_c2c_link_field_layout
#[test]
fn test_c2c_link_field_layout() {
    // 32-bit layout with the link field: slice 31..28 = 5, opcode 27..22, chip_id 21..16,
    // stream 15..10, link 9..6, reserved 5..0 (Transmit: 15..10 reserved).
    let send = Insn::Send { chip_id: 0b101010, stream: 0b100011, link: 0xB };
    let w = encode(&send);
    assert_eq!(w, cat(&[(0x5, 4), (0b000000, 6), (0b101010, 6), (0b100011, 6), (0xB, 4), (0b000000, 6)]));
    assert!(round_trips32(send));
    let decode = |w: u64| isa().decode(32, w);
    assert_eq!(
        decode(cat(&[(0x5, 4), (0b000001, 6), (0b000001, 6), (0b000010, 6), (0x3, 4), (0, 6)])),
        Some(Insn::Receive { chip_id: 0b000001, stream: 0b000010, link: 0x3 })
    );
    assert_eq!(
        decode(cat(&[(0x5, 4), (0b000011, 6), (0b000111, 6), (0b000000, 6), (0xE, 4), (0, 6)])),
        Some(Insn::Transmit { chip_id: 0b000111, link: 0xE })
    );
    // the widened ids: chip 0x1A5 (9 bits), link 0x5D (7 bits)
    let x = Insn::Send { chip_id: 0x1a5, stream: 0b100011, link: 0x5d };
    assert_eq!(
        encode(&x),
        cat(&[(0x5, 4), (0b000000, 6), (0b100101, 6), (0b100011, 6), (0b1101, 4), (0b101, 3), (0b110, 3)])
    );
    assert!(round_trips32(x));
    assert_eq!(
        decode(cat(&[(0x5, 4), (0b000011, 6), (0b100101, 6), (0b000000, 6), (0b1101, 4), (0b101, 3), (0b110, 3)])),
        Some(Insn::Transmit { chip_id: 0x1a5, link: 0x5d })
    );
    assert!(ill32(cat(&[(0x5, 4), (0b000011, 6), (0b000111, 6), (0b000001, 6), (0xE, 4), (0, 6)])));
}
