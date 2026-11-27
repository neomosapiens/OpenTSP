//! The wide chip and link numbers of Send, Receive, Transmit and DeskewRecv: bits 5..0 of the chip are in bits 21..16 and bits 8..6 in
//! bits 2..0, bits 3..0 of the link in bits 9..6 and bits 6..4 in bits 5..3 (`isa.md`, growth limits). A word that was legal keeps its meaning.
use ots_asm::parse_insn;
use ots_isa::{encode, render, Isa};
use ots_profile::Profile;
use std::collections::HashMap;

fn isa() -> Isa {
    Isa::new(&Profile::load("ref").unwrap())
}

fn word(isa: &Isa, t: &str) -> Result<u64, String> {
    parse_insn(isa, t, &HashMap::new(), None).map(|(_, w)| w).map_err(|e| e.code.as_str().to_string())
}

fn at(slice_opcode: u64, chip: u64, stream: u64, link: u64) -> u64 {
    (5 << 28)
        | (slice_opcode << 22)
        | ((chip & 63) << 16)
        | (stream << 10)
        | ((link & 15) << 6)
        | (((link >> 4) & 7) << 3)
        | ((chip >> 6) & 7)
}

#[test]
fn the_upper_bits_are_in_the_low_reserved_bits() {
    let isa = isa();
    assert_eq!(word(&isa, "send 0x1a5, 0x5d, E:0"), Ok(at(0, 0x1a5, 0, 0x5d)));
    assert_eq!(word(&isa, "receive 0x1ff, 0x7f, W:5"), Ok(at(1, 0x1ff, 32 + 5, 0x7f)));
    assert_eq!(word(&isa, "transmit 0x41, 0x10"), Ok(at(3, 0x41, 0, 0x10)));
    assert_eq!(word(&isa, "deskew_recv 0x40, 0x0, E:1"), Ok(at(9, 0x40, 1, 0)));
}

#[test]
fn a_word_that_was_legal_keeps_its_meaning() {
    let isa = isa();
    assert_eq!(word(&isa, "send 0x3, 0xf, E:2"), Ok((5 << 28) | (3 << 16) | (2 << 10) | (0xf << 6)));
    assert_eq!(word(&isa, "transmit 0x3f, 0xf"), Ok((5 << 28) | (3 << 22) | (0x3f << 16) | (0xf << 6)));
}

#[test]
fn the_words_decode_to_the_text_that_made_them() {
    let isa = isa();
    for t in [
        "send 0x1a5, 0x5d, E:0",
        "receive 0x1ff, 0x7f, W:5",
        "transmit 0x41, 0x10",
        "deskew_recv 0x40, 0x0, E:1",
        "send 0x3, 0xf, E:2",
    ] {
        let (insn, w) = parse_insn(&isa, t, &HashMap::new(), None).unwrap();
        assert_eq!(encode(&insn), w, "{t}");
        let back = isa.decode(32, w).unwrap_or_else(|| panic!("{t}: {w:#x} does not decode"));
        assert_eq!(render(&back), t);
    }
}

#[test]
fn a_chip_above_511_or_a_link_above_127_is_out_of_range() {
    let isa = isa();
    assert_eq!(word(&isa, "send 0x200, 0x0, E:0"), Err("OTS-E-OPR-001".to_string()));
    assert_eq!(word(&isa, "send 0x0, 0x80, E:0"), Err("OTS-E-OPR-001".to_string()));
    assert_eq!(word(&isa, "transmit 0x200, 0x0"), Err("OTS-E-OPR-001".to_string()));
}

#[test]
fn a_word_whose_reserved_bits_outside_the_two_fields_are_set_does_not_decode() {
    let isa = isa();
    // Send with bits 5..3 and 2..0 set decodes (chip 0x1c0 + link 0x70 of the 9 and 7 bit numbers); a Transmit with its stream field set does not
    let w = at(0, 0, 0, 0) | 0b111_111;
    assert!(isa.decode(32, w).is_some());
    let t = at(3, 0, 1, 0);
    assert!(isa.decode(32, t).is_none());
}
