// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Queries over the instruction definitions, for tools outside the model. The ISA catalogue
// check (test/isa_db/check_isa_db.py) uses them to compare isa-db/instructions/*.yaml with
// the normative encdec and assembly mappings: Sail is normative, the YAML is checked.

// The assembly text of a word, as the model decodes it ("illegal32 ..." if it does not).
function isa_disasm32(w : bits(32)) -> string = assembly(encdec32(w))
function isa_disasm64(w : bits(64)) -> string = assembly(encdec64(w))

// The text-level decode (postlude/iq.sail decode_text): the assembly text of the instruction that
// starts with units u0 (and u1 for a two-unit instruction), or "illegal" when it does not decode
// (OTS-E-DEC-003).
function isa_decode_text(u0 : bits(32), u1 : bits(32)) -> string =
  match decode_text(u0, u1) { Ok(i) => assembly(i), Err(_) => "illegal" }

// True when the word survives decode -> encode unchanged. (Parsing assembly text back is
// not executable in Sail's C backend; the checker compares the printed text field by field.)
function isa_roundtrip32(w : bits(32)) -> bool = {
  let i = encdec32(w);
  encdec32(i) == w
}

function isa_roundtrip64(w : bits(64)) -> bool = {
  let i = encdec64(w);
  encdec64(i) == w
}

// True when the word decodes to an instruction (for the exhaustive encoding check).
function isa_decodes32(w : bits(32)) -> bool = match encdec32(w) { ILLEGAL32(_) => false, _ => true }
function isa_decodes64(w : bits(64)) -> bool = match encdec64(w) { ILLEGAL64(_) => false, _ => true }

// The gap at which queue q reads (w = false) or writes (w = true) row r (0 = E, 1 = W),
// or -1 when it cannot (core/layout.sail): test/isa_db/check_isa_db.py compares it with the
// `stations` table of isa-db/generated/encoding.json on every queue, row and profile.
function isa_station(q : range(0, 143), r : range(0, 1), w : bool) -> int = {
  let i = issuer_of(q);
  let rw : row = if r == 0 then RowE else RowW;
  let g = if w then write_gap(i.unit, i.slice, rw) else read_gap(i.unit, i.slice, rw);
  match g { Some(x) => x, None() => -1 }
}
