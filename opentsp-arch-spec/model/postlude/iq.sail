// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Parcel and Decode (assembly.md sec. 7.2, 7.3, 7.5;
// ISCA 2020 sec. III-A1: NOP N separates two instructions in program order).
//
// The text of a queue is a sequence of 32-bit units. An instruction starts with a unit whose
// bit 31 is the bundle bit (1: dispatch in one parcel with the next instruction) and whose
// bits 30..28 are the slice tag: tags 0, 3, 5 (CU, MXM, C2C) are one unit, tags 1, 2, 4 (MEM,
// VXM, SXM) two units (bits 63..32 first), tags 6 and 7 do not decode and are one unit
// (OTS-E-DEC-003). The Parcel stage clears the bundle bit before decode.

// Number of units of the instruction that starts with u (1 or 2).
function text_width(u : bits(32)) -> nat = {
  let t = u[30 .. 28];
  if t == 0b001 | t == 0b010 | t == 0b100 then 2 else 1
}

// ---------------------------------------------------------------------------------------
// ECC on instruction text (microarchitecture.md section 16). A text word is four consecutive units of one source
// (core/ecc.sail: 128 data bits, the overlay of the word in iq_wov). Every instruction is checked, before it is
// decoded, in the one or two words its units lie in (the whole words: a fault in a unit the instruction does not use
// counts). The scan is pure: a corrected word is read corrected, an uncorrectable word makes the instruction
// OTS-E-ECC-002; the events, the error registers and the write-back of a correction happen when the parcel dispatches
// (iq_text_check).

// The text word of the unit at IQ position i (-1: none) and its position in the word.
private function iq_unit_word(q : range(0, 144), i : nat) -> int = {
  let t = iq_tag_read(q, ring(iq_head[q] + i));
  if t == iq_no_word then -1 else ediv_int(t, 4)
}

private function iq_unit_lane(q : range(0, 144), i : nat) -> nat = emod_int(iq_tag_read(q, ring(iq_head[q] + i)), 4)

// The unit at IQ position i as the decoder sees it: corrected when its word has a correctable error.
function iq_view(q : range(0, 144), i : nat) -> bits(32) = {
  let u = iq_peek(q, i);
  if not(iq_ecc_live) then u
  else {
    let w = iq_unit_word(q, i);
    if w < 0 then u
    else {
      let e = iq_word_ov(q, w);
      if e == 0b0000000000 then u
      else {
        let k = iq_unit_lane(q, i);
        let d : bits(128) = sail_shiftleft(sail_zero_extend(u, 128), 32 * k);
        let (_, d2, _) = ecc_correct(d, e);
        slice(d2, 32 * k, 32)
      }
    }
  }
}

private function iq_word_unc(q : range(0, 144), w : int) -> bool =
  if w < 0 then false
  else {
    let e = iq_word_ov(q, w);
    if e == 0b0000000000 then false else { let (k, _) = ecc_classify(e); k == ECC_UNCORRECTABLE }
  }

// True when a word of the w units of the instruction at position pos is uncorrectable.
function iq_text_unc(q : range(0, 144), pos : nat, w : nat) -> bool =
  if not(iq_ecc_live) then false
  else iq_word_unc(q, iq_unit_word(q, pos)) | (w == 2 & iq_word_unc(q, iq_unit_word(q, pos + 1)))

// A correctable word found by a dispatching instruction in cycle c: the event, and the write-back (the word's overlay
// goes, a flipped data bit still in the IQ is flipped back).
private function iq_word_check(q : range(0, 144), c : nat, w : int) -> unit =
  if w >= 0 then {
    let wn : nat = max(w, 0);
    let e = iq_word_ov(q, wn);
    if e != 0b0000000000 then {
      let (k, s) = ecc_classify(e);
      match k {
        ECC_OK => (),
        ECC_UNCORRECTABLE => ras_ecc_event(c, q, 2, true, false, 0, emod_int(wn, 16777216), 0, s),
        ECC_CORRECTED => {
          ras_ecc_event(c, q, 2, false, false, 0, emod_int(wn, 16777216), 0, s);
          // the flipped data bit (none when the syndrome names a check bit), in the unit of the IQ that holds it
          let (_, fixed, _) = ecc_correct(sail_zeros(128), e);
          foreach (k from 0 to 3) {
            let m : bits(32) = slice(fixed, 32 * k, 32);
            if m != 0x00000000 then {
              let want = 4 * wn + k;
              foreach (i from 0 to (iq_len[q] - 1)) {
                if iq_tag_read(q, ring(iq_head[q] + i)) == want then iq_unit_write(q, ring(iq_head[q] + i), iq_peek(q, i) ^ m)
              }
            }
          };
          iq_word_ov_set(q, wn, 0b0000000000)
        },
      }
    }
  }

// The units of the head parcel (units of them) leave the IQ in cycle c: every instruction of the parcel (the timing
// prefix included) checks its words, in program order.
function iq_text_check(q : range(0, 144), c : nat, units : nat) -> unit =
  if iq_ecc_live then {
    var pos : nat = 0;
    // every step takes one or two of the parcel's units
    foreach (_step from 0 to (units - 1)) {
      if pos < units then {
        let w = text_width(iq_view(q, pos));
        let a = iq_unit_word(q, pos);
        iq_word_check(q, c, a);
        if w == 2 then { let b = iq_unit_word(q, pos + 1); if b != a then iq_word_check(q, c, b) };
        pos = pos + w
      }
    }
  }

// Decode the instruction that starts with u0 (and u1, for a two-unit instruction).
function decode_text(u0 : bits(32), u1 : bits(32)) -> result(instruction, diag) = {
  let hi : bits(32) = 0b0 @ u0[30 .. 0];
  let t = u0[30 .. 28];
  if t == 0b000 | t == 0b011 | t == 0b101 then match encdec32(hi) { ILLEGAL32(_) => Err(DEC_OPCODE), i => Ok(i) }
  else if t == 0b001 | t == 0b010 | t == 0b100 then match encdec64(hi @ u1) { ILLEGAL64(_) => Err(DEC_OPCODE), i => Ok(i) }
  else Err(DEC_OPCODE)
}

// Decode the instruction of w units at position pos of the IQ of queue q (u1 is read when w is 2).
function iq_decode_at(q : range(0, 144), pos : nat, w : nat) -> result(instruction, diag) =
  if iq_text_unc(q, pos, w) then Err(ECC_TXT)
  else decode_text(iq_view(q, pos), if w == 2 then iq_view(q, pos + 1) else 0x00000000)

// The head parcel of a queue: the leading NOPs (a timing prefix: NOPs with bundle bit 0 before
// the first instruction), then the instructions up to and including the first one with bundle
// bit 0. complete: every unit of the parcel is in the IQ; units: the units it takes from the
// IQ (the prefix included); nops: the sum of the prefix; items: the instructions, decoded.
struct parcel = {
  complete : bool,
  units    : nat,
  nops     : nat,
  n        : nat,
  items    : list(result(instruction, diag)),
}

private let no_parcel : parcel = struct { complete = false, units = 0, nops = 0, n = 0, items = [||] }

private function rev_items(xs : list(result(instruction, diag)), acc : list(result(instruction, diag))) -> list(result(instruction, diag)) =
  match xs { [||] => acc, x :: rest => rev_items(rest, x :: acc) }

function parcel_scan(q : range(0, 144)) -> parcel = {
  let len = iq_len[q];
  var pos : nat = 0;
  var nops : nat = 0;
  var n : nat = 0;
  var items : list(result(instruction, diag)) = [||];
  var done : bool = false;
  var complete : bool = false;
  // every step but the last takes one or two of the len units; the last one finds the end
  foreach (_step from 0 to len) {
    if not(done) then {
      if pos >= len then done = true
      else {
        let u0 = iq_view(q, pos);
        let w = text_width(u0);
        if w == 2 & pos + 1 >= len then done = true
        else {
          let r = iq_decode_at(q, pos, w);
          let bundle = u0[31] == bitone;
          let nop : nat = match r { Ok(NOP(k)) => if not(bundle) & n == 0 then unsigned(k) else 0, _ => 0 };
          pos = pos + w;
          if nop > 0 then nops = nops + nop
          else {
            items = r :: items;
            n = n + 1;
            if not(bundle) then { complete = true; done = true }
          }
        }
      }
    }
  };
  if complete then struct { complete = true, units = pos, nops = nops, n = n, items = rev_items(items, [||]) }
  else no_parcel
}

// The earliest dispatch cycle of a complete head parcel of queue q: one cycle after the
// queue's last dispatch, plus the NOPs before it (sec. 7.3).
function parcel_earliest(q : range(0, 144), p : parcel) -> int = iq_last[q] + 1 + p.nops

// True when the text left in the IQ of queue q can never form a parcel: after leading NOPs
// (a timing prefix with nothing to time: a complete NOP with no successor, not a truncation)
// the text holds the first unit of a two-unit instruction, or instructions up to one whose
// bundle bit is set with nothing after it. Used once, at the end of the run (model_finish).
function iq_stranded(q : range(0, 144)) -> bool = {
  let len = iq_len[q];
  var pos : nat = 0;
  var n : nat = 0;
  var left : bool = false;
  var done : bool = false;
  // every step but the last takes one or two of the len units; the last one finds the end
  foreach (_step from 0 to len) {
    if not(done) then {
      if pos >= len then done = true
      else {
        let u0 = iq_view(q, pos);
        let w = text_width(u0);
        if w == 2 & pos + 1 >= len then { left = true; done = true }
        else {
          let r = iq_decode_at(q, pos, w);
          let bundle = u0[31] == bitone;
          let nop : nat = match r { Ok(NOP(k)) => if not(bundle) & n == 0 then unsigned(k) else 0, _ => 0 };
          pos = pos + w;
          if nop == 0 then {
            n = n + 1;
            left = true;
            if not(bundle) then done = true     // a complete parcel that never dispatched is reported too
          }
        }
      }
    }
  };
  left
}
