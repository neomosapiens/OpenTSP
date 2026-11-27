// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// SECDED ECC on SRAM words, stream registers and data paths (microarchitecture.md
// sec. 15). The published TSP sources give the scheme: the ECC that protects SRAM "is also used
// as they flow on the stream registers on-chip" (ISCA20 sec. II-D); check bits are generated "only
// at the producer" and stored beside the 128-bit memory word as "9-bit ECC, 137-bits in total";
// the scheme "implements SECDED (single-error correction with double-error detection)" and
// a functional slice "will check the ECC bits to ensure data integrity before operating on" a
// stream it consumes (ISCA20 sec. II-D); HC34 p.56 and p.59 list the data paths, the stream
// registers and the SRAM as SECDED protected, with stream registers of 138 bits (32 x 138 bits
// per direction, p.57); ISCA22 sec. 4.5 corrects every single-bit error in situ and detects every
// multi-bit error. They publish no parity-check matrix and no bit positions.
//
// The word of one superlane is 138 bits: bits 0..127 the 16 data bytes (byte l bit b is bit
// 8 l + b), bits 128..136 the check bits c[0..8], bit 137 the spare bit (implementation choice
// of the position; the 138-bit width is the published register width). The code is the extended
// Hamming code (137, 128) [implementation choice]: data bit j (0..127) takes codeword position
// P(j), the j-th integer >= 3 that is not a power of two (3, 5, 6, 7, 9, ..., 136); c[i] (i =
// 0..7) is the XOR of the data bits whose position has bit i set; c[8] is the XOR of the 136
// bits at positions 1..136 (data bits and c[0..7]).
//
// REPRESENTATION. The model stores a superlane word as its 128 data bits as stored (a flipped
// bit is flipped here) and an overlay e of 10 bits: e[8:0] = (the stored check bits) XOR
// code(the stored data bits), e[9] = the stored spare bit. This is the same information as
// (data, check bits): the syndrome of the stored word is s = e[7:0] and q = XOR(e[8:0]) (the
// overall parity), because code(d) is itself a valid codeword; e[8:0] = 0 holds exactly when
// the stored word is a valid codeword. A producer that generates the code writes e[8:0] = 0.
// test_ecc.sail proves the equivalence against an explicit encoder and decoder on every
// single and double flip.

type ecc_ov = bits(10 * superlanes)

let ecc_ov_zero : ecc_ov = sail_zeros(10 * superlanes)

enum ecc_class = {ECC_OK, ECC_CORRECTED, ECC_UNCORRECTABLE}

private function ecc_is_pow2(p : int) -> bool =
  p == 1 | p == 2 | p == 4 | p == 8 | p == 16 | p == 32 | p == 64 | p == 128 | p == 256

private function ecc_log2(p : int) -> int =
  if p >= 256 then 8 else if p >= 128 then 7 else if p >= 64 then 6 else if p >= 32 then 5
  else if p >= 16 then 4 else if p >= 8 then 3 else if p >= 4 then 2 else if p >= 2 then 1 else 0

// The data bit index of codeword position p (p >= 3, not a power of two).
function ecc_data_index(p : int) -> int = p - ecc_log2(p) - 2

// Insert the parity positions 1,2,4,8,...,128 into 128 data bits. The first
// data indices after these holes are 0,1,4,11,26,57,120. Bounded straight-line
// arithmetic lets the SMT backend check all protected bit positions.
function ecc_pos(j : int) -> int =
  j + (if j >= 120 then 9 else if j >= 57 then 8 else if j >= 26 then 7
       else if j >= 11 then 6 else if j >= 4 then 5 else if j >= 1 then 4 else 3)

// XOR of the bits of x (x has at most 10 bits).
function ecc_par10(x : bits(10)) -> bool = {
  var r : bool = false;
  foreach (i from 0 to 9) { if x[i] == bitone then r = not(r) };
  r
}

// The check bits of 128 data bits (the explicit encoder).
function ecc_encode(d : bits(128)) -> bits(9) = {
  var c : bits(9) = 0b000000000;
  var par : bool = false;
  foreach (j from 0 to 127) {
    if d[j] == bitone then {
      let pb : bits(8) = get_slice_int(8, ecc_pos(j), 0);
      foreach (i from 0 to 7) { if pb[i] == bitone then c[i] = ~(c[i]) };
      par = not(par)
    }
  };
  let low : bits(10) = 0b00 @ c[7 .. 0];
  if par != ecc_par10(low) then c[8] = bitone;
  c
}

// The overlay change when bit n of the 138-bit word flips (0..127 data, 128..136 check bits
// c[0..8], 137 the spare bit).
function ecc_flip_delta(n : int) -> bits(10) =
  if n < 128 then {
    let pb : bits(8) = get_slice_int(8, ecc_pos(n), 0);
    let c8 : bool = not(ecc_par10(0b00 @ pb));
    (if c8 then 0b0100000000 else 0b0000000000) | (0b00 @ pb)
  } else if n < 137 then sail_shiftleft(0b0000000001, n - 128)
  else 0b1000000000

// What the check of a superlane word finds, and the position the syndrome names (0 for the
// overall parity bit c[8]; for an uncorrectable word the raw syndrome).
function ecc_classify(e : bits(10)) -> (ecc_class, range(0, 255)) = {
  let s : range(0, 255) = unsigned(e[7 .. 0]);
  let nonzero : bool = e[8 .. 0] != 0b000000000;
  let q : bool = ecc_par10(0b0 @ e[8 .. 0]);
  if not(nonzero) then (ECC_OK, 0)
  else if q then {
    if s == 0 | ecc_is_pow2(s) | (s >= 3 & s <= 136) then (ECC_CORRECTED, s) else (ECC_UNCORRECTABLE, s)
  } else (ECC_UNCORRECTABLE, s)
}

// The reader's view of one superlane: the class, the 128 data bits after a single-bit
// correction, and the position.
function ecc_correct(d : bits(128), e : bits(10)) -> (ecc_class, bits(128), range(0, 255)) = {
  let (k, s) = ecc_classify(e);
  var out : bits(128) = d;
  if k == ECC_CORRECTED & s >= 3 & not(ecc_is_pow2(s)) then
    out = d ^ sail_shiftleft(get_slice_int(128, 1, 0), ecc_data_index(s));
  (k, out, s)
}

// The overlay of a poisoned superlane: c[0] and c[1] inverted (a double error: syndrome 3, even
// parity, so every later reader finds it uncorrectable) [implementation choice]. The spare bit (bit 137) is not part of
// the poison: it belongs to the error report of the stream registers (the spare bits S[k][x], see ecc_report_*).
let ecc_poison_sl : bits(10) = 0b0000000011

function ecc_ov_get(ov : ecc_ov, g : int) -> bits(10) = slice(ov, 10 * g, 10)

function ecc_ov_set(ov : ecc_ov, g : int, e : bits(10)) -> ecc_ov = {
  let m : ecc_ov = sail_shiftleft(get_slice_int(10 * superlanes, 1023, 0), 10 * g);
  let x : ecc_ov = sail_shiftleft(sail_zero_extend(e, 10 * superlanes), 10 * g);
  (ov & ~(m)) | x
}

// The overlay of a vector whose superlanes in m are poisoned and whose others carry a fresh code (a text payload the
// Ifetch read from streams that had an uncorrectable superlane).
function ecc_poison_ov(m : bits(superlanes)) -> ecc_ov = {
  var o : ecc_ov = ecc_ov_zero;
  if m != sail_zeros(superlanes) then {
    foreach (g from 0 to (superlanes - 1)) {
      let bitval : bits(1) = slice(m, g, 1);
      let selected = sub_bits(0b0000000000, 0b000000000 @ bitval);
      o = ecc_ov_set(o, g, selected & ecc_poison_sl)
    }
  };
  o
}

// Matrix-unit check (microarchitecture.md sec. 18): the sum check of an ABC corrected the one wrong lane of its product (`T ecc ... mxm corr`).
val mxm_corr_callback = pure {c: "mxm_corr_callback", cpp: "mxm_corr_callback"} :
  (/* cycle */ nat, /* queue */ range(0, 144), /* plane */ range(0, 3), /* lane */ nat) -> unit

// S[0..3] of a superlane as the reader sees them (core/streams.sail), for the consumer's log of a spare-bit report.
val srf_s03 : (bool, int, int, int) -> bits(4)

// Check the value of a stream operand or of an SRAM word against its overlay, superlane by
// superlane: the reader gets the corrected data; the mask has bit g set for an uncorrectable
// superlane g. No error (the overlay is zero) costs one comparison.
function ecc_check_value(ci : int, q : range(0, 144), kind : range(0, 1), isw : bool, id : range(0, 87), addr : range(0, 65535), v : vec, ov : ecc_ov) -> (vec, bits(superlanes)) = {
  var out : vec = v;
  var bad : bits(superlanes) = sail_zeros(superlanes);
  let c : nat = if ci >= 0 then ci else 0;
  if ov != ecc_ov_zero then {
    foreach (g from 0 to (superlanes - 1)) {
      let e = ecc_ov_get(ov, g);
      // A stream operand whose spare bit is set carries an error report (microarchitecture.md sec. 15.5): logged by its consumer.
      if kind == 0 & e[9] == bitone then ras_spare_event(c, q, isw, id, addr, g, srf_s03(isw, addr, c, g));
      let (k, d2, pos) = ecc_correct(slice(v, 128 * g, 128), e);
      match k {
        ECC_OK => (),
        ECC_CORRECTED => {
          out[128 * g + 127 .. 128 * g] = d2;
          ras_ecc_event(c, q, kind, false, isw, id, addr, g, pos)
        },
        ECC_UNCORRECTABLE => {
          bad[g] = bitone;
          ras_ecc_event(c, q, kind, true, isw, id, addr, g, pos)
        },
      }
    }
  };
  (out, bad)
}

// Superlanes poisoned in the results of the current instruction: the superlanes found
// uncorrectable by the instruction's operand checks (every superlane for a unit whose output
// depends on every input superlane), reported once as OTS-E-ECC-001 after the instruction's
// own checks (postlude/tick.sail execute_element).
register ecc_bad : bits(superlanes) = sail_zeros(superlanes)
register ecc_dep_all : bool = false
// The instruction's per-tile exception (Gather, Scatter) reported another diagnostic as well.
register ecc_after_diag : bool = false
// Superlanes of the results that carry a poison the instruction inherited from state (an MXM product or accumulator
// that an uncorrectable operand reached earlier; a vector that arrived marked uncorrectable over a link). They are
// poisoned in the results like the ones above but are not reported again: the instruction found no error itself.
register ecc_inherit : bits(superlanes) = sail_zeros(superlanes)
// MXM LW and IW from streams: which of the 16 streams of the group had an uncorrectable superlane (set by read_planes16).
register ecc_g16 : vector(16, bool) = vector_init(false)
// Superlanes of a vector that arrived on a link with an uncorrectable FEC error (Receive, microarchitecture.md
// sec. 17): poisoned in the Receive's result like ecc_bad, but reported as OTS-E-LNK-004 instead of OTS-E-ECC-001.
register ecc_link_bad : bits(superlanes) = sail_zeros(superlanes)

function ecc_begin() -> unit = {
  ecc_bad = sail_zeros(superlanes);
  ecc_dep_all = false;
  ecc_after_diag = false;
  ecc_inherit = sail_zeros(superlanes);
  ecc_link_bad = sail_zeros(superlanes)
}

function ecc_poison_mask() -> bits(superlanes) =
  (if ecc_dep_all & ecc_bad != sail_zeros(superlanes) then sail_ones(superlanes) else ecc_bad) | ecc_inherit | ecc_link_bad

// The overlay of a stream result: poisoned superlanes carry the poison pattern, every other
// superlane the freshly generated code (e[8:0] = 0, spare 0).
function ecc_result_ov() -> ecc_ov = {
  let m = ecc_poison_mask();
  var o : ecc_ov = ecc_ov_zero;
  if m != sail_zeros(superlanes) then {
    foreach (g from 0 to (superlanes - 1)) {
      if m[g] == bitone then o = ecc_ov_set(o, g, ecc_poison_sl)
    }
  };
  o
}

// Check superlane g of a stored SRAM word (the Gather and Scatter tile reads of one superlane):
// the corrected 128 data bits and whether the superlane is uncorrectable (then it also marks
// ecc_bad). Like a stream operand, a corrected word stays flipped in the SRAM.
function ecc_check_sl(ci : int, q : range(0, 144), slice_ : range(0, 87), addr : range(0, 65535), g : range(0, superlanes - 1), d : bits(128), e : bits(10)) -> (bits(128), bool) = {
  let c : nat = if ci >= 0 then ci else 0;
  let (k, d2, pos) = ecc_correct(d, e);
  match k {
    ECC_OK => (d, false),
    ECC_CORRECTED => { ras_ecc_event(c, q, 1, false, false, slice_, addr, g, pos); (d2, false) },
    ECC_UNCORRECTABLE => {
      ras_ecc_event(c, q, 1, true, false, slice_, addr, g, pos);
      ecc_bad = ecc_bad | sail_shiftleft(get_slice_int(superlanes, 1, 0), g);
      (d, true)
    },
  }
}
