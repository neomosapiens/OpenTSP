// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Unit tests for the SECDED code of core/ecc.sail. The reference below is written
// again from the definition (codeword positions by counting, syndrome by position, overall
// parity), not from the model's functions, so a change of the model's position function or
// classification is caught by comparison.

private function ref_is_pow2(p : int) -> bool = {
  var q : int = 1;
  while q < p do q = q * 2;
  q == p
}

// The codeword position of data bit j by counting the integers >= 3 that are not a power of two.
private function ref_pos(j : int) -> int = {
  var p : int = 2;
  var cnt : int = -1;
  while cnt < j do { p = p + 1; if not(ref_is_pow2(p)) then cnt = cnt + 1 };
  p
}

private function ref_table() -> vector(128, int) = {
  var t : vector(128, int) = vector_init(0);
  foreach (j from 0 to 127) { t[j] = ref_pos(j) };
  t
}

private function ref_par(x : bits(128)) -> bool = {
  var r : bool = false;
  foreach (i from 0 to 127) { if x[i] == bitone then r = not(r) };
  r
}

// A fixed, irregular 128-bit word.
private function test_word() -> bits(128) = 0x9e3779b97f4a7c15f39cc0605cedc834

// The reference decoder: syndrome by XOR of the positions of the set bits, overall parity of
// the 137 bits. Result: 0 no error, 1 corrected, 2 uncorrectable; and the data after correction.
private function ref_decode(t : vector(128, int), d : bits(128), c : bits(9)) -> (int, bits(128)) = {
  var s : bits(8) = 0x00;
  var par : bool = ref_par(d);
  foreach (j from 0 to 127) { if d[j] == bitone then s = s ^ get_slice_int(8, t[j], 0) };
  foreach (i from 0 to 7) { if c[i] == bitone then { s = s ^ sail_shiftleft(0x01, i); par = not(par) } };
  if c[8] == bitone then par = not(par);
  let sv : int = unsigned(s);
  if sv == 0 & not(par) then (0, d)
  else if not(par) then (2, d)
  else if sv == 0 | ref_is_pow2(sv) then (1, d)
  else {
    var hit : int = -1;
    foreach (j from 0 to 127) { if t[j] == sv then hit = j };
    if hit < 0 then (2, d) else (1, d ^ sail_shiftleft(get_slice_int(128, 1, 0), hit))
  }
}

// Flip codeword bit n (0..127 data, 128..136 check bits c[0..8]).
private function flip_cw(d : bits(128), c : bits(9), n : int) -> (bits(128), bits(9)) =
  if n < 128 then (d ^ sail_shiftleft(get_slice_int(128, 1, 0), n), c)
  else (d, c ^ sail_shiftleft(get_slice_int(9, 1, 0), n - 128))

private function class_code(k : ecc_class) -> int =
  match k { ECC_OK => 0, ECC_CORRECTED => 1, ECC_UNCORRECTABLE => 2 }

// The model's position function equals the counting definition for every data bit.
$[test]
function test_ecc_positions() -> unit = {
  let t = ref_table();
  foreach (j from 0 to 127) { assert(ecc_pos(j) == t[j], "ecc_pos differs from the definition") };
  assert(t[0] == 3 & t[1] == 5 & t[127] == 136, "first and last positions are 3, 5 and 136")
}

// Every single flip of the 137 bits of a codeword is corrected; the explicit decoder, the
// overlay classification and the overlay delta agree; no flip leaves the word unchanged.
$[test]
function test_ecc_single_flips() -> unit = {
  let t = ref_table();
  let d = test_word();
  let c = ecc_encode(d);
  let (k0, d0) = ref_decode(t, d, c);
  assert(k0 == 0 & d0 == d, "a clean word has no error");
  let (kz, _) = ecc_classify(0b0000000000);
  assert(kz == ECC_OK, "a zero overlay is no error");
  foreach (n from 0 to 136) {
    let (d1, c1) = flip_cw(d, c, n);
    let (k, dd) = ref_decode(t, d1, c1);
    assert(k == 1 & dd == d, "reference decoder corrects a single flip");
    // The overlay of the flipped word: stored check bits XOR the code of the stored data.
    let ov : bits(10) = 0b0 @ (c1 ^ ecc_encode(d1));
    assert(ov == ecc_flip_delta(n), "ecc_flip_delta is the overlay of the flip");
    let (km, dm, _) = ecc_correct(d1, ov);
    assert(km == ECC_CORRECTED & dm == d, "the model corrects a single flip");
    // The word the model corrects is the word the reference corrects; a flipped check bit leaves the data as it is.
    assert(dm == dd, "the model and the reference agree on the data")
  }
}

// Every double flip is detected, never miscorrected: 137 * 136 / 2 = 9,316 pairs.
$[test]
function test_ecc_double_flips() -> unit = {
  let t = ref_table();
  let d = test_word();
  let c = ecc_encode(d);
  var pairs : int = 0;
  foreach (a from 0 to 135) {
    foreach (b from (a + 1) to 136) {
      let (d1, c1) = flip_cw(d, c, a);
      let (d2, c2) = flip_cw(d1, c1, b);
      let (k, _) = ref_decode(t, d2, c2);
      assert(k == 2, "reference decoder detects a double flip");
      let (km, _) = ecc_classify(ecc_flip_delta(a) ^ ecc_flip_delta(b));
      assert(km == ECC_UNCORRECTABLE, "the model detects a double flip");
      pairs = pairs + 1
    }
  };
  assert(pairs == 9316, "137 * 136 / 2 pairs")
}

// The explicit encoder is linear: code(a) XOR code(b) = code(a XOR b); the code of zero is zero.
$[test]
function test_ecc_linear() -> unit = {
  let a = test_word();
  let b = 0x0123456789abcdeffedcba9876543210;
  assert(ecc_encode(0x00000000000000000000000000000000) == 0b000000000, "code of zero is zero");
  assert((ecc_encode(a) ^ ecc_encode(b)) == ecc_encode(a ^ b), "the code is linear")
}

// The poison pattern of a result is an uncorrectable word; its spare bit is clear (bit 137 belongs to the error report).
$[test]
function test_ecc_poison() -> unit = {
  let (k, _) = ecc_classify(ecc_poison_sl);
  assert(k == ECC_UNCORRECTABLE, "a poisoned superlane is uncorrectable");
  assert(ecc_poison_sl[9] == bitzero, "poison leaves the spare bit to the error report")
}

// The stream-level check: a clean value passes, one flipped superlane is corrected in the
// reader's copy, a double flip marks its superlane, other superlanes are untouched.
$[test]
function test_ecc_check_value() -> unit = {
  let vv : vec = replicate_bits(0xa5, vlanes);
  let (o0, b0) = ecc_check_value(0, 0, 0, false, 3, 0, vv, ecc_ov_zero);
  assert(o0 == vv & b0 == sail_zeros(superlanes), "a clean value passes unchanged");
  // Flip data bit 77 of superlane 0, as the harness does it: the value changes and the overlay holds the delta.
  let bad_v : vec = vv ^ sail_shiftleft(get_slice_int(8 * vlanes, 1, 0), 77);
  let ov1 = ecc_ov_set(ecc_ov_zero, 0, ecc_flip_delta(77));
  let (o1, b1) = ecc_check_value(0, 0, 0, false, 3, 0, bad_v, ov1);
  assert(o1 == vv & b1 == sail_zeros(superlanes), "a single flip is corrected");
  let ov2 = ecc_ov_set(ecc_ov_zero, 0, ecc_flip_delta(77) ^ ecc_flip_delta(5));
  let (o2, b2) = ecc_check_value(0, 0, 0, false, 3, 0, bad_v ^ sail_shiftleft(get_slice_int(8 * vlanes, 1, 0), 5), ov2);
  assert(b2 == get_slice_int(superlanes, 1, 0), "a double flip marks superlane 0")
}
