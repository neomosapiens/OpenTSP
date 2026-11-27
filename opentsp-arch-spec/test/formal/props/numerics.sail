// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Obligations on the rounding core `fround` of core/numerics/fp.sail . They are stated by the
// structure of a binary format (a significand `mag` and an exponent `e`, value mag * 2^e), not by the steps of `fround`: the
// expected result is built by the obligation from the input alone and compared with what the decoder (`fdecode`) reads back from
// the bit pattern `fround` returns. Round to nearest, ties to even: a tie rounds to the even significand.
//
// Cost note: a query that decodes a symbolic bit pattern costs tens of seconds (symbolic powers of two); prove_extra.sh gives this
// suite a longer per-query limit than the model's own obligations. A comparison written as `d.mag * pow2i(d.e)` is much slower than
// one on `mag` and `e` separately, so the obligations compare the two fields.

// ---- exactness: every finite bit pattern is a fixed point of the decode-and-round pipeline ----

function exact_ok(f : fmt, b : bits(32)) -> bool = {
  let d = fdecode(f, b);
  not(d.cls == FFinite) | fround(f, d.neg, d.mag, d.e) == b
}

$property
function p_exact_f16(x : bits(16)) -> bool = exact_ok(FMT16, 0x0000 @ x)

$property
function p_exact_e5m2(x : bits(8)) -> bool = exact_ok(FMT_E5M2, 0x000000 @ x)

$property
function p_exact_e4m3(x : bits(8)) -> bool = exact_ok(FMT_E4M3, 0x000000 @ x)

// ---- round to nearest, ties to even, for every remainder ----
// mag = c * 2^k + r with c a significand of mbits + 1 bits and r any k-bit remainder, at exponent 0, in the normal range. The expected
// significand is c, or c + 1 when the remainder is above half, or exactly half and c is odd.

function rne_norm_ok(f : fmt, k : int, off : int, r : int) -> bool = {
  let c = pow2i(f.mbits) + off;
  let mag = c * pow2i(k) + r;
  let half = pow2i(k - 1);
  let w = if r > half | (r == half & tmod_int(c, 2) == 1) then c + 1 else c;
  let res = fround(f, false, mag, 0);
  let d = fdecode(f, res);
  let roll = w == pow2i(f.mbits + 1);
  d.cls == FFinite & not(d.neg) & d.mag == (if roll then pow2i(f.mbits) else w) & d.e == (if roll then k + 1 else k)
}

$property
function p_rne_f16_k3(o : bits(10), r : bits(3)) -> bool = rne_norm_ok(FMT16, 3, unsigned(o), unsigned(r))

$property
function p_rne_e5m2_k4(o : bits(2), r : bits(4)) -> bool = rne_norm_ok(FMT_E5M2, 4, unsigned(o), unsigned(r))

$property
function p_rne_e4m3_k3(o : bits(3), r : bits(3)) -> bool = rne_norm_ok(FMT_E4M3, 3, unsigned(o), unsigned(r))

// ---- the subnormal range and its boundary ----
// mag = c * 2^j + r at exponent (emin - mbits) - j, for any (mbits+1)-bit c (so the result runs from zero through the subnormals to
// the second binade): the value in units of 2^(emin - mbits) is c plus the rounding increment. Zero is a zero; below 2^mbits the
// result is subnormal, from 2^mbits on it is normal at the same exponent; 2^(mbits+1) carries into the next binade.

function rne_sub_ok(f : fmt, j : int, c : int, r : int) -> bool = {
  let mag = c * pow2i(j) + r;
  if mag == 0 then true else {
    let half = pow2i(j - 1);
    let w = if r > half | (r == half & tmod_int(c, 2) == 1) then c + 1 else c;
    let res = fround(f, false, mag, f.emin - f.mbits - j);
    let d = fdecode(f, res);
    if w == 0 then d.cls == FZero & not(d.neg)
    else if w == pow2i(f.mbits + 1) then d.cls == FFinite & not(d.neg) & d.mag == pow2i(f.mbits) & d.e == f.emin - f.mbits + 1
    else d.cls == FFinite & not(d.neg) & d.mag == w & d.e == f.emin - f.mbits
  }
}

$property
function p_rne_sub_f16_j2(c : bits(11), r : bits(2)) -> bool = rne_sub_ok(FMT16, 2, unsigned(c), unsigned(r))

$property
function p_rne_sub_e5m2_j3(c : bits(3), r : bits(3)) -> bool = rne_sub_ok(FMT_E5M2, 3, unsigned(c), unsigned(r))

$property
function p_rne_sub_e4m3_j3(c : bits(4), r : bits(3)) -> bool = rne_sub_ok(FMT_E4M3, 3, unsigned(c), unsigned(r))

// ---- the overflow threshold ----
// mag at exponent emax - mbits - 1, emax = bias (bias + 1 for e4m3, whose top exponent code is ordinary; the largest finite value is (2^(mbits+1) - 1) * 2^(emax - mbits), which is
// mag = 2^(mbits+2) - 2 here). The half-way point to the next binade, mag = 2^(mbits+2) - 1, rounds up to infinity (the largest
// finite significand is odd); anything below stays finite. e4m3 has no infinity and its largest finite significand (14) is even, so
// the tie at 432 rounds to 448: every magnitude from 432 (mag = 27 here) saturates to the largest finite value 0x7E, below it the
// result is smaller.

function ovf_ok(f : fmt, mag : int) -> bool = {
  if mag == 0 then true else {
    let emax = if f.ieee then f.bias else f.bias + 1;       // e4m3's top exponent code is an ordinary one
    let res = fround(f, false, mag, emax - f.mbits - 1);
    let d = fdecode(f, res);
    if f.ieee then
      (if mag >= pow2i(f.mbits + 2) - 1 then d.cls == FInf & not(d.neg) else d.cls == FFinite & not(d.neg))
    else
      (if mag >= pow2i(f.mbits + 2) - 5 then res == inf_bits(f, false) else d.cls == FFinite & unsigned(res) < unsigned(inf_bits(f, false)))
  }
}

$property
function p_ovf_f16(m : bits(14)) -> bool = ovf_ok(FMT16, unsigned(m))

$property
function p_ovf_e5m2(m : bits(5)) -> bool = ovf_ok(FMT_E5M2, unsigned(m))

$property
function p_ovf_e4m3(m : bits(6)) -> bool = ovf_ok(FMT_E4M3, unsigned(m))

// ---- anchors: binary16 against definitions that do not use the model's format constants ----
// Every obligation above reads its result back through `fdecode` and builds the expected value from the `fmt` record, so a format whose
// constants are consistent with each other but are not IEEE binary16 (bias 16 with the smallest exponent -15, say) would satisfy
// them. These two obligations fix the format from outside: `fdecode` against the bit fields written with literals (as p_fp8_decode
// does for the FP8 formats), and `fround` against literal bit patterns of well-known binary16 values.

$property
function p_f16_decode(x : bits(16)) -> bool = {
  let d = fdecode(FMT16, 0x0000 @ x);
  let exponent = unsigned(x[14 .. 10]);
  let fraction = unsigned(x[9 .. 0]);
  let class_ok = if exponent == 31 then (if fraction == 0 then d.cls == FInf else d.cls == FNaN)
    else if exponent == 0 & fraction == 0 then d.cls == FZero
    else d.cls == FFinite & d.mag == fraction + (if exponent == 0 then 0 else 1024) & d.e == max(exponent, 1) - 25;
  d.neg == (x[15] == bitone) & class_ok
}

$property
function p_f16_anchors(_u : bits(1)) -> bool = {
  let one = fdecode(FMT16, 0x00003C00);
  one.cls == FFinite & one.mag == 1024 & one.e == -10 &                         // 1.0 = 1024 * 2^-10
  fround(FMT16, false, 1, 0) == 0x00003C00 &                                    // 1.0
  fround(FMT16, true, 2, 0) == 0x0000C000 &                                     // -2.0
  fround(FMT16, false, 1, -1) == 0x00003800 &                                   // 0.5
  fround(FMT16, false, 65504, 0) == 0x00007BFF &                                // the largest finite value
  fround(FMT16, false, 65519, 0) == 0x00007BFF &                                // below the half-way point to 2^16
  fround(FMT16, false, 65520, 0) == 0x00007C00 &                                // the half-way point rounds to infinity (2047 is odd)
  fround(FMT16, false, 1, -14) == 0x00000400 &                                  // the smallest normal value
  fround(FMT16, false, 1023, -24) == 0x000003FF &                               // the largest subnormal value
  fround(FMT16, false, 1, -24) == 0x00000001 &                                  // the smallest subnormal value
  fround(FMT16, false, 1, -25) == 0x00000000 &                                  // half of it: a tie, rounds to the even zero
  fround(FMT16, false, 3, -26) == 0x00000001 &                                  // three quarters of it rounds up
  fround(FMT16, false, 4097, 0) == 0x00006C00 &                                 // 4097 = 2^12 + 1, spacing 4: rounds to 4096
  fround(FMT16, false, 4099, 0) == 0x00006C01 &                                 // 4099 is nearer to 4100
  fround(FMT16, false, 2049, 0) == 0x00006800 &                                 // 2049 is a tie at spacing 2: rounds to the even 2048
  fround(FMT16, false, 2051, 0) == 0x00006802                                   // 2051 is a tie at spacing 2: rounds to the even 2052
}

// ---- bfloat16 (extension bf16; ): the same families as the three formats above, and anchors that do not use FMT_BF16 ----

$property
function p_exact_bf16(x : bits(16)) -> bool = exact_ok(FMT_BF16, 0x0000 @ x)

$property
function p_rne_bf16_k3(o : bits(7), r : bits(3)) -> bool = rne_norm_ok(FMT_BF16, 3, unsigned(o), unsigned(r))

$property
function p_rne_sub_bf16_j2(c : bits(8), r : bits(2)) -> bool = rne_sub_ok(FMT_BF16, 2, unsigned(c), unsigned(r))

$property
function p_ovf_bf16(m : bits(10)) -> bool = ovf_ok(FMT_BF16, unsigned(m))

// bfloat16 is the top 16 bits of a binary32: the value of x as a bfloat16 is the value of x @ 0x0000 as a binary32 (the significand
// scaled by 2^16, the exponent lowered by 16). This ties the format constants of FMT_BF16 to those of FMT32 from outside, as p_f16_decode
// does for binary16 with literals; the special bit patterns are written out.
$property
function p_bf16_decode(x : bits(16)) -> bool = {
  let d = fdecode(FMT_BF16, 0x0000 @ x);
  let t = fdecode(FMT32, x @ 0x0000);
  d.cls == t.cls & d.neg == t.neg &
    (if d.cls == FFinite then t.mag == d.mag * 65536 & t.e == d.e - 16 else true)
}

$property
function p_bf16_specials(_u : bits(1)) -> bool =
  canon_nan(FMT_BF16) == 0x00007FC0 & inf_bits(FMT_BF16, false) == 0x00007F80 & inf_bits(FMT_BF16, true) == 0x0000FF80 &
  zero_bits(FMT_BF16, false) == 0x00000000 & zero_bits(FMT_BF16, true) == 0x00008000 &
  fdecode(FMT_BF16, 0x00007FC0).cls == FNaN & fdecode(FMT_BF16, 0x00007F80).cls == FInf & fdecode(FMT_BF16, 0x00000000).cls == FZero

$property
function p_bf16_anchors(_u : bits(1)) -> bool =
  fround(FMT_BF16, false, 1, 0) == 0x00003F80 &                                 // 1.0
  fround(FMT_BF16, true, 2, 0) == 0x0000C000 &                                  // -2.0
  fround(FMT_BF16, false, 1, -1) == 0x00003F00 &                                // 0.5
  fround(FMT_BF16, false, 255, 120) == 0x00007F7F &                             // the largest finite value
  fround(FMT_BF16, false, 510, 119) == 0x00007F7F &                             // the same value, a larger significand
  fround(FMT_BF16, false, 511, 119) == 0x00007F80 &                             // the half-way point to 2^128 rounds to infinity (511 is odd)
  fround(FMT_BF16, false, 1, -126) == 0x00000080 &                              // the smallest normal value
  fround(FMT_BF16, false, 127, -133) == 0x0000007F &                            // the largest subnormal value
  fround(FMT_BF16, false, 1, -133) == 0x00000001 &                              // the smallest subnormal value
  fround(FMT_BF16, false, 1, -134) == 0x00000000 &                              // half of it: a tie, rounds to the even zero
  fround(FMT_BF16, false, 3, -135) == 0x00000001 &                              // three quarters of it rounds up
  fround(FMT_BF16, false, 257, 0) == 0x00004380 &                               // 257 = 2^8 + 1, spacing 2: a tie, rounds to the even 256
  fround(FMT_BF16, false, 259, 0) == 0x00004382 &                               // 259 is a tie: rounds to the even 260
  fround(FMT_BF16, false, 258, 0) == 0x00004381                                 // 258 is exact
