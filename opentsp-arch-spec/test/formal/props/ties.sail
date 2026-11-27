// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Ties-to-even obligations on the rounding core `fround` of core/numerics/fp.sail . They are stated by the
// structure of a binary format (a significand `mag` and an exponent `e`, value mag * 2^e), not by the steps of `fround`: the
// expected result is built by the obligation from the input alone and compared with what the decoder (`fdecode`) reads back from
// the bit pattern `fround` returns. Round to nearest, ties to even: a tie rounds to the even significand.
//
// Cost note: a query that decodes a symbolic bit pattern costs tens of seconds (symbolic powers of two); prove_extra.sh gives this
// suite a longer per-query limit than the model's own obligations. A comparison written as `d.mag * pow2i(d.e)` is much slower than
// one on `mag` and `e` separately, so the obligations compare the two fields.

// ---- ties to even in the normal range ----
// mag = c * 2^k + 2^(k-1) with c = 2^mbits + off an (mbits+1)-bit significand is exactly half way between c and c + 1 (in units
// of 2^k), at exponent 0. The result is the even one; c + 1 = 2^(mbits+1) carries into the next binade (2^mbits at exponent k + 1).
// A rounding that sends every tie up (the mutant m_s1) differs for every odd c.

function tie_norm_ok(f : fmt, k : int, off : int) -> bool = {
  let c = pow2i(f.mbits) + off;
  let mag = c * pow2i(k) + pow2i(k - 1);
  let w = if tmod_int(c, 2) == 0 then c else c + 1;
  let r = fround(f, false, mag, 0);
  let d = fdecode(f, r);
  let roll = w == pow2i(f.mbits + 1);
  d.cls == FFinite & not(d.neg) & d.mag == (if roll then pow2i(f.mbits) else w) & d.e == (if roll then k + 1 else k)
}

$property
function p_tie_f16_k1(o : bits(10)) -> bool = tie_norm_ok(FMT16, 1, unsigned(o))

$property
function p_tie_f16_k2(o : bits(10)) -> bool = tie_norm_ok(FMT16, 2, unsigned(o))

$property
function p_tie_e5m2_k3(o : bits(2)) -> bool = tie_norm_ok(FMT_E5M2, 3, unsigned(o))

$property
function p_tie_e4m3_k3(o : bits(3)) -> bool = tie_norm_ok(FMT_E4M3, 3, unsigned(o))

// Negative control: the claim "every tie rounds up" is false, so the solver must find a counterexample (an odd c).
function tie_up_claim(f : fmt, k : int, off : int) -> bool = {
  let c = pow2i(f.mbits) + off;
  let mag = c * pow2i(k) + pow2i(k - 1);
  let r = fround(f, false, mag, 0);
  let d = fdecode(f, r);
  d.mag == c + 1 & d.e == k
}

$counterexample
function c_tie_up_is_false_f16(o : bits(10)) -> bool = tie_up_claim(FMT16, 1, unsigned(o))
