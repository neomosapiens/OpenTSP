// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Correctly rounded TanH/Exp/RSqrt/Log (bit-exact contract).

// VXM transcendentals: TanH (21), Exp (22), RSqrt (23), Log (24).
// 
// Contract (opentsp-arch-spec/docs/microarchitecture.md sec. 5): the result is the input's exact
// mathematical function value rounded ONCE, RNE, directly to the destination
// format (f16 or f32). There is no ULP tolerance: results are bit-exact.
// i8 inputs are converted exactly to a real, evaluated, then rounded to i8
// with RNE and saturation.
// 
// Method: fixed-point evaluation with PREC fractional bits, carrying a rigorous
// error bound; the result is refined until the enclosing interval rounds to a
// single destination value (Ziv's strategy). All arithmetic is exact integers.

// A real interval [lo, hi] * 2^-p with integer lo <= hi.
struct ival = { lo : int, hi : int, p : int }

function idiv_floor(a : int, b : int) -> int = {
  // b > 0
  let q = tdiv_int(a, b);
  if q * b > a then q - 1 else q
}

function idiv_ceil(a : int, b : int) -> int = 0 - idiv_floor(0 - a, b)

// Input value x = neg * mag * 2^e as a fixed-point interval at precision p.
function fx_of(neg : bool, mag : int, e : int, p : int) -> ival = {
  let s = e + p;
  if s >= 0 then {
    let v = mag * pow2i(s);
    let v2 = if neg then 0 - v else v;
    struct { lo = v2, hi = v2, p = p }
  } else {
    let d = pow2i(0 - s);
    let lo = idiv_floor(if neg then 0 - mag else mag, d);
    let hi = idiv_ceil(if neg then 0 - mag else mag, d);
    struct { lo = lo, hi = hi, p = p }
  }
}

// exp(y) for y = Y * 2^-p, p >= 0 and |y| <= 1/2, as interval at precision p.
// Taylor series with integer truncation; returns [lo, hi] enclosing exp(y).
function exp_small(y : int, p : int) -> (int, int) = {
  let one = pow2i(p);
  var term : int = one;   // y^k / k!, scaled
  var sum : int = one;
  var k : int = 1;
  var err : int = 2;
  // |term_k| <= floor(|term_(k-1)| / 2), so p + 1 steps reach zero.
  // Each recurrence error is < 2 units; after the zero term the exact tail is < 1.
  // Thus 2 per computed term plus the initial 2 encloses rounding and the tail.
  foreach (_step from 1 to (p + 1)) {
    if (term > 0 | term < 0) then {
      term = tdiv_int(term * y, one);
      term = tdiv_int(term, k);
      sum = sum + term;
      err = err + 2;
      k = k + 1
    }
  };
  (sum - err, sum + err)
}

// ln(2) to precision p, enclosed: atanh series ln2 = 2 * atanh(1/3).
function ln2_iv(p : int) -> (int, int) = {
  let one = pow2i(p + 8);
  var t : int = tdiv_int(one, 3);
  var sum : int = 0;
  var k : int = 0;
  // t starts below 2^(p + 8) and shrinks by a factor of 9 per term, so it is zero after p + 8 terms
  foreach (_step from 0 to (p + 8)) {
    if t > 0 then {
      sum = sum + tdiv_int(t, 2 * k + 1);
      t = tdiv_int(t, 9);
      k = k + 1
    }
  };
  let v = 2 * sum;
  let err = 4 * (k + 2);
  (idiv_floor(v - err, 256), idiv_ceil(v + err, 256))
}

// exp(x) for real interval x at precision p, as (mantissa-interval, exponent):
// exp(x) in [lo, hi] * 2^(n - p). Range reduction x = n ln2 + r.
function exp_iv(x : ival) -> (int, int, int) = {
  let p = x.p;
  let (l2lo, l2hi) = ln2_iv(p);
  let mid = tdiv_int(x.lo + x.hi, 2);
  let n = idiv_floor(mid + tdiv_int(l2lo, 2), l2lo);
  // r = x - n ln2, enclosed
  let rlo = if n >= 0 then x.lo - n * l2hi else x.lo - n * l2lo;
  let rhi = if n >= 0 then x.hi - n * l2lo else x.hi - n * l2hi;
  let (alo, _) = exp_small(rlo, p);
  let (_, bhi) = exp_small(rhi, p);
  (alo, bhi, n)
}

// ln(x) for positive x = mag * 2^e: ln x = (k) ln2 + ln(m), m in [1, 2).
// ln(m) = 2 atanh((m-1)/(m+1)). Returns interval at precision p.
function log_iv(mag : int, e : int, p : int) -> (int, int) = {
  let b = ilog2(mag);
  let k = b + e;                  // x = 2^k * m, m = mag / 2^b in [1,2)
  let one = pow2i(p + 8);
  // z = (m-1)/(m+1) = (mag - 2^b) / (mag + 2^b)
  let num = (mag - pow2i(b)) * one;
  let den = mag + pow2i(b);
  let z = tdiv_int(num, den);
  let z2 = tdiv_int(z * z, one);
  var t : int = z;
  var sum : int = 0;
  var j : int = 0;
  // at most 2000 terms (j = 0 .. 1999); the series stops earlier when a term is zero
  foreach (_step from 0 to 1999) {
    if (t > 0) & j < 2000 then {
      sum = sum + tdiv_int(t, 2 * j + 1);
      t = tdiv_int(t * z2, one);
      j = j + 1
    }
  };
  let lnm = 2 * sum;
  let errm = 4 * (j + 3);
  let (l2lo, l2hi) = ln2_iv(p + 8);
  let lo = (if k >= 0 then k * l2lo else k * l2hi) + lnm - errm;
  let hi = (if k >= 0 then k * l2hi else k * l2lo) + lnm + errm;
  (idiv_floor(lo, 256), idiv_ceil(hi, 256))
}

// Integer square root floor.
function isqrt(n : int) -> int = {
  if n <= 0 then 0 else {
    var x : int = pow2i(tdiv_int(ilog2(n), 2) + 1);
    var y : int = tdiv_int(x + tdiv_int(n, x), 2);
    // Newton's steps from the power of two above the root: the error halves while it is large, then squares at every step, so
    // ilog2(ilog2(n) + 1) + 16 steps (29 for the 4096-bit cap) are more than the iteration ever takes; test/export/loops_equiv.sh checks it
    foreach (_step from 0 to (ilog2(ilog2(n) + 1) + 16)) {
      if y < x then { x = y; y = tdiv_int(x + tdiv_int(n, x), 2) }
    };
    x
  }
}

// Round the real number [lo, hi] * 2^(sh) to format f; Some only if both ends agree.
function round_iv(f : fmt, lo : int, hi : int, sh : int) -> option(bits(32)) = {
  let a = if lo < 0 then fround(f, true, 0 - lo, sh) else fround(f, false, lo, sh);
  let b = if hi < 0 then fround(f, true, 0 - hi, sh) else fround(f, false, hi, sh);
  if a == b then Some(a) else None()
}

// Enclosure of op(a) as [lo, hi] * 2^sh at working precision p. a is finite, in-domain.
function tr_interval(op : int, a : fval, p : int) -> (int, int, int) = {
  if op == 22 then {
    let x = fx_of(a.neg, a.mag, a.e, p);
    let (lo, hi, n) = exp_iv(x);
    (lo, hi, n - p)
  } else if op == 24 then {
    let (lo, hi) = log_iv(a.mag, a.e, p);
    (lo, hi, 0 - p)
  } else if op == 23 then {
    // x = m2 * 2^e2 with e2 even; 1/sqrt(x) = 2^(2p) / sqrt(m2 * 2^(2p)) * 2^(-p - e2/2)
    let e2 : int = if tmod_int(a.e, 2) == 0 then a.e else a.e - 1;
    let m2 : int = if tmod_int(a.e, 2) == 0 then a.mag else a.mag * 2;
    let s = isqrt(m2 * pow2i(2 * p));            // s <= sqrt(m2) 2^p < s + 1
    let num = pow2i(4 * p);
    (tdiv_int(num, s + 1), tdiv_int(num, s) + 1, 0 - 3 * p - tdiv_int(e2, 2))
  } else {
    // tanh(|x|) = 1 - 2 / (E + 1), E = exp(2|x|); odd function.
    let x = fx_of(false, a.mag, a.e + 1, p);
    let (lo, hi, n) = exp_iv(x);
    // exp_iv gives E = [lo, hi] * 2^(n - p); scaled by 2^p that is [lo, hi] * 2^n.
    let elo : int = if n >= 0 then lo * pow2i(n) else idiv_floor(lo, pow2i(0 - n));
    let ehi : int = if n >= 0 then hi * pow2i(n) else idiv_ceil(hi, pow2i(0 - n));
    let one = pow2i(p);
    let big = pow2i(2 * p);
    // E in [elo, ehi] * 2^-p ; 2/(E+1) = 2 * 2^p / (E' + 2^p)
    let tlo : int = big - idiv_ceil(2 * big * one, elo + one);
    let thi : int = big - idiv_floor(2 * big * one, ehi + one);
    let lo2 : int = if a.neg then 0 - thi else tlo;
    let hi2 : int = if a.neg then 0 - tlo else thi;
    (lo2 - 1, hi2 + 1, 0 - 2 * p)
  }
}

// Ziv loop: refine until the enclosure rounds to one value of format f. It terminates
// whenever the exact value is not a rounding boundary of f, which holds for every input
// that reaches it: exp(x), ln(x) and tanh(x) of a nonzero rational x (x != 1 for
// ln) are transcendental (Lindemann-Weierstrass), so never a dyadic boundary; and
// 1/sqrt(x) of a float x is never a midpoint, which would need an odd significand m > 1
// with m^2 dividing a power of two. None() only if 4096 bits do not suffice; the callers
// treat that as a model error, and test_transcendental_converges checks every i8 and f16
// input and a stride of f32 inputs.
function tr_eval(op : int, f : fmt, a : fval) -> option(bits(32)) = {
  var p : int = 64;
  var res : option(bits(32)) = None();
  // p is 64, 128, ... 4096: seven rounds at most
  foreach (_round from 0 to 6) {
    if is_none(res) & p <= 4096 then {
      let (lo, hi, sh) = tr_interval(op, a, p);
      res = round_iv(f, lo, hi, sh);
      p = p * 2
    }
  };
  res
}

// RNE of v * 2^sh to an integer (v may be negative).
function rne_int(v : int, sh : int) -> int = {
  if sh >= 0 then v * pow2i(sh) else {
    let d = pow2i(0 - sh);
    let q = idiv_floor(v, d);
    let rem = v - q * d;
    let half = tdiv_int(d, 2);
    if rem > half | (rem == half & tmod_int(q, 2) != 0) then q + 1 else q
  }
}

// Ziv loop to the nearest integer (RNE); None() as for tr_eval.
function tr_eval_int(op : int, a : fval) -> option(int) = {
  var p : int = 64;
  var res : option(int) = None();
  // p is 64, 128, ... 4096: seven rounds at most
  foreach (_round from 0 to 6) {
    if is_none(res) & p <= 4096 then {
      let (lo, hi, sh) = tr_interval(op, a, p);
      let x = rne_int(lo, sh);
      let y = rne_int(hi, sh);
      if x == y then res = Some(x);
      p = p * 2
    }
  };
  res
}

// Transcendental of an FP element (format f). Edge cases per microarchitecture.md sec. 5.
// None() only if the Ziv loop does not converge (a model error, see tr_eval).
function fp_tr_value(op : int, f : fmt, x : bits(32)) -> option(bits(32)) = {
  let a = fdecode(f, x);
  match a.cls {
    FNaN => Some(canon_nan(f)),
    FInf =>
      Some(if op == 21 then (if a.neg then fp_of_int(f, -1) else fp_of_int(f, 1))
      else if op == 22 then (if a.neg then zero_bits(f, false) else inf_bits(f, false))
      else if op == 23 then (if a.neg then canon_nan(f) else zero_bits(f, false))
      else (if a.neg then canon_nan(f) else inf_bits(f, false))),
    FZero =>
      Some(if op == 21 then zero_bits(f, a.neg)    // tanh(+-0) = the canonical +-0 (no stray bits)
      else if op == 22 then fp_of_int(f, 1)
      else if op == 23 then inf_bits(f, a.neg)    // rsqrt(+-0) = +-Inf (IEEE rSqrt)
      else inf_bits(f, true)),                    // log(+-0) = -Inf
    FFinite =>
      if (op == 23 | op == 24) & a.neg then Some(canon_nan(f))
      else if op == 24 & a.mag == pow2i(0 - a.e) & a.e <= 0 then Some(zero_bits(f, false))  // log(1) = +0
      else if op == 22 & (a.e + ilog2(a.mag)) > 8 then
        Some(if a.neg then zero_bits(f, false) else inf_bits(f, false))  // |x| >= 512: exp saturates
      else if op == 21 & (a.e + ilog2(a.mag)) > 5 then
        Some(fp_of_int(f, if a.neg then -1 else 1))    // |x| >= 64: tanh rounds to +-1 in f32
      else tr_eval(op, f, a)
  }
}

function fp_transcendental(op : int, f : fmt, x : bits(32)) -> bits(32) =
  match fp_tr_value(op, f, x) {
    Some(r) => r,
    None()  => internal_error(__FILE__, __LINE__, "fp_transcendental: Ziv loop did not converge"),
  }

// i8 transcendental: evaluate on the exact integer input, round to i8 (RNE, saturate).
// Out-of-domain or infinite results map as in microarchitecture.md sec. 5: NaN -> 0, +Inf -> 127, -Inf -> -128.
// None() only if the Ziv loop does not converge (a model error, see tr_eval).
function i8_tr_value(op : int, v : int) -> option(int) =
  if v == 0 then Some(if op == 22 then 1 else if op == 23 then 127 else if op == 24 then -128 else 0)
  else if (op == 23 | op == 24) & v < 0 then Some(0)
  else if op == 22 & v > 5 then Some(127)          // e^5 = 148.4 > 127
  else if op == 22 & v < -1 then Some(0)           // e^-1 = 0.37 -> 0; e^-2 smaller
  // rsqrt(4) = 0.5 is the one exact tie among i8 inputs (1/sqrt(v) = n + 1/2 needs
  // v = 4 / (2n + 1)^2): the enclosure never leaves the tie, and RNE gives 0.
  else if op == 23 & v == 4 then Some(0)
  else {
    let a : fval = struct { cls = FFinite, neg = v < 0, mag = if v < 0 then 0 - v else v, e = 0 };
    tr_eval_int(op, a)
  }

function i8_transcendental(op : int, x : bits(32)) -> bits(32) =
  match i8_tr_value(op, s_of_bits(x, 8)) {
    Some(r) => bits_of_s(sat_s(r, 8), 8),
    None()  => internal_error(__FILE__, __LINE__, "i8_transcendental: Ziv loop did not converge"),
  }
