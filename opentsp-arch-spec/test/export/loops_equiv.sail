// The equivalence check of the transcendental helpers whose `while` loops became bounded `foreach` loops (core/numerics/transcendental.sail):
// the model's functions against the loops they replaced, on edge values and random arguments of every size up to the 4096-bit cap.
// loops_equiv.sh substitutes the number of random arguments for ITERATIONS and compiles this file after the model's prelude and numerics modules.

// Independent unbounded-loop reference: exp_small runs until zero, without the old incorrect 399-term cap.
function exp_small_loop(y : int, p : int) -> (int, int) = {
  let one = pow2i(p);
  var term : int = one;
  var sum : int = one;
  var k : int = 1;
  var err : int = 2;
  while (term > 0 | term < 0) do {
    term = tdiv_int(term * y, one);
    term = tdiv_int(term, k);
    sum = sum + term;
    err = err + 2;
    k = k + 1
  };
  (sum - err, sum + err)
}

function ln2_iv_loop(p : int) -> (int, int) = {
  let one = pow2i(p + 8);
  var t : int = tdiv_int(one, 3);
  var sum : int = 0;
  var k : int = 0;
  while t > 0 do {
    sum = sum + tdiv_int(t, 2 * k + 1);
    t = tdiv_int(t, 9);
    k = k + 1
  };
  let v = 2 * sum;
  let err = 4 * (k + 2);
  (idiv_floor(v - err, 256), idiv_ceil(v + err, 256))
}

function log_iv_loop(mag : int, e : int, p : int) -> (int, int) = {
  let b = ilog2(mag);
  let k = b + e;
  let one = pow2i(p + 8);
  let num = (mag - pow2i(b)) * one;
  let den = mag + pow2i(b);
  let z = tdiv_int(num, den);
  let z2 = tdiv_int(z * z, one);
  var t : int = z;
  var sum : int = 0;
  var j : int = 0;
  while (t > 0) & j < 2000 do {
    sum = sum + tdiv_int(t, 2 * j + 1);
    t = tdiv_int(t * z2, one);
    j = j + 1
  };
  let lnm = 2 * sum;
  let errm = 4 * (j + 3);
  let (l2lo, l2hi) = ln2_iv_loop(p + 8);
  let lo = (if k >= 0 then k * l2lo else k * l2hi) + lnm - errm;
  let hi = (if k >= 0 then k * l2hi else k * l2lo) + lnm + errm;
  (idiv_floor(lo, 256), idiv_ceil(hi, 256))
}

function isqrt_loop(n : int) -> int = {
  if n <= 0 then 0 else {
    var x : int = pow2i(tdiv_int(ilog2(n), 2) + 1);
    var y : int = tdiv_int(x + tdiv_int(n, x), 2);
    while y < x do { x = y; y = tdiv_int(x + tdiv_int(n, x), 2) };
    x
  }
}

// The Newton steps isqrt takes: the most of them over the arguments checked is printed, next to the bound of the model.
register isqrt_steps_max : int = 0

function isqrt_steps(n : int) -> int = {
  var steps : int = 0;
  if n > 0 then {
    var x : int = pow2i(tdiv_int(ilog2(n), 2) + 1);
    var y : int = tdiv_int(x + tdiv_int(n, x), 2);
    while y < x do { x = y; y = tdiv_int(x + tdiv_int(n, x), 2); steps = steps + 1 }
  };
  steps
}

function same(a : (int, int), b : (int, int)) -> bool = {
  let (x1, y1) = a;
  let (x2, y2) = b;
  x1 == x2 & y1 == y2
}

function fail(what : string, a : int) -> unit = {
  print_endline("MISMATCH " ^ what ^ " " ^ dec_str(a));
  exit()
}

function check_isqrt(n : int) -> unit = {
  if isqrt(n) != isqrt_loop(n) then fail("isqrt", n);
  let s = isqrt_steps(n);
  if s > isqrt_steps_max then isqrt_steps_max = s;
  // the bound of the model's loop: a step beyond it would not be taken
  if n > 0 & s > ilog2(ilog2(n) + 1) + 17 then fail("isqrt steps exceed the bound", n)
}

function lcg(s : int) -> int = tmod_int(s * 6364136223846793005 + 1442695040888963407, 18446744073709551616)

// The C backend's start-up code calls main; there is nothing to run there (loops_equiv.c calls loops_equiv_run).
val main : unit -> unit
function main() = ()

val loops_equiv_run : unit -> unit

function loops_equiv_run() = {
  var checked : int = 0;
  // ln2_iv: every precision from -20 to 300, then the Ziv precisions and their neighbours
  foreach (p from -20 to 300) { if not(same(ln2_iv(p), ln2_iv_loop(p))) then fail("ln2_iv", p); checked = checked + 1 };
  foreach (j from 6 to 12) {
    foreach (d from 0 to 4) {
      let p = pow2i(j) + d - 2;
      if not(same(ln2_iv(p), ln2_iv_loop(p))) then fail("ln2_iv", p);
      checked = checked + 1
    }
  };
  // isqrt: every power of two to 2^8300 and the integers next to it, the squares of those and the integers next to them
  foreach (j from 0 to 8300) {
    foreach (d from 0 to 4) {
      let n = pow2(j) + d - 2;
      check_isqrt(n);
      let r = pow2i(tdiv_int(j, 2)) + d - 2;
      check_isqrt(r * r + d - 2);
      checked = checked + 2
    }
  };
  // exp_small and log_iv at the precisions the model uses (64 to 4096 and the +8 of log) on the arguments the callers pass: exp_small gets
  // |y| of at most about ln2 scaled, log_iv a positive magnitude
  var s : int = 88172645463325252;
  var i : int = 0;
  while i < ITERATIONS do {
    s = lcg(s);
    let p = 64 * pow2i(tmod_int(tdiv_int(s, 1099511627776), 7));
    s = lcg(s);
    let y = tdiv_int(s * pow2i(p - 64), 18446744073709551616) - pow2i(p - 1);
    if not(same(exp_small(y, p), exp_small_loop(y, p))) then fail("exp_small", y);
    s = lcg(s);
    var mag : int = 1 + tdiv_int(s, pow2i(tdiv_int(s, 288230376151711744)));
    if tmod_int(i, 8) == 0 then { s = lcg(s); mag = mag * 18446744073709551616 + s };
    s = lcg(s);
    let e = tmod_int(tdiv_int(s, 4096), 600) - 300;
    let pl = 64 * pow2i(tmod_int(tdiv_int(s, 1099511627776), 3));
    if not(same(log_iv(mag, e, pl), log_iv_loop(mag, e, pl))) then fail("log_iv", mag);
    s = lcg(s);
    var n : int = tdiv_int(s, pow2i(tdiv_int(s, 288230376151711744)));
    if tmod_int(i, 16) == 0 then {
      var c : int = 1 + tmod_int(tdiv_int(s, 1099511627776), 130);
      while c > 0 do { s = lcg(s); n = n * 18446744073709551616 + s; c = c - 1 }
    };
    check_isqrt(n);
    checked = checked + 3;
    i = i + 1
  };
  print_endline("loops equivalence: " ^ dec_str(checked) ^ " arguments checked, no mismatch; most Newton steps of isqrt " ^ dec_str(isqrt_steps_max))
}
