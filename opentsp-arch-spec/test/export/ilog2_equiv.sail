// The equivalence check of ilog2: the model's loop-free ilog2 (core/numerics/fp.sail) against the one-bit-at-a-time loop it
// replaced, on every power of two from 2^0 to 2^8300 and the four integers next to it, and on random integers of every length up to
// 2^1400. ilog2_equiv.sh substitutes the number of random integers for ITERATIONS and compiles this file after the model's
// prelude and numerics modules.

// The function ilog2 had before the loop-free version.
function ilog2_loop(n : int) -> int = {
  var k : int = 0;
  var m : int = n;
  while m >= 2 do { m = tdiv_int(m, 2); k = k + 1 };
  k
}

function ilog2_check(n : int) -> unit = {
  let a = ilog2(n);
  let b = ilog2_loop(n);
  if a != b then {
    print_endline("MISMATCH n=" ^ dec_str(n) ^ " cascade=" ^ dec_str(a) ^ " loop=" ^ dec_str(b));
    exit()
  }
}

function lcg(s : int) -> int = tmod_int(s * 6364136223846793005 + 1442695040888963407, 18446744073709551616)

function main() -> unit = {
  var checked : int = 0;
  // every power of two below 2^8300 and the integers next to it
  foreach (j from 0 to 8300) {
    foreach (d from 0 to 4) {
      let n = pow2(j) + d - 2;
      if n >= 1 then { ilog2_check(n); checked = checked + 1 }
    }
  };
  // random integers: a 64-bit value shifted right by a random amount, every 16th extended by up to 20 more 64-bit chunks
  var s : int = 88172645463325252;
  var i : int = 0;
  while i < ITERATIONS do {
    s = lcg(s);
    var n : int = tdiv_int(s, pow2i(tdiv_int(s, 288230376151711744)));
    if tmod_int(i, 16) == 0 then {
      var c : int = 1 + tmod_int(tdiv_int(s, 1099511627776), 20);
      while c > 0 do { s = lcg(s); n = n * 18446744073709551616 + s; c = c - 1 }
    };
    ilog2_check(n + 1);
    checked = checked + 1;
    i = i + 1
  };
  print_endline("ilog2 equivalence: " ^ dec_str(checked) ^ " integers checked, no mismatch")
}
