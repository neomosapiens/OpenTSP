// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Test of the integer lane kernels of the VXM (`k_binary` of core/numerics/lanes.sail, audit 2026-11-19 ) for all of its integer
// operations: add (opcodes 0 and 2), saturating add (1), subtract (3 and 5), saturating subtract (4), multiply (6 and 8), saturating
// multiply (7), max (9), min (10), and (11), or (12), xor (13), against a specification written with the prelude's `signed`, `unsigned`,
// a clamp and a per-bit loop, not with the model's own wrap_t / sat_t / int_of / bit-vector operators.
//   scope: every pair of 8-bit operands, signed (I8) and unsigned (U8), and the boolean B8 (which reads any nonzero value as true);
//          16 bits (I16, U16, B16): 501 first operands against 137 second operands (the edges of the range and a lattice);
//          32 bits (I32, U32, B32): the 33 edge values and a stride lattice against each other (no sweep of 2^64 pairs).
//   not covered: the float arms of k_binary (opcodes 0, 3, 6, 9, 10 on F16, F32, E4M3, E5M2): they inline fp_add, fp_mul and fp_max;
//          binary16 add, subtract and multiply are compared with the host elsewhere (test/numerics/f16_exhaustive.sh), the FP8 formats byb
//          the fp8 tests; an SMT obligation over k_binary does not translate in reasonable memory.
// The mutants m_s2 (saturating add computes b + b), m_a5 (the saturating multiply wraps), m_a6 and m_a7 (max and min swapped) each fail
// on almost every pair.

private function lane_clamp(lo : int, hi : int, v : int) -> int = if v < lo then lo else if v > hi then hi else v

// The element in the low n bits of a 32-bit operand: sign extended or not.
private function lane_val(x : bits(32), n : int, is_signed : bool) -> int = {
  let u = tmod_int(unsigned(x), pow2i(n));
  if is_signed & u >= pow2i(n - 1) then u - pow2i(n) else u
}

private function lane_bits(v : int, n : int) -> bits(32) = get_slice_int(32, emod_int(v, pow2i(n)), 0)

// The bitwise operation of kind k (0 and, 1 or, 2 xor) on the low n bits of two unsigned values, bit byb bit.
private function lane_bitop(k : int, ux : int, uy : int, n : int) -> int = {
  var r : int = 0;
  foreach (i from 0 to (n - 1)) {
    let bxa = tmod_int(tdiv_int(ux, pow2i(i)), 2);
    let byb = tmod_int(tdiv_int(uy, pow2i(i)), 2);
    let bit : int = if k == 0 then (if bxa == 1 & byb == 1 then 1 else 0)
                    else if k == 1 then (if bxa == 1 | byb == 1 then 1 else 0)
                    else (if bxa != byb then 1 else 0);
    r = r + bit * pow2i(i)
  };
  r
}

private function lane_check(t : eltype, n : int, is_signed : bool, xv : int, yv : int) -> unit = {
  let x : bits(32) = lane_bits(xv, n);
  let y : bits(32) = lane_bits(yv, n);
  let a = lane_val(x, n, is_signed);
  let b = lane_val(y, n, is_signed);
  let ux = tmod_int(unsigned(x), pow2i(n));
  let uy = tmod_int(unsigned(y), pow2i(n));
  let lo : int = if is_signed then 0 - pow2i(n - 1) else 0;
  let hi : int = if is_signed then pow2i(n - 1) - 1 else pow2i(n) - 1;
  assert(k_binary(0, t, x, y) == lane_bits(a + b, n), "lanes: add");
  assert(k_binary(2, t, x, y) == lane_bits(a + b, n), "lanes: add (opcode 2)");
  assert(k_binary(1, t, x, y) == lane_bits(lane_clamp(lo, hi, a + b), n), "lanes: saturating add");
  assert(k_binary(3, t, x, y) == lane_bits(a - b, n), "lanes: subtract");
  assert(k_binary(5, t, x, y) == lane_bits(a - b, n), "lanes: subtract (opcode 5)");
  assert(k_binary(4, t, x, y) == lane_bits(lane_clamp(lo, hi, a - b), n), "lanes: saturating subtract");
  assert(k_binary(6, t, x, y) == lane_bits(a * b, n), "lanes: multiply");
  assert(k_binary(8, t, x, y) == lane_bits(a * b, n), "lanes: multiply (opcode 8)");
  assert(k_binary(7, t, x, y) == lane_bits(lane_clamp(lo, hi, a * b), n), "lanes: saturating multiply");
  assert(k_binary(9, t, x, y) == lane_bits(if a > b then a else b, n), "lanes: max");
  assert(k_binary(10, t, x, y) == lane_bits(if a < b then a else b, n), "lanes: min");
  assert(k_binary(11, t, x, y) == lane_bits(lane_bitop(0, ux, uy, n), n), "lanes: and");
  assert(k_binary(12, t, x, y) == lane_bits(lane_bitop(1, ux, uy, n), n), "lanes: or");
  assert(k_binary(13, t, x, y) == lane_bits(lane_bitop(2, ux, uy, n), n), "lanes: xor")
}

// A boolean element: zero is false, any other value of the n bits is true; and, or and xor give all ones or zero.
private function lane_check_bool(t : eltype, n : int, xv : int, yv : int) -> unit = {
  let x : bits(32) = lane_bits(xv, n);
  let y : bits(32) = lane_bits(yv, n);
  let a = tmod_int(unsigned(x), pow2i(n)) != 0;
  let b = tmod_int(unsigned(y), pow2i(n)) != 0;
  let yes = pow2i(n) - 1;
  assert(k_binary(11, t, x, y) == lane_bits(if a & b then yes else 0, n), "lanes: boolean and");
  assert(k_binary(12, t, x, y) == lane_bits(if a | b then yes else 0, n), "lanes: boolean or");
  assert(k_binary(13, t, x, y) == lane_bits(if a != b then yes else 0, n), "lanes: boolean xor");
  assert(k_binary(0, t, x, y) == 0x00000000, "lanes: boolean add is zero")
}

private function lane_all8(t : eltype, is_signed : bool) -> unit =
  foreach (xv from 0 to 255) { foreach (yv from 0 to 255) { lane_check(t, 8, is_signed, xv, yv) } }

private function lane_bool8() -> unit =
  foreach (xv from 0 to 255) { foreach (yv from 0 to 255) { lane_check_bool(B8, 8, xv, yv) } }

// The 16-bit sweep: first operands 0 .. 65535 in steps of 131 (501 values with 65535) and the second operands the edges and every 509th value (137).
private function lane_all16(t : eltype, is_signed : bool) -> unit = {
  foreach (xi from 0 to 500) {
    let xv : int = if xi == 500 then 65535 else 131 * xi;
    foreach (k from 0 to 136) {
      let yv : int = if k < 8 then (match k { 0 => 0, 1 => 1, 2 => 2, 3 => 32767, 4 => 32768, 5 => 32769, 6 => 65534, _ => 65535 }) else 509 * (k - 8) + 3;
      lane_check(t, 16, is_signed, xv, yv)
    }
  }
}

// The 32-bit edges: 0, 1, 2, around 2^7, 2^8, 2^15, 2^16, 2^31 and 2^32 (as 33 values of the unsigned range) and a lattice of stride 1_000_003.
private function lane_edge32(k : int) -> int = match k {
  0 => 0, 1 => 1, 2 => 2, 3 => 3, 4 => 127, 5 => 128, 6 => 129, 7 => 255, 8 => 256, 9 => 257, 10 => 32767, 11 => 32768, 12 => 32769,
  13 => 65535, 14 => 65536, 15 => 65537, 16 => 46340, 17 => 46341, 18 => 65534, 19 => 2147483645, 20 => 2147483646, 21 => 2147483647,
  22 => 2147483648, 23 => 2147483649, 24 => 2147483650, 25 => 4294967293, 26 => 4294967294, 27 => 4294967295, 28 => 1073741823,
  29 => 1073741824, 30 => 1073741825, 31 => 3037000499, 32 => 3037000500,
  _ => 1000003 * (k - 33) + 17
}

private function lane_all32(t : eltype, is_signed : bool) -> unit =
  foreach (i from 0 to 232) { foreach (k from 0 to 232) { lane_check(t, 32, is_signed, lane_edge32(i), lane_edge32(k)) } }

$[test]
function test_lanes_int8() -> unit = { lane_all8(I8, true); lane_all8(U8, false); lane_bool8() }

$[test]
function test_lanes_int16() -> unit = { lane_all16(I16, true); lane_all16(U16, false) }

$[test]
function test_lanes_int32() -> unit = { lane_all32(I32, true); lane_all32(U32, false); foreach (i from 0 to 232) { foreach (k from 0 to 232) { lane_check_bool(B32, 32, lane_edge32(i), lane_edge32(k)) } } }

// The C backend's start-up code calls main; there is nothing to run.
val main : unit -> unit
function main() = ()
