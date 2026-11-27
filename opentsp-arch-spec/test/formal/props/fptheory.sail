// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Closed-form cross-check of the binary16 rounding of `fround` (core/numerics/fp.sail) against the floating-point theory of the solver.
// The property says: the result of fround for the value mag * 2^e equals z. A query of this property alone is satisfiable (z is free);
// test/formal/fround_fp_theory.py adds, for each exponent e, the assertion that z is the bit pattern of the IEEE binary16 value that the
// solver's own theory computes for mag * 2^e (an exact conversion of the integer to a wider format, an exact scaling by the power of two, and a
// single round-to-nearest-even conversion to binary16), and requires every such query to be unsatisfiable. The definition of binary16
// is then the solver's, independent of the model's `fmt` constants and of its `fdecode`.

$property
function p_fround_f16_vs_fp(m : bits(15), e : bits(7), z : bits(16)) -> bool =
  unsigned(m) == 0 | fround(FMT16, false, unsigned(m), signed(e)) == 0x0000 @ z
