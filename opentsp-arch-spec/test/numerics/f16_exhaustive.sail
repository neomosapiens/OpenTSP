// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Entry points of the exhaustive binary16 check (f16_exhaustive.sh): the model's own fp_add, fp_sub and fp_mul on two 16-bit
// patterns, with the result as 16 bits. This file is not part of the model; the check builds it with core/numerics into a
// private C program and compares every result with the host's `_Float16` arithmetic (f16_exhaustive.c).

val f16_add : (bits(16), bits(16)) -> bits(16)
function f16_add(x, y) = fp_add(FMT16, 0x0000 @ x, 0x0000 @ y)[15 .. 0]

val f16_sub : (bits(16), bits(16)) -> bits(16)
function f16_sub(x, y) = fp_sub(FMT16, 0x0000 @ x, 0x0000 @ y)[15 .. 0]

val f16_mul : (bits(16), bits(16)) -> bits(16)
function f16_mul(x, y) = fp_mul(FMT16, 0x0000 @ x, 0x0000 @ y)[15 .. 0]

// The C backend's start-up code calls main; there is nothing to run.
val main : unit -> unit
function main() = ()
