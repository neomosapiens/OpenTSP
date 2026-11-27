// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Entry points of the sampled binary32 check (f32_sample.sh): the model's own fp_add, fp_sub and fp_mul on two 32-bit patterns.
// This file is not part of the model; the check builds it with core/numerics into a private C program and compares the results with
// the host's `float` arithmetic (f32_sample.c).

val f32_add : (bits(32), bits(32)) -> bits(32)
function f32_add(x, y) = fp_add(FMT32, x, y)

val f32_sub : (bits(32), bits(32)) -> bits(32)
function f32_sub(x, y) = fp_sub(FMT32, x, y)

val f32_mul : (bits(32), bits(32)) -> bits(32)
function f32_mul(x, y) = fp_mul(FMT32, x, y)

// The C backend's start-up code calls main; there is nothing to run.
val main : unit -> unit
function main() = ()
