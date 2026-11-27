// =====================================================================================
// opentsp TSP architecture model (Sail).
//
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

default Order dec

$include <smt.sail>
$include <option.sail>
$include <result.sail>
$include <arith.sail>
$include <string.sail>
$include <mapping.sail>
$include <vector_dec.sail>
$include <generic_equality.sail>
$include <hex_bits.sail>
$include <dec_bits.sail>

// The operator overloads below follow sail-riscv model/prelude/prelude.sail
// (BSD-2-Clause, (c) the sail-riscv contributors), so that model code reads the same.

function not_bit(b : bits(1)) -> bits(1) = if b == 0b1 then 0b0 else 0b1

overload ~ = {not_bool, not_vec, not_bit}

val not : forall ('p : Bool). bool('p) -> bool(not('p))
function not(b) = not_bool(b)

overload operator & = {and_vec}

overload operator | = {or_vec}

overload operator ^ = {xor_vec, concat_str}

val sub_vec = pure {c: "sub_bits", _: "sub_vec"} : forall 'n. (bits('n), bits('n)) -> bits('n)

val sub_vec_int = pure {c: "sub_bits_int", _: "sub_vec_int"} : forall 'n. (bits('n), int) -> bits('n)

overload operator - = {sub_vec, sub_vec_int}

overload min = {min_int}
overload max = {max_int}

val quot_positive_round_zero = pure {_: "tdiv_int"} : forall 'n 'm, 'n >= 0 & 'm > 0. (int('n), int('m)) -> int(div('n, 'm))
val rem_positive_round_zero = pure {_: "tmod_int"} : forall 'n 'm, 'n >= 0 & 'm > 0. (int('n), int('m)) -> int(mod('n, 'm))

val emod_nat = pure {_: "emod_int"} : forall 'n 'm, 'n >= 0 & 'm > 0. (int('n), int('m)) -> int(mod('n, 'm))
overload operator % = {rem_positive_round_zero, emod_nat}
overload operator / = {quot_positive_round_zero}

// The length of a list: the bound of the loops that take one element off a list per step (a list walk ends after that many steps).
val list_len : forall ('a : Type). list('a) -> nat
function list_len(xs) = match xs { [||] => 0, _ :: rest => 1 + list_len(rest) }
