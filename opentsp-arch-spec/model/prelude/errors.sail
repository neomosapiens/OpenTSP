// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Model-internal exceptions (as in sail-riscv prelude/errors.sail).
// Architectural diagnostics are NOT exceptions; they are values (core/diagnostics.sail).

union exception = {
  Error_not_implemented : string,
  Error_internal_error  : string,
}

val not_implemented : forall ('a : Type). string -> 'a
function not_implemented(message) = throw Error_not_implemented(message)

val internal_error : forall ('a : Type). (string, int, string) -> 'a
// A proof build (`sail -smt -D OPENTSP_PROOF_BUILD`) raises a constant message: the SMT backend cannot translate dec_str and string
// concatenation, so an obligation could not mention a function that may fail. Every other build keeps the message with file and line.
$ifdef OPENTSP_PROOF_BUILD
function internal_error(_, _, _) = throw Error_internal_error("internal_error")
$else
function internal_error(file, line, s) = throw Error_internal_error(file ^ ":" ^ dec_str(line) ^ ": " ^ s)
$endif

function string_of_exception(e : exception) -> string =
  match e {
    Error_not_implemented(s) => "Not implemented error: " ^ s,
    Error_internal_error(s)  => "Internal error: " ^ s,
  }
