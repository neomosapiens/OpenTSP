// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Terminates the scattered instruction definitions (sail-riscv postlude/insts_end.sail).
// The ILLEGAL encdec clauses come last so that every unmatched word decodes to them.

mapping clause encdec32 = ILLEGAL32(w) <-> w
mapping clause encdec64 = ILLEGAL64(w) <-> w

function clause execute ILLEGAL32(_) = Diagnostic(DEC_OPCODE)
function clause execute ILLEGAL64(_) = Diagnostic(DEC_OPCODE)

function clause timing_of ILLEGAL32(_) = T_CU
function clause timing_of ILLEGAL64(_) = T_CU

function clause entry_of ILLEGAL32(_) = "illegal"
function clause entry_of ILLEGAL64(_) = "illegal"

// Every instruction without an element type: the clauses of the typed ones precede this one.
function clause timing_type_of _ = ""

function clause issued_by(ILLEGAL32(_), _) = true
function clause issued_by(ILLEGAL64(_), _) = true

function clause occupancy(_, _) = 1

mapping clause assembly = ILLEGAL32(w) <-> "illegal32" ^ spc() ^ hex_bits_32(w)
mapping clause assembly = ILLEGAL64(w) <-> "illegal64" ^ spc() ^ hex_bits_64(w)

end instruction
end execute
end timing_of
end entry_of
end timing_type_of
end issued_by
end occupancy
end assembly
end encdec32
end encdec64
