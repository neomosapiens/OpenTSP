// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Instruction declarations (sail-riscv sys/insts_begin.sail). Each slice file adds, per
// instruction: a union clause, an encdec mapping clause (32-, 64- or 128-bit layout),
// an execute clause, an assembly clause, and the timing class and unit it belongs to.

scattered union instruction

// What issuing an instruction did. Diagnostics are values (core/diagnostics.sail).
union ExecutionResult = {
  Issued     : unit,
  Diagnostic : diag,
}


// The context of the current issue, set by the cycle driver before calling execute:
// dispatch cycle, issuing unit, and the cycle offset and address offset of an iterated
// MEM reference (postlude/tick.sail).
register issue_cycle : nat = 0
register issue_unit : issuer = struct { unit = U_VXM, slice = 0, plane = 0, sxm = 0 }
register issue_element : nat = 0
// Address offset of an AGT-iterated MEM reference (0 for reference 0).
register issue_addr_offset : int = 0
// Stream id offset of an AGT-iterated MEM reference (StreamStep; 0 for reference 0).
register issue_stream_offset : int = 0
// Queue of the current issue (144 = control channel), and the MEM liquid-thread tag.
register issue_queue : range(0, 144) = 144
register issue_thread : bits(1) = 0b0

val execute : instruction -> ExecutionResult
scattered function execute

val timing_of : instruction -> timing_class
scattered function timing_of

// The instruction's catalogue entry (isa-db/instructions/*.yaml "name"): its row of the
// timing table (core/timing.sail entry_latency).
val entry_of : instruction -> string
scattered function entry_of

// The element type an instruction's timing row may be keyed by, as the assembly names it ("i8"), or ""
// for an instruction without one (core/timing.sail entry_latency_typed).
val timing_type_of : instruction -> string
scattered function timing_type_of

function latency_of_instr(i : instruction) -> latency = entry_latency_typed(entry_of(i), timing_type_of(i), timing_of(i))

// The latency of the current issue (set by the cycle driver with the other issue registers).
register issue_lat : latency = struct { d_func = 0, d_skew = 0 }

// The cycle the current issue executes: its dispatch (reference k of a compound
// instruction: dispatch + k) + d_skew. It reads its stream operands, SRAM and unit state
// in that cycle; its stream results commit at dispatch (+ k) + d_func.
function exec_cycle() -> nat = issue_cycle + issue_element + issue_lat.d_skew
function commit_at() -> nat = issue_cycle + issue_element + issue_lat.d_func

// Units that can issue the instruction.
val issued_by : (instruction, unit_kind) -> bool
scattered function issued_by

// Number of cycles the instruction, dispatched at the given cycle, occupies its queue
// (NOP n: n cycles; every other instruction: 1).
val occupancy : (instruction, nat) -> nat
scattered function occupancy

val assembly : instruction <-> string
scattered mapping assembly

val encdec32 : instruction <-> bits(32)
scattered mapping encdec32

val encdec64 : instruction <-> bits(64)
scattered mapping encdec64

// Words that decode to nothing. Their encdec clauses come last (insts_end.sail).
// There is no 128-bit word: every instruction is 32 or 64 bits (encoding.md).
union clause instruction = ILLEGAL32 : bits(32)
union clause instruction = ILLEGAL64 : bits(64)
