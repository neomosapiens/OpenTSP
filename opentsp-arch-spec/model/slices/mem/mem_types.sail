// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// MEM instruction operand types. They are declared before sys/insts_begin.sail because
// the instruction union refers to them (sail-riscv *_types.sail convention).

enum mem_dir = {MEM_READ, MEM_WRITE}

