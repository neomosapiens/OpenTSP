// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Architectural diagnostics. The rule for what a diagnosed instruction does (microarchitecture.md
// section 8, "Effect of a diagnosed instruction"): ordinary operand/resource failures suppress
// the affected result. Three OPR-001 cases retain good tiles/lanes; integrity diagnostics
// can accompany poisoned results and RAS side effects (sections 15–17). These do not undo
// text consumption or scheduling. Each partial OPR case is marked `DIAG-EFFECT: <name> per-tile|per-lane` at its code in
// slices/, and test/directed/directed.py checks the markers, this list and the table of the
// document against each other.
// DIAG-EFFECT exceptions: gather, scatter, sxselect
// These are values, not exceptions (sail-riscv: ExecutionResult).

enum diag = {
  DEC_OPCODE,       // OTS-E-DEC-003: does not decode (unknown opcode or slice tag 6, 7, reserved bit set, illegal type/mode)
  QUE_BUSY,         // OTS-E-QUE-001: a parcel that breaks the slot rule (each of its instructions)
  QUE_UNIT,         // OTS-E-QUE-002: instruction issued on a queue whose unit lacks it
  TIM_NO_VALUE,     // OTS-E-TIM-001: no value at the reader's gap in the read cycle
  DIR_UPSTREAM,     // OTS-E-DIR-001: the stream's only writers are at or downstream of the reader
  SRF_COLLIDE,      // OTS-E-SRF-002: two writes to one row/stream/gap in one cycle
  MEM_BANK,         // OTS-E-MEM-001: a Read and a Write dispatched together hit the same bank
  OPR_RANGE,        // OTS-E-OPR-001: operand value out of range (for a Gather or Scatter tile address and an sxselect sel, per tile or lane: the good ones still take effect)
  OPR_ALIGN,        // OTS-E-OPR-002: multi-stream operand not aligned to its width
  RES_NOT_READY,    // OTS-E-RES-001: MXM state, or a Permute's SXM map, used before it is ready
  RES_FULL,         // OTS-E-RES-002: MXM weight buffer overfilled
  RES_STATE,        // OTS-E-RES-003: two instructions change one unit-state part in one cycle
  SXM_PERMUTE,      // OTS-E-SXM-001: Permute map is not bijective (a source lane repeats)
  LNK_BUSY,         // OTS-E-LNK-001: Send or Transmit on a link that accepted a vector fewer than c2c.cycles_per_vector cycles ago
  LNK_OVERRUN,      // OTS-E-LNK-002: a vector arrived at a link whose receive FIFO holds c2c.rx_depth vectors and was dropped (reported by the harness)
  LNK_PACKET,       // OTS-E-LNK-003: Send or Transmit released into the cycles of a csr or ctrl packet in flight on the link
  IQ_OVERFLOW,      // OTS-E-IQ-002: fetched text arrived at a full IQ and the units that did not fit were lost (once per queue and cycle, by Enqueue)
  IQ_STAGING,       // OTS-E-IQ-003: an Ifetch or IfetchM found the queue's fetch staging FIFO holding icu.fetch_staging_depth payloads: it fetched nothing
  IQ_STRANDED,      // OTS-E-IQ-004: text was left in an IQ at the end of the run that never completes a parcel (reported once per queue after the last cycle)
  HIO_BUSY,         // OTS-E-HIO-001: PciSend within host.cycles_per_vector cycles of the previous one
  ECC_UNC,          // OTS-E-ECC-001: a stream operand had an uncorrectable (multi-bit) error in some superlane; reported once, after the instruction's own checks
  LNK_UNC,          // OTS-E-LNK-004: a Receive took a vector whose link transfer had an uncorrectable (detected, not corrected) FEC error; the vector is written, its bad superlanes poisoned
  ECC_MXM,          // OTS-E-ECC-003: the sum check of an ABC found an error it cannot correct; the ABC has no effect (microarchitecture.md sec. 18)
  ECC_TXT           // OTS-E-ECC-002: a text word of the instruction had an uncorrectable (multi-bit) error: the instruction consumes its units and has no effect
}

mapping diag_code : diag <-> string = {
  DEC_OPCODE    <-> "OTS-E-DEC-003",
  QUE_BUSY      <-> "OTS-E-QUE-001",
  QUE_UNIT      <-> "OTS-E-QUE-002",
  TIM_NO_VALUE  <-> "OTS-E-TIM-001",
  DIR_UPSTREAM  <-> "OTS-E-DIR-001",
  SRF_COLLIDE   <-> "OTS-E-SRF-002",
  MEM_BANK      <-> "OTS-E-MEM-001",
  OPR_RANGE     <-> "OTS-E-OPR-001",
  OPR_ALIGN     <-> "OTS-E-OPR-002",
  RES_NOT_READY <-> "OTS-E-RES-001",
  RES_FULL      <-> "OTS-E-RES-002",
  RES_STATE     <-> "OTS-E-RES-003",
  SXM_PERMUTE   <-> "OTS-E-SXM-001",
  LNK_BUSY      <-> "OTS-E-LNK-001",
  LNK_OVERRUN   <-> "OTS-E-LNK-002",
  LNK_PACKET    <-> "OTS-E-LNK-003",
  IQ_OVERFLOW   <-> "OTS-E-IQ-002",
  IQ_STAGING    <-> "OTS-E-IQ-003",
  IQ_STRANDED   <-> "OTS-E-IQ-004",
  HIO_BUSY      <-> "OTS-E-HIO-001",
  ECC_UNC       <-> "OTS-E-ECC-001",
  LNK_UNC       <-> "OTS-E-LNK-004",
  ECC_MXM       <-> "OTS-E-ECC-003",
  ECC_TXT       <-> "OTS-E-ECC-002",
}
