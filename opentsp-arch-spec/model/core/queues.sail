// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// The 144 data queues (ISCA 2020 sec. II: 144 independent instruction queues) and the
// unit each one issues to. The per-block totals follow the ScaledML 2020 and ISC 2020
// floorplans: ICU 6 / 14 / 44 / 16 / 44 / 14 / 6, the 14 being "Transposer, Permutor &
// Shifters, PCIe & IO"). The C2C queues are counted in the SXM block (HC34 p.31 draws the
// TX and RX stations inside the SXM). The units inside each block are
// an opentsp decision: 3 queues per MXM plane; per hemisphere 4 distributors and 2
// transposers (HC34 p.31), two shifters used as a pair plus a permuter and a select (ISCA
// 2020 Fig. 8, HC34 p.31), 2 C2C TX and 2 C2C RX queues.
// See opentsp-arch-spec/docs/topology.md sec. 5.
//
//    0 ..  43  MEM west slices 0..43        44 ..  87  MEM east slices 0..43
//   88 .. 103  VXM ALUs 0..15
//  104 .. 115  MXM: plane p at 104 + 3p .. 106 + 3p (planes 0, 1 west; 2, 3 east). With mxm.planes_per_hemisphere = 1 only the
//              first plane of each hemisphere exists: queues 104..106 (plane 0, west) and 110..112 (plane 1, east); queues 107..109
//              and 113..115 are the queues of absent planes. The plane of an east queue is mxm.planes_per_hemisphere + its index
//              in the hemisphere.
//  116 .. 129  SXM west: DST 116..119, TSP 120..121, SHF 122..123, PRM 124, SEL 125,
//              C2C TX 126..127, C2C RX 128..129
//  130 .. 143  SXM east: same order (130 + k)

// The issuing context of an instruction: its queue's unit and instance.
struct issuer = {
  unit  : unit_kind,
  slice : mem_slice,     // MEM: global slice 0..87; others: 0
  plane : range(0, 3),   // MXM plane; others: 0
  sxm   : range(0, 1),   // SXM: 0 = west, 1 = east; others: 0
}

private let no_issuer : issuer = struct { unit = U_VXM, slice = 0, plane = 0, sxm = 0 }

private function sxm_unit_at(k : range(0, 13), east : bool) -> unit_kind =
  if k <= 3 then (if east then U_SXM_E_DST else U_SXM_W_DST)
  else if k <= 5 then (if east then U_SXM_E_TSP else U_SXM_W_TSP)
  else if k <= 7 then (if east then U_SXM_E_SHF else U_SXM_W_SHF)
  else if k == 8 then (if east then U_SXM_E_PRM else U_SXM_W_PRM)
  else if k == 9 then (if east then U_SXM_E_SEL else U_SXM_W_SEL)
  else if k <= 11 then (if east then U_C2C_E_TX else U_C2C_W_TX)
  else (if east then U_C2C_E_RX else U_C2C_W_RX)

function issuer_of(q : queue_id) -> issuer =
  if q <= 43 then { no_issuer with unit = U_MEM_W, slice = q }
  else if q <= 87 then { no_issuer with unit = U_MEM_E, slice = q }
  else if q <= 103 then { no_issuer with unit = U_VXM }
  else if q <= 109 then { no_issuer with unit = U_MXM_W, plane = if q <= 106 then 0 else 1 }
  else if q <= 115 then { no_issuer with unit = U_MXM_E, plane = mxm_planes + (if q <= 112 then 0 else 1) }
  else if q <= 129 then { no_issuer with unit = sxm_unit_at(q - 116, false) }
  else { no_issuer with unit = sxm_unit_at(q - 130, true), sxm = 1 }

// A data queue exists in every profile (the map is fixed); its unit does not when it is the
// queue of a MEM slice that the profile lacks (core/layout.sail mem_slice_present) or of an MXM plane that the
// profile lacks (mxm.planes_per_hemisphere: the index of the plane in its hemisphere is at least the count).
function mxm_queue_present(q : range(104, 115)) -> bool =
  (if q <= 109 then (if q <= 106 then 0 else 1) else (if q <= 112 then 0 else 1)) < mxm_planes

function queue_present(q : range(0, 143)) -> bool =
  if q <= 87 then mem_slice_present(q) else if q >= 104 & q <= 115 then mxm_queue_present(q) else true

// True for the MXM queues (a writer index of the commit ring may also be 144, the control
// channel).
function is_mxm_queue(q : range(0, 144)) -> bool =
  if q <= 143 then (match issuer_of(q).unit { U_MXM_W => true, U_MXM_E => true, _ => false }) else false

// ---------------------------------------------------------------------------------------
// The diagnostic report. Every architectural diagnostic of the model goes through here: the harness callback
// (the `T diag` line, the `E` summary) and, for a code of class `data`, the chip diagnostic log of the CSR space (core/csr.sail
// DIAG_COUNT, DIAG[i], DIAG_POP). The class is the data-dependent column of microarchitecture.md sec. 8 that platform.md sec. 7 names for
// hardware conformance: SXM-001, DEC-003, IQ-002 and IQ-004 whatever the queue; OPR-001 of a MEM queue and of an `sxselect` (`*_SEL`) queue
// (a Gather or Scatter tile address and an `sxselect` select are data; the filter keeps every MEM and `*_SEL` queue: a documented superset,
// as the diagnostic does not say which instruction raised it); TIM-001 of a C2C receive queue.

function diag_number(d : diag) -> range(0, 31) = match d {
  DEC_OPCODE => 0, QUE_BUSY => 1, QUE_UNIT => 2, TIM_NO_VALUE => 3, DIR_UPSTREAM => 4, SRF_COLLIDE => 5, MEM_BANK => 6, OPR_RANGE => 7,
  OPR_ALIGN => 8, RES_NOT_READY => 9, RES_FULL => 10, RES_STATE => 11, SXM_PERMUTE => 12, LNK_BUSY => 13, LNK_OVERRUN => 14, LNK_PACKET => 15,
  IQ_OVERFLOW => 16, IQ_STAGING => 17, IQ_STRANDED => 18, HIO_BUSY => 19, ECC_UNC => 20, LNK_UNC => 21, ECC_MXM => 22, ECC_TXT => 23,
}

function diag_is_data(q : range(0, 144), d : diag) -> bool = {
  let u : option(unit_kind) = if q <= 143 then Some(issuer_of(q).unit) else None();
  match d {
    SXM_PERMUTE => true,
    DEC_OPCODE => true,
    IQ_OVERFLOW => true,
    IQ_STRANDED => true,
    OPR_RANGE => match u { Some(U_MEM_W) => true, Some(U_MEM_E) => true, Some(U_SXM_W_SEL) => true, Some(U_SXM_E_SEL) => true, _ => false },
    TIM_NO_VALUE => match u { Some(U_C2C_W_RX) => true, Some(U_C2C_E_RX) => true, _ => false },
    _ => false,
  }
}

// q is a data queue 0..143, or 144 for the control channel (reported to the harness as 65535).
function diag_report(c : nat, q : range(0, 144), d : diag) -> unit = {
  diag_callback(c, if q == 144 then 65535 else q, diag_code(d));
  if diag_is_data(q, d) then diag_record(c, q, diag_number(d))
}
