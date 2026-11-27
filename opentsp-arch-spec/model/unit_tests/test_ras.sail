// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Unit tests for the error CSRs (core/csr.sail: the error registers and the error interrupt): the counters, the log and its overflow, the
// first-error copy, the severity, the clearing rules and the error interrupt. The layouts below are written again
// from the table at the top of core/csr.sail, not from the model's functions.

private function tv(n : nat) -> bits(64) = get_slice_int(64, n, 0)

private function ras_w0(c : int, kind : int, q : int) -> bits(64) =
  get_slice_int(64, c, 0) | sail_shiftleft(get_slice_int(64, kind, 0), 48) | sail_shiftleft(get_slice_int(64, q, 0), 52)

// One corrected data error of a stream register: queue 88, stream 3, gap 21, superlane 2, position 9.
private function ras_corrected(c : nat) -> unit = ras_ecc_event(c, 88, 0, false, true, 3, 21, 2, 9)

$[test]
function test_ras_record_and_registers() -> unit = {
  host_reset();
  ras_corrected(62);
  assert(csr_read(16) == tv(1), "ERR_STATUS bit 0 is set by a corrected data error");
  assert(csr_read(17) == tv(1), "severity 1 for a corrected error");
  assert(csr_read(18) == tv(1) & csr_read(19) == tv(0), "SBE_COUNT 1, MBE_COUNT 0");
  assert(csr_read(20) == tv(1), "one log entry");
  assert(csr_read(32) == ras_w0(62, 1, 88), "log word 0: cycle, kind 1, queue");
  // word 1: row W (bit 0), stream 3 (bits 5..1), gap 21 (bits 11..6), superlane 2 (47..43), position 9 (56..48)
  assert(csr_read(33) == (get_slice_int(64, 1 + 3 * 2 + 21 * 64, 0) | sail_shiftleft(get_slice_int(64, 2, 0), 43) | sail_shiftleft(get_slice_int(64, 9, 0), 48)), "log word 1: location, superlane, position");
  assert(csr_read(240) == csr_read(32) & csr_read(241) == csr_read(33), "FIRST_ERR is a copy of the first entry");
  assert(csr_read(48) == tv(0) & csr_read(50) == tv(0), "a corrected error raises no interrupt");
  // An uncorrectable data error: bit 1, severity 2, MBE_COUNT, the interrupt (MSI-X vector 0, cause bit 0, count 1).
  ras_ecc_event(70, 88, 1, true, false, 5, 9, 4, 77);
  assert(csr_read(16) == (if ras_log_depth > 1 then tv(3) else tv(3 + 512)), "bits 0 and 1 (and 9 when the log of one entry is full)");
  assert(csr_read(17) == tv(2), "severity 2");
  assert(csr_read(19) == tv(1), "MBE_COUNT 1");
  assert(csr_read(50) == tv(1) & csr_read(48) == tv(1), "the error interrupt: count 1, cause bit 0");
  // SRAM location: slice 5 (bits 6..0), address 9 (bits 19..7)
  if ras_log_depth > 1 then assert(csr_read(34) == ras_w0(70, 2, 88), "second entry, kind 2");
  if ras_log_depth > 1 then assert(csr_read(35) == (get_slice_int(64, 5 + 9 * 128, 0) | sail_shiftleft(get_slice_int(64, 4, 0), 43) | sail_shiftleft(get_slice_int(64, 77, 0), 48)), "SRAM location");
  assert(csr_read(240) == ras_w0(62, 1, 88), "FIRST_ERR still holds the first entry")
}

$[test]
function test_ras_text_event_and_control_queue() -> unit = {
  host_reset();
  // A text word error on the control channel (queue 144 is logged as 0x090): kind 3, text word 5 (bits 31..8), queue 144 & 0xFF.
  ras_ecc_event(11, 144, 2, false, false, 0, 5, 0, 3);
  assert(csr_read(32) == ras_w0(11, 3, 144), "kind 3, the control channel is queue 144 = 0x090");
  assert(csr_read(33) == (get_slice_int(64, 5 * 256 + 144, 0) | sail_shiftleft(get_slice_int(64, 3, 0), 48)), "text location: queue 7..0, word 31..8");
  assert(csr_read(16) == tv(4), "ERR_STATUS bit 2 (corrected text)");
  ras_ecc_event(12, 3, 2, true, false, 0, 6, 0, 9);
  assert(csr_read(16) == (if ras_log_depth > 1 then tv(12) else tv(12 + 512)), "bit 3 (uncorrectable text)");
  assert(csr_read(50) == tv(1), "the uncorrectable text error raises the interrupt")
}

$[test]
function test_ras_log_overflow_pop_and_clear() -> unit = {
  host_reset();
  foreach (i from 0 to ras_log_depth) { ras_corrected(100 + i) };    // depth + 1 events
  assert(csr_read(20) == tv(ras_log_depth), "the log holds ras.log_depth entries");
  assert(csr_read(18) == tv(ras_log_depth + 1), "the counter counts every event");
  assert(csr_read(16) == tv(1 + 512), "bit 9: a dropped event");
  assert(csr_read(32 + 2 * (ras_log_depth - 1)) == ras_w0(100 + ras_log_depth - 1, 1, 88), "the last kept entry; the dropped one is not there");
  csr_write(21, tv(0), 200);
  assert(csr_read(20) == tv(ras_log_depth - 1), "LOG_POP removes one");
  if ras_log_depth > 1 then assert(csr_read(32) == ras_w0(101, 1, 88), "the second entry is now entry 0");
  assert(csr_read(32 + 2 * (ras_log_depth - 1)) == tv(0), "the vacated slot reads 0");
  assert(csr_read(240) == ras_w0(100, 1, 88), "FIRST_ERR keeps the first event after a pop");
  csr_write(16, tv(1), 201);
  assert(csr_read(16) == tv(512), "W1C clears bit 0 only");
  csr_write(16, tv(512), 202);
  assert(csr_read(16) == tv(0) & csr_read(17) == tv(0), "everything cleared, severity 0");
  assert(csr_read(240) == tv(0), "FIRST_ERR is re-armed when ERR_STATUS is clear");
  csr_write(18, tv(5), 203);
  assert(csr_read(18) == tv(0), "any write clears the counter");
  ras_corrected(300);
  assert(csr_read(240) == ras_w0(300, 1, 88), "the next event is the new first error")
}

$[test]
function test_ras_saturation_and_mask() -> unit = {
  host_reset();
  ras_sbe_count = 0x00000000FFFFFFFF;
  ras_corrected(5);
  assert(csr_read(18) == 0x00000000FFFFFFFF, "SBE_COUNT saturates at 2^32 - 1");
  csr_write(22, tv(1), 6);
  assert(csr_read(22) == tv(1), "ERR_MASK is read-write");
  csr_write(16, tv(1), 7);
  ras_corrected(8);
  assert(csr_read(50) == tv(1), "a corrected error raises the interrupt once its bit is enabled");
  ras_corrected(9);
  assert(csr_read(50) == tv(1), "no second interrupt while the bit stays set (a 0 to 1 change only)");
  csr_write(16, tv(1), 10);
  ras_corrected(11);
  assert(csr_read(50) == tv(2), "after the host cleared the bit the next event raises it again");
  // IRQ_MASK bit 0 clear: the cause bit is set and the host is not interrupted.
  csr_write(49, 0x000000000000FFFE, 12);
  csr_write(16, tv(1), 13);
  ras_corrected(14);
  assert(csr_read(50) == tv(2) & csr_read(48) == tv(1), "a masked vector keeps the cause bit and delivers nothing")
}

$[test]
function test_ras_fetch_rt_and_unmapped() -> unit = {
  host_reset();
  assert(csr_read(256) == tv(ras_fetch_latency) & csr_read(256 + 143) == tv(ras_fetch_latency), "FETCH_RT[q] for q 0 and 143");
  assert(csr_read(256 + 144) == tv(0), "no FETCH_RT[144]");
  csr_write(256, tv(77), 1);
  assert(csr_read(256) == tv(ras_fetch_latency), "FETCH_RT is read-only");
  assert(csr_read(23) == tv(0) & csr_read(24) == tv(0), "unmapped registers read 0")
}

// The matrix-unit and link FEC events (kinds 5 to 8). The layouts are written again from the table at the top of core/csr.sail:
// a matrix event has the ABC's queue and plane << 16 | lane (0xFFFF: no lane located), a link event no queue (0xFFF) and link << 16 | flit.
$[test]
function test_ras_matrix_and_link_events() -> unit = {
  host_reset();
  // a corrected lane: kind 5, ERR_STATUS bit 4, SBE_COUNT, severity 1, no interrupt (ERR_MASK reset 0x0AA)
  ras_mxm_event(200, 105, false, 3, 17);
  assert(csr_read(16) == tv(16), "ERR_STATUS bit 4 (matrix corrected)");
  assert(csr_read(17) == tv(1) & csr_read(18) == tv(1) & csr_read(19) == tv(0), "severity 1, SBE_COUNT 1");
  assert(csr_read(32) == ras_w0(200, 5, 105), "kind 5, queue 105");
  assert(csr_read(33) == tv(3 * 65536 + 17), "plane 3 << 16 | lane 17");
  assert(csr_read(50) == tv(0) & csr_read(48) == tv(0), "a corrected matrix error raises no interrupt");
  // an uncorrectable ABC: kind 6, bit 5, MBE_COUNT, severity 2, the error interrupt
  ras_mxm_event(210, 105, true, 1, 65535);
  assert(csr_read(16) == (if ras_log_depth > 1 then tv(48) else tv(48 + 512)), "bits 4 and 5");
  assert(csr_read(17) == tv(2) & csr_read(19) == tv(1), "severity 2, MBE_COUNT 1");
  assert(csr_read(50) == tv(1) & csr_read(48) == tv(1), "the uncorrectable matrix error raises MSI-X vector 0");
  if ras_log_depth > 1 then assert(csr_read(34) == ras_w0(210, 6, 105) & csr_read(35) == tv(65536 + 65535), "kind 6, plane 1, no lane located");
  host_reset();
  // a corrected flit: kind 7, bit 6, no queue (0xFFF), link 5 << 16 | flit 9; an uncorrectable flit: kind 8, bit 7, the interrupt
  ras_link_event(300, 5, 9, false);
  assert(csr_read(16) == tv(64) & csr_read(17) == tv(1) & csr_read(18) == tv(1), "bit 6, severity 1, SBE_COUNT 1");
  assert(csr_read(32) == ras_w0(300, 7, 4095) & csr_read(33) == tv(5 * 65536 + 9), "kind 7, no queue, link 5 flit 9");
  assert(csr_read(50) == tv(0), "a corrected link flit raises no interrupt");
  ras_link_event(301, 5, 10, true);
  assert(csr_read(16) == (if ras_log_depth > 1 then tv(192) else tv(192 + 512)), "bits 6 and 7");
  assert(csr_read(17) == tv(2) & csr_read(19) == tv(1) & csr_read(50) == tv(1), "severity 2, MBE_COUNT 1, the interrupt");
  if ras_log_depth > 1 then assert(csr_read(34) == ras_w0(301, 8, 4095) & csr_read(35) == tv(5 * 65536 + 10), "kind 8, link 5 flit 10");
  assert(csr_read(240) == ras_w0(300, 7, 4095), "FIRST_ERR is the first link event");
  // ERR_MASK bit 6 makes a corrected link flit raise the interrupt as well
  host_reset();
  csr_write(22, tv(64), 0);
  ras_link_event(400, 0, 1, false);
  assert(csr_read(50) == tv(1), "the interrupt of a corrected flit when ERR_MASK bit 6 is set")
}


// The diagnostic log (core/csr.sail): the layout is written again from the table at the top of core/csr.sail.
private function diag_w(c : int, code : int, q : int) -> bits(64) =
  get_slice_int(64, c, 0) | sail_shiftleft(get_slice_int(64, code, 0), 48) | sail_shiftleft(get_slice_int(64, q, 0), 56)

$[test]
function test_diag_log_records_class_data_only() -> unit = {
  host_reset();
  // OPR-001 (code 7) on MEM queue 5 is class data; QUE-002 (code 2) and OPR-001 on a VXM queue (88) are not
  diag_report(10, 5, OPR_RANGE);
  diag_report(11, 5, QUE_UNIT);
  diag_report(12, 88, OPR_RANGE);
  assert(csr_read(1024) == tv(1), "one record: the data-class OPR-001 of a MEM queue");
  assert(csr_read(1040) == diag_w(10, 7, 5), "DIAG[0]: cycle 47..0, code 55..48, queue 63..56");
  // DEC-003 (code 0) of the control channel: queue field 144; IQ-002 (code 16), IQ-004 (18), SXM-001 (12), TIM-001 (3) of a C2C receive queue 128
  diag_report(13, 144, DEC_OPCODE);
  diag_report(14, 7, IQ_OVERFLOW);
  diag_report(15, 144, IQ_STRANDED);
  diag_report(16, 124, SXM_PERMUTE);
  diag_report(17, 128, TIM_NO_VALUE);
  diag_report(18, 8, TIM_NO_VALUE);
  let want = if diag_log_depth < 6 then diag_log_depth else 6;
  assert(csr_read(1024) == tv(want), "the records of class data, in order, up to the depth");
  if diag_log_depth >= 2 then assert(csr_read(1041) == diag_w(13, 0, 144), "DIAG[1]: the control channel is queue 144");
  if diag_log_depth >= 5 then assert(csr_read(1044) == diag_w(16, 12, 124), "DIAG[4]: SXM-001");
  if diag_log_depth >= 6 then assert(csr_read(1045) == diag_w(17, 3, 128), "DIAG[5]: TIM-001 of a C2C receive queue (the TIM-001 of MEM queue 8 is not logged)")
}

$[test]
function test_diag_log_overflow_pop_and_clear() -> unit = {
  host_reset();
  foreach (i from 0 to diag_log_depth) { diag_report(100 + i, 3, DEC_OPCODE) };      // depth + 1 records
  assert(csr_read(1024) == tv(diag_log_depth), "the log holds ras.diag_depth entries");
  assert(csr_read(1026) == tv(1), "DIAG_STATUS bit 0: the record that found the log full was dropped");
  assert(csr_read(1040) == diag_w(100, 0, 3), "DIAG[0] is the oldest record");
  assert(csr_read(1040 + diag_log_depth) == tv(0), "the address after the last entry reads 0");
  csr_write_chip(1024, tv(0), 0);
  csr_write_chip(1040, tv(0), 0);
  assert(csr_read(1024) == tv(diag_log_depth) & csr_read(1040) == diag_w(100, 0, 3), "DIAG_COUNT and DIAG[i] are read-only");
  csr_write_chip(1025, tv(1), 0);
  assert(csr_read(1024) == tv(diag_log_depth - 1), "DIAG_POP removes the oldest");
  if diag_log_depth >= 2 then assert(csr_read(1040) == diag_w(101, 0, 3), "the later entries move down");
  assert(csr_read(1026) == tv(1), "a pop does not clear the overflow flag");
  csr_write_chip(1026, tv(0), 0);
  assert(csr_read(1026) == tv(1), "a write of 0 does not clear it");
  csr_write_chip(1026, tv(1), 0);
  assert(csr_read(1026) == tv(0), "a write of 1 clears the overflow flag");
  foreach (i from 0 to diag_log_depth) { csr_write_chip(1025, tv(1), 0) };
  assert(csr_read(1024) == tv(0) & csr_read(1040) == tv(0), "popping an empty log does nothing")
}

// The array of spare-bit reports keeps the 256 latest (core/streams.sail srf_report): a report on a new register overwrites
// the oldest; the result says whether that one was still in its row (register diag + c <= 44).
$[test]
function test_srf_report_eviction() -> unit = {
  srf_reset();
  let xs : bits(superlanes) = sail_zeros(superlanes);
  let sp : vector(30, bits(superlanes)) = vector_init(sail_zeros(superlanes));
  var evicted : bool = false;
  // 256 registers: the first (gap 20, cycle 100) then 255 on gap 21 at cycles 101..355
  evicted = srf_report(RowE, 20, 100, 1, xs, sp);
  assert(not(evicted), "the first report fits");
  foreach (k from 1 to 255) { evicted = evicted | srf_report(RowE, 21, 100 + k, 1, xs, sp) };
  assert(not(evicted), "256 reports fit");
  assert(not(srf_report(RowE, 20, 100, 1, xs, sp)), "a report of a register that has one merges: nothing is evicted");
  // the 257th register at cycle 120 overwrites the report of cycle 100: it is at register 20 - 100 + 120 = 40 <= 44, still in its row
  assert(srf_report(RowW, 5, 120, 1, xs, sp), "the oldest report was still in its row: evicted live");
  // the next one overwrites the report at cycle 101, gap 21: register 21 - 101 + 200 = 120 > 44, gone
  assert(not(srf_report(RowW, 6, 200, 1, xs, sp)), "an older report that left its row is not a live eviction")
}
