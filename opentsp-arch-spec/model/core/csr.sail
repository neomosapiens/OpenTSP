// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// The CSR space and the host interface of the chip (platform.md
// sec. 1.3, the host path). The published TSP sources place the PCIe host interface in the C2C
// module: a lightweight DMA engine that emplaces a model onto MEM, an entry point for
// bootstrapping, and a general mechanism for passing interrupts to the host (ISCA20 sec. II-6;
// the link is PCIe Gen4 x16, ISCA22 sec. 5.2). They publish no register map: the
// map below is an implementation choice (provenance.yaml, basis implementation-choice).
//
// Every chip has its own host interface (the published sources give each chip a PCIe host interface in its
// C2C module: ISCA20 sec. II-6; ISCA22 sec. 5.2; the serving system takes input at its first chip and gives
// output at its last: SHIP paper sec. 2.1 and 7.1.1), so every chip has its own CSR space, ingress FIFO,
// doorbells, DMA engine and interrupt delivery, and no host instruction is routed over the links.
//
// One CSR space per chip: 4096 registers of 64 bits, address 0x000 .. 0xFFF. An address that
// no register uses reads 0 and ignores writes; so does a write to a read-only register.
// No instruction reads or writes a CSR (the published instruction list has none): the host
// reaches the space through the harness (model_csr_read, model_csr_write), so a program's
// results never depend on it.
//
//   0x000 CHIP_ID      RO   the chip number
//   0x030 IRQ_CAUSE    W1C  bit n: MSI-X vector n was raised since the bit was cleared
//   0x031 IRQ_MASK     RW   bit n: vector n is enabled (reset 0xFFFF); a disabled vector
//                           sets its cause bit and gives no interrupt until it is enabled:
//                           an interrupt is delivered whenever a cause bit is set and the
//                           vector is enabled, so HostNotify delivers at once on an enabled
//                           vector and enabling a vector whose cause bit is set delivers it
//                           (one interrupt, at the cycle of the write); the cause bit stays
//                           set until the host clears it
//   0x032 IRQ_COUNT    RO   interrupts delivered to the host
//   0x200 + l LINK_CORR[l]    RO   l < c2c.links: vectors-with-errors counter of the link: the flits of arriving vectors that the link
//                           FEC corrected in flight (link FEC, microarchitecture.md sec. 17)
//   0x210 + l LINK_UNCORR[l]  RO   l < c2c.links: the flits of arriving vectors that the link FEC detected and could not correct
//   0x500 + (l - 16) LINK_CORR[l], 0x600 + (l - 16) LINK_UNCORR[l]   RO   16 <= l < c2c.links (at most 128 links): the same counters
//                           for the links above 15, whose numbers would collide with the two blocks above
//   0x300 + i DOORBELL[i]  RW  i < host.doorbells; a write with bit 0 set rings it (a queue
//                           parked by HostWait i resumes the cycle after, else the doorbell
//                           stays set until a HostWait takes it); a write with bit 0 clear
//                           clears it; a read gives 1 while it is set
// MSI-X vector 0 is reserved for the error interrupt (the error registers); vectors 1..15 are raised by
// HostNotify.
//
// The error registers and the error interrupt (microarchitecture.md sec. 16). The published TSP sources record corrected errors "in a
// control and status register (CSR) for an error handler to interrogate later" (ISCA20 sec. II-D), say the host can pass
// interrupts for "a multi-bit memory error" (ISCA20 sec. II-6) and place a fetch-latency parameter "in the ICUs CSR space"
// (ISCA20 sec. III-A); they publish no address, field or depth: every address below is an implementation choice.
//   0x010 ERR_STATUS   W1C  the error register: bit 0 corrected data error, 1 uncorrectable data error, 2 corrected
//                           text error, 3 uncorrectable text error, 4 matrix corrected, 5 matrix uncorrectable, 6 link
//                           corrected, 7 link uncorrectable, 8 spare-bit report, 9 log overflow (bits 4 and 5: the matrix-unit
//                           check, microarchitecture.md sec. 18; bits 6 and 7: the link FEC, sec. 17; bit 8 is set by the feature
//                           that reports it)
//   0x011 ERR_SEVERITY RO   2 when an uncorrectable bit (1, 3, 5, 7) of ERR_STATUS is set, else 1 when a corrected bit
//                           (0, 2, 4, 6, 8) is set, else 0 (so it falls when the host clears ERR_STATUS)
//   0x012 SBE_COUNT    W1C-all  corrected errors, saturating at 2^32 - 1; a write of any value clears it
//   0x013 MBE_COUNT    W1C-all  uncorrectable errors, the same
//   0x014 LOG_COUNT    RO   entries in the log (0 .. ras.log_depth)
//   0x015 LOG_POP      WO   a write removes the oldest entry
//   0x016 ERR_MASK     RW   bit n set: a 0 to 1 change of ERR_STATUS bit n raises MSI-X vector 0 (reset 0x0AA, the
//                           uncorrectable bits 1, 3, 5, 7)
//   0x020 + 2 i, 0x021 + 2 i  LOG[i]  RO  entry i, 0 the oldest, i < ras.log_depth (1..8, so the log fits below 0x030)
//         word 0: cycle 47..0, kind 51..48 (1 corrected data, 2 uncorrectable data, 3 corrected text, 4 uncorrectable
//                 text, 5 and 6 matrix, 7 and 8 link, 9 spare-bit report), queue 63..52 (0x090 the control channel, 0xFFF none)
//         word 1: location 42..0 (a stream register: row 0 (1 = W), stream 5..1, gap 11..6; an SRAM word: slice 6..0,
//                 address 22..7; instruction text: queue 7..0, text word 31..8; a matrix event: lane 15..0, plane 17..16, the
//                 lane 0xFFFF when the check located none; a link event: flit 15..0, link 22..16), superlane 47..43,
//                 position 56..48 (0 for a matrix or link event). A matrix event has the ABC's queue, a link event no queue (0xFFF).
//   0x0F0, 0x0F1  FIRST_ERR0/1  RO  a copy of the first log entry since ERR_STATUS was last cleared (even a dropped one)
//   0x100 + q  FETCH_RT[q]  RO  (q = 0 .. 143) the fetch latency of queue q, icu.fetch_latency
// The diagnostic log (microarchitecture.md sec. 16.3): the diagnostic codes of class `data` (the data-dependent
// column of microarchitecture.md sec. 8) in the reporting order of sec. 8, for the host that has only the CSR space.
//   0x400 DIAG_COUNT   RO   entries in the log (0 .. ras.diag_depth)
//   0x401 DIAG_POP     WO   a write removes the oldest entry
//   0x402 DIAG_STATUS  W1C  bit 0: the overflow flag, set when a record found the log full (the record is dropped); a write of 1 clears it
//   0x410 + i  DIAG[i] RO   entry i, 0 the oldest, i < ras.diag_depth (1..64): cycle 47..0, code 55..48 (the position of the code in
//                           the `diag` type of core/diagnostics.sail), queue 63..56 (0 .. 143, 144 the control channel)
// A new event when the log is full sets ERR_STATUS bit 9 and is dropped; the counters still count. An event that sets a bit
// of ERR_STATUS that is enabled in ERR_MASK (a 0 to 1 change) delivers MSI-X vector 0 in the cycle of the event (the cause
// bit 0 of IRQ_CAUSE is set; IRQ_MASK decides whether the host is interrupted, like any vector). The chip does not stop.

let host_cycles_per_vector : range(1, 64) = config host.cycles_per_vector
let host_ingress_depth : range(1, 256) = config host.ingress_depth
let host_doorbells : range(1, 64) = config host.doorbells
let host_remote_csr : bool = config host.remote_csr

register csr_doorbell : vector(64, bool) = vector_init(false)
register csr_irq_cause : bits(64) = 0x0000000000000000
register csr_irq_mask : bits(64) = 0x000000000000FFFF
register csr_irq_count : bits(64) = 0x0000000000000000
// Link FEC counters of this chip's end of link l (a flit counts once, at the arrival cycle of its vector).
register csr_link_corr : vector(128, bits(64)) = vector_init(0x0000000000000000)
register csr_link_unc : vector(128, bits(64)) = vector_init(0x0000000000000000)

let ras_log_depth : range(1, 8) = config ras.log_depth
let diag_log_depth : range(1, 64) = config ras.diag_depth
let ras_fetch_latency : range(1, 64) = config icu.fetch_latency
// The patrol scrubber (postlude/tick.sail, microarchitecture.md sec. 15.4): a scrub cycle every `period` cycles, 0 = off.
let scrub_period : range(0, 65535) = config scrub.period

register ras_err_status : bits(64) = 0x0000000000000000
register ras_err_mask : bits(64) = 0x00000000000000AA
register ras_sbe_count : bits(64) = 0x0000000000000000
register ras_mbe_count : bits(64) = 0x0000000000000000
register ras_log_n : nat = 0
register ras_log0 : vector(8, bits(64)) = vector_init(0x0000000000000000)
register ras_log1 : vector(8, bits(64)) = vector_init(0x0000000000000000)
register ras_first_valid : bool = false
register diag_log : vector(64, bits(64)) = vector_init(0x0000000000000000)
register diag_n : nat = 0
register diag_overflow : bool = false
register ras_first0 : bits(64) = 0x0000000000000000
register ras_first1 : bits(64) = 0x0000000000000000

// A queue parked by HostWait: the dispatch cycle, the doorbell, and the cycle the queue
// resumes once the doorbell has rung (the host's write cycle + 1).
register hw_since : vector(145, option(nat)) = vector_init(None())
register hw_db : vector(145, nat) = vector_init(0)
register hw_release : vector(145, option(nat)) = vector_init(None())

// The cycle at which the PCIe interface last accepted a vector for the host (PciSend).
register pci_accepted : option(nat) = None()

function host_reset() -> unit = {
  csr_doorbell = vector_init(false);
  csr_irq_cause = 0x0000000000000000;
  csr_irq_mask = 0x000000000000FFFF;
  csr_irq_count = 0x0000000000000000;
  csr_link_corr = vector_init(0x0000000000000000);
  csr_link_unc = vector_init(0x0000000000000000);
  ras_err_status = 0x0000000000000000;
  ras_err_mask = 0x00000000000000AA;
  ras_sbe_count = 0x0000000000000000;
  ras_mbe_count = 0x0000000000000000;
  ras_log_n = 0;
  ras_log0 = vector_init(0x0000000000000000);
  ras_log1 = vector_init(0x0000000000000000);
  ras_first_valid = false;
  ras_first0 = 0x0000000000000000;
  ras_first1 = 0x0000000000000000;
  diag_log = vector_init(0x0000000000000000);
  diag_n = 0;
  diag_overflow = false;
  hw_since = vector_init(None());
  hw_db = vector_init(0);
  hw_release = vector_init(None());
  pci_accepted = None()
}

private function csr_value(n : nat) -> bits(64) = get_slice_int(64, n, 0)

function db_idx(a : int) -> range(0, 63) = if 0 <= a & a <= 63 then a else 0

function ras_idx(i : int) -> range(0, 7) = if 0 <= i & i <= 7 then i else 0

function diag_idx(i : int) -> range(0, 63) = if 0 <= i & i <= 63 then i else 0

private function ras_severity() -> bits(64) =
  if and_vec(ras_err_status, 0x00000000000000AA) != 0x0000000000000000 then csr_value(2)
  else if and_vec(ras_err_status, 0x0000000000000155) != 0x0000000000000000 then csr_value(1)
  else csr_value(0)

function csr_read(a : nat) -> bits(64) =
  if a == 0 then csr_value(host_chip_id())
  else if a == 16 then ras_err_status
  else if a == 17 then ras_severity()
  else if a == 18 then ras_sbe_count
  else if a == 19 then ras_mbe_count
  else if a == 20 then csr_value(ras_log_n)
  else if a == 22 then ras_err_mask
  else if a >= 32 & a < 32 + 2 * ras_log_depth then {
    let i = ediv_int(a - 32, 2);
    if i < ras_log_n then (if emod_int(a - 32, 2) == 0 then ras_log0[ras_idx(i)] else ras_log1[ras_idx(i)]) else csr_value(0)
  }
  else if a == 1024 then csr_value(diag_n)
  else if a == 1026 then (if diag_overflow then csr_value(1) else csr_value(0))
  else if a >= 1040 & a < 1040 + diag_log_depth then {
    let i = a - 1040;
    if i < diag_n then diag_log[diag_idx(i)] else csr_value(0)
  }
  else if a == 240 then (if ras_first_valid then ras_first0 else csr_value(0))
  else if a == 241 then (if ras_first_valid then ras_first1 else csr_value(0))
  else if a >= 256 & a < 256 + 144 then csr_value(ras_fetch_latency)
  else if a == 48 then csr_irq_cause
  else if a == 49 then csr_irq_mask
  else if a == 50 then csr_irq_count
  else if a >= 512 & a < 528 then csr_link_corr[a - 512]
  else if a >= 528 & a < 544 then csr_link_unc[a - 528]
  else if a >= 1280 & a < 1392 then csr_link_corr[a - 1264]
  else if a >= 1536 & a < 1648 then csr_link_unc[a - 1520]
  else if a >= 768 & a < 768 + host_doorbells then (if csr_doorbell[db_idx(a - 768)] then csr_value(1) else csr_value(0))
  else csr_value(0)

// The host rings doorbell i at cycle c: the lowest queue parked on it resumes at c + 1.
private function host_ring(i : range(0, 63), c : nat) -> unit = {
  var taken : bool = false;
  foreach (q from 0 to 143) {
    if not(taken) then match (hw_since[q], hw_release[q]) {
      (Some(_), None()) => if hw_db[q] == i then { hw_release[q] = Some(c + 1); taken = true },
      _ => (),
    }
  };
  if not(taken) then csr_doorbell[i] = true
}

// An interrupt reaches the host in cycle c: MSI-X vector n.
private function host_deliver(c : nat, n : range(0, 15)) -> unit = {
  csr_irq_count = add_bits(csr_irq_count, csr_value(1));
  host_irq_callback(c, n)
}

function ras_pop() -> unit =
  if ras_log_n > 0 then {
    foreach (i from 0 to 6) {
      ras_log0[i] = ras_log0[i + 1];
      ras_log1[i] = ras_log1[i + 1]
    };
    ras_log0[7] = 0x0000000000000000;
    ras_log1[7] = 0x0000000000000000;
    ras_log_n = max(ras_log_n - 1, 0)
  }

function diag_pop() -> unit =
  if diag_n > 0 then {
    foreach (i from 0 to 62) { diag_log[i] = diag_log[i + 1] };
    diag_log[63] = 0x0000000000000000;
    diag_n = max(diag_n - 1, 0)
  }

// One record of the diagnostic log (core/queues.sail diag_report decides which codes come here): kept in the order of arrival, or
// dropped with the overflow flag when the log is full.
function diag_record(c : nat, q : range(0, 144), code : range(0, 31)) -> unit =
  if diag_n < diag_log_depth then {
    diag_log[diag_idx(diag_n)] = get_slice_int(8, q, 0) @ get_slice_int(8, code, 0) @ get_slice_int(48, c, 0);
    diag_n = diag_n + 1
  } else diag_overflow = true

function csr_write(a : nat, v : bits(64), c : nat) -> unit =
  if a == 16 then {
    ras_err_status = and_vec(ras_err_status, not_vec(v));
    if ras_err_status == 0x0000000000000000 then ras_first_valid = false
  }
  else if a == 18 then ras_sbe_count = 0x0000000000000000
  else if a == 19 then ras_mbe_count = 0x0000000000000000
  else if a == 21 then ras_pop()
  else if a == 22 then ras_err_mask = v
  else if a == 48 then csr_irq_cause = and_vec(csr_irq_cause, not_vec(v))
  else if a == 49 then {
    let enabled = and_vec(v, not_vec(csr_irq_mask));
    csr_irq_mask = v;
    foreach (n from 0 to 15) {
      if enabled[n] == bitone & csr_irq_cause[n] == bitone then host_deliver(c, n)
    }
  }
  else if a >= 768 & a < 768 + host_doorbells then {
    if v[0] == bitone then host_ring(db_idx(a - 768), c) else csr_doorbell[db_idx(a - 768)] = false
  }


// A write of a CSR of the chip as the host sees it: the diagnostic log registers (DIAG_POP, DIAG_STATUS), and every other register by the function above.
function csr_write_chip(a : nat, v : bits(64), c : nat) -> unit =
  if a == 1025 then diag_pop()
  else if a == 1026 then { if v[0] == bitone then diag_overflow = false }
  else csr_write(a, v, c)

// HostNotify n at cycle c: MSI-X vector n. The cause bit is set; an enabled vector is delivered.
function host_msix(c : nat, n : range(0, 15)) -> unit = {
  csr_irq_cause[n] = bitone;
  if csr_irq_mask[n] == bitone then host_deliver(c, n)
}

// HostWait i dispatched on queue q at cycle c: takes a ringing doorbell, or parks the queue.
function host_wait(q : range(0, 144), c : nat, i : range(0, 63)) -> unit =
  if csr_doorbell[i] then csr_doorbell[i] = false
  else { hw_since[q] = Some(c); hw_db[q] = i; hw_release[q] = None() }

// The PCIe interface accepts one vector per host.cycles_per_vector cycles (an implementation
// choice derived from the published PCIe Gen4 x16 link at 900 MHz: about 35 bytes per cycle,
// a 320-byte vector in about 9.1 cycles, rounded up to 10; for the chip-to-host direction the
// same pacing is applied, which no published source gives).
function pci_busy(c : nat) -> bool = match pci_accepted {
  Some(a) => a + host_cycles_per_vector > c,
  None() => false,
}

// ---------------------------------------------------------------------------------------
// Recording an error event and raising the error interrupt.

private function ras_sat_inc(x : bits(64)) -> bits(64) =
  if x == 0x00000000FFFFFFFF then x else add_bits(x, csr_value(1))

// One event: kind 1..9 (the table at the top of this file), where it happened (queue, 43-bit location, superlane, syndrome
// position). Counted, logged (or dropped on overflow, bit 9) and, when it sets an enabled ERR_STATUS bit, the error
// interrupt (MSI-X vector 0) is delivered in the cycle c. The superlane field of the record is 5 bits, so 32 superlanes
// is the most a profile can have (core/geometry.sail).
function ras_record(c : nat, q : int, kind : range(1, 9), loc : bits(43), sl : range(0, 31), pos : range(0, 511)) -> unit = {
  let unc : bool = kind == 2 | kind == 4 | kind == 6 | kind == 8;
  if unc then ras_mbe_count = ras_sat_inc(ras_mbe_count)
  else if kind != 9 then ras_sbe_count = ras_sat_inc(ras_sbe_count);
  let old = ras_err_status;
  let qf : bits(12) = if q < 0 | q > 4094 then 0xFFF else if q == 144 then 0x090 else get_slice_int(12, q, 0);
  let w0 : bits(64) = qf @ get_slice_int(4, kind, 0) @ get_slice_int(48, c, 0);
  let w1 : bits(64) = 0b0000000 @ get_slice_int(9, pos, 0) @ get_slice_int(5, sl, 0) @ loc;
  ras_err_status[kind - 1] = bitone;
  if not(ras_first_valid) then { ras_first_valid = true; ras_first0 = w0; ras_first1 = w1 };
  if ras_log_n < ras_log_depth then {
    ras_log0[ras_idx(ras_log_n)] = w0;
    ras_log1[ras_idx(ras_log_n)] = w1;
    ras_log_n = ras_log_n + 1
  } else ras_err_status[9] = bitone;
  let rising = and_vec(ras_err_status, not_vec(old));
  if and_vec(rising, ras_err_mask) != 0x0000000000000000 then host_msix(c, 0)
}

private function ras_loc(x : int, shift : int) -> bits(43) = sail_shiftleft(get_slice_int(43, x, 0), shift)

// The model's ECC event (core/ecc.sail, the instruction-text check): the harness trace line and the error registers.
// kind 0 stream operand (id the stream, addr the reader's gap), 1 SRAM word (id the slice, addr the address), 2 instruction
// text (addr the text word).
function ras_ecc_event(c : nat, q : range(0, 144), kind : range(0, 2), unc : bool, isw : bool, id : range(0, 87), addr : range(0, 16777215), sl : range(0, 31), pos : range(0, 255)) -> unit = {
  ecc_event_callback(c, q, kind, unc, isw, id, addr, sl, pos);
  let loc : bits(43) =
    if kind == 0 then ras_loc(addr, 6) | ras_loc(id, 1) | (if isw then ras_loc(1, 0) else ras_loc(0, 0))
    else if kind == 1 then ras_loc(addr, 7) | ras_loc(id, 0)
    else ras_loc(addr, 8) | ras_loc(q, 0);
  let k : range(1, 9) = if kind == 2 then (if unc then 4 else 3) else (if unc then 2 else 1);
  ras_record(c, q, k, loc, sl, pos)
}

// A word the patrol scrubber (postlude/tick.sail scrub_cycle) found in slice s at address a: corrected in place (unc false, kind 1,
// ERR_STATUS bit 0) or left as it is (unc true, kind 2, bit 1). The queue of the entry is the slice's MEM queue.
function ras_scrub_event(c : nat, s : range(0, 87), unc : bool, a : nat, sl : range(0, 31), pos : range(0, 255)) -> unit = {
  scrub_event_callback(c, s, a, sl, unc);
  ras_record(c, s, if unc then 2 else 1, ras_loc(a, 7) | ras_loc(s, 0), sl, pos)
}

// A spare-bit error report (core/ecc.sail ecc_check_value): the trace line and a log entry of kind 9 (ERR_STATUS bit 8) with the
// stream, the superlane and S[0..3] in the position field.
function ras_spare_event(c : nat, q : range(0, 144), isw : bool, id : range(0, 87), gap : range(0, 16777215), sl : range(0, 31), s03 : bits(4)) -> unit = {
  spare_event_callback(c, q, isw, id, gap, sl, unsigned(s03));
  let loc : bits(43) = ras_loc(gap, 6) | ras_loc(id, 1) | (if isw then ras_loc(1, 0) else ras_loc(0, 0));
  ras_record(c, q, 9, loc, sl, unsigned(s03))
}

// The matrix-unit check event (slices/mxm/mxm_insts.sail, microarchitecture.md sec. 18) in the error registers: a lane the sum check
// located and corrected (unc false, kind 5, ERR_STATUS bit 4) or an ABC it found uncorrectable (unc true, kind 6, bit 5; no lane was
// located, so the caller passes 65535). q is the ABC's queue; log word 1 is plane << 16 | lane (the trace line `T ecc ... mxm corr` and
// the diagnostic OTS-E-ECC-003 are the caller's).
function ras_mxm_event(c : nat, q : int, unc : bool, plane : range(0, 3), lane : nat) -> unit =
  ras_record(c, q, if unc then 6 else 5, ras_loc(plane, 16) | ras_loc(lane, 0), 0, 0)

// The link FEC event (the harness, in the arrival cycle of the vector, microarchitecture.md sec. 17) in the error registers: flit `flit` of a
// vector that arrived on link l was corrected (unc false, kind 7, ERR_STATUS bit 6) or detected uncorrectable (unc true, kind 8, bit 7).
// A link event has no queue (0xFFF); log word 1 is link << 16 | flit.
function ras_link_event(c : nat, l : range(0, 127), flit : nat, unc : bool) -> unit =
  ras_record(c, -1, if unc then 8 else 7, ras_loc(l, 16) | ras_loc(flit, 0), 0, 0)
