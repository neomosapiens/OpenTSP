// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Unit tests for MEM dispatch paths (slices/mem/mem_insts.sail
// "Dispatch paths", postlude/tick.sail "Instruction front end"). The end-to-end behaviour
// (records of one cycle, AGT loops per path, a Read and Write pair) is in
// test/directed/directed.py, section "Dispatch".

// Read and Gather use the R path, Write and Scatter the W path; nothing else has a path.
$[test]
function test_dispatch_ref_paths() -> unit = {
  let a = sram_addr_of(1);
  let s = Streamref(0b000000);
  assert(ref_path(MEM_DIRECT(MEM_READ, a, s, 0b0)) == Some(0));
  assert(ref_path(MEM_GATHER(s, s, 0x0, 0b0)) == Some(0));
  assert(ref_path(MEM_DIRECT(MEM_WRITE, a, s, 0b0)) == Some(1));
  assert(ref_path(MEM_SCATTER(s, s, 0x0, 0b1)) == Some(1));
  assert(ref_path(MEM_COUNTDOWN(0x0001)) == None());
  assert(ref_path(IFETCH(s)) == None());
  assert(ref_thread(MEM_SCATTER(s, s, 0x0, 0b1)) == 0b1);
  assert(ref_thread(MEM_DIRECT(MEM_READ, a, s, 0b0)) == 0b0)
}

// The bank of a direct reference includes the AGT offset; indirect references have none.
$[test]
function test_dispatch_ref_bank() -> unit = {
  let s = Streamref(0b000000);
  let lo = MEM_DIRECT(MEM_READ, sram_addr_of(0), s, 0b0);
  assert(match ref_bank(lo, 0) { BankOf(b) => b == 0b0, _ => false });
  assert(match ref_bank(lo, 1 * 2 ^ bank_bit) { BankOf(b) => b == 0b1, _ => false });
  // An address outside the slice uses no bank, so it meets no other access.
  assert(match ref_bank(lo, -1) { BankNone() => true, _ => false });
  assert(match ref_bank(lo, words_per_slice) { BankNone() => true, _ => false });
  assert(not(banks_meet(ref_bank(lo, words_per_slice), BankAny())));
  assert(match ref_bank(MEM_GATHER(s, s, 0x0, 0b0), 0) { BankAny() => true, _ => false })
}

// Configuration records carry no path field: agt_config is the same for both paths.
$[test]
function test_dispatch_config_fields() -> unit = {
  assert(agt_config(MEM_STEP(0x0018)) == Some((AGT_STEP, 0x0018)));
  assert(agt_config(MEM_ITERATIONS(0x006F)) == Some((AGT_ITERATIONS, 0x006F)));
  assert(agt_config(MEM_DIRECT(MEM_READ, sram_addr_of(0), Streamref(0b000000), 0b0)) == None())
}

// Each MEM queue has one AGT per path; reset clears both, and an open W-path configuration
// or a W-path loop is pending work for the harness.
$[test]
function test_dispatch_agt_per_path() -> unit = {
  reset();
  assert(agt_get(5, 0).state == AGT_IDLE & agt_get(5, 1).state == AGT_IDLE);
  agt_set(5, 1, { agt_get(5, 1) with state = AGT_RUN, next = 40 });
  assert(agt_get(5, 0).state == AGT_IDLE);
  assert(not(model_quiescent()));
  assert(model_next_event(10) == 40);
  agt_set(5, 1, { agt_get(5, 1) with state = AGT_CONFIG });
  assert(model_next_event(10) == 11);
  reset();
  assert(agt_get(5, 1).state == AGT_IDLE & model_quiescent());
  assert(not(path_busy(5, 0, 0)) & not(path_busy(5, 1, 0)))
}

// ---------------------------------------------------------------------------------------

// A finished AGT loop holds its path through its trailing waits (the countdown of every
// level after the last reference).
$[test]
function test_dispatch_agt_tail() -> unit = {
  reset();
  let l0 : agt_level = struct { countdown = 3, step = 0, sstep = 0, iters = 2, idx = 1 };
  agt_set(5, 1, { agt_get(5, 1) with state = AGT_RUN, levels = [agt_get(5, 1).levels with 0 = l0], level = 0, pos = 0, last = 40 });
  agt_refresh_test(5, 1);
  assert(agt_get(5, 1).state == AGT_IDLE);
  assert(path_tail(5, 1, 41) & path_tail(5, 1, 43));
  assert(not(path_tail(5, 1, 44)));
  assert(not(path_tail(5, 0, 41)));
  reset();
  assert(not(path_tail(5, 1, 41)))
}
