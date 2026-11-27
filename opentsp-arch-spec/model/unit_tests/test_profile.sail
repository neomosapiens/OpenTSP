// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Unit tests for the profile parameters: sram.words_per_slice,
// geometry.mem_slices_per_hemisphere and mxm.rows. They are profile-generic: every expected
// value is computed from the configuration, and tools/model_unit_tests.sh runs them on ref,
// mini and the test profiles of config/test/.

// The queue map has 144 ids in every profile; MEM slice s exists when (s mod 44) < the key,
// and an absent slice has no station.
$[test]
function test_f3_mem_slices() -> unit = {
  var present : nat = 0;
  foreach (q from 0 to 143) {
    let want : bool = q > 87 | (q % 44) < mem_slices_per_hemisphere;
    // the MXM queues 104..115 of an absent plane (mxm.planes_per_hemisphere 1) do not exist either
    let k : int = if q <= 109 then (if q <= 106 then 0 else 1) else (if q <= 112 then 0 else 1);
    let mxm_absent : bool = q >= 104 & q <= 115 & not(k < mxm_planes);
    assert(queue_present(q) == (want & not(mxm_absent)));
    if q <= 87 then {
      assert(mem_slice_present(q) == want);
      if want then present = present + 1;
      // A MEM slice reads and writes both rows at its quad's gaps, or at none when it is absent.
      let u = issuer_of(q).unit;
      foreach (r from 0 to 1) {
        let rw : row = if r == 0 then RowE else RowW;
        assert((read_gap(u, q, rw) != None()) == want);
        assert((write_gap(u, q, rw) != None()) == want)
      }
    }
  };
  assert(present == 2 * mem_slices_per_hemisphere);
  assert(mem_slices_per_hemisphere % 4 == 0);
  // The slices nearest the VXM always exist (the key is at least 4: one quad).
  assert(mem_slice_present(0) & mem_slice_present(44))
}

// words_per_slice is a power of two from 256 to 2^addr_bits (addr_bits 13 to 16) and the bank bit is its top address bit.
$[test]
function test_f3_sram_geometry() -> unit = {
  assert(addr_bits >= 13 & addr_bits <= 16);
  assert(words_per_slice >= 256 & words_per_slice <= 2 ^ addr_bits);
  assert(2 ^ (bank_bit + 1) == words_per_slice);
  let last = sram_addr_of(words_per_slice - 1);
  assert(sram_index(last) == Some(words_per_slice - 1));
  assert(sram_bank(last) == 0b1);
  assert(sram_bank(sram_addr_of((words_per_slice / 2) - 1)) == 0b0);
  assert(sram_bank(sram_addr_of(words_per_slice / 2)) == 0b1);
  // An address from words_per_slice on is outside the slice (the field has addr_bits bits).
  if words_per_slice < 2 ^ addr_bits then assert(sram_index(sram_addr_of(words_per_slice)) == None())
}

// The weight buffer and an installed matrix hold mxm.rows stream values; the load or install
// that would pass that is OTS-E-RES-002. Feeds 16 streams W:0..15 to the MXM input of the
// instruction issued at cycle c on queue q (read at c + d_skew).
function f3_feed16(c : nat, d_skew : nat) -> unit = {
  let t = c + d_skew;
  match read_gap(U_MXM_W, 0, RowW) {
    Some(g) => if g > 0 & t > 0 then foreach (s from 0 to 15) { let _ = srf_write(RowW, s, g - 1, t - 1, zero_vec); () },
    None() => assert(false, "no MXM read gap"),
  }
}

$[test]
function test_f3_mxm_lw_capacity() -> unit = {
  foreach (k from 0 to (if mxm_planes == 2 then 1 else 0)) {      // no f16 without a plane pair
    let t : mxm_wtype = if k == 0 then W_I8 else W_F16;
    reset();
    let n = (if k == 0 then mxm_rows else 2 * mxm_f16_rows) / 16;
    foreach (i from 0 to n) {
      let c = 100 + 4 * i;
      f3_feed16(c, entry_latency("LW", T_MXM_LW).d_skew);
      slice_test_set_issue_context(104, c);
      if i < n then {
        assert(execute_test(MXM_LW(t, Streamref(0b100000))) == Issued());
        unit_state_flush_test(c);
        assert(mxm[0].buf_rows == 16 * (i + 1))
      } else {
        assert(execute_test(MXM_LW(t, Streamref(0b100000))) == Diagnostic(RES_FULL));
        assert(mxm[0].buf_rows == (if k == 0 then mxm_rows else 2 * mxm_f16_rows))
      }
    }
  }
}

$[test]
function test_f3_mxm_iw_capacity() -> unit = {
  foreach (k from 0 to (if mxm_planes == 2 then 1 else 0)) {      // no f16 without a plane pair
    let t : mxm_wtype = if k == 0 then W_U8 else W_F16;
    reset();
    let n = (if k == 0 then mxm_rows else 2 * mxm_f16_rows) / 16;
    foreach (i from 0 to n) {
      let c = 100 + 4 * i;
      f3_feed16(c, entry_latency("IWS", T_MXM_IW).d_skew);
      slice_test_set_issue_context(104, c);
      if i < n then {
        assert(execute_test(MXM_IWS(t, Streamref(0b100000))) == Issued());
        unit_state_flush_test(c);
        // i8 and u8 rows are stream values; f16 rows take two each.
        assert(mxm[0].rows == (if t == W_F16 then 8 * (i + 1) else 16 * (i + 1)))
      } else {
        assert(execute_test(MXM_IWS(t, Streamref(0b100000))) == Diagnostic(RES_FULL))
      }
    }
  };
  // A plane holds mxm.rows i8 rows or mxm.f16_rows f16 rows.
  if mxm_planes == 2 then assert(mxm[0].rows == mxm_f16_rows)
}
