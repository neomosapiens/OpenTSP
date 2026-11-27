// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// The execute clause of IfetchM (slices/cu/cu_insts.sail). It needs the MEM slice's
// segment (slices/mem/mem_insts.sail), so this file loads after the MEM module.

// Two physical read cycles: retain the first word and its integrity state until
// the second read completes. The scheduler reserves the R port at dispatch.
struct local_fetch = { ready : nat, addr : nat, first : vec, ov : ecc_ov }
register local_fetch_busy : vector(88, nat) = vector_init(0)
register local_fetches : vector(88, option(local_fetch)) = vector_init(None())

// IfetchM a: words a and a+1 of the issuing slice, inside its segment.
function clause execute IFETCHM(SramAddr(a)) = {
  let q = issue_queue;
  let e = unsigned(a) + 1;
  if q > 87 then Diagnostic(QUE_UNIT)
  else if e >= mem_segment[q] | e >= words_per_slice then Diagnostic(OPR_RANGE)
  else if exec_cycle() < local_fetch_busy[q] then Diagnostic(QUE_BUSY)
  else if fill_len(fill_fifo[q]) >= iq_staging_depth then Diagnostic(IQ_STAGING)
  else {
    local_fetch_busy[q] = exec_cycle() + 2;
    local_fetches[q] = Some(struct { ready = exec_cycle() + 1, addr = e,
      first = read_sram_word(issue_unit.slice, unsigned(a)),
      ov = read_sram_ecc(issue_unit.slice, unsigned(a)) });
    Issued()
  }
}

function local_fetch_complete(c : nat) -> unit = {
  foreach (q from 0 to 87) {
    match local_fetches[q] {
      Some(f) => if f.ready <= c then {
        local_fetches[q] = None();
        let ms = issuer_of(q).slice;
        if not(iq_fetch_ov(q, c, f.first, read_sram_word(ms, f.addr),
                          f.ov, read_sram_ecc(ms, f.addr))) then
          diag_report(c, q, IQ_STAGING)
      },
      None() => (),
    }
  }
}
