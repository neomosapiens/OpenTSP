// A symbolic stream port and staging sink. The proof driver inserts the
// production queue transitions at the marked line, binding these ports only.
register bus_word : vec
register bus_bad : bits(superlanes)
register bus_present : bool
register bus_cycle : nat
register bus_reads : nat
register bus_addresses_ok : bool
register slots_used : nat
register sink_full : bool
register sink_attempts : nat
register pair_ok : bool
register expect_first : vec
register expect_second : vec
register expect_first_bad : bits(superlanes)
register expect_second_bad : bits(superlanes)
register probe_diag : int

function probe_read(st : streamref, width : {1, 2, 4}, t : int) -> result(planes, diag) = {
  bus_reads = bus_reads + 1;
  bus_addresses_ok = bus_addresses_ok & width == 1 & stream_id(st) == 7 & t == bus_cycle;
  ecc_bad = bus_bad;
  if bus_present then Ok(one_plane(bus_word)) else Err(TIM_NO_VALUE)
}
function probe_stage(q : range(0, 144), t : nat, a : vec, b : vec, oa : ecc_ov, ob : ecc_ov) -> bool = {
  sink_attempts = sink_attempts + 1;
  pair_ok = q == 88 & t == bus_cycle & a == expect_first & b == expect_second &
    oa == ecc_poison_ov(expect_first_bad) & ob == ecc_poison_ov(expect_second_bad);
  not(sink_full)
}
function probe_diag_id(d : diag) -> diag = d
function probe_report(t : nat, q : int, d : diag) -> unit =
  probe_diag = if t != bus_cycle | q != 88 then 0 else match d { TIM_NO_VALUE => 1, IQ_STAGING => 2, ECC_UNC => 3, _ => 0 }

// INSERT QUEUE TRANSITIONS

$property
function p_fetch_independent_edges(t : bits(12), a : vec, b : vec, ma : bits(superlanes), mb : bits(superlanes),
                                  pa : bool, pb : bool, full_a : bool, full_b : bool) -> bool = {
  let c = unsigned(t);
  stream_fetches[88] = None();
  issue_queue = 88;
  issue_cycle = c;
  issue_element = 0;
  issue_lat = struct {d_skew = 0, d_func = 0};
  bus_reads = 0; bus_addresses_ok = true; bus_cycle = c;
  bus_word = a; bus_bad = ma; bus_present = pa; slots_used = if full_a then iq_staging_depth else 0;
  sink_attempts = 0; pair_ok = false; probe_diag = -1; sink_full = full_b;
  expect_first = a; expect_second = b; expect_first_bad = ma; expect_second_bad = mb;
  let started = probe_start(Streamref(0b000111));
  let start_ok : bool = match started {
    Issued() => pa & not(full_a),
    Diagnostic(TIM_NO_VALUE) => not(pa),
    Diagnostic(IQ_STAGING) => pa & full_a,
    _ => false,
  };
  let first_ok = start_ok & bus_reads == 1 & sink_attempts == 0 & probe_diag == -1;
  bus_cycle = c + 1; bus_word = b; bus_bad = mb; bus_present = pb;
  probe_service(88, c + 1);
  first_ok & bus_addresses_ok & (if not(pa) | full_a then bus_reads == 1 & sink_attempts == 0 & probe_diag == -1
    else bus_reads == 2 & (if not(pb) then sink_attempts == 0 & probe_diag == 1
      else sink_attempts == 1 & pair_ok & probe_diag == (if full_b then 2 else if mb != sail_zeros(superlanes) then 3 else -1)))
}

// Two addressed IQ units can refer to the same protected word or straddle
// words. The oracle represents each distinct word by its SECDED verdict.
register poison_first : bool
register poison_second : bool
register same_word : bool
register clean_decodes : nat
function probe_unit_word(_q : range(0, 144), i : nat) -> int =
  if i == 1 & not(same_word) then 71 else 70
function probe_word_unc(_q : range(0, 144), w : int) -> bool =
  if w == 70 then poison_first else poison_second
function probe_view(_q : range(0, 144), _i : nat) -> bits(32) = 0x52000001
function probe_decode(_hi : bits(32), _lo : bits(32)) -> result(instruction, diag) = {
  clean_decodes = clean_decodes + 1; Ok(C2C_HOSTNOTIFY(0x1))
}

// INSERT DECODE TRANSITIONS

$property
function p_poison_excludes_normal_decode(a : bool, b : bool, same : bool, wide : bool) -> bool = {
  poison_first = a; poison_second = b; same_word = same; clean_decodes = 0; iq_ecc_live = true;
  let bad = a | (wide & not(same) & b);
  match probe_decode_at(88, 0, if wide then 2 else 1) {
    Err(ECC_TXT) => bad & clean_decodes == 0,
    Ok(C2C_HOSTNOTIFY(_)) => not(bad) & clean_decodes == 1,
    _ => false,
  }
}

// For these three register writes the doorbell port must be unreachable.
// The transport records which vectors were delivered, independently of the cause.
register delivered_vectors : bits(16)
register wrong_port : bool
function unexpected_doorbell(_i : range(0, 63), _c : nat) -> unit = wrong_port = true
function probe_irq_delivery(_c : nat, n : range(0, 15)) -> unit = delivered_vectors[n] = bitone

// INSERT CSR TRANSITION

$property
function p_irq_mask_ack(old : bits(16), newmask : bits(16), cause : bits(16), ack : bits(16), status : bits(10)) -> bool = {
  delivered_vectors = 0x0000; wrong_port = false;
  ras_first_valid = status != 0b0000000000;
  ras_err_status = sail_zero_extend(status, 64);
  csr_irq_mask = sail_zero_extend(old, 64);
  csr_irq_cause = sail_zero_extend(cause, 64);
  probe_csr_write(49, sail_zero_extend(newmask, 64), 0);
  let edges = and_vec(cause, and_vec(newmask, not_vec(old)));
  probe_csr_write(48, sail_zero_extend(ack, 64), 1);
  probe_csr_write(16, sail_zero_extend(ack, 64), 1);
  let remaining = and_vec(status, not_vec(ack[9 .. 0]));
  delivered_vectors == edges & not(wrong_port) &
    csr_irq_mask == sail_zero_extend(newmask, 64) &
    csr_irq_cause == sail_zero_extend(and_vec(cause, not_vec(ack)), 64) &
    ras_err_status == sail_zero_extend(remaining, 64) &
    (ras_first_valid == (remaining != 0b0000000000))
}

// Distinct queue and doorbell footprints. A HostWait has no callback or
// ordered output of its own. The order test includes both a pre-rung and an
// absent doorbell on either side and checks the resulting waiter identities.
$property
function p_disjoint_wait_order(a : bool, b : bool, rev : bool, ts : bits(16)) -> bool = {
  let t = unsigned(ts);
  hw_since[3] = None(); hw_since[4] = None();
  hw_release[3] = None(); hw_release[4] = None();
  hw_db[3] = 12; hw_db[4] = 13;
  csr_doorbell[5] = a; csr_doorbell[6] = b;
  if rev then { host_wait(4, t, 6); host_wait(3, t, 5) }
  else { host_wait(3, t, 5); host_wait(4, t, 6) };
  not(csr_doorbell[5]) & not(csr_doorbell[6]) &
    hw_since[3] == (if a then None() else Some(t)) &
    hw_since[4] == (if b then None() else Some(t)) &
    hw_db[3] == (if a then 12 else 5) & hw_db[4] == (if b then 13 else 6) &
    hw_release[3] == None() & hw_release[4] == None()
}

// INSERT RETIRE TRANSITION

// The only scheduled change to these two windows is their common release.
// A literal tick at *any* interior offset preserves the induction hypothesis.
// The endpoint must clear both windows, including for an interval of one.
$property
function p_release_idle_interval(ts : bits(16), gap : bits(8), offset : bits(8)) -> bool = {
  let t = unsigned(ts);
  let end_cycle = t + 1 + unsigned(gap);
  let i = unsigned(offset);
  if i > unsigned(gap) then true else {
    parked_since[3] = Some(t); release_at[3] = Some(end_cycle);
    hw_since[3] = Some(t); hw_release[3] = Some(end_cycle);
    probe_retire(t + i);
    let retained = parked_since[3] == Some(t) & release_at[3] == Some(end_cycle) &
                   hw_since[3] == Some(t) & hw_release[3] == Some(end_cycle);
    probe_retire(end_cycle);
    retained & parked_since[3] == None() & release_at[3] == None() &
      hw_since[3] == None() & hw_release[3] == None()
  }
}
