// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Unit tests for the pure helpers of the cycle driver and the stream register file: each one is a function of explicit values
// (a slot, an AGT, a hold view, tags, a scrub verdict), tested here without the machine state, and the wrappers that apply them to
// the registers are checked to agree with them. The behaviour of the whole machine is in test/directed and the golden vectors.

private function ph_v(b : bits(8)) -> vec = replicate_bits(b, vlanes)

private let ph_empty : srf_write = struct { valid = false, diag = 0, wgap = 0, value = sail_zeros(8 * vlanes), ov = ecc_ov_zero }
private let ph_slot : srf_slot = vector_init(ph_empty)

// ---------------------------------------------------------------------------------------
// SRF slot

// A write replaces its gap only; the reader takes the nearest upstream writer of its diagonal, strictly below its own gap.
$[test]
function test_slot_write_read() -> unit = {
  let s0 = ph_slot;
  assert(not(srf_slot_collides(s0, 3, -10)));
  let s1 = srf_slot_write(s0, 3, -10, ph_v(0x01), ecc_ov_zero);
  assert(srf_slot_collides(s1, 3, -10));
  assert(not(srf_slot_collides(s1, 3, -11)));
  assert(not(srf_slot_collides(s1, 4, -10)));
  let s2 = srf_slot_write(s1, 7, -10, ph_v(0x02), ecc_ov_zero);
  let (f4, x4) = srf_slot_read(s2, 5, -10);
  assert(f4 & x4 == ph_v(0x01));
  let (f9, x9) = srf_slot_read(s2, 9, -10);
  assert(f9 & x9 == ph_v(0x02));
  // a reader at the gap of a write does not see it, nor one on another diagonal
  let (f3, _) = srf_slot_read(s2, 3, -10);
  assert(not(f3));
  let (f7, x7) = srf_slot_read(s2, 7, -10);
  assert(f7 & x7 == ph_v(0x01));
  let (fd, _) = srf_slot_read(s2, 9, -9);
  assert(not(fd));
  assert(srf_slot_best(s2, 9, -10) == 7);
  assert(srf_slot_best(s2, 7, -10) == 3);
  assert(srf_slot_best(s2, 3, -10) == -1);
  // the write is pure: the slot it was applied to is unchanged
  assert(not(srf_slot_collides(s0, 3, -10)))
}

// A chained read sees only the write at the reader's own gap.
$[test]
function test_slot_chain() -> unit = {
  let s1 = srf_slot_write(ph_slot, 12, -3, ph_v(0x33), ecc_ov_zero);
  let (f, x, _) = srf_slot_chain_ov(s1, 12, -3);
  assert(f & x == ph_v(0x33));
  let (g, _, _) = srf_slot_chain_ov(s1, 13, -3);
  assert(not(g));
  let (h, _, _) = srf_slot_chain_ov(s1, 12, -4);
  assert(not(h))
}

// A bit flip changes the value the reader sees and nothing else; a reader that sees no value flips nothing.
$[test]
function test_slot_flip() -> unit = {
  let s1 = srf_slot_write(ph_slot, 4, -2, ph_v(0x00), ecc_ov_zero);
  let s2 = srf_slot_flip(s1, 6, -2, 0, 5);
  let (f, x) = srf_slot_read(s2, 6, -2);
  assert(f & x != ph_v(0x00));
  assert(x == (ph_v(0x00) ^ sail_shiftleft(get_slice_int(8 * vlanes, 1, 0), 5)));
  let s3 = srf_slot_flip(s1, 4, -2, 0, 5);          // the reader at the writer's gap sees no value
  let (_, y) = srf_slot_read(s3, 6, -2);
  assert(y == ph_v(0x00));
  let s4 = srf_slot_flip(s1, 6, -2, 0, 130);        // a check bit flips the overlay, not the data
  let (_, _, ov4) = srf_slot_read_ov(s4, 6, -2);
  assert(ov4 != ecc_ov_zero)
}

// The register file operations agree with the slot functions.
$[test]
function test_srf_wrappers_agree_with_slots() -> unit = {
  srf_reset();
  assert(not(srf_write(RowE, 5, 8, 15, ph_v(0x0a))));
  assert(srf_write(RowE, 5, 8, 15, ph_v(0x0b)));
  assert(not(srf_write(RowE, 5, 13, 20, ph_v(0x0c))));
  let (k, x) = srf_read(RowE, 5, 11, 18);
  assert(k == SRF_FOUND & x == ph_v(0x0b));
  let (k2, x2) = srf_read(RowE, 5, 16, 23);
  assert(k2 == SRF_FOUND & x2 == ph_v(0x0c));
  let (k3, _) = srf_read(RowE, 5, 8, 15);
  assert(k3 != SRF_FOUND);
  let (kc, xc) = srf_read_chain(RowE, 5, 13, 21);
  assert(kc == SRF_FOUND & xc == ph_v(0x0c));
  let (kd, _) = srf_read_chain(RowE, 5, 13, 22);
  assert(kd == SRF_NONE)
}

// Live writes: none upstream of the reader gives the direction diagnosis; one upstream does not.
$[test]
function test_stream_only_downstream() -> unit = {
  srf_reset();
  let _ = srf_write(RowW, 9, 21, 25, ph_v(0x05));
  let _ = srf_write(RowW, 9, 30, 25, ph_v(0x06));
  let (k, _) = srf_read(RowW, 9, 18, 26);
  assert(k == SRF_UPSTREAM);
  let (k2, _) = srf_read(RowW, 9, 25, 40);   // a different diagonal, a write upstream of the reader
  assert(k2 == SRF_NONE)
}

// The report merge: S[0..2] OR together, a higher priority takes the auxiliary streams, a lower one adds nothing, the parity is the
// even parity of S[0..2].
$[test]
function test_rep_merge() -> unit = {
  let e0 : srf_rep = struct { used = true, rw = false, diag = 0, wgap = 5, sp = vector_init(sail_zeros(superlanes)), pr = vector_init(0) };
  var a : vector(30, bits(superlanes)) = vector_init(sail_zeros(superlanes));
  var m0 = a[0]; m0[0] = bitone; a[0] = m0;
  var m5 = a[5]; m5[0] = bitone; a[5] = m5;
  var xs : bits(superlanes) = sail_zeros(superlanes);
  xs[0] = bitone;
  let e1 = srf_rep_merge(e0, 2, xs, a);
  assert(e1.sp[0][0] == bitone & e1.sp[5][0] == bitone & e1.pr[0] == 2);
  assert(e1.sp[3][0] == bitone);                    // one error bit set: odd parity
  var b : vector(30, bits(superlanes)) = vector_init(sail_zeros(superlanes));
  var n1 = b[1]; n1[0] = bitone; b[1] = n1;
  var n6 = b[6]; n6[0] = bitone; b[6] = n6;
  let e2 = srf_rep_merge(e1, 1, xs, b);             // lower priority: S[1] merges, stream 6 does not
  assert(e2.sp[1][0] == bitone & e2.sp[6][0] == bitzero & e2.sp[5][0] == bitone & e2.pr[0] == 2);
  assert(e2.sp[3][0] == bitzero);                   // two error bits: even parity
  let e3 = srf_rep_merge(e2, 3, xs, b);             // higher priority: the auxiliary streams are replaced
  assert(e3.sp[6][0] == bitone & e3.sp[5][0] == bitzero & e3.pr[0] == 3)
}

// ---------------------------------------------------------------------------------------
// Hold decision

private let ph_idle : path_view = struct { pbusy = false, ptail = false, pstate = AGT_IDLE }
private let ph_free : hold_view = struct { vparked = false, vhost = false, vbusy = false, vrepeat = false, vpoll = false, vp = path_views(ph_idle, ph_idle) }

$[test]
function test_hold_decide_queue_flags() -> unit = {
  assert(view_free(ph_free));
  assert(not(hold_decide(ph_free, false, false, None(), false)));
  assert(not(hold_decide(ph_free, false, false, Some(0), false)));
  // a parked queue holds every instruction except its own Sync or Notify
  let pk = { ph_free with vparked = true };
  assert(hold_decide(pk, false, false, None(), false));
  assert(not(hold_decide(pk, true, false, None(), false)));
  assert(not(view_free(pk)));
  // a host park holds the barrier instructions too
  assert(hold_decide({ ph_free with vhost = true }, true, false, None(), false));
  assert(hold_decide({ ph_free with vbusy = true }, false, false, Some(1), true));
  assert(hold_decide({ ph_free with vrepeat = true }, false, false, Some(0), false));
  assert(hold_decide({ ph_free with vpoll = true }, true, false, None(), false))
}

$[test]
function test_hold_decide_paths() -> unit = {
  let busy1 = { ph_free with vp = path_views(ph_idle, { ph_idle with pbusy = true }) };
  assert(not(hold_decide(busy1, false, false, Some(0), false)));
  assert(hold_decide(busy1, false, false, Some(1), false));
  // a whole-queue instruction needs both paths free; an Ifetch on a MEM queue (whole_free) only its own path
  assert(hold_decide(busy1, false, false, None(), false));
  assert(not(hold_decide(busy1, false, true, None(), false)));
  let tail0 = { ph_free with vp = path_views({ ph_idle with ptail = true }, ph_idle) };
  assert(hold_decide(tail0, false, false, Some(0), false));
  assert(not(hold_decide(tail0, false, false, Some(1), false)));
  assert(hold_decide(tail0, false, false, None(), false));
  // a loop runs: it holds its path in both states; a configuration open on a path holds it against all but its records
  let run0 = { ph_free with vp = path_views({ ph_idle with pstate = AGT_RUN }, ph_idle) };
  assert(hold_decide(run0, false, false, Some(0), true));
  assert(hold_decide(run0, false, false, None(), false));
  let cfg1 = { ph_free with vp = path_views(ph_idle, { ph_idle with pstate = AGT_CONFIG }) };
  assert(hold_decide(cfg1, false, false, Some(1), false));
  assert(not(hold_decide(cfg1, false, false, Some(1), true)));
  assert(hold_decide(cfg1, false, false, None(), false));
  assert(not(view_free(cfg1)))
}

private function ph_nat_is(x : option(nat), y : nat) -> bool = match x { Some(t) => t == y, None() => false }
private function ph_nat_none(x : option(nat)) -> bool = match x { None() => true, Some(_) => false }
private function ph_rep_none(x : option(repeat_state)) -> bool = match x { None() => true, Some(_) => false }

private function ph_state(n : int) -> agt_state = if n == 0 then AGT_IDLE else if n == 1 then AGT_CONFIG else AGT_RUN
private function ph_bit(n : int, k : int) -> bool = emod_int(ediv_int(n, pow2i(k)), 2) == 1
private function ph_pview(n : int) -> path_view = struct { pbusy = ph_bit(n, 0), ptail = ph_bit(n, 1), pstate = ph_state(ediv_int(n, 4)) }

// Over every combination of the queue flags and of the two paths' states: a free view holds nothing, and a view with no flag set
// holds a whole-queue instruction exactly when a path is busy, in a tail or has an AGT.
$[test]
function test_hold_decide_exhaustive() -> unit = {
  foreach (f from 0 to 31) {
    foreach (a from 0 to 11) {
      foreach (b from 0 to 11) {
        let hv : hold_view = struct { vparked = ph_bit(f, 0), vhost = ph_bit(f, 1), vbusy = ph_bit(f, 2), vrepeat = ph_bit(f, 3), vpoll = ph_bit(f, 4),
                                     vp = path_views(ph_pview(a), ph_pview(b)) };
        if view_free(hv) then {
          assert(not(hold_decide(hv, false, false, None(), false)));
          assert(not(hold_decide(hv, false, false, Some(0), false)));
          assert(not(hold_decide(hv, false, false, Some(1), true)))
        } else if f == 0 then {
          assert(hold_decide(hv, false, false, None(), false) == view_paths_busy(hv));
          assert(not(hold_decide(hv, false, true, None(), false)))
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------------------
// Port reservation, AGT, Repeat

$[test]
function test_port_reservation() -> unit = {
  assert(path_hold_until(10) == 11);
  assert(fetchm_hold_until(10) == 12);
  let i = IFETCHM(sram_addr_of(5));
  match fetchm_bank_note(i, 10, 100) {
    Some(n) => assert(n.bn_at == 11 & n.bn_addr == 6),
    None() => assert(false),
  };
  // the word after the last of the segment (or of the slice) is not read
  assert(match fetchm_bank_note(i, 10, 6) { None() => true, _ => false });
  assert(match fetchm_bank_note(i, 10, 0) { None() => true, _ => false });
  assert(match fetchm_bank_note(IFETCH(Streamref(0b000000)), 10, 100) { None() => true, _ => false })
}

$[test]
function test_agt_configure_pure() -> unit = {
  let a0 = agt_opened(MEM_DIRECT(MEM_READ, sram_addr_of(0), Streamref(0b000000), 0b0), 5);
  assert(a0.state == AGT_CONFIG & a0.t0 == 5 & a0.last == 5 & a0.pos == -1);
  let (a1, ok1) = agt_configure_pure(a0, AGT_COUNTDOWN, 0x0000);
  assert(ok1 & a1.levels[0].countdown == 1 & a1.pos == 0 & a1.level == 0);   // below 1 counts as 1
  let (a2, ok2) = agt_configure_pure(a1, AGT_STEP, 0xfffe);
  assert(ok2 & a2.levels[0].step == -2 & a2.pos == 1);
  let (a3, ok3) = agt_configure_pure(a2, AGT_COUNTDOWN, 0x0004);              // does not advance in order: the next level
  assert(ok3 & a3.level == 1 & a3.pos == 0 & a3.levels[1].countdown == 4 & a3.levels[0].step == -2);
  let (a4, _) = agt_configure_pure(a3, AGT_COUNTDOWN, 0x0001);
  let (a5, _) = agt_configure_pure(a4, AGT_COUNTDOWN, 0x0001);
  assert(a5.level == 3);
  let (a6, ok6) = agt_configure_pure(a5, AGT_COUNTDOWN, 0x0001);              // a fifth level
  assert(not(ok6) & a6.level == 3 & a6.pos == a5.pos);
  // the wrapper stores the result only when it is accepted
  reset();
  agt_set(7, 0, a5);
  assert(not(agt_configure_test(7, AGT_COUNTDOWN, 0x0001)));
  assert(agt_get(7, 0).level == 3);
  agt_set(7, 0, a0);
  assert(agt_configure_test(7, AGT_ITERATIONS, 0x0003) & agt_get(7, 0).levels[0].iters == 3);
  reset()
}

$[test]
function test_agt_refresh_pure() -> unit = {
  let l0 : agt_level = struct { countdown = 3, step = 0, sstep = 0, iters = 2, idx = 1 };
  let run = { agt_idle with state = AGT_RUN, levels = [agt_idle.levels with 0 = l0], level = 0, pos = 0, last = 40 };
  // no level can advance: idle, held through the trailing waits
  let (a, tail) = agt_refresh_pure(run);
  assert(a.state == AGT_IDLE & ph_nat_is(tail, 44));
  // a level that can advance: the next reference after the countdown
  let l1 = { l0 with idx = 0 };
  let (b, tb) = agt_refresh_pure({ run with levels = [run.levels with 0 = l1] });
  assert(b.state == AGT_RUN & ph_nat_none(tb) & b.next == 44);
  // an AGT that is not running is unchanged
  let (c, tc) = agt_refresh_pure({ run with state = AGT_CONFIG });
  assert(c.state == AGT_CONFIG & ph_nat_none(tc));
  // the end of a configuration runs the loop
  let (d, _) = agt_finish_config_pure({ run with state = AGT_CONFIG, levels = [run.levels with 0 = l1] });
  assert(d.state == AGT_RUN & d.next == 44)
}

$[test]
function test_agt_due_and_advanced() -> unit = {
  let l0 : agt_level = struct { countdown = 2, step = 8, sstep = 1, iters = 3, idx = 0 };
  let run = { agt_idle with state = AGT_RUN, levels = [agt_idle.levels with 0 = l0], level = 0, pos = 0, last = 10, offset = 0, soffset = 0 };
  assert(match agt_due_pure(run, 12, false) { None() => true, _ => false });       // the reference after 1 + countdown = 3 cycles
  assert(match agt_due_pure(run, 13, true) { None() => true, _ => false });        // parked
  match agt_due_pure(run, 13, false) {
    Some((ls, d, s, ss)) => assert(d == 3 & s == 8 & ss == 1 & ls[0].idx == 1),
    None() => assert(false),
  };
  assert(match agt_due_pure(agt_idle, 100, false) { None() => true, _ => false });
  let a = agt_after_reference(run, [run.levels with 0 = { l0 with idx = 1 }], 8, 1, 13);
  assert(a.offset == 8 & a.soffset == 1 & a.last == 13 & a.levels[0].idx == 1)
}

$[test]
function test_cfg_path_pick() -> unit = {
  assert(cfg_path_pick(true, false, 0, 0) == 0);
  assert(cfg_path_pick(false, true, 0, 0) == 1);
  assert(cfg_path_pick(false, false, 0, 0) == 0);
  assert(cfg_path_pick(true, true, 0, 0) == 0);          // a tie: R
  assert(cfg_path_pick(true, true, 1, 0) == 1);          // W has fewer records so far
  assert(cfg_path_pick(true, true, 1, 1) == 0)
}

$[test]
function test_repeat_plan() -> unit = {
  let i = NOP(0x0001);
  // issued at 10, d = 2: iterations at 13, 16, ...; a Repeat dispatched at 13 issues now
  let (now, rest) = repeat_plan(i, 3, 2, 10, 13);
  assert(now);
  match rest { Some(r) => assert(r.left == 2 & r.gap == 3 & r.next == 16), None() => assert(false) };
  // dispatched at 11: the first iteration is not before the Repeat's own cycle, so at 13
  let (now2, rest2) = repeat_plan(i, 3, 2, 10, 11);
  assert(not(now2));
  match rest2 { Some(r) => assert(r.left == 3 & r.next == 13), None() => assert(false) };
  let (now3, rest3) = repeat_plan(i, 1, 0, 10, 20);
  assert(now3 & ph_rep_none(rest3));
  // after an iteration
  match repeat_after(struct { instr = i, left = 3, gap = 3, next = 13 }, 13) {
    Some(r) => assert(r.left == 2 & r.next == 16),
    None() => assert(false),
  };
  assert(match repeat_after(struct { instr = i, left = 1, gap = 3, next = 13 }, 13) { None() => true, _ => false })
}

// ---------------------------------------------------------------------------------------
// Bank rule

private let ph_t0 : ref_tag = struct { thread = 0b0, bank = BankOf(0b0) }
private let ph_t1 : ref_tag = struct { thread = 0b0, bank = BankOf(0b1) }
private let ph_ta : ref_tag = struct { thread = 0b0, bank = BankAny() }
private let ph_u0 : ref_tag = struct { thread = 0b1, bank = BankOf(0b0) }
private let ph_none4 : vector(4, option(ref_tag)) = vector_init(None())
private let ph_none2 : vector(2, option(ref_tag)) = vector_init(None())
private let ph_f4 : vector(4, bool) = vector_init(false)

// A new R reference (index 0) and a new W reference (index 2) of one thread conflict in the same bank, not in different banks, and not
// across threads; an indirect reference (any bank) conflicts with every bank.
$[test]
function test_bank_rule_new_references() -> unit = {
  var go = ph_f4; go[0] = true; go[2] = true;
  var rt = ph_none4; rt[0] = Some(ph_t0); rt[2] = Some(ph_t0);
  let u1 = bank_rule(rt, go, ph_none2, false, BankNone());
  assert(u1.vbank[0] & u1.vbank[2] & not(u1.vbank[1]) & not(u1.vbank[3]));
  rt[2] = Some(ph_t1);
  let w = bank_rule(rt, go, ph_none2, false, BankNone());
  assert(not(w.vbank[0]) & not(w.vbank[2]));
  rt[2] = Some(ph_u0);
  assert(not(bank_rule(rt, go, ph_none2, false, BankNone()).vbank[0]));
  rt[2] = Some(ph_ta);
  assert(bank_rule(rt, go, ph_none2, false, BankNone()).vbank[0]);
  // a reference that failed its checks neither loses nor makes the other lose
  var go2 = ph_f4; go2[0] = true;
  rt[2] = Some(ph_t0);
  let x = bank_rule(rt, go2, ph_none2, false, BankNone());
  assert(not(x.vbank[0]) & not(x.vbank[2]))
}

// An AGT reference on one path against a new reference of the same thread on the other path, and against the other path's AGT reference.
$[test]
function test_bank_rule_loops() -> unit = {
  var go = ph_f4; go[2] = true;
  var rt = ph_none4; rt[2] = Some(ph_t0);
  var lp = ph_none2; lp[0] = Some(ph_t0);
  let u1 = bank_rule(rt, go, lp, false, BankNone());
  assert(u1.vbank[2] & u1.vlblock[0] & u1.vlsrc[0] == 2 & not(u1.vlblock[1]));
  var lp2 = ph_none2; lp2[0] = Some(ph_t0); lp2[1] = Some(ph_t0);
  let w = bank_rule(ph_none4, ph_f4, lp2, false, BankNone());
  assert(w.vlblock[0] & w.vlblock[1] & w.vlsrc[0] == 4 & w.vlsrc[1] == 4);
  var lp3 = ph_none2; lp3[0] = Some(ph_t0); lp3[1] = Some(ph_t1);
  let x = bank_rule(ph_none4, ph_f4, lp3, false, BankNone());
  assert(not(x.vlblock[0]) & not(x.vlblock[1]) & x.vlsrc[0] == -1)
}

// An IfetchM bank: a new W reference or the W path's AGT reference in the same bank loses; the R path's AGT reference does not.
$[test]
function test_bank_rule_ifetchm() -> unit = {
  var go = ph_f4; go[2] = true;
  var rt = ph_none4; rt[2] = Some(ph_t0);
  let u1 = bank_rule(rt, go, ph_none2, true, BankOf(0b0));
  assert(u1.vfm_clash & u1.vbank[2] & not(u1.vfm_loop));
  let w = bank_rule(rt, go, ph_none2, true, BankOf(0b1));
  assert(not(w.vfm_clash) & not(w.vbank[2]));
  let x = bank_rule(rt, go, ph_none2, false, BankOf(0b0));
  assert(not(x.vfm_clash) & not(x.vbank[2]));
  var lp = ph_none2; lp[1] = Some(ph_t0);
  let y = bank_rule(ph_none4, ph_f4, lp, true, BankOf(0b0));
  assert(y.vfm_clash & y.vfm_loop & y.vlblock[1] & not(y.vlblock[0]));
  var lp0 = ph_none2; lp0[0] = Some(ph_t0);
  let z = bank_rule(ph_none4, ph_f4, lp0, true, BankOf(0b0));
  assert(not(z.vfm_clash) & not(z.vlblock[0]))
}

// ---------------------------------------------------------------------------------------
// Scrub arbitration

$[test]
function test_scrub_arbitrate() -> unit = {
  assert(scrub_arbitrate(false, false, false, false) == SCRUB_SKIP);
  assert(scrub_arbitrate(true, true, true, true) == SCRUB_SKIP);
  assert(scrub_arbitrate(true, false, false, false) == SCRUB_PATROL);
  assert(scrub_arbitrate(true, false, true, true) == SCRUB_WRITEBACK);
  assert(scrub_arbitrate(true, false, true, false) == SCRUB_DROP)
}

// The patrol slot is taken by a path held through the Read's dispatch cycle, or by the queue's last issue at that cycle.
$[test]
function test_scrub_slot_taken() -> unit = {
  assert(not(scrub_slot_taken(2, 3, 100, 100, Some(0))));          // before the first dispatch cycle
  assert(scrub_slot_taken(13, 3, 11, 0, None()));                  // d = 10 < 11: path 0 held through it
  assert(not(scrub_slot_taken(14, 3, 11, 0, None())));
  assert(scrub_slot_taken(13, 3, 0, 11, None()));
  assert(scrub_slot_taken(13, 3, 0, 0, Some(10)));
  assert(not(scrub_slot_taken(13, 3, 0, 0, Some(9))));
  assert(not(scrub_slot_taken(13, 3, 0, 0, None())))
}

// The scrub cycles due: one at next, then one every period, while <= upto.
$[test]
function test_scrub_due_count() -> unit = {
  assert(scrub_due_count(0, 4, -1) == 0);
  assert(scrub_due_count(0, 4, 0) == 1);
  assert(scrub_due_count(0, 4, 3) == 1);
  assert(scrub_due_count(0, 4, 4) == 2);
  assert(scrub_due_count(8, 4, 7) == 0);
  assert(scrub_due_count(8, 4, 8) == 1);
  assert(scrub_due_count(8, 4, 20) == 4);
  assert(scrub_due_count(8, 0, 20) == 0)
}

// ---------------------------------------------------------------------------------------
// Commit arbitration, the tick helpers

private function ph_commit(q : range(0, 144), b : range(0, 3), x : bits(8)) -> commit =
  struct { r = RowE, id = 4, g = 20, value = ph_v(x), ov = ecc_ov_zero, q = q, b = b }

$[test]
function test_commit_wins() -> unit = {
  // one writer: it wins with its own value
  let a = ph_commit(2, 0, 0x01);
  let (c1, w1, v1, _) = commit_wins(a, 0, [|a|]);
  assert(not(c1) & w1 & v1 == ph_v(0x01));
  // two writers on one target that are not an OR pair collide; the lowest (queue, plane) writer keeps its value
  let b = ph_commit(5, 0, 0x02);
  let (c2, w2, v2, _) = commit_wins(a, 0, [|a, b|]);
  assert(c2 & w2 & v2 == ph_v(0x01));
  let (c3, w3, v3, _) = commit_wins(b, 1, [|a, b|]);
  assert(c3 & not(w3) & v3 == ph_v(0x02));
  // the writes of one (queue, plane) are one writer: their OR is stored by the first of them
  let a2 = ph_commit(2, 0, 0x10);
  let (c4, w4, v4, _) = commit_wins(a, 0, [|a, a2|]);
  assert(not(c4) & w4 & v4 == ph_v(0x11));
  let (c5, w5, _, _) = commit_wins(a2, 1, [|a, a2|]);
  assert(not(c5) & not(w5));
  // writes to another gap are another target
  let d = { ph_commit(9, 0, 0x04) with g = 21 };
  let (c6, w6, v6, _) = commit_wins(a, 0, [|a, d|]);
  assert(not(c6) & w6 & v6 == ph_v(0x01));
  // a SEL queue and a TSP queue of one hemisphere merge: no collision, the lower writer stores the OR
  let tsp = ph_commit(120, 0, 0x01);
  let sel = ph_commit(125, 0, 0x02);
  let (c7, w7, v7, _) = commit_wins(tsp, 0, [|tsp, sel|]);
  assert(not(c7) & w7 & v7 == ph_v(0x03));
  let (c8, w8, _, _) = commit_wins(sel, 1, [|tsp, sel|]);
  assert(not(c8) & not(w8));
  // two TSP queues are not an OR pair
  let tsp2 = ph_commit(121, 0, 0x04);
  let (c9, _, _, _) = commit_wins(tsp, 0, [|tsp, tsp2|]);
  assert(c9);
  // three writers collide
  let (c10, _, _, _) = commit_wins(tsp, 0, [|tsp, sel, ph_commit(3, 0, 0x08)|]);
  assert(c10)
}

// The commits of a cycle reach the register file through commit_due; a collision keeps the lowest writer's value.
$[test]
function test_commit_due_applies() -> unit = {
  reset();
  pending[7] = [|ph_commit(5, 0, 0x02), ph_commit(2, 0, 0x01)|];
  commit_due(7);
  let (k, x) = srf_read(RowE, 4, 21, 8);
  assert(k == SRF_FOUND & x == ph_v(0x01));
  assert(match pending[7] { [||] => true, _ => false });
  reset()
}

$[test]
function test_tick_helpers() -> unit = {
  assert(wait_reason(true, true, true, true) == "wait-host");
  assert(wait_reason(false, true, true, true) == "wait-barrier");
  assert(wait_reason(false, false, true, true) == "wait-poll");
  assert(wait_reason(false, false, false, true) == "wait-repeat");
  assert(wait_reason(false, false, false, false) == "wait-resource");
  assert(config_ends(AGT_CONFIG, false, true));
  assert(config_ends(AGT_CONFIG, true, false));
  assert(not(config_ends(AGT_CONFIG, true, true)));
  assert(not(config_ends(AGT_RUN, false, false)));
  assert(not(config_ends(AGT_IDLE, false, false)));
  assert(poll_epoch(10, 20, 10, false));
  assert(not(poll_epoch(20, 20, 10, false)));      // not before the cycle after the dispatch
  assert(not(poll_epoch(10, 21, 10, false)));
  assert(not(poll_epoch(10, 20, 10, true)))
}
