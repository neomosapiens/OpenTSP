//! Dispatch: the control channel, a whole-queue instruction, and a parcel on a data queue
//! (`postlude/tick.sail`: `control`, `dispatch_check`, `dispatch_decoded`, `dispatch_body`,
//! `dispatch`), and the MEM slots of a queue (`dispatch_slots`: the fetch slot, the configuration
//! record, the references with the bank rule, and the AGT references due this cycle).
use super::agt::{agt_apply_advance, agt_config, agt_configure, agt_due_lean, agt_nlevels, Agt, AgtDue, AgtState};
use super::classify::CycleRecs;
use super::repeat::{clears_repeat, repeatable, PollState};
use crate::layout::queue_present;
use crate::machine::RunError;
use crate::observe::Observer;
use crate::slices::mem::{banks_meet, is_loopable, tags_clash, BankUse, ClashRef, RefTag};
use crate::Machine;
use ots_isa::{placement, Code as Diag, EntryId, Insn};
use tsp_numerics::fault::model_fault;

impl<const L: usize> Machine<L> {
    /// Does queue `q`'s unit issue `i`?
    // model: sys/insts_begin.sail issued_by (the catalogue's `units`)
    pub fn issued_by(&self, i: &Insn, q: usize) -> bool {
        debug_assert!(q < 144);
        placement(i.id(), q as u16).is_none()
    }

    /// Number of cycles instruction `i`, dispatched at cycle `c`, occupies its queue: Deskew to the
    /// next HAC epoch boundary, RuntimeDeskew `max(t, timing.c2c.rtdeskew_floor)` cycles, every other instruction 1.
    // model: sys/insts_begin.sail occupancy; slices/c2c/c2c_insts.sail; postlude/insts_end.sail
    pub fn occupancy(&self, i: &Insn, c: u64) -> u64 {
        match *i {
            Insn::Deskew => {
                let p = self.geom.hac_period;
                p - c % p
            }
            Insn::RuntimeDeskew { t } => (t as u64).max(self.geom.rtdeskew_floor),
            _ => 1,
        }
    }

    /// What dispatching `i` does once nothing holds it: the unit check. The hold conditions are
    /// those of `hold_check`, which the head parcel has already passed (an assert, not an outcome).
    // model: postlude/tick.sail dispatch_check
    pub fn dispatch_check(&self, q: usize, c: u64, i: &Insn, path: Option<usize>) -> Result<Option<Diag>, RunError> {
        if self.hold_check(q, c, Some(i), path, false) {
            return Err(RunError::fault(c, &format!("dispatch of a held instruction (queue {q}, {i:?}, {path:?})")));
        }
        Ok(if !self.issued_by(i, q) { Some(Diag::Que002) } else { None })
    }

    /// The unit check of [`Machine::dispatch_check`] without the hold test.
    fn unit_check(&self, q: usize, i: &Insn) -> Option<Diag> {
        if !self.issued_by(i, q) {
            Some(Diag::Que002)
        } else {
            None
        }
    }

    /// The parcels the common path takes: one instruction that needs the whole queue (not Repeat, Sync or Notify) on a
    /// queue above the MEM queues, or MEM references only (no configuration record, no Ifetch), on a queue that exists.
    pub fn is_common_parcel(&self, q: usize, r: &CycleRecs) -> bool {
        if !r.ok || r.n == 0 {
            return false;
        }
        match r.single {
            Some(i) => {
                q > 87
                    && !matches!(i.id(), EntryId::Repeat | EntryId::Sync | EntryId::Notify)
                    && queue_present(&self.geom, q)
            }
            None => q <= 87 && r.cfg.is_none() && r.ifetch.is_none() && queue_present(&self.geom, q),
        }
    }

    /// Dispatch a parcel that [`Machine::is_common_parcel`] accepts, on a queue not parked and with both AGT paths idle,
    /// that the tick has found not held at `c`. What `dispatch` would do with it, without what cannot happen here: no
    /// AGT reference is due, no loop clashes with a reference, and the hold test the tick has just made is not made again.
    /// `TSP_ISS_FAST_CHECK` makes it evaluate those and assert them.
    // model: postlude/tick.sail dispatch
    pub(super) fn dispatch_common<O: Observer + ?Sized>(
        &mut self,
        q: usize,
        c: u64,
        r: &CycleRecs,
        obs: &mut O,
    ) -> Result<(), RunError> {
        self.d.fast_dispatches += 1;
        if self.d.fast_check {
            self.check_common(q, c, r);
        }
        match r.single {
            Some(i) => {
                if let Some(d) = self.unit_check(q, &i) {
                    self.diag(obs, q, c, d);
                } else {
                    self.dispatch_body(q, c, i, obs);
                }
                Ok(())
            }
            None if r.n == 1 && self.d.local_fetch_bank[q].is_none_or(|(at, _)| at != u128::from(c)) => {
                let k = r.refs.iter().position(Option::is_some).expect("one reference");
                self.dispatch_one_ref(q, c, k, r.refs[k].expect("one reference"), obs);
                Ok(())
            }
            None => self.dispatch_slots(q, c, r, false, true, obs),
        }
    }

    /// An IfetchM issued at cycle `c`, as a new instruction or as a Repeat iteration: it holds the R path
    /// through the cycle of its second read (`c + 1`) and takes the bank of word `a + 1` in that cycle, so that
    /// `ifetchm a; ifetchm a` and `ifetchm a; repeat 1, d` are one schedule.
    // model: postlude/tick.sail reserve_local_fetch
    pub(super) fn reserve_local_fetch(&mut self, q: usize, i: &Insn, c: u64) {
        self.d.path_busy_until[2 * q] = u128::from(c) + 2;
        if q <= 87 {
            if let Some(note) = super::fetchm_bank_note(i, c, self.mem_segment[q], self.geom.words_per_slice) {
                self.d.local_fetch_bank[q] = Some((note.0, note.1));
            }
        }
    }

    /// A Repeat iteration of `i` at cycle `c`: the execution, and for an IfetchM the reservation of its second beat.
    // model: postlude/tick.sail repeat_issue
    pub(super) fn repeat_issue<O: Observer + ?Sized>(&mut self, q: usize, i: Insn, c: u64, obs: &mut O) {
        self.run_element(q, i, c, 0, 0, 0, obs);
        if matches!(i, Insn::IfetchM { .. }) {
            self.reserve_local_fetch(q, &i, c);
        }
    }

    /// `dispatch_slots` of a parcel whose one instruction is the MEM reference in slot `k` (no other reference to clash
    /// with, no configuration record, no Ifetch, no AGT loop): the unit check, the path held, the element run, the
    /// configuration opened when the reference is loopable, and the queue's last instruction, in the same order.
    fn dispatch_one_ref<O: Observer + ?Sized>(&mut self, q: usize, c: u64, k: usize, i: Insn, obs: &mut O) {
        if let Some(d) = self.unit_check(q, &i) {
            self.diag(obs, q, c, d);
            return;
        }
        let p = usize::from(k >= 2);
        self.d.hold_path(q, p, c);
        self.run_element(q, i, c, 0, 0, 0, obs);
        if is_loopable(&i) {
            self.d.agt_set(q, p, Agt { state: AgtState::Config, instr: Some(i), t0: c, last: c, ..Agt::idle() });
            self.d.agt_records[2 * q + p] = 0;
        }
        self.d.queue_last[q] = Some(i);
        self.d.queue_last_at[q] = c;
    }

    /// `TSP_ISS_FAST_CHECK`: what [`Machine::dispatch_common`] leaves out is what the general path finds.
    fn check_common(&self, q: usize, c: u64, r: &CycleRecs) {
        assert!(self.is_common_parcel(q, r), "fast path: queue {q} at cycle {c}: not a common parcel");
        assert!(!self.d.barrier.queue_parked(q, c), "fast path: queue {q} at cycle {c}: parked");
        assert!(
            self.d.agts[q].state == AgtState::Idle && self.d.agts_w[q].state == AgtState::Idle,
            "fast path: queue {q} at cycle {c}: an AGT is not idle"
        );
        let due = self.due_refs(q, c, false);
        assert!(due[0].is_none() && due[1].is_none(), "fast path: queue {q} at cycle {c}: an AGT reference is due");
        assert!(
            self.loop_tag(q, 0, &due[0]).is_none() && self.loop_tag(q, 1, &due[1]).is_none(),
            "fast path: queue {q} at cycle {c}: a loop tag"
        );
        match r.single {
            Some(i) => assert!(
                !self.hold_check(q, c, Some(&i), None, false),
                "fast path: queue {q} at cycle {c}: {i:?} is held"
            ),
            None => {
                assert!(
                    r.n != 1 || r.refs.iter().filter(|s| s.is_some()).count() == 1,
                    "fast path: queue {q} at cycle {c}: a one-instruction parcel without exactly one reference"
                );
                for (k, slot) in r.refs.iter().enumerate() {
                    if let Some(i) = slot {
                        let p = Some(usize::from(k >= 2));
                        assert!(
                            !self.hold_check(q, c, Some(i), p, false),
                            "fast path: queue {q} at cycle {c}: {i:?} is held"
                        );
                    }
                }
            }
        }
    }

    /// An instruction that needs the whole queue (every instruction other than a MEM reference, a
    /// MEM configuration record or, on a MEM queue, Ifetch).
    // model: postlude/tick.sail dispatch_body
    pub(super) fn dispatch_body<O: Observer + ?Sized>(&mut self, q: usize, c: u64, i: Insn, obs: &mut O) {
        if clears_repeat(&i) {
            self.d.queue_last[q] = None;
        } else if repeatable(&i) {
            self.d.queue_last[q] = Some(i);
            self.d.queue_last_at[q] = c;
        }
        self.d.queue_busy_until[q] = u128::from(c) + u128::from(self.occupancy(&i, c));
        match i.id() {
            EntryId::Sync => self.d.barrier.sync(q, c),
            EntryId::Notify => self.d.barrier.notify(c),
            // HostWait takes a ringing doorbell or parks the queue (a doorbell the profile does not have is
            // OTS-E-OPR-001 at execution and parks nothing)
            EntryId::HostWait => {
                if let Insn::HostWait { i: n } = i {
                    if (n as u64) < self.geom.host_doorbells {
                        self.host_wait(q, c, n as usize);
                    }
                }
            }
            // DeskewRecv: the diagnostics of Receive that do not depend on an arrival are the dispatch's and hold nothing;
            // otherwise the queue polls at each epoch boundary after this cycle (`tick`)
            EntryId::DeskewRecv => {
                if let Insn::DeskewRecv { chip_id, link, stream } = i {
                    match self.deskewrecv_diag(q, chip_id, link, stream) {
                        Some(d) => self.diag(obs, q, c, d),
                        None => {
                            self.d.queue_poll[q] = Some(PollState { chip: chip_id, link, stream, since: c });
                            self.d.poll_n += 1;
                        }
                    }
                }
            }
            _ => {}
        }
        self.run_element(q, i, c, 0, 0, 0, obs);
    }

    // model: postlude/tick.sail dispatch_decoded
    fn dispatch_decoded<O: Observer + ?Sized>(
        &mut self,
        q: usize,
        c: u64,
        i: Insn,
        obs: &mut O,
    ) -> Result<(), RunError> {
        if let Some(d) = self.dispatch_check(q, c, &i, None)? {
            self.diag(obs, q, c, d);
            return Ok(());
        }
        match i {
            Insn::Repeat { n, d } => match self.d.queue_last[q] {
                None => self.diag(obs, q, c, Diag::Res001), // nothing to repeat
                Some(prev) => {
                    self.d.queue_busy_until[q] = u128::from(c) + 1;
                    let (now, rest) = super::repeat::repeat_plan(prev, n as u64, d as u64, self.d.queue_last_at[q], c);
                    if now {
                        if self.iq.observe_sites {
                            obs.site(self.chip_id(), c, q, "repeat", format_args!("{}", self.d.queue_last_at[q]));
                        }
                        self.repeat_issue(q, prev, c, obs);
                        self.d.queue_last_at[q] = c;
                    }
                    self.d.queue_repeat[q] = rest;
                }
            },
            _ => self.dispatch_body(q, c, i, obs),
        }
        Ok(())
    }

    /// A configuration record on path `p`.
    // model: postlude/tick.sail dispatch_cfg
    fn dispatch_cfg<O: Observer + ?Sized>(
        &mut self,
        q: usize,
        c: u64,
        p: usize,
        i: Insn,
        obs: &mut O,
    ) -> Result<(), RunError> {
        if let Some(d) = self.dispatch_check(q, c, &i, Some(p))? {
            self.diag(obs, q, c, d);
            return Ok(());
        }
        // A configuration record continues the configuration of the loopable instruction before it
        // on its path. With no configuration open (a stray record, or after the loop was aborted)
        // it is OTS-E-OPR-001. A fifth level is OTS-E-OPR-001 and aborts the loop.
        let (f, v) = agt_config(&i).ok_or_else(|| RunError::fault(c, "configuration record"))?;
        let mut a = self.d.agt_get(q, p);
        if a.state != AgtState::Config {
            self.diag(obs, q, c, Diag::Opr001);
        } else {
            self.d.agt_records[2 * q + p] += 1;
            if agt_configure(&mut a, f, v) {
                self.d.agt_set(q, p, a);
            } else {
                self.diag(obs, q, c, Diag::Opr001);
                self.d.agt_set(q, p, Agt::idle());
            }
        }
        Ok(())
    }

    /// The AGT reference of path `p` due at cycle `c` (`st`, from `agt_due`): it runs unless blocked
    /// by the bank rule (then it is OTS-E-MEM-001 and has no effect); either way it holds its path
    /// and the loop advances (`agt_refresh`: next reference, or idle).
    // model: postlude/tick.sail run_loop
    #[allow(clippy::too_many_arguments)]
    fn run_loop<O: Observer + ?Sized>(
        &mut self,
        q: usize,
        c: u64,
        p: usize,
        st: AgtDue,
        blocked: bool,
        partner: Option<ClashRef>,
        obs: &mut O,
    ) {
        let Some((_, s, ss)) = st else { return };
        let a = self.d.agt_ref(q, p);
        let (a_instr, a_t0, a_offset, a_soffset, n) = (a.instr, a.t0, a.offset, a.soffset, agt_nlevels(a));
        #[cfg(debug_assertions)]
        let expect = super::agt::agt_advance(&a.levels, n).map(|r| r.0);
        // the element runs and the path is held, then the loop's counters move in place: nothing else of the AGT may
        // change in between (the copying form wrote the fields it had read back, which only holds if none did)
        #[cfg(debug_assertions)]
        let before = *a;
        if blocked {
            self.diag(obs, q, c, Diag::Mem001);
            if let Some(o) = partner {
                let me = self.loop_clash_ref(q, p, s, ss);
                self.bank_report(q, c, &me, &o, obs);
            }
        } else {
            self.run_element(
                q,
                a_instr.unwrap_or_else(|| model_fault("an AGT reference without its instruction")),
                a_t0,
                c.saturating_sub(a_t0),
                a_offset + s,
                a_soffset + ss,
                obs,
            );
        }
        self.d.hold_path(q, p, c);
        #[cfg(debug_assertions)]
        debug_assert_eq!(*self.d.agt_ref(q, p), before, "running an AGT reference changed its AGT");
        let m = self.d.agt_mut(q, p);
        agt_apply_advance(&mut m.levels, n);
        #[cfg(debug_assertions)]
        debug_assert_eq!(Some(m.levels), expect, "the in-place advance differs from agt_advance");
        m.offset = a_offset + s;
        m.soffset = a_soffset + ss;
        m.last = c;
        self.d.agt_refresh(q, p);
    }

    /// The reference of path `p`'s AGT due this cycle as the bank-conflict report needs it.
    // model: postlude/tick.sail loop_clash_ref
    fn loop_clash_ref(&self, q: usize, p: usize, s: i64, ss: i64) -> ClashRef {
        let a = self.d.agt_ref(q, p);
        (
            a.instr.unwrap_or_else(|| model_fault("an AGT reference without its instruction")),
            a.offset + s,
            a.soffset + ss,
        )
    }

    // model: postlude/tick.sail due_ref
    fn due_ref(&self, q: usize, p: usize, st: &AgtDue) -> Option<ClashRef> {
        st.as_ref().map(|&(_, s, ss)| self.loop_clash_ref(q, p, s, ss))
    }

    /// The `(thread, bank)` of path `p`'s AGT reference due this cycle.
    // model: postlude/tick.sail loop_tag
    fn loop_tag(&self, q: usize, p: usize, st: &AgtDue) -> Option<RefTag> {
        st.as_ref().map(|&(_, s, _)| {
            let a = self.d.agt_ref(q, p);
            self.tag_of(
                &a.instr.unwrap_or_else(|| model_fault("an AGT reference without its instruction")),
                a.offset + s,
            )
        })
    }

    /// The AGT references due at cycle `c` on both paths.
    // model: postlude/tick.sail due_refs
    fn due_refs(&self, q: usize, c: u64, parked: bool) -> [AgtDue; 2] {
        [agt_due_lean(self.d.agt_ref(q, 0), c, parked), agt_due_lean(self.d.agt_ref(q, 1), c, parked)]
    }

    /// The records of MEM queue `q` at cycle `c` that fit their slots, with the AGT references due
    /// this cycle. The configuration record dispatches before the AGT references are worked out,
    /// so a reference of this cycle sees it.
    // model: postlude/tick.sail dispatch_slots
    fn dispatch_slots<O: Observer + ?Sized>(
        &mut self,
        q: usize,
        c: u64,
        r: &CycleRecs,
        parked: bool,
        lite: bool,
        obs: &mut O,
    ) -> Result<(), RunError> {
        let mut issued: u32 = 0;
        let mut last: Option<Insn> = None;
        // First and second fetch reads use their own cycle's bank.
        let mut fm_ok = false;
        let prior = self.d.local_fetch_bank[q].filter(|(at, _)| *at == u128::from(c));
        let fm_prior = prior.is_some();
        let mut fm_bank = prior.map_or(BankUse::None, |(_, a)| self.addr_bank(a as i64));
        if let Some(i) = r.ifetch {
            let fpath = matches!(i.id(), EntryId::IfetchM).then_some(0);
            match self.dispatch_check(q, c, &i, fpath)? {
                Some(d) => self.diag(obs, q, c, d),
                None => match i {
                    Insn::IfetchM { addr } => {
                        fm_ok = true;
                        fm_bank = self.addr_bank(addr as i64);
                    }
                    _ => {
                        self.run_element(q, i, c, 0, 0, 0, obs);
                        issued += 1;
                        last = Some(i);
                    }
                },
            }
        }
        if let Some(i) = r.cfg {
            let p = self.d.cfg_path(q);
            self.dispatch_cfg(q, c, p, i, obs)?;
        }
        // `lite` (see `dispatch_common`): no AGT loop on the queue, so no reference is due and no loop clashes.
        let due = if lite { [None, None] } else { self.due_refs(q, c, parked) };
        let looptag: [Option<RefTag>; 2] =
            if lite { [None, None] } else { [self.loop_tag(q, 0, &due[0]), self.loop_tag(q, 1, &due[1])] };
        // the AGT references as the bank-conflict report needs them, taken before any of them runs (running advances the loop)
        let lref: [Option<ClashRef>; 2] =
            if lite { [None, None] } else { [self.due_ref(q, 0, &due[0]), self.due_ref(q, 1, &due[1])] };
        // References: each checked on its own path; then the bank rule, against the other path's
        // reference of the same thread in this cycle (a new one, or an AGT reference).
        let mut go = [false; 4];
        for (k, (slot, ok)) in r.refs.iter().zip(go.iter_mut()).enumerate() {
            if let Some(i) = slot {
                let chk =
                    if lite { self.unit_check(q, i) } else { self.dispatch_check(q, c, i, Some(usize::from(k >= 2)))? };
                match chk {
                    Some(d) => self.diag(obs, q, c, d),
                    None => *ok = true,
                }
            }
        }
        let mut bank = [false; 4];
        for t in 0..2 {
            for p in 0..2 {
                let k = 2 * p + t;
                let o = 2 * (1 - p) + t;
                if let Some(i) = r.refs[k] {
                    if go[k] {
                        let me = self.tag_of(&i, 0);
                        let new_clash = match r.refs[o] {
                            Some(j) => go[o] && tags_clash(me, self.tag_of(&j, 0)),
                            None => false,
                        };
                        let loop_clash = match looptag[1 - p] {
                            Some(tg) => tags_clash(me, tg),
                            None => false,
                        };
                        if new_clash || loop_clash {
                            bank[k] = true;
                        }
                    }
                }
            }
        }
        // the IfetchM bank rule against the W side of the cycle.
        let mut fm_clash = false;
        if fm_ok || fm_prior {
            for t in 0..2 {
                if let Some(j) = r.refs[2 + t] {
                    if go[2 + t] {
                        let jb = self.ref_bank(&j, 0);
                        if banks_meet(jb, fm_bank) {
                            bank[2 + t] = true;
                            fm_clash = true;
                        }
                    }
                }
            }
        }
        let mut fm_loop = false;
        if fm_ok || fm_prior {
            if let Some(tg) = looptag[1] {
                if banks_meet(tg.bank, fm_bank) {
                    fm_loop = true;
                    fm_clash = true;
                }
            }
        }
        if fm_ok {
            if fm_clash {
                self.diag(obs, q, c, Diag::Mem001);
            } else if let Some(i) = r.ifetch {
                self.run_element(q, i, c, 0, 0, 0, obs);
                self.reserve_local_fetch(q, &i, c);
                issued += 1;
                last = Some(i);
            }
        }
        // An AGT reference is blocked by a clashing reference on the other path: the other path's
        // AGT reference, or a new reference that passed its checks.
        let mut lblock = [false; 2];
        let mut lpart: [Option<ClashRef>; 2] = [None, None];
        for p in 0..2 {
            if let Some(me) = looptag[p] {
                let mut hit = match looptag[1 - p] {
                    Some(o) => tags_clash(me, o),
                    None => false,
                };
                if hit {
                    lpart[p] = lref[1 - p];
                }
                for t in 0..2 {
                    if let Some(j) = r.refs[2 * (1 - p) + t] {
                        if go[2 * (1 - p) + t] && tags_clash(me, self.tag_of(&j, 0)) {
                            hit = true;
                            lpart[p] = Some((j, 0, 0));
                        }
                    }
                }
                lblock[p] = hit || (p == 1 && fm_loop);
            }
        }
        if !lite {
            for p in 0..2 {
                self.run_loop(q, c, p, due[p], lblock[p], lpart[p], obs);
            }
        }
        for k in 0..4 {
            if let Some(i) = r.refs[k] {
                if go[k] {
                    if bank[k] {
                        self.diag(obs, q, c, Diag::Mem001);
                        // its partner: the other path's new reference of the thread, else that path's AGT reference
                        let po = if k >= 2 { 0 } else { 1 };
                        let o = 2 * po + k % 2;
                        let mut partner: Option<ClashRef> = None;
                        if let Some(tg) = looptag[po] {
                            if tags_clash(self.tag_of(&i, 0), tg) {
                                partner = lref[po];
                            }
                        }
                        if let Some(j) = r.refs[o] {
                            if go[o] && tags_clash(self.tag_of(&i, 0), self.tag_of(&j, 0)) {
                                partner = Some((j, 0, 0));
                            }
                        }
                        if let Some(x) = partner {
                            self.bank_report(q, c, &(i, 0, 0), &x, obs);
                        }
                    } else {
                        let p = usize::from(k >= 2);
                        self.d.hold_path(q, p, c);
                        issued += 1;
                        last = Some(i);
                        // Reference 0 opens its path's AGT configuration when it dispatches,
                        // whatever its execution (at dispatch + d_skew) reports; two references
                        // dispatched on the path in this cycle open no loop (opentsp decisions).
                        let alone = r.refs[k - (k % 2) + (1 - k % 2)].is_none();
                        self.run_element(q, i, c, 0, 0, 0, obs);
                        if is_loopable(&i) && alone {
                            self.d.agt_set(
                                q,
                                p,
                                Agt { state: AgtState::Config, instr: Some(i), t0: c, last: c, ..Agt::idle() },
                            );
                            self.d.agt_records[2 * q + p] = 0;
                        }
                    }
                }
            }
        }
        // Repeat re-issues the queue's last instruction only when a cycle issued exactly one (a
        // Read/Write pair, or a reference with an Ifetch, is not repeatable).
        if issued > 1 {
            self.d.queue_last[q] = None;
        } else if issued == 1 {
            self.d.queue_last[q] = last;
            self.d.queue_last_at[q] = c;
        }
        Ok(())
    }

    /// Dispatch the parcel of queue `q` that `tick` took from its IQ (or nothing: `r.n == 0`).
    // model: postlude/tick.sail dispatch
    pub fn dispatch<O: Observer + ?Sized>(
        &mut self,
        q: usize,
        c: u64,
        r: &CycleRecs,
        parked: bool,
        obs: &mut O,
    ) -> Result<(), RunError> {
        let n = r.n;
        // The queue of an absent MEM slice has no unit. Every instruction of a parcel it
        // dispatches is OTS-E-QUE-002 (one each, whatever the instruction, including one that does
        // not decode and a parcel that would break the slot rule), and nothing else happens.
        if n != 0 && !queue_present(&self.geom, q) {
            for _ in 0..n {
                self.diag(obs, q, c, Diag::Que002);
            }
            return Ok(());
        }
        let slots = n != 0 && r.ok && r.single.is_none();
        // Without slot instructions, the due AGT references issue on their own (only they can
        // clash). A parked queue and one with both paths idle have none.
        let loops = !parked && (self.d.agts[q].state != AgtState::Idle || self.d.agts_w[q].state != AgtState::Idle);
        if !slots && loops {
            let due = self.due_refs(q, c, parked);
            let both = match (self.loop_tag(q, 0, &due[0]), self.loop_tag(q, 1, &due[1])) {
                (Some(a), Some(b)) => tags_clash(a, b),
                _ => false,
            };
            let l = [self.due_ref(q, 0, &due[0]), self.due_ref(q, 1, &due[1])];
            for (p, d) in due.into_iter().enumerate() {
                self.run_loop(q, c, p, d, both, l[1 - p], obs);
            }
        }
        if n == 0 {
        } else if !r.ok {
            match r.err {
                Some(d) => self.diag(obs, q, c, d), // one instruction that does not decode
                None => {
                    for _ in 0..n {
                        self.diag(obs, q, c, Diag::Que001); // a parcel that breaks the slot rule
                    }
                }
            }
        } else {
            match r.single {
                Some(i) => self.dispatch_decoded(q, c, i, obs)?,
                None => self.dispatch_slots(q, c, r, parked, false, obs)?,
            }
        }
        Ok(())
    }

    /// The control channel carries Sync, Notify and Config only (NOP is the timing prefix of the
    /// next parcel; Repeat, Ifetch, SetVL and every slice instruction are OTS-E-QUE-002). A control
    /// Sync parks every data queue. The control channel has no dispatch paths and no Repeat, so
    /// nothing holds its head parcel: it dispatches when it is complete and due. It has no Ifetch,
    /// so its program is its initial image.
    // model: postlude/tick.sail control
    pub fn control<O: Observer + ?Sized>(&mut self, c: u64, obs: &mut O) {
        let h = self.iq.head(144);
        if !(h.complete && crate::frontend::earliest_after_nops(&self.iq, 144, h.nops) <= c.into()) {
            return;
        }
        let (units, n) = (h.units, h.n);
        let first = h.items.first().copied();
        if self.iq.observe_sites {
            let earliest = crate::frontend::earliest_after_nops(&self.iq, 144, h.nops);
            self.parcel_site(144, c, "dispatch", units, earliest, obs);
        }
        self.text_check(144, c, units, obs);
        self.iq.pop(144, units);
        self.iq.last[144] = i128::from(c);
        self.iq.active[144] = c;
        obs.dispatch(c, 0xffff);
        match (n, first) {
            (1, Some(Err(d))) => self.diag(obs, 144, c, d),
            (1, Some(Ok(i))) => match i.id() {
                EntryId::Sync => self.d.barrier.sync(144, c),
                EntryId::Notify => self.d.barrier.notify(c),
                EntryId::Config => {}
                _ => self.diag(obs, 144, c, Diag::Que002),
            },
            (0, _) => {}
            _ => {
                for _ in 0..n {
                    self.diag(obs, 144, c, Diag::Que001);
                }
            }
        }
    }
}
