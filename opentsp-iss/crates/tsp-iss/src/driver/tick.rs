//! One cycle (`postlude/tick.sail` `tick`, `enqueue`, `reset`). Order (`microarchitecture.md` §1):
//! fetched text enters the IQs, barrier retirement, commits due at c, the elements that execute at c
//! (dispatched at c - d_skew), the control channel, then data queues 0..143; per queue: the next
//! Repeat iteration; the head parcel (complete, due and not held) with the end of an AGT
//! configuration that it does not continue; the AGT references due. A configuration record of an
//! open AGT configuration is applied before the AGT step, so the reference of this cycle sees it. A
//! parked queue's Repeat iterations and AGT references wait; the first one issues in the first
//! unparked cycle, and later ones keep their spacing from it. The driver does not rely on being
//! called every cycle.
use super::agt::AgtState;
use super::classify::{config_for, CycleRecs};
use super::phase;
use super::Driver;
use super::{bits_from, lint, pop_bit, ALL_DATA, LINT_IQ_UNDERFLOW};
use crate::frontend::earliest_after_nops;
use crate::machine::{fault_boundary, RunError};
use crate::observe::{Hold, Observer, Phase};
use crate::Machine;
use ots_isa::{Code as Diag, Insn};

impl<const L: usize> Machine<L> {
    /// The text of the fetches that are due enters the IQ at the start of cycle `c`, before anything
    /// dispatches (design sec. 7.4). A unit lost to a full IQ is the diagnostic OTS-E-IQ-002, once per queue
    /// and cycle. Underflow (sec. 7.7): when this completes the head parcel after its earliest cycle, and
    /// the queue was free in the cycle before, the IQ ran dry: logical time was lost (the lint
    /// OTS-W-IQ-001 when the parcel dispatches).
    // model: postlude/tick.sail enqueue
    fn enqueue<O: Observer + ?Sized>(&mut self, q: usize, c: u64, obs: &mut O) {
        let Some(seg) = self.iq.fill[q].front() else { return };
        if seg.ready > u128::from(c) {
            return;
        }
        let before = self.iq.head_brief(q).0;
        let (pushed, lost) = self.iq.deliver(q, c);
        let m = phase::begin();
        self.iq.parse(&self.isa, q);
        phase::sem_end(phase::PARSE, m);
        if lost {
            self.diag(obs, q, c, Diag::Iq002);
        }
        if pushed && !before {
            self.d.touch(q); // a push past a complete head parcel leaves it as it was
        }
        if pushed && !before && c > 0 {
            let (complete, _, nops) = self.iq.head_brief(q);
            if complete && u128::from(c) > earliest_after_nops(&self.iq, q, nops) && self.free_at(q, c - 1) {
                self.iq.dry[q] = true;
            }
        }
    }

    pub(super) fn parcel_site<O: Observer + ?Sized>(
        &self,
        q: usize,
        c: u64,
        event: &str,
        units: usize,
        earliest: u128,
        obs: &mut O,
    ) {
        let sources = self.iq.site_sources(q, units, self.images[q].len());
        obs.site(self.chip_id(), c, q, event, format_args!("{earliest}{sources}"));
    }

    fn wait_site(&self, q: usize, c: u64) -> &'static str {
        super::wait_reason(
            self.d.barrier.host_parked(q, c),
            self.d.barrier.queue_parked(q, c),
            self.d.queue_poll[q].is_some(),
            self.d.queue_repeat[q].is_some(),
        )
    }

    /// One cycle; a model fault is a panic (see [`Machine::try_tick`]).
    // model: postlude/tick.sail tick
    pub fn tick<O: Observer + ?Sized>(&mut self, c: u64, obs: &mut O) {
        if let Err(e) = self.try_tick(c, obs) {
            panic!("{e}");
        }
    }

    /// One cycle, with a model fault (the Sail `internal_error`) as `Err(RunError::Internal)` naming
    /// the cycle; the machine's state is then unspecified.
    // model: postlude/tick.sail tick
    pub fn try_tick<O: Observer + ?Sized>(&mut self, c: u64, obs: &mut O) -> Result<(), RunError> {
        if super::replay::ENABLED && (self.rp.compiled | self.rp.rec_on) && self.geom.scrub_period == 0 {
            return self.try_tick_replay(c, obs);
        }
        self.try_tick_normal(c, obs)
    }

    /// The tick of a run that neither records nor replays a schedule.
    #[inline(always)]
    pub(crate) fn try_tick_normal<O: Observer + ?Sized>(&mut self, c: u64, obs: &mut O) -> Result<(), RunError> {
        if let Some(message) = &self.edit_failure {
            return Err(RunError::Input(message.clone()));
        }
        self.last_visited = Some(c);
        fault_boundary(c, || self.tick_cycle(c, obs))?
    }

    /// The tick of a run that records or replays a schedule (`driver::replay`), out of line so that the normal tick stays as it was.
    #[cold]
    #[inline(never)]
    fn try_tick_replay<O: Observer + ?Sized>(&mut self, c: u64, obs: &mut O) -> Result<(), RunError> {
        self.last_visited = Some(c);
        if self.rp.compiled {
            fault_boundary(c, || self.tick_compiled(c, obs))?
        } else {
            fault_boundary(c, || self.tick_recording(c, obs))?
        }
    }

    /// Current nominal MEM Reads are zero-skew: they run in dispatch order.
    /// Emit hints from due IQ heads or Repeat before dispatching any queue.
    /// Hints can miss a newly arrived operand or hint a word held by a barrier.
    /// The actual architectural data snapshot and effects stay at dispatch.
    #[cfg(all(
        feature = "native-mem-prefetch",
        any(target_arch = "x86_64", all(target_arch = "aarch64", target_vendor = "apple"))
    ))]
    #[inline(never)]
    fn prefetch_calendar_reads(&self, mut due: [u64; 3], cycle: u64) {
        let cycle = u128::from(cycle);
        while let Some(queue) = pop_bit(&mut due) {
            if let Some(repeat) = self.d.queue_repeat[queue] {
                if repeat.next <= cycle {
                    if let Insn::Read { addr, .. } = repeat.instr {
                        self.prefetch_calendar_word(queue, addr);
                    }
                }
            }
            let head = self.iq.head(queue);
            if head.complete && earliest_after_nops(&self.iq, queue, head.nops) <= cycle {
                for insn in head.recs.refs.iter().flatten() {
                    if let Insn::Read { addr, .. } = *insn {
                        self.prefetch_calendar_word(queue, addr);
                    }
                }
            }
        }
    }

    #[cfg(all(target_arch = "aarch64", target_vendor = "apple", feature = "native-mem-prefetch"))]
    #[inline]
    fn prefetch_calendar_word(&self, queue: usize, addr: u32) {
        if u64::from(addr) >= self.mem_segment[queue] {
            return;
        }
        let Some(index) = self.sram_index(addr) else { return };
        let Some(word) = self.sram.word_ref(queue, index) else { return };
        // The ordinary SRAM word borrow supplies each in-bounds cache-line address.
        // Hints cannot alter the word, architectural state, or its pending write.
        for line in word.chunks(128) {
            // SAFETY: the hint receives an address inside the currently
            // borrowed word. PRFM has no architectural memory effects.
            unsafe {
                std::arch::asm!("prfm pldl1keep, [{address}]", address = in(reg) line.as_ptr(),
                    options(readonly, nostack, preserves_flags));
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "native-mem-prefetch"))]
    #[inline]
    fn prefetch_calendar_word(&self, queue: usize, addr: u32) {
        if u64::from(addr) >= self.mem_segment[queue] {
            return;
        }
        let Some(index) = self.sram_index(addr) else { return };
        let Some(word) = self.sram.word_ref(queue, index) else { return };
        // The ordinary SRAM word borrow supplies each in-bounds cache-line address.
        // Hints cannot alter the word, architectural state, or its pending write.
        for lane in word.chunks(64) {
            unsafe { std::arch::x86_64::_mm_prefetch(lane.as_ptr().cast(), std::arch::x86_64::_MM_HINT_T0) };
        }
        if let Some(offset) = sram_hint_tail(word.as_ptr().addr(), word.len()) {
            // The offset is the final byte in this borrow, on a different line
            // from the last chunk start. Prefetch does not read architectural data.
            unsafe {
                std::arch::x86_64::_mm_prefetch(word.as_ptr().add(offset).cast(), std::arch::x86_64::_MM_HINT_T0)
            };
        }
    }

    pub(crate) fn tick_cycle<O: Observer + ?Sized>(&mut self, c: u64, obs: &mut O) -> Result<(), RunError> {
        self.d.last_tick = c;
        self.scrub_run(c.checked_sub(1), obs);
        self.local_fetch_complete(c, obs);
        let waits = obs.tracks_waits();
        let m = phase::begin();
        let prof = obs.profiling();
        if prof {
            obs.phase(Phase::Driver);
        }
        let mut fills = self.iq.fill_queues();
        while let Some(n) = pop_bit(&mut fills) {
            self.enqueue(n, c, obs);
        }
        phase::end(0, m);
        let m = phase::begin();
        if prof {
            obs.phase(Phase::Commit);
        }
        self.d.barrier.retire(c);
        self.commit_due(c, obs);
        self.stream_fetch_complete(c, obs);
        phase::end(1, m);
        let m = phase::begin();
        if prof {
            obs.phase(Phase::Execute);
        }
        self.execute_due(c, obs);
        phase::end(2, m);
        let m = phase::begin();
        if prof {
            obs.phase(Phase::Control);
        }
        self.control(c, obs);
        phase::end(3, m);
        if self.d.fast_check {
            self.check_early_closed(c);
        }
        let m = phase::begin();
        if prof {
            obs.phase(Phase::Calendar);
        }
        // `CycleRecs` carries seven instructions, so the cycle's records travel by reference, and a queue whose
        // parcel is not due this cycle never asks for them: it dispatches nothing, so no open configuration
        // continues and nothing is hold-checked.
        const NO_RECS: CycleRecs = CycleRecs::none();
        const NO_ITEMS: &[Result<Insn, Diag>] = &[];
        // The queues whose body runs (`wake <= c`) or that report a wait, as a bit set; a change of the barrier
        // during the loop makes every later queue run.
        self.wake_reset_on_barrier();
        let due = self.d.cal_take(c);
        #[cfg(all(
            feature = "native-mem-prefetch",
            any(target_arch = "x86_64", all(target_arch = "aarch64", target_vendor = "apple"))
        ))]
        if self.geom.vlanes == 320 && self.d.index_on {
            let sram_due = [due[0], due[1] & ((1 << 24) - 1), 0];
            if sram_due[0].count_ones() + sram_due[1].count_ones() >= 16 {
                self.prefetch_calendar_reads(sram_due, c);
            }
        }
        if prof {
            obs.phase(Phase::Driver);
        }
        let mut visit = due;
        if waits {
            for (v, i) in visit.iter_mut().zip(self.d.incomplete_set) {
                *v |= i;
            }
        }
        if self.d.poll_n != 0 {
            // a queue whose DeskewRecv looks for its vector at this epoch boundary has its body run
            for q in 0..crate::geom::DATA_QUEUES {
                if self.polls_at(q, c) {
                    visit[q >> 6] |= 1 << (q & 63);
                }
            }
        }
        if self.iq.observe_sites || self.d.index_check || !self.d.index_on {
            if self.d.index_check {
                self.check_calendar(c, &due);
            }
            visit = ALL_DATA;
        }
        phase::end(4, m);
        while let Some(q) = pop_bit(&mut visit) {
            let m = phase::begin();
            if self.d.index_on && self.d.wake[q] > u128::from(c) && !(self.d.poll_n != 0 && self.polls_at(q, c)) {
                if self.d.index_check {
                    self.check_skip(q, c);
                }
                // The calendar skips held queues, but Sail observes a complete, due head
                // on each tick even when another queue or an external event caused the tick.
                if self.iq.observe_sites {
                    let (complete, units, nops) = self.iq.head_brief(q);
                    let earliest = earliest_after_nops(&self.iq, q, nops);
                    if complete && earliest <= c.into() {
                        self.parcel_site(q, c, self.wait_site(q, c), units, earliest, obs);
                    }
                }
                if waits && self.d.incomplete[q] {
                    obs.wait(c, q as u16, Hold::IncompleteText, c); // a queue that is not due reports only this
                }
                phase::end(5, m);
                continue;
            }
            phase::end(5, m);
            let m = phase::begin();
            let (complete, units, nops) = self.iq.head_brief(q);
            let earliest = earliest_after_nops(&self.iq, q, nops);
            let due = complete && u128::from(c) >= earliest;
            let saved_head = due.then(|| *self.iq.head(q).recs);
            let recs = saved_head.as_ref().unwrap_or(&NO_RECS);
            phase::end(6, m);
            let m = phase::begin();
            let parked = self.d.barrier.queue_parked(q, c);
            // A DeskewRecv looks for its vector at each epoch boundary after its dispatch, in a cycle in which a barrier
            // does not park the queue; a vector that has arrived makes it a Receive dispatched in this cycle.
            if self.d.poll_n != 0 && !parked && self.polls_at(q, c) {
                let p = self.d.queue_poll[q].expect("polls_at");
                if self.deskewrecv_ready(c, p.link) {
                    obs.deskew_released(q as u16, p.since, c);
                    self.d.queue_poll[q] = None;
                    self.d.poll_n -= 1;
                    let rcv = Insn::Receive { chip_id: p.chip, link: p.link, stream: p.stream };
                    self.d.queue_busy_until[q] = u128::from(c) + u128::from(self.occupancy(&rcv, c));
                    self.run_element(q, rcv, c, 0, 0, 0, obs);
                }
            }
            if self.d.queue_repeat[q].as_ref().is_some_and(|r| r.next <= u128::from(c)) && !parked {
                let r = self.d.queue_repeat[q].expect("checked");
                if self.iq.observe_sites {
                    obs.site(self.chip_id(), c, q, "repeat", format_args!("{}", self.d.queue_last_at[q]));
                }
                self.repeat_issue(q, r.instr, c, obs);
                self.d.queue_last_at[q] = c;
                self.d.queue_busy_until[q] = u128::from(c) + 1;
                self.d.queue_repeat[q] = super::repeat::repeat_after(r, c);
            }
            phase::end(7, m);
            let m = phase::begin();
            // A configuration ends at the first cycle in which the queue dispatches no
            // configuration instruction (a parcel that is not due, not complete or held dispatches
            // nothing).
            let items = if due { self.iq.head(q).items } else { NO_ITEMS };
            let mut go = due && !self.parcel_held(q, c, recs, items);
            let mut ended = false;
            for path in 0..2 {
                if super::config_ends(self.d.agt_ref(q, path).state, go, config_for(recs)) {
                    self.d.agt_finish_config(q, path);
                    ended = true;
                }
            }
            if ended && due && !go {
                go = !self.parcel_held(q, c, recs, items); // the ended configuration no longer holds the head
            }
            if self.iq.observe_sites && due {
                self.parcel_site(q, c, if go { "dispatch" } else { self.wait_site(q, c) }, units, earliest, obs);
            }
            if waits && !go {
                if due {
                    if let Some(why) = self.parcel_hold(q, c, recs, items) {
                        obs.wait(
                            c,
                            q as u16,
                            why,
                            u64::try_from(earliest).expect("due parcel is within current cycle"),
                        );
                    }
                } else if self.iq.len[q] > 0 && !complete {
                    obs.wait(c, q as u16, Hold::IncompleteText, c);
                }
            }
            phase::end(8, m);
            let m = phase::begin();
            let r2 = if go { recs } else { &NO_RECS };
            if go {
                self.text_check(q, c, units, obs);
                self.iq.pop(q, units);
                self.iq.last[q] = i128::from(c);
                self.iq.active[q] = c;
                obs.dispatch(c, q as u16);
                if self.iq.dry[q] {
                    self.iq.dry[q] = false;
                    lint(obs, q, c, LINT_IQ_UNDERFLOW);
                }
            }
            phase::end(9, m);
            let m = phase::begin();
            // AGT references are due after the cycle's configuration record (if any) is applied,
            // so dispatch() works them out (each holds only its own path).
            if prof {
                obs.phase(Phase::Dispatch);
            }
            let (r_idle, w_idle) = (self.d.agts[q].state == AgtState::Idle, self.d.agts_w[q].state == AgtState::Idle);
            if go {
                if self.d.fast_on && !parked && r_idle && w_idle && self.is_common_parcel(q, r2) {
                    self.dispatch_common(q, c, r2, obs)?;
                } else {
                    self.dispatch(q, c, r2, parked, obs)?;
                }
            } else if self.d.fast_on && !(r_idle && w_idle) {
                self.dispatch(q, c, r2, parked, obs)?; // AGT references of a queue that dispatches nothing
            } else if !self.d.fast_on {
                self.dispatch(q, c, r2, parked, obs)?;
            }
            phase::end(10, m);
            let m = phase::begin();
            if prof {
                obs.phase(Phase::Driver);
            }
            if go && self.d.early_close && !prof {
                self.close_config_early(q, c);
            }
            let (w, inc) = self.wake_of(q, c);
            self.d.set_wake(q, w);
            self.d.incomplete[q] = inc;
            let bit = 1u64 << (q & 63);
            if inc {
                self.d.incomplete_set[q >> 6] |= bit;
            } else {
                self.d.incomplete_set[q >> 6] &= !bit;
            }
            self.d.ev_invalidate(q);
            if self.wake_reset_on_barrier() {
                visit = bits_from(&ALL_DATA, q + 1);
            }
            phase::end(11, m);
        }
        let m = phase::begin();
        if prof {
            obs.phase(Phase::Flush);
        }
        self.unit_state_flush(c, obs);
        self.scrub_run(Some(c), obs);
        phase::end(12, m);
        Ok(())
    }

    /// A reference that was just dispatched opened an AGT configuration, which ends at the next tick's visit of the
    /// queue unless that tick dispatches a configuration record. When it cannot (the head parcel, as the IQ stands,
    /// is not due next cycle or carries no record, and no text arrives next cycle that could complete a head that
    /// is not complete), the end is done now: `agt_finish_config` as that visit would, and the tick at `c + 1` is
    /// still made (`cfg_tick`). The visit is then not made, and the queue's wake cycle is that of an idle AGT.
    fn close_config_early(&mut self, q: usize, c: u64) {
        let open = |d: &Driver<L>, p: usize| {
            let a = d.agt_ref(q, p);
            a.state == AgtState::Config && super::agt::agt_nlevels(a) == 0
        };
        if !(open(&self.d, 0) || open(&self.d, 1)) {
            return;
        }
        let (complete, _, nops) = self.iq.head_brief(q);
        let safe = if complete {
            nops > 0 || !config_for(self.iq.head(q).recs)
        } else {
            self.iq.fill[q].front().is_none_or(|s| s.ready > u128::from(c) + 1)
        };
        if !safe {
            return;
        }
        for p in 0..2 {
            if open(&self.d, p) {
                self.d.agt_finish_config(q, p);
                self.d.cfg_tick = u128::from(c) + 1;
                self.d.early_closes += 1;
                if self.d.fast_check {
                    self.d.early_closed.push((q, p, c));
                }
            }
        }
    }

    /// Queue `q` has a DeskewRecv in its polling loop and `c` is an epoch boundary after its dispatch.
    // model: postlude/tick.sail tick (the `queue_poll` match)
    fn polls_at(&self, q: usize, c: u64) -> bool {
        self.d.queue_poll[q].is_some_and(|p| super::poll_epoch(p.since, c, self.geom.hac_period, false))
    }

    /// `TSP_ISS_FAST_CHECK`: at the tick after an early end of a configuration (after its arrivals are in), the tick is
    /// at `c + 1`, and the visit that was saved would have found no configuration record to continue it.
    fn check_early_closed(&mut self, c: u64) {
        for (q, p, c0) in std::mem::take(&mut self.d.early_closed) {
            assert_eq!(c, c0 + 1, "early close of queue {q} at cycle {c0}: the next tick is at cycle {c}");
            let (complete, _, nops) = self.iq.head_brief(q);
            let due = complete && u128::from(c) >= earliest_after_nops(&self.iq, q, nops);
            assert!(
                !(due && config_for(self.iq.head(q).recs)),
                "early close of queue {q} at cycle {c0}: a configuration record is due at {c}"
            );
            assert!(
                self.d.agt_ref(q, p).state == AgtState::Idle,
                "early close of queue {q} at cycle {c0}: state at {c}"
            );
        }
    }

    /// Every `wake` depends on the barrier: when it changed, they are all 0 (`true`).
    fn wake_reset_on_barrier(&mut self) -> bool {
        if self.d.wake_gen == self.d.barrier.gen {
            return false;
        }
        self.d.wake_all();
        self.d.wake_gen = self.d.barrier.gen;
        true
    }

    /// After the tick body of data queue `q` at cycle `c`: a cycle before which no later tick body of the queue can
    /// act or report another wait, as the queue and the barrier stand (a lower bound). The body acts when its
    /// complete head parcel is due and not held, a Repeat iteration or an AGT reference is due and the queue is not
    /// parked, or an AGT configuration is open. A due parcel that is held at `c` stays held, for the same reason,
    /// until a hold condition can change: a busy-until, path or tail cycle, a release cycle (or a change of the
    /// queue's state or of the barrier, which recompute this). With it: the IQ holds text whose head parcel is
    /// incomplete.
    pub(crate) fn wake_of(&self, q: usize, c: u64) -> (u128, bool) {
        let d = &self.d;
        let b = &d.barrier;
        let mut park_end = 0;
        if !b.quiet() {
            for k in [q, 144] {
                if b.parked_since[k].is_some() {
                    park_end = park_end.max(b.release_at[k].unwrap_or(u128::MAX));
                }
            }
        }
        let mut w = u128::MAX;
        for p in 0..2 {
            let a = d.agt_ref(q, p);
            match a.state {
                AgtState::Config => w = 0,
                AgtState::Run => w = w.min(a.next.max(park_end)),
                AgtState::Idle => {}
            }
        }
        if let Some(r) = &d.queue_repeat[q] {
            w = w.min(r.next.max(park_end));
        }
        let mut incomplete = false;
        if self.iq.len[q] > 0 {
            let (complete, _, nops) = self.iq.head_brief(q);
            if complete {
                let e = earliest_after_nops(&self.iq, q, nops);
                w = w.min(if e > c.into() { e } else { self.hold_change_after(q, c) });
            }
            incomplete = !complete;
        }
        (w, incomplete)
    }

    /// The first cycle after `c` at which a hold condition of queue `q` that depends on time can change.
    fn hold_change_after(&self, q: usize, c: u64) -> u128 {
        let d = &self.d;
        let b = &d.barrier;
        // A release cycle is set only for a parked queue.
        let (r0, r1, r2) = if b.quiet() {
            (u128::MAX, u128::MAX, u128::MAX)
        } else {
            (
                b.release_at[q].unwrap_or(u128::MAX),
                b.release_at[144].unwrap_or(u128::MAX),
                b.hw_release[q].unwrap_or(u128::MAX),
            )
        };
        [
            d.queue_busy_until[q],
            d.path_busy_until[2 * q],
            d.path_busy_until[2 * q + 1],
            d.path_tail_until[2 * q],
            d.path_tail_until[2 * q + 1],
            r0,
            r1,
            r2,
        ]
        .into_iter()
        .filter(|&t| t > c.into())
        .min()
        .unwrap_or(u128::MAX)
    }

    /// `TSP_ISS_INDEX_CHECK`: the calendar gave every queue whose wake cycle is due at `c`.
    fn check_calendar(&self, c: u64, due: &[u64; 3]) {
        assert_eq!(
            self.d.barrier.parked_counter(),
            self.d.barrier.parked_count_scan(),
            "barrier: parked counter at cycle {c}"
        );
        for q in 0..=144 {
            assert!(self.iq.head_matches_scan(&self.isa, q), "event core: parsed head of queue {q} at cycle {c}");
        }
        let mut fills = [0u64; 3];
        for (q, f) in self.iq.fill.iter().enumerate() {
            fills[q >> 6] |= (!f.is_empty() as u64) << (q & 63);
        }
        assert_eq!(fills, self.iq.fill_queues(), "event core: fill set at cycle {c}");
        for q in 0..144 {
            let listed = due[q >> 6] >> (q & 63) & 1 == 1;
            assert!(
                listed || self.d.wake[q] > u128::from(c),
                "event core: queue {q} due at cycle {c} but not in the calendar"
            );
            let inc = self.d.incomplete_set[q >> 6] >> (q & 63) & 1 == 1;
            assert_eq!(inc, self.d.incomplete[q], "event core: incomplete set of queue {q} at cycle {c}");
        }
    }

    /// `TSP_ISS_INDEX_CHECK`: queue `q`, skipped at cycle `c`, has a wake cycle that its state still gives, and by
    /// the tick body's own conditions would have done nothing.
    fn check_skip(&self, q: usize, c: u64) {
        let w = self.wake_of(q, c);
        assert_eq!((self.d.wake[q], self.d.incomplete[q]), w, "event index: queue {q} at cycle {c}: stale wake cycle");
        let parked = self.d.barrier.queue_parked(q, c);
        let (complete, _, nops) = self.iq.head_brief(q);
        let due = complete && u128::from(c) >= earliest_after_nops(&self.iq, q, nops) && {
            let h = self.iq.head(q);
            !self.parcel_held(q, c, h.recs, h.items)
        };
        let repeat = self.d.queue_repeat[q].as_ref().is_some_and(|r| r.next <= u128::from(c)) && !parked;
        let agt = (0..2).any(|p| {
            let a = self.d.agt_ref(q, p);
            a.state == AgtState::Config || super::agt::agt_due_lean(a, c, parked).is_some()
        });
        assert!(!(due || repeat || agt), "event index: queue {q} skipped at cycle {c} but can act");
    }
}

#[cfg(all(target_arch = "x86_64", feature = "native-mem-prefetch"))]
#[inline]
fn sram_hint_tail(address: usize, size: usize) -> Option<usize> {
    let last = size.checked_sub(1)?;
    ((address & 63) + (last & 63) >= 64).then_some(last)
}

#[cfg(all(test, target_arch = "x86_64", feature = "native-mem-prefetch"))]
#[path = "../../tests/private/sram_prefetch_lines.rs"]
mod physical_line_tests;
