//! Event queries for the harness (`postlude/harness_api.sail`). Idle-cycle skipping must
//! preserve architectural state and required ordered effects. Optional wait observations
//! sample visited cycles and may differ between literal and skipped execution.
use super::agt::AgtState;
use super::classify::CycleRecs;
use super::{pop_bit, EXEC_SLOTS, PENDING_SLOTS};
use crate::frontend::earliest_after_nops;
use crate::Machine;
use ots_isa::{Code as Diag, Insn};

/// The earlier of two optional cycles (`None`: no event).
// model: postlude/harness_api.sail earliest
fn earliest(a: Option<u128>, b: Option<u128>) -> Option<u128> {
    match (a, b) {
        (None, b) => b,
        (a, None) => a,
        (Some(a), Some(b)) => Some(a.min(b)),
    }
}

/// The first cycle `t > c` with `t mod m == k`.
fn next_slot(c: u128, k: usize, m: usize) -> u128 {
    // c is an actual u64 tick widened before adding; its next slot may exceed u64::MAX.
    let (c1, m) = (c + 1, m as u128);
    c1 + (k as u128 + m - c1 % m) % m
}

impl<const L: usize> Machine<L> {
    /// The earliest cycle `t >= base` at which the head parcel is not held, assuming nothing else
    /// changes (`None`: never, it waits for another event). The hold conditions that depend on time
    /// only end at the busy-until, tail and release cycles, so these are the only candidates.
    // model: postlude/tick.sail parcel_dispatchable
    /// `bound`: a cycle another event already holds, so only an answer below it matters (`None`: any).
    fn parcel_dispatchable(
        &self,
        q: usize,
        base: u128,
        bound: Option<u128>,
        r: &CycleRecs,
        items: &[Result<Insn, Diag>],
    ) -> Option<u128> {
        let d = &self.d;
        // `base` stands in for a release cycle the barrier does not have: it is a candidate already, so
        // repeating it changes no minimum. No candidate is below `base`, so `base` itself ends the search.
        let cands = [
            base,
            d.queue_busy_until[q],
            d.path_busy_until[2 * q],
            d.path_busy_until[2 * q + 1],
            d.path_tail_until[2 * q],
            d.path_tail_until[2 * q + 1],
            d.barrier.release_at[q].unwrap_or(base),
            d.barrier.release_at[144].unwrap_or(base),
            d.barrier.hw_release[q].unwrap_or(base),
        ];
        let mut best = bound;
        for t in cands {
            if t >= base && best.is_none_or(|b| t < b) && !self.parcel_held(q, t, r, items) {
                best = Some(t);
                if t == base {
                    break;
                }
            }
        }
        if best == bound {
            return None;
        }
        best
    }

    /// After a tick at cycle `c`: the earliest cycle `> c` at which the complete head parcel of data
    /// queue `q` can dispatch.
    // model: postlude/tick.sail parcel_event
    pub fn parcel_event(&self, q: usize, c: u128) -> Option<u128> {
        self.parcel_event_below(q, c, None)
    }

    /// [`Machine::parcel_event`] when only an answer below `bound` matters: `None` for a queue that cannot beat it
    /// (`model_next_event` takes a minimum, so a candidate at or above the best found changes nothing).
    fn parcel_event_below(&self, q: usize, c: u128, bound: Option<u128>) -> Option<u128> {
        if self.iq.len[q] == 0 {
            return None;
        }
        let h = self.iq.head(q);
        if !h.complete {
            return None;
        }
        let base = (c + 1).max(earliest_after_nops(&self.iq, q, h.nops));
        if bound.is_some_and(|b| base >= b) {
            return None; // no candidate of this queue is below `base`
        }
        self.parcel_dispatchable(q, base, bound, h.recs, h.items)
    }

    // model: postlude/tick.sail control_event
    pub fn control_event(&self, c: u128) -> Option<u128> {
        let (complete, _, nops) = self.iq.head_brief(144);
        complete.then(|| (c + 1).max(earliest_after_nops(&self.iq, 144, nops)))
    }

    /// True when nothing is left to happen: no commit is pending, no fetched text is on its way,
    /// and no queue has Repeat iterations, AGT references or a head parcel left to dispatch, except
    /// queues parked by a Sync no Notify has released (they never resume).
    // model: postlude/tick.sail model_idle; postlude/harness_api.sail model_quiescent
    pub fn model_idle(&self) -> bool {
        let d = &self.d;
        if !d.stream_fetches.is_empty()
            || !d.local_fetches.is_empty()
            || d.pending.iter().any(|s| !s.is_empty())
            || d.execute_ring.iter().any(|s| !s.is_empty())
        {
            return false;
        }
        for q in 0..144 {
            let held = d.barrier.stuck(q) || d.barrier.stuck(144);
            if (d.agts[q].state != AgtState::Idle || d.agts_w[q].state != AgtState::Idle || d.queue_repeat[q].is_some())
                && !held
            {
                return false;
            }
            if self.parcel_event(q, u128::from(d.last_tick)).is_some() {
                return false;
            }
        }
        if self.control_event(u128::from(d.last_tick)).is_some() {
            return false;
        }
        self.iq.fill.iter().all(|f| f.is_empty())
    }

    /// A queue whose pending Sync has no Notify yet (it waits for a record, not for time).
    // model: postlude/harness_api.sail held
    fn held(&self, q: usize) -> bool {
        self.d.barrier.stuck(q) || self.d.barrier.stuck(144)
    }

    /// The first cycle at or after `t` at which queue `q` (or the control channel) is not parked
    /// by a Sync whose release cycle is already known.
    // model: postlude/harness_api.sail unparked_from
    fn unparked_from(&self, q: usize, t: u128) -> u128 {
        let mut u = t;
        for _pass in 0..2 {
            for k in [q, 144] {
                if let (Some(s), Some(r)) = (self.d.barrier.parked_since[k], self.d.barrier.release_at[k]) {
                    if u128::from(s) <= u && u < r {
                        u = r;
                    }
                }
            }
        }
        u
    }

    /// The first epoch boundary (a multiple of `timing.c2c.hac_period`) at or after cycle `c`.
    // model: postlude/tick.sail next_boundary
    fn next_boundary(&self, c: u128) -> u128 {
        let p = u128::from(self.geom.hac_period);
        c + (p - c % p) % p
    }

    /// The earliest epoch boundary at which a queue with a DeskewRecv in its polling loop can find its vector (`None`: none):
    /// the boundary at or after the arrival of the oldest vector the harness knows on its link and after the last tick, moved to
    /// the first boundary from the release of a barrier that parks the queue there. A queue whose link has no vector
    /// waits for the harness to deliver one. Asked between ticks, by the harness (`next_time`).
    // model: postlude/harness_api.sail model_poll_event
    pub fn poll_event(&self) -> Option<u128> {
        let d = &self.d;
        if d.poll_n == 0 {
            return None;
        }
        let mut best: Option<u128> = None;
        for q in 0..crate::geom::DATA_QUEUES {
            let Some(p) = d.queue_poll[q] else { continue };
            let Some(a) = self.deskewrecv_next_arrival(p.link) else { continue };
            if self.held(q) {
                continue;
            }
            let mut b = self.next_boundary(a.max(u128::from(d.last_tick) + 1));
            if d.barrier.queue_parked(q, b) {
                b = self.next_boundary(self.unparked_from(q, b));
            }
            best = earliest(best, Some(b));
        }
        best
    }

    /// After `tick(c)`: the earliest cycle `t > c` at which `tick(t)` can do something (`None`:
    /// none). The candidates are a pending commit, an element waiting to execute (dispatch +
    /// d_skew), fetched text that enters an IQ (the head of a fill FIFO, from its ready cycle on,
    /// then every cycle), the head parcel of a queue that can dispatch (not before its earliest
    /// cycle, and when nothing holds it any more: `parcel_event`), an open AGT configuration on
    /// either dispatch path (it ends at the first cycle in which the queue dispatches no
    /// configuration instruction, so every cycle counts), the next AGT reference and the next
    /// Repeat iteration (delayed to the release of a known barrier window; work held by an
    /// unreleased Sync waits for a Notify).
    // model: postlude/harness_api.sail model_next_event
    pub fn model_next_event(&self, c: u64) -> Option<u128> {
        let c = u128::from(c);
        if !self.d.index_on {
            return self.model_next_event_scan(c);
        }
        self.d.ev_begin_query(c);
        let t = self.model_next_event_indexed(c);
        if self.d.index_check {
            assert_eq!(t, self.model_next_event_scan(c), "event index: model_next_event after cycle {c}");
        }
        t
    }

    /// Data queue `q`'s part of `model_next_event` after a tick at `c`: its AGTs, its Repeat and its head parcel.
    fn queue_event(&self, q: usize, c: u128) -> Option<u128> {
        let d = &self.d;
        let mut best: Option<u128> = d
            .local_fetches
            .iter()
            .map(|f| f.ready)
            .chain(d.stream_fetches.iter().map(|f| u128::from(f.issue.exec_cycle()) + 1))
            .map(|due| (c + 1).max(due))
            .min();
        for p in 0..2 {
            let a = d.agt_ref(q, p);
            if a.state == AgtState::Config {
                return Some(c + 1);
            } else if a.state == AgtState::Run && !self.held(q) {
                best = earliest(best, Some(self.unparked_from(q, a.next.max(c + 1))));
            }
        }
        if let Some(r) = &d.queue_repeat[q] {
            if !self.held(q) {
                best = earliest(best, Some(self.unparked_from(q, r.next.max(c + 1))));
            }
        }
        if best == Some(c + 1) {
            return best;
        }
        earliest(best, self.parcel_event_below(q, c, best))
    }

    /// [`Machine::model_next_event`] from the event structures: the rings' and the fill FIFOs' occupancy sets, and a
    /// min-tree of the queues' parts. A queue's part is computed again only when the queue acted or took text into
    /// an incomplete head (`ev_invalidate`), when the barrier changed (`ev_gen`), or when the cycle reached it.
    /// Until then it holds: each part is the first cycle at or after `max(c + 1, x)` at which nothing holds the
    /// queue, and a hold ends only at a cycle of the queue's state, so a later `c` below the answer finds the same
    /// answer.
    fn model_next_event_indexed(&self, c: u128) -> Option<u128> {
        let d = &self.d;
        if d.cfg_tick == c + 1 {
            return Some(c + 1); // a configuration was open at the end of this tick (`close_config_early`)
        }
        let mut best: Option<u128> = d
            .local_fetches
            .iter()
            .map(|f| f.ready)
            .chain(d.stream_fetches.iter().map(|f| u128::from(f.issue.exec_cycle()) + 1))
            .map(|due| (c + 1).max(due))
            .min();
        let mut occ = d.pending_occ.rotate_right(((c + 1) % PENDING_SLOTS as u128) as u32);
        while occ != 0 {
            let i = occ.trailing_zeros() as u128;
            occ &= occ - 1;
            if !d.pending[((c + 1 + i) % PENDING_SLOTS as u128) as usize].is_empty() {
                best = earliest(best, Some(c + 1 + i));
                break;
            }
        }
        let mut occ = d.exec_occ.rotate_right(((c + 1) % EXEC_SLOTS as u128) as u32);
        while occ != 0 {
            let i = occ.trailing_zeros() as u128;
            occ &= occ - 1;
            if !d.execute_ring[((c + 1 + i) % EXEC_SLOTS as u128) as usize].is_empty() {
                best = earliest(best, Some(c + 1 + i));
                break;
            }
        }
        if best == Some(c + 1) {
            return best;
        }
        let mut fills = self.iq.fill_queues();
        while let Some(n) = pop_bit(&mut fills) {
            if let Some(seg) = self.iq.fill[n].front() {
                best = earliest(best, Some((c + 1).max(seg.ready)));
            }
        }
        best = earliest(best, self.control_event(c));
        if best == Some(c + 1) {
            return best;
        }
        let mut dirty = d.ev_take_dirty();
        while let Some(n) = pop_bit(&mut dirty) {
            d.ev_set(n, self.queue_event(n, c).unwrap_or(u128::MAX));
        }
        loop {
            let (m, q) = d.ev_min();
            if m > c {
                return earliest(best, (m != u128::MAX).then_some(m));
            }
            d.ev_set(q, self.queue_event(q, c).unwrap_or(u128::MAX));
        }
    }

    /// [`Machine::model_next_event`] by the full scan of the queues: the reference the index is checked against.
    pub fn model_next_event_scan(&self, c: u128) -> Option<u128> {
        let d = &self.d;
        // The result is a minimum over the candidates, and none of them is below `c + 1`: once a candidate is
        // `c + 1` nothing else can lower it, and a candidate at or above the best found changes nothing. The cheap
        // candidates come first so that the per-queue scan runs against the tightest bound.
        let mut best: Option<u128> = d
            .local_fetches
            .iter()
            .map(|f| f.ready)
            .chain(d.stream_fetches.iter().map(|f| u128::from(f.issue.exec_cycle()) + 1))
            .map(|due| (c + 1).max(due))
            .min();
        for k in 0..PENDING_SLOTS {
            if !d.pending[k].is_empty() {
                best = earliest(best, Some(next_slot(c, k, PENDING_SLOTS)));
            }
        }
        for k in 0..EXEC_SLOTS {
            if !d.execute_ring[k].is_empty() {
                best = earliest(best, Some(next_slot(c, k, EXEC_SLOTS)));
            }
        }
        for q in 0..=144 {
            if let Some(seg) = self.iq.fill[q].front() {
                best = earliest(best, Some((c + 1).max(seg.ready)));
            }
        }
        best = earliest(best, self.control_event(c));
        if best == Some(c + 1) {
            return best;
        }
        for q in 0..144 {
            for p in 0..2 {
                let a = d.agt_ref(q, p);
                if a.state == AgtState::Config {
                    best = earliest(best, Some(c + 1));
                } else if a.state == AgtState::Run && !self.held(q) {
                    best = earliest(best, Some(self.unparked_from(q, a.next.max(c + 1))));
                }
            }
            if let Some(r) = &d.queue_repeat[q] {
                if !self.held(q) {
                    best = earliest(best, Some(self.unparked_from(q, r.next.max(c + 1))));
                }
            }
            if best == Some(c + 1) {
                return best;
            }
            best = earliest(best, self.parcel_event_below(q, c, best));
        }
        best
    }
}
