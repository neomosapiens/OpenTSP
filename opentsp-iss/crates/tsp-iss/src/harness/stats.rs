//! The counters behind `--stats`: an [`Observer`] that counts what the hooks report and
//! forwards every event to the observer it wraps, so the trace and the results are the wrapped
//! observer's.
//!
//! Wait cycles are counted per queue and hold reason, in cycles, not in ticks: the machine skips the
//! cycles at which nothing happens, so a wait observed at tick `c` that goes on at the next tick `c'`
//! with the same queue still waiting counts the cycles in between too, a wait that ends in a
//! dispatch at `d` counts the cycles up to `d - 1`, and the first wait counts the cycles since the
//! parcel was due. The counts are therefore the same when
//! `--tick-every-cycle` ticks the idle cycles.
use crate::geom::QUEUES;
use crate::observe::{Hold, Observer, Row};
use ots_isa::Code;
use std::collections::BTreeMap;

/// The hold reasons in report order, with their names in the JSON.
pub const REASONS: [(Hold, &str); 7] = [
    (Hold::Parked, "parked"),
    (Hold::Busy, "busy"),
    (Hold::Path, "path"),
    (Hold::Loop, "loop"),
    (Hold::Repeat, "repeat"),
    (Hold::Configuration, "configuration"),
    (Hold::IncompleteText, "incomplete_text"),
];

/// The index of `h` in [`REASONS`].
fn reason(h: Hold) -> usize {
    REASONS.iter().position(|(r, _)| *r == h).expect("every hold is listed")
}

/// The queue slot of a queue id (the control channel, 0xffff, is the last).
pub fn slot(queue: u16) -> usize {
    if queue == 0xffff {
        QUEUES - 1
    } else {
        queue as usize
    }
}

/// The queue id of a slot.
pub fn queue_of(slot: usize) -> u16 {
    if slot == QUEUES - 1 {
        0xffff
    } else {
        slot as u16
    }
}

/// What one chip's run did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Counters {
    /// Parcels dispatched, per queue slot.
    pub dispatches: Vec<u64>,
    /// Wait cycles, per queue slot and reason ([`REASONS`] order).
    pub waits: Vec<[u128; 7]>,
    /// Bytes of text fetched.
    pub fetch_bytes: u64,
    /// Diagnostics per code.
    pub by_code: BTreeMap<&'static str, u64>,
    /// Per queue: the cycle and reason of the last wait, while it goes on.
    pending: Vec<Option<(u64, usize)>>,
}

impl Default for Counters {
    fn default() -> Counters {
        Counters {
            dispatches: vec![0; QUEUES],
            waits: vec![[0; 7]; QUEUES],
            fetch_bytes: 0,
            by_code: BTreeMap::new(),
            pending: vec![None; QUEUES],
        }
    }
}

impl Counters {
    /// The wait cycles of each reason over all queues.
    pub fn wait_totals(&self) -> [u128; 7] {
        let mut t = [0; 7];
        for w in &self.waits {
            for (a, b) in t.iter_mut().zip(w) {
                *a += b;
            }
        }
        t
    }
}

/// An observer that counts into `counts` and forwards to `inner`.
pub struct StatsObserver<'a, O: Observer + ?Sized> {
    pub inner: &'a mut O,
    pub counts: &'a mut Counters,
}

impl<O: Observer + ?Sized> Observer for StatsObserver<'_, O> {
    fn srf(&mut self, c: u64, row: Row, stream: u8, gap: u8, v: &[u8]) {
        self.inner.srf(c, row, stream, gap, v);
    }

    fn sram(&mut self, c: u64, slice: u8, addr: u32, v: &[u8]) {
        self.inner.sram(c, slice, addr, v);
    }

    fn diag(&mut self, c: u64, queue: u16, code: Code) {
        *self.counts.by_code.entry(code.as_str()).or_insert(0) += 1;
        self.inner.diag(c, queue, code);
    }

    fn lint(&mut self, c: u64, queue: u16, code: &'static str) {
        self.inner.lint(c, queue, code);
    }

    fn dispatch(&mut self, c: u64, queue: u16) {
        let s = slot(queue);
        self.counts.dispatches[s] += 1;
        if let Some((from, r)) = self.counts.pending[s].take() {
            self.counts.waits[s][r] += u128::from(c.saturating_sub(1).saturating_sub(from));
        }
        self.inner.dispatch(c, queue);
    }

    fn wait(&mut self, c: u64, queue: u16, why: Hold, since: u64) {
        let s = slot(queue);
        let r = reason(why);
        match self.counts.pending[s] {
            Some((from, was)) => self.counts.waits[s][was] += u128::from(c - from - 1),
            None => self.counts.waits[s][r] += u128::from(c - since.min(c)),
        }
        self.counts.waits[s][r] += 1;
        self.counts.pending[s] = Some((c, r));
        self.inner.wait(c, queue, why, since);
    }

    fn fetch(&mut self, c: u64, queue: u16, bytes: u32) {
        self.counts.fetch_bytes += bytes as u64;
        self.inner.fetch(c, queue, bytes);
    }

    fn host_dma(&mut self, c: u64, what: crate::observe::HostDma) {
        self.inner.host_dma(c, what);
    }

    fn host_irq(&mut self, c: u64, vector: u8) {
        self.inner.host_irq(c, vector);
    }

    fn host_csr(&mut self, c: u64, addr: u64, value: u64) {
        self.inner.host_csr(c, addr, value);
    }

    fn host_csr_remote(&mut self, c: u64, chip: u16, addr: u64, value: u64) {
        self.inner.host_csr_remote(c, chip, addr, value);
    }

    fn ecc(&mut self, e: &crate::ecc::EccEvent) {
        self.inner.ecc(e);
    }
    fn link_fec(&mut self, c: u64, link: u8, unc: bool, flit: u8) {
        self.inner.link_fec(c, link, unc, flit);
    }
    fn mxm_corr(&mut self, c: u64, queue: u16, plane: u8, lane: u16) {
        self.inner.mxm_corr(c, queue, plane, lane);
    }
    fn scrub(&mut self, c: u64, slice: u8, addr: u32, sl: u8, unc: bool) {
        self.inner.scrub(c, slice, addr, sl, unc);
    }

    fn host_pkt(&mut self, c: u64, ctrl: bool, link: u8) {
        self.inner.host_pkt(c, ctrl, link);
    }

    fn site(&mut self, chip: u16, cycle: u64, q: usize, event: &str, detail: std::fmt::Arguments<'_>) {
        self.inner.site(chip, cycle, q, event, detail);
    }
    fn tracks_waits(&self) -> bool {
        true
    }
}

/// The IQ levels of a chip's queues that ever held text: queue id, fewest units held after a
/// dispatch, most held.
pub fn iq_levels<const L: usize>(m: &crate::machine::Machine<L>) -> Vec<(u16, usize, usize)> {
    (0..QUEUES)
        .filter_map(|s| {
            let st = m.queue_state(queue_of(s))?;
            (st.iq_max > 0).then_some((queue_of(s), st.iq_min, st.iq_max))
        })
        .collect()
}

/// The `replay` object of the `--stats` document: whether this build has schedule replay, and the counters of the
/// process's bounded schedule cache (`driver::replay::ReplayCache`): the runs it served (`hits`), the runs it did not
/// (`misses`), the replays that diverged from their recording and took the normal driver (`divergences`), the schedules
/// it dropped for the budget (`evictions`), and what it retains now against its limits.
pub fn replay_json(enabled: bool, c: &crate::driver::replay::ReplayCounters) -> serde_json::Value {
    serde_json::json!({
        "enabled": enabled,
        "hits": c.hits,
        "misses": c.misses,
        "divergences": c.divergences,
        "evictions": c.evictions,
        "recordings": c.recordings,
        "oversize": c.oversize,
        "refused": c.refused,
        "shadow_checks": c.shadow_checks,
        "shadow_mismatches": c.shadow_mismatches,
        "param_served": c.param_served,
        "param_refused": c.param_refused,
        "param_fallback": c.param_fallback,
        "param_recordings": c.param_recordings,
        "param_first_sight": c.param_first_sight,
        "param_small": c.param_small,
        "param_unbuildable": c.param_unbuildable,
        "schedules": c.schedules,
        "retained_bytes": c.retained_bytes,
        "budget_bytes": c.budget_bytes,
        "max_schedules": c.max_schedules,
    })
}
