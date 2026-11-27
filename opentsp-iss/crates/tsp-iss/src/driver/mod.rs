//! The cycle driver (`postlude/tick.sail`, `postlude/harness_api.sail`).
//!
//! The TSP has no fetch/decode/execute loop of its own: 144 queues issue and results land at
//! later cycles. Every queue (and the control channel, 144) has an IQ (`frontend`) of text units;
//! the driver takes each queue's head parcel, dispatches it when it is due and nothing holds it,
//! and ticks the cycles at which something can happen (`events`).
//!
//! One module per group of Sail functions, each function tagged `// model: postlude/tick.sail barrier_sync`.
//! The state is [`Driver`] (the `register`s of `tick.sail`); the functions are methods of
//! [`Machine`](crate::Machine) so they can reach the IQs, the units and the observer's reports in
//! the order the model does. It covers the front end, the barrier, Repeat, the execute ring, the
//! control channel and `dispatch`; the AGT records (`agt`) are used by MEM dispatch
//! (`dispatch_slots`). The commit ring is `commit` (`commit_due` with SRF-002, the thread merge and the
//! SEL+TSP OR); `flush` is `unit_state_flush` (the MXM planes' and the SXM maps' staging).
pub mod agt;
pub mod barrier;
pub mod classify;
pub mod commit;
pub mod dispatch;
pub mod events;
pub mod execute;
pub mod flush;
pub mod phase;
pub mod repeat;
pub mod replay;
#[cfg(feature = "replay")]
pub mod replay_param;
pub mod tick;

use crate::geom::{DATA_QUEUES, QUEUES};
use crate::observe::Observer;
use agt::{Agt, AgtState};
use barrier::Barrier;
use commit::Commit;
use execute::ExecElement;
use ots_isa::{Code as Diag, Insn};
use repeat::{PollState, RepeatState};
use std::cell::Cell;

/// Slots of the commit ring (`pending`), of the execute ring.
pub const PENDING_SLOTS: usize = 128;
pub const EXEC_SLOTS: usize = 16;
/// Slots of the wake calendar; leaves of the event min-tree (a power of two at least `DATA_QUEUES`).
pub const CAL_SLOTS: usize = 256;
const EV_LEAVES: usize = 256;
/// The 144 data queues as a bit set.
pub const ALL_DATA: [u64; 3] = [!0, !0, (1 << 16) - 1];

/// Path hold through the issue cycle (`path_hold_until`).
// model: postlude/tick.sail path_hold_until
pub fn path_hold_until(c: u64) -> u128 {
    u128::from(c) + 1
}

/// IfetchM hold through the second read (`fetchm_hold_until`).
// model: postlude/tick.sail fetchm_hold_until
pub fn fetchm_hold_until(c: u64) -> u128 {
    u128::from(c) + 2
}

/// Bank note of an IfetchM's second word, or none when the word is outside the segment or slice.
// model: postlude/tick.sail fetchm_bank_note
pub fn fetchm_bank_note(i: &Insn, c: u64, seg: u64, words_per_slice: u64) -> Option<(u128, u32)> {
    match i {
        Insn::IfetchM { addr } => {
            let e = *addr as u64 + 1;
            if e < seg && e < words_per_slice {
                Some((u128::from(c) + 1, e as u32))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Why a due head parcel waits (observation site name), in priority order.
// model: postlude/tick.sail wait_reason
pub fn wait_reason(host: bool, parked: bool, poll: bool, repeating: bool) -> &'static str {
    if host {
        "wait-host"
    } else if parked {
        "wait-barrier"
    } else if poll {
        "wait-poll"
    } else if repeating {
        "wait-repeat"
    } else {
        "wait-resource"
    }
}

/// An open AGT configuration ends when the queue dispatches no configuration record this cycle.
// model: postlude/tick.sail config_ends
pub fn config_ends(st: AgtState, go: bool, cfg_in_cycle: bool) -> bool {
    st == AgtState::Config && !(go && cfg_in_cycle)
}

/// A DeskewRecv looks for its vector at each epoch boundary after its dispatch, when not parked.
// model: postlude/tick.sail poll_epoch
pub fn poll_epoch(since: u64, c: u64, period: u64, parked: bool) -> bool {
    c > since && period != 0 && c.is_multiple_of(period) && !parked
}

/// The lowest set bit at or above `from` of a 144-bit set.
pub fn next_bit(set: &[u64; 3], from: usize) -> Option<usize> {
    let mut w = from >> 6;
    if w >= 3 {
        return None;
    }
    let mut bits = set[w] & (!0u64 << (from & 63));
    loop {
        if bits != 0 {
            let n = (w << 6) + bits.trailing_zeros() as usize;
            return (n < 145).then_some(n);
        }
        w += 1;
        if w >= 3 {
            return None;
        }
        bits = set[w];
    }
}

/// Remove the lowest set bit of a 145-bit set and return its index.
#[inline(always)]
pub fn pop_bit(set: &mut [u64; 3]) -> Option<usize> {
    for (w, word) in set.iter_mut().enumerate() {
        let b = *word;
        if b != 0 {
            *word = b & (b - 1);
            return Some((w << 6) + b.trailing_zeros() as usize);
        }
    }
    None
}

/// The bits of `set` at or above `from`.
pub fn bits_from(set: &[u64; 3], from: usize) -> [u64; 3] {
    let mut r = *set;
    for (w, x) in r.iter_mut().enumerate() {
        let lo = w << 6;
        if from >= lo + 64 {
            *x = 0;
        } else if from > lo {
            *x &= !0u64 << (from - lo);
        }
    }
    r
}

/// Captured first half of a two-cycle local instruction fetch.
#[derive(Debug)]
pub struct LocalFetch {
    pub queue: usize,
    pub ready: u128,
    pub addr: u32,
    pub first: [u8; crate::geom::MAX_LANES],
    pub checks: Vec<u16>,
}

#[derive(Debug)]
pub(crate) struct StreamFetch {
    pub issue: crate::slices::Issue,
    pub stream: u32,
    pub first: [u8; crate::geom::MAX_LANES],
    pub bad: u32,
}

/// The registers of `tick.sail`.
#[derive(Debug)]
pub struct Driver<const L: usize = { crate::geom::MAX_LANES }> {
    pub barrier: Barrier,
    /// Per-queue busy-until cycle (index 144: the control channel).
    pub queue_busy_until: [u128; QUEUES],
    /// Per-path busy-until cycle of a MEM queue (index `2q + path`).
    pub path_busy_until: [u128; 2 * DATA_QUEUES],
    pub(crate) stream_fetches: Vec<StreamFetch>,
    pub local_fetches: Vec<LocalFetch>,
    pub local_fetch_busy: [u128; 88],
    pub local_fetch_bank: [Option<(u128, u32)>; DATA_QUEUES],
    /// The trailing waits of a finished AGT loop.
    pub path_tail_until: [u128; 2 * DATA_QUEUES],
    /// One AGT per dispatch path: the R path's and the W path's.
    pub agts: [Agt; DATA_QUEUES],
    pub agts_w: [Agt; DATA_QUEUES],
    /// Configuration records each path's open configuration has received (index `2q + path`).
    pub agt_records: [u64; 2 * DATA_QUEUES],
    pub queue_last: [Option<Insn>; DATA_QUEUES],
    /// The cycle the queue last issued `queue_last`.
    pub queue_last_at: [u64; DATA_QUEUES],
    /// The patrol scrubber (`scrub.rs`): the next scrub cycle not yet run and the scrub pointer of each MEM slice.
    pub scrub_next: u128,
    pub scrub_ptr: [u32; 88],
    pub scrub_pending: [Option<Box<crate::scrub::ScrubWrite>>; 88],
    pub queue_repeat: [Option<RepeatState>; DATA_QUEUES],
    /// DeskewRecv (`postlude/tick.sail` `queue_poll`): the instruction that holds queue `q` until its vector has arrived at an
    /// epoch boundary. `poll_n` entries are set.
    pub queue_poll: [Option<PollState>; DATA_QUEUES],
    pub poll_n: u32,
    pub execute_ring: [Vec<ExecElement>; EXEC_SLOTS],
    pub pending: [Vec<Commit<L>>; PENDING_SLOTS],
    pub last_tick: u64,
    /// The event index (not a register of the model). `wake[q]`: no tick before it can make data queue `q` act; a
    /// function of the queue's head parcel, Repeat and AGTs, so whatever writes them must call [`Driver::touch`]
    /// or recompute it.
    pub wake: [u128; DATA_QUEUES],
    /// The IQ holds text and its head parcel is incomplete (valid with `wake`): a skipped queue's only wait report.
    pub incomplete: [bool; DATA_QUEUES],
    /// `incomplete` as a bit set.
    pub incomplete_set: [u64; 3],
    /// The barrier generation `wake` was computed at: a change of the barrier sets every `wake` to 0.
    pub wake_gen: u64,
    /// The wake calendar: slot `t mod 256` holds the queues whose `wake` was set to `t`, for `cal_next <= t <
    /// cal_next + 256`; `cal_occ` marks the non-empty slots, `cal_now` the queues due before `cal_next`, `cal_far`
    /// the wakes beyond the ring (`cal_far_min` the earliest). A queue may also sit at an older wake cycle: the
    /// tick skips it when its `wake` is later.
    cal: Box<[[u64; 3]; CAL_SLOTS]>,
    cal_occ: [u64; CAL_SLOTS / 64],
    cal_now: [u64; 3],
    cal_far: Vec<(u128, usize)>,
    cal_far_min: u128,
    cal_next: u128,
    /// Queue `q`'s part of `model_next_event` (`u128::MAX`: none) at leaf `EV_LEAVES + q` of a min-tree, valid
    /// unless `q` is in `ev_dirty` or the barrier left `ev_gen`.
    ev_tree: Box<[Cell<u128>]>,
    ev_dirty: [Cell<u64>; 3],
    pub(crate) ev_gen: Cell<u64>,
    ev_query_at: Cell<Option<u128>>,
    /// Occupancy of the commit ring's and the execute ring's slots: a superset of the non-empty slots.
    pub pending_occ: u128,
    pub exec_occ: u16,
    /// `commit_due`'s scratch list (kept for its allocation).
    pub commit_order: Vec<u32>,
    /// `commit_due`'s duplicate detector: one bit per (row, stream, gap) target, all zero between calls.
    pub commit_seen: Vec<u64>,
    /// `TSP_ISS_INDEX_CHECK`: every tick and every `model_next_event` asserts the index against the full scan.
    pub index_check: bool,
    /// `TSP_ISS_INDEX=0`: the index is off and every tick and `model_next_event` scan all queues.
    pub index_on: bool,
    /// `TSP_ISS_FAST=0`: the common-case dispatch path is off (every parcel takes the general path).
    pub fast_on: bool,
    /// `TSP_ISS_FAST_CHECK`: the common-case path evaluates what it leaves out and asserts the answer it assumes.
    pub fast_check: bool,
    /// The run loop closes an AGT configuration that nothing can continue at the end of the tick that opened it
    /// (`Machine::close_config_early`); set by `Machine::run_until`, never by a direct `tick`.
    pub early_close: bool,
    /// The cycle a tick is due at only because a configuration was closed early (it was open then: `model_next_event`).
    pub cfg_tick: u128,
    /// `TSP_ISS_FAST_CHECK`: the configurations closed early and the cycle: the next tick must be at `cycle + 1`, and
    /// the visit it saves must find that nothing continues them.
    pub early_closed: Vec<(usize, usize, u64)>,
    /// How many parcels took the common-case dispatch, and how many configurations were closed early (the tests
    /// assert that the paths they check were taken).
    pub fast_dispatches: u64,
    pub early_closes: u64,
}

impl Driver {
    /// The registers of a machine at the widest vector: `Driver::with_lanes`.
    pub fn new(barrier_release_latency: u64) -> Driver {
        Driver::with_lanes(barrier_release_latency)
    }
}

impl<const L: usize> Driver<L> {
    // model: postlude/tick.sail reset (the driver's registers)
    pub fn with_lanes(barrier_release_latency: u64) -> Driver<L> {
        Driver {
            barrier: Barrier::with_latency(barrier_release_latency),
            queue_busy_until: [0; QUEUES],
            path_busy_until: [0; 2 * DATA_QUEUES],
            stream_fetches: Vec::new(),
            local_fetches: Vec::new(),
            local_fetch_busy: [0; 88],
            local_fetch_bank: [None; DATA_QUEUES],
            path_tail_until: [0; 2 * DATA_QUEUES],
            agts: [Agt::idle(); DATA_QUEUES],
            agts_w: [Agt::idle(); DATA_QUEUES],
            agt_records: [0; 2 * DATA_QUEUES],
            queue_last: [None; DATA_QUEUES],
            queue_last_at: [0; DATA_QUEUES],
            scrub_next: 0,
            scrub_ptr: [0; 88],
            scrub_pending: std::array::from_fn(|_| None),
            queue_repeat: [None; DATA_QUEUES],
            queue_poll: [None; DATA_QUEUES],
            poll_n: 0,
            execute_ring: std::array::from_fn(|_| Vec::new()),
            pending: std::array::from_fn(|_| Vec::new()),
            last_tick: 0,
            wake: [0; DATA_QUEUES],
            incomplete: [false; DATA_QUEUES],
            incomplete_set: [0; 3],
            wake_gen: 0,
            cal: Box::new([[0; 3]; CAL_SLOTS]),
            cal_occ: [0; CAL_SLOTS / 64],
            cal_now: ALL_DATA,
            cal_far: Vec::new(),
            cal_far_min: u128::MAX,
            cal_next: 0,
            ev_tree: (0..2 * EV_LEAVES).map(|_| Cell::new(u128::MAX)).collect(),
            ev_dirty: ALL_DATA.map(Cell::new),
            ev_gen: Cell::new(0),
            ev_query_at: Cell::new(None),
            pending_occ: 0,
            exec_occ: 0,
            commit_order: Vec::new(),
            commit_seen: Vec::new(),
            index_check: std::env::var_os("TSP_ISS_INDEX_CHECK").is_some_and(|v| !v.is_empty() && v != "0"),
            index_on: std::env::var_os("TSP_ISS_INDEX").is_none_or(|v| v != "0"),
            fast_on: std::env::var_os("TSP_ISS_FAST").is_none_or(|v| v != "0"),
            fast_check: std::env::var_os("TSP_ISS_FAST_CHECK").is_some_and(|v| !v.is_empty() && v != "0"),
            early_close: false,
            cfg_tick: u128::MAX,
            early_closed: Vec::new(),
            fast_dispatches: 0,
            early_closes: 0,
        }
    }

    /// Rebuild derived scheduling structures after an external architectural-state edit.
    pub(crate) fn rebuild_after_edit(&mut self) {
        self.barrier.rebuild_after_edit();
        self.wake = [0; DATA_QUEUES];
        self.incomplete = [false; DATA_QUEUES];
        self.incomplete_set = [0; 3];
        self.wake_gen = self.barrier.gen;
        self.cal.fill([0; 3]);
        self.cal_occ.fill(0);
        self.cal_now = ALL_DATA;
        self.cal_far.clear();
        self.cal_far_min = u128::MAX;
        self.cal_next = u128::from(self.last_tick);
        for (cell, bits) in self.ev_dirty.iter().zip(ALL_DATA) {
            cell.set(bits);
        }
        self.ev_tree.iter().for_each(|cell| cell.set(u128::MAX));
        self.ev_gen.set(self.barrier.gen);
        self.ev_query_at.set(None);
        self.pending_occ =
            self.pending.iter().enumerate().fold(0, |bits, (i, xs)| bits | (u128::from(!xs.is_empty()) << i));
        self.exec_occ =
            self.execute_ring.iter().enumerate().fold(0, |bits, (i, xs)| bits | (u16::from(!xs.is_empty()) << i));
        self.poll_n = self.queue_poll.iter().filter(|p| p.is_some()).count() as u32;
        self.commit_order.clear();
        self.commit_seen.fill(0);
    }

    /// Queue `q`'s head parcel, Repeat or AGTs may have changed: both index entries are recomputed.
    pub fn touch(&mut self, q: usize) {
        if q < DATA_QUEUES {
            self.wake[q] = 0;
            self.cal_now[q >> 6] |= 1 << (q & 63);
            self.ev_invalidate(q);
        }
    }

    /// Every queue's `wake` is 0 (the barrier changed).
    pub fn wake_all(&mut self) {
        self.wake = [0; DATA_QUEUES];
        self.cal_now = ALL_DATA;
    }

    /// Set queue `q`'s `wake` to `w` and enter it in the calendar.
    pub fn set_wake(&mut self, q: usize, w: u128) {
        self.wake[q] = w;
        self.cal_schedule(q, w);
    }

    /// Enter queue `q` in the calendar at cycle `w` (`u128::MAX`: never).
    pub fn cal_schedule(&mut self, q: usize, w: u128) {
        let bit = 1u64 << (q & 63);
        if w < self.cal_next {
            self.cal_now[q >> 6] |= bit;
        } else if w - self.cal_next < CAL_SLOTS as u128 {
            let s = (w % CAL_SLOTS as u128) as usize;
            self.cal[s][q >> 6] |= bit;
            self.cal_occ[s >> 6] |= 1 << (s & 63);
        } else if w != u128::MAX {
            self.cal_far.push((w, q));
            self.cal_far_min = self.cal_far_min.min(w);
        }
    }

    /// The queues the calendar holds for cycles up to `c` (ticks come in increasing order), removed from it.
    pub fn cal_take(&mut self, c: u64) -> [u64; 3] {
        let c = u128::from(c);
        if c >= self.cal_next {
            let n = c - self.cal_next + 1;
            if n >= CAL_SLOTS as u128 {
                self.cal_drain(0, CAL_SLOTS);
            } else {
                let lo = (self.cal_next % CAL_SLOTS as u128) as usize;
                let hi = lo + n as usize;
                if hi <= CAL_SLOTS {
                    self.cal_drain(lo, hi);
                } else {
                    self.cal_drain(lo, CAL_SLOTS);
                    self.cal_drain(0, hi - CAL_SLOTS);
                }
            }
            self.cal_next = c + 1;
            if self.cal_far_min < (self.cal_next + CAL_SLOTS as u128) {
                let (mut i, mut min) = (0, u128::MAX);
                while i < self.cal_far.len() {
                    let (w, q) = self.cal_far[i];
                    if w < (self.cal_next + CAL_SLOTS as u128) {
                        self.cal_far.swap_remove(i);
                        self.cal_schedule(q, w);
                    } else {
                        min = min.min(w);
                        i += 1;
                    }
                }
                self.cal_far_min = min;
            }
        }
        std::mem::take(&mut self.cal_now)
    }

    /// Move the calendar slots `lo..hi` into `cal_now`.
    fn cal_drain(&mut self, lo: usize, hi: usize) {
        let mut w = lo >> 6;
        while w << 6 < hi {
            let from = (w << 6).max(lo);
            let to = ((w + 1) << 6).min(hi);
            let span = if to - from == 64 { !0 } else { ((1u64 << (to - from)) - 1) << (from & 63) };
            let mut bits = self.cal_occ[w] & span;
            self.cal_occ[w] &= !span;
            while bits != 0 {
                let s = (w << 6) + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                for i in 0..3 {
                    self.cal_now[i] |= self.cal[s][i];
                }
                self.cal[s] = [0; 3];
            }
            w += 1;
        }
    }

    /// The public query may move backwards after a speculative future query. Cached
    /// clamped event times are valid monotonically, so a backwards query invalidates them.
    pub(crate) fn ev_begin_query(&self, cycle: u128) {
        if self.ev_query_at.replace(Some(cycle)).is_some_and(|previous| cycle < previous) {
            for (word, bits) in self.ev_dirty.iter().zip(ALL_DATA) {
                word.set(bits);
            }
        }
    }

    /// Queue `q`'s part of `model_next_event` must be computed again.
    pub(crate) fn ev_invalidate(&self, q: usize) {
        let word = &self.ev_dirty[q >> 6];
        word.set(word.get() | (1 << (q & 63)));
    }

    /// The queues whose part of `model_next_event` must be computed again (all of them when the barrier
    /// changed), removed from the set.
    pub(crate) fn ev_take_dirty(&self) -> [u64; 3] {
        let dirty = std::array::from_fn(|i| self.ev_dirty[i].replace(0));
        if self.ev_gen.get() != self.barrier.gen {
            self.ev_gen.set(self.barrier.gen);
            return ALL_DATA;
        }
        dirty
    }

    /// Set queue `q`'s part of `model_next_event` (`u128::MAX`: none).
    pub(crate) fn ev_set(&self, q: usize, v: u128) {
        let t = &self.ev_tree;
        let mut i = EV_LEAVES + q;
        t[i].set(v);
        while i > 1 {
            i >>= 1;
            t[i].set(t[2 * i].get().min(t[2 * i + 1].get()));
        }
    }

    /// The earliest part of `model_next_event` over the queues, and a queue that has it.
    pub fn ev_min(&self) -> (u128, usize) {
        let t = &self.ev_tree;
        let m = t[1].get();
        let mut i = 1;
        while i < EV_LEAVES {
            i = if t[2 * i].get() == m { 2 * i } else { 2 * i + 1 };
        }
        (m, i - EV_LEAVES)
    }

    /// Queue `q`'s cached part of `model_next_event` (for the check).
    pub fn ev_get(&self, q: usize) -> u128 {
        self.ev_tree[EV_LEAVES + q].get()
    }

    // model: postlude/tick.sail agt_get
    pub fn agt_get(&self, q: usize, p: usize) -> Agt {
        *self.agt_ref(q, p)
    }

    /// [`Driver::agt_get`] without the copy: an `Agt` carries four levels, and most readers want one field of it.
    pub fn agt_ref(&self, q: usize, p: usize) -> &Agt {
        if p == 0 {
            &self.agts[q]
        } else {
            &self.agts_w[q]
        }
    }

    /// In-place access to the AGT of path `p` of queue `q`, marking the queue changed as [`Driver::agt_set`] does: the
    /// driver's per-element bookkeeping (`run_loop`, `agt_refresh`) edits the few fields it changes instead of copying the
    /// four levels out and back.
    #[inline]
    pub fn agt_mut(&mut self, q: usize, p: usize) -> &mut Agt {
        self.touch(q);
        if p == 0 {
            &mut self.agts[q]
        } else {
            &mut self.agts_w[q]
        }
    }

    // model: postlude/tick.sail agt_set
    #[inline]
    pub fn agt_set(&mut self, q: usize, p: usize, a: Agt) {
        self.touch(q);
        if p == 0 {
            self.agts[q] = a;
        } else {
            self.agts_w[q] = a;
        }
    }

    // model: postlude/tick.sail path_busy
    pub fn path_busy(&self, q: usize, p: usize, c: impl TryInto<u128>) -> bool {
        let c = c.try_into().ok().expect("nonnegative cycle");
        c < self.path_busy_until[2 * q + p]
    }

    // model: postlude/tick.sail path_tail
    pub fn path_tail(&self, q: usize, p: usize, c: impl TryInto<u128>) -> bool {
        let c = c.try_into().ok().expect("nonnegative cycle");
        c < self.path_tail_until[2 * q + p]
    }

    // model: postlude/tick.sail hold_path
    pub fn hold_path(&mut self, q: usize, p: usize, c: u64) {
        self.path_busy_until[2 * q + p] = u128::from(c) + 1;
    }
}

/// Report a diagnostic of queue `q` (144 is the control channel, reported as 65535) to the observer.
pub fn report<O: Observer + ?Sized>(obs: &mut O, q: usize, c: u64, d: Diag) {
    obs.diag(c, if q == 144 { 0xffff } else { q as u16 }, d);
}

/// The position of a code in the `diag` type of `core/diagnostics.sail`: the code field of a diagnostic log entry.
// model: core/queues.sail diag_number
pub fn diag_number(d: Diag) -> Option<u64> {
    use ots_isa::Code::*;
    Some(match d {
        Dec003 => 0,
        Que001 => 1,
        Que002 => 2,
        Tim001 => 3,
        Dir001 => 4,
        Srf002 => 5,
        Mem001 => 6,
        Opr001 => 7,
        Opr002 => 8,
        Res001 => 9,
        Res002 => 10,
        Res003 => 11,
        Sxm001 => 12,
        Lnk001 => 13,
        Lnk002 => 14,
        Lnk003 => 15,
        Iq002 => 16,
        Iq003 => 17,
        Iq004 => 18,
        Hio001 => 19,
        Ecc001 => 20,
        Lnk004 => 21,
        Ecc003 => 22,
        Ecc002 => 23,
        _ => return None,
    })
}

/// Is the report of code `d` on queue `q` of class `data`: the `data` entries and data exceptions of the `Class` column of `microarchitecture.md` §8 that `platform.md` §7 names
/// for hardware conformance (the rules that read SRAM words, stream values or the instruction text)? IQ-002 and IQ-004 are in it.
/// OPR-001 is kept for every MEM queue and every `*_SEL` queue (the report does not say which instruction raised it: a documented
/// superset of the Gather and Scatter tile addresses and the `sxselect` select).
// model: core/queues.sail diag_is_data
pub fn diag_is_data(q: usize, d: Diag) -> bool {
    use ots_isa::Code::*;
    let queue = if q == 144 { 0xffff } else { q as u16 };
    let unit = ots_isa::UNITS.iter().find(|(n, qs)| *n != "any" && qs.contains(&queue)).map_or("", |(n, _)| *n);
    match d {
        Sxm001 | Dec003 | Iq002 | Iq004 => true,
        Opr001 => unit.starts_with("MEM") || unit.ends_with("_SEL"),
        Tim001 => unit.starts_with("C2C") && unit.ends_with("_RX"),
        _ => false,
    }
}

impl<const L: usize> crate::machine::Machine<L> {
    /// Report a diagnostic of queue `q`: to the observer, and for a code of class `data` into the chip's diagnostic log (the
    /// CSR space, `diag_report` of `core/queues.sail`): kept in the order of arrival, or dropped with the overflow flag when it is full.
    // model: core/queues.sail diag_report
    pub fn diag<O: Observer + ?Sized>(&mut self, obs: &mut O, q: usize, c: u64, d: Diag) {
        report(obs, q, c, d);
        if d == Diag::Dec003 && self.iq.taken_fetched(q) {
            // the unit came from a fetch or the host, not from the object: a data event whatever the table says of the code
            obs.diag_fetched(c, if q == 144 { 0xffff } else { q as u16 }, d);
        }
        if d == Diag::Opr001 && self.ecc.data_opr.take() {
            // the instruction that raised it (a Gather or Scatter tile address, an `sxselect` select) makes it data
            obs.diag_data(c, if q == 144 { 0xffff } else { q as u16 }, d);
        }
        if diag_is_data(q, d) {
            if let Some(n) = diag_number(d) {
                let r = &mut self.host.ras;
                if (r.diag.len() as u64) < self.geom.ras_diag_depth {
                    r.diag.push((q as u64 & 0xFF) << 56 | n << 48 | (c & 0xFFFF_FFFF_FFFF));
                } else {
                    r.diag_overflow = true;
                }
            }
        }
    }
}

impl<const L: usize> crate::machine::Machine<L> {
    /// The end-of-run report of the queues that hold text they never dispatched because they are parked for good: a
    /// HostWait that no doorbell released, a DeskewRecv whose vector never arrived, a Sync that no Notify releases (a
    /// Sync of the control channel parks every data queue). The model reports no diagnostic for them (`finish`), so a run
    /// that ends this way has judged only the text before the wait. A queue whose last instruction is the wait holds no
    /// text and is not reported.
    pub fn report_parked<O: Observer + ?Sized>(&self, obs: &mut O) {
        let b = &self.d.barrier;
        let control = match (b.parked_since[144], b.release_at[144]) {
            (Some(since), None) => Some(since),
            _ => None,
        };
        for q in 0..crate::geom::DATA_QUEUES {
            if self.iq.len[q] == 0 {
                continue;
            }
            let why = if let (Some(since), None) = (b.parked_since[q], b.release_at[q]) {
                Some((since, "sync"))
            } else if b.hw_for_good(q) {
                b.hw_since[q].map(|since| (since, "host-wait"))
            } else if let Some(p) = self.d.queue_poll[q] {
                Some((p.since, "deskew-recv"))
            } else {
                control.map(|since| (since, "control-sync"))
            };
            if let Some((since, cause)) = why {
                obs.parked_text(q as u16, since, cause);
            }
        }
    }
}

// model: core/lints.sail LINT_IQ_UNDERFLOW
pub const LINT_IQ_UNDERFLOW: &str = "OTS-W-IQ-001";

// model: core/lints.sail LINT_RAS_EVICT
pub const LINT_RAS_EVICT: &str = "OTS-W-RAS-001";

/// Report a simulator lint of queue `q` (144 is the control channel, reported as 65535).
// model: core/lints.sail lint
pub fn lint<O: Observer + ?Sized>(obs: &mut O, q: usize, c: u64, code: &'static str) {
    obs.lint(c, if q == 144 { 0xffff } else { q as u16 }, code);
}
