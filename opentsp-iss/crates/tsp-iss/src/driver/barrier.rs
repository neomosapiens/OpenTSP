//! The barrier (`postlude/tick.sail`, "Barrier"). A Sync that dispatches parks its queue (on the
//! control channel: every data queue) from its dispatch cycle; a Notify that dispatches at N on any
//! queue releases every Sync dispatched at or before N, at N + `barrier.release_latency`. A Sync or
//! Notify that does not dispatch has no effect. A parked queue issues nothing: its head parcel, its
//! AGT references and its Repeat iterations wait (they resume, shifted, when the queue is
//! released); a Sync or Notify at its head is not held by the barrier.
use crate::geom::QUEUES;

/// `parked_since[q]`: dispatch cycle of the queue's pending Sync, if any; `release_at[q]`: the cycle
/// its pending Sync retires, once a Notify has been seen; `last_notify`: dispatch cycle of the latest
/// Notify (a Sync dispatched later in that same cycle, on a higher queue, is released by it too, so
/// the result does not depend on the order in which the queues of one cycle are visited).
#[derive(Clone, Debug)]
pub struct Barrier {
    pub parked_since: [Option<u64>; QUEUES],
    pub release_at: [Option<u128>; QUEUES],
    pub last_notify: Option<u64>,
    latency: u64,
    /// No release cycle in `release_at` is below it (`u128::MAX`: none is set): `retire` has nothing to do before it.
    next_retire: u128,
    /// Bumped by every change of the barrier: the driver's event index holds answers that depend on it.
    pub gen: u64,
    /// The number of entries of `parked_since` that are set: no queue is parked while it is 0.
    parked_n: u32,
    /// HostWait (`core/csr.sail`): the dispatch cycle of the queue's pending HostWait, the doorbell it waits on, and the
    /// cycle it resumes once the host has rung it (the ring's cycle + 1). A queue parked this way holds Sync and
    /// Notify too. `hw_n` entries of `hw_since` are set.
    pub hw_since: [Option<u64>; QUEUES],
    pub hw_db: [u8; QUEUES],
    pub hw_release: [Option<u128>; QUEUES],
    hw_n: u32,
}

/// The barrier window as a pure function of one Sync and its release cycle.
// model: postlude/tick.sail in_barrier_window
pub fn in_barrier_window(sync: Option<u64>, release: Option<u128>, c: u128) -> bool {
    match (sync, release) {
        (Some(s), Some(r)) => u128::from(s) <= c && c < r,
        (Some(s), None) => u128::from(s) <= c,
        _ => false,
    }
}

impl Barrier {
    pub fn new() -> Barrier {
        Barrier {
            parked_since: [None; QUEUES],
            release_at: [None; QUEUES],
            last_notify: None,
            latency: 0,
            next_retire: u128::MAX,
            gen: 0,
            parked_n: 0,
            hw_since: [None; QUEUES],
            hw_db: [0; QUEUES],
            hw_release: [None; QUEUES],
            hw_n: 0,
        }
    }

    /// With `barrier.release_latency` of the profile.
    pub fn with_latency(latency: u64) -> Barrier {
        Barrier { latency, ..Barrier::new() }
    }

    pub(crate) fn rebuild_after_edit(&mut self) {
        self.parked_n = self.parked_since.iter().filter(|x| x.is_some()).count() as u32;
        self.hw_n = self.hw_since.iter().filter(|x| x.is_some()).count() as u32;
        self.next_retire = self.release_at.iter().flatten().copied().min().unwrap_or(u128::MAX);
        self.gen = self.gen.wrapping_add(1);
    }

    pub fn reset(&mut self) {
        *self = Barrier { gen: self.gen + 1, ..Barrier::with_latency(self.latency) };
    }

    // model: postlude/tick.sail parked_one
    fn parked_one(&self, k: usize, c: impl TryInto<u128>) -> bool {
        let c = c.try_into().ok().expect("nonnegative cycle");
        in_barrier_window(self.parked_since[k], self.release_at[k], c)
    }

    /// Queue `q` (or the control channel, 144) is parked at cycle `c`.
    // model: postlude/tick.sail queue_parked
    pub fn queue_parked(&self, q: usize, c: impl TryInto<u128>) -> bool {
        let c = c.try_into().ok().expect("nonnegative cycle");
        self.parked_n != 0 && (self.parked_one(q, c) || (q != 144 && self.parked_one(144, c)))
    }

    /// The number of set entries of `parked_since`, counted (what `parked_n` must equal).
    pub fn parked_count_scan(&self) -> u32 {
        self.parked_since.iter().filter(|p| p.is_some()).count() as u32
    }

    /// No queue is parked, whatever the cycle.
    pub fn quiet(&self) -> bool {
        self.parked_n == 0 && self.hw_n == 0
    }

    /// Queue `q` is parked by HostWait at cycle `c`: from its dispatch until the cycle before the host's ring takes
    /// effect. Unlike a barrier park it holds Sync and Notify too.
    // model: postlude/tick.sail host_parked
    pub fn host_parked(&self, q: usize, c: impl TryInto<u128>) -> bool {
        let c = c.try_into().ok().expect("nonnegative cycle");
        self.hw_n != 0 && in_barrier_window(self.hw_since[q], self.hw_release[q], c)
    }

    /// Parked by a HostWait the host has not rung: it never resumes by itself.
    // model: postlude/harness_api.sail hw_for_good
    pub fn hw_for_good(&self, k: usize) -> bool {
        self.hw_n != 0 && matches!((self.hw_since[k], self.hw_release[k]), (Some(_), None))
    }

    /// HostWait on doorbell `db` dispatched on queue `q` at cycle `c` finds no ringing doorbell: the queue parks.
    // model: core/csr.sail host_wait
    pub fn hw_park(&mut self, q: usize, c: u64, db: usize) {
        if self.hw_since[q].is_none() {
            self.hw_n += 1;
        }
        self.hw_since[q] = Some(c);
        self.hw_db[q] = db as u8;
        self.hw_release[q] = None;
        self.gen += 1;
    }

    /// The host rings doorbell `db` at cycle `c`: the lowest queue (0..143) parked on it resumes at `c + 1`. False when
    /// no queue waits (the doorbell then stays set).
    // model: core/csr.sail host_ring
    pub fn hw_ring(&mut self, db: usize, c: u64) -> bool {
        if self.hw_n == 0 {
            return false;
        }
        for q in 0..crate::geom::DATA_QUEUES {
            if let (Some(_), None) = (self.hw_since[q], self.hw_release[q]) {
                if self.hw_db[q] as usize == db {
                    self.hw_release[q] = Some(u128::from(c) + 1);
                    self.next_retire = self.next_retire.min(u128::from(c) + 1);
                    self.gen += 1;
                    return true;
                }
            }
        }
        false
    }

    /// The counter of parked queues, for the check.
    pub fn parked_counter(&self) -> u32 {
        self.parked_n
    }

    // model: postlude/tick.sail barrier_parked
    pub fn barrier_parked(&self, c: impl TryInto<u128>) -> bool {
        let c = c.try_into().ok().expect("nonnegative cycle");
        self.parked_one(144, c)
    }

    /// A Sync dispatched on queue `k` at cycle `c`.
    // model: postlude/tick.sail barrier_sync
    pub fn sync(&mut self, k: usize, c: u64) {
        if self.parked_since[k].is_none() {
            self.parked_n += 1;
        }
        self.parked_since[k] = Some(c);
        self.release_at[k] =
            if self.last_notify == Some(c) { Some(u128::from(c) + u128::from(self.latency)) } else { None };
        self.next_retire = self.next_retire.min(u128::from(c) + u128::from(self.latency));
        self.gen += 1;
    }

    /// A Notify dispatched at cycle `n` releases every Sync dispatched at or before `n`.
    // model: postlude/tick.sail barrier_notify
    pub fn notify(&mut self, n: u64) {
        self.last_notify = Some(n);
        self.gen += 1;
        for k in 0..QUEUES {
            if let (Some(s), None) = (self.parked_since[k], self.release_at[k]) {
                if s <= n {
                    self.release_at[k] = Some(u128::from(n) + u128::from(self.latency));
                    self.next_retire = self.next_retire.min(u128::from(n) + u128::from(self.latency));
                }
            }
        }
    }

    /// Retire Syncs whose release cycle has passed (so a later Sync can park again).
    // model: postlude/tick.sail barrier_retire
    pub fn retire(&mut self, c: impl TryInto<u128>) {
        let c = c.try_into().ok().expect("nonnegative cycle");
        if c < self.next_retire {
            return;
        }
        let mut next = u128::MAX;
        if self.hw_n != 0 {
            for k in 0..QUEUES {
                if let Some(r) = self.hw_release[k] {
                    if c >= r {
                        if self.hw_since[k].take().is_some() {
                            self.hw_n -= 1;
                        }
                        self.hw_release[k] = None;
                        self.gen += 1;
                    } else {
                        next = next.min(r);
                    }
                }
            }
        }
        for k in 0..QUEUES {
            if let Some(r) = self.release_at[k] {
                if c >= r {
                    if self.parked_since[k].take().is_some() {
                        self.parked_n -= 1;
                    }
                    self.release_at[k] = None;
                    self.gen += 1;
                } else {
                    next = next.min(r);
                }
            }
        }
        self.next_retire = next;
    }

    /// Parked by a Sync that no Notify has released: it never resumes by itself.
    // model: postlude/tick.sail stuck
    pub fn stuck(&self, k: usize) -> bool {
        self.parked_n != 0 && matches!((self.parked_since[k], self.release_at[k]), (Some(_), None))
    }
}

impl Default for Barrier {
    fn default() -> Self {
        Barrier::new()
    }
}
