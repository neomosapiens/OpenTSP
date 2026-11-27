//! Instructions per parcel and the holds (`postlude/tick.sail`; design
//! sec. 7.5, 7.6). A non-MEM queue dispatches one instruction per parcel. A MEM queue has slots:
//! one Ifetch; per liquid thread one R-type reference (Read, Gather) and one W-type reference
//! (Write, Scatter); one configuration record; or one instruction that needs the whole queue,
//! alone. When a parcel's instructions do not fit, every instruction of the parcel is OTS-E-QUE-001
//! and none dispatches, so the result does not depend on the order of the instructions.
use super::agt::AgtState;
use crate::observe::Hold;
use crate::Machine;
use ots_isa::{Code as Diag, EntryId, Insn};

/// `cycle_recs`: the parcel of a queue sorted into slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CycleRecs {
    /// The instructions dispatch (each in its slot).
    pub ok: bool,
    /// Number of instructions.
    pub n: usize,
    /// A lone instruction that does not decode.
    pub err: Option<Diag>,
    /// A lone instruction that needs the whole queue.
    pub single: Option<Insn>,
    pub ifetch: Option<Insn>,
    /// Index `2 * path + thread`.
    pub refs: [Option<Insn>; 4],
    /// The configuration record (one per cycle).
    pub cfg: Option<Insn>,
}

impl CycleRecs {
    // model: postlude/tick.sail no_recs
    pub const fn none() -> CycleRecs {
        CycleRecs { ok: true, n: 0, err: None, single: None, ifetch: None, refs: [None; 4], cfg: None }
    }
}

/// The dispatch path of a reference: R (Read, Gather) 0, W (Write, Scatter) 1.
// model: slices/mem/mem_insts.sail ref_path
pub fn ref_path(i: &Insn) -> Option<usize> {
    match i {
        Insn::Read { .. } | Insn::Gather { .. } => Some(0),
        Insn::Write { .. } | Insn::Scatter { .. } => Some(1),
        _ => None,
    }
}

/// Does `i` configure an AGT loop (`Entry::agt_config` is set)? Read from the instruction itself: the catalogue row of
/// an entry is a large record that the dispatch would otherwise fetch for every instruction.
// model: slices/mem/mem_insts.sail agt_config
pub fn is_agt_config(i: &Insn) -> bool {
    matches!(i, Insn::Countdown { .. } | Insn::Step { .. } | Insn::StreamStep { .. } | Insn::Iterations { .. })
}

/// The liquid-thread tag of a reference (0 for anything else).
// model: slices/mem/mem_insts.sail ref_thread
pub fn ref_thread(i: &Insn) -> u8 {
    match *i {
        Insn::Read { thread, .. }
        | Insn::Write { thread, .. }
        | Insn::Gather { thread, .. }
        | Insn::Scatter { thread, .. } => thread as u8 & 1,
        _ => 0,
    }
}

/// Sort the decoded records of MEM queue `q` into their slots; `false` when two share one.
// model: postlude/tick.sail place
pub fn place(x: &mut CycleRecs, i: Insn) -> bool {
    if is_agt_config(&i) {
        if x.cfg.is_some() {
            return false;
        }
        x.cfg = Some(i);
        return true;
    }
    if let Some(p) = ref_path(&i) {
        let k = 2 * p + ref_thread(&i) as usize;
        if x.refs[k].is_some() {
            return false;
        }
        x.refs[k] = Some(i);
        return true;
    }
    match i.id() {
        // IfetchM takes the fetch slot too
        EntryId::Ifetch | EntryId::IfetchM => {
            if x.ifetch.is_some() {
                return false;
            }
            x.ifetch = Some(i);
            true
        }
        _ => false,
    }
}

/// The instructions of a parcel on queue `q`, sorted into slots (one that needs the whole queue,
/// or that does not decode, is left to `dispatch`). A NOP inside a parcel is never alone and fits
/// no slot (design sec. 7.3).
// model: postlude/tick.sail classify_items
pub fn classify_items(q: usize, items: &[Result<Insn, Diag>]) -> CycleRecs {
    let mut r = CycleRecs::none();
    classify_into(q, items, &mut r);
    r
}

/// [`classify_items`] written into `r`, which holds [`CycleRecs::none`]: a record is 300 bytes, and the parse makes one
/// per parcel in the place where it is kept.
pub fn classify_into(q: usize, items: &[Result<Insn, Diag>], r: &mut CycleRecs) {
    let n = items.len();
    r.n = n;
    match items {
        [] => {}
        [x] => match x {
            Err(d) => {
                r.ok = false;
                r.err = Some(*d);
            }
            Ok(i) => {
                // `place` changes nothing when it refuses
                if q > 87 || !place(r, *i) {
                    r.single = Some(*i);
                }
            }
        },
        _ => {
            if q > 87 {
                r.ok = false;
                return;
            }
            let mut ok = true;
            for y in items {
                match y {
                    Err(_) => ok = false,
                    Ok(i) => {
                        let fit = place(r, *i);
                        ok &= fit;
                    }
                }
            }
            // IfetchM reads SRAM through the R port, for every tile (both liquid
            // threads), so it does not fit with an R-type reference (Read, Gather) of any thread.
            let fetchm = matches!(r.ifetch.map(|i| i.id()), Some(EntryId::IfetchM));
            if fetchm && (r.refs[0].is_some() || r.refs[1].is_some()) {
                ok = false;
            }
            r.ok = ok;
        }
    }
}

/// True when the cycle carries a configuration record: open configurations continue.
// model: postlude/tick.sail config_for
pub fn config_for(r: &CycleRecs) -> bool {
    r.ok && r.cfg.is_some()
}

impl<const L: usize> Machine<L> {
    /// A whole-queue instruction needs both paths idle: no AGT loop running or configuration open
    /// (state not idle: an open configuration runs its loop with the fields received so far, so a
    /// reference may be due in this very cycle), no path busy or in a tail.
    /// The reason, if one holds: a busy path or a tail is `Path`, a running loop `Loop`, an open
    /// configuration `Configuration`.
    // model: postlude/tick.sail view_paths_busy
    fn any_path_busy(&self, q: usize, c: impl TryInto<u128>) -> Option<Hold> {
        let c = c.try_into().ok().expect("nonnegative cycle");
        let d = &self.d;
        if d.path_busy(q, 0, c) || d.path_busy(q, 1, c) || d.path_tail(q, 0, c) || d.path_tail(q, 1, c) {
            return Some(Hold::Path);
        }
        match (d.agts[q].state, d.agts_w[q].state) {
            (AgtState::Idle, AgtState::Idle) => None,
            (AgtState::Config, _) | (_, AgtState::Config) => Some(Hold::Configuration),
            _ => Some(Hold::Loop),
        }
    }

    /// Is instruction `i` of the head parcel of queue `q` held at cycle `c` (design sec. 7.6)? An
    /// instruction on dispatch path `path` needs only its path free; any other instruction the whole
    /// queue; `strict`: the parcel breaks the slot rule, so every instruction needs the whole queue
    /// (both paths, Ifetch included). `i` is `None` for an instruction that does not decode. Held when
    /// - the queue is parked by a Sync (a queue's own Sync and Notify are the barrier; the control
    ///   channel's Sync parks every data queue);
    /// - a Repeat runs on the queue;
    /// - the queue is busy (`c < busy-until`: Deskew; an instruction issued this cycle);
    /// - its path is busy, in the tail of a finished loop, runs a loop, or holds an open
    ///   configuration that `i` does not continue.
    // model: postlude/tick.sail hold_check
    pub fn hold_check(
        &self,
        q: usize,
        c: impl TryInto<u128>,
        i: Option<&Insn>,
        path: Option<usize>,
        strict: bool,
    ) -> bool {
        let c = c.try_into().ok().expect("nonnegative cycle");
        self.hold_reason(q, c, i, path, strict).is_some()
    }

    /// [`Machine::hold_check`] with the reason (the first that holds, in the order of the list there: parked, Repeat, busy, then the path; which
    /// of them holds never changes the result, only the reason reported).
    pub fn hold_reason(
        &self,
        q: usize,
        c: impl TryInto<u128>,
        i: Option<&Insn>,
        path: Option<usize>,
        strict: bool,
    ) -> Option<Hold> {
        let c = c.try_into().ok().expect("nonnegative cycle");
        let own_barrier = matches!(i, Some(Insn::Sync | Insn::Notify));
        let ifetch = !strict && matches!(i, Some(Insn::Ifetch { .. } | Insn::IfetchM { .. })); // IfetchM
        if !own_barrier && self.d.barrier.queue_parked(q, c) {
            return Some(Hold::Parked);
        }
        if self.d.barrier.host_parked(q, c) {
            return Some(Hold::Parked); // HostWait: the queue waits for the host's ring (holds Sync and Notify too)
        }
        if self.d.queue_repeat[q].is_some() {
            return Some(Hold::Repeat);
        }
        if self.d.poll_n != 0 && self.d.queue_poll[q].is_some() {
            return Some(Hold::Repeat); // a DeskewRecv polls: it holds the queue like a Repeat that runs
        }
        if c < self.d.queue_busy_until[q] {
            return Some(Hold::Busy);
        }
        match path {
            // A path's loop runs while its configuration arrives: it holds the path against
            // new references in both states; only its configuration records get through.
            Some(p) => {
                let a = self.d.agt_ref(q, p);
                if self.d.path_busy(q, p, c) || self.d.path_tail(q, p, c) {
                    Some(Hold::Path)
                } else if a.state == AgtState::Run {
                    Some(Hold::Loop)
                } else if a.state == AgtState::Config && !i.is_some_and(is_agt_config) {
                    Some(Hold::Configuration)
                } else {
                    None
                }
            }
            None if ifetch && q <= 87 => None,
            None => self.any_path_busy(q, c),
        }
    }

    /// The parcel's dispatch needs: is any instruction of the parcel (classified as `r`) held at
    /// cycle `c`?
    // model: postlude/tick.sail parcel_held
    pub fn parcel_held(&self, q: usize, c: impl TryInto<u128>, r: &CycleRecs, items: &[Result<Insn, Diag>]) -> bool {
        let c = c.try_into().ok().expect("nonnegative cycle");
        if self.d.fast_on
            && self.d.barrier.quiet()
            && self.d.queue_repeat[q].is_none()
            && self.d.poll_n == 0
            && self.is_common_parcel(q, r)
        {
            let held = self.held_common(q, c, r);
            if self.d.fast_check {
                assert_eq!(held, self.parcel_hold(q, c, r, items).is_some(), "fast path: queue {q} at cycle {c}: held");
            }
            return held;
        }
        self.parcel_hold(q, c, r, items).is_some()
    }

    /// [`Machine::parcel_held`] of a common parcel (see `is_common_parcel`) on a queue with nothing parked and no
    /// Repeat: held by the queue's busy cycle or by the dispatch path of an instruction (a busy or tail cycle, a loop
    /// or an open configuration; no instruction of such a parcel continues a configuration).
    fn held_common(&self, q: usize, c: impl TryInto<u128>, r: &CycleRecs) -> bool {
        let c = c.try_into().ok().expect("nonnegative cycle");
        let d = &self.d;
        if c < d.queue_busy_until[q] {
            return true;
        }
        if r.single.is_some() {
            return self.any_path_busy(q, c).is_some();
        }
        for (k, slot) in r.refs.iter().enumerate() {
            if slot.is_some() {
                let p = usize::from(k >= 2);
                if d.path_busy(q, p, c) || d.path_tail(q, p, c) || d.agt_ref(q, p).state != AgtState::Idle {
                    return true;
                }
            }
        }
        false
    }

    /// [`Machine::parcel_held`] with the reason of the first instruction that is held.
    pub fn parcel_hold(
        &self,
        q: usize,
        c: impl TryInto<u128>,
        r: &CycleRecs,
        items: &[Result<Insn, Diag>],
    ) -> Option<Hold> {
        let c = c.try_into().ok().expect("nonnegative cycle");
        if r.n == 0 {
            return None;
        }
        if !r.ok {
            // A parcel that breaks the slot rule (or an instruction that does not decode): every
            // instruction needs the whole queue.
            return items.iter().find_map(|y| match y {
                Ok(i) => self.hold_reason(q, c, Some(i), None, true),
                Err(_) => self.hold_reason(q, c, None, None, true),
            });
        }
        if let Some(i) = &r.single {
            return self.hold_reason(q, c, Some(i), None, false);
        }
        // The first instruction held, in slot order: the later ones are not looked at.
        if let Some(i) = &r.ifetch {
            let p = matches!(i.id(), EntryId::IfetchM).then_some(0);
            if let h @ Some(_) = self.hold_reason(q, c, Some(i), p, false) {
                return h;
            }
        }
        if let Some(i) = &r.cfg {
            if let h @ Some(_) = self.hold_reason(q, c, Some(i), Some(self.d.cfg_path(q)), false) {
                return h;
            }
        }
        for k in 0..4 {
            if let Some(i) = &r.refs[k] {
                if let h @ Some(_) = self.hold_reason(q, c, Some(i), Some(usize::from(k >= 2)), false) {
                    return h;
                }
            }
        }
        None
    }

    /// Queue `q` is free at cycle `t` (design sec. 7.7): not parked, not busy, no Repeat, and both
    /// paths idle (no loop or open configuration, not busy, not in a tail). The control channel is
    /// always free.
    // model: postlude/tick.sail free_at
    pub fn free_at(&self, q: usize, t: impl TryInto<u128>) -> bool {
        let t = t.try_into().ok().expect("nonnegative cycle");
        if q == 144 {
            return true;
        }
        let d = &self.d;
        !d.barrier.queue_parked(q, t)
            && !d.barrier.host_parked(q, t)
            && t >= d.queue_busy_until[q]
            && d.queue_repeat[q].is_none()
            && !(d.poll_n != 0 && d.queue_poll[q].is_some())
            && d.agts[q].state == AgtState::Idle
            && d.agts_w[q].state == AgtState::Idle
            && !d.path_busy(q, 0, t)
            && !d.path_busy(q, 1, t)
            && !d.path_tail(q, 0, t)
            && !d.path_tail(q, 1, t)
    }
}
