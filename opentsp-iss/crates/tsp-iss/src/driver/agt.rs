//! The address generation table, one per queue and dispatch path (the nested-loop rule and the
//! configuration rules of `microarchitecture.md` §4; HC34 p.25).
//!
//! A loopable MEM instruction (Read, Write, Gather, Scatter) is reference 0 at its dispatch cycle
//! t0; `Countdown`/`Step`/`StreamStep`/`Iterations` records on the same queue in the following
//! consecutive cycles configure up to four nested loop levels, inner to outer. The model's text
//! is in `postlude/tick.sail` ("Address generation table"); the timing follows the pseudocode
//! statement by statement: Issue takes one cycle and every `Wait(countdown_x)` is unconditional.
//!
//! This module holds the state, the counters and the end-of-configuration step (`agt_finish_config`,
//! `agt_refresh`, `agt_due`): the cycle driver reads them. Opening a configuration and running the
//! references belongs to MEM dispatch (`dispatch_slots`).
use super::Driver;
use ots_isa::Insn;

/// `agt_level`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AgtLevel {
    pub countdown: u64,
    pub step: i64,
    pub sstep: i64,
    pub iters: u64,
    pub idx: u64,
}

/// `agt_default_level`: "0, 0, or 2".
pub const AGT_DEFAULT_LEVEL: AgtLevel = AgtLevel { countdown: 0, step: 0, sstep: 0, iters: 2, idx: 0 };

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgtState {
    Idle,
    Config,
    Run,
}

/// `agt`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Agt {
    pub state: AgtState,
    /// The loopable instruction (`None` when idle).
    pub instr: Option<Insn>,
    /// Dispatch cycle of reference 0.
    pub t0: u64,
    pub levels: [AgtLevel; 4],
    /// Level being configured.
    pub level: usize,
    /// Last field written in that level (-1: none).
    pub pos: i64,
    /// Address offset of the last reference.
    pub offset: i64,
    /// Stream id offset of the last reference (StreamStep).
    pub soffset: i64,
    /// Cycle of the last reference.
    pub last: u64,
    /// Cycle of the next reference (valid in `Run`).
    pub next: u128,
}

impl Agt {
    // model: postlude/tick.sail agt_idle
    pub const fn idle() -> Agt {
        Agt {
            state: AgtState::Idle,
            instr: None,
            t0: 0,
            levels: [AGT_DEFAULT_LEVEL; 4],
            level: 0,
            pos: -1,
            offset: 0,
            soffset: 0,
            last: 0,
            next: 0,
        }
    }
}

/// The AGT field an instruction configures and its raw 16-bit value: the position in the
/// `Countdown`, `Step`, `StreamStep`, `Iterations` order (0..3).
// model: slices/mem/mem_insts.sail agt_config
pub fn agt_config(i: &Insn) -> Option<(u8, u16)> {
    i.entry().agt_config.map(|p| (p, i.fields()[0] as u16))
}

/// The result of [`agt_advance`]: the counters after the step, the delay from the previous
/// reference, the step and the stream step of the advancing level.
pub type AgtStep = Option<([AgtLevel; 4], u64, i64, i64)>;

/// Apply one configuration write of field `fp` with value `v`; `false` when it would need a fifth
/// level.
// model: postlude/tick.sail agt_configure
pub fn agt_configure(a: &mut Agt, fp: u8, v: u16) -> bool {
    let fp = fp as i64;
    let mut b = *a;
    let mut ok = true;
    if fp <= b.pos {
        if b.level >= 3 {
            ok = false;
        } else {
            b.level += 1;
            b.pos = -1;
        }
    }
    if ok {
        let mut l = b.levels[b.level];
        match fp {
            0 => l.countdown = (v as u64).max(1),
            1 => l.step = v as i16 as i64,
            2 => l.sstep = v as i64,
            _ => l.iters = (v as u64).max(1),
        }
        b.levels[b.level] = l;
        b.pos = fp;
        *a = b;
    }
    ok
}

/// The loop counters of the next reference after the one at counters `levels`, over the first `n`
/// levels (the levels received so far); `None` when none of them can advance. Returns the delay
/// from the previous reference (its Issue cycle, plus `Wait(countdown)` of every level that wrapped
/// and of the level that advances) and the step and stream step of the advancing level.
// model: postlude/tick.sail agt_advance
pub fn agt_advance(levels: &[AgtLevel; 4], n: usize) -> AgtStep {
    let mut ls = *levels;
    let mut wait: u64 = 1;
    for k in 0..4 {
        if k < n {
            let mut l = ls[k];
            wait += l.countdown;
            if l.idx + 1 < l.iters {
                l.idx += 1;
                ls[k] = l;
                return Some((ls, wait, l.step, l.sstep));
            }
            l.idx = 0;
            ls[k] = l;
        }
    }
    None
}

/// The delay [`agt_advance`] returns, without building the advanced levels: the delay from the previous reference to the next
/// one over the first `n` levels, `None` when none of them can advance.
#[inline]
pub fn agt_next_wait(levels: &[AgtLevel; 4], n: usize) -> Option<u64> {
    let mut wait: u64 = 1;
    for (k, l) in levels.iter().enumerate() {
        if k < n {
            wait += l.countdown;
            if l.idx + 1 < l.iters {
                return Some(wait);
            }
        }
    }
    None
}

/// What the driver needs of a due AGT reference: the delay from the previous reference and the step and stream step of the
/// advancing level. The levels themselves advance in place ([`agt_apply_advance`]) when the reference has run, so the
/// 160 bytes of the four levels are not copied out of the AGT and back on every element.
pub type AgtDue = Option<(u64, i64, i64)>;

/// [`agt_due`] without the levels: the same reference is due at the same cycle, with the same delay, step and stream step.
#[inline]
pub fn agt_due_lean(a: &Agt, c: u64, parked: bool) -> AgtDue {
    if a.state == AgtState::Idle || parked {
        return None;
    }
    let n = agt_nlevels(a);
    let mut wait: u64 = 1;
    for (k, l) in a.levels.iter().enumerate() {
        if k < n {
            wait += l.countdown;
            if l.idx + 1 < l.iters {
                return (u128::from(a.last) + u128::from(wait) <= u128::from(c)).then_some((wait, l.step, l.sstep));
            }
        }
    }
    None
}

/// Advance the counters of `levels` over the first `n` levels as [`agt_advance`] does, in place: the first level that can
/// advance counts up, the levels below it wrap to 0. Does nothing to a loop with no level left to advance.
#[inline]
pub fn agt_apply_advance(levels: &mut [AgtLevel; 4], n: usize) {
    for l in levels.iter_mut().take(n) {
        if l.idx + 1 < l.iters {
            l.idx += 1;
            return;
        }
        l.idx = 0;
    }
}

/// Number of levels received so far.
// model: postlude/tick.sail agt_nlevels
pub fn agt_nlevels(a: &Agt) -> usize {
    if a.pos < 0 {
        0
    } else {
        a.level + 1
    }
}

/// The AGT reference due at cycle `c`, as the counters, delay, step and stream step of
/// [`agt_advance`] (with the fields received so far, also while the configuration is still
/// arriving); `None` when none is due (idle, parked, waiting for its countdown, or every level
/// received so far exhausted).
// model: postlude/tick.sail agt_due
pub fn agt_due(a: &Agt, c: u64, parked: bool) -> AgtStep {
    if a.state == AgtState::Idle || parked {
        return None;
    }
    match agt_advance(&a.levels, agt_nlevels(a)) {
        None => None,
        Some((ls, d, st, ss)) => (u128::from(a.last) + u128::from(d) <= u128::from(c)).then_some((ls, d, st, ss)),
    }
}

impl<const L: usize> Driver<L> {
    /// With the configuration of path `p` of queue `q` closed (`Run`): record the cycle of the next
    /// reference, or go idle when no level can advance.
    // model: postlude/tick.sail agt_refresh
    pub fn agt_refresh(&mut self, q: usize, p: usize) {
        let a = self.agt_ref(q, p);
        if a.state != AgtState::Run {
            return;
        }
        // a level can advance: only `next` changes, in place
        if let Some(d) = agt_next_wait(&a.levels, agt_nlevels(a)) {
            let next = u128::from(a.last) + u128::from(d);
            self.agt_mut(q, p).next = next;
            return;
        }
        let a = *a;
        match agt_advance(&a.levels, agt_nlevels(&a)) {
            None => {
                // the trailing waits of every level the loop has.
                let mut tail = u128::from(a.last) + 1;
                for k in 0..agt_nlevels(&a) {
                    tail += u128::from(a.levels[k].countdown);
                }
                self.path_tail_until[2 * q + p] = tail;
                self.agt_set(q, p, Agt::idle());
            }
            Some((_, d, _, _)) => self.agt_set(q, p, Agt { next: u128::from(a.last) + u128::from(d), ..a }),
        }
    }

    /// The configuration of path `p` of queue `q` ended (no configuration record at this cycle).
    // model: postlude/tick.sail agt_finish_config
    pub fn agt_finish_config(&mut self, q: usize, p: usize) {
        let a = self.agt_ref(q, p);
        if agt_nlevels(a) == 0 {
            // No configuration record followed the reference: no level can advance, so the loop is over at once
            // (what `agt_refresh` finds), with the trailing wait of one cycle.
            self.path_tail_until[2 * q + p] = u128::from(a.last) + 1;
            self.agt_set(q, p, Agt::idle());
            return;
        }
        let a = *a;
        self.agt_set(q, p, Agt { state: AgtState::Run, ..a });
        self.agt_refresh(q, p);
    }

    /// The path a configuration record configures: the path whose AGT is configuring; when both
    /// are, the one with fewer records so far (R on a tie), so a Read/Write pair's records alternate
    /// R, W, R, W, ...; with neither (a stray record), R.
    // model: postlude/tick.sail cfg_path
    pub fn cfg_path(&self, q: usize) -> usize {
        let r = self.agts[q].state == AgtState::Config;
        let w = self.agts_w[q].state == AgtState::Config;
        if r && w {
            usize::from(self.agt_records[2 * q + 1] < self.agt_records[2 * q])
        } else {
            usize::from(w)
        }
    }
}
