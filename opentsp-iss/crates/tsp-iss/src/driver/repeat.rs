//! Repeat (`postlude/tick.sail`): `Repeat n, d` re-issues the queue's previous instruction `n` more
//! with `d` cycles between iterations, so the spacing is `d + 1`, counted from that instruction's own issue
//! cycle `p`: iteration k at `p + k (d + 1)`. The first iteration is never before the Repeat's own cycle `r`: it
//! issues at `max(p + d + 1, r)`, the rest `d + 1` apart. The queue is busy meanwhile. The dispatch of a
//! Repeat is in `dispatch::dispatch_decoded`, the iterations are driven by `tick`.
use ots_isa::{EntryId, Insn};

/// `repeat_state`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RepeatState {
    pub instr: Insn,
    pub left: u64,
    pub gap: u128,
    pub next: u128,
}

/// `poll_state` of `postlude/tick.sail`: a DeskewRecv that has dispatched at `since` and not yet found its vector. At each
/// epoch boundary after `since` it becomes a Receive of these operands, dispatched in that cycle, once a vector has
/// arrived on the link.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PollState {
    pub chip: u32,
    pub link: u32,
    pub stream: u32,
    pub since: u64,
}

/// The instruction a Repeat re-issues: the queue's last dispatched instruction other than NOP,
/// Repeat and AGT configuration (a Read/Write pair is not repeatable).
// model: postlude/tick.sail repeatable
pub fn repeatable(i: &Insn) -> bool {
    !matches!(i.id(), EntryId::NOP | EntryId::Repeat) && !super::classify::is_agt_config(i)
}

/// Sync, Notify, Deskew and RuntimeDeskew act at dispatch (the barrier, the queue hold), which an
/// iteration would not repeat, so they leave nothing to repeat.
// model: postlude/tick.sail clears_repeat
pub fn clears_repeat(i: &Insn) -> bool {
    matches!(
        i.id(),
        EntryId::Sync
            | EntryId::Notify
            | EntryId::Deskew
            | EntryId::RuntimeDeskew
            | EntryId::HostWait
            | EntryId::DeskewRecv
    )
}

/// First iteration and remainder of a Repeat (`postlude/tick.sail` `repeat_plan`).
// model: postlude/tick.sail repeat_plan
pub fn repeat_plan(prev: Insn, n: u64, d: u64, last_at: u64, c: u64) -> (bool, Option<RepeatState>) {
    let gap = u128::from(d) + 1;
    let first = (u128::from(last_at) + gap).max(u128::from(c));
    if first == u128::from(c) {
        let rest = if n <= 1 {
            None
        } else {
            Some(RepeatState { instr: prev, left: n.saturating_sub(1), gap, next: u128::from(c) + gap })
        };
        (true, rest)
    } else {
        (false, Some(RepeatState { instr: prev, left: n, gap, next: first }))
    }
}

/// The Repeat that remains after an iteration at cycle `c`.
// model: postlude/tick.sail repeat_after
pub fn repeat_after(r: RepeatState, c: u64) -> Option<RepeatState> {
    if r.left <= 1 {
        None
    } else {
        Some(RepeatState { left: r.left.saturating_sub(1), next: u128::from(c) + r.gap, ..r })
    }
}
