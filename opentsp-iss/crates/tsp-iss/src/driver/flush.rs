//! End of cycle: apply the staged MXM plane parts and SXM maps (`postlude/tick.sail`
//! `unit_state_flush`). Queues that staged the same part of one plane, or the same
//! SXM map, in this cycle raise OTS-E-RES-003 (once each, in queue order) and none of their staged
//! changes take effect; an ACC among them also loses its scheduled stream writes (an MXM queue's
//! only stream writes committing at that cycle). The MXM half is `slices/mxm`, the SXM maps' half `slices/sxm`.
use crate::layout::{issuer_of, UnitKind};
use crate::observe::Observer;
use crate::Machine;
use ots_isa::{Code as Diag, EntryId};

/// True for the MXM queues (a writer index of the commit ring may also be 144, the control channel).
// model: core/queues.sail is_mxm_queue
fn is_mxm_queue(q: usize) -> bool {
    q <= 143 && matches!(issuer_of(q).unit, UnitKind::MxmW | UnitKind::MxmE)
}

impl<const L: usize> Machine<L> {
    // model: postlude/tick.sail unit_state_flush
    pub fn unit_state_flush<O: Observer + ?Sized>(&mut self, c: u64, obs: &mut O) {
        if self.mxm.staged.is_empty() && self.sxm.staged.is_empty() {
            return;
        }
        let mut bad = [false; 145];
        let mxm_collided = self.mxm.mark_clashing_queues(&mut bad);
        let sxm_collided = self.sxm.mark_clashing_queues(&mut bad);
        let collided = mxm_collided || sxm_collided;
        if collided {
            for (q, b) in bad.iter().enumerate().take(144) {
                if *b {
                    self.diag(obs, q, c, Diag::Res003);
                }
            }
        }
        self.mxm.flush(&bad);
        self.sxm.flush(&bad);
        if collided {
            // An ACC staged in its execute cycle c (dispatch + d_skew) commits at dispatch + d_func.
            let la = self.timing.row(EntryId::ACC.index());
            let k = (c as i64)
                .wrapping_sub(la.d_skew as i64)
                .wrapping_add(la.d_func as i64)
                .rem_euclid(super::PENDING_SLOTS as i64) as usize; // PENDING_SLOTS divides 2^64, so the wrap does not change the slot
            self.d.pending[k].retain(|e| !(bad[e.q] && is_mxm_queue(e.q)));
        }
    }
}
