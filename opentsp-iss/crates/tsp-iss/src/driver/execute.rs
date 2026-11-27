//! The execute stage (`postlude/tick.sail`). Reference k of an instruction dispatched
//! at `c0` issues at `c0 + k` and EXECUTES at `c0 + k + d_skew` (its entry's latency): it reads its
//! stream operands, SRAM and unit state in that cycle, and its diagnostics are reported at that
//! cycle. Dispatch checks (queue, unit, barrier, slots, banks) stay at the dispatch cycle. An
//! element with `d_skew > 0` waits in the 16-slot execute ring; `tick(c)` runs the elements due at
//! `c`, in dispatch order, before the queues dispatch.
use super::{phase, EXEC_SLOTS};
use crate::observe::Observer;
use crate::slices::{Exec, Issue};
use crate::Machine;
use ots_isa::Insn;
use ots_profile::Latency;

/// `exec_element`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecElement {
    pub q: usize,
    pub instr: Insn,
    pub c0: u64,
    pub k: u64,
    pub off: i64,
    pub soff: i64,
}

impl<const L: usize> Machine<L> {
    // model: postlude/tick.sail execute_element
    pub(crate) fn execute_element<O: Observer + ?Sized>(&mut self, e: &ExecElement, obs: &mut O) {
        let lat = self.latency_of_instr(&e.instr);
        self.execute_element_with(e, lat, obs)
    }

    /// [`Machine::execute_element`] for an element whose latency the caller has looked up already.
    fn execute_element_with<O: Observer + ?Sized>(&mut self, e: &ExecElement, lat: Latency, obs: &mut O) {
        let ctx =
            Issue { cycle: e.c0, queue: e.q, element: e.k, addr_offset: e.off, stream_offset: e.soff, thread: 0, lat };
        let m = phase::begin();
        let prof = obs.profiling();
        if prof {
            obs.exec_element(e);
            obs.exec_begin(e.instr.id().index(), e.q, ctx.exec_cycle());
        }
        if self.iq.observe_sites {
            obs.site(self.chip_id(), ctx.exec_cycle(), e.q, e.instr.entry().name, format_args!("{} {}", e.c0, e.k));
        }
        let live = self.ecc.live;
        if live {
            self.ecc.begin();
        }
        self.ecc.data_opr.set(false);
        let x = self.execute(&e.instr, &ctx, obs);
        phase::sem_end(phase::SEM, m);
        if live {
            self.ecc_finish(e.q, ctx.exec_cycle(), x, obs);
        } else if let Exec::Diagnostic(d) = x {
            self.diag(obs, e.q, ctx.exec_cycle(), d);
        }
        if prof {
            obs.exec_end();
        }
    }

    /// Issue one reference of the instruction dispatched at cycle `c0`, `k` cycles later, with an
    /// AGT address offset and stream offset.
    // model: postlude/tick.sail run_element
    #[allow(clippy::too_many_arguments)] // the arguments of the model's run_element
    pub fn run_element<O: Observer + ?Sized>(
        &mut self,
        q: usize,
        i: Insn,
        c0: u64,
        k: u64,
        off: i64,
        soff: i64,
        obs: &mut O,
    ) {
        let e = ExecElement { q, instr: i, c0, k, off, soff };
        let lat = self.latency_of_instr(&i);
        let s = lat.d_skew as u64;
        if s == 0 {
            self.execute_element_with(&e, lat, obs);
        } else {
            let x = ((u128::from(c0) + u128::from(k) + u128::from(s)) % EXEC_SLOTS as u128) as usize;
            self.d.execute_ring[x].push(e);
            self.d.exec_occ |= 1 << x;
        }
    }

    /// The elements whose execute cycle is `c` (`d_skew <= 15`, so the 16-slot ring never aliases).
    // model: postlude/tick.sail execute_due
    pub fn execute_due<O: Observer + ?Sized>(&mut self, c: u64, obs: &mut O) {
        let x = (c % EXEC_SLOTS as u64) as usize;
        if self.d.execute_ring[x].is_empty() {
            return;
        }
        self.d.exec_occ &= !(1 << x);
        let mut xs = std::mem::take(&mut self.d.execute_ring[x]);
        for e in &xs {
            self.execute_element(e, obs);
        }
        if self.d.execute_ring[x].is_empty() {
            xs.clear();
            self.d.execute_ring[x] = xs; // the slot keeps its allocation
        }
    }
}
