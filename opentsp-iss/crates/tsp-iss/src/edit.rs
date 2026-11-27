//! Explicit editing of debugger state, separated from the hot execution path.
use crate::driver::Driver;
use crate::ecc::Ecc;
use crate::frontend::Iq;
use crate::links::LinkPort;
use crate::slices::{activity::Activity, mxm::Mxm, sxm::Sxm};
use crate::sram::{Masks, Sram};
use crate::srf::Srf;
use crate::Machine;
use ots_isa::Isa;
use ots_profile::TimingTable;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Architectural state that a debugger may edit. Profile geometry, ISA and timing are immutable:
/// construct a new machine to change them. Derived scheduler structures are rebuilt afterwards.
pub struct StateEditor<'a, const L: usize> {
    pub geom: &'a crate::Geom,
    pub isa: &'a Isa,
    pub timing: &'a TimingTable,
    pub driver: &'a mut Driver<L>,
    pub iq: &'a mut Iq,
    pub masks: &'a mut Masks,
    pub sram: &'a mut Sram,
    pub srf: &'a mut Srf,
    pub mxm: &'a mut Mxm,
    pub sxm: &'a mut Sxm,
    pub mem_segment: &'a mut [u64; 88],
    pub c2c_accepted: &'a mut [Option<u64>; crate::geom::MAX_LINKS],
    pub ecc: &'a mut Ecc,
    pub links: &'a mut Box<dyn LinkPort>,
    pub activity: &'a mut Activity,
}

struct Repair<'a, const L: usize>(&'a mut Machine<L>);
impl<const L: usize> Drop for Repair<'_, L> {
    fn drop(&mut self) {
        // Editing may unwind. Rebuild during unwinding too, and never double-panic if a
        // debugger left an impossible raw state: subsequent execution reports the error.
        let machine = &mut *self.0;
        let repaired = catch_unwind(AssertUnwindSafe(|| {
            machine.iq.rebuild_after_edit(&machine.isa)?;
            machine.d.rebuild_after_edit();
            // A raw edit can introduce work that is already due. Revisit every queue
            // on the next tick; wake_of assumes the queue has already been visited.
            machine.rp = crate::driver::replay::Rp::new();
            machine.cursor = machine.last_visited.map_or(0, |c| u128::from(c) + 1);
            machine.t_next = match machine.last_visited {
                Some(cycle) => machine.model_next_event(cycle),
                None => Some(u128::from(machine.host.start)),
            };
            Ok::<(), String>(())
        }));
        machine.edit_failure = match repaired {
            Ok(Ok(())) => None,
            Ok(Err(message)) => Some(message),
            Err(_) => Some("state edit left an invalid architectural state".into()),
        };
    }
}

impl<const L: usize> Machine<L> {
    pub fn sram(&self) -> &Sram {
        &self.sram
    }
    pub fn srf(&self) -> &Srf {
        &self.srf
    }
    pub fn mxm(&self) -> &Mxm {
        &self.mxm
    }
    pub fn sxm(&self) -> &Sxm {
        &self.sxm
    }
    pub fn mem_segments(&self) -> &[u64; 88] {
        &self.mem_segment
    }
    pub fn c2c_accepted(&self) -> &[Option<u64>; crate::geom::MAX_LINKS] {
        &self.c2c_accepted
    }
    pub fn replay_state(&self) -> &crate::driver::replay::Rp {
        &self.rp
    }
    pub fn activity(&self) -> &Activity {
        &self.activity
    }
    pub fn ecc(&self) -> EccView<'_> {
        EccView(&self.ecc)
    }
    pub fn driver(&self) -> &Driver<L> {
        &self.d
    }
    pub fn iq(&self) -> &Iq {
        &self.iq
    }
    pub fn masks(&self) -> &Masks {
        &self.masks
    }
    pub fn isa(&self) -> &Isa {
        &self.isa
    }
    pub fn timing(&self) -> &TimingTable {
        &self.timing
    }

    /// Edit architectural state through one invalidation boundary. A panic in `edit` is
    /// propagated after rebuilding the caches; an invalid resulting shape is returned as an
    /// error and execution is blocked until a successful edit or reset. Recorded/replayed
    /// schedules are discarded because their history no longer describes this state.
    pub fn edit_state<R>(&mut self, edit: impl FnOnce(StateEditor<'_, L>) -> R) -> Result<R, String> {
        self.mxm_bf16_weights.iter_mut().for_each(|entry| *entry = None);
        let repair = Repair(self);
        let out = edit(StateEditor {
            geom: &repair.0.geom,
            isa: &repair.0.isa,
            timing: &repair.0.timing,
            driver: &mut repair.0.d,
            iq: &mut repair.0.iq,
            masks: &mut repair.0.masks,
            sram: &mut repair.0.sram,
            srf: &mut repair.0.srf,
            mxm: &mut repair.0.mxm,
            sxm: &mut repair.0.sxm,
            mem_segment: &mut repair.0.mem_segment,
            c2c_accepted: &mut repair.0.c2c_accepted,
            ecc: &mut repair.0.ecc,
            links: &mut repair.0.links,
            activity: &mut repair.0.activity,
        });
        drop(repair);
        match &self.edit_failure {
            Some(message) => Err(message.clone()),
            None => Ok(out),
        }
    }
}

/// Read-only integrity state; mutable cells and event buffers are not exposed.
pub struct EccView<'a>(&'a Ecc);
impl EccView<'_> {
    pub fn live(&self) -> bool {
        self.0.live
    }
    pub fn sram_ov(&self, slice: usize, addr: u32) -> u32 {
        self.0.sram_ov(slice, addr)
    }
    pub fn get(&self, id: u32, superlane: usize) -> u16 {
        self.0.get(id, superlane)
    }
}
