//! The units' instructions (`sys/insts_begin.sail`, `postlude/insts_end.sail`, `slices/*`). One
//! module per slice; `execute` dispatches on the catalogue entry.
//!
//! The units are the control unit (`cu`), the MEM slices (`mem`), the VXM (`vxm`), the MXM (`mxm`),
//! the SXM (`sxm`) and the C2C unit (`c2c`): every unit's instructions are executed.
pub mod activity;
pub mod c2c;
pub mod cu;
pub mod energy;
pub mod mem;
pub mod mxm;
pub mod mxm_check;
#[cfg(feature = "opprof")]
pub mod opprof;
pub mod pool;
pub mod sxm;
pub mod vxm;

use crate::observe::Observer;
use crate::Machine;
use ots_isa::{Code as Diag, Insn, ENTRIES};
pub use ots_profile::Latency;
use std::sync::LazyLock;

/// What issuing an instruction did (`ExecutionResult`): diagnostics are values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exec {
    Issued,
    Diagnostic(Diag),
}

/// The context of the current issue (the `issue_*` registers of `sys/insts_begin.sail`): the
/// dispatch cycle, the issuing queue (144: the control channel), the cycle offset and address and
/// stream offsets of an iterated MEM reference, the liquid-thread tag and the entry's latency.
#[derive(Clone, Copy, Debug)]
pub struct Issue {
    pub cycle: u64,
    pub queue: usize,
    pub element: u64,
    pub addr_offset: i64,
    pub stream_offset: i64,
    pub thread: u8,
    pub lat: Latency,
}

impl Issue {
    /// The cycle the current issue executes: its dispatch (reference k of a compound instruction:
    /// dispatch + k) + d_skew. It reads its stream operands, SRAM and unit state in that cycle.
    // model: sys/insts_begin.sail exec_cycle
    pub fn exec_cycle(&self) -> u64 {
        self.cycle + self.element + self.lat.d_skew as u64
    }

    /// The cycle its stream results commit: dispatch (+ k) + d_func.
    // model: sys/insts_begin.sail commit_at
    pub fn commit_at(&self) -> u128 {
        u128::from(self.cycle) + u128::from(self.element) + u128::from(self.lat.d_func)
    }
}

/// The unit that executes each catalogue entry, by `EntryId::index()` (the codes of `UNIT_*`), looked up once per
/// instruction instead of comparing the entry's slice name.
static UNIT_OF: LazyLock<[u8; ENTRIES.len()]> = LazyLock::new(|| {
    let mut t = [0u8; ENTRIES.len()];
    for e in ENTRIES.iter() {
        t[e.id.index()] = match e.slice {
            "CU" => UNIT_CU,
            "MEM" => UNIT_MEM,
            "VXM" => UNIT_VXM,
            "MXM" => UNIT_MXM,
            "SXM" => UNIT_SXM,
            "C2C" => UNIT_C2C,
            other => unreachable!("no unit {other} for the entry {:?}", e.id),
        };
    }
    t
});
const UNIT_CU: u8 = 0;
const UNIT_MEM: u8 = 1;
const UNIT_VXM: u8 = 2;
const UNIT_MXM: u8 = 3;
const UNIT_SXM: u8 = 4;
const UNIT_C2C: u8 = 5;

impl<const L: usize> Machine<L> {
    /// The latency of the instruction: the row of its element type (`timing.overrides` row `<Entry>.<type>`), else its
    /// catalogue entry's row (`timing.overrides`, else its class).
    // model: sys/insts_begin.sail latency_of_instr
    pub fn latency_of_instr(&self, i: &Insn) -> Latency {
        let index = i.id().index();
        if self.timing.has_typed(index) {
            if let Some(code) = i.get("type") {
                return self.timing.row_typed(index, code);
            }
        }
        self.timing.row(index)
    }

    /// Execute `i` in the context `ctx`.
    // model: sys/insts_begin.sail execute
    pub fn execute<O: Observer + ?Sized>(&mut self, i: &Insn, ctx: &Issue, obs: &mut O) -> Exec {
        #[cfg(feature = "opprof")]
        {
            opprof::timed(i.id().index(), || self.execute_unit(i, ctx, obs))
        }
        #[cfg(not(feature = "opprof"))]
        self.execute_unit(i, ctx, obs)
    }

    #[inline(always)]
    fn execute_unit<O: Observer + ?Sized>(&mut self, i: &Insn, ctx: &Issue, obs: &mut O) -> Exec {
        // the activity counters (`activity`) count an instruction that issued, once, in the arm of the unit that ran it, when
        // counting is on: the flag is read only for an instruction that issued
        match UNIT_OF[i.id().index()] {
            UNIT_CU => {
                let x = self.execute_cu(i, ctx, obs);
                if x == Exec::Issued && self.activity.enabled() {
                    self.note_cu(i, ctx);
                }
                x
            }
            UNIT_MEM => {
                let x = self.execute_mem(i, ctx, obs);
                if x == Exec::Issued && self.activity.enabled() {
                    self.note_mem(i, ctx);
                }
                x
            }
            UNIT_VXM => {
                let x = self.execute_vxm(i, ctx, obs);
                if x == Exec::Issued && self.activity.enabled() {
                    self.note_vxm(i, ctx);
                }
                x
            }
            UNIT_MXM => {
                let x = self.execute_mxm(i, ctx, obs);
                if x == Exec::Issued && self.activity.enabled() {
                    self.note_mxm(i, ctx);
                }
                x
            }
            UNIT_SXM => {
                let x = self.execute_sxm(i, ctx, obs);
                if x == Exec::Issued && self.activity.enabled() {
                    self.note_sxm(i, ctx);
                }
                x
            }
            UNIT_C2C => {
                let x = self.execute_c2c(i, ctx, obs);
                if x == Exec::Issued && self.activity.enabled() {
                    self.note_c2c(i, ctx);
                }
                x
            }
            _ => unreachable!("no unit for the entry {:?}", i.id()),
        }
    }
}
