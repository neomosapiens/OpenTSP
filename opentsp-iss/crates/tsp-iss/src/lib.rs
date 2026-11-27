//! The opentsp instruction-set simulator for specification 0.1.
//!
//! A structural port of the Sail model (`opentsp-arch-spec/model/`), drop-in compatible with the
//! reference harness `tsp_sim` (`opentsp-arch-spec/c_emulator/tsp_sim.c`): the same argv, stdin
//! protocol, stdout bytes and exit codes (`opentsp-iss/README.md`).
//!
//! Modules: the harness (`harness`: options, input, output, link wiring, several chips in lock-step
//! windows), the configuration (`geom`, `ots-profile`), the front end (`frontend`, `layout`), the cycle
//! driver (`driver`: classify, holds, barrier, Repeat, execute ring, commit ring, control channel,
//! dispatch, `model_next_event`), the units (`slices`: CU, MEM, VXM, MXM, SXM, C2C), the stream register
//! file (`srf`), the operand functions (`operands`), SRAM access and the tile masks (`sram`), the link
//! FIFOs (`links`) and the native subcommands `run` and `vectors` (`native`): `.elf`, `.safetensors` and Q
//! text through `ots-format`, and the golden vectors in-process, and the host-visible platform
//! registers and flows of one chip (`platform`). The `T srf` and `T sram` events reach the observer.
pub mod copy;
pub mod ctl;
pub mod driver;
pub mod ecc;
mod edit;
pub use edit::{EccView, StateEditor};
pub mod frontend;
pub mod geom;
pub mod harness;
pub mod host;
pub mod info;
pub mod layout;
pub mod links;
pub mod machine;
pub mod native;
pub mod observe;
pub mod operands;
pub mod platform;
pub mod scrub;
pub mod slices;
pub mod sram;
pub mod srf;
pub mod verify;

pub use geom::Geom;
pub use links::{LinkPort, NoLinks};
pub use machine::{LoadError, Machine, QueueState, RunError, RunSummary};
pub use observe::{Hold, NoObserver, Observer, Row};
// model: core/diagnostics.sail diag, diag_code
pub use ots_isa::Code as Diag;
