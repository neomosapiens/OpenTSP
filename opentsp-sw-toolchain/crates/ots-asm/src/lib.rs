//! The assembler and disassembler of opentsp specification 0.1 (`opentsp-sw-toolchain/README.md`).
//!
//! [`parse_insn`] reads one instruction of text, with the reject precedence of `assembly.md` §5;
//! [`decode_by_reparse`] is the decode rule of `tsp_as`, which the generated decoder of `ots-isa` is
//! held to. [`program`] reads the program file `.ots` and lowers it to the text of each queue,
//! [`checks`] holds the bundle checks and the AGT warnings, and [`dis`] lists text images with the
//! model's annotations, and [`patch`] rewrites operands of an object's text at the sites of a patch table. Objects, weight packages and manifests go through `ots_format::{object,
//! weights, manifest}`. The commands are in [`cli::run`].
//!
//! The behaviour is that of `opentsp-arch-spec/tools/tsp_as.py`, function by function.
pub mod checks;
pub mod cli;
pub mod dis;
pub mod insn;
pub mod patch;
pub mod program;
mod pynum;

pub use checks::Warning;
pub use dis::disassemble;
pub use insn::{decode_by_reparse, parse_insn};
pub use program::{assemble, Program, Record};

pub use ots_format::load_profile;
