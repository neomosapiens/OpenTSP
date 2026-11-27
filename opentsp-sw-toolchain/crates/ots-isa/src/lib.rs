//! The instruction set of opentsp specification 0.1: tables, decoder, encoder and renderer
//! generated at build time from `opentsp-arch-spec/isa-db/generated/encoding.json`.
//!
//! - [`Insn`]: one variant per catalogue entry, one `u32` per operand (named as in the catalogue);
//!   [`EntryId`] and [`ENTRIES`] describe the entries ([`Entry`], [`Var`], [`KINDS`]).
//! - [`Isa`]: what depends on the profile (the `max_expr` of an operand, the SRAM size and bank
//!   bit): [`Isa::decode`], [`Isa::decode_text`] (the decode rule of the text layer), the
//!   per-parcel slot rule [`Isa::group_codes`], and the stations.
//! - Free functions that do not depend on the profile: [`encode`], [`render`], [`placement`],
//!   [`queue_id`], [`queue_name`], [`unit_queues`].
//!
//! The Sail model decides what a word means; `encoding.json` is generated from it and is read at
//! build time, so a change of the catalogue changes this crate with no hand edit. The ISS uses the
//! same decoder (`opentsp-iss/README.md` §4.7). Nothing here parses assembly
//! text: that is `ots-asm`.
mod error;
mod isa;
mod queues;
mod types;

pub use error::AsmError;
pub use isa::{encode, entry, placement, render, render_fields, Access, Isa, SlotSet};
pub use ots_profile::{Profile, TimingClass};
pub use queues::{canonical_queue_name, is_queue, queue_id, queue_key, queue_name, unit_queues, DATA_QUEUES};
pub use types::{Entry, Kind, MaxExpr, Station, StationExpr, Syntax, Var};

// build.rs output: generated tables, not hand-written code
#[allow(clippy::all)]
mod tables {
    use crate::types::*;
    include!(concat!(env!("OUT_DIR"), "/tables.rs"));
}
pub use tables::*;

/// The first 12 bytes of SHA-256 of `encoding.json` as built into this crate (the object's
/// encoding digest).
pub fn encoding_digest12() -> [u8; 12] {
    use sha2::{Digest, Sha256};
    let d: [u8; 32] = Sha256::digest(ENCODING_JSON).into();
    d[..12].try_into().unwrap()
}

impl EntryId {
    /// The position in the catalogue (the index into [`ENTRIES`] and into the timing table).
    pub fn index(self) -> usize {
        self as usize
    }

    /// The entry's table row.
    pub fn entry(self) -> &'static Entry {
        &ENTRIES[self as usize]
    }
}

impl Insn {
    /// The entry's table row.
    pub fn entry(&self) -> &'static Entry {
        self.id().entry()
    }

    /// An operand by its catalogue name.
    pub fn get(&self, name: &str) -> Option<u32> {
        let f = self.fields();
        self.entry().vars.iter().position(|v| v.name == name).map(|i| f[i])
    }
}

/// The number of text units the instruction at a queue's text starting with unit `u0` takes
/// (the slice tag, bits 30..28 of `u0`, gives the layout: narrow 1, wide 2), or `None` for a tag that
/// is neither (OTS-E-DEC-003, one unit consumed). The bundle bit is ignored.
pub fn decode_text_units(u0: u32) -> Option<u8> {
    let tag = ((u0 >> 28) & 7) as u8;
    if NARROW_TAGS.contains(&tag) {
        Some(1)
    } else if WIDE_TAGS.contains(&tag) {
        Some(2)
    } else {
        None
    }
}
