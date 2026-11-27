//! The formats of opentsp specification 0.1 around the instruction text: the lowering of a cycle
//! view to per-queue text and its inverse, the harness's `Q` lines (`text`); objects, revision 2
//! (`object`); weight packages, `.safetensors` (`weights`); multi-chip manifests, `.otsm` (`manifest`; `assembly.md` §4).
//!
//! Ports of `lower`, `raise_images`, `word_units`, `q_lines`, `parse_images`, `is_text_input`,
//! `write_object`, `read_object`, `write_weights`, `read_weights`, `bind_weights`, `preload_text`, `write_manifest`, `read_manifest` and `manifest_input` of `opentsp-arch-spec/tools/tsp_as.py`; the normative lowering is
//! `assembly.md` §7.9.
mod elf;
pub mod externs;
pub mod manifest;
pub mod object;
pub mod text;
pub mod weights;

pub use externs::{check_extern, is_ident, Extern};
pub use manifest::{manifest_input, read_manifest, write_manifest, write_manifest_for, MAX_CHIPS, MAX_WIRED_CHIPS};
pub use object::{read_object, write_object};
pub use ots_isa::AsmError;
pub use weights::{bind_weights, merge_preloads, object_preloads, preload_text, read_weights, write_weights};

use ots_isa::Code;
use ots_profile::Profile;

/// The profile of a name, or of the file `--config` names (`tsp_as.load_profile`): an unknown
/// name, a missing file, a profile of another specification version or one without a key of a profile variable is `OTS-E-ASM-003`.
pub fn load_profile(name: &str, config: Option<&str>) -> Result<Profile, AsmError> {
    let r = match config {
        Some(path) => Profile::load_file(name, path),
        None => Profile::load(name),
    };
    let profile = r.map_err(|e| AsmError::new(Code::Asm003, e.to_string()))?;
    profile.check_assembler_timing().map_err(|e| AsmError::new(Code::Asm003, e))?;
    // a file read without validation may lack a key an operand bound reads (`tsp_as.py` refuses it the same way)
    if let Some((name, expr)) = profile.missing_variable() {
        return Err(AsmError::new(
            Code::Asm003,
            format!("profile {}: {expr}: no value for {name}, a key of its expression is missing", profile.name()),
        ));
    }
    Ok(profile)
}
