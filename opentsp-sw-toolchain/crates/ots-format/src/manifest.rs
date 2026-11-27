//! Manifests, `.otsm` (`assembly.md` §4 "Manifest"): the multi-chip container.
//! A manifest names the K chips of a run and embeds one object per chip, all of one profile. Ports of
//! `write_manifest`, `read_manifest`, `manifest_input` and `chip_count_ok` of
//! `opentsp-arch-spec/tools/tsp_as.py`. Everything is little-endian.
//!
//! ```text
//! header (64 bytes)  magic "OTSM" | major 0 | minor 1 | header bytes u16 = 64 | profile (16 bytes,
//!                    NUL-padded ASCII) | chip count K u32 | topology u32 = 0 | format revision u32 = 2 |
//!                    reserved u32 = 0 | encoding digest (12 bytes) | profile digest (12 bytes)
//! entries            K x (offset u32 from the start of the file, size u32); entry 0 starts at 64 + 8 K,
//!                    entry i + 1 where entry i ends, the last at the end of the file
//! objects            the K embedded ELF objects back to back in chip order
//! ```
//!
//! The header's digests are those of an object header. The manifest has no start
//! offsets and no weights: a run takes one weight package per chip, in chip order.
use crate::{
    load_profile,
    object::{ascii_field, name_field, read_object},
    text::input_text,
    weights::{bind_weights, merge_preloads, object_preloads, preload_text, read_weights},
};
use ots_isa::{encoding_digest12, AsmError, Code};

pub const MAGIC: &[u8; 4] = b"OTSM";
/// Bumped by every change to the manifest format.
pub const FORMAT_REVISION: u32 = 2;
/// The `tsp_sim --chips K` wiring (`microarchitecture.md` §10), the only topology of 0.1.
pub const TOPOLOGY_NODE: u32 = 0;
pub const HEADER_SIZE: usize = 64;
pub const ENTRY_SIZE: usize = 8;
/// Major and minor of specification 0.1.
const SPEC_VERSION: (u8, u8) = (0, 1);

fn bad(m: impl Into<String>) -> AsmError {
    AsmError::new(Code::Asm003, m)
}

fn le32(d: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(d[at..at + 4].try_into().unwrap())
}

/// The most chips of the `--chips K` wiring: 33 nodes of 8 chips (ISCA22 sec. 2.2).
pub const MAX_WIRED_CHIPS: u64 = 264;
/// The most chips any topology names (`chip_id` is 9 bits).
pub const MAX_CHIPS: u64 = 512;

/// `K` chips of a manifest as a reader takes them: 1..=512. The wiring is the harness's: `--chips K` (a Dragonfly of nodes
/// of 8 chips, the last may hold fewer) takes 1..=264, a topology file up to 512.
pub fn chip_count_ok(k: u64) -> bool {
    (1..=MAX_CHIPS).contains(&k)
}

/// The manifest of `objects` (one per chip, in chip order) for `profile`; `config` is `--config`.
///
/// The count is checked first (`OTS-E-ASM-003`), then every object is read ([`read_object`], with
/// `profile` and `config`): it must be a valid object of the profile, and the message names the chip.
/// The offsets are those of the layout; the digests are those of the profile as loaded (`config`, or the
/// shipped file). Profile names longer than 16 ASCII bytes are rejected.
pub fn write_manifest(profile: &str, objects: &[Vec<u8>], config: Option<&str>) -> Result<Vec<u8>, AsmError> {
    write_manifest_for(profile, objects, config, MAX_WIRED_CHIPS)
}

/// [`write_manifest`] for a system of at most `max_chips` chips: [`MAX_WIRED_CHIPS`] for the `--chips K` wiring, up to
/// [`MAX_CHIPS`] for a topology file (`mpack --topology`).
pub fn write_manifest_for(
    profile: &str,
    objects: &[Vec<u8>],
    config: Option<&str>,
    max_chips: u64,
) -> Result<Vec<u8>, AsmError> {
    if !(1..=max_chips.min(MAX_CHIPS)).contains(&(objects.len() as u64)) {
        let what = if max_chips > MAX_WIRED_CHIPS {
            "a topology file names"
        } else {
            "a manifest of the --chips wiring holds"
        };
        return Err(bad(format!("{} chips: {what} 1..{} objects", objects.len(), max_chips.min(MAX_CHIPS))));
    }
    for (i, o) in objects.iter().enumerate() {
        read_object(o, Some(profile), config).map_err(|e| AsmError::new(e.code, format!("chip {i}: {}", e.msg)))?;
    }
    // every object is of `profile`, so the profile loads
    let prof = load_profile(profile, config)?;
    let total: usize = objects.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(HEADER_SIZE + objects.len() * ENTRY_SIZE + total);
    out.extend_from_slice(MAGIC);
    out.extend([SPEC_VERSION.0, SPEC_VERSION.1]);
    out.extend((HEADER_SIZE as u16).to_le_bytes());
    out.extend(name_field(profile));
    out.extend((objects.len() as u32).to_le_bytes());
    out.extend(TOPOLOGY_NODE.to_le_bytes());
    out.extend(FORMAT_REVISION.to_le_bytes());
    out.extend(0u32.to_le_bytes());
    out.extend(encoding_digest12());
    out.extend(prof.digest12());
    debug_assert_eq!(out.len(), HEADER_SIZE);
    let mut off = (HEADER_SIZE + objects.len() * ENTRY_SIZE) as u64;
    for o in objects {
        out.extend((off as u32).to_le_bytes());
        out.extend((o.len() as u32).to_le_bytes());
        off += o.len() as u64;
    }
    for o in objects {
        out.extend_from_slice(o);
    }
    Ok(out)
}

/// `(profile name, the K objects)` of a manifest. `profile` is `--profile` (when given, the manifest's
/// profile must be it) and `config` is `--config` (the file that stands for the profile's
/// `config/<name>.json`, for the profile digest and for the checks of the embedded objects).
///
/// Rejects, with `OTS-E-ASM-003` and no location (the command names it), in this order: a length under
/// 64 or another magic; the version and the header size; the format revision and the reserved word; the
/// topology; the profile field, then `profile`, then the profile itself (unknown, or of another
/// version); the two digests; a chip count that is not [`chip_count_ok`]; a length under the header plus
/// the entries; per entry, in order: an offset that is not the end of the previous entry (the first:
/// the end of the table), a size of 0, an entry that runs past the file, and an embedded object that is
/// not a valid object of the profile ([`read_object`]; the code is the object's, the message names the
/// entry); at last a file longer than the last entry.
pub fn read_manifest<'a>(
    data: &'a [u8],
    profile: Option<&str>,
    config: Option<&str>,
) -> Result<(String, Vec<&'a [u8]>), AsmError> {
    if data.len() < HEADER_SIZE || &data[..4] != MAGIC {
        return Err(bad("not an OTSM manifest (bad magic)"));
    }
    let (major, minor, hsize) = (data[4], data[5], u16::from_le_bytes([data[6], data[7]]) as usize);
    if (major, minor) != SPEC_VERSION || hsize != HEADER_SIZE {
        return Err(bad(format!("manifest is spec version {major}.{minor}, this tool reads 0.1 only")));
    }
    let (k, topology, rev, reserved) = (le32(data, 24) as u64, le32(data, 28), le32(data, 32), le32(data, 36));
    if rev != FORMAT_REVISION || reserved != 0 {
        return Err(bad(format!(
            "manifest format revision {rev} (reserved {reserved:#x}); this tool reads revision {FORMAT_REVISION}"
        )));
    }
    if topology != TOPOLOGY_NODE {
        return Err(bad(format!("manifest topology {topology}: 0 is the only topology of 0.1")));
    }
    let name = ascii_field(&data[8..24], "manifest profile")?;
    if profile.is_some_and(|p| p != name) {
        return Err(bad(format!("manifest is for profile {name:?}, not {:?}", profile.unwrap())));
    }
    let prof = load_profile(&name, config)?;
    if data[40..52] != encoding_digest12() {
        return Err(bad("manifest was built against another encoding.json (queue map or encodings)"));
    }
    if data[52..64] != prof.digest12() {
        return Err(bad(format!("manifest was built against another configuration of profile {name:?}")));
    }
    if !(1..=MAX_CHIPS).contains(&k) {
        return Err(bad(format!("manifest chip count {k}: must be 1..{MAX_CHIPS}")));
    }
    let table_end = HEADER_SIZE as u64 + k * ENTRY_SIZE as u64;
    if (data.len() as u64) < table_end {
        return Err(bad("manifest size does not match its header"));
    }
    let mut expect = table_end;
    let mut objects = Vec::with_capacity(k as usize);
    for i in 0..k as usize {
        let at = HEADER_SIZE + i * ENTRY_SIZE;
        let (off, size) = (le32(data, at) as u64, le32(data, at + 4) as u64);
        if off != expect || size == 0 || off + size > data.len() as u64 {
            return Err(bad(format!("manifest entry {i}: offset or size does not follow the previous entry")));
        }
        let o = &data[off as usize..(off + size) as usize];
        read_object(o, Some(&name), config)
            .map_err(|e| AsmError::new(e.code, format!("manifest entry {i}: {}", e.msg)))?;
        objects.push(o);
        expect = off + size;
    }
    if expect != data.len() as u64 {
        return Err(bad("manifest size does not match its entries"));
    }
    Ok((name, objects))
}

/// The `tsp_sim --chips K` input of a manifest (`mobjinput`): for every chip i, in order, a `C <i>` line
/// (hex), the `S` lines of its externs filled from `weight_packages[i]` (when packages are given:
/// exactly one per chip, in chip order, else `OTS-E-ASM-004`; none: no `S` lines) and the `Q` lines of its
/// images, as `objinput` prints them for one object. The manifest is read first ([`read_manifest`]);
/// nothing is returned when a chip fails.
pub fn manifest_input(data: &[u8], weight_packages: &[Vec<u8>], config: Option<&str>) -> Result<String, AsmError> {
    let (name, objects) = read_manifest(data, None, config)?;
    if !weight_packages.is_empty() && weight_packages.len() != objects.len() {
        return Err(AsmError::new(
            Code::Asm004,
            format!("{} weight packages for {} chips: one per chip", weight_packages.len(), objects.len()),
        ));
    }
    let mut out = String::new();
    for (i, o) in objects.iter().enumerate() {
        let (_, images, externs) = read_object(o, Some(&name), config)?;
        out.push_str(&format!("C {i:x}\n"));
        let mut tensors = object_preloads(o, &name, &externs, config, 1)?;
        if let Some(pk) = weight_packages.get(i) {
            merge_preloads(&mut tensors, read_weights(pk, Some(&name), config)?)?;
            out.push_str(&preload_text(&bind_weights(&externs, &tensors)?));
        } else if !tensors.is_empty() {
            let embedded: Vec<_> = externs.into_iter().filter(|x| tensors.contains_key(&x.name)).collect();
            out.push_str(&preload_text(&bind_weights(&embedded, &tensors)?));
        }
        out.push_str(&input_text(&images));
    }
    Ok(out)
}
