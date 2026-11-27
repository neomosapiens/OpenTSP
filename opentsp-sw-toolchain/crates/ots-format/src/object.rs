//! Resolved program objects carried in OpenTSP ELF32 sections.
use crate::{
    externs::{Extern, ExternIndex},
    load_profile,
    text::{raise_images, Images},
};
use ots_isa::{queue_key, queue_name, AsmError, Code, Isa, ENCODING_JSON, PADDING};
use ots_profile::Profile;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const MAGIC: &[u8; 4] = b"\x7fELF";
/// Bumped by every change to the object or weight-package format.
pub const FORMAT_REVISION: u32 = 1;
pub const HEADER_SIZE: usize = 52;
pub const IMAGE_ENTRY_SIZE: usize = 8;
pub const EXTERN_ENTRY_SIZE: usize = 32;

fn bad(m: impl Into<String>) -> AsmError {
    AsmError::new(Code::Asm003, m)
}

fn le16(d: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(d[at..at + 2].try_into().unwrap())
}

fn le32(d: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(d[at..at + 4].try_into().unwrap())
}

/// The object of a program: `images` in text order (ascending queue, control channel last), the
/// `externs` in declaration order, for `profile` (its validated name fits the 16-byte field without truncation).
pub fn write_object(profile: &Profile, images: &Images, externs: &[Extern]) -> Vec<u8> {
    write_object_sections(profile, images, externs, &[])
}

pub fn write_object_sections(
    profile: &Profile,
    images: &Images,
    externs: &[Extern],
    extra: &[(&str, &[u8])],
) -> Vec<u8> {
    let mut note = Vec::new();
    note.extend(8u32.to_le_bytes());
    note.extend(84u32.to_le_bytes());
    note.extend(1u32.to_le_bytes());
    note.extend(crate::elf::NOTE_NAME);
    note.extend(1u16.to_le_bytes());
    note.extend(0u16.to_le_bytes());
    note.extend(name_field(profile.name()));
    note.extend(Sha256::digest(ENCODING_JSON));
    note.extend(Sha256::digest(profile.bytes()));
    let mut out = (images.len() as u32).to_le_bytes().to_vec();
    for (q, us) in images.iter() {
        out.extend(q.to_le_bytes());
        out.extend(0u16.to_le_bytes());
        out.extend((us.len() as u32).to_le_bytes());
    }
    for (_, us) in images.iter() {
        for u in us {
            out.extend(u.to_le_bytes());
        }
    }
    let mut placement = (externs.len() as u32).to_le_bytes().to_vec();
    for x in externs {
        placement.extend(name_field(&x.name));
        placement.extend((x.slice as u16).to_le_bytes());
        placement.extend(0u16.to_le_bytes());
        placement.extend((x.addr as u32).to_le_bytes());
        placement.extend((x.size as u32).to_le_bytes());
        placement.extend(0u32.to_le_bytes());
    }
    crate::elf::write_sections(&note, &out, &placement, extra)
}

/// Replace initial queue text while preserving all embedded load sources.
pub fn replace_images(
    data: &[u8],
    profile: &Profile,
    images: &Images,
    externs: &[Extern],
) -> Result<Vec<u8>, AsmError> {
    let sections = crate::elf::read(data)?;
    let extra: Vec<_> = [".tsp.text", ".tsp.const", ".tsp.zero", ".tsp.recipe"]
        .into_iter()
        .filter_map(|name| sections.get(name).map(|bytes| (name, bytes.as_slice())))
        .collect();
    Ok(write_object_sections(profile, images, externs, &extra))
}

/// `name` as 16 bytes, NUL-padded (cut at 16).
pub(crate) fn name_field(name: &str) -> [u8; 16] {
    let mut b = [0u8; 16];
    let n = name.len().min(16);
    b[..n].copy_from_slice(&name.as_bytes()[..n]);
    b
}

/// A NUL-padded ASCII field: the text, then only NUL bytes (`ASM-003` otherwise, and for an empty
/// text).
pub(crate) fn ascii_field(raw: &[u8], what: &str) -> Result<String, AsmError> {
    let end = raw.iter().position(|b| *b == 0).unwrap_or(raw.len());
    let (text, rest) = raw.split_at(end);
    if rest.iter().any(|b| *b != 0) || text.is_empty() {
        return Err(bad(format!("{what}: not NUL-padded text")));
    }
    if !text.is_ascii() {
        return Err(bad(format!("{what}: not ASCII")));
    }
    Ok(String::from_utf8(text.to_vec()).unwrap())
}

/// `(profile name, images, externs)` of a binary object. `profile` is `--profile` (when given, the
/// object's profile must be it) and `config` is `--config` (the file that stands for the profile's
/// `config/<name>.json`, for the profile digest and for the checks that depend on the profile).
///
/// Rejects, with `OTS-E-ASM-003` and no location (the command names it), every other format and
/// version and every entry that breaks `assembly.md` §4, in this order: an ELF file; a length under 64
/// or another magic; the version and the header size; the format revision and the reserved word; the
/// profile field, then `--profile`, then the profile itself (unknown, or of another version); the two
/// digests; a length under the header plus the image table; a length that is not the header, the table,
/// the units and the externs; per image, in table order: the pad, the queue, the unit count, the
/// canonical order and `icu.iq_depth`; the parcels that do not fit the queue's slots; per extern, in
/// order: the pad fields, the name and the extern rules ([`check_extern`]).
pub fn read_object(
    data: &[u8],
    profile: Option<&str>,
    config: Option<&str>,
) -> Result<(String, Images, Vec<Extern>), AsmError> {
    let wire = crate::elf::read(data)?;
    let note = &wire[".note.opentsp"];
    if note.len() != 104
        || le32(note, 0) != 8
        || le32(note, 4) != 84
        || le32(note, 8) != 1
        || &note[12..20] != crate::elf::NOTE_NAME
        || le16(note, 20) != 1
        || le16(note, 22) != 0
    {
        return Err(bad("invalid OpenTSP ELF ABI note"));
    }
    let name = ascii_field(&note[24..40], "profile")?;
    if profile.is_some_and(|p| p != name) {
        return Err(bad(format!("object is for profile {name:?}, not {:?}", profile.unwrap())));
    }
    let prof = load_profile(&name, config)?;
    if note[40..72] != Sha256::digest(ENCODING_JSON)[..] {
        return Err(bad("object was built against another encoding.json"));
    }
    if note[72..104] != Sha256::digest(prof.bytes())[..] {
        return Err(bad(format!("object was built against another configuration of profile {name:?}")));
    }
    let data = &wire[".tsp.iq"];
    let loads = &wire[".tsp.load"];
    if data.len() < 4 || loads.len() < 4 {
        return Err(bad("truncated OpenTSP section"));
    }
    let nimg = le32(data, 0) as usize;
    let next = le32(loads, 0) as usize;
    let table_end = 4 + nimg * IMAGE_ENTRY_SIZE;
    if nimg > 145 || data.len() < table_end || loads.len() != 4 + next * EXTERN_ENTRY_SIZE {
        return Err(bad("ELF section length does not match its directory"));
    }
    let table: Vec<_> = (0..nimg)
        .map(|k| 4 + k * IMAGE_ENTRY_SIZE)
        .map(|at| (le16(data, at), le16(data, at + 2), le32(data, at + 4)))
        .collect();
    let nunits: u128 = table.iter().map(|t| t.2 as u128).sum();
    if data.len() as u128 != table_end as u128 + 4 * nunits {
        return Err(bad("ELF IQ section length differs"));
    }
    let isa = Isa::new(&prof);
    let cap = (prof.iq_depth() / 4) as usize;
    let mut images = Images::new();
    let mut p = table_end;
    let mut prev: Option<(bool, u16)> = None;
    for (q, z, n) in table {
        let n = n as usize;
        if z != 0 || queue_name(q).is_none() || n == 0 {
            return Err(bad(format!("bad image entry (queue {q:#x}, {n} units)")));
        }
        let key = queue_key(q);
        if prev.is_some_and(|k| k >= key) {
            return Err(bad(format!(
                "image of queue {q:#x}: not in canonical order (ascending queue, control channel last) or duplicated"
            )));
        }
        prev = Some(key);
        let us: Vec<u32> = data[p..p + 4 * n].as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
        p += 4 * n;
        if us.iter().filter(|u| **u != PADDING).count() > cap {
            return Err(bad(format!("image of queue {q:#x}: {n} units do not fit icu.iq_depth ({cap} units)")));
        }
        images.insert(q, us);
    }
    // The parcels of an image must fit the queue's slots (microarchitecture.md §3, "Parcels and
    // slots"; encoding.json `mem_slots`), as in the assembler.
    let mut groups: BTreeMap<(u16, i64), Vec<(u64, u8)>> = BTreeMap::new();
    for r in raise_images(&images) {
        groups.entry((r.queue, r.cycle)).or_default().push((r.word, r.units));
    }
    for ((q, _), g) in &groups {
        if g.len() > 1 {
            let dec: Vec<_> = g.iter().map(|(w, nu)| isa.decode_word(*w, *nu).ok()).collect();
            if isa.group_codes(*q, &dec).contains(&Some(Code::Que001)) {
                return Err(bad(format!(
                    "{} instructions in one parcel of queue {q:#x} do not fit the queue's slots (microarchitecture.md section 3, Instruction front end)",
                    g.len()
                )));
            }
        }
    }
    let mut externs: Vec<Extern> = Vec::with_capacity(next);
    let mut seen = ExternIndex::new();
    p = 4;
    for i in 0..next {
        let e = &loads[p..p + EXTERN_ENTRY_SIZE];
        p += EXTERN_ENTRY_SIZE;
        if le16(e, 18) != 0 || le32(e, 28) != 0 {
            return Err(bad(format!("extern entry {i}: non-zero pad field")));
        }
        let x = Extern {
            name: ascii_field(&e[..16], &format!("extern entry {i} name"))?,
            slice: le16(e, 16) as u64,
            addr: le32(e, 20) as u64,
            size: le32(e, 24) as u64,
        };
        seen.check(&x, &prof).map_err(|e| bad(format!("extern entry {i}: {}", e.msg)))?;
        seen.push(&x);
        externs.push(x);
    }
    debug_assert_eq!(p, loads.len());
    Ok((name, images, externs))
}
