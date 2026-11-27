//! Wire envelope for resolved OpenTSP ELF32 objects.
use ots_isa::{AsmError, Code};
use std::collections::BTreeMap;
use std::ffi::{c_int, c_uint, c_void};

const IDENT: &[u8; 16] = b"\x7fELF\x01\x01\x01\0\0\0\0\0\0\0\0\0";
const HEADER: usize = 52;
const SECTION: usize = 40;
const MAX_SECTION: usize = 64 * 1024 * 1024;
const COMPRESSED: u32 = 0x800;
pub(crate) const NOTE_NAME: &[u8] = b"OpenTSP\0";

#[link(name = "zstd")]
unsafe extern "C" {
    fn ZSTD_compressBound(size: usize) -> usize;
    fn ZSTD_compress(dst: *mut c_void, cap: usize, src: *const c_void, size: usize, level: c_int) -> usize;
    fn ZSTD_decompress(dst: *mut c_void, cap: usize, src: *const c_void, size: usize) -> usize;
    fn ZSTD_findFrameCompressedSize(src: *const c_void, size: usize) -> usize;
    fn ZSTD_isError(code: usize) -> c_uint;
}

fn bad(message: &str) -> AsmError {
    AsmError::new(Code::Asm003, message.to_owned())
}

fn compress(raw: &[u8]) -> Vec<u8> {
    // libzstd receives exactly the source and allocated destination ranges.
    let bound = unsafe { ZSTD_compressBound(raw.len()) };
    let mut out = vec![0; 12 + bound];
    out[..4].copy_from_slice(&2u32.to_le_bytes());
    out[4..8].copy_from_slice(&(raw.len() as u32).to_le_bytes());
    out[8..12].copy_from_slice(&4u32.to_le_bytes());
    let written = unsafe { ZSTD_compress(out[12..].as_mut_ptr().cast(), bound, raw.as_ptr().cast(), raw.len(), 3) };
    assert_eq!(unsafe { ZSTD_isError(written) }, 0, "encoding an in-memory Zstandard section");
    out.truncate(12 + written);
    out
}

fn decode(encoded: &[u8]) -> Result<Vec<u8>, AsmError> {
    if encoded.len() < 12 || u32_at(encoded, 0) != 2 || u32_at(encoded, 8) != 4 {
        return Err(bad("invalid OpenTSP ELF compression"));
    }
    let size = u32_at(encoded, 4) as usize;
    if size > MAX_SECTION {
        return Err(bad("OpenTSP ELF section expands beyond its limit"));
    }
    let frame = &encoded[12..];
    if unsafe { ZSTD_findFrameCompressedSize(frame.as_ptr().cast(), frame.len()) } != frame.len() {
        return Err(bad("expected one complete ELF Zstandard frame"));
    }
    let mut bytes = vec![0; size];
    let n = unsafe { ZSTD_decompress(bytes.as_mut_ptr().cast(), size, frame.as_ptr().cast(), frame.len()) };
    if n != size {
        return Err(bad("ELF decompressed section length differs"));
    }
    Ok(bytes)
}

fn u16_at(data: &[u8], p: usize) -> u16 {
    u16::from_le_bytes(data[p..p + 2].try_into().unwrap())
}

fn u32_at(data: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(data[p..p + 4].try_into().unwrap())
}

fn align(out: &mut Vec<u8>, alignment: usize) {
    out.resize(out.len().next_multiple_of(alignment), 0);
}

fn words(out: &mut Vec<u8>, values: &[u32]) {
    for value in values {
        out.extend(value.to_le_bytes());
    }
}

pub(crate) fn write_sections(note: &[u8], iq: &[u8], placement: &[u8], extra: &[(&str, &[u8])]) -> Vec<u8> {
    let mut sections = vec![
        (".note.opentsp", 7, 0, note.to_vec()),
        (".tsp.iq", 1, COMPRESSED, compress(iq)),
        (".tsp.load", 1, COMPRESSED, compress(placement)),
    ];
    sections.extend(extra.iter().map(|(name, raw)| (*name, 1, COMPRESSED, compress(raw))));
    let section_count = sections.len() as u16;
    let mut out = vec![0; HEADER];
    let mut names = vec![0];
    let mut table = vec![0; SECTION];
    for (name, kind, flags, data) in sections {
        align(&mut out, 4);
        words(&mut table, &[names.len() as u32, kind, flags, 0, out.len() as u32, data.len() as u32, 0, 0, 4, 0]);
        names.extend(name.as_bytes());
        names.push(0);
        out.extend(data);
    }
    let strname = names.len() as u32;
    names.extend(b".shstrtab\0");
    words(&mut table, &[strname, 3, 0, 0, out.len() as u32, names.len() as u32, 0, 0, 1, 0]);
    out.extend(names);
    align(&mut out, 4);
    let shoff = out.len() as u32;
    out.extend(table);
    out[..16].copy_from_slice(IDENT);
    out[16..18].copy_from_slice(&1u16.to_le_bytes());
    out[20..24].copy_from_slice(&1u32.to_le_bytes());
    out[32..36].copy_from_slice(&shoff.to_le_bytes());
    out[40..42].copy_from_slice(&(HEADER as u16).to_le_bytes());
    out[46..48].copy_from_slice(&(SECTION as u16).to_le_bytes());
    out[48..50].copy_from_slice(&(section_count + 2).to_le_bytes());
    out[50..52].copy_from_slice(&(section_count + 1).to_le_bytes());
    out
}

pub(crate) fn read(data: &[u8]) -> Result<BTreeMap<&str, Vec<u8>>, AsmError> {
    if data.len() < HEADER || &data[..16] != IDENT {
        return Err(bad("expected an OpenTSP ELF32 program; rebuild old object files"));
    }
    if u16_at(data, 16) != 1
        || u16_at(data, 18) != 0
        || u32_at(data, 20) != 1
        || u32_at(data, 24) != 0
        || u32_at(data, 28) != 0
        || u32_at(data, 36) != 0
        || u16_at(data, 40) as usize != HEADER
        || u16_at(data, 42) != 0
        || u16_at(data, 44) != 0
        || u16_at(data, 46) as usize != SECTION
    {
        return Err(bad("invalid OpenTSP ELF header"));
    }
    let offset = u32_at(data, 32) as usize;
    let n = u16_at(data, 48) as usize;
    let stridx = u16_at(data, 50) as usize;
    if !(4..=9).contains(&n)
        || stridx == 0
        || stridx >= n
        || offset < HEADER
        || !offset.is_multiple_of(4)
        || offset.checked_add(n * SECTION) != Some(data.len())
    {
        return Err(bad("invalid OpenTSP ELF section directory"));
    }
    let entry = |i| &data[offset + i * SECTION..offset + (i + 1) * SECTION];
    if entry(0).iter().any(|&byte| byte != 0) {
        return Err(bad("invalid ELF null section"));
    }
    let mut ranges = Vec::new();
    for i in 1..n {
        let e = entry(i);
        let a = u32_at(e, 16) as usize;
        let size = u32_at(e, 20) as usize;
        let align = u32_at(e, 32) as usize;
        let kind = u32_at(e, 4);
        let flags = u32_at(e, 8);
        let b = a.checked_add(size).ok_or_else(|| bad("ELF section length overflow"))?;
        if !(align == 1 || align == 4)
            || !a.is_multiple_of(align)
            || a < HEADER
            || b > offset
            || size > MAX_SECTION
            || [12, 24, 28, 36].into_iter().any(|p| u32_at(e, p) != 0)
            || !matches!(kind, 1 | 3 | 7)
            || !matches!(flags, 0 | COMPRESSED)
            || (kind != 1 && flags != 0)
        {
            return Err(bad("ELF section outside the payload or invalid attributes"));
        }
        ranges.push((a, b));
    }
    let mut ordered = ranges.clone();
    ordered.sort_unstable();
    if ordered.windows(2).any(|w| w[0].1 > w[1].0) || u32_at(entry(stridx), 4) != 3 {
        return Err(bad("overlapping ELF sections or invalid name table"));
    }
    let (a, b) = ranges[stridx - 1];
    let names = &data[a..b];
    let mut result = BTreeMap::new();
    for (i, (a, b)) in ranges.into_iter().enumerate() {
        let e = entry(i + 1);
        let at = u32_at(e, 0) as usize;
        let name = names
            .get(at..)
            .and_then(|tail| tail.split(|&x| x == 0).next())
            .and_then(|s| std::str::from_utf8(s).ok())
            .ok_or_else(|| bad("invalid ELF section name"))?;
        if at == 0 || at + name.len() >= names.len() || name.is_empty() {
            return Err(bad("invalid ELF section name"));
        }
        let kind = match name {
            ".note.opentsp" => 7,
            ".tsp.iq" | ".tsp.load" | ".tsp.text" | ".tsp.const" | ".tsp.zero" | ".tsp.recipe" => 1,
            ".shstrtab" => 3,
            _ => return Err(bad("unknown OpenTSP ELF section")),
        };
        if u32_at(e, 4) != kind || result.contains_key(name) {
            return Err(bad("duplicate or invalid OpenTSP ELF section"));
        }
        let payload = &data[a..b];
        let contents = if u32_at(e, 8) == COMPRESSED { decode(payload)? } else { payload.to_vec() };
        result.insert(name, contents);
    }
    if [".note.opentsp", ".tsp.iq", ".tsp.load", ".shstrtab"].iter().any(|s| !result.contains_key(s)) {
        return Err(bad("OpenTSP ELF lacks a required section"));
    }
    Ok(result)
}
