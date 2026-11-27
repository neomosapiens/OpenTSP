//! Named Safetensors SRAM images and validated extern binding.
use crate::{externs::Extern, load_profile};
use ots_isa::{AsmError, Code, ELTYPE_CODES};
use ots_profile::Profile;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// The dtype of a tensor of raw bytes (no element type).
pub const RAW_DTYPE: u8 = 255;
/// A tensor name: `[A-Za-z_][A-Za-z0-9_]{0,14}`.
const NAME_MAX: usize = 15;

fn bad(m: impl Into<String>) -> AsmError {
    AsmError::new(Code::Asm003, m)
}

fn no_weights(m: impl Into<String>) -> AsmError {
    AsmError::new(Code::Asm004, m)
}

/// The first 16 bytes of SHA-256 of `data`. On macOS the system's CommonCrypto computes it (about six times
/// the speed of the `sha2` crate there, which has no hardware path on that platform); elsewhere, and for an
/// input of 4 GiB or more, the `sha2` crate does. Both give the same bytes (`tests/load_speed.rs`).
pub fn digest16(data: &[u8]) -> [u8; 16] {
    let d: [u8; 32] = sha256(data);
    d[..16].try_into().unwrap()
}

/// Which code computes [`digest16`]: `"commoncrypto"` or `"sha2"`.
pub fn digest_backend() -> &'static str {
    if cfg!(target_os = "macos") {
        "commoncrypto"
    } else {
        "sha2"
    }
}

#[cfg(target_os = "macos")]
fn sha256(data: &[u8]) -> [u8; 32] {
    extern "C" {
        // `CC_SHA256(const void *data, CC_LONG len, unsigned char *md)` of CommonCrypto (libSystem); `CC_LONG` is `u32`
        fn CC_SHA256(data: *const std::ffi::c_void, len: u32, md: *mut u8) -> *mut u8;
    }
    let Ok(len) = u32::try_from(data.len()) else {
        return Sha256::digest(data).into();
    };
    let mut md = [0u8; 32];
    // SAFETY: `data` is valid for `len` bytes, `md` for the 32 bytes of a SHA-256 digest, and the call keeps no pointer.
    unsafe { CC_SHA256(data.as_ptr().cast(), len, md.as_mut_ptr()) };
    md
}

#[cfg(not(target_os = "macos"))]
fn sha256(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}

/// A tensor to pack: `dtype` an element-type name of `encoding.json` (`None`: raw bytes), `words`
/// of `vlanes` bytes each, lane 0 first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorIn {
    pub name: String,
    pub dtype: Option<String>,
    pub words: Vec<Vec<u8>>,
}

/// A tensor of a package: its (informational) dtype code and its image, a whole number of words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub dtype: u8,
    vlanes: usize,
    image: Vec<u8>,
}

impl Tensor {
    /// The number of words.
    pub fn len(&self) -> usize {
        self.image.len() / self.vlanes
    }

    pub fn is_empty(&self) -> bool {
        self.image.is_empty()
    }

    /// Word `k`, `vlanes` bytes, lane 0 first.
    pub fn word(&self, k: usize) -> &[u8] {
        &self.image[k * self.vlanes..(k + 1) * self.vlanes]
    }

    /// The words in order.
    pub fn words(&self) -> impl Iterator<Item = &[u8]> {
        self.image.chunks_exact(self.vlanes)
    }
}

/// A word the loader writes into SRAM before cycle 0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Preload<'a> {
    pub slice: u64,
    pub addr: u64,
    pub word: &'a [u8],
}

/// `[A-Za-z_][A-Za-z0-9_]{0,14}`, the whole text.
fn is_tensor_name(s: &str) -> bool {
    s.len() <= NAME_MAX && crate::externs::is_ident(s)
}

fn hash_text(image: &[u8]) -> String {
    sha256(image).iter().map(|b| format!("{b:02x}")).collect()
}

/// Encode named raw SRAM words as interoperable Safetensors U8 tensors.
pub fn write_weights(profile: &Profile, tensors: &[TensorIn]) -> Result<Vec<u8>, AsmError> {
    use serde_json::{json, Map, Value};
    let lanes = profile.vlanes() as usize;
    let mut attrs = Map::new();
    attrs.insert("opentsp.profile".into(), json!(profile.name()));
    let mut header = Map::new();
    let mut images = Vec::new();
    let mut ordered = BTreeMap::new();
    for t in tensors {
        if !is_tensor_name(&t.name) {
            return Err(no_weights(format!("tensor name {:?} is not an identifier of at most 15 characters", t.name)));
        }
        if t.words.is_empty() || t.words.iter().any(|w| w.len() != lanes) {
            return Err(no_weights(format!(
                "tensor {}: every word must be {lanes} bytes, and there must be one",
                t.name
            )));
        }
        let dtype = match &t.dtype {
            None => RAW_DTYPE,
            Some(name) => ELTYPE_CODES
                .iter()
                .find(|(k, _)| *k == name.as_str())
                .map(|(_, code)| *code as u8)
                .ok_or_else(|| no_weights(format!("tensor {}: unknown element type {name:?}", t.name)))?,
        };
        if ordered.insert(t.name.as_str(), (t, dtype)).is_some() {
            return Err(no_weights(format!("tensor name {:?} occurs more than once", t.name)));
        }
    }
    for (t, dtype) in ordered.into_values() {
        let start = images.len();
        for word in &t.words {
            images.extend(word);
        }
        attrs.insert(format!("dtype.{}", t.name), json!(dtype.to_string()));
        attrs.insert(format!("sha256.{}", t.name), json!(hash_text(&images[start..])));
        header.insert(
            t.name.clone(),
            json!({"dtype":"U8", "shape":[t.words.len(), lanes],
                                            "data_offsets":[start, images.len()]}),
        );
    }
    header.insert("__metadata__".into(), Value::Object(attrs));
    let mut encoded = serde_json::to_vec(&header).expect("Safetensors metadata is JSON");
    encoded.resize(encoded.len().next_multiple_of(8), b' ');
    let mut output = Vec::with_capacity(8 + encoded.len() + images.len());
    output.extend((encoded.len() as u64).to_le_bytes());
    output.extend(encoded);
    output.extend(images);
    Ok(output)
}

struct UniqueValue(serde_json::Value);

impl<'de> serde::Deserialize<'de> for UniqueValue {
    fn deserialize<D: serde::Deserializer<'de>>(input: D) -> Result<Self, D::Error> {
        use serde::de::{Error, MapAccess, SeqAccess, Visitor};
        use serde_json::{Map, Value};
        struct UniqueVisitor;
        impl<'de> Visitor<'de> for UniqueVisitor {
            type Value = UniqueValue;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("unique Safetensors metadata")
            }
            fn visit_bool<E: Error>(self, v: bool) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::from(v)))
            }
            fn visit_u64<E: Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::from(v)))
            }
            fn visit_i64<E: Error>(self, v: i64) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::from(v)))
            }
            fn visit_str<E: Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::from(v)))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut out = Vec::new();
                while let Some(item) = seq.next_element::<UniqueValue>()? {
                    out.push(item.0);
                }
                Ok(UniqueValue(Value::Array(out)))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut out = Map::new();
                while let Some((key, UniqueValue(value))) = map.next_entry::<String, UniqueValue>()? {
                    if out.insert(key, value).is_some() {
                        return Err(A::Error::custom("duplicate Safetensors metadata key"));
                    }
                }
                Ok(UniqueValue(Value::Object(out)))
            }
        }
        input.deserialize_any(UniqueVisitor)
    }
}

/// Decode profile-bound raw SRAM word arrays from Safetensors.
pub fn read_weights(
    data: &[u8],
    profile: Option<&str>,
    config: Option<&str>,
) -> Result<BTreeMap<String, Tensor>, AsmError> {
    use serde_json::Value;
    const MAX_HEADER: usize = 16 * 1024 * 1024;
    if data.len() < 8 || data.starts_with(b"OTSW") {
        return Err(bad("expected OpenTSP Safetensors; rebuild old weight packages"));
    }
    let length = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
    if length == 0 || length > MAX_HEADER || length > data.len() - 8 || data[8] != b'{' {
        return Err(bad("invalid Safetensors header length"));
    }
    let UniqueValue(mut header) = serde_json::from_slice(&data[8..8 + length])
        .map_err(|error| bad(format!("invalid Safetensors metadata: {error}")))?;
    let entries = header.as_object_mut().ok_or_else(|| bad("invalid Safetensors header"))?;
    let attrs = entries.remove("__metadata__").ok_or_else(|| bad("missing OpenTSP Safetensors metadata"))?;
    let attrs = attrs
        .as_object()
        .filter(|m| m.values().all(Value::is_string))
        .ok_or_else(|| bad("invalid OpenTSP Safetensors metadata"))?;
    let name =
        attrs.get("opentsp.profile").and_then(Value::as_str).ok_or_else(|| bad("missing Safetensors profile"))?;
    if profile.is_some_and(|p| p != name) {
        return Err(bad("Safetensors profile differs from the object"));
    }
    let lanes = load_profile(name, config)?.vlanes() as usize;
    let bytes = &data[8 + length..];
    let mut ranges = Vec::new();
    let pair = |v: Option<&Value>| -> Option<(usize, usize)> {
        let list = v?.as_array()?;
        if list.len() != 2 {
            return None;
        }
        Some((usize::try_from(list[0].as_u64()?).ok()?, usize::try_from(list[1].as_u64()?).ok()?))
    };
    for (name, item) in entries.iter() {
        let (start, end) = pair(item.get("data_offsets")).ok_or_else(|| bad("invalid SRAM tensor offsets"))?;
        ranges.push((start, end, name, item));
    }
    ranges.sort_unstable_by(|a, b| (a.0, a.1, a.2).cmp(&(b.0, b.1, b.2)));
    let mut out = BTreeMap::new();
    let mut expected = 0;
    for (start, end, name, item) in ranges {
        if !is_tensor_name(name) || item.get("dtype").and_then(Value::as_str) != Some("U8") {
            return Err(bad("invalid SRAM tensor"));
        }
        let (words, width) = pair(item.get("shape")).ok_or_else(|| bad("invalid SRAM tensor shape"))?;
        if width != lanes
            || words == 0
            || start >= end
            || end > bytes.len()
            || words.checked_mul(lanes) != Some(end - start)
        {
            return Err(bad(format!("tensor {name}: invalid image length")));
        }
        let dtype = attrs
            .get(&format!("dtype.{name}"))
            .and_then(Value::as_str)
            .and_then(|s| s.parse::<u8>().ok())
            .filter(|d| *d == RAW_DTYPE || ELTYPE_CODES.iter().any(|(_, c)| *c == u32::from(*d)))
            .ok_or_else(|| bad(format!("tensor {name}: invalid element type metadata")))?;
        if start != expected {
            return Err(bad("Safetensors payload has holes or overlaps"));
        }
        let image = &bytes[start..end];
        if attrs.get(&format!("sha256.{name}")).and_then(Value::as_str) != Some(hash_text(image).as_str()) {
            return Err(no_weights(format!("tensor {name}: image does not match its digest")));
        }
        out.insert(name.clone(), Tensor { dtype, vlanes: lanes, image: image.to_vec() });
        expected = end;
    }
    if expected != bytes.len() {
        return Err(bad("Safetensors payload has trailing bytes"));
    }
    Ok(out)
}

pub fn object_preloads(
    data: &[u8],
    profile: &str,
    externs: &[Extern],
    config: Option<&str>,
    event: u32,
) -> Result<BTreeMap<String, Tensor>, AsmError> {
    if crate::text::is_text_input(data) {
        return Ok(BTreeMap::new());
    }
    let sections = crate::elf::read(data)?;
    let regions: BTreeMap<_, _> = externs.iter().map(|x| (x.name.as_str(), x.size)).collect();
    let mut out = BTreeMap::new();
    let mut seen = std::collections::HashSet::new();
    for section in [".tsp.text", ".tsp.const"] {
        if let Some(raw) = sections.get(section) {
            for (name, tensor) in read_weights(raw, Some(profile), config)? {
                if !seen.insert(name.clone()) || regions.get(name.as_str()) != Some(&(tensor.len() as u64)) {
                    return Err(no_weights(format!("invalid embedded SRAM extent {name}")));
                }
                out.insert(name, tensor);
            }
        }
    }
    if let Some(raw) = sections.get(".tsp.zero") {
        if raw.len() % 20 != 0 {
            return Err(bad("invalid ELF zero range directory"));
        }
        let lanes = load_profile(profile, config)?.vlanes() as usize;
        for record in raw.chunks_exact(20) {
            let name = crate::object::ascii_field(&record[..16], "zero range")?;
            let phases = u32::from_le_bytes(record[16..20].try_into().unwrap());
            let Some(words) = regions.get(name.as_str()).copied() else {
                return Err(no_weights(format!("invalid embedded zero range {name}")));
            };
            if !seen.insert(name.clone()) || !matches!(phases, 1 | 3) {
                return Err(no_weights(format!("invalid embedded zero range {name}")));
            }
            let size = usize::try_from(words)
                .ok()
                .and_then(|n| n.checked_mul(lanes))
                .ok_or_else(|| no_weights("zero range too large"))?;
            if phases & event != 0 {
                out.insert(name, Tensor { dtype: RAW_DTYPE, vlanes: lanes, image: vec![0; size] });
            }
        }
    }
    Ok(out)
}

pub fn merge_preloads(into: &mut BTreeMap<String, Tensor>, from: BTreeMap<String, Tensor>) -> Result<(), AsmError> {
    for (name, tensor) in from {
        if into.contains_key(&name) {
            return Err(no_weights(format!("duplicate tensor {name} across weight packages")));
        }
        into.insert(name, tensor);
    }
    Ok(())
}

/// The SRAM preloads the loader writes before cycle 0, in extern order: every extern takes the tensor
/// of its name, of exactly its size (`OTS-E-ASM-004` otherwise, for the first extern that fails).
pub fn bind_weights<'a>(
    externs: &[Extern],
    tensors: &'a BTreeMap<String, Tensor>,
) -> Result<Vec<Preload<'a>>, AsmError> {
    let mut pre = Vec::new();
    for x in externs {
        let Some(t) = tensors.get(&x.name) else {
            return Err(no_weights(format!("extern {}: no tensor of that name in the weight package", x.name)));
        };
        if t.len() as u64 != x.size {
            return Err(no_weights(format!("extern {}: tensor has {} words, the extern {}", x.name, t.len(), x.size)));
        }
        pre.extend(t.words().enumerate().map(|(k, w)| Preload { slice: x.slice, addr: x.addr + k as u64, word: w }));
    }
    Ok(pre)
}

/// The harness `S` lines of SRAM preloads (`S <slice> <addr> <b0> .. <bn>`, hex, lane 0 first).
pub fn preload_text(pre: &[Preload]) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    for p in pre {
        write!(s, "S {:x} {:x}", p.slice, p.addr).unwrap();
        for b in p.word {
            write!(s, " {b:x}").unwrap();
        }
        s.push('\n');
    }
    s
}
