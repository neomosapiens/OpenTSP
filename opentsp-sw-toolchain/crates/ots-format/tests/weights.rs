//! Safetensors framing, physical tensor validation and ordered SRAM binding.
use ots_format::{
    weights::{bind_weights, preload_text, read_weights, write_weights, Tensor, TensorIn, RAW_DTYPE},
    Extern,
};
use ots_isa::{AsmError, Code};
use ots_profile::Profile;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../ots-asm/tests/data");

fn ref_profile() -> Profile {
    Profile::load("ref").unwrap()
}

fn mini_profile() -> Profile {
    Profile::load("mini").unwrap()
}

fn digest(b: &[u8]) -> [u8; 32] {
    let d: [u8; 32] = Sha256::digest(b).into();
    d
}

fn name16(s: &[u8]) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[..s.len()].copy_from_slice(s);
    b
}

/// A physical tensor and its independently writable metadata.
#[derive(Clone)]
struct Ent {
    name: [u8; 16],
    words: u32,
    dtype: u8,
    z1: u8,
    z2: u16,
    off: Option<u64>,
    dg: Option<[u8; 32]>,
    image: Vec<u8>,
}

/// An entry for `words` words of `vl` bytes, every byte `fill`; offset and digest are filled in by `Pkg`.
fn ent(name: &str, words: u32, vl: usize, fill: u8) -> Ent {
    Ent {
        name: name16(name.as_bytes()),
        words,
        dtype: 0,
        z1: 0,
        z2: 0,
        off: None,
        dg: None,
        image: vec![fill; words as usize * vl],
    }
}

struct Pkg {
    profile: [u8; 16],
    tensors: Vec<Ent>,
    trailer: Vec<u8>,
}

impl Pkg {
    fn new(profile: &str, tensors: Vec<Ent>) -> Pkg {
        Pkg { profile: name16(profile.as_bytes()), tensors, trailer: vec![] }
    }

    fn bytes(&self) -> Vec<u8> {
        use serde_json::{json, Map};
        let name = |data: &[u8]| {
            let last = data.iter().rposition(|&byte| byte != 0).map_or(0, |i| i + 1);
            String::from_utf8_lossy(&data[..last]).into_owned()
        };
        let mut metadata = Map::new();
        metadata.insert("opentsp.profile".into(), json!(name(&self.profile)));
        let mut fields = Vec::new();
        let mut images = Vec::new();
        let mut tensors = self.tensors.iter().collect::<Vec<_>>();
        tensors.sort_by_key(|tensor| tensor.name);
        for t in tensors {
            let key = name(&t.name);
            let start = t.off.unwrap_or(images.len() as u64);
            images.extend(&t.image);
            let hex: String = t.dg.unwrap_or_else(|| digest(&t.image)).iter().map(|b| format!("{b:02x}")).collect();
            metadata
                .insert(format!("dtype.{key}"), json!(if t.z1 == 0 { t.dtype.to_string() } else { "invalid".into() }));
            metadata.insert(format!("sha256.{key}"), json!(hex));
            let dtype = if t.z2 == 0 { "U8" } else { "I16" };
            let lanes = if t.words == 0 { 0 } else { t.image.len() / t.words as usize };
            let item = json!({"dtype": dtype, "shape": [t.words as usize, lanes],
                              "data_offsets": [start, images.len() as u64]});
            fields.push(format!("{}:{item}", serde_json::to_string(&key).unwrap()));
        }
        fields.insert(0, format!("\"__metadata__\":{}", json!(metadata)));
        let header = format!("{{{}}}", fields.join(","));
        finish_package(header.into_bytes(), &images, &self.trailer)
    }
}

fn finish_package(mut header: Vec<u8>, images: &[u8], trailer: &[u8]) -> Vec<u8> {
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend(header);
    out.extend(images);
    out.extend(trailer);
    out
}

fn payload_start(data: &[u8]) -> usize {
    8 + u64::from_le_bytes(data[..8].try_into().unwrap()) as usize
}

fn header(data: &[u8]) -> serde_json::Value {
    serde_json::from_slice(&data[8..payload_start(data)]).unwrap()
}

fn code_of(r: Result<BTreeMap<String, Tensor>, AsmError>) -> (Code, String) {
    let e = r.expect_err("must be rejected");
    (e.code, e.msg)
}

fn rejects(data: &[u8], profile: Option<&str>, code: Code, why: &str) {
    let (c, msg) = code_of(read_weights(data, profile, None));
    assert_eq!(c, code, "{why}: {msg}");
}

fn good() -> Pkg {
    Pkg::new("ref", vec![ent("a", 2, 320, 1), ent("b", 1, 320, 2)])
}

// ---- layout ------------------------------------------------------------------------------------------

#[test]
fn the_raw_dtype_code_is_reserved() {
    assert_eq!(RAW_DTYPE, 255);
}

#[test]
fn a_package_of_two_tensors_is_read_back_word_by_word_lane_zero_first() {
    let mut p = good();
    p.tensors[0].image = (0..640).map(|i| (i % 251) as u8).collect();
    p.tensors[0].dtype = 1;
    p.tensors[1].dtype = 255;
    let t = read_weights(&p.bytes(), None, None).unwrap();
    assert_eq!(t.keys().collect::<Vec<_>>(), ["a", "b"]);
    let a = &t["a"];
    assert_eq!((a.dtype, a.len(), a.is_empty()), (1, 2, false));
    assert_eq!(a.word(0), &p.tensors[0].image[..320]);
    assert_eq!(a.word(1), &p.tensors[0].image[320..]);
    assert_eq!(t["b"].dtype, 255);
    assert_eq!(t["b"].word(0), vec![2u8; 320].as_slice());
}

#[test]
fn assigned_element_type_codes_and_raw_are_the_only_valid_dtypes() {
    for dt in (0..=12).chain([16, 255]) {
        let mut p = good();
        p.tensors[0].dtype = dt;
        assert!(read_weights(&p.bytes(), None, None).is_ok(), "dtype {dt}");
    }
    for dt in (13..255).filter(|&dt| dt != 16) {
        let mut p = good();
        p.tensors[1].dtype = dt;
        rejects(&p.bytes(), None, Code::Asm003, &format!("dtype {dt}"));
    }
}

#[test]
fn a_package_of_no_tensors_is_valid() {
    let p = Pkg::new("ref", vec![]);
    assert_eq!(payload_start(&p.bytes()), p.bytes().len());
    assert!(read_weights(&p.bytes(), None, None).unwrap().is_empty());
}

#[test]
fn the_word_size_is_the_profiles_vlanes() {
    let mini = Pkg::new("mini", vec![ent("a", 3, 16, 7)]);
    let t = read_weights(&mini.bytes(), Some("mini"), None).unwrap();
    assert_eq!(t["a"].len(), 3);
    assert_eq!(t["a"].word(2).len(), 16);
    // the same bytes read as ref: 48 bytes are not a whole 320-byte word, so the image is truncated
    let as_ref = Pkg { profile: name16(b"ref"), ..mini };
    rejects(&as_ref.bytes(), None, Code::Asm003, "mini image as ref");
}

#[test]
fn a_profile_digest_does_not_exist_so_a_package_for_one_profile_reads_under_another_config() {
    // the package names the profile and nothing else about it; `--config` stands for the profile file
    let p = good();
    let dir = std::env::temp_dir().join(format!("ots-weights-cfg-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = dir.join("ref22.json");
    let text =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../opentsp-arch-spec/config/ref.json"))
            .unwrap();
    std::fs::write(&cfg, text.replace("\"mem_slices_per_hemisphere\": 44", "\"mem_slices_per_hemisphere\": 22"))
        .unwrap();
    assert!(read_weights(&p.bytes(), None, Some(cfg.to_str().unwrap())).is_ok());
    std::fs::remove_dir_all(&dir).ok();
}

// ---- rejections of read_weights, in the order the reader checks ---------------------------------------

#[test]
fn rejects_a_short_file_and_legacy_weight_formats() {
    let good = good().bytes();
    for len in [0, 1, 4, 7, 8, 31] {
        rejects(&good[..len], None, Code::Asm003, &format!("{len} bytes"));
    }
    for magic in [b"OTSO", b"OTSW", b"OTSX", b"otsw"] {
        let mut invalid = good.clone();
        invalid[..4].copy_from_slice(magic);
        let (code, message) = code_of(read_weights(&invalid, None, None));
        assert_eq!(code, Code::Asm003);
        if magic == b"OTSW" {
            assert!(message.contains("rebuild"), "{message}");
        }
    }
}

#[test]
fn rejects_an_invalid_safetensors_header() {
    let valid = good().bytes();
    for n in [0u64, 1, 16 * 1024 * 1024 + 1, u64::MAX] {
        let mut b = valid.clone();
        b[..8].copy_from_slice(&n.to_le_bytes());
        rejects(&b, None, Code::Asm003, "invalid header length");
    }
    for invalid in [b"[]".as_slice(), b"{{", b"{\"__metadata__\":42}", b"{}"] {
        let b = finish_package(invalid.to_vec(), &[], &[]);
        rejects(&b, None, Code::Asm003, "invalid metadata");
    }
}

#[test]
fn rejects_a_bad_profile_field_a_profile_that_is_not_the_requested_one_and_an_unknown_profile() {
    for (what, field) in [
        ("empty", [0u8; 16]),
        ("text after NUL", *b"ref\0\0\0\0\0\0\0\0\0\0\0\0x"),
        ("no NUL, 16 bytes, unknown", *b"abcdefghijklmnop"),
        ("non-ASCII", name16(b"r\xe9f")),
    ] {
        let p = Pkg { profile: field, ..good() };
        rejects(&p.bytes(), None, Code::Asm003, what);
    }
    rejects(&good().bytes(), Some("mini"), Code::Asm003, "--profile mini for a ref package");
    rejects(&Pkg::new("nosuch", vec![]).bytes(), None, Code::Asm003, "unknown profile");
    rejects(&Pkg::new("REF", vec![]).bytes(), None, Code::Asm003, "profile names are case-sensitive");
    // --profile equal to the field is accepted
    assert!(read_weights(&good().bytes(), Some("ref"), None).is_ok());
}

#[test]
fn rejects_a_header_that_is_cut_short() {
    let b = good().bytes();
    for cut in [8, 9, 79, 80, payload_start(&b) - 1] {
        rejects(&b[..cut], None, Code::Asm003, &format!("cut {cut}"));
    }
}

#[test]
fn rejects_a_bad_entry_pad_words_dtype_duplicate_name_offset() {
    type Mutation = (&'static str, Box<dyn Fn(&mut Pkg)>);
    let cases: Vec<Mutation> = vec![
        ("pad byte", Box::new(|p| p.tensors[1].z1 = 1)),
        ("pad half-word", Box::new(|p| p.tensors[1].z2 = 1)),
        ("zero words", Box::new(|p| p.tensors[1].words = 0)),
        ("dtype 13", Box::new(|p| p.tensors[0].dtype = 13)),
        ("duplicate name", Box::new(|p| p.tensors[1].name = name16(b"a"))),
        ("offset overlap", Box::new(|p| p.tensors[1].off = Some(639))),
        ("offset too high", Box::new(|p| p.tensors[0].off = Some(1))),
        ("second offset", Box::new(|p| p.tensors[1].off = Some(0))),
        ("offset 2^64-1", Box::new(|p| p.tensors[1].off = Some(u64::MAX))),
    ];
    for (what, f) in cases {
        let mut p = good();
        f(&mut p);
        rejects(&p.bytes(), None, Code::Asm003, what);
    }
}

#[test]
fn rejects_a_tensor_name_that_is_not_nul_padded_ascii_text() {
    for (what, name) in [
        ("empty", [0u8; 16]),
        ("text after NUL", *b"a\0b\0\0\0\0\0\0\0\0\0\0\0\0\0"),
        ("non-ASCII", name16(b"\xe9")),
        ("16 characters, no NUL, non-ASCII at the end", *b"abcdefghijklmno\xff"),
    ] {
        let mut p = good();
        p.tensors[0].name = name;
        rejects(&p.bytes(), None, Code::Asm003, what);
    }
}

// The reader enforces the identifier rule of the writer.
#[test]
fn rejects_a_tensor_name_that_is_not_an_identifier_of_at_most_15_characters() {
    for (what, name) in [
        ("starts with a digit", name16(b"1x")),
        ("has a hyphen", name16(b"a-b")),
        ("has a space", name16(b"a b")),
        ("has a dot", name16(b"a.b")),
        ("ends in a line feed", name16(b"x\n")),
        ("has a control character", name16(b"a\x01b")),
        ("fills the field (16 characters)", name16(b"abcdefghijklmnop")),
    ] {
        let mut p = good();
        p.tensors[0].name = name;
        rejects(&p.bytes(), None, Code::Asm003, what);
        let mut p = good();
        p.tensors[1].name = name;
        rejects(&p.bytes(), None, Code::Asm003, what);
    }
}

#[test]
fn the_reader_accepts_every_name_the_writer_accepts() {
    for name in [&b"a"[..], b"_", b"_a1", b"A9", b"abcdefghijklmno", b"x_y_z"] {
        let mut p = good();
        p.tensors[0].name = name16(name);
        let t = read_weights(&p.bytes(), None, None).unwrap();
        assert!(t.contains_key(std::str::from_utf8(name).unwrap()), "{name:?}");
    }
}

#[test]
fn rejects_a_truncated_image_trailing_bytes_and_a_wrong_size() {
    let b = good().bytes();
    let (c, m) = code_of(read_weights(&b[..b.len() - 1], None, None));
    assert_eq!(c, Code::Asm003);
    assert!(m.contains("offset") || m.contains("length"), "{m}");
    for cut in [128, 129, 128 + 319, 128 + 320, 128 + 641] {
        rejects(&b[..cut], None, Code::Asm003, &format!("cut at {cut}"));
    }
    for extra in [1usize, 4, 320, 1000] {
        let mut p = good();
        p.trailer = vec![0; extra];
        let (c, m) = code_of(read_weights(&p.bytes(), None, None));
        assert_eq!(c, Code::Asm003, "{extra} trailing bytes");
        assert!(m.contains("trailing"), "{m}");
    }
    // a word count larger than the file holds
    let mut p = good();
    p.tensors[1].words = u32::MAX;
    rejects(&p.bytes(), None, Code::Asm003, "word count 2^32-1");
}

#[test]
fn a_digest_that_does_not_match_is_asm_004_and_is_found_in_table_order() {
    let mut p = good();
    let mut dg = digest(&p.tensors[0].image);
    dg[0] ^= 1;
    p.tensors[0].dg = Some(dg);
    rejects(&p.bytes(), None, Code::Asm004, "digest of the first tensor");
    // a flipped bit of any image byte
    let b = good().bytes();
    for i in [payload_start(&b), payload_start(&b) + 319, payload_start(&b) + 640, b.len() - 1] {
        let mut b = b.clone();
        b[i] ^= 0x10;
        rejects(&b, None, Code::Asm004, &format!("image byte {i}"));
    }
    let mut p = good();
    let mut wrong = digest(&p.tensors[1].image);
    wrong[5] ^= 1;
    p.tensors[1].dg = Some(wrong);
    rejects(&p.bytes(), None, Code::Asm004, "digest of the second tensor");
}

#[test]
fn the_table_and_the_images_are_checked_tensor_by_tensor() {
    // tensor 0 has a wrong digest and tensor 1 a bad pad: the reader reaches tensor 0's image first
    let mut p = good();
    let mut dg = digest(&p.tensors[0].image);
    dg[3] ^= 0x80;
    p.tensors[0].dg = Some(dg);
    p.tensors[1].z1 = 9;
    rejects(&p.bytes(), None, Code::Asm004, "digest of tensor 0 before the pad of tensor 1");
    // and the other way round: a bad pad in tensor 0 is found before a wrong digest in tensor 1
    let mut q = good();
    q.tensors[0].z2 = 9;
    let mut dg = digest(&q.tensors[1].image);
    dg[3] ^= 0x80;
    q.tensors[1].dg = Some(dg);
    rejects(&q.bytes(), None, Code::Asm003, "pad of tensor 0 before the digest of tensor 1");
    // a wrong digest is also found before trailing bytes
    let mut r = good();
    r.trailer = vec![1];
    r.tensors[1].dg = Some([0; 32]);
    rejects(&r.bytes(), None, Code::Asm004, "digest before the size");
}

#[test]
fn the_digest_is_over_the_image_and_not_the_name_or_the_word_count() {
    let mut p = good();
    let parsed = header(&p.bytes());
    let expected: String = digest(&p.tensors[0].image).iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(parsed["__metadata__"]["sha256.a"], expected);
    p.tensors[0].dg = Some(digest(&p.tensors[1].image));
    p.tensors[1].dg = Some(digest(&p.tensors[0].image));
    rejects(&p.bytes(), None, Code::Asm004, "swapped digests");
}

// ---- the writer ---------------------------------------------------------------------------------------

fn tin(name: &str, dtype: Option<&str>, words: Vec<Vec<u8>>) -> TensorIn {
    TensorIn { name: name.to_string(), dtype: dtype.map(str::to_string), words }
}

#[test]
fn write_weights_lays_out_header_table_and_images() {
    let w0 = vec![(0..320).map(|i| (i % 256) as u8).collect::<Vec<u8>>(), vec![9; 320]];
    let w1 = vec![vec![5; 320]];
    let out =
        write_weights(&ref_profile(), &[tin("w0", Some("i8"), w0.clone()), tin("rawt", None, w1.clone())]).unwrap();
    let mut expect = Pkg::new("ref", vec![ent("w0", 2, 320, 0), ent("rawt", 1, 320, 0)]);
    expect.tensors[0].image = w0.concat();
    expect.tensors[0].dtype = 0;
    expect.tensors[1].image = w1.concat();
    expect.tensors[1].dtype = 255;
    assert_eq!(out, expect.bytes());
    assert_eq!(out.len() - payload_start(&out), 3 * 320);
}

#[test]
fn write_weights_uses_the_element_type_codes_of_the_catalogue() {
    let codes = [
        ("i8", 0u8),
        ("f16", 1),
        ("i32", 2),
        ("f32", 3),
        ("u8", 4),
        ("i16", 5),
        ("u16", 6),
        ("u32", 7),
        ("b8", 8),
        ("b16", 9),
        ("b32", 10),
        ("e4m3", 11),
        ("e5m2", 12),
        ("bf16", 16),
    ];
    for (name, code) in codes {
        let out = write_weights(&mini_profile(), &[tin("t", Some(name), vec![vec![0; 16]])]).unwrap();
        assert_eq!(header(&out)["__metadata__"]["dtype.t"], code.to_string(), "{name}");
    }
    let raw = write_weights(&mini_profile(), &[tin("t", None, vec![vec![0; 16]])]).unwrap();
    assert_eq!(header(&raw)["__metadata__"]["dtype.t"], "255");
    // every code the writer emits is one the reader accepts
    assert_eq!(ots_isa::ELTYPE_CODES.len(), codes.len());
}

#[test]
fn write_weights_preserves_the_profile_name_and_full_tensor_names() {
    let out = write_weights(&mini_profile(), &[tin("abcdefghijklmno", None, vec![vec![1; 16]])]).unwrap();
    let h = header(&out);
    assert_eq!(h["__metadata__"]["opentsp.profile"], "mini");
    assert_eq!(h["abcdefghijklmno"]["shape"], serde_json::json!([1, 16]));
    assert_eq!(h["abcdefghijklmno"]["data_offsets"], serde_json::json!([0, 16]));
}

#[test]
fn write_weights_rejects_a_bad_name_a_wrong_word_size_and_no_words_as_asm_004() {
    let ok = vec![vec![0u8; 320]];
    for name in ["", "1a", "a b", "a-b", "abcdefghijklmnop", "a\n", "\u{e9}", "a.b", "x\0"] {
        let e = write_weights(&ref_profile(), &[tin(name, None, ok.clone())]).unwrap_err();
        assert_eq!(e.code, Code::Asm004, "name {name:?}");
        assert!(e.msg.contains("identifier"), "{}", e.msg);
    }
    for good_name in ["a", "_", "_1", "A9", "abcdefghijklmno", "x_y_z"] {
        assert!(write_weights(&ref_profile(), &[tin(good_name, None, ok.clone())]).is_ok(), "{good_name}");
    }
    for words in [vec![vec![0u8; 319]], vec![vec![0u8; 321]], vec![vec![0u8; 320], vec![0u8; 16]], vec![vec![]], vec![]]
    {
        let e = write_weights(&ref_profile(), &[tin("t", None, words)]).unwrap_err();
        assert_eq!(e.code, Code::Asm004);
    }
    // the first failing tensor reports; the name is checked before the words
    let e =
        write_weights(&ref_profile(), &[tin("a", None, ok.clone()), tin("1b", None, vec![]), tin("c", None, vec![])])
            .unwrap_err();
    assert!(e.msg.contains("1b"), "{}", e.msg);
}

#[test]
fn write_weights_rejects_a_duplicate_name_as_asm_004_and_the_reader_rejects_a_package_that_has_one() {
    let t = tin("same", None, vec![vec![1; 16]]);
    let e = write_weights(&mini_profile(), &[t.clone(), t]).unwrap_err();
    assert_eq!(e.code, Code::Asm004);
    assert!(e.msg.contains("same") && e.msg.contains("more than once"), "{}", e.msg);
    // the first of the two reports when a name is also bad; a duplicate under another element type is a duplicate
    let e = write_weights(
        &mini_profile(),
        &[
            tin("a", None, vec![vec![1; 16]]),
            tin("b", Some("i8"), vec![vec![1; 16]]),
            tin("a", Some("i8"), vec![vec![1; 16]]),
        ],
    )
    .unwrap_err();
    assert_eq!(e.code, Code::Asm004);
    let package = Pkg::new("mini", vec![ent("same", 1, 16, 1), ent("same", 1, 16, 2)]);
    rejects(&package.bytes(), None, Code::Asm003, "duplicate name in a package");
}

#[test]
fn write_weights_rejects_an_unknown_element_type_as_asm_004_after_the_name_and_words_of_that_tensor() {
    let e = write_weights(&mini_profile(), &[tin("t", Some("f64"), vec![vec![0; 16]])]).unwrap_err();
    assert_eq!(e.code, Code::Asm004);
    assert!(e.msg.contains("f64"), "{}", e.msg);
    let e = write_weights(&mini_profile(), &[tin("1t", Some("f64"), vec![vec![0; 16]])]).unwrap_err();
    assert!(e.msg.contains("identifier"), "{}", e.msg);
    assert_eq!(
        write_weights(&mini_profile(), &[tin("t", Some("raw"), vec![vec![0; 16]])]).unwrap_err().code,
        Code::Asm004
    );
}

#[test]
fn what_the_writer_writes_the_reader_reads() {
    let w: Vec<Vec<u8>> = (0..5).map(|k| (0..320).map(|l| (k * 7 + l) as u8).collect()).collect();
    let out =
        write_weights(&ref_profile(), &[tin("w", Some("f16"), w.clone()), tin("x", None, vec![vec![3; 320]])]).unwrap();
    let t = read_weights(&out, Some("ref"), None).unwrap();
    assert_eq!(t["w"].dtype, 1);
    assert_eq!(t["w"].len(), 5);
    for (k, word) in w.iter().enumerate() {
        assert_eq!(t["w"].word(k), word.as_slice());
    }
    assert_eq!(t["x"].dtype, 255);
}

// ---- the packages tsp_as.py wpack wrote -----------------------------------------------------------------

/// `pat(t, n, vl)`: the bytes of tensor `t`, `n` words of `vl` lanes (the generator of the fixtures).
fn pat(t: usize, n: usize, vl: usize) -> Vec<Vec<u8>> {
    (0..n).map(|j| (0..vl).map(|l| ((t * 53 + j * 31 + l * 7 + 1) & 0xff) as u8).collect()).collect()
}

#[test]
fn the_writer_is_byte_identical_to_tsp_as_wpack() {
    for (suffix, profile, vl) in [("", ref_profile(), 320usize), ("_mini", mini_profile(), 16)] {
        let example = write_weights(
            &profile,
            &[tin("x", Some("i8"), vec![vec![3; vl]]), tin("y", Some("i8"), vec![vec![4; vl]])],
        )
        .unwrap();
        assert_eq!(example, std::fs::read(format!("{DATA}/example{suffix}.safetensors")).unwrap(), "example{suffix}");
        let rich = write_weights(
            &profile,
            &[
                tin("w0", Some("i8"), pat(0, 3, vl)),
                tin("raw_t", None, pat(1, 1, vl)),
                tin("f", Some("f16"), pat(2, 2, vl)),
                tin("abcdefghijklmno", Some("u32"), pat(3, 1, vl)),
                tin("x", Some("i8"), vec![vec![3; vl]]),
                tin("y", Some("i8"), vec![vec![4; vl]]),
            ],
        )
        .unwrap();
        assert_eq!(rich, std::fs::read(format!("{DATA}/rich{suffix}.safetensors")).unwrap(), "rich{suffix}");
    }
}

#[test]
fn the_reader_reads_the_packages_tsp_as_wpack_wrote() {
    let t = read_weights(&std::fs::read(format!("{DATA}/rich.safetensors")).unwrap(), Some("ref"), None).unwrap();
    assert_eq!(t.keys().map(String::as_str).collect::<Vec<_>>(), ["abcdefghijklmno", "f", "raw_t", "w0", "x", "y"]);
    assert_eq!((t["w0"].dtype, t["raw_t"].dtype, t["f"].dtype, t["abcdefghijklmno"].dtype), (0, 255, 1, 7));
    for (name, i, n) in [("w0", 0, 3), ("raw_t", 1, 1), ("f", 2, 2), ("abcdefghijklmno", 3, 1)] {
        let want = pat(i, n, 320);
        assert_eq!(t[name].len(), n);
        for (k, w) in want.iter().enumerate() {
            assert_eq!(t[name].word(k), w.as_slice(), "{name} word {k}");
        }
    }
    assert!(read_weights(&std::fs::read(format!("{DATA}/rich_mini.safetensors")).unwrap(), Some("mini"), None).is_ok());
    // a mini package is not a ref one
    rejects(
        &std::fs::read(format!("{DATA}/rich_mini.safetensors")).unwrap(),
        Some("ref"),
        Code::Asm003,
        "mini package, --profile ref",
    );
}

// ---- bind_weights and preload_text ---------------------------------------------------------------------

fn x(name: &str, slice: u64, addr: u64, size: u64) -> Extern {
    Extern { name: name.to_string(), slice, addr, size }
}

fn tensors(list: &[(&str, usize)]) -> BTreeMap<String, Tensor> {
    let mut data = Pkg::new("mini", vec![]);
    for (n, words) in list {
        data.tensors.push(ent(n, *words as u32, 16, 0));
    }
    // distinct bytes per word: word k of tensor j is filled with 16 * j + k
    for (j, t) in data.tensors.iter_mut().enumerate() {
        for (k, c) in t.image.chunks_mut(16).enumerate() {
            c.fill((16 * j + k) as u8);
        }
    }
    read_weights(&data.bytes(), None, None).unwrap()
}

#[test]
fn bind_weights_gives_one_preload_per_word_in_extern_order() {
    let t = tensors(&[("a", 2), ("b", 1), ("unused", 4)]);
    let pre = bind_weights(&[x("b", 9, 0x30, 1), x("a", 5, 0x10, 2)], &t).unwrap();
    let got: Vec<(u64, u64, u8)> = pre.iter().map(|p| (p.slice, p.addr, p.word[0])).collect();
    // b is tensor 1 (fill 16), a is tensor 0 (fills 0, 1)
    assert_eq!(got, vec![(9, 0x30, 16), (5, 0x10, 0), (5, 0x11, 1)]);
    assert!(pre.iter().all(|p| p.word.len() == 16));
}

#[test]
fn bind_weights_of_no_externs_is_no_preloads_and_a_package_may_hold_more_tensors() {
    assert!(bind_weights(&[], &tensors(&[("a", 1)])).unwrap().is_empty());
    assert!(bind_weights(&[], &tensors(&[])).unwrap().is_empty());
}

#[test]
fn bind_weights_rejects_a_missing_tensor_and_another_word_count_as_asm_004() {
    let t = tensors(&[("a", 2)]);
    let e = bind_weights(&[x("b", 0, 0, 1)], &t).unwrap_err();
    assert_eq!(e.code, Code::Asm004);
    assert!(e.msg.contains("no tensor"), "{}", e.msg);
    for size in [1, 3, 4] {
        let e = bind_weights(&[x("a", 0, 0, size)], &t).unwrap_err();
        assert_eq!(e.code, Code::Asm004, "size {size}");
        assert!(e.msg.contains("words"), "{}", e.msg);
    }
    // the first failing extern reports, and nothing is returned for the ones before it
    let e = bind_weights(&[x("a", 0, 0, 2), x("c", 1, 0, 1), x("d", 2, 0, 1)], &t).unwrap_err();
    assert!(e.msg.contains("extern c"), "{}", e.msg);
}

#[test]
fn names_are_case_sensitive_when_binding() {
    let t = tensors(&[("a", 1)]);
    assert_eq!(bind_weights(&[x("A", 0, 0, 1)], &t).unwrap_err().code, Code::Asm004);
}

#[test]
fn preload_text_is_one_s_line_per_word_with_unpadded_lower_case_hex() {
    let t = tensors(&[("a", 2)]);
    let pre = bind_weights(&[x("a", 0x2c, 0x1fff, 2)], &t).unwrap();
    assert_eq!(
        preload_text(&pre),
        "S 2c 1fff 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\nS 2c 2000 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n"
    );
    let t = tensors(&[("a", 1), ("b", 1)]);
    let pre = bind_weights(&[x("b", 4, 0, 1)], &t).unwrap();
    assert_eq!(preload_text(&pre), "S 4 0 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10\n");
    assert_eq!(preload_text(&[]), "");
}

#[test]
fn preload_text_prints_every_byte_value_without_padding() {
    let mut p = Pkg::new("mini", vec![ent("a", 1, 16, 0)]);
    p.tensors[0].image =
        (0u8..16).map(|i| [0, 1, 9, 10, 15, 16, 0x7f, 0x80, 0xff, 0xa0, 0x0a, 0xf0, 2, 3, 4, 5][i as usize]).collect();
    let t = read_weights(&p.bytes(), None, None).unwrap();
    let pre = bind_weights(&[x("a", 0, 0, 1)], &t).unwrap();
    assert_eq!(preload_text(&pre), "S 0 0 0 1 9 a f 10 7f 80 ff a0 a f0 2 3 4 5\n");
}
