//! Manifests, `.otsm` (`assembly.md` §4): the 64-byte header, the
//! K (offset, size) entries, the K embedded objects, and every rejection of the reader
//! (`OTS-E-ASM-003`, in the order of `tsp_as.read_manifest`). The manifests here are built byte by byte,
//! so the reader and the writer are each held to the layout and not to one another; the manifests
//! `tsp_as.py mpack` wrote (`ots-asm/tests/data/*.otsm`) are compared byte for byte in `ots-asm/tests/manifest.rs`.
//! `manifest_input` is the `tsp_sim --chips K` input of `mobjinput`.
use ots_format::{
    bind_weights,
    manifest::{
        chip_count_ok, manifest_input, read_manifest, write_manifest, ENTRY_SIZE, FORMAT_REVISION, HEADER_SIZE,
        TOPOLOGY_NODE,
    },
    preload_text, read_weights,
    text::{input_text, Images},
    weights::TensorIn,
    write_object, write_weights, Extern,
};
use ots_isa::{queue_id, AsmError, Code, CONTROL};
use ots_profile::Profile;
use sha2::{Digest, Sha256};

const SPEC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../opentsp-arch-spec");

fn digest12(path: &str) -> [u8; 12] {
    let d: [u8; 32] = Sha256::digest(std::fs::read(path).unwrap()).into();
    d[..12].try_into().unwrap()
}

fn name16(s: &str) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[..s.len()].copy_from_slice(s.as_bytes());
    b
}

fn mem(i: usize) -> u16 {
    queue_id(&format!("MEM[{i}]")).unwrap()
}

/// A valid object of `profile`, distinguished by `tag` (its text and its one extern).
fn object(profile: &str, tag: u32) -> Vec<u8> {
    let mut images = Images::new();
    images.extend(mem(4), &[0x21, 0x1000_0010, tag]);
    images.extend(CONTROL, &[0x30]);
    let externs = [Extern { name: format!("w{tag}"), slice: 4, addr: 0x10 + tag as u64, size: 2 }];
    write_object(&Profile::load(profile).unwrap(), &images, &externs)
}

/// A hand-built manifest: header fields, then the entry table, then the objects.
#[derive(Clone)]
struct Man {
    magic: [u8; 4],
    major: u8,
    minor: u8,
    hsize: u16,
    profile: [u8; 16],
    k: Option<u32>,
    topology: u32,
    revision: u32,
    reserved: u32,
    enc: [u8; 12],
    prof: [u8; 12],
    objects: Vec<Vec<u8>>,
    /// replaces the entry (offset, size) of chip i
    entries: Vec<(usize, u32, u32)>,
    trailer: Vec<u8>,
}

impl Man {
    fn new(profile: &str, objects: Vec<Vec<u8>>) -> Man {
        Man {
            magic: *b"OTSM",
            major: 0,
            minor: 1,
            hsize: 64,
            profile: name16(profile),
            k: None,
            topology: 0,
            revision: 2,
            reserved: 0,
            enc: digest12(&format!("{SPEC}/isa-db/generated/encoding.json")),
            prof: digest12(&format!("{SPEC}/config/{profile}.json")),
            objects,
            entries: vec![],
            trailer: vec![],
        }
    }

    fn bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend(self.magic);
        b.push(self.major);
        b.push(self.minor);
        b.extend(self.hsize.to_le_bytes());
        b.extend(self.profile);
        b.extend(self.k.unwrap_or(self.objects.len() as u32).to_le_bytes());
        b.extend(self.topology.to_le_bytes());
        b.extend(self.revision.to_le_bytes());
        b.extend(self.reserved.to_le_bytes());
        b.extend(self.enc);
        b.extend(self.prof);
        assert_eq!(b.len(), 64);
        let mut off = 64 + 8 * self.objects.len() as u32;
        for (i, o) in self.objects.iter().enumerate() {
            let (off_i, size_i) = self.entries.iter().find(|e| e.0 == i).map_or((off, o.len() as u32), |e| (e.1, e.2));
            b.extend(off_i.to_le_bytes());
            b.extend(size_i.to_le_bytes());
            off += o.len() as u32;
        }
        for o in &self.objects {
            b.extend(o);
        }
        b.extend(&self.trailer);
        b
    }
}

fn rejected(r: Result<(String, Vec<&[u8]>), AsmError>, what: &str) -> AsmError {
    match r {
        Err(e) => {
            assert_eq!(e.code, Code::Asm003, "{what}: {e}");
            assert!(e.where_.is_none(), "{what}: the command is the location");
            e
        }
        Ok(_) => panic!("{what}: accepted"),
    }
}

fn read(m: &Man) -> Result<usize, AsmError> {
    read_manifest(&m.bytes(), None, None).map(|(_, o)| o.len())
}

fn rej(m: &Man, what: &str) -> AsmError {
    rejected(read_manifest(&m.bytes(), None, None), what)
}

fn two(profile: &str) -> Vec<Vec<u8>> {
    vec![object(profile, 0), object(profile, 1)]
}

#[test]
fn the_layout_constants() {
    assert_eq!((HEADER_SIZE, ENTRY_SIZE, FORMAT_REVISION, TOPOLOGY_NODE), (64, 8, 2, 0));
}

#[test]
fn the_chip_counts_are_1_to_512() {
    for k in 0..=600u64 {
        assert_eq!(chip_count_ok(k), (1..=512).contains(&k), "K = {k}");
    }
    assert!(!chip_count_ok(u64::from(u32::MAX)));
}

#[test]
fn a_writer_without_a_topology_takes_the_chips_of_the_wiring_and_one_with_it_up_to_512() {
    use ots_format::{write_manifest_for, MAX_CHIPS, MAX_WIRED_CHIPS};
    assert_eq!((MAX_WIRED_CHIPS, MAX_CHIPS), (264, 512));
    let e = write_manifest("ref", &vec![Vec::new(); 265], None).unwrap_err();
    assert!(e.msg.contains("265 chips") && e.msg.contains("--chips wiring"), "{e}");
    let e = write_manifest_for("ref", &vec![Vec::new(); 513], None, MAX_CHIPS).unwrap_err();
    assert!(e.msg.contains("513 chips") && e.msg.contains("topology file"), "{e}");
    // 300 chips: refused for the wiring, accepted by the count rule of a topology (the objects are then read, the first is empty)
    let e = write_manifest_for("ref", &vec![Vec::new(); 300], None, MAX_CHIPS).unwrap_err();
    assert!(e.msg.starts_with("chip 0: "), "{e}");
}

#[test]
fn a_hand_built_manifest_is_read() {
    let objs = two("ref");
    let bytes = Man::new("ref", objs.clone()).bytes();
    let (p, got) = read_manifest(&bytes, None, None).unwrap();
    assert_eq!(p, "ref");
    assert_eq!(got.iter().map(|o| o.to_vec()).collect::<Vec<_>>(), objs);
    let bytes = Man::new("mini", vec![object("mini", 7)]).bytes();
    let (p, got) = read_manifest(&bytes, Some("mini"), None).unwrap();
    assert_eq!((p.as_str(), got.len()), ("mini", 1));
}

#[test]
fn the_writer_makes_the_hand_built_bytes() {
    for (profile, n) in [("ref", 1usize), ("ref", 2), ("mini", 3), ("mini", 8), ("ref", 16), ("mini", 24), ("mini", 64)]
    {
        let objs: Vec<Vec<u8>> = (0..n).map(|i| object(profile, i as u32)).collect();
        let want = Man::new(profile, objs.clone()).bytes();
        assert_eq!(write_manifest(profile, &objs, None).unwrap(), want, "{profile} x {n}");
        assert_eq!(&want[..8], b"OTSM\x00\x01\x40\x00");
        assert_eq!(u32::from_le_bytes(want[24..28].try_into().unwrap()) as usize, n);
    }
}

#[test]
fn the_header_binds_the_interface_and_chip_count() {
    let o = object("ref", 0);
    let m = write_manifest("ref", std::slice::from_ref(&o), None).unwrap();
    // the encoding and profile digests are those of an object of the profile
    assert_eq!(m[40..52], o[92..104]);
    assert_eq!(m[52..64], o[124..136]);
    assert_eq!(m[8..24], o[76..92]);
    // entry 0 starts at 64 + 8 K and takes the whole object
    assert_eq!(&m[64..72], &[&72u32.to_le_bytes()[..], &(o.len() as u32).to_le_bytes()[..]].concat()[..]);
    assert_eq!(&m[72..], &o[..]);
}

#[test]
fn the_writer_refuses_a_chip_count_the_wiring_cannot_take() {
    for n in [0usize, 265, 300, 512] {
        let objs: Vec<Vec<u8>> = (0..n).map(|i| object("mini", i as u32)).collect();
        let e = write_manifest("mini", &objs, None).unwrap_err();
        assert_eq!((e.code, e.where_.as_deref()), (Code::Asm003, None), "{n} chips");
        assert!(e.msg.contains("chips"), "{e}");
    }
}

#[test]
fn the_writer_checks_every_object_and_names_the_chip() {
    let good = object("ref", 0);
    let mut bad_magic = object("ref", 1);
    bad_magic[0] = b'X';
    let e = write_manifest("ref", &[good.clone(), bad_magic], None).unwrap_err();
    assert_eq!(e.code, Code::Asm003);
    assert!(e.msg.starts_with("chip 1: "), "{e}");
    // an object of another profile, an empty file, a manifest
    let e = write_manifest("ref", &[object("mini", 0)], None).unwrap_err();
    assert!(e.msg.starts_with("chip 0: ") && e.msg.contains("mini"), "{e}");
    assert_eq!(write_manifest("ref", &[Vec::new()], None).unwrap_err().code, Code::Asm003);
    let m = write_manifest("ref", std::slice::from_ref(&good), None).unwrap();
    assert_eq!(write_manifest("ref", &[m], None).unwrap_err().code, Code::Asm003);
    // the count is checked first
    let e = write_manifest("ref", &vec![Vec::new(); 265], None).unwrap_err();
    assert!(e.msg.contains("265 chips"), "{e}");
    // an unknown profile is ASM-003 too
    assert_eq!(write_manifest("nosuch", &[good], None).unwrap_err().code, Code::Asm003);
}

#[test]
fn the_reader_takes_every_chip_count_of_the_wiring_and_no_other() {
    // one object read per chip: the edges of every run of counts (the predicate itself is tested above)
    let one = object("mini", 0);
    for k in [1usize, 2, 7, 8, 9, 15, 16, 17, 23, 24, 25, 56, 63, 64, 65] {
        let m = Man::new("mini", vec![one.clone(); k]);
        assert_eq!(read(&m).is_ok(), chip_count_ok(k as u64), "K = {k}");
    }
    // the count is the header's: it must match the entries
    let m = Man { k: Some(9), ..Man::new("mini", vec![object("mini", 0)]) };
    rej(&m, "K = 9 of 1 object");
    let m = Man { k: Some(0), ..Man::new("mini", vec![object("mini", 0)]) };
    rej(&m, "K = 0");
    let m = Man { k: Some(u32::MAX), ..Man::new("mini", vec![object("mini", 0)]) };
    rej(&m, "K = 2^32-1");
}

#[test]
fn an_empty_or_short_file_or_another_magic_is_rejected() {
    rejected(read_manifest(b"", None, None), "empty");
    rejected(read_manifest(b"OTSM", None, None), "magic alone");
    let good = Man::new("ref", two("ref")).bytes();
    rejected(read_manifest(&good[..63], None, None), "63 bytes");
    for m in [*b"OTSO", *b"OTSW", *b"otsm", *b"\0\0\0\0", *b"\x7fELF"] {
        rej(&Man { magic: m, ..Man::new("ref", two("ref")) }, "magic");
    }
    // an object is not a manifest
    rejected(read_manifest(&object("ref", 0), None, None), "an object");
}

#[test]
fn the_version_and_the_header_size_must_be_0_1_and_64() {
    for (major, minor, hsize) in
        [(0, 2, 64), (1, 1, 64), (1, 0, 64), (0, 0, 64), (0, 1, 60), (0, 1, 68), (0, 1, 0), (255, 255, 64)]
    {
        rej(
            &Man { major, minor, hsize, ..Man::new("ref", two("ref")) },
            &format!("version {major}.{minor}, header {hsize}"),
        );
    }
}

#[test]
fn the_format_revision_and_the_reserved_word_must_be_2_and_0() {
    for (revision, reserved) in [(0, 0), (1, 0), (3, 0), (2, 1), (2, 0x8000_0000), (u32::MAX, u32::MAX)] {
        rej(
            &Man { revision, reserved, ..Man::new("ref", two("ref")) },
            &format!("revision {revision}, reserved {reserved:#x}"),
        );
    }
}

#[test]
fn the_topology_must_be_0() {
    for topology in [1, 2, 0x100, u32::MAX] {
        let e = rej(&Man { topology, ..Man::new("ref", two("ref")) }, &format!("topology {topology}"));
        assert!(e.msg.contains("topology"), "{e}");
    }
}

#[test]
fn the_profile_field_is_nul_padded_ascii_text() {
    let with = |p: [u8; 16]| Man { profile: p, ..Man::new("ref", two("ref")) };
    let mut b = name16("ref");
    b[5] = b'x'; // text after a NUL
    rej(&with(b), "text after NUL");
    rej(&with([0; 16]), "empty");
    let mut b = name16("ref");
    b[1] = 0xe9; // not ASCII
    rej(&with(b), "non-ASCII");
    let mut b = name16("ref");
    b[15] = 1;
    rej(&with(b), "a byte at the end");
    for p in ["nosuch", "REF", "re f", "ref\n", "a/b"] {
        rej(&with(name16(p)), p);
    }
    rej(&with(*b"abcdefghijklmnop"), "16 characters");
}

#[test]
fn the_profile_option_must_equal_the_header() {
    let bytes = Man::new("mini", vec![object("mini", 0)]).bytes();
    assert!(read_manifest(&bytes, None, None).is_ok());
    assert!(read_manifest(&bytes, Some("mini"), None).is_ok());
    let e = rejected(read_manifest(&bytes, Some("ref"), None), "--profile ref of a mini manifest");
    assert!(e.msg.contains("mini"), "{e}");
    rejected(read_manifest(&bytes, Some(""), None), "--profile empty");
}

#[test]
fn the_digests_must_match_the_encoding_and_the_profile() {
    let base = || Man::new("ref", two("ref"));
    let mut enc = base().enc;
    enc[11] ^= 1;
    let e = rej(&Man { enc, ..base() }, "encoding digest");
    assert!(e.msg.contains("encoding.json"), "{e}");
    let mut prof = base().prof;
    prof[0] ^= 0x80;
    let e = rej(&Man { prof, ..base() }, "profile digest");
    assert!(e.msg.contains("configuration"), "{e}");
    rej(&Man { prof: Man::new("mini", vec![]).prof, ..base() }, "mini digest on ref");
    // a profile read from --config is held to its own bytes
    let dir = std::env::temp_dir().join(format!("ots-manifest-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = dir.join("ref22.json");
    let cfg_s = cfg.to_str().unwrap();
    let text = std::fs::read_to_string(format!("{SPEC}/config/ref.json")).unwrap();
    std::fs::write(&cfg, text.replace("\"mem_slices_per_hemisphere\": 44", "\"mem_slices_per_hemisphere\": 22"))
        .unwrap();
    let shipped = base().bytes();
    rejected(read_manifest(&shipped, None, Some(cfg_s)), "a manifest against another configuration");
    let objs = vec![write_object(&Profile::load_file("ref", &cfg).unwrap(), &Images::new(), &[])];
    let own = write_manifest("ref", &objs, Some(cfg_s)).unwrap();
    assert!(read_manifest(&own, None, Some(cfg_s)).is_ok());
    rejected(read_manifest(&own, None, None), "a manifest of the --config file, read against the shipped one");
    // the writer refuses objects of the shipped configuration under --config
    assert_eq!(write_manifest("ref", &[object("ref", 0)], Some(cfg_s)).unwrap_err().code, Code::Asm003);
    let old = dir.join("old.json");
    std::fs::write(&old, text.replace("\"spec_version\": \"0.1\"", "\"spec_version\": \"0.2\"")).unwrap();
    rejected(read_manifest(&shipped, None, Some(old.to_str().unwrap())), "config of version 0.2");
    rejected(read_manifest(&shipped, None, Some(dir.join("none.json").to_str().unwrap())), "missing config");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_chip_count_is_checked_after_the_digests() {
    // K = 513 is not a chip count, but a bad digest is reported first (the text of the message shows it)
    let mut enc = Man::new("ref", vec![]).enc;
    enc[0] ^= 1;
    let m = Man { k: Some(513), enc, ..Man::new("ref", two("ref")) };
    assert!(rej(&m, "bad digest and bad K").msg.contains("encoding.json"));
    let e = rej(&Man { k: Some(513), ..Man::new("ref", two("ref")) }, "bad K");
    assert!(e.msg.contains("chip count"), "{e}");
}

#[test]
fn the_size_must_hold_the_entry_table() {
    // K = 2 with 64 + 15 bytes: the table does not fit
    let good = Man::new("ref", two("ref")).bytes();
    for n in [64usize, 64 + 8, 64 + 15] {
        let e = rejected(read_manifest(&good[..n], None, None), &format!("{n} bytes"));
        assert!(e.msg.contains("size"), "{e}");
    }
    // K = 64 needs 64 + 512 bytes
    let big = Man::new("mini", vec![object("mini", 0); 64]).bytes();
    assert!(read_manifest(&big, None, None).is_ok());
    rejected(read_manifest(&big[..64 + 511], None, None), "K = 64, table cut");
}

#[test]
fn an_entry_must_follow_its_predecessor_and_have_a_size() {
    let objs = || two("ref");
    let n0 = objs()[0].len() as u32;
    let n1 = objs()[1].len() as u32;
    let start = 64 + 16;
    // entry 0 elsewhere; entry 1 elsewhere; a gap; an overlap
    for (what, entries) in [
        ("entry 0 offset + 1", vec![(0usize, start + 1, n0)]),
        ("entry 0 offset - 1", vec![(0, start - 1, n0)]),
        ("entry 0 at 0", vec![(0, 0, n0)]),
        ("entry 1 gap", vec![(1, start + n0 + 1, n1)]),
        ("entry 1 overlaps", vec![(1, start + n0 - 1, n1)]),
        ("entry 1 swapped with 0", vec![(0, start + n0, n1), (1, start, n0)]),
        ("entry 0 size 0", vec![(0, start, 0)]),
        ("entry 1 size 0", vec![(1, start + n0, 0)]),
        ("entry 0 past the end", vec![(0, start, n0 + n1 + 1)]),
        ("entry 1 past the end", vec![(1, start + n0, n1 + 1)]),
        ("entry 1 size 2^32-1", vec![(1, start + n0, u32::MAX)]),
        ("entry 1 offset 2^32-1", vec![(1, u32::MAX, 1)]),
    ] {
        let e = rej(&Man { entries, ..Man::new("ref", objs()) }, what);
        assert!(e.msg.starts_with("manifest entry") || e.msg.contains("size"), "{what}: {e}");
    }
    // a shorter entry that still follows its predecessor leaves a gap before the next
    rej(&Man { entries: vec![(0, start, n0 - 1)], ..Man::new("ref", objs()) }, "entry 0 one byte short");
}

#[test]
fn the_manifest_ends_with_its_last_entry() {
    let m = Man { trailer: vec![0], ..Man::new("ref", two("ref")) };
    let e = rej(&m, "one byte after the last entry");
    assert!(e.msg.contains("size"), "{e}");
    rej(&Man { trailer: vec![0; 64], ..Man::new("ref", two("ref")) }, "64 bytes after");
    let good = Man::new("ref", two("ref")).bytes();
    rejected(read_manifest(&good[..good.len() - 1], None, None), "one byte short");
}

#[test]
fn every_embedded_object_is_read_and_the_message_names_the_chip() {
    let mut cut = object("ref", 1);
    cut.truncate(cut.len() - 1);
    let e = rej(&Man::new("ref", vec![object("ref", 0), cut]), "a truncated object");
    assert!(e.msg.starts_with("manifest entry 1: "), "{e}");
    let mut magic = object("ref", 0);
    magic[3] = b'X';
    assert!(rej(&Man::new("ref", vec![magic, object("ref", 1)]), "bad magic").msg.starts_with("manifest entry 0: "));
    // an object of the other profile in a manifest of ref
    let e = rej(&Man::new("ref", vec![object("ref", 0), object("ref", 1), object("mini", 0)]), "an object of mini");
    assert!(e.msg.starts_with("manifest entry 2: ") && e.msg.contains("mini"), "{e}");
    // a manifest in a manifest
    let inner = write_manifest("ref", &[object("ref", 0)], None).unwrap();
    rej(&Man::new("ref", vec![inner]), "a manifest as a chip");
}

#[test]
fn the_reader_stops_at_the_first_fault_in_the_order_of_the_reference() {
    // one fault from each check, then the same plus a fault of every later check: the first message stays
    let good = || Man::new("ref", two("ref"));
    let mut bad_obj = object("ref", 0);
    bad_obj[0] = 0;
    let mut enc = good().enc;
    enc[0] ^= 1;
    let faulty = Man {
        magic: *b"OTSM",
        major: 0,
        minor: 2,
        revision: 9,
        topology: 3,
        profile: name16("nosuch"),
        enc,
        k: Some(513),
        trailer: vec![1],
        objects: vec![bad_obj],
        ..good()
    };
    type Step = (&'static str, fn(&mut Man), &'static str);
    let steps: [Step; 8] = [
        ("version", |m| m.minor = 1, "revision"),
        ("revision", |m| m.revision = 2, "topology"),
        ("topology", |m| m.topology = 0, "profile"),
        ("profile", |m| m.profile = name16("ref"), "encoding.json"),
        ("digest", |m| m.enc = Man::new("ref", vec![]).enc, "chip count"),
        ("chip count", |m| m.k = None, "manifest entry 0"),
        ("object", |m| m.objects = two("ref"), "size"),
        ("trailer", |m| m.trailer.clear(), ""),
    ];
    let mut m = faulty;
    for (what, fix, next) in steps {
        let e = read(&m).unwrap_err();
        assert!(e.msg.contains(&what_marker(what)), "{what}: {e}");
        fix(&mut m);
        if next.is_empty() {
            assert!(read(&m).is_ok());
        }
    }
}

fn what_marker(what: &str) -> String {
    match what {
        "version" => "spec version",
        "revision" => "format revision",
        "topology" => "topology",
        "profile" => "profile",
        "digest" => "encoding.json",
        "chip count" => "chip count",
        "object" => "manifest entry 0",
        "trailer" => "size",
        _ => unreachable!(),
    }
    .to_string()
}

// ---- manifest_input ----

fn weights(profile: &str, names: &[&str], words: usize, seed: u8) -> Vec<u8> {
    let p = Profile::load(profile).unwrap();
    let vl = p.vlanes() as usize;
    let tensors: Vec<TensorIn> = names
        .iter()
        .enumerate()
        .map(|(t, n)| TensorIn {
            name: n.to_string(),
            dtype: Some("i8".into()),
            words: (0..words)
                .map(|j| (0..vl).map(|l| seed.wrapping_add((t * 7 + j * 3 + l) as u8)).collect())
                .collect(),
        })
        .collect();
    write_weights(&p, &tensors).unwrap()
}

/// The input of chip `i` as `objinput` prints it for one object.
fn chip_input(obj: &[u8], package: Option<&[u8]>, profile: &str) -> String {
    let (_, images, externs) = ots_format::read_object(obj, Some(profile), None).unwrap();
    let mut out = String::new();
    if let Some(w) = package {
        let t = read_weights(w, Some(profile), None).unwrap();
        out += &preload_text(&bind_weights(&externs, &t).unwrap());
    }
    out + &input_text(&images)
}

#[test]
fn the_input_has_a_c_line_then_the_q_lines_of_every_chip() {
    for profile in ["ref", "mini"] {
        let objs: Vec<Vec<u8>> = (0..3).map(|i| object(profile, i)).collect();
        let m = write_manifest(profile, &objs, None).unwrap();
        let mut want = String::new();
        for (i, o) in objs.iter().enumerate() {
            want += &format!("C {i:x}\n{}", chip_input(o, None, profile));
        }
        assert_eq!(manifest_input(&m, &[], None).unwrap(), want, "{profile}");
        assert!(want.starts_with("C 0\nQ "), "{want}");
    }
}

#[test]
fn the_chip_numbers_are_hex() {
    let objs: Vec<Vec<u8>> = (0..16).map(|i| object("mini", i % 3)).collect();
    let out = manifest_input(&write_manifest("mini", &objs, None).unwrap(), &[], None).unwrap();
    let c: Vec<&str> = out.lines().filter(|l| l.starts_with("C ")).collect();
    assert_eq!(c.len(), 16);
    assert_eq!((c[9], c[10], c[15]), ("C 9", "C a", "C f"));
}

#[test]
fn a_weight_package_per_chip_puts_the_s_lines_after_the_c_line() {
    let objs = vec![object("mini", 0), object("mini", 1)];
    let m = write_manifest("mini", &objs, None).unwrap();
    let pk = vec![weights("mini", &["w0"], 2, 1), weights("mini", &["w1", "unused"], 2, 50)];
    let mut want = String::new();
    for (i, o) in objs.iter().enumerate() {
        want += &format!("C {i:x}\n{}", chip_input(o, Some(&pk[i]), "mini"));
    }
    let got = manifest_input(&m, &pk, None).unwrap();
    assert_eq!(got, want);
    let lines: Vec<&str> = got.lines().collect();
    assert_eq!(lines[0], "C 0");
    assert!(lines[1].starts_with("S 4 10 ") && lines[2].starts_with("S 4 11 "), "{got}");
    assert!(lines[3].starts_with("Q "));
    // the packages are bound in chip order: the packages of chips 0 and 1 swapped do not bind
    let swapped = vec![pk[1].clone(), pk[0].clone()];
    assert_eq!(manifest_input(&m, &swapped, None).unwrap_err().code, Code::Asm004);
}

#[test]
fn the_number_of_weight_packages_is_zero_or_one_per_chip() {
    let objs = vec![object("mini", 0), object("mini", 1), object("mini", 2)];
    let m = write_manifest("mini", &objs, None).unwrap();
    let p = |n: u32| weights("mini", &[&format!("w{n}")], 2, n as u8);
    assert!(manifest_input(&m, &[p(0), p(1), p(2)], None).is_ok());
    for k in [1usize, 2, 4] {
        let pk: Vec<Vec<u8>> = (0..k as u32).map(p).collect();
        let e = manifest_input(&m, &pk, None).unwrap_err();
        assert_eq!(e.code, Code::Asm004, "{k} packages");
        assert!(e.msg.contains(&format!("{k} weight packages for 3 chips")), "{e}");
    }
}

#[test]
fn a_manifest_that_does_not_read_is_asm_003_before_the_packages_are_counted() {
    let e = manifest_input(b"OTSM", &[Vec::new()], None).unwrap_err();
    assert_eq!(e.code, Code::Asm003);
}

#[test]
fn a_package_that_does_not_bind_or_read_is_the_error_of_objinput() {
    let m = write_manifest("mini", &[object("mini", 0), object("mini", 1)], None).unwrap();
    let ok0 = weights("mini", &["w0"], 2, 1);
    // chip 1's package misses its tensor / has another word count: ASM-004
    assert_eq!(
        manifest_input(&m, &[ok0.clone(), weights("mini", &["w0"], 2, 1)], None).unwrap_err().code,
        Code::Asm004
    );
    assert_eq!(
        manifest_input(&m, &[ok0.clone(), weights("mini", &["w1"], 3, 1)], None).unwrap_err().code,
        Code::Asm004
    );
    // a package for another profile, a package that is not one: ASM-003
    assert_eq!(manifest_input(&m, &[ok0.clone(), weights("ref", &["w1"], 2, 1)], None).unwrap_err().code, Code::Asm003);
    assert_eq!(manifest_input(&m, &[ok0.clone(), b"OTSW".to_vec()], None).unwrap_err().code, Code::Asm003);
    // a digest mismatch: ASM-004
    let mut bad = weights("mini", &["w1"], 2, 1);
    let n = bad.len();
    bad[n - 1] ^= 1;
    assert_eq!(manifest_input(&m, &[ok0, bad], None).unwrap_err().code, Code::Asm004);
}

#[test]
fn nothing_is_returned_when_a_later_chip_fails() {
    // the result is the whole text or an error, never the text of the chips before the failure
    let m = write_manifest("mini", &[object("mini", 0), object("mini", 1)], None).unwrap();
    assert!(manifest_input(&m, &[weights("mini", &["w0"], 2, 1), weights("mini", &["zz"], 2, 1)], None).is_err());
}

#[test]
fn an_extern_free_manifest_with_packages_prints_no_s_lines() {
    let mut images = Images::new();
    images.extend(mem(4), &[0x21]);
    let o = write_object(&Profile::load("mini").unwrap(), &images, &[]);
    let m = write_manifest("mini", std::slice::from_ref(&o), None).unwrap();
    let pk = weights("mini", &["unused"], 1, 3);
    assert_eq!(manifest_input(&m, &[pk], None).unwrap(), format!("C 0\n{}", input_text(&images)));
}
