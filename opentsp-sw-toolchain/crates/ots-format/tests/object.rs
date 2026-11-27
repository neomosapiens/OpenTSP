//! Hand-built ELF32 input and resolved-program validation.
use ots_format::{
    object::{read_object, write_object, FORMAT_REVISION, HEADER_SIZE},
    text::Images,
    Extern,
};
use ots_isa::{queue_id, AsmError, Code, CONTROL};
use ots_profile::Profile;
use sha2::{Digest, Sha256};

const SPEC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../opentsp-arch-spec");

fn digest(path: &str) -> [u8; 32] {
    let d: [u8; 32] = Sha256::digest(std::fs::read(path).unwrap()).into();
    d
}

/// A hand-built object: header fields, then the image table, the units, the externs.
struct Obj {
    magic: [u8; 4],
    major: u8,
    minor: u8,
    hsize: u16,
    profile: [u8; 16],
    nimage: Option<u32>,
    nextern: Option<u32>,
    revision: u32,
    reserved: u32,
    enc: [u8; 32],
    prof: [u8; 32],
    images: Vec<(u16, u16, Vec<u32>)>,
    externs: Vec<[u8; 32]>,
    trailer: Vec<u8>,
}

fn name16(s: &str) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[..s.len()].copy_from_slice(s.as_bytes());
    b
}

fn ext(name: &[u8], slice: u16, addr: u32, size: u32) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[..name.len()].copy_from_slice(name);
    b[16..18].copy_from_slice(&slice.to_le_bytes());
    b[20..24].copy_from_slice(&addr.to_le_bytes());
    b[24..28].copy_from_slice(&size.to_le_bytes());
    b
}

impl Obj {
    fn new(profile: &str) -> Obj {
        Obj {
            magic: *b"\x7fELF",
            major: 1,
            minor: 1,
            hsize: 52,
            profile: name16(profile),
            nimage: None,
            nextern: None,
            revision: 1,
            reserved: 0,
            enc: digest(&format!("{SPEC}/isa-db/generated/encoding.json")),
            prof: digest(&format!("{SPEC}/config/{profile}.json")),
            images: vec![],
            externs: vec![],
            trailer: vec![],
        }
    }

    fn image(mut self, q: u16, units: &[u32]) -> Obj {
        self.images.push((q, 0, units.to_vec()));
        self
    }

    fn bytes(&self) -> Vec<u8> {
        let mut note = Vec::new();
        note.extend(8u32.to_le_bytes());
        note.extend(84u32.to_le_bytes());
        note.extend(1u32.to_le_bytes());
        note.extend(b"OpenTSP\0");
        note.extend((self.revision as u16).to_le_bytes());
        note.extend((self.reserved as u16).to_le_bytes());
        note.extend(self.profile);
        note.extend(self.enc);
        note.extend(self.prof);
        let mut iq = self.nimage.unwrap_or(self.images.len() as u32).to_le_bytes().to_vec();
        for (queue, reserved, units) in &self.images {
            iq.extend(queue.to_le_bytes());
            iq.extend(reserved.to_le_bytes());
            iq.extend((units.len() as u32).to_le_bytes());
        }
        for (_, _, units) in &self.images {
            for unit in units {
                iq.extend(unit.to_le_bytes());
            }
        }
        let mut load = self.nextern.unwrap_or(self.externs.len() as u32).to_le_bytes().to_vec();
        for entry in &self.externs {
            load.extend(entry);
        }
        load.extend(&self.trailer);

        let names = b"\0.note.opentsp\0.tsp.iq\0.tsp.load\0.shstrtab\0";
        let mut out = vec![0; 52];
        out[..4].copy_from_slice(&self.magic);
        out[4..7].copy_from_slice(&[self.major, self.minor, 1]);
        out[16..18].copy_from_slice(&1u16.to_le_bytes());
        out[20..24].copy_from_slice(&1u32.to_le_bytes());
        out[40..42].copy_from_slice(&self.hsize.to_le_bytes());
        out[46..52].copy_from_slice(&[40, 0, 5, 0, 4, 0]);
        let mut headers = vec![0; 40];
        for (name, kind, alignment, section) in
            [(1, 7, 4, note.as_slice()), (15, 1, 4, iq.as_slice()), (23, 1, 4, load.as_slice()), (33, 3, 1, names)]
        {
            let offset = out.len() as u32;
            for word in [name, kind, 0, 0, offset, section.len() as u32, 0, 0, alignment, 0] {
                headers.extend(word.to_le_bytes());
            }
            out.extend(section);
        }
        out.resize(out.len().next_multiple_of(4), 0);
        let directory = out.len() as u32;
        out[32..36].copy_from_slice(&directory.to_le_bytes());
        out.extend(headers);
        out
    }
}

fn read(o: &Obj) -> Result<(String, Images, Vec<Extern>), AsmError> {
    read_object(&o.bytes(), None, None)
}

fn rejected(r: Result<(String, Images, Vec<Extern>), AsmError>, what: &str) {
    match r {
        Err(e) => {
            assert_eq!(e.code, Code::Asm003, "{what}: {e}");
            assert!(e.where_.is_none(), "{what}: the command is the location");
        }
        Ok(_) => panic!("{what}: accepted"),
    }
}

fn mem(i: usize) -> u16 {
    queue_id(&format!("MEM[{i}]")).unwrap()
}

#[test]
fn header_is_52_bytes_and_abi_is_1() {
    assert_eq!((HEADER_SIZE, FORMAT_REVISION), (52, 1));
}

#[test]
fn a_hand_built_object_is_read() {
    let o = Obj {
        externs: vec![ext(b"x", 4, 0x10, 1), ext(b"y", 5, 0, 1)],
        ..Obj::new("ref").image(mem(4), &[0x21, 0x1000_0010, 0]).image(CONTROL, &[0x30])
    };
    let (p, images, externs) = read(&o).unwrap();
    assert_eq!(p, "ref");
    let got: Vec<(u16, Vec<u32>)> = images.iter().map(|(q, u)| (q, u.to_vec())).collect();
    assert_eq!(got, vec![(mem(4), vec![0x21, 0x1000_0010, 0]), (CONTROL, vec![0x30])]);
    assert_eq!(
        externs,
        vec![
            Extern { name: "x".into(), slice: 4, addr: 0x10, size: 1 },
            Extern { name: "y".into(), slice: 5, addr: 0, size: 1 }
        ]
    );
}

#[test]
fn an_object_of_no_text_and_no_extern_is_valid() {
    let o = Obj::new("mini");
    let (p, images, externs) = read(&o).unwrap();
    assert_eq!((p.as_str(), images.len(), externs.len()), ("mini", 0, 0));
    let profile = Profile::load("mini").unwrap();
    assert_eq!(read_object(&write_object(&profile, &Images::new(), &[]), None, None).unwrap(), read(&o).unwrap());
}

#[test]
fn writer_matches_the_independent_section_builder() {
    let o = Obj {
        externs: vec![ext(b"w0", 7, 0x1ff0, 0x10)],
        ..Obj::new("ref").image(3, &[0x21]).image(mem(7), &[1, 2, 3]).image(CONTROL, &[0x40])
    };
    let profile = Profile::load("ref").unwrap();
    let mut images = Images::new();
    images.extend(CONTROL, &[0x40]);
    images.extend(mem(7), &[1, 2, 3]);
    images.extend(3, &[0x21]);
    let externs = [Extern { name: "w0".into(), slice: 7, addr: 0x1ff0, size: 0x10 }];
    assert_eq!(read_object(&write_object(&profile, &images, &externs), None, None).unwrap(), read(&o).unwrap());
}

#[test]
fn the_note_contains_full_interface_digests() {
    // the encoding digest the writer uses is that of encoding.json as shipped; the profile digest
    // that of the profile file (the embedded copies are the shipped files)
    for p in ["ref", "mini"] {
        let bytes = write_object(&Profile::load(p).unwrap(), &Images::new(), &[]);
        assert_eq!(&bytes[92..124], &digest(&format!("{SPEC}/isa-db/generated/encoding.json")), "{p}");
        assert_eq!(&bytes[124..156], &digest(&format!("{SPEC}/config/{p}.json")), "{p}");
    }
}

#[test]
fn an_unrelated_elf_is_rejected() {
    let mut b = b"\x7fELF".to_vec();
    b.extend([0u8; 100]);
    rejected(read_object(&b, None, None), "ELF");
    // the ELF test comes before the length test
    rejected(read_object(b"\x7fELF", None, None), "short ELF");
}

#[test]
fn a_bad_magic_or_a_short_file_is_rejected() {
    rejected(read_object(b"", None, None), "empty");
    rejected(read_object(b"OTSO", None, None), "magic alone");
    let good = Obj::new("ref").bytes();
    rejected(read_object(&good[..63], None, None), "63 bytes");
    for m in [*b"OTSW", *b"otso", *b"OTSM", *b"\0\0\0\0"] {
        rejected(read(&Obj { magic: m, ..Obj::new("ref") }), "magic");
    }
}

#[test]
fn only_elf32_little_endian_with_the_52_byte_header_is_supported() {
    for (major, minor, hsize) in
        [(1, 2, 52), (2, 1, 52), (1, 0, 52), (0, 0, 52), (1, 1, 60), (1, 1, 68), (1, 1, 0), (255, 255, 52)]
    {
        rejected(
            read(&Obj { major, minor, hsize, ..Obj::new("ref") }),
            &format!("version {major}.{minor}, header {hsize}"),
        );
    }
}

#[test]
fn the_note_abi_and_reserved_field_must_be_1_and_0() {
    for (revision, reserved) in [(2, 0), (3, 0), (0, 0), (1, 1), (1, 0x8000), (u32::MAX, u32::MAX)] {
        rejected(
            read(&Obj { revision, reserved, ..Obj::new("ref") }),
            &format!("revision {revision}, reserved {reserved:#x}"),
        );
    }
}

#[test]
fn the_profile_field_is_nul_padded_ascii_text() {
    let ok = |p: [u8; 16]| read(&Obj { profile: p, ..Obj::new("ref") });
    let mut b = name16("ref");
    b[5] = b'x'; // text after a NUL
    rejected(ok(b), "text after NUL");
    rejected(ok([0; 16]), "empty");
    let mut b = name16("ref");
    b[1] = 0xe9; // not ASCII
    rejected(ok(b), "non-ASCII");
    // the same bytes, after the NUL, are text too
    let mut b = name16("ref");
    b[15] = 1;
    rejected(ok(b), "a byte at the end");
    // an unknown profile, a name outside [a-z0-9_]+, a profile of 16 characters with no NUL
    for p in ["nosuch", "REF", "re f", "ref\n", "a/b"] {
        rejected(read(&Obj { profile: name16(p), ..Obj::new("ref") }), p);
    }
    rejected(read(&Obj { profile: *b"abcdefghijklmnop", ..Obj::new("ref") }), "16 characters");
}

#[test]
fn the_profile_option_must_equal_the_header() {
    let bytes = Obj::new("mini").bytes();
    assert!(read_object(&bytes, None, None).is_ok());
    assert!(read_object(&bytes, Some("mini"), None).is_ok());
    rejected(read_object(&bytes, Some("ref"), None), "--profile ref of a mini object");
    rejected(read_object(&bytes, Some(""), None), "--profile empty");
}

#[test]
fn the_digests_must_match_the_encoding_and_the_profile() {
    let mut enc = Obj::new("ref").enc;
    enc[11] ^= 1;
    rejected(read(&Obj { enc, ..Obj::new("ref") }), "encoding digest");
    let mut prof = Obj::new("ref").prof;
    prof[0] ^= 0x80;
    rejected(read(&Obj { prof, ..Obj::new("ref") }), "profile digest");
    // the digest of the other profile
    rejected(read(&Obj { prof: Obj::new("mini").prof, ..Obj::new("ref") }), "mini digest on ref");
    // a profile read from --config is held to its own bytes
    let dir = std::env::temp_dir().join(format!("ots-object-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = dir.join("ref22.json");
    let text = std::fs::read_to_string(format!("{SPEC}/config/ref.json")).unwrap();
    std::fs::write(&cfg, text.replace("\"mem_slices_per_hemisphere\": 44", "\"mem_slices_per_hemisphere\": 22"))
        .unwrap();
    let bytes = Obj::new("ref").bytes();
    rejected(read_object(&bytes, None, Some(cfg.to_str().unwrap())), "ref object against another configuration");
    let p = Profile::load_file("ref", &cfg).unwrap();
    let own = write_object(&p, &Images::new(), &[]);
    assert!(read_object(&own, None, Some(cfg.to_str().unwrap())).is_ok());
    rejected(read_object(&own, None, None), "an object of the --config file, read against the shipped one");
    // a config of another spec version
    let old = dir.join("old.json");
    std::fs::write(&old, text.replace("\"spec_version\": \"0.1\"", "\"spec_version\": \"0.2\"")).unwrap();
    rejected(read_object(&bytes, None, Some(old.to_str().unwrap())), "config of version 0.2");
    rejected(read_object(&bytes, None, Some(dir.join("none.json").to_str().unwrap())), "missing config");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_size_must_match_the_header() {
    let o = Obj::new("ref").image(mem(4), &[0x21, 0x1000_0010, 0]).image(CONTROL, &[0x30]);
    let good = o.bytes();
    assert!(read_object(&good, None, None).is_ok());
    // truncated by a byte, in the units, in the table, in the externs; extended by a byte
    for n in [good.len() - 1, 64 + 8, 64 + 16, 64 + 15, 64] {
        rejected(read_object(&good[..n], None, None), &format!("{n} bytes"));
    }
    let mut longer = good.clone();
    longer.push(0);
    rejected(read_object(&longer, None, None), "one byte more");
    rejected(
        read(&Obj { trailer: vec![0; 32], ..Obj::new("ref").image(mem(4), &[0x21]) }),
        "an extern's worth of bytes the header does not count",
    );
    // counts that do not match what follows
    rejected(read(&Obj { nimage: Some(3), ..Obj::new("ref").image(mem(4), &[0x21]) }), "nimage too large");
    rejected(read(&Obj { nimage: Some(0), ..Obj::new("ref").image(mem(4), &[0x21]) }), "nimage too small");
    rejected(read(&Obj { nimage: Some(u32::MAX), ..Obj::new("ref") }), "nimage 2^32-1");
    rejected(read(&Obj { nextern: Some(1), ..Obj::new("ref") }), "nextern too large");
    rejected(read(&Obj { nextern: Some(u32::MAX), ..Obj::new("ref") }), "nextern 2^32-1");
    rejected(
        read(&Obj { nextern: Some(0), externs: vec![ext(b"x", 4, 0, 1)], ..Obj::new("ref") }),
        "nextern too small",
    );
    // a unit count that points past the end
    let mut b = good.clone();
    b[156 + 8..156 + 12].copy_from_slice(&u32::MAX.to_le_bytes());
    rejected(read_object(&b, None, None), "unit count 2^32-1");
}

#[test]
fn the_size_is_checked_before_the_image_entries() {
    // a bad entry in a truncated file is a size error; both are ASM-003, so the order shows only in the
    // text of the message
    let e = read_object(&Obj { images: vec![(mem(4), 1, vec![1])], ..Obj::new("ref") }.bytes()[..70], None, None)
        .unwrap_err();
    assert!(e.msg.contains("directory"), "{e}");
}

#[test]
fn an_image_entry_has_a_zero_pad_a_queue_and_a_unit() {
    rejected(read(&Obj { images: vec![(mem(4), 1, vec![0x21])], ..Obj::new("ref") }), "non-zero pad");
    rejected(read(&Obj::new("ref").image(144, &[0x21])), "queue 0x90");
    rejected(read(&Obj::new("ref").image(0xfffe, &[0x21])), "queue 0xfffe");
    rejected(read(&Obj::new("ref").image(0x100, &[0x21])), "queue 0x100");
    rejected(read(&Obj::new("ref").image(mem(4), &[])), "no unit");
    // the shipped queues, all 144 and the control channel, are accepted (with one NOP unit)
    for q in (0..144u16).chain([CONTROL]) {
        assert!(read(&Obj::new("ref").image(q, &[1])).is_ok(), "queue {q:#x}");
    }
}

#[test]
fn the_images_are_in_ascending_queue_order_with_the_control_channel_last() {
    let a = || Obj::new("ref");
    assert!(read(&a().image(1, &[1]).image(2, &[1]).image(CONTROL, &[1])).is_ok());
    rejected(read(&a().image(2, &[1]).image(1, &[1])), "descending");
    rejected(read(&a().image(1, &[1]).image(1, &[1])), "duplicate");
    rejected(read(&a().image(CONTROL, &[1]).image(1, &[1])), "control first");
    rejected(read(&a().image(CONTROL, &[1]).image(CONTROL, &[1])), "control twice");
    rejected(read(&a().image(143, &[1]).image(142, &[1])), "143 then 142");
}

#[test]
fn the_text_of_a_queue_fits_iq_depth() {
    // ref: icu.iq_depth 8192 bytes? the cap is a quarter of it, in units; padding does not count
    for p in ["ref", "mini"] {
        let profile = Profile::load(p).unwrap();
        let cap = (profile.iq_depth() / 4) as usize;
        let o = |n: usize| Obj::new(p).image(4, &vec![1; n]);
        assert!(read(&o(cap)).is_ok(), "{p}: {cap} units");
        rejected(read(&o(cap + 1)), &format!("{p}: {} units", cap + 1));
        // padding units are never enqueued
        let mut us = vec![1; cap];
        us.extend([0xFFFF_FFFF; 5]);
        assert!(read(&Obj::new(p).image(4, &us)).is_ok(), "{p}: cap + padding");
        us.push(1);
        rejected(read(&Obj::new(p).image(4, &us)), &format!("{p}: cap + 1 and padding"));
        // the cap is per queue
        assert!(read(&Obj::new(p).image(4, &vec![1; cap]).image(5, &vec![1; cap])).is_ok(), "{p}: two queues");
    }
}

#[test]
fn a_parcel_must_fit_the_slots_of_its_queue() {
    // two instructions in one parcel (the first unit carries the bundle bit): on a VXM queue, which has no
    // slot rule for more than one instruction, that is QUE-001 at assembly and ASM-003 here
    let vxm = queue_id("VXM[0]").unwrap();
    let nop = 0x0000_fcb3u32; // a 32-bit instruction (`nop 0xfcb3`)
    let two = [nop | 0x8000_0000, nop];
    rejected(read(&Obj::new("ref").image(vxm, &two)), "two instructions in a VXM parcel");
    // one instruction in a parcel is fine, and so are two in two parcels
    assert!(read(&Obj::new("ref").image(vxm, &[nop])).is_ok());
    assert!(read(&Obj::new("ref").image(vxm, &[nop, nop])).is_ok());
    // two MEM instructions on different slots: read (R path) and write (W path)... the words are from tsp_as `insn`
    let (rd, wr) = (0x1000_0010_0000_0000u64, 0x1040_0010_0000_0000u64); // `read 0x10, E:0, 0x0` and `write 0x10, E:0, 0x0`
    let us = |a: u64, b: u64| [((a >> 32) as u32) | 0x8000_0000, a as u32, (b >> 32) as u32, b as u32];
    let q = mem(4);
    assert!(read(&Obj::new("ref").image(q, &us(rd, wr))).is_ok(), "a read and a write fit the two slots");
    rejected(read(&Obj::new("ref").image(q, &us(rd, rd))), "two reads on the one slot");
    // an undecodable instruction among several
    let bad = [0x6FFF_FFFFu32 | 0x8000_0000, nop];
    rejected(read(&Obj::new("ref").image(q, &bad)), "an undecodable unit in a parcel of two");
    // a single undecodable instruction is not the reader's business: `dis` annotates it
    assert!(read(&Obj::new("ref").image(q, &[0x6FFF_FFFF])).is_ok());
}

#[test]
fn an_extern_entry_has_zero_pads_a_text_name_and_the_extern_rules() {
    let e = |x: [u8; 32]| read(&Obj { externs: vec![x], ..Obj::new("ref") });
    assert!(e(ext(b"x", 4, 0x10, 1)).is_ok());
    let mut b = ext(b"x", 4, 0x10, 1);
    b[18] = 1; // the pad after the slice
    rejected(e(b), "pad 1");
    let mut b = ext(b"x", 4, 0x10, 1);
    b[28] = 1; // the pad after the size
    rejected(e(b), "pad 2");
    let mut b = ext(b"x", 4, 0x10, 1);
    b[5] = b'y'; // text after a NUL
    rejected(e(b), "name with text after the NUL");
    rejected(e(ext(b"", 4, 0x10, 1)), "empty name");
    rejected(e(ext(b"\xe9", 4, 0x10, 1)), "non-ASCII name");
    rejected(e(ext(b"1x", 4, 0x10, 1)), "a name that is not an identifier");
    rejected(e(ext(b"a b", 4, 0x10, 1)), "a name with a space");
    rejected(e(ext(b"a-b", 4, 0x10, 1)), "a name with a dash");
    assert!(e(ext(b"_a9", 4, 0x10, 1)).is_ok());
    // a name of 16 characters has no NUL and is longer than 15
    let mut b = ext(b"", 4, 0x10, 1);
    b[..16].copy_from_slice(b"abcdefghijklmnop");
    rejected(e(b), "16 characters");
    let mut b = ext(b"", 4, 0x10, 1);
    b[..15].copy_from_slice(b"abcdefghijklmno");
    assert!(e(b).is_ok(), "15 characters");
    rejected(e(ext(b"x", 88, 0x10, 1)), "slice 88");
    rejected(e(ext(b"x", 4, 0, 0)), "size 0");
    rejected(e(ext(b"x", 4, 0x1fff, 2)), "past the end");
    rejected(e(ext(b"x", 4, u32::MAX, 2)), "address overflow");
    assert!(e(ext(b"x", 4, 0x1fff, 1)).is_ok(), "the last word");
    let two = |a: [u8; 32], b: [u8; 32]| read(&Obj { externs: vec![a, b], ..Obj::new("ref") });
    rejected(two(ext(b"x", 4, 0, 1), ext(b"x", 5, 0, 1)), "a name twice");
    rejected(two(ext(b"x", 4, 0, 4), ext(b"y", 4, 3, 1)), "overlap");
    assert!(two(ext(b"x", 4, 0, 4), ext(b"y", 4, 4, 1)).is_ok(), "adjacent");
    assert!(two(ext(b"x", 4, 0, 4), ext(b"y", 5, 0, 4)).is_ok(), "other slice");
}

#[test]
fn an_extern_slice_must_exist_in_the_profile() {
    // With 22 MEM slices a hemisphere, slices 22..43 do not exist (slice mod 44 >= 22)
    let dir = std::env::temp_dir().join(format!("ots-object-f3-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = dir.join("ref22.json");
    let text = std::fs::read_to_string(format!("{SPEC}/config/ref.json")).unwrap();
    std::fs::write(&cfg, text.replace("\"mem_slices_per_hemisphere\": 44", "\"mem_slices_per_hemisphere\": 22"))
        .unwrap();
    let p = Profile::load_file("ref", &cfg).unwrap();
    let c = cfg.to_str().unwrap();
    let with = |sl: u16| {
        let x = Extern { name: "x".into(), slice: sl as u64, addr: 0, size: 1 };
        read_object(&write_object(&p, &Images::new(), &[x]), None, Some(c))
    };
    assert!(with(21).is_ok());
    rejected(with(22), "the first absent slice");
    rejected(with(43), "the last absent slice");
    assert!(with(44).is_ok(), "slice 44 is the first of the other hemisphere");
    rejected(with(66), "slice 66 (22 of the other hemisphere)");
    assert!(with(65).is_ok());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn externs_of_one_slice_may_touch_but_not_overlap() {
    let p = Profile::load("ref").unwrap();
    let x = |name: &str, slice: u64, addr: u64, size: u64| Extern { name: name.into(), slice, addr, size };
    let known = [x("a", 4, 0x10, 0x10)];
    for (y, why) in [
        (x("b", 4, 0x00, 0x10), "just below"),
        (x("b", 4, 0x20, 1), "just above"),
        (x("b", 5, 0x10, 0x10), "another slice"),
    ] {
        assert_eq!(ots_format::check_extern(&y, &p, &known), Ok(()), "{why}");
    }
    for (y, why) in [
        (x("b", 4, 0x00, 0x11), "its last word"),
        (x("b", 4, 0x1f, 4), "its first word"),
        (x("b", 4, 0x14, 2), "inside"),
        (x("b", 4, 0, 0x100), "around"),
    ] {
        let e = ots_format::check_extern(&y, &p, &known).unwrap_err();
        assert_eq!((e.code, e.msg.as_str()), (Code::Asm002, "extern b overlaps a in slice 4"), "{why}");
    }
}

#[test]
fn a_line_feed_in_an_extern_name_is_rejected() {
    // `tsp_as.check_extern` matches the whole name (`IDENT.fullmatch`), so not even a final "\n"
    rejected(read(&Obj { externs: vec![ext(b"x\n", 4, 0x10, 1)], ..Obj::new("ref") }), "a final line feed");
    // nor a second one, nor a line feed inside
    rejected(read(&Obj { externs: vec![ext(b"x\n\n", 4, 0x10, 1)], ..Obj::new("ref") }), "two line feeds");
    rejected(read(&Obj { externs: vec![ext(b"x\ny", 4, 0x10, 1)], ..Obj::new("ref") }), "a line feed inside");
    rejected(read(&Obj { externs: vec![ext(b"\n", 4, 0x10, 1)], ..Obj::new("ref") }), "a line feed alone");
}

#[test]
fn a_profile_name_of_more_than_16_characters_is_rejected_before_packing() {
    let dir = std::env::temp_dir().join(format!("ots-object-long-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = dir.join("long.json");
    std::fs::copy(format!("{SPEC}/config/ref.json"), &cfg).unwrap();
    assert!(Profile::load_file("profile_of_twenty_x", &cfg).is_err());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn round_trip_of_written_objects() {
    for p in ["ref", "mini"] {
        let profile = Profile::load(p).unwrap();
        let mut images = Images::new();
        images.extend(CONTROL, &[0x30, 0xFFFF_FFFF]);
        images.extend(7, &[0x21]);
        images.extend(0, &[1]);
        let externs = vec![
            Extern { name: "a".into(), slice: 0, addr: 0, size: 8 },
            Extern { name: "b_1".into(), slice: 1, addr: 8, size: 8 },
        ];
        let bytes = write_object(&profile, &images, &externs);
        assert_eq!(&bytes[..4], b"\x7fELF");
        let (name, got, got_ext) = read_object(&bytes, Some(p), None).unwrap();
        assert_eq!((name.as_str(), got, got_ext), (p, images, externs));
    }
}
