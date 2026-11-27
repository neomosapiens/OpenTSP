//! `ots-asm patch`: the patched object equals the object assembled from the program with
//! the new operand written literally (the differential), the table and the values are checked, and a patch that would
//! put a read and a write of one thread into one bank is refused with `OTS-E-MEM-001` and writes nothing.
use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
};

fn ots_asm(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ots-asm")).args(args).output().unwrap()
}

fn text(o: &Output) -> (i32, String, String) {
    (
        o.status.code().unwrap(),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

struct Tmp(PathBuf);

impl Tmp {
    fn new(tag: &str) -> Tmp {
        let d = std::env::temp_dir().join(format!("ots-asm-patch-{tag}-{}", std::process::id()));
        fs::create_dir_all(&d).unwrap();
        Tmp(d)
    }
    fn path(&self, n: &str) -> String {
        self.0.join(n).to_str().unwrap().to_string()
    }
    fn write(&self, n: &str, s: &str) -> String {
        let p = self.path(n);
        fs::write(&p, s).unwrap();
        p
    }
}

/// A program with a read and a write of MEM[4] in one bundle (banks 0 and 1), a permute, and a VXM add.
fn source(read: u32, write: u32, map: u32) -> String {
    format!(
        ".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=2 {{\n  MEM[4]: read {read:#x}, E:0, 0x0\n  MEM[4]: write {write:#x}, E:1, 0x0\n}}\nbundle @cycle=9 {{\n  VXM[0]: add.i8 E:8, E:0, E:4\n}}\nbundle @cycle=12 {{\n  SXM.E.PRM: permute E:1, {map:#x}, E:2\n}}\n"
    )
}

fn assemble(t: &Tmp, name: &str, src: &str) -> Vec<u8> {
    let s = t.write(&format!("{name}.ots"), src);
    let o = t.path(&format!("{name}.elf"));
    assert_eq!(text(&ots_asm(&["asm", &s, "-o", &o])), (0, String::new(), String::new()));
    fs::read(o).unwrap()
}

/// Units: 0 = gap NOP, 1-2 read, 3-4 write on queue 4; permute on queue 0x8a after a NOP gap: units 1-2.
const TABLE: &str = r#"{"format": 1, "sites": [
  {"name": "rd", "queue": 4, "unit": 1, "op": "read", "field": "addr", "kind": "sram_addr", "min": 0, "max": 4095},
  {"name": "wr", "queue": 4, "unit": 3, "op": "write", "field": "addr", "kind": "sram_addr", "min": 4096, "max": 8191},
  {"name": "pm", "queue": 138, "unit": 1, "op": "permute", "field": "map", "kind": "map_id", "min": 0, "max": 15}
]}"#;

fn patch(t: &Tmp, table: &str, values: &str, out: &str) -> (i32, String, String) {
    let (table, values) = (t.write("table.json", table), t.write("values.json", values));
    text(&ots_asm(&["patch", &t.path("p.elf"), "--table", &table, "--values", &values, "-o", out]))
}

#[test]
fn the_patched_object_is_the_object_of_the_patched_program() {
    let t = Tmp::new("diff");
    fs::write(t.path("p.elf"), assemble(&t, "base", &source(0x10, 0x1010, 3))).unwrap();
    let out = t.path("out.elf");
    // two address sites and a two-unit field that straddles its units (the permute map is bits 33..30)
    for (rd, wr, pm) in [(0x20, 0x1234, 0xa), (0, 8191, 15), (4095, 4096, 0), (0x10, 0x1010, 3)] {
        let (rc, stdout, stderr) = patch(&t, TABLE, &format!("{{\"rd\": {rd}, \"wr\": {wr}, \"pm\": {pm}}}"), &out);
        assert_eq!((rc, stderr.as_str()), (0, ""), "{rd} {wr} {pm}");
        assert_eq!(stdout.lines().count(), 3, "{stdout}");
        assert_eq!(fs::read(&out).unwrap(), assemble(&t, "want", &source(rd, wr, pm)), "{rd:#x} {wr:#x} {pm:#x}");
    }
    // a site without a value keeps the operand
    let (rc, stdout, _) = patch(&t, TABLE, "{\"pm\": 9}", &out);
    assert_eq!(rc, 0);
    assert_eq!(stdout, "pm = 0x9 (was 0x3; queue 138, unit 1)\n");
    assert_eq!(fs::read(&out).unwrap(), assemble(&t, "want", &source(0x10, 0x1010, 9)));
}

#[test]
fn a_patch_into_a_bank_clash_is_refused_with_mem_001_and_writes_nothing() {
    let t = Tmp::new("bank");
    fs::write(t.path("p.elf"), assemble(&t, "base", &source(0x10, 0x1010, 3))).unwrap();
    let wide = TABLE.replace("\"max\": 4095", "\"max\": 8191");
    let out = t.path("clash.elf");
    // the read moves to the write's bank (bank bit 12 of the address)
    let (rc, stdout, stderr) = patch(&t, &wide, "{\"rd\": 4128}", &out);
    assert_eq!((rc, stdout.as_str()), (1, ""));
    assert!(stderr.starts_with("patch: OTS-E-MEM-001: the patch of site rd makes an invalid parcel"), "{stderr}");
    assert!(!std::path::Path::new(&out).exists());
    // moving the write into the read's bank is refused the same way
    let wide_w = TABLE.replace("\"min\": 4096", "\"min\": 0");
    let (rc, _, stderr) = patch(&t, &wide_w, "{\"wr\": 64}", &out);
    assert_eq!(rc, 1);
    assert!(stderr.starts_with("patch: OTS-E-MEM-001: "), "{stderr}");
    // and both together, ending in different banks, are accepted
    let both = TABLE.replace("\"max\": 4095", "\"max\": 8191").replace("\"min\": 4096", "\"min\": 0");
    assert_eq!(patch(&t, &both, "{\"rd\": 4128, \"wr\": 32}", &out).0, 0);
}

#[test]
fn a_patched_word_that_no_longer_decodes_in_a_two_instruction_parcel_is_dec_003() {
    // : the read's address moves beyond sram.addr_bits (13): the word is a DEC-003 word, and the check that is named is the
    // decode, not the slot rule of the parcel it breaks
    let t = Tmp::new("dec");
    fs::write(t.path("p.elf"), assemble(&t, "base", &source(0x10, 0x1010, 3))).unwrap();
    let wide = TABLE.replace("\"max\": 4095", "\"max\": 16383");
    let out = t.path("dec.elf");
    let (rc, stdout, stderr) = patch(&t, &wide, "{\"rd\": 8192}", &out);
    assert_eq!((rc, stdout.as_str()), (1, ""), "{stderr}");
    assert!(
        stderr.starts_with(
            "patch: OTS-E-DEC-003: the patch of site rd makes an invalid parcel: unit 1 of MEM[4] does not decode"
        ),
        "{stderr}"
    );
    assert!(!std::path::Path::new(&out).exists());
}

#[test]
fn values_and_sites_are_checked_before_anything_is_written() {
    let t = Tmp::new("checks");
    fs::write(t.path("p.elf"), assemble(&t, "base", &source(0x10, 0x1010, 3))).unwrap();
    let out = t.path("never.elf");
    let cases: Vec<(String, String, &str)> = vec![
        (TABLE.into(), "{\"rd\": 4096}".into(), "OTS-E-OPR-001: site rd: value 4096 outside [0, 4095]"),
        (TABLE.into(), "{\"pm\": 16}".into(), "OTS-E-OPR-001: site pm: value 16 outside [0, 15]"),
        (TABLE.into(), "{\"nope\": 1}".into(), "OTS-E-ASM-001: patch values: \"nope\" is not a site of the table"),
        (TABLE.into(), "{\"rd\": -1}".into(), "OTS-E-ASM-001: patch value rd: expected a non-negative integer"),
        (TABLE.into(), "[1]".into(), "OTS-E-ASM-001: patch values: expected a JSON object"),
        (TABLE.into(), "{".into(), "OTS-E-ASM-001: patch values: not JSON"),
        (
            TABLE.replace("\"unit\": 1,", "\"unit\": 2,"),
            "{\"rd\": 1}".into(),
            "OTS-E-ASM-001: site rd: unit 2 of queue 4 is not the first unit of an instruction",
        ),
        (
            TABLE.replace("\"read\"", "\"write\""),
            "{\"rd\": 1}".into(),
            "OTS-E-ASM-001: site rd: unit 1 of queue 4 holds read, not write",
        ),
        (
            TABLE.replace("\"addr\"", "\"bogus\""),
            "{\"rd\": 1}".into(),
            "OTS-E-ASM-001: site rd: read has no operand \"bogus\"",
        ),
        (
            TABLE.replace("sram_addr\", \"min\": 0", "map_id\", \"min\": 0"),
            "{\"rd\": 1}".into(),
            "OTS-E-ASM-001: site rd: operand addr of read is of kind sram_addr, not map_id",
        ),
        (
            TABLE.replace("\"unit\": 1,", "\"unit\": 99,"),
            "{\"rd\": 1}".into(),
            "OTS-E-ASM-001: site rd: unit 99 of queue 4 is not the first unit of an instruction",
        ),
        (
            TABLE.replace("\"queue\": 4,", "\"queue\": 5,"),
            "{\"rd\": 1}".into(),
            "OTS-E-ASM-001: site rd: the object has no text for queue 5",
        ),
        (TABLE.replace("\"format\": 1", "\"format\": 2"), "{}".into(), "OTS-E-ASM-001: patch table: format 2"),
        (
            TABLE.replace("\"name\": \"wr\"", "\"name\": \"rd\""),
            "{}".into(),
            "OTS-E-ASM-001: patch table site 1: site rd is declared twice",
        ),
        (
            TABLE.replace("\"min\": 0, \"max\": 4095", "\"min\": 9, \"max\": 4"),
            "{}".into(),
            "OTS-E-ASM-001: patch table site rd: min 9 is above max 4",
        ),
        (
            TABLE.replace("\"min\": 0, \"max\": 4095", "\"min\": 0, \"max\": 4095, \"stride\": 4, \"span\": 2"),
            "{\"rd\": 5}".into(),
            "",
        ),
        (
            TABLE.replace("\"min\": 0, \"max\": 4095", "\"min\": 0, \"max\": 4095, \"stride\": 4, \"span\": 2"),
            "{\"rd\": 2}".into(),
            "OTS-E-OPR-001: site rd: value 0x2 is not on the grid of stride 4 from 0x0",
        ),
    ];
    for (i, (table, values, want)) in cases.iter().enumerate() {
        let (rc, stdout, stderr) = patch(&t, table, values, &out);
        if want.is_empty() {
            assert_eq!((rc, stderr.as_str()), (0, ""), "case {i}");
            fs::remove_file(&out).unwrap();
            continue;
        }
        assert_eq!((rc, stdout.as_str()), (1, ""), "case {i}: {stderr}");
        assert!(stderr.starts_with(&format!("patch: {want}")), "case {i}: {stderr}");
        assert!(!std::path::Path::new(&out).exists(), "case {i}: an object was written");
    }
}

#[test]
fn usage_and_unreadable_files() {
    let t = Tmp::new("usage");
    fs::write(t.path("p.elf"), assemble(&t, "base", &source(0x10, 0x1010, 3))).unwrap();
    let table = t.write("t.json", TABLE);
    let values = t.write("v.json", "{}");
    let obj = t.path("p.elf");
    let out = t.path("o.elf");
    for args in [
        vec!["patch"],
        vec!["patch", &obj, "--table", &table, "--values", &values],
        vec!["patch", &obj, "--table", &table, "-o", &out],
        vec!["patch", "--table", &table, "--values", &values, "-o", &out],
    ] {
        let (rc, stdout, stderr) = text(&ots_asm(&args));
        assert_eq!((rc, stdout.as_str()), (2, ""), "{args:?}");
        assert!(stderr.starts_with("usage:"), "{stderr}");
    }
    let (rc, _, stderr) =
        text(&ots_asm(&["patch", "/no/such.elf", "--table", &table, "--values", &values, "-o", &out]));
    assert_eq!(rc, 1);
    assert!(stderr.starts_with("patch: /no/such.elf: "), "{stderr}");
    // an object of another profile is read as every command reads it
    let (rc, _, stderr) =
        text(&ots_asm(&["patch", &obj, "--profile", "mini", "--table", &table, "--values", &values, "-o", &out]));
    assert_eq!(rc, 1);
    assert!(stderr.contains("OTS-E-ASM-003"), "{stderr}");
}
