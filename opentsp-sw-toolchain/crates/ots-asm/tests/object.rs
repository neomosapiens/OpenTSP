//! Objects through the command line: `asm -o` writes the bytes `tsp_as.py asm -o` writes,
//! and `dis` and `objinput` read an object. The expected files in `tests/data/` were produced by
//! `opentsp-arch-spec/tools/tsp_as.py` (`asm -o`, `dis`, `objinput`) from the programs next to them;
//! `parcel_src.*` from `tsp_as.py` with its one sort line made stable (source order),
//! `parcel_word.*` kept from `tsp_as.py` (ascending word order: valid input for the readers). The rejections of the reader are
//! in `ots-format/tests/object.rs`; the differential run (`tests/diff_tsp_as.py`) holds both to
//! mutations of every corpus.
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

fn data(name: &str) -> PathBuf {
    PathBuf::from(format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR")))
}

fn bytes(name: &str) -> Vec<u8> {
    fs::read(data(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn run(args: &[&str]) -> Output {
    let mut waited = 0;
    loop {
        match Command::new(env!("CARGO_BIN_EXE_ots-asm")).args(args).output() {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && waited < 100 => {
                waited += 1;
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            r => return r.unwrap(),
        }
    }
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
        let d = std::env::temp_dir().join(format!("ots-asm-object-{tag}-{}", std::process::id()));
        fs::create_dir_all(&d).unwrap();
        Tmp(d)
    }
    fn path(&self, n: &str) -> String {
        self.0.join(n).to_str().unwrap().to_string()
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).ok();
    }
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[test]
fn asm_dash_o_of_the_section_3_example_is_the_object_tsp_as_writes() {
    for (src, obj) in [("example.ots", "example.elf"), ("example_mini.ots", "example_mini.elf")] {
        let t = Tmp::new("example");
        let out = t.path("out.elf");
        let (rc, stdout, stderr) = text(&run(&["asm", s(&data(src)), "-o", &out]));
        assert_eq!((rc, stdout.as_str(), stderr.as_str()), (0, "", ""), "{src}");
        assert_eq!(fs::read(&out).unwrap(), bytes(obj), "{src}");
    }
}

#[test]
fn the_example_object_is_elf32_with_the_documented_program() {
    let bytes = bytes("example.elf");
    assert_eq!(&bytes[..7], b"\x7fELF\x01\x01\x01");
    let (profile, images, externs) = ots_format::read_object(&bytes, None, None).unwrap();
    assert_eq!(profile, "ref");
    assert_eq!(images.len(), 4);
    assert_eq!(externs.len(), 2);
}

#[test]
fn input_and_dash_o_together_print_the_q_lines_and_write_the_object() {
    let t = Tmp::new("both");
    let out = t.path("out.elf");
    let (rc, stdout, _) = text(&run(&["asm", s(&data("example.ots")), "--input", "-o", &out]));
    assert_eq!(rc, 0);
    assert_eq!(stdout.as_bytes(), bytes("example.q"));
    assert_eq!(fs::read(&out).unwrap(), bytes("example.elf"));
}

#[test]
fn the_parcel_order_in_the_object_is_source_order() {
    let t = Tmp::new("parcel");
    let a = t.path("src.elf");
    assert_eq!(text(&run(&["asm", s(&data("parcel.ots")), "-o", &a])).0, 0);
    assert_eq!(fs::read(&a).unwrap(), bytes("parcel_src.elf"));
    // `parcel_word.elf` is the same program with the parcel in ascending word order (what `tsp_as.py` made
    // before the source-order rule): an object a reader still accepts, no longer one the assembler writes
    assert_ne!(bytes("parcel_src.elf"), bytes("parcel_word.elf"));
}

#[test]
fn a_rejected_program_writes_no_object() {
    let t = Tmp::new("reject");
    let src = t.path("p.ots");
    fs::write(&src, ".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=1 {\n  MEM[4]: bogus\n}\n").unwrap();
    let out = t.path("out.elf");
    let (rc, stdout, stderr) = text(&run(&["asm", &src, "-o", &out]));
    assert_eq!((rc, stdout.as_str()), (1, ""));
    assert!(stderr.starts_with(&format!("{src}:5: OTS-E-")), "{stderr}");
    assert!(!std::path::Path::new(&out).exists());
}

#[test]
fn warnings_are_printed_and_the_object_is_written() {
    let t = Tmp::new("warn");
    let src = t.path("p.ots");
    // an address past the end of the slice is a warning, not an error
    fs::write(
        &src,
        ".arch opentsp-tsa\n.profile mini\n.section text\nbundle @cycle=1 {\n  MEM[4]: read 0x1000, E:0, 0x0\n}\n",
    )
    .unwrap();
    let out = t.path("out.elf");
    let (rc, _, stderr) = text(&run(&["asm", &src, "-o", &out]));
    assert_eq!(rc, 0);
    assert!(stderr.contains("(warning)"), "{stderr}");
    assert!(fs::metadata(&out).is_ok());
}

#[test]
fn an_empty_program_is_a_valid_elf() {
    let t = Tmp::new("empty");
    let src = t.path("p.ots");
    fs::write(&src, ".arch opentsp-tsa\n.profile ref\n.section text\n").unwrap();
    let out = t.path("out.elf");
    assert_eq!(text(&run(&["asm", &src, "-o", &out])).0, 0);
    assert!(ots_format::read_object(&fs::read(&out).unwrap(), None, None).is_ok());
    let (rc, stdout, _) = text(&run(&["objinput", &out]));
    assert_eq!((rc, stdout.as_str()), (0, ""));
    let (rc, listing, _) = text(&run(&["dis", &out]));
    assert_eq!((rc, listing.as_str()), (0, ".arch opentsp-tsa\n.profile ref\n.section text\n"));
}

#[test]
fn dis_of_an_object_is_the_listing_with_its_externs() {
    for name in ["example", "example_mini", "parcel_src", "parcel_word"] {
        let (rc, out, err) = text(&run(&["dis", s(&data(&format!("{name}.elf")))]));
        assert_eq!((rc, err.as_str()), (0, ""), "{name}");
        assert_eq!(out.as_bytes(), bytes(&format!("{name}.dis")), "{name}");
    }
    // `--profile` that equals the header is accepted, another one is ASM-003
    let ex = data("example_mini.elf");
    assert_eq!(text(&run(&["dis", s(&ex), "--profile", "mini"])).0, 0);
    let (rc, out, err) = text(&run(&["dis", s(&ex), "--profile", "ref"]));
    assert_eq!((rc, out.as_str()), (1, ""));
    assert!(err.starts_with("dis: OTS-E-ASM-003: "), "{err}");
}

#[test]
fn objinput_of_an_object_is_the_q_lines() {
    for name in ["example", "example_mini", "parcel_src", "parcel_word"] {
        let (rc, out, err) = text(&run(&["objinput", s(&data(&format!("{name}.elf")))]));
        assert_eq!((rc, err.as_str()), (0, ""), "{name}");
        assert_eq!(out.as_bytes(), bytes(&format!("{name}.objinput")), "{name}");
    }
    assert_eq!(bytes("example.objinput"), bytes("example.q"));
}

#[test]
fn objinput_of_text_prints_the_text_lines_and_does_not_load_a_profile() {
    let t = Tmp::new("objtext");
    let q = t.path("a.txt");
    fs::write(&q, "Q 4 00000021 10000010 00000000\n\nQ 4 00000000\nQ ffff 00000030\nQ 5 1\n").unwrap();
    // tsp_as never builds the ISA for `objinput` of text, so a profile that does not exist is no error
    for argv in [vec!["objinput", &q[..]], vec!["objinput", &q[..], "--profile", "nosuch"]] {
        let (rc, out, err) = text(&run(&argv));
        assert_eq!((rc, err.as_str()), (0, ""));
        assert_eq!(out, "Q 4 00000021 10000010 00000000 00000000\nQ 5 00000001\nQ ffff 00000030\n");
    }
    // `dis` of the same text does need it
    let (rc, _, err) = text(&run(&["dis", &q, "--profile", "nosuch"]));
    assert_eq!(rc, 1);
    assert!(err.starts_with("dis: OTS-E-ASM-003: "), "{err}");
}

#[test]
fn a_text_file_with_non_ascii_or_a_bad_line_is_asm_003_for_both_commands() {
    let t = Tmp::new("badtext");
    let q = t.path("a.txt");
    for body in [&b"Q 4 \xe9\n"[..], b"Q 4 0\nR 4 0\n", b"Q 90 0\n", b"Q 4 123456789\n"] {
        fs::write(&q, body).unwrap();
        for cmd in ["dis", "objinput"] {
            let (rc, out, err) = text(&run(&[cmd, &q]));
            assert_eq!((rc, out.as_str()), (1, ""), "{cmd}");
            assert!(err.contains(": OTS-E-ASM-003: "), "{cmd}: {err}");
        }
    }
}

#[test]
fn a_bad_object_is_asm_003_with_the_command_as_its_location() {
    let t = Tmp::new("badobj");
    let good = bytes("example.elf");
    let bad = t.path("bad.elf");
    let mut cases: Vec<(&str, Vec<u8>)> = Vec::new();
    cases.push(("truncated", good[..good.len() - 1].to_vec()));
    let mut m = good.clone();
    m[40] ^= 1;
    cases.push(("encoding digest", m));
    let mut m = good.clone();
    m[32] = 1;
    cases.push(("revision", m));
    let mut m = b"\x7fELF".to_vec();
    m.extend([0; 60]);
    cases.push(("ELF", m));
    cases.push(("binary but not an object", vec![0xff; 10]));
    for (what, b) in cases {
        fs::write(&bad, &b).unwrap();
        for cmd in ["dis", "objinput"] {
            let (rc, out, err) = text(&run(&[cmd, &bad]));
            assert_eq!((rc, out.as_str()), (1, ""), "{what} {cmd}");
            assert!(err.starts_with(&format!("{cmd}: OTS-E-ASM-003: ")), "{what} {cmd}: {err}");
        }
    }
}

#[test]
fn a_missing_file_is_an_error_for_the_three_commands() {
    for cmd in ["dis", "objinput"] {
        let (rc, _, err) = text(&run(&[cmd, "/nonexistent/x.elf"]));
        assert_eq!(rc, 1, "{cmd}");
        assert!(err.starts_with(&format!("{cmd}: ")), "{err}");
    }
    // no operand: usage
    assert_eq!(text(&run(&["dis"])).0, 2);
    assert_eq!(text(&run(&["objinput"])).0, 2);
    assert_eq!(text(&run(&["asm", "x.ots", "-o"])).0, 2, "-o without a value");
}

#[test]
fn an_object_of_a_config_file_is_read_with_that_config() {
    // (the 22-slice configuration is not one the model accepts: `--no-validate` lets the commands read it, see config_validate.rs)
    let t = Tmp::new("config");
    let cfg = t.path("ref22.json");
    let ref_json =
        fs::read_to_string(format!("{}/../../../opentsp-arch-spec/config/ref.json", env!("CARGO_MANIFEST_DIR")))
            .unwrap();
    fs::write(&cfg, ref_json.replace("\"mem_slices_per_hemisphere\": 44", "\"mem_slices_per_hemisphere\": 22"))
        .unwrap();
    let src = t.path("p.ots");
    fs::write(&src, ".arch opentsp-tsa\n.profile ref\n.extern a, slice=4, addr=0x0, size=0x2\n.section text\nbundle @cycle=1 {\n  MEM[4]: read a, E:0, 0x0\n}\n").unwrap();
    let out = t.path("o.elf");
    assert_eq!(text(&run(&["asm", &src, "-o", &out, "--no-validate", "--config", &cfg])).0, 0);
    // the digest of the object is that of the config file
    let (rc, listing, err) = text(&run(&["dis", &out, "--no-validate", "--config", &cfg]));
    assert_eq!((rc, err.as_str()), (0, ""));
    assert!(listing.contains(".extern a, slice=4, addr=0x0, size=0x2\n"), "{listing}");
    assert!(listing.contains("MEM[4]: read 0x0, E:0, 0x0"), "{listing}");
    let (rc, _, err) = text(&run(&["dis", &out]));
    assert_eq!(rc, 1, "read against the shipped ref.json");
    assert!(err.starts_with("dis: OTS-E-ASM-003: "), "{err}");
    let (rc, _, _) = text(&run(&["objinput", &out, "--no-validate", "--config", &cfg]));
    assert_eq!(rc, 0);
}

#[test]
fn dis_of_an_object_with_an_annotation_is_exit_1() {
    // a single instruction on a queue it does not belong to: QUE-002 at the listing
    let t = Tmp::new("annot");
    let src = t.path("p.ots");
    fs::write(
        &src,
        ".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=1 {\n  MEM[4]: read 0x0, E:0, 0x0\n}\n",
    )
    .unwrap();
    let out = t.path("o.elf");
    assert_eq!(text(&run(&["asm", &src, "-o", &out])).0, 0);
    let bytes = fs::read(&out).unwrap();
    let (_, images, externs) = ots_format::read_object(&bytes, None, None).unwrap();
    let vxm = ots_isa::queue_id("VXM[0]").unwrap();
    let mut displaced = ots_format::text::Images::new();
    displaced.insert(vxm, images.iter().next().unwrap().1.to_vec());
    let profile = ots_format::load_profile("ref", None).unwrap();
    fs::write(&out, ots_format::write_object(&profile, &displaced, &externs)).unwrap();
    let (rc, listing, _) = text(&run(&["dis", &out]));
    assert_eq!(rc, 1);
    assert!(listing.contains("; OTS-E-QUE-002"), "{listing}");
}

#[test]
fn objinput_with_a_missing_weight_package_is_refused() {
    // a package that cannot be read is an error, never silently ignored
    let (rc, out, _) = text(&run(&["objinput", s(&data("example.elf")), "--weights", "/nonexistent.safetensors"]));
    assert_ne!(rc, 0);
    assert_eq!(out, "");
}
