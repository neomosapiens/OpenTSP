//! The disassembler (`assembly.md` §4 "Disassembly"): the listing grammar, the
//! annotations of the model's diagnostics, the exit status. The expected results are those of the
//! checked-in corpora `opentsp-arch-spec/test/isa_db/corpus/<profile>/{text,groups}.txt` (printed by
//! the model through `tsp_as.py`) and of `tsp_as.py dis` itself (`opentsp-sw-toolchain/tests/diff_tsp_as.py`
//! holds the two to the golden vectors and to mutations).
use ots_asm::disassemble;
use ots_format::{text::Images, Extern};
use ots_isa::{queue_name, Isa, BUNDLE_BIT, CONTROL};
use ots_profile::Profile;
use std::process::{Command, Output};

const PROFILES: [&str; 2] = ["ref", "mini"];

fn corpus(profile: &str, name: &str) -> Vec<Vec<String>> {
    let path = format!("{}/../../../opentsp-arch-spec/test/isa_db/corpus/{profile}/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect()
}

fn isa(profile: &str) -> Isa {
    Isa::new(&Profile::load(profile).unwrap())
}

fn hex(s: &str) -> u64 {
    u64::from_str_radix(s, 16).unwrap()
}

/// The listing lines of one instruction each (the bundle lines and the header are dropped).
fn body(listing: &str) -> Vec<String> {
    listing.lines().filter(|l| l.starts_with("  ") && !l.starts_with("  .")).map(str::to_string).collect()
}

/// `(code, text)` of a listing line `  <queue>: <text>[   ; <code>[: ...]]`.
fn split(line: &str) -> (String, String) {
    let b = line.split_once(": ").unwrap().1;
    match b.split_once("   ; ") {
        Some((t, n)) => (n.split(':').next().unwrap().to_string(), t.to_string()),
        None => ("-".to_string(), b.to_string()),
    }
}

/// `(cycle, queue)` order of a listing: by cycle, then queues in order of their text, control last.
fn order_key(k: usize, q: u16) -> (usize, bool, u16) {
    (k, q == CONTROL, q)
}

#[test]
fn text_txt_listing_gives_every_corpus_code_and_text() {
    let mut n = 0;
    for profile in PROFILES {
        let isa = isa(profile);
        let mut images = Images::new();
        let mut expect: Vec<(usize, u16, String, String)> = Vec::new();
        for row in corpus(profile, "text.txt") {
            let q = hex(&row[0]) as u16;
            if row[4].starts_with("nop ") {
                continue; // a NOP is a timing prefix, with no listing line
            }
            let k = expect.iter().filter(|e| e.1 == q).count();
            expect.push((k, q, row[3].clone(), row[4].clone()));
            let mut units = vec![hex(&row[1]) as u32];
            if row[2] != "-" {
                units.push(hex(&row[2]) as u32);
            }
            images.extend(q, &units);
        }
        expect.sort_by_key(|e| order_key(e.0, e.1));
        let (listing, bad) = disassemble(&isa, &images, &[]);
        let lines = body(&listing);
        assert_eq!(lines.len(), expect.len(), "{profile}: lines");
        for ((_, q, code, text), line) in expect.iter().zip(&lines) {
            let (got_code, got_text) = split(line);
            assert!(line.starts_with(&format!("  {}: ", queue_name(*q).unwrap())), "{profile}: {line}");
            assert_eq!(&got_code, code, "{profile}: {line} (corpus {text})");
            if !text.is_empty() {
                assert_eq!(&got_text, text, "{profile}: {line}");
            }
            n += 1;
        }
        assert_eq!(bad, expect.iter().filter(|e| e.2 != "-").count(), "{profile}: annotated count");
    }
    assert!(n > 2000, "{n}");
}

#[test]
fn groups_txt_listing_gives_every_group_code() {
    let mut n = 0;
    for profile in PROFILES {
        let isa = isa(profile);
        let mut images = Images::new();
        let mut expect: Vec<(usize, u16, String, String)> = Vec::new();
        let mut groups: std::collections::BTreeMap<u32, Vec<(u16, u64, String, String)>> = Default::default();
        for r in corpus(profile, "groups.txt") {
            groups.entry(r[0].parse().unwrap()).or_default().push((
                hex(&r[1]) as u16,
                hex(&r[2]),
                r[3].clone(),
                r[4].clone(),
            ));
        }
        let mut nq = std::collections::HashMap::new();
        for rs in groups.values() {
            let q = rs[0].0;
            let k = *nq.entry(q).and_modify(|k| *k += 1).or_insert(0usize);
            let mut units = Vec::new();
            for (j, (_, w0, code, text)) in rs.iter().enumerate() {
                let mut u = ots_format::text::word_units(*w0);
                if j < rs.len() - 1 {
                    u[0] |= BUNDLE_BIT;
                }
                units.extend(u);
                expect.push((k, q, code.clone(), text.clone()));
            }
            images.extend(q, &units);
        }
        expect.sort_by_key(|e| order_key(e.0, e.1));
        let (listing, _) = disassemble(&isa, &images, &[]);
        let lines = body(&listing);
        assert_eq!(lines.len(), expect.len(), "{profile}: lines");
        for ((_, _, code, text), line) in expect.iter().zip(&lines) {
            let (got_code, got_text) = split(line);
            assert_eq!(&got_code, code, "{profile}: {line} (corpus {text})");
            assert_eq!(&got_text, text, "{profile}: {line}");
            n += 1;
        }
    }
    assert!(n > 1500, "{n}");
}

fn dis_text(images: &Images, profile: &str) -> (String, usize) {
    disassemble(&isa(profile), images, &[])
}

#[test]
fn header_grammar() {
    let (l, bad) = dis_text(&Images::new(), "ref");
    assert_eq!((l.as_str(), bad), (".arch opentsp-tsa\n.profile ref\n.section text\n", 0));
    let (l, _) = dis_text(&Images::new(), "mini");
    assert_eq!(l, ".arch opentsp-tsa\n.profile mini\n.section text\n");
}

#[test]
fn externs_are_listed_between_the_profile_and_the_section() {
    let x = |name: &str, slice, addr, size| Extern { name: name.to_string(), slice, addr, size };
    let (l, bad) = disassemble(&isa("ref"), &Images::new(), &[x("w0", 4, 0x10, 0x1), x("w1", 5, 0, 0x100)]);
    assert_eq!(
        l,
        ".arch opentsp-tsa\n.profile ref\n.extern w0, slice=4, addr=0x10, size=0x1\n.extern w1, slice=5, addr=0x0, size=0x100\n.section text\n"
    );
    assert_eq!(bad, 0);
}

/// Q lines of a text file of the harness form, to images.
fn images_of(text: &str) -> Images {
    ots_format::text::parse_images(text, "t").unwrap()
}

#[test]
fn bundles_one_per_cycle_queues_in_order_of_first_appearance() {
    // the program of tests/data/example.ots lowered by `tsp_as.py asm --input`
    let q = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/example.q")).unwrap();
    let (l, bad) = dis_text(&images_of(&q), "ref");
    assert_eq!(bad, 0);
    assert_eq!(
        l,
        ".arch opentsp-tsa\n.profile ref\n.section text\n\
bundle @cycle=33 {\n  MEM[4]: read 0x10, E:0, 0x0\n  MEM[5]: read 0x0, E:4, 0x0\n}\n\
bundle @cycle=40 {\n  VXM[0]: add.i8 E:8, E:0, E:4\n}\n\
bundle @cycle=48 {\n  MEM[44]: write 0x9, E:8, 0x0\n}\n"
    );
}

#[test]
fn a_clean_listing_assembles_back_to_the_same_text_in_source_order() {
    use ots_asm::program::assemble;
    let src = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/example.ots")).unwrap();
    let (_, prog) = assemble(&src, "example.ots", None, &mut |_| {}).unwrap();
    let images = prog.images();
    let (listing, bad) = dis_text(&images, "ref");
    assert_eq!(bad, 0);
    let (_, back) = assemble(listing.as_bytes(), "listing.ots", None, &mut |w| panic!("{}", w.render())).unwrap();
    assert_eq!(back.images(), images);
}

#[test]
fn an_unsorted_parcel_round_trips_in_source_order() {
    use ots_asm::program::assemble;
    // two instructions of one parcel, the larger word first: the listing keeps the parcel's order
    let src = ".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=0 {\n  MEM[4]: write 0x20, E:0, 0x0\n  MEM[4]: read 0x10, E:0, 0x1\n}\n";
    let (_, prog) = assemble(src.as_bytes(), "p.ots", None, &mut |_| {}).unwrap();
    let images = prog.images();
    let (listing, bad) = dis_text(&images, "ref");
    assert_eq!(bad, 0, "{listing}");
    let lines = body(&listing);
    assert!(lines[0].contains("write 0x20") && lines[1].contains("read 0x10"), "{listing}");
    let (_, back) = assemble(listing.as_bytes(), "l.ots", None, &mut |_| {}).unwrap();
    assert_eq!(back.images(), images);
}

#[test]
fn annotations_in_the_models_order() {
    // an undecodable 32-bit word (tag 6): DEC-003, annotated in place of the text
    let mut im = Images::new();
    im.extend(4, &[0x6fff_ffff]);
    let (l, bad) = dis_text(&im, "ref");
    assert_eq!(bad, 1);
    assert!(l.contains("  MEM[4]: illegal32 0x6fffffff   ; OTS-E-DEC-003\n"), "{l}");
    // a decodable instruction on a queue that cannot issue it: the placement code
    let isa = isa("ref");
    let w = ots_asm::parse_insn(&isa, "add.i8 E:8, E:0, E:4", &Default::default(), None).unwrap().1;
    let mut im = Images::new();
    im.extend(4, &ots_format::text::word_units(w));
    let (l, bad) = disassemble(&isa, &im, &[]);
    assert_eq!(bad, 1);
    assert!(l.contains("  MEM[4]: add.i8 E:8, E:0, E:4   ; OTS-E-QUE-002\n"), "{l}");
    // a NOP-only queue lists nothing
    let (l, bad) = dis_text(&images_of("Q 4 00000030\n"), "ref");
    assert_eq!((bad, l.as_str()), (0, ".arch opentsp-tsa\n.profile ref\n.section text\n"));
}

#[test]
fn a_group_that_does_not_fit_is_annotated_on_each_instruction_over_its_own_text() {
    // two reads in one parcel on MEM[4] (one R slot per thread): QUE-001 on both
    let mut im = Images::new();
    // use the model's own words: two `read` instructions of thread 0 in one parcel
    let isa = isa("ref");
    let r = |t: &str| ots_asm::parse_insn(&isa, t, &Default::default(), None).unwrap().1;
    let mut u0 = ots_format::text::word_units(r("read 0x10, E:0, 0x0"));
    u0[0] |= BUNDLE_BIT;
    let u1 = ots_format::text::word_units(r("read 0x20, E:4, 0x0"));
    im.extend(4, &u0);
    im.extend(4, &u1);
    let (l, bad) = disassemble(&isa, &im, &[]);
    assert_eq!(bad, 2, "{l}");
    let lines = body(&l);
    assert_eq!(lines.len(), 2);
    assert!(lines[0].starts_with("  MEM[4]: read 0x10, E:0, 0x0   ; OTS-E-QUE-001"), "{l}");
    assert!(lines[1].starts_with("  MEM[4]: read 0x20, E:4, 0x0   ; OTS-E-QUE-001"), "{l}");
}

// ---- the command line ----

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

fn tmp(name: &str, bytes: &[u8]) -> String {
    let dir = std::env::temp_dir().join(format!("ots-asm-dis-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(name);
    std::fs::write(&p, bytes).unwrap();
    p.to_string_lossy().into_owned()
}

fn text(o: &Output) -> (i32, String, String) {
    (
        o.status.code().unwrap(),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

#[test]
fn dis_of_a_clean_text_exits_0_and_defaults_to_the_ref_profile() {
    let f = format!("{}/tests/data/example.q", env!("CARGO_MANIFEST_DIR"));
    let (rc, out, err) = text(&run(&["dis", &f]));
    assert_eq!((rc, err.as_str()), (0, ""));
    assert!(out.starts_with(".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=33 {\n"), "{out}");
    // a blank file is a text image with no instruction
    let b = tmp("blank.txt", b"  \n\n");
    assert_eq!(
        text(&run(&["dis", &b])),
        (0, ".arch opentsp-tsa\n.profile ref\n.section text\n".to_string(), String::new())
    );
    let (rc, out, _) = text(&run(&["dis", "--profile", "mini", &b]));
    assert_eq!(rc, 0);
    assert!(out.contains(".profile mini\n"));
}

#[test]
fn dis_exits_1_when_anything_is_annotated() {
    let f = tmp("bad.txt", b"Q 4 6fffffff\n");
    let (rc, out, err) = text(&run(&["dis", &f]));
    assert_eq!(rc, 1);
    assert!(out.contains("  MEM[4]: illegal32 0x6fffffff   ; OTS-E-DEC-003\n"), "{out}");
    assert_eq!(err, "");
}

#[test]
fn dis_input_errors_are_asm_003_on_stderr() {
    for (name, data, want) in [
        ("l.txt", &b"Q 4 zz\n"[..], "dis: OTS-E-ASM-003: l.txt:1: "),
        ("q.txt", b"Q 90 00000001\n", "dis: OTS-E-ASM-003: q.txt:1: "),
        ("n.txt", b"Q 4 00000000\nQ\n", "dis: OTS-E-ASM-003: n.txt:2: "),
        ("u.txt", b"Q 4 0000000\xc3\xa9\n", "dis: OTS-E-ASM-003: "),
    ] {
        let f = tmp(name, data);
        let (rc, out, err) = text(&run(&["dis", &f]));
        assert_eq!((rc, out.as_str()), (1, ""), "{name}");
        let want = want.replace(name, &f);
        assert!(err.starts_with(&want), "{name}: {err}");
    }
    let (rc, _, err) = text(&run(&["dis", "--profile", "nosuch", &tmp("e.txt", b"")]));
    assert_eq!(rc, 1);
    assert!(err.starts_with("dis: OTS-E-ASM-003: "), "{err}");
}

#[test]
fn dis_without_a_file_is_usage() {
    let (rc, _, err) = text(&run(&["dis"]));
    assert_eq!(rc, 2);
    assert!(err.contains("usage"), "{err}");
}
