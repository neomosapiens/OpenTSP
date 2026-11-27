//! Program files (`assembly.md` §3), the static checks at bundle close and the lowering of a program
//! to the text of each queue. The expected results are those of
//! `opentsp-arch-spec/tools/tsp_as.py` (`tests/data/example.q` was printed by it; the error and
//! warning tables are checked against it by `opentsp-sw-toolchain/tests/diff_tsp_as.py`).
use ots_asm::program::{assemble, Program};
use ots_format::text::input_text;
use ots_isa::{queue_id, AsmError};

/// With `OTS_A4_DUMP=<dir>` every program of this file is also written to `<dir>/<hash>.ots`
/// (the checked-in `tests/programs/` of the differential `diff_tsp_as.py` were made this way).
fn dump(src: &str) {
    if let Ok(dir) = std::env::var("OTS_A4_DUMP") {
        let h = src.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(format!("{dir}/case_{h:016x}.ots"), src).unwrap();
    }
}

fn run(src: &str, config: Option<&str>) -> (Result<Program, AsmError>, Vec<String>) {
    if config.is_none() {
        dump(src);
    }
    let mut warns = Vec::new();
    let r = assemble(src.as_bytes(), "p.ots", config, &mut |w| warns.push(w.render())).map(|(_, p)| p);
    (r, warns)
}

/// `<where>: <code>` of an error.
fn err_of(src: &str) -> String {
    match run(src, None).0 {
        Ok(_) => "ok".to_string(),
        Err(e) => format!("{}: {}", e.where_.as_deref().unwrap_or("?"), e.code.as_str()),
    }
}

/// `<where>: <code> (warning)` of each warning of an accepted program.
fn warns_of(src: &str) -> Vec<String> {
    let (r, w) = run(src, None);
    assert!(r.is_ok(), "{:?}", r.err());
    w.iter()
        .map(|l| {
            l.splitn(3, ": ").take(2).collect::<Vec<_>>().join(": ").replace("(warning)", "").trim().to_string()
                + " (warning)"
        })
        .collect()
}

const HEAD: &str = ".arch opentsp-tsa\n.profile ref\n";

fn prog(externs: &str, body: &str) -> String {
    format!("{HEAD}{externs}.section text\n{body}")
}

fn bundles(body: &[(u32, &[&str])]) -> String {
    let mut s = String::new();
    for (c, lines) in body {
        s += &format!("bundle @cycle={c} {{\n");
        for l in *lines {
            s += &format!("  {l}\n");
        }
        s += "}\n";
    }
    prog("", &s)
}

// ---- the example of assembly.md §3 ----------------------------------------------------------

#[test]
fn the_section_3_example_lowers_to_the_q_lines_of_tsp_as() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/example.ots")).unwrap();
    let want = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/example.q")).unwrap();
    let (p, w) = run(&src, None);
    let p = p.unwrap();
    assert!(w.is_empty(), "{w:?}");
    assert_eq!(p.profile, "ref");
    assert_eq!(p.externs.iter().map(|x| x.name.as_str()).collect::<Vec<_>>(), ["x", "y"]);
    assert_eq!(p.records.len(), 4);
    assert_eq!(input_text(&p.images()), want);
}

// ---- file structure: ASM-001 / ASM-003 -------------------------------------------------------

#[test]
fn structure_errors_carry_their_line_and_code() {
    let ok = bundles(&[(0, &["VXM[0]: sync"])]);
    assert_eq!(err_of(&ok), "ok");
    let cases: &[(&str, &str)] = &[
        ("", "p.ots:1: OTS-E-ASM-001"),
        ("; only a comment\n\n", "p.ots:1: OTS-E-ASM-001"),
        (".arch opentsp-tsa\n.profile ref\n", "p.ots:1: OTS-E-ASM-001"),
        (".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=0 {\n", "p.ots:end: OTS-E-ASM-001"),
        ("\t.arch opentsp-tsa\n", "p.ots:1: OTS-E-ASM-001"),
        (".arch opentsp-tsa\n; caf\u{e9}\n", "p.ots:2: OTS-E-ASM-001"),
        (".arch opentsp-other\n", "p.ots:1: OTS-E-ASM-003"),
        (".profile ref\n", "p.ots:1: OTS-E-ASM-001"),
        (".arch opentsp-tsa\n.arch opentsp-tsa\n", "p.ots:2: OTS-E-ASM-001"),
        (".arch opentsp-tsa\n.profile nosuch\n", "p.ots:2: OTS-E-ASM-003"),
        (".arch opentsp-tsa\n.profile ref\n.profile ref\n", "p.ots:3: OTS-E-ASM-001"),
        (
            ".arch opentsp-tsa\n.profile ref\n.section text\n.extern x, slice=4, addr=0x0, size=0x1\n",
            "p.ots:4: OTS-E-ASM-001",
        ),
        (".arch opentsp-tsa\n.profile ref\n.section text\n.section text\n", "p.ots:4: OTS-E-ASM-001"),
        (".arch opentsp-tsa\n.profile ref\n.section data\n", "p.ots:3: OTS-E-ASM-001"),
        (".arch opentsp-tsa\n.profile ref\n.data\n", "p.ots:3: OTS-E-ASM-001"),
        (".arch opentsp-tsa\n.profile ref\nbundle @cycle=0 {\n", "p.ots:3: OTS-E-ASM-001"),
        (".arch opentsp-tsa\n.profile ref\n.section text\nfoo\n", "p.ots:4: OTS-E-ASM-001"),
        (
            ".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=0 {\nbundle @cycle=1 {\n",
            "p.ots:5: OTS-E-ASM-001",
        ),
        (
            ".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=0 {\n}\nbundle @cycle=0 {\n}\n",
            "p.ots:6: OTS-E-ASM-001",
        ),
        (".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=4294967296 {\n}\n", "p.ots:4: OTS-E-ASM-001"),
        (
            ".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=99999999999999999999999 {\n}\n",
            "p.ots:4: OTS-E-ASM-001",
        ),
        (
            ".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=0 {\n  VXM[0] sync\n}\n",
            "p.ots:5: OTS-E-ASM-001",
        ),
        (".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=0 {\n  : sync\n}\n", "p.ots:5: OTS-E-ASM-001"),
        (".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=0 {\n  VXM[0]:\n}\n", "p.ots:5: OTS-E-ASM-001"),
        // a CR is an ordinary character: it ends no line and breaks the match
        (".arch opentsp-tsa\r\n", "p.ots:1: OTS-E-ASM-001"),
    ];
    for (src, want) in cases {
        assert_eq!(err_of(src), *want, "{src:?}");
    }
    // the first error aborts; a tab in a comment is still an error
    assert_eq!(err_of(&ok.replace("sync", "sync ; a\ttab")), "p.ots:5: OTS-E-ASM-001");
    // spaces: any amount around ":" and the directive operands; the last line needs no line feed
    let loose = ".arch   opentsp-tsa  \n.profile    ref\n.section   text   ; c\nbundle   @cycle=3  {  \n   VXM[0]   :   sync   \n  }  ";
    assert_eq!(err_of(loose), "ok");
    // bundle @cycle=0 { with no slot is accepted and its cycle counts
    assert_eq!(err_of(&bundles(&[(5, &[]), (5, &[])])), "p.ots:6: OTS-E-ASM-001");
    assert_eq!(err_of(&bundles(&[(5, &[]), (6, &[])])), "ok");
}

#[test]
fn queue_names_and_instruction_texts() {
    let one = |l: &str| err_of(&bundles(&[(0, &[l])]));
    assert_eq!(one("cu: sync"), "p.ots:5: OTS-E-ASM-005");
    assert_eq!(one("MEM [0]: sync"), "p.ots:5: OTS-E-ASM-005");
    assert_eq!(one("MEM[88]: sync"), "p.ots:5: OTS-E-ASM-005");
    assert_eq!(one("MEM[007]: sync"), "ok");
    assert_eq!(one("MEM.E[0]: sync"), "ok");
    assert_eq!(one("CU: sync"), "ok");
    // unknown queue before the text is tried
    assert_eq!(one("nosuch: bogus"), "p.ots:5: OTS-E-ASM-005");
    assert_eq!(one("VXM[0]: bogus"), "p.ots:5: OTS-E-DEC-003");
    assert_eq!(one("VXM[0]: nop 0x0"), "p.ots:5: OTS-E-OPR-001");
    assert_eq!(one("VXM[0]: add.i8 E:8, E:0, E:40"), "p.ots:5: OTS-E-OPR-001");
}

// ---- externs: ASM-002 ------------------------------------------------------------------------

#[test]
fn extern_rules() {
    let with = |x: &str, l: &str| err_of(&prog(&format!("{x}\n"), &format!("bundle @cycle=0 {{\n  {l}\n}}\n")));
    let ok = ".extern x, slice=4, addr=0x10, size=0x2";
    assert_eq!(with(ok, "MEM[4]: read x+0x1, E:0, 0x0"), "ok");
    assert_eq!(with(ok, "MEM[4]: read x+0x2, E:0, 0x0"), "p.ots:6: OTS-E-ASM-002", "offset outside the extern");
    assert_eq!(with(ok, "MEM[4]: read nosuch, E:0, 0x0"), "p.ots:6: OTS-E-ASM-002", "undefined");
    assert_eq!(with(ok, "MEM[5]: read x, E:0, 0x0"), "p.ots:6: OTS-E-ASM-002", "another slice");
    assert_eq!(with(ok, "MEM[4]: ifetchm x+0x1"), "p.ots:6: OTS-E-ASM-002", "ifetchm reads two words");
    assert_eq!(with(ok, "MEM[4]: ifetchm x"), "ok");
    assert_eq!(with(&format!("{ok}\n{ok}"), "MEM[4]: sync"), "p.ots:4: OTS-E-ASM-002", "duplicate");
    assert_eq!(with(".extern longname_of_16_chr, slice=4, addr=0x0, size=0x1", "CU: sync"), "p.ots:3: OTS-E-ASM-002");
    assert_eq!(
        with(".extern x, slice=88, addr=0x0, size=0x1", "CU: sync"),
        "p.ots:3: OTS-E-ASM-002",
        "slice out of range"
    );
    assert_eq!(with(".extern x, slice=4, addr=0x0, size=0x0", "CU: sync"), "p.ots:3: OTS-E-ASM-002", "empty");
    assert_eq!(
        with(".extern x, slice=4, addr=0x1ffe, size=0x3", "CU: sync"),
        "p.ots:3: OTS-E-ASM-002",
        "beyond words_per_slice"
    );
    assert_eq!(with(".extern x, slice=4, addr=0x1ffe, size=0x2", "CU: sync"), "ok");
    assert_eq!(
        with(&format!("{ok}\n.extern y, slice=4, addr=0x11, size=0x4"), "CU: sync"),
        "p.ots:4: OTS-E-ASM-002",
        "overlap"
    );
    assert_eq!(
        with(&format!("{ok}\n.extern y, slice=5, addr=0x11, size=0x4"), "CU: sync"),
        "ok",
        "other slice: no overlap"
    );
    assert_eq!(with(&format!("{ok}\n.extern y, slice=4, addr=0x12, size=0x4"), "CU: sync"), "ok", "adjacent");
    // the text of a declaration
    assert_eq!(with(".extern x slice=4, addr=0x0, size=0x1", "CU: sync"), "p.ots:3: OTS-E-ASM-001");
    assert_eq!(with(".extern 1x, slice=4, addr=0x0, size=0x1", "CU: sync"), "p.ots:3: OTS-E-ASM-001");
    assert_eq!(with(".extern x , slice=4,addr=0x0 ,size=0x1", "CU: sync"), "ok");
    assert_eq!(with(".extern x, slice=4, addr=16, size=0x1", "CU: sync"), "p.ots:3: OTS-E-ASM-001", "addr is hex");
    assert_eq!(with(".extern", "CU: sync"), "p.ots:3: OTS-E-ASM-001");
    // a huge value does not wrap
    assert_eq!(with(".extern x, slice=4, addr=0xffffffffffffffffff, size=0x1", "CU: sync"), "p.ots:3: OTS-E-ASM-002");
    assert_eq!(with(".extern x, slice=4, addr=0x0, size=0xffffffffffffffffff", "CU: sync"), "p.ots:3: OTS-E-ASM-002");
}

#[test]
fn an_extern_in_an_absent_mem_slice_is_asm_002_and_its_queue_is_que_002() {
    // A profile with 22 MEM slices a hemisphere (slice mod 44 >= 22 does not exist)
    let ref_json =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../opentsp-arch-spec/config/ref.json"))
            .unwrap();
    // set the key whatever its current value or spacing (the specification files may change)
    let key = "\"mem_slices_per_hemisphere\"";
    let at = ref_json.find(key).expect("ref.json has geometry.mem_slices_per_hemisphere");
    let colon = at + key.len() + ref_json[at + key.len()..].find(':').unwrap() + 1;
    let end = colon + ref_json[colon..].find([',', '\n', '}']).unwrap();
    let mem22 = format!("{} 22{}", &ref_json[..colon], &ref_json[end..]);
    // the process id and the test name make the path this test's own
    let path = std::env::temp_dir().join(format!("ots-asm-a4-mem22-{}.json", std::process::id()));
    std::fs::write(&path, mem22).unwrap();
    let cfg = path.to_str().unwrap();
    let code = |src: &str| match run(src, Some(cfg)).0 {
        Ok(_) => "ok".to_string(),
        Err(e) => format!("{}: {}", e.where_.unwrap(), e.code.as_str()),
    };
    assert_eq!(code(&prog(".extern x, slice=21, addr=0x0, size=0x1\n", "")), "ok");
    assert_eq!(code(&prog(".extern x, slice=22, addr=0x0, size=0x1\n", "")), "p.ots:3: OTS-E-ASM-002");
    assert_eq!(code(&prog(".extern x, slice=66, addr=0x0, size=0x1\n", "")), "p.ots:3: OTS-E-ASM-002");
    assert_eq!(code(&prog(".extern x, slice=44, addr=0x0, size=0x1\n", "")), "ok");
    let b = |q: &str| code(&bundles(&[(0, &[&format!("{q}: sync")])]));
    assert_eq!(b("MEM[21]"), "ok");
    assert_eq!(b("MEM[22]"), "p.ots:5: OTS-E-QUE-002");
    assert_eq!(b("MEM[43]"), "p.ots:5: OTS-E-QUE-002");
    assert_eq!(b("MEM[44]"), "ok");
    assert_eq!(b("MEM.E[22]"), "p.ots:5: OTS-E-QUE-002");
    assert_eq!(b("VXM[0]"), "ok");
    assert_eq!(b("CU"), "ok");
    std::fs::remove_file(path).unwrap();
}

// ---- bundle checks: QUE-001, QUE-002, MEM-001 --------------------------------------------------

#[test]
fn bundle_checks_run_in_the_models_order() {
    let b = |ls: &[&str]| err_of(&bundles(&[(0, ls)]));
    // slot rule, reported at the line of the group's LAST instruction
    assert_eq!(b(&["VXM[0]: sync", "VXM[0]: sync"]), "p.ots:6: OTS-E-QUE-001");
    assert_eq!(b(&["VXM[0]: sync", "VXM[0]: sync", "VXM[0]: sync"]), "p.ots:7: OTS-E-QUE-001");
    assert_eq!(
        b(&["MEM[4]: read 0x0, E:0, 0x0", "MEM[4]: read 0x1, E:1, 0x0"]),
        "p.ots:6: OTS-E-QUE-001",
        "two R of one thread"
    );
    assert_eq!(
        b(&["MEM[4]: read 0x0, E:0, 0x0", "MEM[4]: read 0x1, E:1, 0x1", "MEM[4]: ifetch E:2"]),
        "ok",
        "the §3 example of one legal slot set"
    );
    assert_eq!(
        b(&["MEM[4]: sync", "MEM[4]: read 0x0, E:0, 0x0"]),
        "p.ots:6: OTS-E-QUE-001",
        "a whole-queue instruction with another"
    );
    assert_eq!(
        b(&["MEM[4]: ifetchm 0x0", "MEM[4]: read 0x0, E:0, 0x0"]),
        "p.ots:6: OTS-E-QUE-001",
        "ifetchm takes the R slot of both threads"
    );
    assert_eq!(b(&["MEM[4]: ifetchm 0x0", "MEM[4]: write 0x1000, E:0, 0x0"]), "ok");
    assert_eq!(
        b(&["VXM[0]: nop 0x2", "VXM[0]: add.i8 E:8, E:0, E:4"]),
        "p.ots:6: OTS-E-QUE-001",
        "nop with another is a violation"
    );
    assert_eq!(b(&["VXM[0]: nop 0x2"]), "ok", "a lone nop is a timing instruction");
    // placement
    assert_eq!(b(&["VXM[0]: read 0x0, E:0, 0x0"]), "p.ots:5: OTS-E-QUE-002");
    assert_eq!(b(&["MEM[4]: add.i8 E:8, E:0, E:4"]), "p.ots:5: OTS-E-QUE-002");
    assert_eq!(b(&["CU: ifetch E:2"]), "p.ots:5: OTS-E-QUE-002");
    assert_eq!(b(&["CU: read 0x0, E:0, 0x0"]), "p.ots:5: OTS-E-QUE-002");
    // QUE-001 on a queue before QUE-002 on the same queue, per queue in order of first line
    assert_eq!(b(&["VXM[0]: read 0x0, E:0, 0x0", "VXM[0]: sync"]), "p.ots:6: OTS-E-QUE-001");
    assert_eq!(
        b(&["VXM[1]: read 0x0, E:0, 0x0", "VXM[0]: sync", "VXM[0]: sync"]),
        "p.ots:5: OTS-E-QUE-002",
        "VXM[1] is first"
    );
    assert_eq!(
        b(&["VXM[0]: sync", "VXM[1]: read 0x0, E:0, 0x0", "VXM[0]: sync"]),
        "p.ots:7: OTS-E-QUE-001",
        "VXM[0] is first"
    );
    // bank rule
    assert_eq!(
        b(&["MEM[4]: read 0x0, E:0, 0x0", "MEM[4]: write 0x1, E:1, 0x0"]),
        "p.ots:5: OTS-E-MEM-001",
        "same bank: at the first clashing instruction"
    );
    assert_eq!(b(&["MEM[4]: read 0x0, E:0, 0x0", "MEM[4]: write 0x1000, E:1, 0x0"]), "ok", "banks differ (bit 12)");
    assert_eq!(b(&["MEM[4]: read 0x0, E:0, 0x0", "MEM[4]: write 0x1, E:1, 0x1"]), "ok", "other thread");
    assert_eq!(
        b(&["MEM[4]: write 0x1, E:1, 0x0", "MEM[4]: gather E:0, E:2, 0x0, 0x0"]),
        "p.ots:5: OTS-E-MEM-001",
        "indirect clashes with the other path"
    );
    // a word outside the slice addresses no bank (mini: 256 words, bank bit 7)
    let mini = ".arch opentsp-tsa\n.profile mini\n.section text\nbundle @cycle=0 {\n  MEM[4]: read 0x180, E:0, 0x0\n  MEM[4]: write 0x100, E:1, 0x0\n}\n";
    assert_eq!(err_of(mini), "ok");
    assert_eq!(err_of(&mini.replace("write 0x100", "write 0x180")), "ok", "both outside");
    assert_eq!(
        err_of(&mini.replace("write 0x100", "write 0x80")),
        "ok",
        "banks 1 and 1: the read is outside, no clash"
    );
    assert_eq!(
        err_of(&mini.replace("read 0x180", "read 0x81").replace("write 0x100", "write 0x80")),
        "p.ots:5: OTS-E-MEM-001"
    );
    // the checks are per bundle: the same instructions in two cycles are fine
    assert_eq!(err_of(&bundles(&[(0, &["VXM[0]: sync"]), (1, &["VXM[0]: sync"])])), "ok");
    // an error in a later bundle names its line
    assert_eq!(
        err_of(&bundles(&[(0, &["VXM[0]: sync"]), (1, &["VXM[0]: sync", "VXM[0]: sync"])])),
        "p.ots:9: OTS-E-QUE-001"
    );
}

// ---- warnings ---------------------------------------------------------------------------------

#[test]
fn an_address_beyond_the_slice_warns_at_its_line_and_the_program_stays_valid() {
    // ref: 8192 words, a 13-bit field: only the second word of an ifetchm can be beyond
    let w = warns_of(&bundles(&[(0, &["MEM[4]: ifetchm 0x1fff"]), (1, &["MEM[4]: ifetchm 0x1ffe"])]));
    assert_eq!(w, ["p.ots:5: OTS-E-OPR-001 (warning)"]);
    // the mini profile has 256 words
    let mini =
        ".arch opentsp-tsa\n.profile mini\n.section text\nbundle @cycle=0 {\n  MEM[4]: read 0x100, E:0, 0x0\n}\n";
    let (r, w) = run(mini, None);
    assert!(r.is_ok());
    assert_eq!(w.len(), 1);
    assert!(w[0].starts_with("p.ots:5: OTS-E-OPR-001 (warning): "), "{w:?}");
}

const R: &str = "MEM[4]: read 0x0, E:0, 0x0";
const W: &str = "MEM[4]: write 0x1000, E:1, 0x0";

#[test]
fn agt_the_us12340300_pair_is_silent() {
    // a read + write pair opens one configuration a path; the records alternate R, W
    let b = bundles(&[
        (0, &[R, W]),
        (1, &["MEM[4]: countdown 0x1"]),
        (2, &["MEM[4]: countdown 0x1"]),
        (3, &["MEM[4]: step 0x1"]),
        (4, &["MEM[4]: step 0x1"]),
        (5, &["MEM[4]: iterations 0x2"]),
        (6, &["MEM[4]: iterations 0x2"]),
    ]);
    assert_eq!(warns_of(&b), Vec::<String>::new());
    // a single reference with its configuration, four levels, a gap ends it
    let one = bundles(&[
        (0, &[R]),
        (1, &["MEM[4]: countdown 0x1"]),
        (2, &["MEM[4]: step 0x1"]),
        (3, &["MEM[4]: iterations 0x2"]),
    ]);
    assert_eq!(warns_of(&one), Vec::<String>::new());
}

#[test]
fn agt_warnings() {
    // a configuration with no reference before it
    assert_eq!(warns_of(&bundles(&[(0, &["MEM[4]: countdown 0x1"])])), ["p.ots:5: OTS-E-OPR-001 (warning)"]);
    // a reference in the same cycle does not count
    assert_eq!(warns_of(&bundles(&[(0, &[R, "MEM[4]: countdown 0x1"])])), ["p.ots:6: OTS-E-OPR-001 (warning)"]);
    // a late record: a cycle's gap ends the configuration
    assert_eq!(warns_of(&bundles(&[(0, &[R]), (2, &["MEM[4]: countdown 0x1"])])), ["p.ots:8: OTS-E-OPR-001 (warning)"]);
    // a cycle of the queue without a configuration record ends it
    assert_eq!(
        warns_of(&bundles(&[(0, &[R]), (1, &["MEM[4]: sync"]), (2, &["MEM[4]: countdown 0x1"])])),
        ["p.ots:11: OTS-E-OPR-001 (warning)"]
    );
    // a configuration on another queue is stray
    assert_eq!(warns_of(&bundles(&[(0, &[R]), (1, &["MEM[5]: countdown 0x1"])])), ["p.ots:8: OTS-E-OPR-001 (warning)"]);
    // a fifth level: countdown at or before the last field starts the next level
    let mut v: Vec<(u32, Vec<&str>)> = vec![(0, vec![R])];
    for c in 1..=5 {
        v.push((c, vec!["MEM[4]: countdown 0x1"]));
    }
    let slices: Vec<(u32, &[&str])> = v.iter().map(|(c, l)| (*c, l.as_slice())).collect();
    // levels 0..3 take four countdowns; the fifth is beyond; the open configuration is then dropped
    assert_eq!(warns_of(&bundles(&slices)), ["p.ots:20: OTS-E-OPR-001 (warning)"]);
    // a reference on a path whose configuration is open
    assert_eq!(
        warns_of(&bundles(&[(0, &[R]), (1, &["MEM[4]: countdown 0x1", "MEM[4]: read 0x2, E:2, 0x0"])])),
        ["p.ots:9: OTS-E-QUE-001 (warning)"]
    );
    // two references on one path never open a configuration (the second is the slot violation anyway)
    // a reference beside an ifetch opens its path (a pair of paths, one reference each)
    assert_eq!(
        warns_of(&bundles(&[(0, &[R, "MEM[4]: ifetch E:2"]), (1, &["MEM[4]: countdown 0x1"])])),
        Vec::<String>::new()
    );
    // a gather opens its path too
    assert_eq!(
        warns_of(&bundles(&[(0, &["MEM[4]: gather E:0, E:2, 0x0, 0x0"]), (1, &["MEM[4]: countdown 0x1"])])),
        Vec::<String>::new()
    );
    // an ifetchm takes R of both threads: it never opens a path
    assert_eq!(
        warns_of(&bundles(&[(0, &["MEM[4]: ifetchm 0x0"]), (1, &["MEM[4]: countdown 0x1"])])),
        ["p.ots:8: OTS-E-OPR-001 (warning)"]
    );
    // and a reference on a configuring path warns once for each slot the instruction takes: ifetchm: R of both threads
    assert_eq!(
        warns_of(&bundles(&[(0, &[R]), (1, &["MEM[4]: countdown 0x1", "MEM[4]: ifetchm 0x0"])])),
        ["p.ots:9: OTS-E-QUE-001 (warning)", "p.ots:9: OTS-E-QUE-001 (warning)"]
    );
}

#[test]
fn an_image_longer_than_iq_depth_warns_after_the_file() {
    // icu.iq_depth = 1280 bytes: 320 units. 330 one-unit parcels on VXM[0] (cycles 0, 2, 4, ...) make 659 units
    let mut body = String::new();
    for i in 0..330 {
        body += &format!("bundle @cycle={} {{\n  VXM[0]: sync\n}}\n", 2 * i);
    }
    let (r, w) = run(&prog("", &body), None);
    let p = r.unwrap();
    assert_eq!(w.len(), 1, "{w:?}");
    assert!(w[0].starts_with("p.ots: OTS-E-ASM-003 (warning): "), "{}", w[0]);
    assert_eq!(p.images().get(queue_id("VXM[0]").unwrap()).unwrap().len(), 659);
    // 320 units exactly: no warning
    let mut body = String::new();
    for i in 0..160 {
        body += &format!("bundle @cycle={} {{\n  VXM[0]: sync\n}}\n", 2 * i);
    }
    let (r, w) = run(&prog("", &body), None);
    assert!(r.is_ok());
    assert!(w.is_empty(), "{w:?}");
}

// ---- lowering and the order of a parcel -----------------------------------------------

#[test]
fn a_parcel_keeps_source_order() {
    // MEM[4], one cycle: a W (word 0x10400000...) before an R (0x10000000...): source order W, R
    let b = bundles(&[(3, &["MEM[4]: write 0x1000, E:1, 0x0", "MEM[4]: read 0x0, E:0, 0x0", "MEM[4]: ifetch E:2"])]);
    let p = run(&b, None).0.unwrap();
    let q = queue_id("MEM[4]").unwrap();
    let src = p.images().get(q).unwrap().clone();
    // NOP 3, then the three instructions: bundle bit on the first two
    assert_eq!(src[0], 3);
    assert_eq!(src.len(), 1 + 2 + 2 + 1);
    assert_eq!(src[1], 0x8000_0000 | 0x1040_1000, "the write comes first: {src:x?}");
    // not in ascending word order: the read (a smaller word) follows the write
    assert!(src[3] & 0x7FFF_FFFF < src[1] & 0x7FFF_FFFF, "{src:x?}");
    // cycles and queues are ordered: two queues, interleaved lines
    let b = bundles(&[(0, &["VXM[1]: sync", "VXM[0]: sync"]), (1, &["VXM[0]: sync"])]);
    let p = run(&b, None).0.unwrap();
    assert_eq!(p.images().iter().map(|(q, _)| q).collect::<Vec<_>>(), [88, 89]);
}

#[test]
fn nop_gaps_and_the_control_channel_lower_like_tsp_as() {
    // cycle 70000: one gap of 65535 and one of 4465 on the first parcel of the queue
    let b = bundles(&[(70000, &["VXM[0]: sync"]), (70002, &["VXM[0]: sync"])]);
    let p = run(&b, None).0.unwrap();
    let us = p.images().get(88).unwrap().clone();
    assert_eq!(us, vec![0xFFFF, 70000 - 65535, 0x0040_0000, 1, 0x0040_0000]);
    // CU last; a lone nop makes no parcel; the others keep their cycles
    let b = bundles(&[(0, &["CU: sync", "MEM[0]: sync"]), (1, &["VXM[0]: nop 0x4"]), (2, &["VXM[0]: sync"])]);
    let p = run(&b, None).0.unwrap();
    let text = input_text(&p.images());
    assert_eq!(text, "Q 0 00400000\nQ 58 00000002 00400000\nQ ffff 00400000\n");
}

// ---- the host instructions ----------------------------------------------------

#[test]
fn the_host_program_lowers_to_the_q_lines_of_tsp_as() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/host.ots")).unwrap();
    let want = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/host.q")).unwrap();
    let (p, w) = run(&src, None);
    assert!(w.is_empty(), "{w:?}");
    assert_eq!(input_text(&p.unwrap().images()), want);
}

#[test]
fn deskew_recv_is_placed_on_the_rx_queues_like_receive() {
    let b = |lines: &[&str]| err_of(&bundles(&[(3, lines)]));
    assert_eq!(b(&["C2C.W.RX[0]: deskew_recv 0x0, 0x0, E:3", "C2C.E.RX[1]: deskew_recv 0x0, 0x1, W:4"]), "ok");
    assert_eq!(b(&["C2C.W.TX[0]: deskew_recv 0x0, 0x0, E:3"]), "p.ots:5: OTS-E-QUE-002");
    assert_eq!(b(&["VXM[0]: deskew_recv 0x0, 0x0, E:3"]), "p.ots:5: OTS-E-QUE-002");
}

#[test]
fn host_instructions_follow_the_placement_and_slot_rules() {
    let b = |lines: &[&str]| err_of(&bundles(&[(3, lines)]));
    // PciReceive on the RX queues, PciSend on the TX queues, HostWait and HostNotify on any data queue
    assert_eq!(b(&["C2C.W.RX[0]: pci_receive W:24", "C2C.E.RX[1]: pci_receive E:31"]), "ok");
    assert_eq!(b(&["C2C.E.TX[1]: pci_send E:0"]), "ok");
    assert_eq!(b(&["MXM[3].ACC: host_wait 0x3f"]), "ok");
    assert_eq!(b(&["SXM.E.SEL: host_notify 0xf"]), "ok");
    assert_eq!(b(&["MEM[87]: host_notify 0x1"]), "ok");
    assert_eq!(b(&["C2C.W.TX[0]: pci_receive W:24"]), "p.ots:5: OTS-E-QUE-002");
    assert_eq!(b(&["C2C.W.RX[1]: pci_send W:0"]), "p.ots:5: OTS-E-QUE-002");
    assert_eq!(b(&["MEM[4]: pci_send W:0"]), "p.ots:5: OTS-E-QUE-002");
    assert_eq!(b(&["CU: host_wait 0x1"]), "p.ots:5: OTS-E-QUE-002");
    // one instruction alone in a queue's slot: a second one in the same bundle does not fit
    assert_eq!(b(&["VXM[0]: host_wait 0x1", "VXM[0]: host_notify 0x1"]), "p.ots:6: OTS-E-QUE-001");
    assert_eq!(b(&["C2C.W.RX[0]: pci_receive W:24", "C2C.W.RX[0]: host_wait 0x1"]), "p.ots:6: OTS-E-QUE-001");
}
