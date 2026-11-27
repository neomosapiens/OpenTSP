//! Lexical rules (`assembly.md` §1), operand checks and the reject precedence (§5) of one
//! instruction; the cases are those of `check_asm_examples.py` §4 and the directed cases of `tsp_as.py`.
use ots_asm::parse_insn;
use ots_format::Extern;
use ots_isa::{render, Isa};
use ots_profile::Profile;
use std::collections::HashMap;

fn isa(profile: &str) -> Isa {
    Isa::new(&Profile::load(profile).unwrap())
}

fn parse(isa: &Isa, t: &str) -> Result<String, String> {
    parse_insn(isa, t, &HashMap::new(), None).map(|(i, _)| render(&i)).map_err(|e| e.code.as_str().to_string())
}

fn code(isa: &Isa, t: &str) -> Option<String> {
    parse(isa, t).err()
}

#[test]
fn lexical_rules() {
    let isa = isa("ref");
    let cases: &[(&str, Option<&str>)] = &[
        ("add.i8 E:8,E:0,E:4", Some("add.i8 E:8, E:0, E:4")), // zero spaces after ','
        ("add.i8   E:8 ,  E:0 , E:4", Some("add.i8 E:8, E:0, E:4")), // spaces around ','
        ("read 0x0010, E:0, 0x0", Some("read 0x10, E:0, 0x0")), // leading zeros
        ("read 0X10, E:0, 0x0", None),                        // 0X is not hex
        ("read 0x1F, E:0, 0x0", Some("read 0x1f, E:0, 0x0")), // upper-case digits
        ("add.i8 E:007, E:0, E:4", Some("add.i8 E:7, E:0, E:4")), // decimal leading zeros
        ("acc.i32 E:0, replace, 01", Some("acc.i32 E:0, replace, 1")),
        ("add.i8\tE:8, E:0, E:4", None), // tab
        ("ADD.i8 E:8, E:0, E:4", None),  // upper-case mnemonic
        ("add.I8 E:8, E:0, E:4", None),
        ("add.i8E:8, E:0, E:4", None), // no space after the mnemonic
        ("read -0x1, E:0, 0x0", None),
        ("read 0x, E:0, 0x0", None),
        ("sync ", Some("sync")),
        (" sync", Some("sync")),
        ("  nop 0x1  ", Some("nop 0x1")),
    ];
    for (t, want) in cases {
        assert_eq!(parse(&isa, t).ok().as_deref(), *want, "{t:?}");
    }
}

#[test]
fn tab_and_non_ascii_are_asm_001_whatever_else_is_wrong() {
    let isa = isa("ref");
    for t in ["sync\t", "\tsync", "nop\t0x1", "caf\u{e9}", "sync \u{e9}", "bogus\t\u{e9}"] {
        assert_eq!(code(&isa, t).as_deref(), Some("OTS-E-ASM-001"), "{t:?}");
    }
    // a CR is an ordinary character, not a space
    assert_eq!(code(&isa, "sync\r").as_deref(), Some("OTS-E-DEC-003"));
    assert_eq!(code(&isa, "sync\n").as_deref(), Some("OTS-E-DEC-003"));
}

#[test]
fn error_classes_and_their_precedence() {
    let isa = isa("ref");
    // no template matches: DEC-003; an empty text too
    for t in ["", "bogus", "nop", "nop 1", "read 0x0, E:0", "read 0x0, E:0, 0x0,", "add.i8 E:8, E:0"] {
        assert_eq!(code(&isa, t).as_deref(), Some("OTS-E-DEC-003"), "{t:?}");
    }
    // a template matches, an operand is out of range: OPR-001
    assert_eq!(code(&isa, "nop 0x0").as_deref(), Some("OTS-E-OPR-001"));
    assert_eq!(code(&isa, "nop 0x10000").as_deref(), Some("OTS-E-OPR-001"));
    assert_eq!(code(&isa, "add.i8 E:32, E:0, E:4").as_deref(), Some("OTS-E-OPR-001")); // stream above 31
    assert_eq!(code(&isa, "lw.i8 E:99999999999999999999999999").as_deref(), Some("OTS-E-OPR-001"));
    assert_eq!(code(&isa, "nop 0x99999999999999999999999999").as_deref(), Some("OTS-E-OPR-001"));
    // a group base (stream_group16) must be a multiple of 16
    let g = parse(&isa, "ifetch E:16");
    assert!(g.is_ok() || g == Err("OTS-E-OPR-001".into()), "{g:?}");
    // a symbol with no extern is ASM-002
    assert_eq!(code(&isa, "read sym, E:0, 0x0").as_deref(), Some("OTS-E-ASM-002"));
}

#[test]
fn shift_amount_bound_follows_the_profile() {
    let r = isa("ref");
    let m = isa("mini");
    assert!(parse(&r, "shift E:0, up, 0x13f, E:1").is_ok());
    assert_eq!(code(&r, "shift E:0, up, 0x140, E:1").as_deref(), Some("OTS-E-OPR-001"));
    assert!(parse(&m, "shift E:0, up, 0xf, E:1").is_ok());
    assert_eq!(code(&m, "shift E:0, up, 0x10, E:1").as_deref(), Some("OTS-E-OPR-001"));
}

#[test]
fn name_kinds_report_dec_003_for_a_value_that_is_not_legal_for_the_entry() {
    let isa = isa("ref");
    // a boolean element type with an operand that is not a boolean value, or a type the entry does not take
    for (t, want) in [("add.i8 E:0, E:1, E:2", None), ("cmp.eq.i8 E:0, E:1, E:2", None)] {
        assert_eq!(code(&isa, t).as_deref(), want, "{t:?}");
    }
    let mut seen_dec = false;
    for ty in ["i8", "f16", "i32", "f32", "u8", "i16", "u16", "u32", "b8", "b16", "b32"] {
        for op in ["add", "and", "mul", "cmp.lt", "relu", "convert", "reduce.add"] {
            let t = match op {
                "convert" => format!("convert.{ty}.f32 E:0, E:1"),
                "relu" => format!("relu.{ty} E:0, E:1"),
                _ => format!("{op}.{ty} E:0, E:1, E:2"),
            };
            seen_dec |= code(&isa, &t).as_deref() == Some("OTS-E-DEC-003");
        }
    }
    assert!(seen_dec, "no (type, op) pair gave DEC-003");
}

fn ext(name: &str, slice: u64, addr: u64, size: u64) -> (String, Extern) {
    (name.to_string(), Extern { name: name.to_string(), slice, addr, size })
}

#[test]
fn symbols_need_an_extern_inside_its_range_and_slice() {
    let isa = isa("ref");
    let syms: HashMap<String, Extern> =
        [ext("a", 1, 0x20, 0x10), ext("w", 1, 0x40, 1), ext("w2", 1, 0x50, 2)].into_iter().collect();
    let p = |t: &str, q: Option<u16>| {
        parse_insn(&isa, t, &syms, q).map(|(i, w)| (render(&i), w)).map_err(|e| e.code.as_str().to_string())
    };
    assert_eq!(p("read a+0x3, E:0, 0x0", Some(1)).unwrap().0, "read 0x23, E:0, 0x0");
    assert_eq!(p("read a, E:0, 0x0", None).unwrap().0, "read 0x20, E:0, 0x0");
    assert_eq!(p("read a+0xf, E:0, 0x0", Some(1)).unwrap().0, "read 0x2f, E:0, 0x0");
    assert_eq!(p("read a+0x10, E:0, 0x0", Some(1)), Err("OTS-E-ASM-002".into())); // outside
    assert_eq!(p("read a, E:0, 0x0", Some(2)), Err("OTS-E-ASM-002".into())); // another slice
    assert_eq!(p("read b, E:0, 0x0", Some(1)), Err("OTS-E-ASM-002".into())); // undefined
                                                                             // spacing around '+' is not allowed: the text matches no template
    assert_eq!(p("read a  +  0x3, E:0, 0x0", Some(1)), Err("OTS-E-DEC-003".into()));
    assert_eq!(p("read a +0x3, E:0, 0x0", Some(1)), Err("OTS-E-DEC-003".into()));
    assert_eq!(p("read a+ 0x3, E:0, 0x0", Some(1)), Err("OTS-E-DEC-003".into()));
    // IfetchM reads words a and a+1: a symbol needs two words of its extern
    assert_eq!(p("ifetchm w", Some(1)), Err("OTS-E-ASM-002".into()));
    assert!(p("ifetchm w2", Some(1)).is_ok());
    assert_eq!(p("ifetchm w2+0x1", Some(1)), Err("OTS-E-ASM-002".into()));
    // a numeric address does no extern check
    assert!(p("ifetchm 0x40", Some(1)).is_ok());
}

#[test]
fn every_template_names_every_operand_once_and_parses_its_own_canonical_text() {
    // the compile step asserts the first; the second: render of random legal values parses to the same word
    for profile in ["ref", "mini"] {
        let isa = isa(profile);
        for e in ots_isa::ENTRIES.iter() {
            let mut f = [0u32; ots_isa::MAX_VARS];
            for (i, v) in e.vars.iter().enumerate() {
                f[i] = v.legal.map_or(0, |l| l[0]);
            }
            if !isa.fields_legal(e, &f) {
                continue;
            }
            let text = ots_isa::render_fields(e, &f).unwrap();
            let (_, w) = parse_insn(&isa, &text, &HashMap::new(), None)
                .unwrap_or_else(|er| panic!("{}: {text:?}: {er}", e.name));
            assert_eq!(w, e.encode_fields(&f), "{}: {text:?}", e.name);
        }
    }
}

#[test]
fn repeat_accepts_d_zero_and_ifetch_takes_any_stream() {
    // Repeat's d field is 0..15 (spacing d + 1); n = 0 stays illegal
    for p in ["ref", "mini"] {
        let isa = isa(p);
        assert_eq!(parse(&isa, "repeat 0x1, 0x0").as_deref(), Ok("repeat 0x1, 0x0"), "{p}");
        assert_eq!(parse(&isa, "repeat 0xfff, 0xf").as_deref(), Ok("repeat 0xfff, 0xf"), "{p}");
        assert_eq!(code(&isa, "repeat 0x0, 0x0").as_deref(), Some("OTS-E-OPR-001"), "{p}");
        assert_eq!(code(&isa, "repeat 0x1, 0x10").as_deref(), Some("OTS-E-OPR-001"), "{p}");
        // Ifetch: one stream operand, no even-stream rule
        for s in ["E:0", "E:1", "E:5", "E:31", "W:5", "W:31"] {
            assert_eq!(parse(&isa, &format!("ifetch {s}")).as_deref(), Ok(format!("ifetch {s}").as_str()), "{p} {s}");
        }
    }
}

#[test]
fn the_host_instructions_assemble_with_the_ranges_of_the_catalogue() {
    // The host instructions: the mnemonics are those of the catalogue (`pci_receive`, `pci_send`, `host_wait`,
    // `host_notify`); PciReceive takes S24..S31 on either row, HostNotify n = 1..15, HostWait any 6-bit doorbell
    for p in ["ref", "mini"] {
        let isa = isa(p);
        for (text, word) in [
            ("pci_receive E:24", "51406000"),
            ("pci_receive W:31", "5140fc00"),
            ("pci_send E:0", "51800000"),
            ("pci_send W:31", "5180fc00"),
            ("host_wait 0x0", "51c00000"),
            ("host_wait 0x3f", "51c0003f"),
            ("host_notify 0x1", "52000001"),
            ("host_notify 0xf", "5200000f"),
        ] {
            assert_eq!(parse(&isa, text).as_deref(), Ok(text), "{p} {text}");
            let (i, _) = parse_insn(&isa, text, &HashMap::new(), None).unwrap();
            assert_eq!(format!("{:08x}", ots_isa::encode(&i)), word, "{p} {text}");
        }
        for (text, c) in [
            ("pci_receive E:2", "OTS-E-OPR-001"),
            ("pci_receive W:23", "OTS-E-OPR-001"),
            ("pci_receive E:32", "OTS-E-OPR-001"),
            ("pci_send E:32", "OTS-E-OPR-001"),
            ("host_wait 0x40", "OTS-E-OPR-001"),
            ("host_notify 0x0", "OTS-E-OPR-001"),
            ("host_notify 0x10", "OTS-E-OPR-001"),
            ("pcireceive E:24", "OTS-E-DEC-003"),
            ("hostwait 0x1", "OTS-E-DEC-003"),
            ("host_notifyx 0x1", "OTS-E-DEC-003"),
        ] {
            assert_eq!(code(&isa, text).as_deref(), Some(c), "{p} {text}");
        }
    }
}

#[test]
fn deskew_recv_assembles_like_receive_with_opcode_9() {
    // : `deskew_recv chip_id, link, stream` has the fields of `receive` (chip 21..16, stream 15..10, link 9..6) with opcode 9
    for p in ["ref", "mini"] {
        let isa = isa(p);
        for (text, word) in [
            ("deskew_recv 0x0, 0x0, E:0", "52400000"),
            ("deskew_recv 0x3f, 0xf, W:31", "527fffc0"),
            ("deskew_recv 0x1, 0x3, E:2", "524108c0"),
        ] {
            assert_eq!(parse(&isa, text).as_deref(), Ok(text), "{p} {text}");
            let (i, _) = parse_insn(&isa, text, &HashMap::new(), None).unwrap();
            assert_eq!(format!("{:08x}", ots_isa::encode(&i)), word, "{p} {text}");
        }
        for (text, c) in [
            ("deskew_recv 0x200, 0x0, E:0", "OTS-E-OPR-001"),
            ("deskew_recv 0x0, 0x80, E:0", "OTS-E-OPR-001"),
            ("deskew_recv 0x0, 0x0, E:32", "OTS-E-OPR-001"),
            ("deskewrecv 0x0, 0x0, E:0", "OTS-E-DEC-003"),
        ] {
            assert_eq!(code(&isa, text).as_deref(), Some(c), "{p} {text}");
        }
    }
}

// tsp_as.py quotes the text of OTS-E-DEC-003 with json.dumps(text, ensure_ascii=False)
#[test]
fn dec003_quotes_the_text_as_json_does() {
    let isa = isa("ref");
    let msg = |t: &str| parse_insn(&isa, t, &HashMap::new(), None).unwrap_err().msg;
    assert_eq!(msg("frob 1"), r#"no instruction matches "frob 1""#);
    assert_eq!(msg("frob \"1\\"), r#"no instruction matches "frob \"1\\""#);
    assert_eq!(msg("frob\u{b}1"), r#"no instruction matches "frob\u000b1""#); // Rust's Debug form is \u{b}
    assert_eq!(msg("frob\u{8}\u{c}1"), r#"no instruction matches "frob\b\f1""#);
}

/// The message of an out-of-range signed operand states its two bounds as plain integers, as the
/// reference assembler does: `out of range -8191..8191`, not a Rust `Some(-8191)..Some(8191)`.
#[test]
fn out_of_range_message_has_plain_integer_bounds() {
    let isa = isa("ref");
    let e = parse_insn(&isa, "step 0x2000", &HashMap::new(), None).unwrap_err();
    assert_eq!(e.code.as_str(), "OTS-E-OPR-001");
    assert!(e.msg.ends_with("8192 out of range -8191..8191"), "{}", e.msg);
    assert!(!e.msg.contains("Some("), "{}", e.msg);
}

#[test]
fn bf16_needs_the_extension() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../opentsp-arch-spec/config/test/ext_bf16.json");
    let with = Isa::new(&Profile::load_file("ext_bf16", path).unwrap());
    let without = isa("ref");
    let ok = parse_insn(&with, "add.bf16 E:8, E:0, E:4", &HashMap::new(), None).unwrap_or_else(|e| panic!("{}", e.msg));
    assert_eq!(render(&ok.0), "add.bf16 E:8, E:0, E:4");
    assert_eq!(code(&without, "add.bf16 E:8, E:0, E:4").as_deref(), Some("OTS-E-DEC-003"));
    assert!(parse(&without, "add.f16 E:8, E:0, E:4").is_ok());
}

#[test]
fn extension_refusal_names_config_file() {
    // --config path kept on the Isa; default profile name stays "ref".
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../opentsp-arch-spec/config/test/ref_ext.json");
    // ref_ext declares wide_acc/block_fp8 but not bf16.
    let p = Profile::load_file("ref", path).unwrap();
    let isa = Isa::with_where(&p, format!("file {path}"));
    let err = parse_insn(&isa, "abc.bf16 W:24", &Default::default(), None).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("needs the extension bf16"), "{msg}");
    assert!(msg.contains("ref_ext.json"), "{msg}");
    assert!(!msg.contains("(profile ref)"), "{msg}");
}

#[test]
fn profile_bound_refusal_names_the_active_configuration() {
    let p = Profile::load("mini").unwrap();
    let isa = Isa::with_where(&p, "file custom-mini.json".into());
    let error = parse_insn(&isa, "acc.i32 E:0, replace, 7", &HashMap::new(), None).unwrap_err();
    assert_eq!(error.code.as_str(), "OTS-E-OPR-001");
    assert!(error.to_string().contains("(file custom-mini.json)"), "{error}");
    assert!(!error.to_string().contains("(profile mini)"), "{error}");
}
