//! The checked-in assembler corpora (`opentsp-arch-spec/test/isa_db/corpus/<profile>/`), read
//! live, against the generated decoder, renderer and placement. The expected results are those of
//! `tools/tsp_as.py` (itself held to the Sail model).
use ots_isa::{
    canonical_queue_name, decode_text_units, encode, placement, queue_id, queue_name, render, Code, EntryId, Insn, Isa,
    CONTROL, ENTRIES,
};
use ots_profile::Profile;

const PROFILES: [&str; 2] = ["ref", "mini"];

fn corpus(profile: &str, name: &str) -> Vec<Vec<String>> {
    let path = format!("{}/../../../opentsp-arch-spec/test/isa_db/corpus/{profile}/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| {
            // reject.txt: split at the LAST tab (the text may contain tabs); the others at every tab
            if name == "reject.txt" {
                let (a, b) = l.rsplit_once('\t').unwrap();
                vec![a.to_string(), b.to_string()]
            } else {
                l.split('\t').map(str::to_string).collect()
            }
        })
        .collect()
}

fn isa(profile: &str) -> Isa {
    Isa::new(&Profile::load(profile).unwrap())
}

fn hex(s: &str) -> u64 {
    u64::from_str_radix(s, 16).unwrap()
}

#[test]
fn text_txt_decode_and_placement() {
    let mut n = 0;
    for profile in PROFILES {
        let isa = isa(profile);
        for r in corpus(profile, "text.txt") {
            let (q, u0, u1, code, text) = (hex(&r[0]) as u16, hex(&r[1]) as u32, &r[2], &r[3], &r[4]);
            let mut units = vec![u0];
            if u1 != "-" {
                units.push(hex(u1) as u32);
            }
            let (got_code, got_text) = match isa.decode_text(&units) {
                Err(c) => {
                    let w =
                        if units.len() == 1 { units[0] as u64 } else { ((units[0] as u64) << 32) | units[1] as u64 };
                    (c.as_str().to_string(), format!("illegal{} {w:#x}", if units.len() == 1 { 32 } else { 64 }))
                }
                Ok((insn, used)) => {
                    assert_eq!(used as usize, units.len(), "{profile} {q:x} {u0:08x}: units consumed");
                    (placement(insn.id(), q).map_or("-".to_string(), |c| c.as_str().to_string()), render(&insn))
                }
            };
            assert_eq!((&got_code, &got_text), (code, text), "{profile} queue {q:x} units {units:08x?}");
            n += 1;
        }
    }
    assert!(n >= 3000, "{n} rows");
}

#[test]
fn accept_words_decode_to_their_entry() {
    let mut n = 0;
    for profile in PROFILES {
        let isa = isa(profile);
        for r in corpus(profile, "accept.txt") {
            let (text, width, w) = (&r[0], r[1].parse::<u8>().unwrap(), hex(&r[2]));
            let insn = isa.decode(width, w).unwrap_or_else(|| panic!("{profile}: {text:?} = {w:#x} does not decode"));
            assert_eq!(encode(&insn), w, "{profile}: {text:?}: encode(decode(w))");
            let e = insn.id().entry();
            assert_eq!(e.width, width);
            assert_eq!(
                text.trim().split(' ').next().unwrap().split('.').next().unwrap(),
                e.mnemonic.split('.').next().unwrap(),
                "{profile}: {text:?} decodes to {}",
                e.name
            );
            // the canonical text is a fixed point of render for the same word
            assert!(render(&insn).starts_with(e.mnemonic.split('.').next().unwrap()));
            n += 1;
        }
    }
    assert!(n >= 1300, "{n} rows");
}

#[test]
fn decode_by_field_checks_agrees_with_the_model_text_on_every_corpus_word() {
    // The reference assembler decodes a word only if rendering and re-parsing it gives that word. The corpora
    // carry, for 1500 text starts per profile, the result of that rule (code and canonical text); the
    // field-check decoder must agree on each, and on every accepted word its entry must be the entry
    // the canonical text names.
    for profile in PROFILES {
        let isa = isa(profile);
        for r in corpus(profile, "text.txt") {
            let units: Vec<u32> =
                std::iter::once(hex(&r[1]) as u32).chain((r[2] != "-").then(|| hex(&r[2]) as u32)).collect();
            let got = isa.decode_text(&units);
            assert_eq!(got.is_err(), r[3] == "OTS-E-DEC-003", "{profile} {units:08x?}: corpus code {}", r[3]);
            if let Ok((insn, _)) = got {
                assert_eq!(render(&insn), r[4]);
                assert_eq!(isa.decode(insn.id().entry().width, encode(&insn)).map(|i| i.id()), Some(insn.id()));
            }
        }
    }
}

#[test]
fn encode_decode_round_trip_of_every_entry_with_pseudo_random_legal_fields() {
    let mut s = 0x2545F4914F6CDD1Du64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for profile in PROFILES {
        let isa = isa(profile);
        for e in ENTRIES.iter() {
            let mut ok = 0;
            for _ in 0..300 {
                let mut f = [0u32; ots_isa::MAX_VARS];
                for (i, v) in e.vars.iter().enumerate() {
                    let nb = v.nbits as u32;
                    f[i] = (next() & ((1u64 << nb) - 1)) as u32;
                }
                let insn = Insn::from_fields(e.id, &f);
                let w = encode(&insn);
                match isa.decode(e.width, w) {
                    Some(d) if !isa.fields_legal(e, &f) => {
                        // illegal fields whose word is another entry's (ABC with type 3 is the word of ABC8)
                        assert_ne!(d.id(), insn.id(), "{} {w:#x}: illegal fields decode to the entry itself", e.name);
                        assert_eq!(encode(&d), w);
                    }
                    Some(d) => {
                        assert_eq!(d, insn, "{} {w:#x}", e.name);
                        assert_eq!(encode(&d), w);
                        ok += 1;
                    }
                    None => assert!(!isa.fields_legal(e, &f), "{} {w:#x}: legal fields but no decode", e.name),
                }
            }
            assert!(
                ok > 0 || e.vars.iter().any(|v| v.legal.is_some() || v.max_expr.is_some() || v.max.is_some()),
                "{}: no legal random value",
                e.name
            );
        }
    }
}

#[test]
fn a_changed_fixed_or_reserved_bit_does_not_decode_to_the_same_entry() {
    let isa = isa("ref");
    // `sync` is 0x00400000 with every other bit fixed
    assert_eq!(isa.decode(32, 0x0040_0000).map(|i| i.id()), Some(EntryId::Sync));
    assert!(isa.decode(32, 0x0040_0001).is_none());
    assert!(isa.decode(32, 0x0040_0000 | 1 << 20).is_none());
    // a bit that is neither fixed nor an operand is reserved and must be zero. The 0.1 catalogue has
    // none (every `-` of a match pattern is an operand bit), so this holds for the rule itself: for
    // every entry the three masks cover the word.
    for e in ENTRIES.iter() {
        let all = if e.width == 64 { u64::MAX } else { (1u64 << e.width) - 1 };
        let reserved = all & !e.mask & !e.var_mask;
        for b in 0..e.width {
            if reserved >> b & 1 == 1 {
                assert!(isa.decode(e.width, e.fixed | 1 << b).is_none(), "{}: reserved bit {b} set", e.name);
            }
        }
    }
}

#[test]
fn queues_names_aliases_and_units() {
    assert_eq!(queue_id("MEM[0]"), Some(0));
    assert_eq!(queue_id("MEM[007]"), Some(7));
    assert_eq!(queue_id("MEM[000]"), Some(0));
    assert_eq!(queue_id("MEM[144]"), None);
    assert_eq!(queue_id("cu"), None);
    assert_eq!(queue_id("MEM [0]"), None);
    assert_eq!(queue_id("CU"), Some(CONTROL));
    assert_eq!(queue_name(CONTROL), Some("CU"));
    assert_eq!(queue_name(143).map(queue_id), Some(Some(143)));
    assert_eq!(canonical_queue_name("MEM[007][0]"), "MEM[7][0]");
    assert_eq!(canonical_queue_name("MEM[]"), "MEM[]");
    assert_eq!(canonical_queue_name("MEM[0"), "MEM[0");
    for q in 0..144u16 {
        assert_eq!(queue_id(queue_name(q).unwrap()), Some(q));
    }
    assert_eq!(ots_isa::unit_queues("VXM").map(|q| q.len()), Some(16));
}

#[test]
fn placement_follows_the_units_and_the_control_channel() {
    assert_eq!(placement(EntryId::Read, 0), None);
    assert_eq!(placement(EntryId::Read, 88), Some(Code::Que002));
    assert_eq!(placement(EntryId::Read, CONTROL), Some(Code::Que002));
    assert_eq!(placement(EntryId::Sync, CONTROL), None);
    assert_eq!(placement(EntryId::Sync, 5), None);
    assert_eq!(placement(EntryId::Add, 88), None);
    assert_eq!(placement(EntryId::Add, 0xFFFE), Some(Code::Que002));
    assert_eq!(placement(EntryId::NOP, CONTROL), None);
}

#[test]
fn decode_rule_corners() {
    let isa = isa("ref");
    // tag 6 and tag 7 do not decode, one unit consumed; a wide tag needs the second unit
    assert_eq!(isa.decode_text(&[0x6FFF_FFFF]), Err(Code::Dec003));
    assert_eq!(isa.decode_text(&[0x7000_0000]), Err(Code::Dec003));
    assert_eq!(isa.decode_text(&[0x1000_0000]), Err(Code::Dec003));
    // the bundle bit is the front end's: clear before decoding
    let (a, _) = isa.decode_text(&[0x0040_0000]).unwrap();
    let (b, _) = isa.decode_text(&[0x8040_0000]).unwrap();
    assert_eq!(a, b);
    assert_eq!(decode_text_units(0x0000_0001), Some(1));
    assert_eq!(decode_text_units(0x1000_0000), Some(2));
    assert_eq!(decode_text_units(0x6000_0000), None);
}

#[test]
fn shift_amount_depends_on_the_profile() {
    // lane_count: max_expr vlanes-1 (0x13f on ref, 0xf on mini)
    let r = isa("ref");
    let m = isa("mini");
    let w_of = |isa: &Isa, amount: u32| {
        let e = ENTRIES.iter().find(|e| e.name == "Shift").unwrap();
        let mut f = [0u32; ots_isa::MAX_VARS];
        for (i, v) in e.vars.iter().enumerate() {
            f[i] = if v.name == "amount" { amount } else { 0 };
        }
        let w = encode(&Insn::from_fields(e.id, &f));
        isa.decode(e.width, w).is_some()
    };
    assert!(w_of(&r, 0x13f) && !w_of(&r, 0x140));
    assert!(w_of(&m, 0xf) && !w_of(&m, 0x10));
}

#[test]
fn max_of_is_the_profile_variable_read_once_at_construction() {
    // : `Isa::new` reads each profile variable a `max_expr` names once (MAX_EXPR_VARS, MaxExpr::idx); every bound of every
    // bounded operand must equal what the by-name lookup gives, on profiles with different sizes
    assert!(!ots_isa::MAX_EXPR_BOUNDS.is_empty());
    let mut bounded = 0;
    let dir = format!("{}/../../../opentsp-arch-spec/config", env!("CARGO_MANIFEST_DIR"));
    for profile in ["ref", "mini", "test/sl8", "test/sl32"] {
        let p = Profile::load_file(profile.rsplit('/').next().unwrap(), format!("{dir}/{profile}.json")).unwrap();
        let isa = Isa::new(&p);
        for e in ENTRIES.iter() {
            for v in e.vars.iter().filter(|v| v.max_expr.is_some()) {
                let me = v.max_expr.unwrap();
                assert_eq!(ots_isa::MAX_EXPR_BOUNDS[me.idx as usize], (me.var, me.offset), "{} {}", e.name, v.name);
                let want = p.variable(me.var).unwrap_or(p.vlanes()) as i64 - me.offset as i64;
                let want = v.max.map_or(want, |m| m.min(want));
                assert_eq!(isa.max_of(v), Some(want), "{profile} {} {}", e.name, v.name);
                bounded += 1;
            }
        }
    }
    assert!(bounded >= 4 * 6, "the bounded operands of the catalogue: {bounded}");
}
