//! `groups.txt` of the assembler corpora: instructions that share one parcel of one queue, through
//! the slot rule (`encoding.json` `mem_slots`): the model's group code per instruction.
use ots_format::text::word_units;
use ots_isa::{render, Code, Isa};
use ots_profile::Profile;
use std::collections::BTreeMap;

#[test]
fn groups_txt_slot_rule_and_bank_rule() {
    let mut rows_seen = 0;
    for profile in ["ref", "mini"] {
        let isa = Isa::new(&Profile::load(profile).unwrap());
        let path = format!(
            "{}/../../../opentsp-arch-spec/test/isa_db/corpus/{profile}/groups.txt",
            env!("CARGO_MANIFEST_DIR")
        );
        type Group = (u16, Vec<(u64, String, String)>);
        let mut groups: BTreeMap<u32, Group> = BTreeMap::new();
        for l in std::fs::read_to_string(path).unwrap().lines().filter(|l| !l.starts_with('#') && !l.is_empty()) {
            let t: Vec<&str> = l.split('\t').collect();
            let g = groups.entry(t[0].parse().unwrap()).or_insert((u16::from_str_radix(t[1], 16).unwrap(), vec![]));
            g.1.push((u64::from_str_radix(t[2], 16).unwrap(), t[3].to_string(), t[4].to_string()));
        }
        let mut with_code = 0;
        for (n, (q, rows)) in &groups {
            let insns: Vec<_> =
                rows.iter().map(|(w, _, _)| isa.decode_text(&word_units(*w)).ok().map(|(i, _)| i)).collect();
            assert!(insns.iter().all(Option::is_some), "{profile} group {n}: a corpus word does not decode");
            let codes = isa.group_codes(*q, &insns);
            for ((i, c), (w, code, text)) in insns.iter().zip(&codes).zip(rows) {
                assert_eq!(&render(i.as_ref().unwrap()), text, "{profile} group {n} word {w:x}");
                assert_eq!(c.map_or("-", Code::as_str), code, "{profile} group {n} queue {q:x} word {w:x} ({text})");
                with_code += c.is_some() as u32;
                rows_seen += 1;
            }
        }
        assert!(with_code > 100, "{profile}: only {with_code} coded rows");
    }
    assert!(rows_seen >= 1500, "{rows_seen}");
}

#[test]
fn group_corner_cases() {
    let isa = Isa::new(&Profile::load("ref").unwrap());
    let d = |w: u64| Some(isa.decode_text(&word_units(w)).unwrap().0);
    // a lone instruction has no group code, whatever it is
    assert_eq!(isa.group_codes(5, &[d(0x0040_0000)]), vec![None]);
    // several records on a non-MEM queue, or an undecodable one among several: QUE-001 on each
    assert_eq!(isa.group_codes(100, &[d(0x0040_0000), d(0x0080_0000)]), vec![Some(Code::Que001); 2]);
    assert_eq!(isa.group_codes(5, &[d(0x0040_0000), None]), vec![Some(Code::Que001); 2]);
    assert!(isa.group_codes(5, &[]).is_empty());
}
