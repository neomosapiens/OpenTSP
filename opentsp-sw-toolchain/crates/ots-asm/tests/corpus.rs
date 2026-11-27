//! The checked-in assembler corpora (`opentsp-arch-spec/test/isa_db/corpus/<profile>/`), read live,
//! through the instruction parser. The expected results are those of `tools/tsp_as.py`.
use ots_asm::parse_insn;
use ots_isa::Isa;
use ots_profile::Profile;
use std::collections::HashMap;

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

#[test]
fn accept_txt() {
    let mut n = 0;
    for profile in PROFILES {
        let isa = isa(profile);
        for r in corpus(profile, "accept.txt") {
            let (text, width, w) = (&r[0], r[1].parse::<u8>().unwrap(), u64::from_str_radix(&r[2], 16).unwrap());
            let (insn, word) =
                parse_insn(&isa, text, &HashMap::new(), None).unwrap_or_else(|e| panic!("{profile}: {text:?}: {e}"));
            assert_eq!((insn.entry().width, word), (width, w), "{profile}: {text:?}");
            assert_eq!(ots_isa::encode(&insn), w, "{profile}: {text:?}: encode of the parsed instruction");
            n += 1;
        }
    }
    assert!(n >= 1300, "{n} rows");
}

#[test]
fn reject_txt() {
    let mut n = 0;
    for profile in PROFILES {
        let isa = isa(profile);
        for r in corpus(profile, "reject.txt") {
            let (text, code) = (&r[0], &r[1]);
            match parse_insn(&isa, text, &HashMap::new(), None) {
                Ok((i, w)) => panic!("{profile}: {text:?} is accepted ({} {w:#x}), want {code}", i.entry().name),
                Err(e) => assert_eq!(e.code.as_str(), code, "{profile}: {text:?}: {e}"),
            }
            n += 1;
        }
    }
    assert!(n >= 1700, "{n} rows");
}
