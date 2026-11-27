//! `tsp_as` decodes a word only if rendering and re-parsing it gives exactly that
//! word. `ots-isa` decodes by field checks; here both rules run on the same words and must agree:
//! every corpus word, and seeded random words around every catalogue entry.
use ots_asm::{decode_by_reparse, parse_insn};
use ots_isa::{encode, render, Insn, Isa, ENTRIES, MAX_VARS};
use ots_profile::Profile;
use std::collections::HashMap;

const PROFILES: [&str; 2] = ["ref", "mini"];

fn corpus(profile: &str, name: &str) -> Vec<Vec<String>> {
    let path = format!("{}/../../../opentsp-arch-spec/test/isa_db/corpus/{profile}/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect()
}

fn hex(s: &str) -> u64 {
    u64::from_str_radix(s, 16).unwrap()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn agree(isa: &Isa, profile: &str, width: u8, w: u64, what: &str) -> bool {
    let a = isa.decode(width, w);
    let b = decode_by_reparse(isa, width, w);
    assert_eq!(a, b, "{profile} {what}: width {width} word {w:#x}: field checks vs render-then-parse");
    if let Some(i) = a {
        assert_eq!(encode(&i), w);
        // and the canonical text is a fixed point of the parser
        let (j, pw) = parse_insn(isa, &render(&i), &HashMap::new(), None).unwrap();
        assert_eq!((pw, j.entry().width), (w, width));
    }
    a.is_some()
}

#[test]
fn decoders_agree_on_every_corpus_word() {
    let mut n = 0;
    for profile in PROFILES {
        let isa = Isa::new(&Profile::load(profile).unwrap());
        for r in corpus(profile, "accept.txt") {
            assert!(
                agree(&isa, profile, r[1].parse().unwrap(), hex(&r[2]), &r[0]),
                "{profile}: accepted {:?} must decode",
                r[0]
            );
            n += 1;
        }
        for r in corpus(profile, "text.txt") {
            let u0 = hex(&r[1]) as u32 & 0x7FFF_FFFF;
            let (width, w) = if r[2] == "-" { (32, u0 as u64) } else { (64, ((u0 as u64) << 32) | hex(&r[2])) };
            agree(&isa, profile, width, w, &r[4]);
            n += 1;
        }
        for r in corpus(profile, "groups.txt") {
            let w0 = hex(&r[2]);
            let width =
                if w0 >> 32 == 0 && ots_isa::NARROW_TAGS.contains(&(((w0 >> 28) & 0xF) as u8)) { 32 } else { 64 };
            agree(&isa, profile, width, w0, &r[4]);
            n += 1;
        }
    }
    assert!(n > 6000, "{n} words");
}

#[test]
fn decoders_agree_on_seeded_random_words_of_every_entry() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut decoded = 0;
    let mut total = 0;
    for profile in PROFILES {
        let isa = Isa::new(&Profile::load(profile).unwrap());
        for e in ENTRIES.iter() {
            for k in 0..300 {
                let mut f = [0u32; MAX_VARS];
                for (i, v) in e.vars.iter().enumerate() {
                    let nb = v.nbits as u32;
                    let r = rng.next();
                    f[i] = match (k % 3, v.legal) {
                        (0, Some(l)) => l[(r as usize >> 8) % l.len()],
                        (1, _) => (r & ((1u64 << nb) - 1)) as u32,
                        _ => ((r & 0xFF) as u32).min(((1u64 << nb) - 1) as u32), // small values
                    };
                }
                let mut w = encode(&Insn::from_fields(e.id, &f));
                match k % 7 {
                    1 => w ^= 1 << (rng.next() % e.width as u64), // flip any bit: a fixed bit, or an operand
                    2 => w |= rng.next() & !e.mask & !e.var_mask & if e.width == 64 { u64::MAX } else { 0xFFFF_FFFF },
                    _ => {}
                }
                decoded += agree(&isa, profile, e.width, w, e.name) as u32;
                total += 1;
            }
        }
    }
    assert!(decoded > total / 4, "only {decoded} of {total} random words decode: the test is too weak");
}

#[test]
fn decoders_agree_on_random_words_of_either_width() {
    let mut rng = Rng(0xDEADBEEF12345);
    for profile in PROFILES {
        let isa = Isa::new(&Profile::load(profile).unwrap());
        for _ in 0..3000 {
            let w = rng.next();
            agree(&isa, profile, 64, w, "random");
            agree(&isa, profile, 32, w & 0xFFFF_FFFF, "random");
        }
    }
}
