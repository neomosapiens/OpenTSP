//! The IQ decodes a text word with a bucketed search of the catalogue (`decode_text_fast`: only the entries whose
//! fixed bits agree with the word's ten bits under the bundle bit are tried). It must give what the scan of the whole
//! catalogue (`decode_text`, the model's rule) gives for every pair of units: random words, words built from each
//! entry's fixed bits with random operand and reserved bits, and the tags that are not layouts.
use ots_isa::{Isa, ENTRIES};
use ots_profile::Profile;
use tsp_iss::frontend::{decode_text, decode_text_fast};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn check(isa: &Isa, u0: u32, u1: u32, what: &str) {
    assert_eq!(decode_text_fast(isa, u0, u1), decode_text(isa, u0, u1), "{what}: units {u0:#010x} {u1:#010x}");
}

#[test]
fn random_words_decode_as_the_catalogue_scan_does() {
    for name in ["ref", "mini"] {
        let isa = Isa::new(&Profile::load(name).unwrap());
        let mut r = Rng(0x9E37_79B9_7F4A_7C15);
        for _ in 0..400_000 {
            let w = r.next();
            check(&isa, (w >> 32) as u32, w as u32, name);
        }
    }
}

#[test]
fn words_of_every_entry_decode_as_the_catalogue_scan_does() {
    for name in ["ref", "mini"] {
        let isa = Isa::new(&Profile::load(name).unwrap());
        let mut r = Rng(0xD1B5_4A32_D192_ED03);
        let mut legal = 0;
        for e in ENTRIES.iter() {
            for k in 0..20_000 {
                // the fixed bits, random operand bits, and now and then random reserved bits and a bundle bit
                let mut w = e.fixed | (r.next() & e.var_mask);
                if k % 7 == 0 {
                    w |= r.next() & !e.mask & !e.var_mask;
                }
                let (u0, u1) = if e.width == 64 { ((w >> 32) as u32, w as u32) } else { (w as u32, 0) };
                let u0 = if k % 5 == 0 { u0 | 1 << 31 } else { u0 };
                check(&isa, u0, u1, e.name);
                legal += decode_text(&isa, u0, u1).is_ok() as usize;
            }
        }
        assert!(legal > 100_000, "{name}: the words must reach the legality checks, {legal} decoded");
    }
}

#[test]
fn tags_that_are_not_layouts_and_bundle_bits_decode_as_the_scan_does() {
    let isa = Isa::new(&Profile::load("ref").unwrap());
    for tag in 0..8u32 {
        for low in [0u32, 1, 0xFFFF, 0x0FFF_FFFF, 0x0123_4567] {
            for bundle in [0u32, 1 << 31] {
                for u1 in [0u32, 0xFFFF_FFFF, 0x89AB_CDEF] {
                    check(&isa, bundle | tag << 28 | low, u1, "tag sweep");
                }
            }
        }
    }
}
