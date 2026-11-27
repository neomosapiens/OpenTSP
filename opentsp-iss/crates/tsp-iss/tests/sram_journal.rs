//! The SRAM's page journal (`Sram::journal_begin`, `journal_undo`): what a served run of the parametrised replay keeps so that the run
//! can be repeated on the normal driver. Run: `cargo test -p tsp-iss --features replay --test sram_journal`.
#![cfg(feature = "replay")]
use ots_profile::Profile;
use tsp_iss::sram::Sram;
use tsp_iss::Geom;

fn contents(s: &Sram) -> Vec<(usize, u64, Vec<u8>)> {
    s.nonzero_words().map(|(sl, a, w)| (sl, a, w.to_vec())).collect()
}

#[test]
fn undo_puts_back_every_page_written_since_the_journal_began() {
    let g = Geom::new(&Profile::load("mini").unwrap());
    let mut s = Sram::new(&g);
    let vl = g.vlanes as usize;
    let (mid, far) = (100, 200); // pages of 64 words: 1 and 3 (mini has 256 words a slice)
    assert!(g.words_per_slice > far);
    // before the journal: words on three pages (two words of one page, one of another, one on another slice)
    s.write(3, 0, &vec![7u8; vl]);
    s.write(3, 1, &vec![8u8; vl]);
    s.write(3, mid, &vec![9u8; vl]);
    s.write(5, 0, &vec![1u8; vl]);
    let before = contents(&s);
    let pages_before = s.pages_allocated();
    assert!(s.journal_pages().is_none());
    s.journal_begin();
    // during: an overwrite of a word, a second write to the same word, a word on a page that did not exist (slice 9), a zero over a word
    s.write(3, 0, &vec![0xaau8; vl]);
    s.write(3, 0, &vec![0xbbu8; vl]);
    s.write(3, 7, &vec![0xccu8; vl]);
    s.write(9, far, &vec![0xddu8; vl]);
    s.write(5, 0, &vec![0u8; vl]);
    assert_ne!(contents(&s), before);
    assert_eq!(s.journal_pages(), Some(3), "one pre-image per page written, whatever the number of writes");
    assert_eq!(s.journal_undo(), 3);
    assert_eq!(contents(&s), before, "the SRAM is what it was at the start");
    assert_eq!(s.pages_allocated(), pages_before, "a page that did not exist does not exist again");
    assert_eq!(s.journal_pages(), None);
    assert_eq!(s.journal_undo(), 0, "no journal, nothing to undo");
}

#[test]
fn a_journal_that_ends_keeps_what_the_run_wrote_and_costs_no_pre_image_afterwards() {
    let g = Geom::new(&Profile::load("mini").unwrap());
    let mut s = Sram::new(&g);
    let vl = g.vlanes as usize;
    s.journal_begin();
    s.write(1, 0, &vec![5u8; vl]);
    s.journal_end();
    s.write(1, 1, &vec![6u8; vl]);
    assert_eq!(s.journal_undo(), 0);
    assert_eq!(contents(&s).len(), 2);
}
