//! SRAM (`core/sram.sail`): each of the 88 MEM slices holds `words_per_slice` words of one stream
//! value each (ISCA 2020 sec. II-B). The storage itself is the harness's in `tsp_sim`
//! (`read_sram_word`, `write_sram_word`); here it is lazily allocated pages of 64 words of
//! `vlanes` bytes, so memory follows use.
//!
//! On top of it are the model's functions: `sram_index`, `sram_bank`, `sram_read`,
//! `sram_write` (with its callback, `T sram`), the tile masks `tile_active` and `merge_tiles`.
use crate::geom::{Geom, MAX_LANES, SLICES};
use crate::machine::Machine;
use crate::observe::Observer;
use crate::srf::Vec320;

const PAGE_WORDS: u64 = 64;

/// The tile masks of `core/sram.sail`: `queue_vl` (SetVL: superlanes at or beyond it are inactive,
/// per issuing queue) and `mem_thread_split` (Config: the tiles below it are thread 0, per MEM
/// slice. Reset: every superlane active, every tile thread 0. The users are `write_planes`
/// (`operands`) and the MEM instructions' tile gating.
#[derive(Clone, Debug)]
pub struct Masks {
    pub queue_vl: [u8; 144],
    pub mem_thread_split: [u8; 88],
}

impl Masks {
    pub fn new(superlanes: u64) -> Masks {
        Masks { queue_vl: [superlanes as u8; 144], mem_thread_split: [superlanes as u8; 88] }
    }

    // model: core/sram.sail masks_reset
    pub fn reset(&mut self, superlanes: u64) {
        *self = Masks::new(superlanes);
    }
}

/// Tile g is written when g < vl and (thread mask) g belongs to thread t.
// model: core/sram.sail tile_active
pub fn tile_active(vl: u64, split: u64, use_thread: bool, t: u8, g: u64) -> bool {
    g < vl && (!use_thread || if t == 0 { g < split } else { g >= split })
}

/// Merge `new` into `old`, keeping old's bytes in inactive tiles (a tile is 16 lanes, one
/// superlane).
// model: core/sram.sail merge_tiles
pub fn merge_tiles(
    old: &Vec320,
    new: &Vec320,
    vl: u64,
    split: u64,
    use_thread: bool,
    t: u8,
    superlanes: u64,
) -> Vec320 {
    let mut out = *old;
    if all_tiles_active(vl, split, use_thread, t, superlanes) {
        let n = 16 * superlanes as usize;
        crate::copy::copy(&mut out[..n], &new[..n]);
        return out;
    }
    for g in 0..superlanes {
        if tile_active(vl, split, use_thread, t, g) {
            let r = 16 * g as usize..16 * (g as usize + 1);
            out[r.clone()].copy_from_slice(&new[r]);
        }
    }
    out
}

/// `merge_tiles(&ZERO_VEC, new, ...)` for `new` the first bytes of a vector (`None`: all zero, as an unwritten SRAM word):
/// the active tiles of `new`, zero elsewhere, in one copy.
#[inline]
pub fn merge_word(new: Option<&[u8]>, vl: u64, split: u64, use_thread: bool, t: u8, superlanes: u64) -> Vec320 {
    let mut out = [0u8; MAX_LANES];
    merge_word_into(&mut out, new, vl, split, use_thread, t, superlanes);
    out
}

/// [`merge_word`] into `out`, which must be all zero on entry (the bytes of inactive tiles stay as they are).
#[inline]
pub fn merge_word_into(
    out: &mut [u8],
    new: Option<&[u8]>,
    vl: u64,
    split: u64,
    use_thread: bool,
    t: u8,
    superlanes: u64,
) {
    let Some(new) = new else { return };
    if all_tiles_active(vl, split, use_thread, t, superlanes) {
        let n = (16 * superlanes as usize).min(new.len());
        crate::copy::copy(&mut out[..n], &new[..n]);
        return;
    }
    for g in 0..superlanes {
        if tile_active(vl, split, use_thread, t, g) {
            let r = 16 * g as usize..(16 * (g as usize + 1)).min(new.len());
            if r.start < r.end {
                crate::copy::copy(&mut out[r.clone()], &new[r]);
            }
        }
    }
}

/// `tile_active` for every tile below `superlanes`, without asking tile by tile.
pub fn all_tiles_active(vl: u64, split: u64, use_thread: bool, t: u8, superlanes: u64) -> bool {
    superlanes == 0 || (vl >= superlanes && (!use_thread || if t == 0 { split >= superlanes } else { split == 0 }))
}

pub struct Sram {
    vlanes: usize,
    words_per_slice: u64,
    pages_per_slice: u64,
    pages: Vec<Option<Box<[u8]>>>,
    pub(crate) revision: [u128; SLICES],
    /// The pre-images of the pages written since [`Sram::journal_begin`] (a parametrised replay run keeps them so that a
    /// fallback can put the SRAM back to the state at the start of the run).
    #[cfg(feature = "replay")]
    journal: Option<Journal>,
}

/// The first contents of every page written while a journal is on: the page index and the page as it was (`None`: never written).
#[cfg(feature = "replay")]
#[derive(Default)]
struct Journal {
    /// One bit for each of a page's 64 words, including writes restored before completion.
    seen: Vec<u64>,
    saved: Vec<(usize, Option<Box<[u8]>>)>,
}

impl Sram {
    pub fn new(g: &Geom) -> Sram {
        let pages_per_slice = g.words_per_slice.div_ceil(PAGE_WORDS);
        Sram {
            vlanes: g.vlanes as usize,
            revision: [0; SLICES],
            words_per_slice: g.words_per_slice,
            pages_per_slice,
            pages: vec![None; (pages_per_slice as usize) * SLICES],
            #[cfg(feature = "replay")]
            journal: None,
        }
    }

    /// Start keeping the pre-image of every page that is written from now on (a journal that is on is dropped first).
    #[cfg(feature = "replay")]
    pub fn journal_begin(&mut self) {
        // One exact word-write mask per page; the saved pre-image still occurs only once per page.
        self.journal = Some(Journal { seen: vec![0; self.pages.len()], saved: Vec::new() });
    }

    /// Drop the journal: the SRAM keeps what it holds.
    #[cfg(feature = "replay")]
    pub fn journal_end(&mut self) {
        self.journal = None;
    }

    /// Whether a journal is on, and the pages it holds.
    #[cfg(feature = "replay")]
    pub fn journal_pages(&self) -> Option<usize> {
        self.journal.as_ref().map(|j| j.saved.len())
    }

    /// Exact word-granular write history for the replay text guard.
    /// The caller validates the SRAM range. A write followed by restoration
    /// still counts; neighboring data on the same page does not force fallback.
    #[cfg(feature = "replay")]
    pub(crate) fn journal_wrote_range(&self, slice: usize, addr: u64, words: u64) -> bool {
        let Some(j) = &self.journal else { return false };
        if words == 0 {
            return false;
        }
        let first = self.locate(slice, addr).0;
        let last = self.locate(slice, addr + words - 1).0;
        (first..=last).any(|p| {
            let lo = if p == first { addr % PAGE_WORDS } else { 0 };
            let hi = if p == last { (addr + words - 1) % PAGE_WORDS + 1 } else { PAGE_WORDS };
            let mask = (u64::MAX << lo) & (u64::MAX >> (PAGE_WORDS - hi));
            j.seen[p] & mask != 0
        })
    }

    /// Put every page written since [`Sram::journal_begin`] back to its first contents and drop the journal: the SRAM as it
    /// was at the start of the run. Returns the pages restored (0 when no journal was on).
    #[cfg(feature = "replay")]
    pub fn journal_undo(&mut self) -> usize {
        let Some(j) = self.journal.take() else { return 0 };
        let n = j.saved.len();
        for (p, old) in j.saved {
            self.pages[p] = old;
        }
        n
    }

    fn locate(&self, slice: usize, addr: u64) -> (usize, usize) {
        assert!(slice < SLICES && addr < self.words_per_slice, "SRAM slice {slice} address {addr} out of range");
        (
            (slice as u64 * self.pages_per_slice + addr / PAGE_WORDS) as usize,
            ((addr % PAGE_WORDS) as usize) * self.vlanes,
        )
    }

    /// The word at `slice`, `addr`: `vlanes` bytes, lane 0 first, then zeros up to `MAX_LANES`; an
    /// untouched word is all zero.
    pub fn read(&self, slice: usize, addr: u64) -> [u8; MAX_LANES] {
        let (p, off) = self.locate(slice, addr);
        let mut w = [0u8; MAX_LANES];
        if let Some(page) = &self.pages[p] {
            crate::copy::copy(&mut w[..self.vlanes], &page[off..off + self.vlanes]);
        }
        w
    }

    /// The word at `slice`, `addr` as stored (`vlanes` bytes), `None` when its page was never written (all zero).
    pub fn word_ref(&self, slice: usize, addr: u64) -> Option<&[u8]> {
        let (p, off) = self.locate(slice, addr);
        self.pages[p].as_ref().map(|page| &page[off..off + self.vlanes])
    }

    /// Write a word: `bytes` beyond `vlanes` are dropped, missing lanes are zero.
    pub fn write(&mut self, slice: usize, addr: u64, bytes: &[u8]) {
        self.revision[slice] += 1;
        let (p, off) = self.locate(slice, addr);
        let vl = self.vlanes;
        #[cfg(feature = "replay")]
        if let Some(j) = &mut self.journal {
            if j.seen[p] == 0 {
                j.saved.push((p, self.pages[p].clone()));
            }
            j.seen[p] |= 1u64 << (addr % PAGE_WORDS);
        }
        let page = self.pages[p].get_or_insert_with(|| vec![0u8; PAGE_WORDS as usize * vl].into_boxed_slice());
        let word = &mut page[off..off + vl];
        let n = bytes.len().min(vl);
        word[n..].fill(0);
        crate::copy::copy(&mut word[..n], &bytes[..n]);
    }

    /// The words that are not all zero, as (slice, address, `vlanes` bytes), in slice then address order: the
    /// contents of the SRAM, independent of which pages happen to be allocated.
    pub fn nonzero_words(&self) -> impl Iterator<Item = (usize, u64, &[u8])> + '_ {
        let (vl, per_slice, words) = (self.vlanes, self.pages_per_slice, self.words_per_slice);
        self.pages.iter().enumerate().filter_map(|(i, p)| p.as_ref().map(|p| (i, p))).flat_map(move |(i, page)| {
            let (slice, first) = (i as u64 / per_slice, (i as u64 % per_slice) * PAGE_WORDS);
            page.chunks_exact(vl)
                .enumerate()
                .map(move |(k, w)| (slice as usize, first + k as u64, w))
                .filter(move |(_, addr, w)| *addr < words && w.iter().any(|b| *b != 0))
        })
    }

    /// Pages allocated so far (a test hook for the lazy allocation).
    pub fn pages_allocated(&self) -> usize {
        self.pages.iter().filter(|p| p.is_some()).count()
    }
}

impl<const L: usize> Machine<L> {
    /// The word index of address `a` (`addr_bits` bits), if it is inside the slice.
    // model: core/sram.sail sram_index
    pub fn sram_index(&self, a: u32) -> Option<u64> {
        let i = a as u64;
        (i < self.geom.words_per_slice).then_some(i)
    }

    /// Bank of an address: bit `bank_bit` (ISCA 2020 sec. III-B: the bank bit is exposed).
    // model: core/sram.sail sram_bank
    pub fn sram_bank(&self, a: u32) -> u8 {
        ((a >> self.geom.bank_bit) & 1) as u8
    }

    // model: core/sram.sail sram_read
    pub fn sram_read(&self, s: usize, a: u32) -> Option<Vec320> {
        self.sram_index(a).map(|i| self.sram.read(s, i))
    }

    /// Write a word and tell the observer (`sram_write_callback`: cycle, slice, address, and the
    /// word). False, and nothing happens, when the address is outside the slice.
    // model: core/sram.sail sram_write
    pub fn sram_write<O: Observer + ?Sized>(&mut self, c: u64, s: usize, a: u32, v: &Vec320, obs: &mut O) -> bool {
        match self.sram_index(a) {
            Some(i) => {
                self.sram.write(s, i, v);
                let vl = self.geom.vlanes as usize;
                obs.sram(c, s as u8, a, &v[..vl]);
                true
            }
            None => false,
        }
    }
}

// Exact word history must remain independent of the once-per-page rollback capture.
#[cfg(all(test, feature = "replay"))]
mod journal_tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn word_history_matches_an_independent_set_across_page_and_slice_edges() {
        for profile in ["mini", "ref"] {
            let g = Geom::new(&ots_profile::Profile::load(profile).unwrap());
            let mut s = Sram::new(&g);
            s.journal_begin();
            let mut written = BTreeSet::new();
            // A neighbor first captures page0; later writes within that page must still set their own bits.
            for (slice, addr) in [(0, 1), (0, 0), (0, 63), (0, 64), (0, 65), (0, 127), (0, 128), (87, 63)] {
                s.write(slice, addr, &[0xa5]);
                s.write(slice, addr, &[]); // restoring bytes must not erase history
                written.insert((slice, addr));
                for query_slice in [0, 1, 87] {
                    for start in 0..130 {
                        for end in start..=130 {
                            let expected = written.iter().any(|&(ss, a)| ss == query_slice && start <= a && a < end);
                            assert_eq!(
                                s.journal_wrote_range(query_slice, start, end - start),
                                expected,
                                "{profile} slice{query_slice} {start}..{end} after {slice}:{addr}"
                            );
                        }
                    }
                }
            }
            assert_eq!(s.journal_pages(), Some(4), "one preimage per dirty page despite many word writes");
            assert_eq!(s.journal_undo(), 4);
            assert_eq!(s.journal_pages(), None);
            assert!(!s.journal_wrote_range(0, 0, 130));
        }
    }
}
