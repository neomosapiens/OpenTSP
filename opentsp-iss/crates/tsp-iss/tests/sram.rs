//! Tests of `sram`, moved from the module's in-source test block (they use the crate's public API).
use ots_profile::Profile;
use tsp_iss::geom::{Geom, MAX_LANES};
use tsp_iss::sram::*;

fn sram(name: &str) -> Sram {
    Sram::new(&Geom::new(&Profile::load(name).unwrap()))
}

#[test]
fn untouched_words_are_zero_and_allocation_is_lazy() {
    let mut s = sram("ref");
    assert_eq!(s.pages_allocated(), 0);
    assert_eq!(s.read(87, 8191), [0; tsp_iss::geom::MAX_LANES]);
    s.write(87, 8191, &[1, 2, 3]);
    assert_eq!(s.pages_allocated(), 1);
    let w = s.read(87, 8191);
    assert_eq!(&w[..4], &[1, 2, 3, 0]);
    assert_eq!(s.read(87, 8190), [0; tsp_iss::geom::MAX_LANES]);
    assert_eq!(s.read(86, 8191), [0; tsp_iss::geom::MAX_LANES]);
}

#[test]
fn write_replaces_the_word_and_drops_bytes_beyond_vlanes() {
    let mut s = sram("mini");
    let all: Vec<u8> = (0..=255).collect();
    s.write(3, 5, &all[..40]);
    let w = s.read(3, 5);
    assert_eq!(&w[..16], &all[..16]);
    assert!(w[16..].iter().all(|b| *b == 0));
    s.write(3, 5, &[9]);
    let w = s.read(3, 5);
    assert_eq!(w[0], 9);
    assert!(w[1..].iter().all(|b| *b == 0));
}

#[test]
#[should_panic(expected = "out of range")]
fn out_of_range_panics() {
    sram("mini").read(0, 256);
}

// ---- I3: the SRAM functions of core/sram.sail and the tile masks ---------------------------

use tsp_iss::links::NoLinks;
use tsp_iss::observe::Observer;
use tsp_iss::srf::Vec320;

fn machine(name: &str) -> tsp_iss::Machine {
    tsp_iss::Machine::new(&Profile::load(name).unwrap(), 0, Box::new(NoLinks)).unwrap()
}

fn vec_of(b: u8, n: usize) -> Vec320 {
    let mut x = [0u8; MAX_LANES];
    x[..n].fill(b);
    x
}

#[derive(Default)]
struct Writes(Vec<(u64, u8, u32, Vec<u8>)>);
impl Observer for Writes {
    fn sram(&mut self, c: u64, slice: u8, addr: u32, v: &[u8]) {
        self.0.push((c, slice, addr, v.to_vec()));
    }
}

// model: core/sram.sail tile_active
#[test]
fn sram_tile_active_follows_vl_and_the_thread_split() {
    // no thread gating: tiles below the vector length
    assert!(tile_active(3, 20, false, 0, 2) && !tile_active(3, 20, false, 0, 3));
    // thread 0: tiles below the split; thread 1: the others; both below vl
    assert!(tile_active(20, 7, true, 0, 6) && !tile_active(20, 7, true, 0, 7));
    assert!(!tile_active(20, 7, true, 1, 6) && tile_active(20, 7, true, 1, 7) && tile_active(20, 7, true, 1, 19));
    assert!(!tile_active(10, 7, true, 1, 10), "vl still limits thread 1");
    // thread tag ignored when not gated
    assert!(tile_active(20, 7, false, 1, 0));
}

// model: core/sram.sail merge_tiles
#[test]
fn sram_merge_tiles_keeps_old_bytes_in_inactive_tiles() {
    let old = vec_of(0xaa, 320);
    let new = vec_of(0x55, 320); // the 20 tiles of ref; the lanes above are not tiles of the profile
    let m = merge_tiles(&old, &new, 20, 7, true, 1, 20);
    assert!(m[..7 * 16].iter().all(|b| *b == 0xaa) && m[7 * 16..320].iter().all(|b| *b == 0x55));
    let m = merge_tiles(&old, &new, 2, 20, false, 0, 20);
    assert!(m[..32].iter().all(|b| *b == 0x55) && m[32..320].iter().all(|b| *b == 0xaa));
    // the mini profile has one superlane: lanes beyond it are untouched
    let m = merge_tiles(&vec_of(0, 16), &vec_of(1, 16), 1, 1, false, 0, 1);
    assert_eq!(m, vec_of(1, 16));
    // a zero base stays zero outside the active tiles
    let z = merge_tiles(&[0; MAX_LANES], &new, 5, 20, false, 0, 20);
    assert!(z[..80].iter().all(|b| *b == 0x55) && z[80..].iter().all(|b| *b == 0));
}

// model: core/sram.sail sram_index, sram_bank
#[test]
fn sram_index_and_bank_follow_the_profile() {
    let m = machine("mini");
    assert_eq!(
        (m.sram_index(0), m.sram_index(255), m.sram_index(256), m.sram_index(8191)),
        (Some(0), Some(255), None, None)
    );
    assert_eq!((m.sram_bank(127), m.sram_bank(128), m.sram_bank(255), m.sram_bank(0)), (0, 1, 1, 0));
    let m = machine("ref");
    assert_eq!((m.sram_index(8191), m.sram_index(0)), (Some(8191), Some(0)));
    assert_eq!((m.sram_bank(4095), m.sram_bank(4096), m.sram_bank(8191)), (0, 1, 1));
}

// model: core/sram.sail sram_read, sram_write
#[test]
fn sram_write_stores_reports_and_sram_read_returns_it() {
    let mut m = machine("mini");
    let mut w = Writes::default();
    assert_eq!(m.sram_read(3, 9), Some([0; MAX_LANES]));
    assert!(m.sram_write(40, 3, 9, &vec_of(0x7e, 16), &mut w));
    assert_eq!(m.sram_read(3, 9), Some(vec_of(0x7e, 16)));
    assert_eq!(w.0, vec![(40, 3, 9, vec![0x7e; 16])]);
    // out of range: no write, no callback
    assert!(!m.sram_write(41, 3, 256, &vec_of(1, 16), &mut w));
    assert_eq!(m.sram_read(3, 256), None);
    assert_eq!(w.0.len(), 1);
    // lanes beyond vlanes are dropped; the word is replaced, not merged
    let mut x = vec_of(0x11, 16);
    x[20] = 0xff;
    m.sram_write(42, 3, 9, &x, &mut w);
    assert_eq!(m.sram_read(3, 9), Some(vec_of(0x11, 16)));
    // the public view reads the same word
    assert_eq!(m.sram_word(3, 9).unwrap(), vec_of(0x11, 16));
    m.sram_write(43, 87, 255, &vec_of(0x22, 16), &mut w);
    assert_eq!(w.0.last().unwrap(), &(43, 87, 255, vec![0x22; 16]));
}
