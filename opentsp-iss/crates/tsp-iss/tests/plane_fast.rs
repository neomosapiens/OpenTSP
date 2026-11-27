//! The single-stream operand paths (`read_plane`, `write_plane`) and the whole-word fast paths of `merge_tiles` and the
//! SRAM write against the general routines they shortcut: `read_planes`, `write_planes`, a tile-by-tile merge and a
//! zero-fill-then-copy write. Every mask shape: all tiles active, none, a SetVL cut, and each thread split.
use ots_isa::Code;
use ots_profile::{Latency, Profile};
use tsp_iss::geom::MAX_LANES;
use tsp_iss::layout::{read_gap, RowId, UnitKind};
use tsp_iss::links::NoLinks;
use tsp_iss::observe::NoObserver;
use tsp_iss::operands::*;
use tsp_iss::slices::Issue;
use tsp_iss::sram::{merge_tiles, tile_active};
use tsp_iss::srf::Vec320;
use tsp_iss::Machine;

fn machine(name: &str) -> Machine {
    Machine::new(&Profile::load(name).unwrap(), 0, Box::new(NoLinks)).unwrap()
}

fn ctx(q: usize, thread: u8) -> Issue {
    Issue {
        cycle: 0,
        queue: q,
        element: 0,
        addr_offset: 0,
        stream_offset: 0,
        thread,
        lat: Latency { d_func: 1, d_skew: 0 },
    }
}

fn pattern(seed: u8, vl: usize) -> Vec320 {
    let mut x = [0u8; MAX_LANES];
    for (l, b) in x[..vl].iter_mut().enumerate() {
        *b = seed.wrapping_mul(31).wrapping_add(l as u8).wrapping_mul(7) | 1;
    }
    x
}

/// The literal tile-by-tile merge.
fn merge_reference(old: &Vec320, new: &Vec320, vl: u64, split: u64, use_thread: bool, t: u8, sl: u64) -> Vec320 {
    let mut out = *old;
    for g in 0..sl {
        if tile_active(vl, split, use_thread, t, g) {
            let r = 16 * g as usize..16 * (g as usize + 1);
            out[r.clone()].copy_from_slice(&new[r]);
        }
    }
    out
}

#[test]
fn merge_tiles_equals_the_tile_by_tile_merge_for_every_mask() {
    for sl in [1u64, 4, 20] {
        let old = pattern(3, 16 * sl as usize);
        let new = pattern(9, 16 * sl as usize);
        for vl in 0..=sl + 1 {
            for split in 0..=sl + 1 {
                for use_thread in [false, true] {
                    for t in 0..2u8 {
                        let got = merge_tiles(&old, &new, vl, split, use_thread, t, sl);
                        let want = merge_reference(&old, &new, vl, split, use_thread, t, sl);
                        assert_eq!(got, want, "sl {sl} vl {vl} split {split} thread-gated {use_thread} t {t}");
                    }
                }
            }
        }
    }
}

#[test]
fn write_plane_schedules_what_write_planes_schedules() {
    for (q, threads) in [(100usize, false), (3, true), (104, false)] {
        for vl in [0u8, 1, 7, 19, 20] {
            for split in [0u8, 5, 20] {
                for thread in 0..2u8 {
                    let mut a = machine("ref");
                    let mut b = machine("ref");
                    let sl = a.geom().superlanes as usize;
                    let v = pattern(5, a.geom().vlanes as usize);
                    for m in [&mut a, &mut b] {
                        m.edit_state(|state| state.masks.queue_vl[q] = vl).unwrap();
                        if threads {
                            m.edit_state(|state| state.masks.mem_thread_split[q] = split).unwrap();
                        }
                    }
                    let c = ctx(q, thread);
                    let ra = a.write_planes(&c, 4, &one_plane(v), 1, 77);
                    let rb = b.write_plane(&c, 4, &v, 77);
                    assert_eq!(ra, rb);
                    assert_eq!(
                        a.driver().pending[77 % a.driver().pending.len()],
                        b.driver().pending[77 % b.driver().pending.len()],
                        "q {q} vl {vl} split {split} t {thread} sl {sl}"
                    );
                    assert_eq!(a.driver().pending_occ, b.driver().pending_occ);
                }
            }
        }
    }
}

#[test]
fn write_plane_reports_what_write_planes_reports() {
    let v = pattern(2, 320);
    for (q, s) in [(104usize, 32 | 2), (100, 3), (120, 32 | 1), (120, 1)] {
        let mut a = machine("ref");
        let mut b = machine("ref");
        let ra = a.write_planes(&ctx(q, 0), s, &one_plane(v), 1, 9);
        let rb = b.write_plane(&ctx(q, 0), s, &v, 9);
        assert_eq!(ra, rb, "queue {q} stream {s}");
        assert_eq!(a.driver().pending, b.driver().pending);
    }
}

#[test]
fn read_plane_returns_what_read_planes_returns() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    let gv = read_gap(m.geom(), UnitKind::Vxm, 0, RowId::E).unwrap();
    let v = pattern(8, vl);
    m.srf_write(RowId::E, 5, gv - 2, 30, &v, &mut NoObserver);
    m.srf_write(RowId::E, 9, gv + 2, 25, &v, &mut NoObserver);
    for (s, t) in [(5u32, 32i64), (5, 31), (5, 33), (9, 27), (6, 40), (32 | 5, 32)] {
        let want = m.read_planes(&ctx(100, 0), s, 1, t).map(|p| p[0]);
        assert_eq!(m.read_plane(&ctx(100, 0), s, t), want, "stream {s} cycle {t}");
    }
    assert_eq!(m.read_plane(&ctx(100, 0), 5, 32), Ok(v));
    assert_eq!(m.read_plane(&ctx(120, 0), 3, 5), Err(Code::Dir001)); // a unit with no read gap on the row
}
