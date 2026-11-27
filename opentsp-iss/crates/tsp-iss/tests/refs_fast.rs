//! The copy-free operand paths (`read_planes_refs`, `read_plane_ref`, `Srf::read_ref`, `write_plane_sram`, `merge_word`)
//! against the copying routines they replace (`read_planes_core`, `read_plane`, `Srf::read`, `sram_read` then
//! `write_plane`, a tile-by-tile merge into a zero vector): the same bytes, the same errors, the same commits.
use ots_isa::Code;
use ots_profile::{Latency, Profile};
use tsp_iss::geom::MAX_LANES;
use tsp_iss::layout::{read_gap, RowId, UnitKind};
use tsp_iss::links::NoLinks;
use tsp_iss::observe::NoObserver;
use tsp_iss::operands::{push_merged, Commit, CommitAt};
use tsp_iss::slices::Issue;
use tsp_iss::sram::{merge_word, tile_active};
use tsp_iss::srf::{SrfLookup, Vec320};
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

fn merge_reference(new: &[u8], vl: u64, split: u64, use_thread: bool, t: u8, sl: u64) -> Vec320 {
    let mut out = [0u8; MAX_LANES];
    for g in 0..sl {
        if tile_active(vl, split, use_thread, t, g) {
            let r = 16 * g as usize..16 * (g as usize + 1);
            out[r.clone()].copy_from_slice(&new[r]);
        }
    }
    out
}

#[test]
fn merge_word_equals_the_tile_by_tile_merge_into_zero_for_every_mask() {
    for sl in [1u64, 4, 20] {
        let new = pattern(9, 16 * sl as usize);
        for vl in 0..=sl + 1 {
            for split in 0..=sl + 1 {
                for use_thread in [false, true] {
                    for t in 0..2u8 {
                        let got = merge_word(Some(&new[..16 * sl as usize]), vl, split, use_thread, t, sl);
                        let want = merge_reference(&new, vl, split, use_thread, t, sl);
                        assert_eq!(got, want, "sl {sl} vl {vl} split {split} thread-gated {use_thread} t {t}");
                    }
                }
            }
        }
        assert_eq!(merge_word(None, 20, 20, false, 0, sl), [0u8; MAX_LANES], "an unwritten word is zero");
    }
}

#[test]
fn read_planes_refs_returns_what_read_planes_core_returns() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    let gv = read_gap(m.geom(), UnitKind::Vxm, 0, RowId::E).unwrap();
    // streams 4..8 written upstream of the VXM at cycle 30, stream 12 only downstream of it, stream 20 at the gap itself
    for (k, s) in (4u8..8).enumerate() {
        m.srf_write(RowId::E, s, gv - 2, 30, &pattern(10 + k as u8, vl), &mut NoObserver);
    }
    m.srf_write(RowId::E, 12, gv + 2, 25, &pattern(3, vl), &mut NoObserver);
    m.srf_write(RowId::E, 20, gv, 31, &pattern(4, vl), &mut NoObserver);
    m.srf_write(RowId::E, 21, gv, 31, &pattern(5, vl), &mut NoObserver);
    let c = ctx(100, 0);
    // (stream, width, cycle, chain)
    let cases = [
        (4u32, 4u8, 32i64, false),
        (4, 2, 32, false),
        (4, 1, 32, false),
        (5, 1, 32, false),
        (4, 4, 31, false),
        (4, 4, 33, false),
        (12, 1, 27, false),
        (12, 4, 27, false),
        (8, 4, 32, false),
        (6, 4, 32, false),
        (5, 4, 32, false),
        (20, 1, 32, true),
        (20, 2, 32, true),
        (20, 2, 31, true),
        (20, 4, 32, true),
        (21, 1, 32, true),
        (32 | 4, 4, 32, false),
        (0, 4, 32, false),
    ];
    for (s, w, t, chain) in cases {
        let want = m.read_planes_core(&c, s, w, t, chain);
        let got = m.read_planes_refs(&c, s, w, t, chain);
        match (&want, &got) {
            (Ok(p), Ok(r)) => {
                for b in 0..4 {
                    assert_eq!(r[b].len(), vl);
                    assert_eq!(r[b], &p[b][..vl], "stream {s} width {w} cycle {t} chain {chain} plane {b}");
                    assert!(p[b][vl..].iter().all(|&x| x == 0));
                }
            }
            (Err(a), Err(b)) => assert_eq!(a, b, "stream {s} width {w} cycle {t} chain {chain}"),
            _ => {
                panic!("stream {s} width {w} cycle {t} chain {chain}: {want:?} against {:?}", got.as_ref().map(|_| ()))
            }
        }
    }
    // a unit with no read gap on the row
    assert_eq!(m.read_planes_refs(&ctx(120, 0), 3, 1, 5, false).err(), Some(Code::Dir001));
}

#[test]
fn srf_read_ref_returns_what_read_returns() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    m.srf_write(RowId::W, 3, 10, 40, &pattern(6, vl), &mut NoObserver);
    for (g, t) in [(20u8, 50i64), (20, 49), (20, 51), (5, 50), (10, 50), (30, 70)] {
        let (look, v) = m.srf().read(RowId::W, 3, g, t);
        let (look2, r) = m.srf().read_ref(RowId::W, 3, g, t);
        assert_eq!(look, look2, "gap {g} cycle {t}");
        match r {
            Some(r) => {
                assert_eq!(look, SrfLookup::Found);
                assert_eq!(r, &v[..vl]);
            }
            None => assert_ne!(look, SrfLookup::Found),
        }
    }
}

/// The flow rule written the plain way (the scan the occupancy bitmask replaced): the reader at gap `g` in cycle `t`
/// sees the nearest write at a gap below `g` whose diagonal `gap - cycle` equals `g - t`; a write at the reader's own gap
/// is never seen. `Srf::read` now forwards to `read_ref`, so this model, not `read`, is what `read_ref` is held to.
#[test]
fn srf_read_ref_equals_the_plain_scan_down_from_the_reader_on_many_writes() {
    use std::collections::HashMap;
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    // (slot of the diagonal, gap) -> (diagonal, seed): what the 64 slots of 44 gaps hold, the newest write winning
    let mut model: HashMap<(i64, u8), (i64, u8)> = HashMap::new();
    let mut x = 0x9E37_79B9u32;
    let mut next = |n: u32| {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (x >> 8) % n
    };
    let (mut found, mut missed) = (0, 0);
    let mut seed = 0u8;
    // the machine's order: the commits of cycle c, then the reads of that cycle and of the next ones (a read reaches
    // back at most 44 cycles from the commit it sees, and is never behind the newest commit)
    for c in 0..260u64 {
        for _ in 0..next(4) {
            let g = next(44) as u8;
            seed = seed % 250 + 1;
            m.srf_write(RowId::W, 3, g, c, &pattern(seed, vl), &mut NoObserver);
            let d = g as i64 - c as i64;
            model.insert((d.rem_euclid(64), g), (d, seed));
        }
        for g in 0..44u8 {
            for t in [c as i64, c as i64 + 1, c as i64 + 17] {
                let d = g as i64 - t;
                let want =
                    (0..g).rev().find_map(|w| model.get(&(d.rem_euclid(64), w)).filter(|e| e.0 == d).map(|e| e.1));
                let (look, got) = m.srf().read_ref(RowId::W, 3, g, t);
                match (want, got) {
                    (Some(seed), Some(v)) => {
                        assert_eq!(look, SrfLookup::Found, "gap {g} cycle {t}");
                        assert_eq!(v, &pattern(seed, vl)[..vl], "gap {g} cycle {t}");
                        found += 1;
                    }
                    (None, None) => {
                        assert_ne!(look, SrfLookup::Found, "gap {g} cycle {t}");
                        missed += 1;
                    }
                    (w, r) => panic!("gap {g} cycle {t}: the model has {w:?}, read_ref {:?}", r.map(|r| r[0])),
                }
            }
        }
    }
    assert!(found > 200 && missed > 200, "the sweep must exercise both outcomes ({found} found, {missed} missed)");
}

/// The arena recycles a value 64 cycles after its commit: a long run of commits holds a few dozen vectors, not one per
/// entry ever written, and every value written in the last 44 cycles is intact when it is read.
#[test]
fn the_arena_holds_the_commits_of_the_last_64_cycles_and_not_more() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    let mut peak = 0;
    for c in 0..3000u64 {
        for k in 0..3u8 {
            // distinct streams and gaps, so no entry is overwritten and the old arena (freed on overwrite) would grow
            let (s, g) = ((c % 31) as u8, (k as u64 * 7 + c % 29) as u8 % 44);
            m.srf_write(RowId::E, s, g, c, &pattern((c % 251) as u8, vl), &mut NoObserver);
        }
        // a probe stream: gap 3 at every cycle, read 40 cycles later at gap 43 (the same diagonal), with its own bytes
        m.srf_write(RowId::E, 31, 3, c, &pattern((c % 251) as u8, vl), &mut NoObserver);
        peak = peak.max(m.srf().values_live());
        if c >= 40 {
            let (look, v) = m.srf().read_ref(RowId::E, 31, 43, c as i64);
            assert_eq!(look, SrfLookup::Found, "cycle {c}");
            assert_eq!(v.unwrap(), &pattern(((c - 40) % 251) as u8, vl)[..vl], "cycle {c}");
        }
    }
    assert!(peak <= 4 * 65 + 4, "peak {peak} values live");
}

/// A write at the reader's own gap in the same cycle (the same diagonal) is not seen by the flow rule (it is only
/// seen by the chained read).
#[test]
fn a_write_at_the_readers_own_gap_is_not_seen_by_the_flow_read() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    m.srf_write(RowId::W, 3, 10, 40, &pattern(6, vl), &mut NoObserver);
    assert_ne!(m.srf().read_ref(RowId::W, 3, 10, 40).0, SrfLookup::Found);
    assert_eq!(m.srf().read_ref(RowId::W, 3, 11, 41).0, SrfLookup::Found);
}

#[test]
fn read_plane_ref_returns_what_read_plane_returns() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    let gv = read_gap(m.geom(), UnitKind::Vxm, 0, RowId::E).unwrap();
    m.srf_write(RowId::E, 5, gv - 2, 30, &pattern(8, vl), &mut NoObserver);
    m.srf_write(RowId::E, 9, gv + 2, 25, &pattern(8, vl), &mut NoObserver);
    for (q, s, t) in [(100usize, 5u32, 32i64), (100, 5, 31), (100, 9, 27), (100, 6, 40), (100, 32 | 5, 32), (120, 3, 5)]
    {
        let want = m.read_plane(&ctx(q, 0), s, t);
        let got = Machine::read_plane_ref(m.srf(), m.geom(), &ctx(q, 0), s, t);
        match (want, got) {
            (Ok(v), Ok(r)) => assert_eq!(r, &v[..vl], "queue {q} stream {s} cycle {t}"),
            (Err(a), Err(b)) => assert_eq!(a, b, "queue {q} stream {s} cycle {t}"),
            (a, b) => panic!("queue {q} stream {s} cycle {t}: {a:?} against {:?}", b.map(|_| ())),
        }
    }
}

#[test]
fn write_plane_sram_schedules_what_sram_read_then_write_plane_schedules() {
    for (q, threads) in [(3usize, true), (20, true), (60, true)] {
        for vl in [0u8, 1, 7, 19, 20] {
            for split in [0u8, 5, 20] {
                for thread in 0..2u8 {
                    for written in [false, true] {
                        let mut a = machine("ref");
                        let mut b = machine("ref");
                        let vlanes = a.geom().vlanes as usize;
                        for m in [&mut a, &mut b] {
                            m.edit_state(|state| state.masks.queue_vl[q] = vl).unwrap();
                            if threads {
                                m.edit_state(|state| state.masks.mem_thread_split[q] = split).unwrap();
                            }
                            if written {
                                m.edit_state(|state| state.sram.write(q, 7, &pattern(5, vlanes)[..vlanes])).unwrap();
                            }
                        }
                        let c = ctx(q, thread);
                        let ra = a.write_plane_sram(&c, 4, q, 7, 77);
                        let word = b.sram_read(q, 7).unwrap();
                        let rb = b.write_plane(&c, 4, &word, 77);
                        assert_eq!(ra, rb);
                        assert_eq!(
                            a.driver().pending[77 % a.driver().pending.len()],
                            b.driver().pending[77 % b.driver().pending.len()],
                            "q {q} vl {vl} split {split} t {thread} written {written}"
                        );
                        assert_eq!(a.driver().pending_occ, b.driver().pending_occ);
                    }
                }
            }
        }
    }
}

#[test]
fn read_planes16_boxed_returns_what_read_planes16_returns() {
    // variant 0: group 16..31 complete; 1: stream 24 never written; 2: stream 18 written at another cycle, so no value reaches the unit at the read cycle
    for variant in 0..3 {
        let mut m = machine("ref");
        let vl = m.geom().vlanes as usize;
        let gm = read_gap(m.geom(), UnitKind::MxmW, 0, RowId::W).unwrap();
        for b in 0..16u8 {
            match (variant, b) {
                (1, 8) => {}
                (2, 2) => {
                    m.srf_write(RowId::W, 16 + b, gm - 2, 25, &pattern(9, vl), &mut NoObserver);
                }
                _ => {
                    m.srf_write(RowId::W, 16 + b, gm - 2, 30, &pattern(40 + b, vl), &mut NoObserver);
                }
            }
        }
        let c = ctx(104, 0);
        for (s, t) in [(48u32, 32i64), (48, 33), (48, 27), (49, 32), (32, 32), (16, 32)] {
            let want = m.read_planes16(&c, s, t);
            let got = m.read_planes16_boxed(&c, s, t);
            match (&want, &got) {
                (Ok(a), Ok(b)) => {
                    for k in 0..16 {
                        assert_eq!(a[k], b[k], "variant {variant} stream {s} cycle {t} row {k}");
                    }
                }
                (Err(a), Err(b)) => assert_eq!(a, b, "variant {variant} stream {s} cycle {t}"),
                _ => panic!(
                    "variant {variant} stream {s} cycle {t}: {:?} against {:?}",
                    want.as_ref().map(|_| ()),
                    got.as_ref().map(|_| ())
                ),
            }
        }
        // the three outcomes are reached
        match variant {
            0 => assert!(m.read_planes16(&c, 48, 32).is_ok()),
            1 => assert_eq!(m.read_planes16(&c, 48, 32).err(), Some(Code::Tim001)),
            _ => assert!(m.read_planes16(&c, 48, 32).is_err()),
        }
        assert_eq!(m.read_planes16(&c, 49, 32).err(), Some(Code::Opr002));
        // a unit with no read gap on the row
        assert_eq!(m.read_planes16_boxed(&ctx(120, 0), 48, 5).err(), Some(Code::Dir001));
    }
}

#[test]
fn ifetchm_fetches_the_stored_words_and_zeros_for_untouched_ones() {
    use tsp_iss::slices::Exec;
    for name in ["ref", "mini"] {
        let mut m = machine(name);
        let vl = m.geom().vlanes as usize;
        let n = vl / 4;
        let word = |seed: u8| -> Vec<u8> { pattern(seed, vl)[..vl].to_vec() };
        let units_of = |w: &[u8]| -> Vec<u32> {
            w.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        };
        m.preload(4, 10, &word(1)).unwrap();
        m.preload(4, 11, &word(2)).unwrap();
        m.preload(4, 21, &word(3)).unwrap();
        // (address, the two words it must fetch)
        let zero = vec![0u8; vl];
        let far = (m.geom().words_per_slice - 4) as u32; // a page nothing was written to
        let cases = [
            (10u32, word(1), word(2)),
            (20, zero.clone(), word(3)),
            (21, word(3), zero.clone()),
            (30, zero.clone(), zero.clone()),
            (far, zero.clone(), zero.clone()),
        ];
        for (index, (a, w0, w1)) in cases.into_iter().enumerate() {
            let cycle = 30 + 2 * index as u64;
            let before = m.iq().fill[4].len();
            assert_eq!(
                m.execute_cu(&ots_isa::Insn::IfetchM { addr: a }, &Issue { cycle, ..ctx(4, 0) }, &mut NoObserver),
                Exec::Issued
            );
            assert_eq!(m.iq().fill[4].len(), before, "second SRAM read has not happened");
            m.tick(cycle + 1, &mut NoObserver);
            let seg = m.iq().fill[4].back().unwrap();
            assert_eq!(m.iq().fill[4].len(), before + 1);
            let want: Vec<u32> = units_of(&w0).into_iter().chain(units_of(&w1)).collect();
            assert_eq!(seg.units.len(), 2 * n);
            assert_eq!(seg.units.iter().copied().collect::<Vec<u32>>(), want, "{name} address {a}");
            m.edit_state(|state| state.iq.fill[4].clear()).unwrap(); // inspect each payload before delivery, independently
        }
    }
}

#[test]
fn push_merged_appends_what_a_push_of_merge_word_appends() {
    let mut n = 0;
    for superlanes in [1u64, 8, 20] {
        let full = pattern(7, MAX_LANES);
        let short = pattern(9, 16 * superlanes as usize);
        for vl in 0..=superlanes + 1 {
            for split in 0..=superlanes + 1 {
                for use_thread in [false, true] {
                    for t in 0..2u8 {
                        for new in [None, Some(&short[..]), Some(&full[..])] {
                            for before in [0usize, 3] {
                                let at = CommitAt { r: RowId::W, id: 5, g: 17, q: 44, b: 2, ov: 0 };
                                let seed = |k: usize| Commit {
                                    r: RowId::E,
                                    id: k as u8,
                                    g: 1,
                                    value: pattern(k as u8, 40),
                                    q: k,
                                    b: 0,
                                    ov: 0,
                                };
                                let mut got: Vec<Commit> = (0..before).map(seed).collect();
                                let mut want = got.clone();
                                push_merged(&mut got, at, new, (vl, split, use_thread, t, superlanes));
                                want.push(Commit {
                                    r: at.r,
                                    id: at.id,
                                    g: at.g,
                                    value: merge_word(new, vl, split, use_thread, t, superlanes),
                                    q: at.q,
                                    b: at.b,
                                    ov: 0,
                                });
                                assert_eq!(
                                    got, want,
                                    "superlanes {superlanes} vl {vl} split {split} thread {use_thread}/{t}"
                                );
                                n += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(n > 1000);
}
