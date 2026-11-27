//! Tests of `slices/mem`, moved from the module's in-source test block (they use the crate's public API).
//! Ports of the MEM and AGT unit tests of the Sail model (`unit_tests/test_dispatch.sail`,
//! `test_slices.sail`: `test_dispatch_ref_paths`, `test_dispatch_ref_bank`,
//! `test_dispatch_config_fields`, `test_dispatch_agt_per_path`, `test_agt_encodings`,
//! `test_gather_scatter_one_map`, `test_setsegment`; the AGT counters themselves are ported in
//! `driver::agt`), the worked AGT examples end to end, and the ISS's own
//! tests of the per-tile effects, the bank rule and the slots.
use ots_isa::Insn;
use tsp_iss::geom::MAX_LANES;
use tsp_iss::observe::Observer;
use tsp_iss::slices::mem::*;
use tsp_iss::slices::{Exec, Issue};
use tsp_iss::srf::Vec320;
use tsp_iss::{Diag, Machine};

use ots_isa::{encode, Isa};
use ots_profile::{Latency, Profile};
use tsp_iss::driver::agt::{Agt, AgtState};
use tsp_iss::driver::classify::{ref_path, ref_thread};
use tsp_iss::layout::{read_gap, write_gap, RowId, UnitKind};
use tsp_iss::links::NoLinks;

fn machine(name: &str) -> Machine {
    let mut m = Machine::new(&Profile::load(name).unwrap(), 0, Box::new(NoLinks)).unwrap();
    m.edit_state(|state| state.activity.set_enabled(true)).unwrap(); // the tests read the activity counters, which are off unless asked for
    m
}

fn ctx(q: usize, cycle: u64) -> Issue {
    Issue {
        cycle,
        queue: q,
        element: 0,
        addr_offset: 0,
        stream_offset: 0,
        thread: 0,
        lat: Latency { d_func: 5, d_skew: 0 },
    }
}

#[derive(Default)]
struct Rec {
    srf: Vec<(u64, RowId, u8, u8, Vec<u8>)>,
    sram: Vec<(u64, u8, u32, Vec<u8>)>,
    diags: Vec<(u64, u16, Diag)>,
}

impl Observer for Rec {
    fn srf(&mut self, c: u64, row: tsp_iss::observe::Row, s: u8, g: u8, v: &[u8]) {
        let r = if row == tsp_iss::observe::Row::East { RowId::E } else { RowId::W };
        self.srf.push((c, r, s, g, v.to_vec()));
    }
    fn sram(&mut self, c: u64, slice: u8, addr: u32, v: &[u8]) {
        self.sram.push((c, slice, addr, v.to_vec()));
    }
    fn diag(&mut self, c: u64, queue: u16, code: Diag) {
        self.diags.push((c, queue, code));
    }
}

fn units(i: &Insn) -> Vec<u32> {
    let w = encode(i);
    if i.entry().width == 64 {
        vec![(w >> 32) as u32, w as u32]
    } else {
        vec![w as u32]
    }
}

/// Text for queue `q` from `(cycle, instruction)` records, a bundle for the records of one cycle.
fn load_at(m: &mut Machine, q: u16, recs: &[(u64, Insn)]) {
    let mut us: Vec<u32> = Vec::new();
    let mut last: i64 = -1;
    let mut k = 0;
    while k < recs.len() {
        let c = recs[k].0;
        let mut j = k;
        while j < recs.len() && recs[j].0 == c {
            j += 1;
        }
        if c as i64 - last - 1 > 0 {
            us.extend(units(&Insn::NOP { n: (c as i64 - last - 1) as u32 }));
        }
        for (n, (_, i)) in recs[k..j].iter().enumerate() {
            let mut u = units(i);
            if n + 1 < j - k {
                u[0] |= 0x8000_0000;
            }
            us.extend(u);
        }
        last = c as i64;
        k = j;
    }
    m.load_image(q, &us).unwrap();
}

fn vec_of(f: impl Fn(usize) -> u8) -> Vec320 {
    let mut v = [0u8; MAX_LANES];
    for (l, b) in v.iter_mut().enumerate().take(320) {
        *b = f(l);
    }
    v
}

fn word(m: &Machine, vl: usize, f: impl Fn(usize) -> u8) -> Vec<u8> {
    let mut v = vec_of(&f);
    v[vl..].fill(0);
    let _ = m;
    v[..vl].to_vec()
}

const E0: u32 = 0;
const W16: u32 = 32 + 16;

// model: unit_tests/test_dispatch.sail test_dispatch_ref_paths
#[test]
fn test_dispatch_ref_paths() {
    let (a, s) = (1, 0);
    assert_eq!(ref_path(&Insn::Read { addr: a, dst: s, thread: 0 }), Some(0));
    assert_eq!(ref_path(&Insn::Gather { dst: s, map: s, k: 0, thread: 0 }), Some(0));
    assert_eq!(ref_path(&Insn::Write { addr: a, src: s, thread: 0 }), Some(1));
    assert_eq!(ref_path(&Insn::Scatter { src: s, map: s, k: 0, thread: 1 }), Some(1));
    assert_eq!(ref_path(&Insn::Countdown { d: 1 }), None);
    assert_eq!(ref_path(&Insn::Ifetch { stream: s }), None);
    assert_eq!(ref_thread(&Insn::Scatter { src: s, map: s, k: 0, thread: 1 }), 1);
    assert_eq!(ref_thread(&Insn::Read { addr: a, dst: s, thread: 0 }), 0);
}

// model: slices/mem/mem_insts.sail is_loopable
#[test]
fn only_the_references_are_loopable() {
    for id in ots_isa::EntryId::ALL {
        let loopable = matches!(
            id,
            ots_isa::EntryId::Read | ots_isa::EntryId::Write | ots_isa::EntryId::Gather | ots_isa::EntryId::Scatter
        );
        assert_eq!(ots_isa::entry(id).agt_loopable, loopable, "{id:?}");
    }
    assert!(is_loopable(&Insn::Read { addr: 0, dst: 0, thread: 0 }));
    assert!(!is_loopable(&Insn::SetSegment { n: 16 }));
}

// model: unit_tests/test_dispatch.sail test_dispatch_ref_bank
#[test]
fn test_dispatch_ref_bank() {
    let m = machine("ref");
    let lo = Insn::Read { addr: 0, dst: 0, thread: 0 };
    let bank = 1i64 << m.geom().bank_bit;
    assert_eq!(m.ref_bank(&lo, 0), BankUse::Of(0));
    assert_eq!(m.ref_bank(&lo, bank), BankUse::Of(1));
    // An address outside the slice uses no bank, so it meets no other access
    assert_eq!(m.ref_bank(&lo, -1), BankUse::None);
    assert_eq!(m.ref_bank(&lo, m.geom().words_per_slice as i64), BankUse::None);
    assert!(!banks_meet(m.ref_bank(&lo, m.geom().words_per_slice as i64), BankUse::Any));
    assert_eq!(m.ref_bank(&Insn::Gather { dst: 0, map: 0, k: 0, thread: 0 }, 0), BankUse::Any);
    assert!(
        banks_meet(BankUse::Any, BankUse::Of(1))
            && banks_meet(BankUse::Of(1), BankUse::Of(1))
            && !banks_meet(BankUse::Of(0), BankUse::Of(1))
    );
    // the bank bit follows the profile
    let m = machine("mini");
    assert_eq!((m.addr_bank(127), m.addr_bank(128), m.addr_bank(256)), (BankUse::Of(0), BankUse::Of(1), BankUse::None));
}

// model: unit_tests/test_dispatch.sail test_dispatch_agt_per_path
#[test]
fn test_dispatch_agt_per_path() {
    let mut m = machine("ref");
    assert!(m.driver().agt_get(5, 0).state == AgtState::Idle && m.driver().agt_get(5, 1).state == AgtState::Idle);
    m.edit_state(|state| {
        state.driver.agt_set(5, 1, Agt { state: AgtState::Run, next: 40, ..state.driver.agt_get(5, 1) })
    })
    .unwrap();
    assert_eq!(m.driver().agt_get(5, 0).state, AgtState::Idle);
    assert!(!m.quiescent());
    assert_eq!(m.next_event(10), Some(40));
    m.edit_state(|state| state.driver.agt_set(5, 1, Agt { state: AgtState::Config, ..state.driver.agt_get(5, 1) }))
        .unwrap();
    assert_eq!(m.next_event(10), Some(11));
    m.reset();
    assert!(m.driver().agt_get(5, 1).state == AgtState::Idle && m.quiescent());
    assert!(!m.driver().path_busy(5, 0, 0) && !m.driver().path_busy(5, 1, 0));
}

// model: unit_tests/test_slices.sail test_agt_encodings, test_gather_scatter_one_map
#[test]
fn the_mem_encodings_round_trip_and_reject_reserved_bits() {
    let m = machine("ref");
    let isa: &Isa = m.isa();
    let ill = |w: u64| isa.decode(64, w).is_none();
    let st = encode(&Insn::Step { a: 0x18 });
    assert_eq!((st >> 60, (st >> 54) & 63, (st >> 16) & 0xffff), (1, 0b000110, 0x18));
    assert_eq!(isa.decode(64, st), Some(Insn::Step { a: 0x18 }));
    assert!(!ill(encode(&Insn::Step { a: 0xFFFF })), "step -1");
    assert!(ill((1 << 60) | (0b000110 << 54) | (0x2000 << 16)), "step 8192 is out of range");
    assert_eq!(isa.decode(64, encode(&Insn::Iterations { n: 0x6F })), Some(Insn::Iterations { n: 0x6F }));
    // ReadWrite is gone: opcode 2 does not decode
    assert!(ill((1 << 60) | (0b000010 << 54) | (0x1000 << 32) | (0b000001 << 26)));
    // Gather and Scatter: one map stream; k in 51..48, thread in 53, bit 52 reserved
    let g = encode(&Insn::Gather { dst: 0b000011, map: 0b100101, k: 0, thread: 0 });
    assert_eq!(((g >> 42) & 63, (g >> 36) & 63, g & ((1 << 36) - 1)), (0b000011, 0b100101, 0));
    assert!(ill(g | 1));
    assert_eq!(isa.decode(64, g), Some(Insn::Gather { dst: 0b000011, map: 0b100101, k: 0, thread: 0 }));
    let g2 = encode(&Insn::Scatter { src: 0b000011, map: 0b100101, k: 0xA, thread: 1 });
    assert!((g2 >> 53) & 1 == 1 && (g2 >> 48) & 15 == 0xA);
    assert!(ill(g2 | (1 << 52)));
    let wr = encode(&Insn::Write { addr: 1, src: 0, thread: 1 });
    assert!((wr >> 53) & 1 == 1 && isa.decode(64, wr) == Some(Insn::Write { addr: 1, src: 0, thread: 1 }));
    // StreamStep s = 0..31
    let ss = encode(&Insn::StreamStep { s: 2 });
    assert_eq!(((ss >> 54) & 63, (ss >> 16) & 0xffff), (0b001011, 2));
    assert!(ill((ss & !(0xffffu64 << 16)) | (0x20 << 16)));
    // SetSegment n <= 8192, reserved bits zero
    assert!(ill(encode(&Insn::SetSegment { n: 0x2000 }) | 0x1_0000));
}

// model: unit_tests/test_slices.sail test_setsegment
#[test]
fn test_setsegment() {
    let mut m = machine("ref");
    let mut o = Rec::default();
    assert_eq!(m.mem_segments()[0], 8192);
    assert_eq!(m.execute(&Insn::SetSegment { n: 16 }, &ctx(0, 10), &mut o), Exec::Issued);
    assert_eq!(m.mem_segments()[0], 16);
    assert_eq!(m.mem_segments()[1], 8192, "per slice");
    // a reference at or beyond the segment is OPR-001
    assert_eq!(
        m.execute(&Insn::Read { addr: 16, dst: 0, thread: 0 }, &ctx(0, 11), &mut o),
        Exec::Diagnostic(Diag::Opr001)
    );
    assert_eq!(m.execute(&Insn::Read { addr: 15, dst: 0, thread: 0 }, &ctx(0, 11), &mut o), Exec::Issued);
    // the segment is limited to the slice, and reset restores it
    assert_eq!(m.execute(&Insn::SetSegment { n: 8192 }, &ctx(44, 12), &mut o), Exec::Issued);
    assert_eq!(m.mem_segments()[44], 8192);
    let mut m = machine("mini");
    assert_eq!(m.execute(&Insn::SetSegment { n: 4096 }, &ctx(3, 0), &mut o), Exec::Issued);
    assert_eq!(m.mem_segments()[3], 256, "min(n, words_per_slice)");
    m.execute(&Insn::SetSegment { n: 10 }, &ctx(3, 0), &mut o);
    m.reset();
    assert_eq!(m.mem_segments()[3], 256);
}

// the AGT records, Power instructions: no effect on their own
#[test]
fn agt_records_and_power_instructions_execute_as_no_ops() {
    let mut m = machine("mini");
    let mut o = Rec::default();
    for i in [
        Insn::Countdown { d: 1 },
        Insn::Step { a: 3 },
        Insn::StreamStep { s: 1 },
        Insn::Iterations { n: 2 },
        Insn::PowerConfig { v: 7 },
        Insn::DeepSleep,
    ] {
        assert_eq!(m.execute(&i, &ctx(5, 3), &mut o), Exec::Issued, "{i:?}");
    }
    assert!(o.srf.is_empty() && o.sram.is_empty() && o.diags.is_empty());
    assert_eq!(m.driver().pending.iter().map(|s| s.len()).sum::<usize>(), 0);
}

#[test]
fn a_read_commits_its_word_at_the_slice_write_gap_and_tiles_follow_the_thread_split() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    let w = word(&m, vl, |l| 1 + (l % 200) as u8);
    m.preload(4, 9, &w[..m.geom().vlanes as usize]).unwrap();
    let mut o = Rec::default();
    let g = write_gap(m.geom(), UnitKind::MemW, 4, RowId::E).unwrap();
    // no split: every tile
    assert_eq!(m.execute(&Insn::Read { addr: 9, dst: E0, thread: 0 }, &ctx(4, 20), &mut o), Exec::Issued);
    m.commit_due(25, &mut o);
    assert_eq!(o.srf.len(), 1);
    assert_eq!((o.srf[0].0, o.srf[0].1, o.srf[0].2, o.srf[0].3), (25, RowId::E, 0, g));
    assert_eq!(&o.srf[0].4[..], &w[..vl]);
    // Config SL = 7 (tiles 0..6 are thread 0, the rest thread 1): thread 1 sees tiles 7.. only
    let mut o = Rec::default();
    m.edit_state(|state| state.masks.mem_thread_split[4] = 7).unwrap();
    m.execute(&Insn::Read { addr: 9, dst: E0 | 1, thread: 1 }, &ctx(4, 30), &mut o);
    m.commit_due(35, &mut o);
    let got = &o.srf[0].4;
    assert!(
        got[..7 * 16].iter().all(|b| *b == 0) && got[7 * 16..] == w[7 * 16..vl],
        "thread 1: the tiles from the split"
    );
    // SetVL 10: tiles from 10 are inactive in both threads
    let mut o = Rec::default();
    m.edit_state(|state| state.masks.queue_vl[4] = 10).unwrap();
    m.execute(&Insn::Read { addr: 9, dst: E0 | 2, thread: 1 }, &ctx(4, 40), &mut o);
    m.commit_due(45, &mut o);
    let got = &o.srf[0].4;
    assert!(
        got[..7 * 16].iter().all(|b| *b == 0)
            && got[7 * 16..10 * 16] == w[7 * 16..10 * 16]
            && got[10 * 16..].iter().all(|b| *b == 0)
    );
}

#[test]
fn a_write_replaces_only_the_active_tiles_and_reads_its_stream_at_the_execute_cycle() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    let old = word(&m, vl, |_| 0xaa);
    m.preload(0, 5, &old[..m.geom().vlanes as usize]).unwrap();
    let rg = read_gap(m.geom(), UnitKind::MemW, 0, RowId::W).unwrap();
    // the stream W:16 reaches slice 0's read gap in cycle 50: a writer one register upstream (gaps grow along the flow on both rows)
    let mut o = Rec::default();
    m.edit_state(|state| state.srf.write(RowId::W, 16, rg - 1, 49, &vec_of(|_| 0x55))).unwrap();
    let ctx0 = ctx(0, 50);
    assert_eq!(m.execute(&Insn::Write { addr: 5, src: W16, thread: 0 }, &ctx0, &mut o), Exec::Issued);
    assert_eq!(o.sram.len(), 1);
    assert_eq!(m.sram_word(0, 5).unwrap()[..vl], vec_of(|_| 0x55)[..vl]);
    // a Write with only thread 1's tiles (split 7) keeps tiles 0..6
    m.preload(0, 6, &old[..m.geom().vlanes as usize]).unwrap();
    m.edit_state(|state| state.masks.mem_thread_split[0] = 7).unwrap();
    m.edit_state(|state| state.srf.write(RowId::W, 16, rg - 1, 59, &vec_of(|_| 0x33))).unwrap();
    assert_eq!(m.execute(&Insn::Write { addr: 6, src: W16, thread: 1 }, &ctx(0, 60), &mut o), Exec::Issued);
    let got = m.sram_word(0, 6).unwrap();
    assert!(got[..7 * 16].iter().all(|b| *b == 0xaa) && got[7 * 16..vl].iter().all(|b| *b == 0x33));
    // a missing operand is TIM-001 and writes nothing; an address outside the slice is OPR-001
    assert_eq!(
        m.execute(&Insn::Write { addr: 7, src: W16, thread: 0 }, &ctx(0, 70), &mut o),
        Exec::Diagnostic(Diag::Tim001)
    );
    assert_eq!(m.sram_word(0, 7).unwrap(), [0; MAX_LANES]);
    let mut mm = machine("mini");
    assert_eq!(
        mm.execute(&Insn::Write { addr: 256, src: W16, thread: 0 }, &ctx(0, 9), &mut o),
        Exec::Diagnostic(Diag::Opr001)
    );
}

#[test]
fn stream_and_address_offsets_are_range_checked_without_wrapping() {
    let mut m = machine("mini");
    let mut o = Rec::default();
    m.preload(2, 255, &[7; 16]).unwrap();
    let at = |off: i64, soff: i64| Issue { addr_offset: off, stream_offset: soff, ..ctx(2, 5) };
    let rd = |a: u32, s: u32| Insn::Read { addr: a, dst: s, thread: 0 };
    assert_eq!(m.execute(&rd(250, E0), &at(5, 0), &mut o), Exec::Issued, "255 is the last word");
    assert_eq!(m.execute(&rd(250, E0), &at(6, 0), &mut o), Exec::Diagnostic(Diag::Opr001), "256 is outside");
    assert_eq!(m.execute(&rd(3, E0), &at(-4, 0), &mut o), Exec::Diagnostic(Diag::Opr001), "no wrap below 0");
    assert_eq!(m.execute(&rd(3, E0 | 29), &at(0, 2), &mut o), Exec::Issued, "stream 31 on the same row");
    assert_eq!(
        m.execute(&rd(3, E0 | 29), &at(0, 3), &mut o),
        Exec::Diagnostic(Diag::Opr001),
        "stream 32 does not exist"
    );
    assert_eq!(m.execute(&rd(3, W16), &at(0, -17), &mut o), Exec::Diagnostic(Diag::Opr001), "stream -1");
    // StreamStep keeps the row: W:16 + 2 is W:18
    let g = write_gap(m.geom(), UnitKind::MemW, 2, RowId::W).unwrap();
    m.execute(&rd(3, W16), &at(0, 2), &mut o);
    m.commit_due(10, &mut o);
    assert_eq!(o.srf.iter().find(|x| x.1 == RowId::W).map(|x| (x.1, x.2, x.3)), Some((RowId::W, 18, g)));
}

// Each tile of a Gather acts alone
#[test]
fn a_gather_drives_zero_on_a_bad_tiles_lane_and_writes_the_good_lanes() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    // slice 0 holds, at address 40 + g, the word whose every lane is 0x10 + g
    for g in 0..20usize {
        m.preload(0, 40 + g as u32, &word(&m, vl, |l| 0x10 + (l / 16) as u8)).unwrap();
    }
    // the map word: tile g takes address 40 + g, except tile 3 (outside the slice: 0x1FFF is 8191, in; use the segment)
    m.edit_state(|state| state.mem_segment[0] = 60).unwrap();
    let map = vec_of(|l| match l % 16 {
        0 => (40 + l / 16) as u8,
        1 => 0,
        _ => 0,
    });
    let mut map = map;
    map[16 * 3] = 0x70; // address 112 >= segment 60: tile 3 is bad
    let rg = read_gap(m.geom(), UnitKind::MemW, 0, RowId::E).unwrap();
    m.edit_state(|state| state.srf.write(RowId::E, 1, rg - 1, 29, &map)).unwrap();
    let mut o = Rec::default();
    let k = 5;
    let r = m.execute(&Insn::Gather { dst: E0 | 2, map: E0 | 1, k, thread: 0 }, &ctx(0, 30), &mut o);
    assert_eq!(r, Exec::Diagnostic(Diag::Opr001), "one OPR-001 for any bad tile");
    m.commit_due(35, &mut o);
    let out = &o.srf[0].4;
    for g in 0..20usize {
        for l in 0..16usize {
            let want = if l == k as usize && g != 3 { 0x10 + g as u8 } else { 0 };
            assert_eq!(out[16 * g + l], want, "tile {g} lane {l}");
        }
    }
}

// Each tile of a Scatter acts alone, in ascending order
#[test]
fn a_scatter_writes_the_good_tiles_and_reports_one_opr001() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    m.edit_state(|state| state.mem_segment[0] = 60).unwrap();
    for g in 0..20u32 {
        m.preload(0, 40 + g, &word(&m, vl, |_| 0xee)).unwrap();
    }
    let mut map = vec_of(|l| if l % 16 == 0 { (40 + l / 16) as u8 } else { 0 });
    map[16 * 3] = 0x70; // tile 3 bad
    map[16 * 7] = 0x70; // tile 7 bad
    let src = vec_of(|l| 1 + (l / 16) as u8);
    let rg = read_gap(m.geom(), UnitKind::MemW, 0, RowId::E).unwrap();
    m.edit_state(|state| state.srf.write(RowId::E, 1, rg - 1, 29, &map)).unwrap();
    m.edit_state(|state| state.srf.write(RowId::E, 2, rg - 1, 29, &src)).unwrap();
    let mut o = Rec::default();
    let k = 2;
    let r = m.execute(&Insn::Scatter { src: E0 | 2, map: E0 | 1, k, thread: 0 }, &ctx(0, 30), &mut o);
    assert_eq!(r, Exec::Diagnostic(Diag::Opr001));
    let addrs: Vec<u32> = o.sram.iter().map(|x| x.2).collect();
    let want: Vec<u32> = (0..20u32).filter(|g| *g != 3 && *g != 7).map(|g| 40 + g).collect();
    assert_eq!(addrs, want, "one write per good tile, ascending");
    let w = m.sram_word(0, 40).unwrap();
    assert_eq!((w[2], w[0], w[3]), (1, 0xee, 0xee), "lane 16g + k of the word, the rest unchanged");
    assert_eq!(m.sram_word(0, 43).unwrap()[..vl], vec_of(|_| 0xee)[..vl], "the bad tile wrote nothing");
    // an inactive tile checks no address and writes nothing: SetVL 3 leaves tiles 0..2 only
    let mut m2 = machine("ref");
    m2.edit_state(|state| state.masks.queue_vl[0] = 3).unwrap();
    m2.edit_state(|state| state.srf.write(RowId::E, 1, rg - 1, 29, &map)).unwrap();
    m2.edit_state(|state| state.srf.write(RowId::E, 2, rg - 1, 29, &src)).unwrap();
    let mut o2 = Rec::default();
    let r = m2.execute(&Insn::Scatter { src: E0 | 2, map: E0 | 1, k, thread: 0 }, &ctx(0, 30), &mut o2);
    assert_eq!(o2.sram.len(), 3, "tiles 0..2");
    assert_eq!(r, Exec::Issued, "the bad tiles 3 and 7 are inactive: no address check");
}

// ---- the cycle driver: slots, loops and the bank rule ------------------------------------------

fn tick_all(m: &mut Machine, o: &mut Rec) {
    m.run(100_000, o).unwrap();
}

const BANK_WORDS: u32 = 4096; // the ref profile's bank bit is 12

fn read_i(addr: u32, dst: u32, thread: u32) -> Insn {
    Insn::Read { addr, dst, thread }
}
fn write_i(addr: u32, src: u32, thread: u32) -> Insn {
    Insn::Write { addr, src, thread }
}

/// A worked AGT example: Read at T, Countdown 1 at T+1, Step 2 at T+2, Iterations 3 at
/// T+3: references at T, T+2, T+4 (period 1 + countdown; the loop runs while it is configured).
#[test]
fn an_agt_read_loop_references_at_the_countdown_period_with_the_step() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    for k in 0..3u32 {
        m.preload(4, 16 + 2 * k, &word(&m, vl, |_| k as u8 + 1)).unwrap();
    }
    load_at(
        &mut m,
        4,
        &[
            (10, read_i(16, E0, 0)),
            (11, Insn::Countdown { d: 1 }),
            (12, Insn::Step { a: 2 }),
            (13, Insn::Iterations { n: 3 }),
        ],
    );
    let mut o = Rec::default();
    tick_all(&mut m, &mut o);
    let d = m.latency_of_instr(&read_i(0, 0, 0)).d_func as u64;
    let seen: Vec<(u64, u8)> = o.srf.iter().map(|x| (x.0 - d, x.4[0])).collect();
    assert_eq!(seen, vec![(10, 1), (12, 2), (14, 3)]);
    assert!(o.diags.is_empty(), "{:?}", o.diags);
    assert!(m.quiescent());
}

/// Two levels: inner 3 references step 2, outer 2 references step 10, countdowns 1 and 4:
/// offsets 0, 2, 4 | 14, 16, 18 and periods 2, 2, 6, 2, 2 (the Sail test of the counters, end to end).
#[test]
fn two_nested_levels_give_the_documented_offsets_and_cycles() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    for a in [0u32, 2, 4, 14, 16, 18] {
        m.preload(4, a, &word(&m, vl, |_| a as u8 + 1)).unwrap();
    }
    let recs = [
        (10, read_i(0, E0, 0)),
        (11, Insn::Countdown { d: 1 }),
        (12, Insn::Step { a: 2 }),
        (13, Insn::Iterations { n: 3 }),
        (14, Insn::Countdown { d: 4 }),
        (15, Insn::Step { a: 10 }),
        (16, Insn::Iterations { n: 2 }),
    ];
    load_at(&mut m, 4, &recs);
    let mut o = Rec::default();
    tick_all(&mut m, &mut o);
    let d = m.latency_of_instr(&read_i(0, 0, 0)).d_func as u64;
    let seen: Vec<(u64, u8)> = o.srf.iter().map(|x| (x.0 - d, x.4[0])).collect();
    // the references use the fields received so far: 10 (offset 0), 12 (2), then the second level's
    // fields arrive while the first level still runs; the full nest takes over from there
    assert_eq!(seen.len(), 6, "{seen:?}");
    let offs: Vec<u8> = seen.iter().map(|x| x.1 - 1).collect();
    assert_eq!(offs, vec![0, 2, 4, 14, 16, 18]);
    assert!(o.diags.is_empty(), "{:?}", o.diags);
}

/// A read loop (step 24) and a write loop (step 1) of 111 iterations each on one slice: a pair's
/// records alternate R, W, R, W; each loop has 112
/// references (Iterations n makes n references).
#[test]
fn the_us12340300_pair_runs_two_112_reference_loops_on_one_slice() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    let t0 = 40u64;
    for k in 0..112u32 {
        m.preload(43, 0x1000 + 24 * k, &word(&m, vl, |_| (k % 250) as u8 + 1)).unwrap();
    }
    let recs = [
        (t0, read_i(0x1000, E0, 0)),
        (t0, write_i(0x00FF, W16, 0)),
        (t0 + 1, Insn::Step { a: 24 }),
        (t0 + 2, Insn::Step { a: 1 }),
        (t0 + 3, Insn::Iterations { n: 112 }),
        (t0 + 4, Insn::Iterations { n: 112 }),
    ];
    load_at(&mut m, 43, &recs);
    // the write's source stream W:16: west slice 1 (quad 0, W gap 24) Reads address k 15 cycles
    // before the k-th write reference (slice 43, quad 10, reads W at 34: 10 registers away, plus
    // the Read's d_func of 5)
    let wr_exp: Vec<u64> = [t0, t0 + 2].into_iter().chain(t0 + 4..t0 + 114).collect();
    let mut feed = Vec::new();
    for (k, c) in wr_exp.iter().enumerate() {
        m.preload(1, k as u32, &word(&m, vl, |_| (k % 200) as u8 + 3)).unwrap();
        feed.push((c - 15, read_i(k as u32, W16, 0)));
    }
    load_at(&mut m, 1, &feed);
    let mut o = Rec::default();
    tick_all(&mut m, &mut o);
    let d = m.latency_of_instr(&read_i(0, 0, 0)).d_func as u64;
    let rd_exp: Vec<u64> = [t0, t0 + 1].into_iter().chain(t0 + 3..t0 + 113).collect();
    let rd: Vec<(u64, u8)> = o.srf.iter().filter(|x| x.1 == RowId::E).map(|x| (x.0 - d, x.4[0])).collect();
    assert_eq!(rd.iter().map(|x| x.0).collect::<Vec<_>>(), rd_exp);
    assert!(rd.iter().enumerate().all(|(k, x)| x.1 == (k % 250) as u8 + 1), "the k-th reference reads 0x1000 + 24k");
    assert_eq!(
        o.sram.iter().map(|x| (x.0, x.2)).collect::<Vec<_>>(),
        wr_exp.iter().enumerate().map(|(k, c)| (*c, 0xFF + k as u32)).collect::<Vec<_>>()
    );
    for k in 0..112u32 {
        assert_eq!(m.sram_word(43, 0xFF + k).unwrap()[0], (k % 200) as u8 + 3);
    }
    assert!(o.diags.is_empty(), "{:?}", o.diags);
    // the two paths hold themselves apart: the records of the order of the pair do not matter
    let mut m2 = machine("ref");
    for k in 0..112u32 {
        m2.preload(43, 0x1000 + 24 * k, &word(&m2, vl, |_| (k % 250) as u8 + 1)).unwrap();
    }
    let mut swapped = recs;
    swapped.swap(0, 1);
    load_at(&mut m2, 43, &swapped);
    let mut o2 = Rec::default();
    tick_all(&mut m2, &mut o2);
    assert_eq!(
        o2.srf.iter().filter(|x| x.1 == RowId::E).map(|x| x.0).collect::<Vec<_>>(),
        rd.iter().map(|x| x.0 + d).collect::<Vec<_>>()
    );
}

#[test]
fn a_stray_record_and_a_fifth_level_are_opr001() {
    // no loop open: a configuration record is OPR-001 at its dispatch
    let mut m = machine("mini");
    load_at(&mut m, 4, &[(3, Insn::Step { a: 1 })]);
    let mut o = Rec::default();
    tick_all(&mut m, &mut o);
    assert_eq!(o.diags, vec![(3, 4, Diag::Opr001)]);
    // a fifth level aborts the loop with OPR-001 at its record's cycle
    let mut m = machine("mini");
    let mut recs = vec![(5u64, read_i(0, E0, 0))];
    for k in 0..5u64 {
        recs.push((6 + k, Insn::Countdown { d: 1 }));
    }
    load_at(&mut m, 4, &recs);
    let mut o = Rec::default();
    tick_all(&mut m, &mut o);
    assert_eq!(o.diags, vec![(10, 4, Diag::Opr001)]);
    assert!(m.quiescent());
}

#[test]
fn two_references_in_one_slot_are_que001_and_none_runs() {
    let mut m = machine("mini");
    load_at(&mut m, 4, &[(5, read_i(0, E0, 0)), (5, read_i(1, E0 | 1, 0))]);
    let mut o = Rec::default();
    tick_all(&mut m, &mut o);
    assert_eq!(o.diags, vec![(5, 4, Diag::Que001), (5, 4, Diag::Que001)]);
    assert!(o.srf.is_empty());
    // a Read and a Write of the same thread are two slots and issue together
    let mut m = machine("mini");
    load_at(&mut m, 4, &[(5, read_i(0, E0, 0)), (5, write_i(130, E0 | 1, 0))]);
    let mut o = Rec::default();
    tick_all(&mut m, &mut o);
    assert!(!o.diags.contains(&(5, 4, Diag::Que001)), "{:?}", o.diags);
}

/// The bank rule (MEM-001): R and W references of one thread in one cycle meet in a bank. Each is
/// MEM-001 and none runs; different banks, or different threads, issue together.
#[test]
fn the_bank_rule_blocks_a_pair_in_one_bank_of_one_thread() {
    // same bank, same thread
    let mut m = machine("ref");
    load_at(&mut m, 4, &[(10, read_i(0, E0, 0)), (10, write_i(8, E0 | 1, 0))]);
    let mut o = Rec::default();
    tick_all(&mut m, &mut o);
    assert_eq!(o.diags, vec![(10, 4, Diag::Mem001), (10, 4, Diag::Mem001)]);
    assert!(o.srf.is_empty() && o.sram.is_empty());
    // different banks: both run (the Write's operand is missing: TIM-001, at its execute cycle)
    let mut m = machine("ref");
    load_at(&mut m, 4, &[(10, read_i(0, E0, 0)), (10, write_i(BANK_WORDS, E0 | 1, 0))]);
    let mut o = Rec::default();
    tick_all(&mut m, &mut o);
    assert_eq!(o.srf.len(), 1);
    assert!(o.diags.iter().all(|d| d.2 != Diag::Mem001));
    // the same bank but different threads use different tiles: no conflict
    let mut m = machine("ref");
    load_at(&mut m, 4, &[(10, read_i(0, E0, 0)), (10, write_i(8, E0 | 1, 1))]);
    let mut o = Rec::default();
    tick_all(&mut m, &mut o);
    assert!(o.diags.iter().all(|d| d.2 != Diag::Mem001), "{:?}", o.diags);
    // an indirect reference may use any bank: Gather + Write conflict
    let mut m = machine("ref");
    load_at(
        &mut m,
        4,
        &[(10, Insn::Gather { dst: E0, map: E0 | 1, k: 0, thread: 0 }), (10, write_i(BANK_WORDS, E0 | 2, 0))],
    );
    let mut o = Rec::default();
    tick_all(&mut m, &mut o);
    assert_eq!(o.diags.iter().filter(|d| d.2 == Diag::Mem001).count(), 2, "{:?}", o.diags);
    // an address outside the slice uses no bank: the Read runs
    let mut m = machine("mini");
    load_at(&mut m, 4, &[(10, read_i(0, E0, 0)), (10, write_i(256, E0 | 1, 0))]);
    let mut o = Rec::default();
    tick_all(&mut m, &mut o);
    assert!(o.diags.iter().all(|d| d.2 != Diag::Mem001), "{:?}", o.diags);
    assert_eq!(o.srf.len(), 1);
}

/// An AGT reference is blocked (MEM-001, no effect) by a new reference of the same thread on the
/// other path in the same bank; it still holds its path and the loop advances.
#[test]
fn an_agt_reference_is_blocked_by_a_new_reference_in_its_bank() {
    let mut m = machine("ref");
    // R loop at 10 (step 1, 3 references: 10, 11, 12); a Write to the same bank arrives at 11
    load_at(
        &mut m,
        4,
        &[
            (10, read_i(0, E0, 0)),
            (11, Insn::Step { a: 1 }),
            (11, write_i(8, E0 | 1, 0)),
            (12, Insn::Iterations { n: 3 }),
        ],
    );
    let mut o = Rec::default();
    tick_all(&mut m, &mut o);
    // the cycle-11 Write is a second record in a cycle with a configuration record: the three fit
    // (one configuration, one W). The R path's reference of cycle 11 meets it in bank 0.
    let mem001: Vec<u64> = o.diags.iter().filter(|d| d.2 == Diag::Mem001).map(|d| d.0).collect();
    assert_eq!(mem001, vec![11, 11], "the Write and the loop's reference: {:?}", o.diags);
    assert!(o.sram.is_empty(), "the blocked Write has no effect");
    // the blocked reference still advanced the loop: the references of 10 and 12 commit
    let d = m.latency_of_instr(&read_i(0, 0, 0)).d_func as u64;
    assert_eq!(o.srf.iter().map(|x| x.0 - d).collect::<Vec<_>>(), vec![10, 12]);
}

#[test]
fn a_config_record_goes_to_the_path_with_fewer_records() {
    let mut m = machine("ref");
    // a Read and a Write at 10, then three records: R, W, R (the W path keeps its defaults)
    load_at(
        &mut m,
        4,
        &[
            (10, read_i(0, E0, 0)),
            (10, write_i(BANK_WORDS, E0 | 1, 0)),
            (11, Insn::Step { a: 3 }),
            (12, Insn::Step { a: 5 }),
            (13, Insn::Iterations { n: 4 }),
        ],
    );
    let mut o = Rec::default();
    m.tick(10, &mut o);
    assert_eq!((m.driver().agts[4].state, m.driver().agts_w[4].state), (AgtState::Config, AgtState::Config));
    m.tick(11, &mut o);
    assert_eq!((m.driver().agt_records[8], m.driver().agt_records[9]), (1, 0));
    m.tick(12, &mut o);
    assert_eq!((m.driver().agt_records[8], m.driver().agt_records[9]), (1, 1));
    assert_eq!((m.driver().agts[4].levels[0].step, m.driver().agts_w[4].levels[0].step), (3, 5));
    m.tick(13, &mut o);
    assert_eq!((m.driver().agt_records[8], m.driver().agt_records[9]), (2, 1));
    assert_eq!(m.driver().agts[4].levels[0].iters, 4);
}

#[test]
fn an_absent_slice_reports_que002_for_each_instruction() {
    // With fewer MEM slices the queue of an absent slice has no unit (the shipped profiles
    // have all 44; the profile is changed here)
    let mut c: serde_json::Value =
        serde_json::from_slice(include_bytes!("../../../../opentsp-arch-spec/config/ref.json")).unwrap();
    c["geometry"]["mem_slices_per_hemisphere"] = 8.into();
    let p = Profile::from_bytes("few", &serde_json::to_vec(&c).unwrap()).unwrap();
    let mut m = Machine::new(&p, 0, Box::new(NoLinks)).unwrap();
    load_at(&mut m, 9, &[(5, read_i(0, E0, 0)), (6, Insn::Step { a: 1 })]);
    load_at(&mut m, 7, &[(5, read_i(0, E0, 0))]);
    let mut o = Rec::default();
    tick_all(&mut m, &mut o);
    assert_eq!(
        o.diags,
        vec![(5, 9, Diag::Que002), (6, 9, Diag::Que002)],
        "queue 7 (present) reads, queue 9 has no unit"
    );
    assert_eq!(o.srf.len(), 1);
}

#[test]
fn repeat_re_issues_the_last_reference_but_not_a_pair() {
    let mut m = machine("ref");
    // : the spacing is d + 1; d = 0 is legal (consecutive cycles)
    load_at(&mut m, 4, &[(10, read_i(0, E0, 0)), (11, Insn::Repeat { n: 2, d: 0 })]);
    let mut o = Rec::default();
    tick_all(&mut m, &mut o);
    assert!(o.diags.is_empty(), "{:?}", o.diags);
    let d = m.latency_of_instr(&read_i(0, 0, 0)).d_func as u64;
    assert_eq!(o.srf.iter().map(|x| x.0 - d).collect::<Vec<_>>(), vec![10, 11, 12]);
    let mut m = machine("ref");
    load_at(&mut m, 4, &[(10, read_i(0, E0, 0)), (11, Insn::Repeat { n: 2, d: 1 })]);
    let mut o = Rec::default();
    tick_all(&mut m, &mut o);
    assert!(o.diags.is_empty(), "{:?}", o.diags);
    assert_eq!(o.srf.iter().map(|x| x.0 - d).collect::<Vec<_>>(), vec![10, 12, 14]);
    // a pair issued in one cycle is not repeatable
    let mut m = machine("ref");
    load_at(
        &mut m,
        4,
        &[(10, read_i(0, E0, 0)), (10, write_i(BANK_WORDS, E0 | 1, 0)), (11, Insn::Repeat { n: 1, d: 1 })],
    );
    let mut o = Rec::default();
    tick_all(&mut m, &mut o);
    assert!(o.diags.contains(&(11, 4, Diag::Res001)), "{:?}", o.diags);
}

#[test]
fn a_scatter_counts_a_read_and_a_write_per_active_tile() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    for g in 0..20u32 {
        m.preload(0, 40 + g, &word(&m, vl, |_| 0xee)).unwrap();
    }
    m.edit_state(|state| state.masks.queue_vl[0] = 3).unwrap();
    let map = vec_of(|l| if l % 16 == 0 { (40 + l / 16) as u8 } else { 0 });
    let rg = read_gap(m.geom(), UnitKind::MemW, 0, RowId::E).unwrap();
    m.edit_state(|state| state.srf.write(RowId::E, 1, rg - 1, 29, &map)).unwrap();
    m.edit_state(|state| state.srf.write(RowId::E, 2, rg - 1, 29, &vec_of(|l| 1 + (l / 16) as u8))).unwrap();
    let mut o = Rec::default();
    assert_eq!(
        m.execute(&Insn::Scatter { src: E0 | 2, map: E0 | 1, k: 2, thread: 0 }, &ctx(0, 30), &mut o),
        Exec::Issued
    );
    let s = m.activity().mem[0];
    assert_eq!((s.reads, s.writes), (1, 1), "one reference each");
    assert_eq!((s.read_bytes, s.write_bytes), (48, 48), "three tiles of 16 bytes");
    assert_eq!(s.write_events, 3, "one SRAM word write per good tile, as the trace shows them");
    assert_eq!(o.sram.len(), 3);
}
