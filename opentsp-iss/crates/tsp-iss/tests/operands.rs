//! Tests of `operands`, moved from the module's in-source test block (they use the crate's public API).
//! Tests of `sys/operands.sail`: `aligned`, `read_planes_core`, `read_planes16`, `write_planes`,
//! `words_to_planes` (the model has no unit test of its own for them; the cases follow the
//! comments of the Sail source and the flow rule of `microarchitecture.md` §2).
use ots_isa::Code;
use ots_profile::{Latency, Profile};
use tsp_iss::geom::MAX_LANES;
use tsp_iss::layout::{read_gap, write_gap, RowId, UnitKind};
use tsp_iss::links::NoLinks;
use tsp_iss::observe::NoObserver;
use tsp_iss::operands::*;
use tsp_iss::slices::Issue;
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

fn byte_vec(b: u8, vl: usize) -> Vec320 {
    let mut x = [0u8; MAX_LANES];
    x[..vl].fill(b);
    x
}

/// Stream code as encoded: bit 5 the row.
const fn e(i: u32) -> u32 {
    i
}
const fn w(i: u32) -> u32 {
    32 | i
}

const VXM: usize = 100;

// model: sys/operands.sail aligned
#[test]
fn operands_aligned_to_their_width() {
    for id in 0..32u32 {
        assert!(aligned(e(id), 1));
        assert_eq!(aligned(w(id), 2), id % 2 == 0);
        assert_eq!(aligned(e(id), 4), id % 4 == 0 && id + 4 <= 32);
    }
    assert!(aligned(e(30), 2) && !aligned(e(31), 2) && aligned(w(28), 4) && !aligned(w(30), 4));
}

#[test]
fn operands_stream_codes() {
    assert_eq!((stream_row(e(3)), stream_id(e(3))), (RowId::E, 3));
    assert_eq!((stream_row(w(31)), stream_id(w(31))), (RowId::W, 31));
}

// model: sys/operands.sail read_planes_core
#[test]
fn operands_read_planes_flow_and_errors() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    let gv = read_gap(m.geom(), UnitKind::Vxm, 0, RowId::E).unwrap();
    assert!(gv >= 3);
    let up = gv - 3; // a writer 3 registers upstream of the VXM
    let mut o = NoObserver;
    m.srf_write(RowId::E, 4, up, 10, &byte_vec(0x11, vl), &mut o);
    m.srf_write(RowId::E, 5, up, 10, &byte_vec(0x22, vl), &mut o);
    // one stream, i8
    let p = m.read_planes(&ctx(VXM, 0), e(4), 1, 13).unwrap();
    assert_eq!(p[0], byte_vec(0x11, vl));
    assert_eq!(p[1], [0; MAX_LANES]);
    // two streams: plane b is stream 4 + b
    let p = m.read_planes(&ctx(VXM, 0), e(4), 2, 13).unwrap();
    assert_eq!((p[0], p[1], p[2]), (byte_vec(0x11, vl), byte_vec(0x22, vl), [0; MAX_LANES]));
    // a cycle too early or late: timing error, whatever the width
    assert_eq!(m.read_planes(&ctx(VXM, 0), e(4), 1, 12), Err(Code::Tim001));
    assert_eq!(m.read_planes(&ctx(VXM, 0), e(4), 2, 14), Err(Code::Tim001));
    // plane 2 and 3 are missing
    assert_eq!(m.read_planes(&ctx(VXM, 0), e(4), 4, 13), Err(Code::Tim001));
    // misaligned operand
    assert_eq!(m.read_planes(&ctx(VXM, 0), e(5), 2, 13), Err(Code::Opr002));
    assert_eq!(m.read_planes(&ctx(VXM, 0), e(30), 4, 13), Err(Code::Opr002));
    // nothing on the other row
    assert_eq!(m.read_planes(&ctx(VXM, 0), w(4), 1, 13), Err(Code::Tim001));
}

#[test]
fn operands_direction_beats_timing_and_a_unit_without_a_read_gap_is_dir() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    let gv = read_gap(m.geom(), UnitKind::Vxm, 0, RowId::E).unwrap();
    let mut o = NoObserver;
    // stream 8: the only writer is downstream of the VXM reader
    m.srf_write(RowId::E, 9, gv + 2, 25, &byte_vec(5, vl), &mut o);
    // stream 8 has nothing: TIM-001; stream 9 only downstream: DIR-001
    assert_eq!(m.read_planes(&ctx(VXM, 0), e(8), 1, 26), Err(Code::Tim001));
    assert_eq!(m.read_planes(&ctx(VXM, 0), e(9), 1, 26), Err(Code::Dir001));
    // a pair whose first plane is missing and whose second is upstream: DIR-001, in either order
    assert_eq!(m.read_planes(&ctx(VXM, 0), e(8), 2, 26), Err(Code::Dir001));
    m.srf_write(RowId::E, 10, gv - 1, 40, &byte_vec(1, vl), &mut o);
    m.srf_write(RowId::E, 11, gv + 1, 40, &byte_vec(1, vl), &mut o);
    assert_eq!(m.read_planes(&ctx(VXM, 0), e(10), 2, 41), Err(Code::Dir001), "plane 0 found, plane 1 upstream");
    // the SXM west transposer reads no E stream: DIR-001, before the alignment check
    assert_eq!(read_gap(m.geom(), UnitKind::SxmWTsp, 0, RowId::E), None);
    assert_eq!(m.read_planes(&ctx(120, 0), e(3), 1, 5), Err(Code::Dir001));
    assert_eq!(m.read_planes(&ctx(120, 0), e(3), 2, 5), Err(Code::Dir001));
    // on its own row it reads
    assert_eq!(m.read_planes(&ctx(120, 0), w(3), 1, 5), Err(Code::Tim001));
}

// model: sys/operands.sail read_planes_core (chain)
#[test]
fn operands_chained_read_takes_the_write_at_the_readers_own_gap() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    let gv = read_gap(m.geom(), UnitKind::Vxm, 0, RowId::E).unwrap();
    m.srf_write(RowId::E, 6, gv, 30, &byte_vec(0x33, vl), &mut NoObserver);
    let p = m.read_planes_core(&ctx(VXM, 0), e(6), 1, 31, true).unwrap();
    assert_eq!(p[0], byte_vec(0x33, vl));
    assert_eq!(m.read_planes_core(&ctx(VXM, 0), e(6), 1, 32, true), Err(Code::Tim001));
    // the flow rule never sees it: a writer at the reader's own gap is not upstream of it
    assert_eq!(m.read_planes_core(&ctx(VXM, 0), e(6), 1, 31, false), Err(Code::Dir001));
}

// model: sys/operands.sail read_planes16
#[test]
fn operands_read_planes16() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    let gv = read_gap(m.geom(), UnitKind::Vxm, 0, RowId::E).unwrap();
    let mut o = NoObserver;
    for b in 0..16u8 {
        m.srf_write(RowId::E, 16 + b, gv - 2, 7, &byte_vec(b + 1, vl), &mut o);
    }
    let g = m.read_planes16(&ctx(VXM, 0), e(16), 9).unwrap();
    for (b, plane) in g.iter().enumerate() {
        assert_eq!(*plane, byte_vec(b as u8 + 1, vl), "stream {}", 16 + b);
    }
    assert_eq!(m.read_planes16(&ctx(VXM, 0), e(17), 9), Err(Code::Opr002));
    assert_eq!(m.read_planes16(&ctx(VXM, 0), e(16), 10), Err(Code::Tim001));
    assert_eq!(m.read_planes16(&ctx(VXM, 0), e(0), 9), Err(Code::Tim001));
    assert_eq!(m.read_planes16(&ctx(120, 0), e(0), 9), Err(Code::Dir001));
    // a downstream writer in one of the 16 streams
    m.srf_write(RowId::E, 3, gv + 1, 20, &byte_vec(1, vl), &mut o);
    assert_eq!(m.read_planes16(&ctx(VXM, 0), e(0), 21), Err(Code::Dir001));
}

// model: sys/operands.sail write_planes
#[test]
fn operands_write_planes_schedule_commits_in_the_pending_ring() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    let mut p = NO_PLANES;
    p[0] = byte_vec(0xa0, vl);
    p[1] = byte_vec(0xa1, vl);
    m.write_planes(&ctx(VXM, 0), e(4), &p, 2, 50 + 128).unwrap();
    let gv = write_gap(m.geom(), UnitKind::Vxm, 0, RowId::E).unwrap();
    let slot = &m.driver().pending[50];
    assert_eq!(slot.len(), 2);
    // the model prepends to its list, so the ISS stores the list reversed: oldest first
    assert_eq!((slot[0].r, slot[0].id, slot[0].g, slot[0].q, slot[0].b), (RowId::E, 4, gv, VXM, 0));
    assert_eq!((slot[1].id, slot[1].b), (5, 1));
    assert_eq!(slot[0].value, byte_vec(0xa0, vl));
    assert_eq!(slot[1].value, byte_vec(0xa1, vl));
    assert_eq!(m.driver().pending.iter().map(Vec::len).sum::<usize>(), 2);
}

#[test]
fn operands_write_planes_errors_schedule_nothing() {
    let mut m = machine("ref");
    let p = one_plane(byte_vec(1, 320));
    // an MXM west queue writes row E only
    assert_eq!(m.write_planes(&ctx(104, 0), w(2), &p, 1, 9), Err(Code::Dir001));
    assert!(m.write_planes(&ctx(104, 0), e(2), &p, 1, 9).is_ok());
    // misaligned
    assert_eq!(m.write_planes(&ctx(VXM, 0), e(3), &p, 2, 9), Err(Code::Opr002));
    assert_eq!(m.write_planes(&ctx(VXM, 0), e(28), &p, 4, 9), Ok(()));
    assert_eq!(m.write_planes(&ctx(VXM, 0), e(30), &p, 4, 9), Err(Code::Opr002));
    // a unit with no write gap on the row is DIR-001 before the alignment check
    assert_eq!(m.write_planes(&ctx(104, 0), w(3), &p, 2, 9), Err(Code::Dir001));
    assert_eq!(m.driver().pending[9].len(), 1 + 4);
}

#[test]
fn operands_inactive_tiles_produce_zero() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    let p = one_plane(byte_vec(0xff, vl));
    // SetVL 3 on the VXM queue: superlanes 3.. are zero
    m.edit_state(|state| state.masks.queue_vl[VXM] = 3).unwrap();
    m.write_planes(&ctx(VXM, 0), e(0), &p, 1, 20).unwrap();
    let v = m.driver().pending[20][0].value;
    assert!(v[..48].iter().all(|b| *b == 0xff) && v[48..].iter().all(|b| *b == 0), "{v:?}");
    // a MEM queue (<= 87) applies its liquid-thread split: thread 1 writes tiles 7.. only
    m.edit_state(|state| state.masks.mem_thread_split[5] = 7).unwrap();
    m.write_planes(&ctx(5, 1), e(0), &p, 1, 21).unwrap();
    let v = m.driver().pending[21][0].value;
    assert!(v[..7 * 16].iter().all(|b| *b == 0) && v[7 * 16..vl].iter().all(|b| *b == 0xff));
    m.write_planes(&ctx(5, 0), e(1), &p, 1, 22).unwrap();
    let v = m.driver().pending[22][0].value;
    assert!(v[..7 * 16].iter().all(|b| *b == 0xff) && v[7 * 16..].iter().all(|b| *b == 0));
    // a non-MEM queue ignores the thread tag and the split
    m.edit_state(|state| state.masks.queue_vl[VXM] = 20).unwrap();
    m.edit_state(|state| state.masks.mem_thread_split[0] = 1).unwrap();
    m.write_planes(&ctx(VXM, 1), e(2), &p, 1, 23).unwrap();
    assert!(m.driver().pending[23][0].value[..vl].iter().all(|b| *b == 0xff));
}

// model: sys/operands.sail words_to_planes
#[test]
fn operands_words_to_planes_is_little_endian_per_lane() {
    let mut xs = [0u32; MAX_LANES];
    for (l, x) in xs.iter_mut().enumerate() {
        *x = (l as u32) | ((l as u32 ^ 0xff) << 8) | (0xab << 16) | (0xcd << 24);
    }
    let vl = 48;
    let pl = words_to_planes(&xs, vl);
    #[allow(clippy::needless_range_loop)] // the lane index is also the expected value
    for l in 0..vl {
        assert_eq!((pl[0][l], pl[1][l], pl[2][l], pl[3][l]), (l as u8, l as u8 ^ 0xff, 0xab, 0xcd));
    }
    assert!((vl..MAX_LANES).all(|l| (0..4).all(|b| pl[b][l] == 0)));
    assert_eq!(one_plane([7; MAX_LANES])[0], [7; MAX_LANES]);
    assert_eq!(one_plane([7; MAX_LANES])[3], [0; MAX_LANES]);
}

// A scheduled write is part of the file when the driver reaches its cycle; a read in cycle t sees
// only commits of earlier cycles.
#[test]
fn operands_a_scheduled_write_is_read_after_its_commit_cycle() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    let p = one_plane(byte_vec(0x5a, vl));
    let gw = write_gap(m.geom(), UnitKind::Vxm, 0, RowId::E).unwrap();
    m.write_planes(&ctx(VXM, 0), e(9), &p, 1, 12).unwrap();
    // before the commit nothing is in the file
    assert_eq!(m.srf_value(tsp_iss::observe::Row::East, 9, gw + 1, 13), None);
    m.tick(12, &mut NoObserver);
    assert!(m.driver().pending[12].is_empty());
    assert_eq!(m.srf_value(tsp_iss::observe::Row::East, 9, gw + 1, 13), Some(byte_vec(0x5a, vl)));
    assert_eq!(m.srf_value(tsp_iss::observe::Row::East, 9, gw + 3, 15), Some(byte_vec(0x5a, vl)));
}
