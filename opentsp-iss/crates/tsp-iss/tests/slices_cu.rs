//! Tests of `slices/cu`, moved from the module's in-source test block (they use the crate's public API).
//! Ports of the fetch tests of the Sail model (`unit_tests/test_slices.sail`:
//! `test_ifetch_one_stream`, `test_ifetch_rx_queue`, `test_ifetch_local_memory`), and the ISS's
//! own tests of the fetch path end to end: the latency, the fill rate, the padding's share of the
//! rate, queueing behind an earlier payload, IQ-001 (underflow) and IQ-002 (overflow), and the
//! conditions that reject an `IfetchM` (`unit_tests/test_frontend.sail` has the IQ-level tests,
//! ported in `frontend`; `test/directed/frontend.py` has the end-to-end ones against `tsp_sim`).
use ots_isa::Insn;
use tsp_iss::observe::Observer;
use tsp_iss::slices::{Exec, Issue};
use tsp_iss::Machine;

use ots_isa::{encode, Code};
use ots_profile::{Latency, Profile};
use tsp_iss::layout::{read_gap, RowId, UnitKind};
use tsp_iss::links::NoLinks;
use tsp_iss::observe::NoObserver;

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
        lat: Latency { d_func: 0, d_skew: 0 },
    }
}

fn zero(m: &Machine) -> tsp_iss::srf::Vec320 {
    let _ = m;
    [0u8; tsp_iss::geom::MAX_LANES]
}

fn units(i: &Insn) -> Vec<u32> {
    let w = encode(i);
    if i.entry().width == 64 {
        vec![(w >> 32) as u32, w as u32]
    } else {
        vec![w as u32]
    }
}

/// The units as the bytes of one SRAM word, padded with 0xFFFFFFFF.
fn word_of(m: &Machine, us: &[u32]) -> Vec<u8> {
    let n = m.geom().vlanes as usize / 4;
    (0..n).flat_map(|k| us.get(k).copied().unwrap_or(0xFFFF_FFFF).to_le_bytes()).collect()
}

#[derive(Default)]
struct Rec {
    diags: Vec<(u64, u16, Code)>,
    dispatches: Vec<(u64, u16)>,
    lints: Vec<(u64, u16, &'static str)>,
}

impl Observer for Rec {
    fn diag(&mut self, c: u64, q: u16, code: Code) {
        self.diags.push((c, q, code));
    }
    fn dispatch(&mut self, c: u64, q: u16) {
        self.dispatches.push((c, q));
    }
    fn lint(&mut self, c: u64, q: u16, code: &'static str) {
        self.lints.push((c, q, code));
    }
}

// model: unit_tests/test_slices.sail test_ifetch_one_stream
#[test]
fn ifetch_one_stream() {
    let vl = machine("ref").geom().vlanes as usize;
    let g = read_gap(machine("ref").geom(), UnitKind::Vxm, 0, RowId::E).unwrap();
    let z = [0u8; tsp_iss::geom::MAX_LANES];
    let e = |s: u32| Insn::Ifetch { stream: s };
    let mut m = machine("ref");
    // a VXM queue: stream E:4 written at the gap upstream of the VXM reaches it at cycle 30, and the next
    // vector of it at cycle 31 (the pair is two consecutive vectors of ONE stream, read at t and t + 1)
    let _ = m.edit_state(|state| state.srf.write(RowId::E, 4, g - 1, 29, &z[..vl])).unwrap();
    assert_eq!(m.execute_cu(&e(0b000100), &ctx(88, 30), &mut NoObserver), Exec::Issued);
    m.tick(31, &mut NoObserver);
    assert!(m.iq().fill[88].is_empty());
    let mut m = machine("ref");
    let _ = m.edit_state(|state| state.srf.write(RowId::E, 4, g - 1, 29, &z[..vl])).unwrap();
    let _ = m.edit_state(|state| state.srf.write(RowId::E, 4, g - 1, 30, &z[..vl])).unwrap();
    assert_eq!(m.execute_cu(&e(0b000100), &ctx(88, 30), &mut NoObserver), Exec::Issued);
    m.tick(31, &mut NoObserver);
    let seg = m.iq().fill[88].front().expect("the payload is staged");
    assert_eq!(
        seg.ready,
        (30 + 1 + m.geom().fetch_latency).into(),
        "the payload is complete after the second read: t + 1 + fetch_latency"
    );
    // the second vector of another stream does not make a pair
    let mut m = machine("ref");
    let _ = m.edit_state(|state| state.srf.write(RowId::E, 4, g - 1, 29, &z[..vl])).unwrap();
    let _ = m.edit_state(|state| state.srf.write(RowId::E, 5, g - 1, 30, &z[..vl])).unwrap();
    assert_eq!(m.execute_cu(&e(0b000100), &ctx(88, 30), &mut NoObserver), Exec::Issued);
    m.tick(31, &mut NoObserver);
    assert!(m.iq().fill[88].is_empty());
    // an odd stream is legal
    let mut m = machine("ref");
    let _ = m.edit_state(|state| state.srf.write(RowId::E, 5, g - 1, 29, &z[..vl])).unwrap();
    let _ = m.edit_state(|state| state.srf.write(RowId::E, 5, g - 1, 30, &z[..vl])).unwrap();
    assert_eq!(m.execute_cu(&e(0b000101), &ctx(88, 30), &mut NoObserver), Exec::Issued);
}

// model: unit_tests/test_slices.sail test_ifetch_rx_queue
#[test]
fn ifetch_rx_queue() {
    let mut m = machine("ref");
    let vl = m.geom().vlanes as usize;
    let z = zero(&m);
    let tx = read_gap(m.geom(), UnitKind::C2cWTx, 0, ROW_W).unwrap_or(1);
    let g0 = tx.saturating_sub(1);
    let e = |s: u32| Insn::Ifetch { stream: s };
    // the west C2C RX queue (128) fetches from the row arriving at the chip edge, W
    let _ = m.edit_state(|state| state.srf.write(RowId::W, 4, g0, 29, &z[..vl])).unwrap();
    let _ = m.edit_state(|state| state.srf.write(RowId::W, 4, g0, 30, &z[..vl])).unwrap();
    assert_eq!(m.execute_cu(&e(0b100100), &ctx(128, 30), &mut NoObserver), Exec::Issued);
    assert_eq!(m.execute_cu(&e(0b000100), &ctx(128, 30), &mut NoObserver), Exec::Diagnostic(Code::Dir001));
    // the east RX queue (142), row E
    let mut m = machine("ref");
    let _ = m.edit_state(|state| state.srf.write(RowId::E, 4, g0, 29, &z[..vl])).unwrap();
    let _ = m.edit_state(|state| state.srf.write(RowId::E, 4, g0, 30, &z[..vl])).unwrap();
    assert_eq!(m.execute_cu(&e(0b000100), &ctx(142, 30), &mut NoObserver), Exec::Issued);
    assert_eq!(read_gap(m.geom(), UnitKind::C2cERx, 0, RowId::E), read_gap(m.geom(), UnitKind::C2cETx, 0, RowId::E));
}

const ROW_W: RowId = RowId::W;

// model: unit_tests/test_slices.sail test_ifetch_local_memory
#[test]
fn ifetch_local_memory() {
    for name in ["ref", "mini"] {
        let mut m = machine(name);
        let wps = m.geom().words_per_slice as u32;
        let f = |a: u32| Insn::IfetchM { addr: a };
        assert_eq!(m.execute_cu(&f(0x10), &ctx(4, 30), &mut NoObserver), Exec::Issued);
        m.tick(31, &mut NoObserver);
        // the second word of the pair must be inside the slice (and the segment)
        assert_eq!(m.execute_cu(&f(wps - 1), &ctx(4, 30), &mut NoObserver), Exec::Diagnostic(Code::Opr001));
        assert_eq!(m.execute_cu(&f(wps - 2), &ctx(4, 32), &mut NoObserver), Exec::Issued);
        m.tick(33, &mut NoObserver);
        m.edit_state(|state| state.mem_segment[4] = 16).unwrap();
        assert_eq!(m.execute_cu(&f(0x0F), &ctx(4, 30), &mut NoObserver), Exec::Diagnostic(Code::Opr001));
        assert_eq!(m.execute_cu(&f(0x0E), &ctx(4, 34), &mut NoObserver), Exec::Issued);
        // only MEM queues issue it; the dispatch check is QUE-002 (and execute is not reached), the
        // execute clause itself still answers QUE-002 outside the MEM queues
        assert_eq!(m.execute_cu(&f(1), &ctx(100, 30), &mut NoObserver), Exec::Diagnostic(Code::Que002));
        assert_eq!(m.execute_cu(&f(1), &ctx(144, 30), &mut NoObserver), Exec::Diagnostic(Code::Que002));
    }
}

/// An IfetchM of word `a` at the machine's queue `q`, cycle 0; the fetched word is `us`.
fn run_ifetchm(name: &str, a: u32, us: &[u32], image_tail: &[u32]) -> (Machine, Rec) {
    let mut m = machine(name);
    let w0 = word_of(&m, us);
    m.preload(4, a, &w0).unwrap();
    m.preload(4, a + 1, &word_of(&m, &[])).unwrap();
    let mut img = units(&Insn::IfetchM { addr: a });
    img.extend_from_slice(image_tail);
    m.load_image(4, &img).unwrap();
    let mut r = Rec::default();
    m.run(1_000, &mut r).unwrap();
    (m, r)
}

#[test]
fn fetched_text_dispatches_after_the_latency_at_the_fill_rate() {
    for name in ["ref", "mini"] {
        let m0 = machine(name);
        let lat = 1 + m0.geom().fetch_latency;
        let rate = m0.iq().rate() as u64;
        // one instruction per unit (up to six): the n-th SetVL dispatches the cycle after the (n-1)-th, from
        // the cycle its unit has arrived (rate units per cycle), never before IfetchM + latency
        let k_n = 6.min(m0.geom().vlanes / 4); // a word holds vlanes / 4 units
        let text: Vec<u32> = (0..k_n as u32).flat_map(|n| units(&Insn::SetVL { n: 1 + n })).collect();
        let (_, r) = run_ifetchm(name, 1, &text, &[]);
        let ds: Vec<u64> = r.dispatches.iter().filter(|(_, q)| *q == 4).map(|(c, _)| *c).collect();
        assert_eq!(ds.len() as u64, 1 + k_n, "the IfetchM and the fetched SetVLs");
        assert_eq!(ds[0], 0);
        let mut want = Vec::new();
        let mut last = 0u64;
        for k in 0..k_n {
            let arrives = lat + k / rate;
            let c = arrives.max(last + 1);
            want.push(c);
            last = c;
        }
        assert_eq!(&ds[1..], &want[..], "{name}");
        // the first parcel completes after its earliest cycle (1) with the queue free before: the
        // lint OTS-W-IQ-001, which is no diagnostic
        assert_eq!(r.lints, vec![(lat, 4, "OTS-W-IQ-001")], "{name}");
        assert!(r.diags.is_empty(), "{name}");
    }
}

#[test]
fn padding_takes_its_share_of_the_fill_rate() {
    for name in ["ref", "mini"] {
        let m0 = machine(name);
        let lat = 1 + m0.geom().fetch_latency;
        let rate = m0.iq().rate();
        let n = m0.geom().vlanes as usize / 4;
        if rate >= n {
            continue;
        }
        // `rate` padding units fill cycle `lat`; the first instruction enters the next cycle
        let mut text = vec![0xFFFF_FFFFu32; rate];
        text.extend(units(&Insn::SetVL { n: 2 }));
        let (_, r) = run_ifetchm(name, 1, &text, &[]);
        let ds: Vec<u64> = r.dispatches.iter().filter(|(_, q)| *q == 4).map(|(c, _)| *c).collect();
        assert_eq!(ds, vec![0, lat + 1], "{name}");
        assert!(r.diags.is_empty(), "padding is no DEC-003: {:?}", r.diags);
    }
}

#[test]
fn a_second_payload_queues_behind_the_first() {
    let name = "ref";
    let mut m = machine(name);
    let n = m.geom().vlanes as usize / 4;
    let rate = m.iq().rate();
    let lat = 1 + m.geom().fetch_latency;
    m.preload(4, 1, &word_of(&m, &units(&Insn::SetVL { n: 3 }))).unwrap();
    m.preload(4, 2, &word_of(&m, &[])).unwrap();
    m.preload(4, 3, &word_of(&m, &units(&Insn::SetVL { n: 4 }))).unwrap();
    m.preload(4, 4, &word_of(&m, &[])).unwrap();
    let mut img = units(&Insn::IfetchM { addr: 1 });
    img.extend(units(&Insn::IfetchM { addr: 3 }));
    m.load_image(4, &img).unwrap();
    let mut r = Rec::default();
    m.run(1_000, &mut r).unwrap();
    let ds: Vec<u64> = r.dispatches.iter().filter(|(_, q)| *q == 4).map(|(c, _)| *c).collect();
    // IfetchM at 0 and 2 (two successive SRAM reads each); the first payload has 2n units (one SetVL, then padding, then none), the
    // second starts when the first has been taken at `rate` units per cycle
    let first_done = lat + (2 * n).div_ceil(rate) as u64;
    assert_eq!(ds[..2], [0, 2]);
    assert_eq!(ds[2], lat, "the first SetVL");
    assert_eq!(ds[3], first_done, "the second payload's SetVL enters after the first payload's last unit");
    assert_eq!(m.masks().queue_vl[4], 4.min(m.geom().superlanes) as u8);
}

#[test]
fn a_payload_that_does_not_fit_overflows_with_one_iq_002_per_cycle() {
    let name = "ref";
    let m0 = machine(name);
    let cap = m0.iq().cap();
    let rate = m0.iq().rate();
    let lat = 1 + m0.geom().fetch_latency;
    let n = m0.geom().vlanes as usize / 4;
    // after the IfetchM dispatches, two units of room remain; the payload is all non-padding
    let ones = vec![1u32; n];
    let mut m = machine(name);
    m.preload(4, 1, &word_of(&m, &ones)).unwrap();
    m.preload(4, 2, &word_of(&m, &ones)).unwrap();
    let mut img = units(&Insn::IfetchM { addr: 1 });
    img.extend(vec![0xFFFFu32; cap - 2]); // NOP 65535: a prefix without an instruction: it never dispatches
    m.load_image(4, &img).unwrap();
    let mut r = Rec::default();
    m.run_until(1_000, 1_000_000, &mut r).unwrap();
    let steps = (2 * n).div_ceil(rate) as u64;
    let got: Vec<u64> = r.diags.iter().filter(|(_, q, c)| *q == 4 && *c == Code::Iq002).map(|(c, _, _)| *c).collect();
    assert_eq!(got, (lat..lat + steps).collect::<Vec<_>>());
    assert!(r.lints.iter().all(|(_, _, c)| *c != "OTS-W-IQ-002"), "the lint is retired");
}

#[test]
fn an_ifetchm_pair_outside_the_segment_fetches_nothing() {
    let mut m = machine("ref");
    m.edit_state(|state| state.mem_segment[4] = 4).unwrap();
    m.preload(4, 3, &word_of(&m, &units(&Insn::SetVL { n: 3 }))).unwrap();
    m.load_image(4, &units(&Insn::IfetchM { addr: 3 })).unwrap();
    let mut r = Rec::default();
    m.run(1_000, &mut r).unwrap();
    assert_eq!(r.diags, vec![(0, 4, Code::Opr001)]);
    assert!(m.iq().fill[4].is_empty() && m.iq().len[4] == 0);
}

#[test]
fn an_ifetchm_counts_two_words_read_from_its_slice() {
    let mut m = machine("ref");
    let vb = m.geom().vlanes;
    let c = ctx(4, 30);
    assert_eq!(m.execute(&Insn::IfetchM { addr: 0x10 }, &c, &mut NoObserver), Exec::Issued);
    let s = m.activity().mem[4];
    assert_eq!((s.reads, s.read_bytes, s.writes), (2, 2 * vb, 0));
    assert_eq!(m.activity().issued[ots_isa::EntryId::IfetchM.index()], 1);
    assert_eq!((m.activity().active_cycles[0], m.activity().active_cycles[5]), (1, 1), "MEM and CU each active once");
    // an IfetchM outside the segment raises a diagnostic and counts nowhere
    m.edit_state(|state| state.mem_segment[4] = 16).unwrap();
    assert_eq!(m.execute(&Insn::IfetchM { addr: 0x0F }, &ctx(4, 40), &mut NoObserver), Exec::Diagnostic(Code::Opr001));
    assert_eq!(m.activity().mem[4].reads, 2);
}
