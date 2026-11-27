//! Tests of `slices/c2c`, moved from the module's in-source test block (they use the crate's public API).
//! The C2C unit tests of the Sail model (`unit_tests/test_c2c.sail`: `test_c2c_link_instructions`,
//! `test_c2c_published_constants`, `test_c2c_link_busy`) and the ISS's own form of them: Send, Transmit and Receive on a looped-back chip, the checks and their order, and that a
//! rejected Receive leaves the FIFO unchanged.
use ots_isa::Insn;
use tsp_iss::geom::MAX_LANES;
use tsp_iss::observe::Observer;
use tsp_iss::slices::{Exec, Issue};
use tsp_iss::srf::Vec320;
use tsp_iss::{Diag, Machine};

use ots_profile::Profile;
use tsp_iss::harness::links::PlanLinks;
use tsp_iss::layout::{issuer_of, read_gap, write_gap, RowId};
use tsp_iss::srf::SrfLookup;

/// Every profile the Sail unit tests run on: the shipped ones and `config/test/`.
const PROFILES: [&str; 7] = ["ref", "mini", "link_tight", "mid_min", "mini_small", "ref_slow", "ref_small"];

fn machine(name: &str) -> Machine {
    // the test profiles live in config/test/
    let p = if matches!(name, "ref" | "mini") {
        Profile::load(name)
    } else {
        Profile::load_file(
            name,
            format!("{}/../../../opentsp-arch-spec/config/test/{name}.json", env!("CARGO_MANIFEST_DIR")),
        )
    }
    .unwrap();
    let port = PlanLinks::loopback(&tsp_iss::Geom::new(&p));
    let mut m = Machine::new(&p, 0, Box::new(port)).unwrap();
    m.edit_state(|state| state.activity.set_enabled(true)).unwrap(); // the tests read the activity counters, which are off unless asked for
    m
}

#[derive(Default)]
struct Rec {
    diags: Vec<(u64, u16, Diag)>,
}

impl Observer for Rec {
    fn diag(&mut self, c: u64, queue: u16, code: Diag) {
        self.diags.push((c, queue, code));
    }
}

fn exec(m: &mut Machine, q: usize, c: u64, i: &Insn) -> Exec {
    let ctx = Issue {
        cycle: c,
        queue: q,
        element: 0,
        addr_offset: 0,
        stream_offset: 0,
        thread: 0,
        lat: m.latency_of_instr(i),
    };
    m.execute(i, &ctx, &mut Rec::default())
}

fn diag(d: Diag) -> Exec {
    Exec::Diagnostic(d)
}

// West TX queues 126, 127, RX 128, 129; the east ones are the same + 14 (core/queues.sail).
const TX: usize = 126;
const RX: usize = 128;

/// The row queue `q` reads, and the row it writes.
fn read_row(m: &Machine, q: usize) -> Option<RowId> {
    let i = issuer_of(q);
    [RowId::E, RowId::W].into_iter().find(|r| read_gap(m.geom(), i.unit, i.slice as usize, *r).is_some())
}
fn write_row(m: &Machine, q: usize) -> Option<RowId> {
    let i = issuer_of(q);
    [RowId::E, RowId::W].into_iter().find(|r| write_gap(m.geom(), i.unit, i.slice as usize, *r).is_some())
}
fn sref(r: RowId, id: u32) -> u32 {
    id + if r == RowId::W { 32 } else { 0 }
}

fn vector(m: &Machine, seed: u8) -> Vec320 {
    let mut v = [0u8; MAX_LANES];
    for (l, x) in v.iter_mut().enumerate().take(m.geom().vlanes as usize) {
        *x = seed.wrapping_add(l as u8);
    }
    v
}

/// Feed stream `id` of the TX queue's row so that a read in cycle `t` at its gap finds `v`.
fn feed_tx(m: &mut Machine, id: u32, v: &Vec320, t: u64) {
    let r = read_row(m, TX).unwrap();
    let i = issuer_of(TX);
    let g = read_gap(m.geom(), i.unit, i.slice as usize, r).unwrap();
    m.edit_state(|state| state.srf.write(r, id as u8, g - 1, t - 1, v)).unwrap();
}

fn send(chip: u32, link: u32, s: u32) -> Insn {
    Insn::Send { chip_id: chip, link, stream: s }
}
fn receive(chip: u32, link: u32, s: u32) -> Insn {
    Insn::Receive { chip_id: chip, link, stream: s }
}

/// The value the RX queue committed on stream `id` in cycle `commit`, after ticking from the
/// dispatch cycle `from` to it (the commit ring has 128 slots, so a tick must not start earlier).
fn received(m: &mut Machine, id: u32, from: u64, commit: u64) -> Option<Vec320> {
    for c in from..=commit {
        m.tick(c, &mut Rec::default());
    }
    let r = write_row(m, RX).unwrap();
    let i = issuer_of(RX);
    let g = write_gap(m.geom(), i.unit, i.slice as usize, r).unwrap();
    match m.srf().read(r, id as u8, g + 1, commit as i64 + 1) {
        (SrfLookup::Found, v) => Some(v),
        _ => None,
    }
}

#[test]
fn a_looped_back_send_arrives_after_the_link_latency_and_receive_writes_it_to_a_stream() {
    let mut m = machine("ref");
    let lat = m.latency_of_instr(&send(0, 3, 0));
    let r = write_row(&m, RX).unwrap();
    let rd = read_row(&m, TX).unwrap();
    let v = vector(&m, 9);
    let c = 40u64;
    feed_tx(&mut m, 5, &v, c + lat.d_skew as u64);
    assert_eq!(exec(&mut m, TX, c, &send(0, 3, sref(rd, 5))), Exec::Issued);
    let sent = c + lat.d_skew as u64;
    let nominal = m.geom().link_latency;
    // one cycle early: nothing has arrived; the arrival cycle: the vector is taken
    let early = sent + nominal - 1 - lat.d_skew as u64;
    assert_eq!(exec(&mut m, RX, early, &receive(0, 3, sref(r, 2))), diag(Diag::Tim001));
    let on_time = sent + nominal - lat.d_skew as u64;
    assert_eq!(exec(&mut m, RX, on_time, &receive(0, 3, sref(r, 2))), Exec::Issued);
    assert_eq!(received(&mut m, 2, on_time, on_time + lat.d_func as u64), Some(v));
    // the FIFO is empty again
    assert_eq!(exec(&mut m, RX, on_time + 1, &receive(0, 3, sref(r, 2))), diag(Diag::Tim001));
}

#[test]
fn transmit_sends_the_notification_vector() {
    let mut m = machine("ref");
    let lat = m.latency_of_instr(&Insn::Transmit { chip_id: 0, link: 0 });
    let r = write_row(&m, RX).unwrap();
    assert_eq!(exec(&mut m, TX, 10, &Insn::Transmit { chip_id: 0, link: 0 }), Exec::Issued);
    let at = 10 + lat.d_skew as u64 + m.geom().link_latency - lat.d_skew as u64;
    assert_eq!(exec(&mut m, RX, at, &receive(0, 0, sref(r, 0))), Exec::Issued);
    let v = received(&mut m, 0, at, at + lat.d_func as u64).unwrap();
    assert_eq!(v[0], 1);
    assert!(v[1..].iter().all(|b| *b == 0));
}

#[test]
fn the_chip_id_and_the_link_are_checked_first() {
    let mut m = machine("ref");
    let r = write_row(&m, RX).unwrap();
    let rd = read_row(&m, TX).unwrap();
    // one chip, links loop back: the peer of every link is chip 0
    assert_eq!(exec(&mut m, TX, 10, &send(1, 0, sref(rd, 0))), diag(Diag::Opr001));
    assert_eq!(exec(&mut m, TX, 10, &Insn::Transmit { chip_id: 7, link: 2 }), diag(Diag::Opr001));
    assert_eq!(exec(&mut m, RX, 10, &receive(2, 0, sref(r, 0))), diag(Diag::Opr001));
    // a link beyond c2c.links (mini has fewer than 16 or not: use the geometry)
    let n = m.geom().links as u32;
    if n < 16 {
        assert_eq!(exec(&mut m, TX, 10, &send(0, n, sref(rd, 0))), diag(Diag::Opr001));
        assert_eq!(exec(&mut m, RX, 10, &receive(0, n, sref(r, 0))), diag(Diag::Opr001));
    }
    // the chip check comes before the operand read: no stream was fed, still OPR-001
    assert_eq!(exec(&mut m, TX, 10, &send(5, 0, sref(rd, 0))), diag(Diag::Opr001));
    // a good chip and link but nothing fed: TIM-001 from the read
    assert_eq!(exec(&mut m, TX, 10, &send(0, 0, sref(rd, 0))), diag(Diag::Tim001));
}

#[test]
fn a_rejected_receive_leaves_the_fifo_unchanged() {
    let mut m = machine("ref");
    let lat = m.latency_of_instr(&send(0, 1, 0));
    let rd = read_row(&m, TX).unwrap();
    let r = write_row(&m, RX).unwrap();
    let v = vector(&m, 3);
    feed_tx(&mut m, 0, &v, 20 + lat.d_skew as u64);
    assert_eq!(exec(&mut m, TX, 20, &send(0, 1, sref(rd, 0))), Exec::Issued);
    let at = 20 + m.geom().link_latency;
    // the destination row is the one the RX unit cannot write (h2): DIR-001, the vector stays
    let wrong = if r == RowId::E { RowId::W } else { RowId::E };
    assert_eq!(exec(&mut m, RX, at, &receive(0, 1, sref(wrong, 0))), diag(Diag::Dir001));
    assert_eq!(exec(&mut m, RX, at, &receive(0, 1, sref(r, 0))), Exec::Issued);
    assert_eq!(received(&mut m, 0, at, at + lat.d_func as u64), Some(v));
}

#[test]
fn deskew_and_runtime_deskew_execute_as_no_ops() {
    let mut m = machine("ref");
    assert_eq!(exec(&mut m, 30, 5, &Insn::Deskew), Exec::Issued);
    assert_eq!(exec(&mut m, 30, 5, &Insn::RuntimeDeskew { t: 9 }), Exec::Issued);
}

// model: unit_tests/test_c2c.sail test_c2c_link_busy
#[test]
fn a_link_accepts_one_vector_per_cycles_per_vector() {
    for name in ["ref", "link_tight"] {
        let mut m = machine(name);
        let n = m.geom().cycles_per_vector;
        let t = |chip, link| Insn::Transmit { chip_id: chip, link };
        m.reset();
        assert_eq!(exec(&mut m, TX, 100, &t(0, 0)), Exec::Issued, "{name}");
        assert_eq!(exec(&mut m, TX, 100, &t(0, 0)), diag(Diag::Lnk001), "{name}");
        assert_eq!(exec(&mut m, TX, 100, &t(0, 1)), Exec::Issued, "{name}");
        assert_eq!(exec(&mut m, TX, 100, &send(0, 0, 0)), diag(Diag::Lnk001), "{name}");
        assert_eq!(exec(&mut m, TX, 100 + n - 1, &t(0, 0)), diag(Diag::Lnk001), "{name}");
        assert_eq!(exec(&mut m, TX, 100 + n, &t(0, 0)), Exec::Issued, "{name}");
        // the window is measured from the last accepted vector, not from a refused one
        assert_eq!(exec(&mut m, TX, 100 + 2 * n - 1, &t(0, 0)), diag(Diag::Lnk001), "{name}");
        assert_eq!(exec(&mut m, TX, 100 + 2 * n, &t(0, 0)), Exec::Issued, "{name}");
        // OPR-001 is checked first and occupies nothing; a Send that finds no data occupies nothing
        assert_eq!(exec(&mut m, TX, 1000, &t(3, 4)), diag(Diag::Opr001), "{name}");
        assert_eq!(exec(&mut m, TX, 1000, &send(0, 4, 0b100000)), diag(Diag::Tim001), "{name}");
        assert_eq!(exec(&mut m, TX, 1000, &t(0, 4)), Exec::Issued, "{name}");
        m.reset();
        assert_eq!(exec(&mut m, TX, 5, &t(0, 0)), Exec::Issued, "{name}");
        m.reset();
        assert_eq!(exec(&mut m, TX, 0, &t(0, 0)), Exec::Issued, "{name}");
        assert!(m.geom().rx_depth >= 1);
    }
}

// model: unit_tests/test_c2c.sail test_c2c_link_instructions
#[test]
fn test_c2c_link_instructions() {
    // The unit tests run on every profile; the platform is chip 0 with every link looped back.
    for name in PROFILES {
        let mut m = machine(name);
        let rd = read_row(&m, TX).unwrap();
        assert_eq!(m.geom().links, 16, "{name}");
        let (w_tx, w_rx, e_tx, e_rx, vxm) = (126, 128, 140, 142, 88);
        let s = send(0, 0, 0);
        assert!(m.issued_by(&s, w_tx), "{name}");
        assert!(!m.issued_by(&s, w_rx) && !m.issued_by(&s, vxm), "{name}");
        let r = receive(0, 0, 0);
        assert!(m.issued_by(&r, e_rx) && !m.issued_by(&r, e_tx), "{name}");
        let t = |link| Insn::Transmit { chip_id: 0, link };
        assert!(m.issued_by(&t(0xF), e_tx) && !m.issued_by(&t(0), e_rx), "{name}");
        assert_eq!(s.entry().timing, ots_profile::TimingClass::C2c, "{name}");
        m.reset();
        // Receive with nothing arrived is a timing error.
        assert_eq!(exec(&mut m, RX, 10, &receive(0, 0x0, 1)), diag(Diag::Tim001), "{name}");
        assert_eq!(exec(&mut m, RX, 10, &receive(0, 0x9, 1)), diag(Diag::Tim001), "{name}");
        assert_eq!(exec(&mut m, RX, 10, &receive(0b000011, 0x0, 1)), diag(Diag::Opr001), "{name}");
        assert_eq!(exec(&mut m, TX, 10, &t(0)), Exec::Issued, "{name}");
        assert_eq!(exec(&mut m, TX, 10, &t(0xF)), Exec::Issued, "{name}");
        assert_eq!(
            exec(&mut m, TX, 10, &Insn::Transmit { chip_id: 0b000101, link: 0x3 }),
            diag(Diag::Opr001),
            "{name}"
        );
        // Send reads its stream at the TX gap: nothing is there.
        assert_eq!(exec(&mut m, TX, 10, &send(0, 0x7, sref(rd, 0))), diag(Diag::Tim001), "{name}");
        // c2c_notification: what Transmit sent is lane 0 = 1, every other lane 0.
        let lat = m.latency_of_instr(&t(0));
        let at = 10 + m.geom().link_latency;
        let w = write_row(&m, RX).unwrap();
        assert_eq!(exec(&mut m, RX, at, &receive(0, 0, sref(w, 0))), Exec::Issued, "{name}");
        let v = received(&mut m, 0, at, at + lat.d_func as u64).unwrap();
        assert_eq!(v[0], 1, "{name}");
        assert!(v[1..].iter().all(|b| *b == 0), "{name}");
    }
}

// model: unit_tests/test_c2c.sail test_c2c_published_constants
#[test]
fn test_c2c_published_constants() {
    // The HAC epoch is 252 cycles: an 8-bit counter with 4 values reserved (ISCA 2022 sec. 3.2
    // footnote 4); the link latency bounds come from ISCA 2022 Table 2.
    for name in PROFILES {
        let m = machine(name);
        assert_eq!(m.geom().hac_period, 252, "{name}");
        assert_eq!(m.occupancy(&Insn::Deskew, 251), 1, "{name}");
        assert_eq!(m.occupancy(&Insn::Deskew, 252), 252, "{name}");
        let g = m.geom();
        assert_eq!((g.link_latency_min, g.link_latency, g.link_latency_max), (209, 216, 228), "{name}");
    }
}

#[test]
fn the_c2c_activity_counts_vectors_and_sync_packets_per_link() {
    let mut m = machine("ref");
    let lat = m.latency_of_instr(&send(0, 3, 0));
    let r = write_row(&m, RX).unwrap();
    let rd = read_row(&m, TX).unwrap();
    let c = 40u64;
    let v = vector(&m, 9);
    feed_tx(&mut m, 5, &v, c + lat.d_skew as u64);
    assert_eq!(exec(&mut m, TX, c, &send(0, 3, sref(rd, 5))), Exec::Issued);
    let on_time = c + m.geom().link_latency;
    assert_eq!(exec(&mut m, RX, on_time, &receive(0, 3, sref(r, 2))), Exec::Issued);
    // a Transmit on another link, later than the first vector's occupancy of its own link
    assert_eq!(exec(&mut m, TX, c + 100, &Insn::Transmit { chip_id: 0, link: 1 }), Exec::Issued);
    let a = m.activity();
    assert_eq!((a.c2c_sent[3], a.c2c_received[3], a.c2c_sync[3]), (1, 1, 0));
    assert_eq!((a.c2c_sent[1], a.c2c_sync[1]), (0, 1), "a sync packet is not a data vector");
    assert_eq!(a.srf_writes.iter().sum::<u64>(), 1, "the Receive wrote one stream");
    let j = a.to_json();
    assert_eq!((j["c2c"]["sent_bytes"].as_u64(), j["c2c"]["received_bytes"].as_u64()), (Some(320), Some(320)));
    assert_eq!(j["c2c"]["per_link"].as_array().unwrap().len(), 2);
    // a Receive that finds nothing arrived (TIM-001) counts nowhere
    assert!(matches!(exec(&mut m, RX, c + 200, &receive(0, 3, sref(r, 2))), Exec::Diagnostic(_)));
    assert_eq!(m.activity().c2c_received[3], 1);
}

#[test]
fn host_vectors_are_counted_in_both_directions() {
    use tsp_iss::host::{HostEvent, HostKind};
    // three Reads fill streams 0..2 from slice 44; PciSend at cycles 100 (accepted), 105 (inside the interval: a
    // diagnostic, counted nowhere) and 110 (accepted)
    let mut m = machine("ref");
    m.load_image(44, &[0x4b, 0x1000_0001, 0x8000_0000, 4, 0x1000_0002, 0x8400_0000, 4, 0x1000_0003, 0x8800_0000])
        .unwrap();
    m.load_image(126, &[0x64, 0x5180_8000, 4, 0x5180_8400, 4, 0x5180_8800]).unwrap();
    m.run(10_000, &mut Rec::default()).unwrap();
    let a = m.activity();
    assert_eq!((a.host_out, a.host_in), (2, 0), "two vectors accepted by the host, one refused");
    assert_eq!(a.mem[44].reads, 3);
    let j = a.to_json();
    assert_eq!((j["host"]["out_vectors"].as_u64(), j["host"]["out_bytes"].as_u64()), (Some(2), Some(640)));
    // a vector the host's DMA engine pushed, taken by PciReceive S24 on RX queue 142
    let mut m = machine("ref");
    let mut b = Box::new([0u8; MAX_LANES]);
    b[..320].fill(7);
    m.set_host_script(vec![HostEvent { cycle: 0, kind: HostKind::Dma(b) }], 0);
    m.load_image(142, &[100, 0x5140_e000]).unwrap();
    m.run(10_000, &mut Rec::default()).unwrap();
    assert_eq!((m.activity().host_in, m.activity().host_out), (1, 0));
    assert_eq!(m.activity().srf_writes, [0, 1], "the vector landed on a W stream");
    assert_eq!(m.activity().to_json()["host"]["in_bytes"], 320);
}
