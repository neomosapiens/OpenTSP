//! Tests of `frontend`, moved from the module's in-source test block (they use the crate's public API).
use ots_isa::encode;
use ots_isa::{Insn, Isa};
use ots_profile::Profile;
use tsp_iss::frontend::*;
use tsp_iss::{Diag, Geom};

fn setup(name: &str) -> (Iq, Isa, Geom) {
    let p = Profile::load(name).unwrap();
    let g = Geom::new(&p);
    (Iq::new(&g), Isa::new(&p), g)
}

fn unit(bundle: bool, i: Insn) -> u32 {
    (if bundle { 0x8000_0000 } else { 0 }) | encode(&i) as u32
}
fn nop_u(n: u32) -> u32 {
    unit(false, Insn::NOP { n })
}
fn setvl_u(n: u32) -> u32 {
    unit(false, Insn::SetVL { n })
}
fn load(iq: &mut Iq, q: usize, us: &[u32]) {
    for &u in us {
        iq.push(q, u);
    }
}

// test_iq_ring
#[test]
fn test_iq_ring() {
    let (mut iq, _, g) = setup("ref");
    assert_eq!(iq.cap(), g.iq_units);
    let cap = iq.cap();
    for k in 0..cap {
        assert!(iq.push(7, k as u32));
    }
    assert_eq!(iq.len[7], cap);
    assert!(!iq.push(7, 0x1234_5678), "full: the unit is lost");
    assert!(iq.len[7] == cap && iq.peek(7, 0) == 0 && iq.peek(7, 5) == 5);
    iq.pop(7, 3);
    assert!(iq.len[7] == cap - 3 && iq.peek(7, 0) == 3);
    assert!(iq.push(7, 0xAAAA_0001) && iq.push(7, 0xAAAA_0002), "wraps");
    assert!(
        iq.len[7] == cap - 1
            && iq.peek(7, cap - 4) == (cap - 1) as u32
            && iq.peek(7, cap - 3) == 0xAAAA_0001
            && iq.peek(7, cap - 2) == 0xAAAA_0002
    );
    assert!(iq.len[8] == 0 && iq.len[144] == 0, "queues are independent");
    iq.clear();
    assert_eq!(iq.len[7], 0);
}

// test_iq_fill
#[test]
fn test_iq_fill() {
    for name in ["ref", "mini"] {
        let (mut iq, _, g) = setup(name);
        let vl = g.vlanes as usize;
        let mut a = vec![0u8; vl];
        let mut b = vec![0u8; vl];
        a[..4].copy_from_slice(&[0x11, 0x22, 0x33, 0x44]);
        a[4..8].fill(0xFF); // padding
        b[0] = 0x55;
        iq.fetch(3, 10, &a, &b);
        let t = 10 + g.fetch_latency;
        assert_eq!(iq.deliver(3, t - 1), (false, false));
        assert_eq!(iq.len[3], 0, "not ready yet");
        let (p1, l1) = iq.deliver(3, t);
        assert!(p1 && !l1);
        // the first unit is lanes 0..3; the second (padding) is dropped but used a place
        assert_eq!(iq.peek(3, 0), 0x4433_2211);
        let rate = iq.rate();
        assert_eq!(iq.len[3], if rate >= 2 { rate - 1 } else { 1 });
        // a second fetch queues behind the first
        iq.fetch(3, 10, &b, &a);
        let mut c = t + 1;
        let mut guard = 0;
        while !iq.fill[3].is_empty() && guard < 1000 {
            iq.deliver(3, c);
            c += 1;
            guard += 1;
        }
        assert!(guard < 1000);
        // both payloads are in, in order: 2 x 2 vectors of vlanes bytes, minus the two padding units
        assert_eq!(iq.len[3], (vl - 2).min(iq.cap()));
    }
}

// test_iq_overflow
#[test]
fn test_iq_overflow() {
    let (mut iq, _, g) = setup("ref");
    for _ in 0..iq.cap() {
        assert!(iq.push(9, 1));
    }
    let z = vec![0u8; g.vlanes as usize];
    iq.fetch(9, 0, &z, &z); // all-zero units are not padding
    let (p, l) = iq.deliver(9, g.fetch_latency);
    assert!(!p && l && iq.len[9] == iq.cap());
}

// test_parcel_scan
#[test]
fn test_parcel_scan() {
    let (mut iq, isa, _) = setup("ref");
    assert!(!iq.parcel_scan(&isa, 5).complete, "empty IQ");
    load(&mut iq, 5, &[nop_u(3), nop_u(4)]);
    assert!(!iq.parcel_scan(&isa, 5).complete, "only a prefix");
    load(&mut iq, 5, &[setvl_u(5)]);
    let p = iq.parcel_scan(&isa, 5);
    assert!(p.complete && p.nops == 7 && p.n == 1 && p.units == 3);
    assert_eq!(parcel_earliest(&iq, 5, &p), 7, "last = -1: the sum of the prefix");
    iq.last[5] = 100;
    assert_eq!(parcel_earliest(&iq, 5, &iq.parcel_scan(&isa, 5)), 108);
    // a bundle: two instructions, the first with the bundle bit
    iq.clear();
    load(&mut iq, 6, &[unit(true, Insn::Sync), setvl_u(2), setvl_u(3)]);
    let q = iq.parcel_scan(&isa, 6);
    assert!(q.complete && q.n == 2 && q.units == 2 && q.nops == 0);
    // a NOP inside the parcel is a member (it breaks the slot rule), not a prefix
    iq.clear();
    load(&mut iq, 6, &[unit(true, Insn::NOP { n: 2 }), setvl_u(2)]);
    let r = iq.parcel_scan(&isa, 6);
    assert!(r.complete && r.n == 2 && r.nops == 0);
    iq.clear();
    load(&mut iq, 6, &[unit(true, Insn::SetVL { n: 2 }), nop_u(2)]);
    let s = iq.parcel_scan(&isa, 6);
    assert!(s.complete && s.n == 2 && s.nops == 0);
    // a bundle that continues past the text is not complete
    iq.clear();
    load(&mut iq, 6, &[unit(true, Insn::Sync)]);
    assert!(!iq.parcel_scan(&isa, 6).complete);
    // a 64-bit instruction is two units; the first alone is not complete
    iq.clear();
    let rd = encode(&Insn::Read { addr: 1, dst: 0, thread: 0 });
    load(&mut iq, 6, &[(rd >> 32) as u32]);
    assert!(!iq.parcel_scan(&isa, 6).complete);
    load(&mut iq, 6, &[rd as u32]);
    let t = iq.parcel_scan(&isa, 6);
    assert!(t.complete && t.n == 1 && t.units == 2);
    // a tag-6 unit is one undecodable instruction
    iq.clear();
    load(&mut iq, 6, &[0x6000_0000, setvl_u(2)]);
    let u = iq.parcel_scan(&isa, 6);
    assert!(u.complete && u.n == 1 && u.units == 1 && u.items[..] == [Err(Diag::Dec003)]);
}

// test_decode_text (the part about the text layer)
#[test]
fn test_decode_text() {
    let (_, isa, _) = setup("ref");
    assert_eq!((text_width(0x1000_0000), text_width(0x2000_0000), text_width(0x4000_0000)), (2, 2, 2));
    assert_eq!(
        (
            text_width(0),
            text_width(0x3000_0000),
            text_width(0x5000_0000),
            text_width(0x6040_0000),
            text_width(0x7040_0000)
        ),
        (1, 1, 1, 1, 1)
    );
    // a reserved bit makes a word undecodable; NOP 0 does not decode
    assert_eq!(decode_text(&isa, 0x0040_0001, 0), Err(Diag::Dec003));
    assert_eq!(decode_text(&isa, 0, 0), Err(Diag::Dec003));
    assert_eq!(
        decode_text(&isa, nop_u(5) | 0x8000_0000, 0),
        Ok(Insn::NOP { n: 5 }),
        "the bundle bit is cleared before decode"
    );
}

// test_iq_staging_bound: the fetch staging FIFO holds icu.fetch_staging_depth payloads; a fetch that finds it full
// fetches nothing; a payload that has entered the IQ no longer counts; the other queues are independent.
// model: unit_tests/test_frontend.sail test_iq_staging_bound
#[test]
fn test_iq_staging_bound() {
    for name in ["ref", "mini"] {
        let (mut iq, _, g) = setup(name);
        let z = vec![0u8; g.vlanes as usize];
        let depth = g.fetch_staging_depth as usize;
        assert!(depth >= 1);
        for _ in 0..depth {
            assert!(iq.fetch(5, 10, &z, &z));
        }
        assert_eq!(iq.fill[5].len(), depth);
        assert!(!iq.fetch(5, 10, &z, &z), "{name}: a full staging FIFO fetches nothing");
        assert_eq!(iq.fill[5].len(), depth, "the refused fetch left no trace");
        assert!(iq.fetch(6, 10, &z, &z) && iq.fetch(144, 10, &z, &z));
        // drain queue 5 cycle by cycle: a head payload leaves the FIFO with its last unit
        let mut t = 10 + g.fetch_latency;
        while iq.fill[5].len() == depth {
            iq.deliver(5, t);
            t += 1;
        }
        assert_eq!(iq.fill[5].len(), depth - 1);
        assert!(iq.fetch(5, t, &z, &z));
    }
}

// test_iq_stranded: the end-of-run check; text that cannot form a parcel is stranded, a trailing NOP prefix is not.
// model: unit_tests/test_frontend.sail test_iq_stranded
#[test]
fn test_iq_stranded() {
    let (mut iq, isa, _) = setup("ref");
    assert!(!iq.stranded(&isa, 100));
    load(&mut iq, 100, &[nop_u(5), nop_u(7)]);
    assert!(!iq.stranded(&isa, 100), "a NOP with no successor is no truncation");
    iq.clear();
    load(&mut iq, 100, &[unit(true, Insn::SetVL { n: 2 })]);
    assert!(iq.stranded(&isa, 100), "the last instruction has the bundle bit");
    iq.clear();
    load(&mut iq, 100, &[nop_u(3), unit(true, Insn::SetVL { n: 2 })]);
    assert!(iq.stranded(&isa, 100));
    iq.clear();
    load(&mut iq, 100, &[0x1000_0000]);
    assert!(iq.stranded(&isa, 100), "the first unit of a MEM instruction only");
    iq.clear();
    load(&mut iq, 100, &[0x2000_0000]);
    assert!(iq.stranded(&isa, 100), "... of a VXM instruction");
    iq.clear();
    load(&mut iq, 144, &[unit(true, Insn::Notify)]);
    assert!(iq.stranded(&isa, 144) && !iq.stranded(&isa, 100));
}

#[test]
fn test_text_word_epoch_boundary() {
    let (mut iq, _, _) = setup("mini");
    iq.track_text();
    let first = 1usize << 32;
    iq.image_done(88, first);
    assert!(iq.fetch_ov(88, 0, &[0; 16], &[0; 16], vec![3, 0]));
    let ready = iq.fill[88].front().unwrap().ready;
    iq.inject(88, ready, 1);
    for cycle in ready..ready + 9 {
        iq.deliver(88, cycle);
    }
    let word = (first / 4) as u128;
    assert_eq!(iq.unit_word(88, 0), Some(word));
    assert_eq!(iq.tx.ov(88, word), 3);
    assert_eq!(iq.unit_word(88, 8), None);
}

/// A queue may be parsed against different ISA profiles. Ref accepts this lane shift and mini rejects it,
/// even if precisely these units were seen before. The bundle marker is independent of decoded operands.
#[test]
fn repeated_units_respect_the_active_profile_and_bundle_boundary() {
    let (mut iq, full, _) = setup("ref");
    let mini = Isa::new(&Profile::load("mini").unwrap());
    let shift = Insn::Shift { dst: 0, src: 0, dir: 0, amount: mini.profile().vlanes() as u32 };
    let word = encode(&shift);
    for (isa, expected) in [(&full, Ok(shift)), (&mini, Err(Diag::Dec003)), (&full, Ok(shift))] {
        iq.clear();
        load(&mut iq, 0, &[(word >> 32) as u32, word as u32]);
        iq.parse(isa, 0);
        assert_eq!(iq.head(0).items, [expected]);
    }

    iq.clear();
    load(&mut iq, 0, &[unit(true, Insn::Sync), unit(false, Insn::Sync)]);
    iq.parse(&full, 0);
    assert_eq!(iq.head(0).items, [Ok(Insn::Sync), Ok(Insn::Sync)]);
    assert!(iq.head_matches_scan(&full, 0));
}

/// A cached instruction with intact data bits still fails if two ECC check bits are corrupted.
#[test]
fn cached_units_are_checked_for_uncorrectable_text_errors() {
    let (mut iq, isa, _) = setup("ref");
    iq.track_text();
    assert!(iq.push_image(0, unit(false, Insn::Sync), 0));
    iq.parse(&isa, 0);
    assert_eq!(iq.head(0).items, [Ok(Insn::Sync)]);
    for bit in [128, 129] {
        assert!(iq.flip_text(0, 0, bit));
    }
    iq.reparse_reset(0);
    iq.parse(&isa, 0);
    assert_eq!(iq.head(0).items, [Err(Diag::Ecc002)]);
}

// the origin of the text taken from a queue: the object's own image, or later text (a fetch, the host)
#[test]
fn the_origin_of_the_parcel_last_taken_is_the_image_or_later_text() {
    let (mut iq, _, _) = setup("ref");
    let (a, b, c) = (nop_u(1), setvl_u(2), nop_u(3));
    for (k, u) in [a, b].into_iter().enumerate() {
        assert!(iq.push_image(5, u, k));
    }
    iq.image_done(5, 2);
    assert!(!iq.taken_fetched(5), "nothing taken yet");
    iq.pop(5, 1);
    assert!(!iq.taken_fetched(5), "the first unit of the image");
    // text that arrives after the image (a fetch pushes at the tail of the IQ)
    assert!(iq.push(5, c));
    iq.pop(5, 1);
    assert!(!iq.taken_fetched(5), "the last unit of the image");
    iq.pop(5, 1);
    assert!(iq.taken_fetched(5), "a unit after the image");
    // another queue has its own image length and count
    assert!(!iq.taken_fetched(6));
    assert!(iq.push_image(6, a, 0));
    iq.image_done(6, 1);
    iq.pop(6, 1);
    assert!(!iq.taken_fetched(6));
    assert!(iq.taken_fetched(5), "queue 5 is unchanged");
    // a reset forgets both
    iq.clear();
    assert!(!iq.taken_fetched(5));
    assert!(iq.push(5, a));
    iq.pop(5, 1);
    assert!(iq.taken_fetched(5), "a queue with no image: all its text came later");
}
