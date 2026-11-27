//! Tests of `srf`, moved from the module's in-source test block (they use the crate's public API).
//! Ports of `unit_tests/test_streams.sail` (the stream register file tests), and the ISS's own:
//! the diagonal slots, lazy allocation, the value arena, the commit callback.
use ots_profile::Profile;
use tsp_iss::geom::MAX_LANES;
use tsp_iss::layout::RowId::{E, W};
use tsp_iss::links::NoLinks;
use tsp_iss::observe::{Observer, Row};
use tsp_iss::srf::*;

/// `v(b)`: a vector of `vlanes` copies of byte `b`.
fn v(vl: usize, b: u8) -> Vec320 {
    let mut x = [0u8; MAX_LANES];
    x[..vl].fill(b);
    x
}

fn srf(vl: usize) -> Srf {
    Srf::new(vl)
}

// model: unit_tests/test_streams.sail test_srf_visible_once
#[test]
fn srf_visible_once() {
    let mut s = srf(320);
    s.write(E, 3, 18, 15, &v(320, 0x2a));
    let (k1, x) = s.read(E, 3, 21, 18);
    assert!(k1 == SrfLookup::Found && x == v(320, 0x2a));
    assert_eq!(s.read(E, 3, 21, 17).0, SrfLookup::None);
    assert_eq!(s.read(E, 3, 21, 19).0, SrfLookup::None);
}

// model: unit_tests/test_streams.sail test_srf_overwrite
#[test]
fn srf_overwrite_nearest_upstream_writer_wins() {
    let mut s = srf(320);
    s.write(E, 2, 8, 15, &v(320, 0x01)); // diagonal -7
    s.write(E, 2, 13, 20, &v(320, 0x02)); // diagonal -7, downstream writer
    let (k1, a) = s.read(E, 2, 11, 18); // between: sees the first
    assert!(k1 == SrfLookup::Found && a == v(320, 0x01));
    let (k2, b) = s.read(E, 2, 18, 25); // after: sees the second
    assert!(k2 == SrfLookup::Found && b == v(320, 0x02));
}

// model: unit_tests/test_streams.sail test_srf_collision
#[test]
fn srf_collision() {
    let mut s = srf(320);
    assert!(!s.write(W, 0, 18, 10, &v(320, 0x01)));
    assert!(s.write(W, 0, 18, 10, &v(320, 0x02)));
    // the second write replaced the first
    assert_eq!(s.read(W, 0, 21, 13).1, v(320, 0x02));
}

// model: unit_tests/test_streams.sail test_srf_upstream
#[test]
fn srf_upstream() {
    let mut s = srf(320);
    s.write(E, 8, 21, 25, &v(320, 0x05)); // VXM writes E at gap 21
    assert_eq!(s.read(E, 8, 18, 22).0, SrfLookup::Upstream); // west MEM quad 0 reads E at gap 18
}

// model: unit_tests/test_streams.sail test_srf_rows_independent
#[test]
fn srf_rows_independent() {
    let mut s = srf(320);
    s.write(E, 1, 18, 10, &v(320, 0x11));
    assert_eq!(s.read(W, 1, 21, 13).0, SrfLookup::None);
}

// model: unit_tests/test_streams.sail test_srf_upstream_live_only
#[test]
fn srf_upstream_live_only() {
    let mut s = srf(320);
    s.write(E, 8, 21, 25, &v(320, 0x05));
    assert_eq!(s.read(E, 8, 18, 200).0, SrfLookup::None); // the write left the row long ago
    let mut s = srf(320);
    s.write(E, 9, 5, 10, &v(320, 0x01)); // stale upstream write
    s.write(E, 9, 21, 100, &v(320, 0x02)); // live downstream write
    assert_eq!(s.read(E, 9, 18, 102).0, SrfLookup::Upstream);
}

// model: core/streams.sail srf_read_chain: only a write AT the reader's own gap in cycle t - 1
#[test]
fn srf_read_chain_sees_only_the_write_at_its_own_gap_one_cycle_earlier() {
    let mut s = srf(320);
    s.write(E, 4, 21, 30, &v(320, 0x77));
    let (k, x) = s.read_chain(E, 4, 21, 31);
    assert!(k == SrfLookup::Found && x == v(320, 0x77));
    assert_eq!(s.read_chain(E, 4, 21, 30).0, SrfLookup::None);
    assert_eq!(s.read_chain(E, 4, 21, 32).0, SrfLookup::None);
    assert_eq!(s.read_chain(E, 4, 22, 31).0, SrfLookup::None);
    // the flow rule never sees a write at the reader's own gap: it is at or downstream of the
    // reader, so the read is a direction error
    assert_eq!(s.read(E, 4, 21, 31).0, SrfLookup::Upstream);
    assert_eq!(s.read(E, 5, 21, 31).0, SrfLookup::None);
}

// A value written at gap g in cycle c is at register g + 1 + k in cycle c + 1 + k, to the end of the
// row (register 44), and gone after it.
#[test]
fn srf_value_flows_one_register_per_cycle_until_it_falls_off() {
    let mut s = srf(16);
    s.write(W, 31, 3, 100, &v(16, 0xa5));
    for k in 0..41u64 {
        let g = (4 + k) as u8;
        assert_eq!(s.read(W, 31, g, 101 + k as i64).1, v(16, 0xa5), "register {g}");
        assert_eq!(s.read(W, 31, g, 102 + k as i64).0, SrfLookup::None, "a cycle too late at register {g}");
    }
    // register 44 is beyond the highest reader gap (43): the write is live while register <= 44
    assert_eq!(s.read(W, 31, 43, 140).1, v(16, 0xa5));
    assert_eq!(s.read(W, 31, 43, 141).0, SrfLookup::None);
}

// Two writes whose diagonals differ by 64 share a slot and a gap: the later write replaces the
// earlier one (P-SRF-3: slots never alias within the live window, so this is only a storage fact).
#[test]
fn srf_slot_aliasing_replaces_the_older_write() {
    let mut s = srf(16);
    s.write(E, 0, 10, 20, &v(16, 1)); // diagonal -10
    s.write(E, 0, 10, 20 + 64, &v(16, 2)); // diagonal -74: the same slot (-10 mod 64 = 54), same gap
    assert_eq!(s.read(E, 0, 12, 22 + 64).1, v(16, 2));
    assert_eq!(s.read(E, 0, 12, 22).0, SrfLookup::None);
    // not a collision: the diagonals differ
    assert!(!s.write(E, 0, 10, 20 + 128, &v(16, 3)));
}

#[test]
fn srf_reset_empties_every_stream_and_storage_is_lazy() {
    let mut s = srf(16);
    assert_eq!(s.streams_allocated(), 0);
    s.write(E, 3, 18, 15, &v(16, 9));
    s.write(W, 3, 18, 15, &v(16, 9));
    assert_eq!(s.streams_allocated(), 2);
    s.reset();
    assert_eq!(s.streams_allocated(), 0);
    assert_eq!(s.read(E, 3, 21, 18).0, SrfLookup::None);
    // the arena is released too
    assert_eq!(s.values_live(), 0);
}

#[test]
fn srf_overwritten_values_are_recycled() {
    let mut s = srf(16);
    for i in 0..200u64 {
        // the same slot and gap every time (diagonals 64 apart): the arena never grows past one value
        s.write(E, 0, 5, 1000 + 64 * i, &v(16, i as u8));
    }
    assert_eq!(s.values_live(), 1);
}

#[test]
fn srf_lanes_beyond_vlanes_are_not_stored() {
    let mut s = srf(16);
    let mut x = v(16, 0x11);
    x[16..40].fill(0xee);
    s.write(E, 0, 5, 10, &x);
    let (_, y) = s.read(E, 0, 6, 11);
    assert_eq!(y, v(16, 0x11));
}

// srf_write_callback: the machine reports the write (row, stream, gap) to the observer, then the
// value when the machine is asked to trace values.
#[derive(Default)]
struct Writes(Vec<(u64, Row, u8, u8, Vec<u8>)>);
impl Observer for Writes {
    fn srf(&mut self, c: u64, row: Row, s: u8, g: u8, v: &[u8]) {
        self.0.push((c, row, s, g, v.to_vec()));
    }
}

fn machine(name: &str) -> tsp_iss::Machine {
    tsp_iss::Machine::new(&Profile::load(name).unwrap(), 0, Box::new(NoLinks)).unwrap()
}

#[test]
fn srf_write_calls_the_callback_and_machine_srf_value_reads_back() {
    let mut m = machine("mini");
    let mut w = Writes::default();
    assert!(!m.srf_write(W, 6, 19, 40, &v(16, 0x42), &mut w));
    assert!(m.srf_write(W, 6, 19, 40, &v(16, 0x43), &mut w));
    assert_eq!(w.0.len(), 2);
    assert_eq!((w.0[0].0, w.0[0].1, w.0[0].2, w.0[0].3), (40, Row::West, 6, 19));
    assert_eq!(w.0[0].4, vec![0x42u8; 16]);
    // the value is at register 22 in cycle 43; a reader at gap 22 sees it
    assert_eq!(m.srf_value(Row::West, 6, 22, 43), Some(v(16, 0x43)));
    assert_eq!(m.srf_value(Row::West, 6, 22, 44), None);
    assert_eq!(m.srf_value(Row::East, 6, 22, 43), None);
}

#[test]
fn reset_empties_the_stream_register_file() {
    let mut m = machine("ref");
    m.srf_write(E, 0, 10, 5, &v(320, 1), &mut tsp_iss::NoObserver);
    assert!(m.srf_value(Row::East, 0, 12, 7).is_some());
    m.reset();
    assert_eq!(m.srf_value(Row::East, 0, 12, 7), None);
}

// The ISS's storage (lazy streams, slots of 44 entries, an arena of ids) against a literal port of
// the model's representation: `register srf_e : vector(32, vector(64, vector(44, srf_write)))`.
mod literal {
    use super::*;
    use tsp_iss::layout::RowId;

    #[derive(Clone, Copy)]
    struct W {
        valid: bool,
        diag: i64,
        value: u8, // a tag: the byte every lane holds
    }

    pub struct Model {
        rows: Vec<Vec<[[W; 44]; 64]>>,
    }

    impl Model {
        pub fn new() -> Model {
            let e = W { valid: false, diag: 0, value: 0 };
            Model { rows: vec![vec![[[e; 44]; 64]; 32]; 2] }
        }
        fn row(r: RowId) -> usize {
            if r == E {
                0
            } else {
                1
            }
        }
        pub fn write(&mut self, r: RowId, s: u8, g: u8, c: u64, value: u8) -> bool {
            let d = g as i64 - c as i64;
            let slot = &mut self.rows[Self::row(r)][s as usize][d.rem_euclid(64) as usize];
            let old = slot[g as usize];
            slot[g as usize] = W { valid: true, diag: d, value };
            old.valid && old.diag == d
        }
        pub fn read(&self, r: RowId, s: u8, g: u8, t: i64) -> (SrfLookup, u8) {
            let d = g as i64 - t;
            let slot = &self.rows[Self::row(r)][s as usize][d.rem_euclid(64) as usize];
            let (mut best, mut value) = (-1i64, 0u8);
            for (w, &e) in slot.iter().enumerate().take(44) {
                if e.valid && e.diag == d && (w as u8) < g && w as i64 > best {
                    best = w as i64;
                    value = e.value;
                }
            }
            if best >= 0 {
                return (SrfLookup::Found, value);
            }
            let (mut any, mut up) = (false, false);
            for k in 0..64 {
                for w in 0..44 {
                    let e = self.rows[Self::row(r)][s as usize][k][w];
                    if e.valid && e.diag + t <= 44 {
                        any = true;
                        if (w as u8) < g {
                            up = true;
                        }
                    }
                }
            }
            (if any && !up { SrfLookup::Upstream } else { SrfLookup::None }, 0)
        }
        pub fn read_chain(&self, r: RowId, s: u8, g: u8, t: i64) -> (SrfLookup, u8) {
            let d = g as i64 - t + 1;
            let e = self.rows[Self::row(r)][s as usize][d.rem_euclid(64) as usize][g as usize];
            if e.valid && e.diag == d {
                (SrfLookup::Found, e.value)
            } else {
                (SrfLookup::None, 0)
            }
        }
    }
}

#[test]
fn srf_storage_equals_the_literal_model_on_random_writes_and_reads() {
    let mut x: u64 = 0x2545_f491_4f6c_dd1d;
    let mut rnd = move |n: u64| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x % n
    };
    for vl in [16usize, 320] {
        let mut s = srf(vl);
        let mut m = literal::Model::new();
        let mut cycle = 100u64;
        for step in 0..12_000u32 {
            // few streams and gaps, cycles that advance slowly: many collisions, aliasing, upstream cases
            let (r, st, g) = (if rnd(2) == 0 { E } else { W }, rnd(3) as u8, rnd(44) as u8);
            if rnd(3) > 0 {
                let c = cycle + rnd(4);
                let tag = (step % 251) as u8 + 1;
                assert_eq!(s.write(r, st, g, c, &v(vl, tag)), m.write(r, st, g, c, tag), "step {step}: write");
            } else {
                let t = (cycle + rnd(60)) as i64 - 8;
                let got = s.read(r, st, g, t);
                let want = m.read(r, st, g, t);
                assert_eq!((got.0, got.1[0]), want, "step {step}: read {r:?} {st} gap {g} t {t}");
                assert!(got.1[..vl].iter().all(|b| *b == want.1) && got.1[vl..].iter().all(|b| *b == 0));
                let got = s.read_chain(r, st, g, t);
                let want = m.read_chain(r, st, g, t);
                assert_eq!((got.0, got.1[0]), want, "step {step}: chain read");
            }
            if rnd(5) == 0 {
                cycle += 1;
            }
        }
        // the arena holds at most one value per entry: 2 rows x 3 streams x 64 slots x 44 gaps
        assert!(s.values_live() <= 2 * 3 * 64 * 44);
    }
}
