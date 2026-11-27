//! `Srf::read` against a brute-force model of `core/streams.sail`: per (row, stream) 64 diagonal slots of 44 gap
//! entries, a write overwriting its entry, and a read taking the live write of the reader's diagonal with the largest
//! gap below the reader. Random writes and reads close in time (so slots fill, alias and get overwritten).
use std::collections::HashMap;
use tsp_iss::layout::RowId;
use tsp_iss::srf::{Srf, SrfLookup};

type Key = (usize, u8, usize, u8);

fn slot(d: i64) -> usize {
    d.rem_euclid(64) as usize
}

fn model(m: &HashMap<Key, (i64, Vec<u8>)>, r: usize, s: u8, g: u8, t: i64) -> (SrfLookup, Vec<u8>) {
    let d = g as i64 - t;
    for w in (0..g.min(44)).rev() {
        if let Some((dd, v)) = m.get(&(r, s, slot(d), w)) {
            if *dd == d {
                return (SrfLookup::Found, v.clone());
            }
        }
    }
    let live: Vec<u8> =
        m.iter().filter(|((rr, ss, _, _), (dd, _))| *rr == r && *ss == s && dd + t <= 44).map(|(k, _)| k.3).collect();
    if !live.is_empty() && live.iter().all(|w| *w >= g) {
        (SrfLookup::Upstream, vec![0; tsp_iss::geom::MAX_LANES])
    } else {
        (SrfLookup::None, vec![0; tsp_iss::geom::MAX_LANES])
    }
}

#[test]
fn reads_match_a_brute_force_model_of_the_slots() {
    let mut x: u64 = 0x2545_f491_4f6c_dd1d;
    let mut next = move |n: u64| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x % n
    };
    for vl in [320usize, 7] {
        let mut srf = Srf::new(vl);
        let mut m: HashMap<Key, (i64, Vec<u8>)> = HashMap::new();
        let mut c: u64 = 100;
        for i in 0..40_000u64 {
            c += next(3);
            let r = next(2) as usize;
            let row = if r == 0 { RowId::E } else { RowId::W };
            let s = next(3) as u8;
            let g = next(46) as u8;
            if next(2) == 0 && g < 44 {
                let mut v = vec![0u8; tsp_iss::geom::MAX_LANES];
                for (k, b) in v.iter_mut().enumerate().take(vl) {
                    *b = (i as u8).wrapping_add(k as u8);
                }
                srf.write(row, s, g, c, &v);
                m.insert((r, s, slot(g as i64 - c as i64), g), (g as i64 - c as i64, v));
            } else {
                let t = c as i64 + next(50) as i64 - 5;
                let (k, v) = srf.read(row, s, g, t);
                let (wk, wv) = model(&m, r, s, g, t);
                assert_eq!(k, wk, "read {i}: row {r} stream {s} gap {g} cycle {t}");
                assert_eq!(&v[..], &wv[..], "read {i}: row {r} stream {s} gap {g} cycle {t}");
            }
        }
    }
}
