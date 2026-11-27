//! The ABC product with its output lanes split over a pool of threads equals the serial loop, lane for lane, for every
//! split width, on random and edge inputs: f16 (specials, subnormals, sparse and dense activations, the all-zero
//! activation, a short vector), int8 of every sign mix, and any pool size. The lanes of an ABC are independent and the
//! reduction order inside a lane is untouched, so nothing else may differ.
use tsp_iss::geom::MAX_LANES;
use tsp_iss::operands::{Planes, NO_PLANES};
use tsp_iss::slices::mxm::{abc_product_serial, abc_product_split, WType};
use tsp_iss::slices::pool::Pool;
use tsp_iss::srf::Vec320;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

const F16_EDGES: [u16; 14] =
    [0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0x0400, 0x5640];

/// An f16 element: edge values and random finite ones; specials only when `specials`.
fn f16(rng: &mut Rng, specials: bool) -> u16 {
    let r = rng.next();
    loop {
        let v = if r.is_multiple_of(5) { F16_EDGES[(r >> 8) as usize % F16_EDGES.len()] } else { (r >> 16) as u16 };
        if specials || v & 0x7c00 != 0x7c00 {
            return v;
        }
        let r2 = rng.next();
        let v = (r2 >> 16) as u16 & 0x83ff | ((r2 >> 40) as u16 % 30) << 10;
        if v & 0x7c00 != 0x7c00 {
            return v;
        }
    }
}

fn f16_rows(rng: &mut Rng, rows: usize, vl: usize, specials: bool) -> Vec<Vec320> {
    let mut w = vec![[0u8; MAX_LANES]; 2 * rows];
    for r in 0..rows {
        #[allow(clippy::needless_range_loop)] // the lane indexes the plane and the draw
        for l in 0..vl {
            let sp = specials && rng.next().is_multiple_of(40);
            let v = f16(rng, sp);
            w[2 * r][l] = v as u8;
            w[2 * r + 1][l] = (v >> 8) as u8;
        }
    }
    w
}

fn f16_act(rng: &mut Rng, vl: usize, density: u64, specials: bool) -> Planes {
    let mut x = NO_PLANES;
    #[allow(clippy::needless_range_loop)] // the lane indexes both planes
    for l in 0..vl {
        if rng.next() % 100 < density {
            let sp = specials && rng.next().is_multiple_of(50);
            let v = f16(rng, sp);
            x[0][l] = v as u8;
            x[1][l] = (v >> 8) as u8;
        }
    }
    x
}

fn int_rows(rng: &mut Rng, rows: usize, vl: usize) -> Vec<Vec320> {
    (0..rows)
        .map(|_| {
            let mut v = [0u8; MAX_LANES];
            for b in v[..vl].iter_mut() {
                *b = rng.next() as u8;
            }
            v
        })
        .collect()
}

fn pools() -> Vec<Pool> {
    vec![Pool::new(0), Pool::new(1), Pool::new(3)]
}

#[test]
fn f16_split_equals_serial_for_every_width() {
    let mut rng = Rng(0x5911_7f16);
    let pools = pools();
    for (rows, vl) in [(160usize, 320usize), (160, 64), (23, 320), (1, 320), (16, 17), (40, 8)] {
        for (density, specials) in [(20u64, false), (100, false), (0, false), (20, true), (3, true)] {
            let w = f16_rows(&mut rng, rows, vl, specials);
            let x = f16_act(&mut rng, vl, density, specials);
            let want = abc_product_serial(WType::F16, &w, rows, vl, false, false, &x);
            for pool in &pools {
                for chunk in 1..=rows + 1 {
                    let got = abc_product_split(WType::F16, &w, rows, vl, false, false, &x, pool, chunk);
                    assert_eq!(
                        got[..],
                        want[..],
                        "rows {rows} vl {vl} density {density} specials {specials} chunk {chunk}"
                    );
                }
            }
        }
    }
}

#[test]
fn int8_split_equals_serial_for_every_width_and_sign_mix() {
    let mut rng = Rng(0x1818_5911);
    let pools = pools();
    for (rows, vl) in [(320usize, 320usize), (100, 64), (17, 320), (1, 5)] {
        let w = int_rows(&mut rng, rows, vl);
        let mut x = NO_PLANES;
        x[0] = int_rows(&mut rng, 1, vl)[0];
        for (t, ws, xs) in
            [(WType::I8, true, true), (WType::U8, false, false), (WType::I8, false, true), (WType::U8, true, false)]
        {
            let want = abc_product_serial(t, &w, rows, vl, ws, xs, &x);
            for pool in &pools {
                for chunk in (1..=rows + 1).step_by(if rows > 100 { 7 } else { 1 }) {
                    let got = abc_product_split(t, &w, rows, vl, ws, xs, &x, pool, chunk);
                    assert_eq!(got[..], want[..], "rows {rows} vl {vl} {t:?} ws {ws} xs {xs} chunk {chunk}");
                }
            }
        }
    }
}

#[test]
fn the_lanes_past_the_rows_stay_zero() {
    let mut rng = Rng(7);
    let w = f16_rows(&mut rng, 10, 64, false);
    let x = f16_act(&mut rng, 64, 50, false);
    let got = abc_product_split(WType::F16, &w, 10, 64, false, false, &x, &Pool::new(3), 3);
    assert!(got[10..].iter().all(|&v| v == 0));
}
