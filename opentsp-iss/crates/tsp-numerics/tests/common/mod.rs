//! Shared by the differential tests: job evaluation, and the input generators of
//! `tools/numerics_jobs.py` (mirrored; a divergence makes the hash tests fail).
// Every test crate includes this module and uses a subset of it.
#![allow(dead_code)]

use std::path::PathBuf;
use tsp_numerics::lanes::{self, ReduceOp};
use tsp_numerics::{fp, mxm, transcendental, EType};

/// splitmix64
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

pub fn ty(code: u32) -> EType {
    EType::from_code(code as u8).expect("type code")
}

pub fn mask_of(n: u32) -> u64 {
    (1u64 << n) - 1
}

pub const F16_EDGES: [u32; 27] = [
    0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x0400, 0x0401, 0x3BFF, 0x3C00, 0xBC00, 0x3C01, 0x4000, 0x4400,
    0x5640, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7C01, 0x7E00, 0xFE00, 0x7FFF, 0x0200, 0x3555, 0x6800, 0x0100,
];
pub const F32_EDGES: [u32; 25] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x007F_FFFF,
    0x0080_0000,
    0x0080_0001,
    0x3F7F_FFFF,
    0x3F80_0000,
    0xBF80_0000,
    0x3F80_0001,
    0x4000_0000,
    0x4B80_0000,
    0x7F7F_FFFF,
    0xFF7F_FFFF,
    0x7F80_0000,
    0xFF80_0000,
    0x7F80_0001,
    0x7FC0_0000,
    0xFFC0_0000,
    0x7FFF_FFFF,
    0x3380_0000,
    0x3EAA_AAAB,
    0x5F00_0000,
    0x0040_0000,
];

pub fn edges(t: EType) -> Vec<u32> {
    match t {
        EType::F16 => return F16_EDGES.to_vec(),
        EType::F32 => return F32_EDGES.to_vec(),
        _ => {}
    }
    let mask = mask_of(t.nbits());
    let h = mask >> 1;
    let vals = [0, 1, 2, 3, h - 1, h, h + 1, h + 2, mask - 2, mask - 1, mask, mask >> 2, (mask >> 2) + 1];
    let mut out: Vec<u32> = Vec::new();
    for v in vals {
        let v = (v & mask) as u32;
        if !out.contains(&v) {
            out.push(v);
        }
    }
    out
}

pub fn rand_float(rng: &mut Rng, t: EType) -> u32 {
    let n = t.nbits();
    let mask = mask_of(n);
    let f = lanes::fmt_of(t);
    let mut r = rng.next();
    let k = r & 7;
    r >>= 3;
    if k < 3 {
        return (rng.next() & mask) as u32;
    }
    let sign = r & 1;
    let e = f.bias as i64 + ((r >> 1) % 17) as i64 - 8;
    let m = rng.next() & ((1u64 << f.mbits) - 1);
    ((sign << (n - 1)) | ((e as u64) << f.mbits) | m) as u32
}

pub fn rand_int(rng: &mut Rng, t: EType) -> u32 {
    let n = t.nbits();
    let mask = mask_of(n);
    let r = rng.next();
    let k = r & 3;
    if k == 1 {
        return (rng.next() & mask_of(n / 2)) as u32;
    }
    if k == 2 {
        let base = [0, mask >> 1, (mask >> 1) + 1, mask][((r >> 2) & 3) as usize];
        return (base.wrapping_add((r >> 4) & 3).wrapping_sub(1) & mask) as u32;
    }
    (rng.next() & mask) as u32
}

pub fn rand_val(rng: &mut Rng, t: EType) -> u32 {
    if t.is_float() {
        rand_float(rng, t)
    } else {
        rand_int(rng, t)
    }
}

pub fn rand_pair(rng: &mut Rng, t: EType) -> (u32, u32) {
    if t.is_float() {
        let n = t.nbits();
        let sign = 1u32 << (n - 1);
        let mask = mask_of(n) as u32;
        let x = rand_float(rng, t);
        let k = rng.next() & 3;
        let y = if k == 1 {
            x ^ sign
        } else if k == 2 {
            (x ^ sign) ^ (rng.next() & 0xF) as u32
        } else {
            rand_float(rng, t)
        };
        (x, y & mask)
    } else {
        let x = rand_val(rng, t);
        let y = rand_val(rng, t);
        (x, y)
    }
}

/// An input set of a hash spec.
#[derive(Clone, Copy, Debug)]
pub enum Inp {
    All8,
    All16,
    Rand(u64, usize),
}

pub fn unary_inputs(t: EType, inp: Inp) -> Vec<u32> {
    match inp {
        Inp::All8 => (0..256).collect(),
        Inp::All16 => (0..65536).collect(),
        Inp::Rand(seed, n) => {
            let mut rng = Rng(seed);
            let mut v = edges(t);
            for _ in 0..n {
                v.push(rand_val(&mut rng, t));
            }
            v
        }
    }
}

pub fn binary_inputs(t: EType, inp: Inp) -> Vec<(u32, u32)> {
    match inp {
        Inp::All8 => (0..256u32).flat_map(|x| (0..256u32).map(move |y| (x, y))).collect(),
        Inp::All16 => panic!("no all16 for pairs"),
        Inp::Rand(seed, n) => {
            let mut rng = Rng(seed);
            let e = edges(t);
            let mut v: Vec<(u32, u32)> = e.iter().flat_map(|&x| e.iter().map(move |&y| (x, y))).collect();
            for _ in 0..n {
                v.push(rand_pair(&mut rng, t));
            }
            v
        }
    }
}

fn f16_finite(rng: &mut Rng, lo: u64, hi: u64) -> u16 {
    let r = rng.next();
    let e = lo + (r >> 1) % (hi - lo + 1);
    (((r & 1) << 15) | (e << 10) | (rng.next() & 0x3FF)) as u16
}

/// Case `idx` of the f16 dot product: (weights, activations) as f16 patterns.
pub fn dot_case(rng: &mut Rng, idx: usize, lanes: usize) -> (Vec<u16>, Vec<u16>) {
    let k = idx % 6;
    let mut ws: Vec<u16> = Vec::new();
    let mut xs: Vec<u16> = Vec::new();
    match k {
        0 => {
            for _ in 0..lanes {
                ws.push(f16_finite(rng, 0, 30));
                xs.push(f16_finite(rng, 0, 30));
            }
        }
        1 => {
            for _ in 0..lanes {
                ws.push(f16_finite(rng, 10, 20));
                xs.push(f16_finite(rng, 10, 20));
            }
        }
        2 => {
            for _ in 0..lanes / 2 {
                let w = f16_finite(rng, 8, 22);
                let x = f16_finite(rng, 8, 22);
                ws.push(w);
                ws.push(w);
                xs.push(x);
                xs.push(x ^ 0x8000);
            }
            while ws.len() < lanes {
                ws.push(0);
                xs.push(0);
            }
            if idx & 1 == 1 {
                xs[0] ^= 1;
            }
        }
        3 => {
            for _ in 0..lanes {
                let a = if rng.next() & 1 == 1 { f16_finite(rng, 1, 3) } else { f16_finite(rng, 27, 30) };
                ws.push(a);
                let b = if rng.next() & 1 == 1 { f16_finite(rng, 1, 3) } else { f16_finite(rng, 27, 30) };
                xs.push(b);
            }
        }
        4 => {
            if idx.is_multiple_of(4) {
                for _ in 0..lanes {
                    let s = (rng.next() & 1) as u16;
                    ws.push(s << 15);
                    xs.push((1 - s) << 15);
                }
            } else {
                for _ in 0..lanes {
                    if rng.next() & 1 == 1 {
                        ws.push(((rng.next() & 1) as u16) << 15);
                        xs.push(f16_finite(rng, 0, 30));
                    } else {
                        ws.push(f16_finite(rng, 8, 22));
                        xs.push(f16_finite(rng, 8, 22));
                    }
                }
            }
        }
        _ => {
            for _ in 0..lanes {
                ws.push(f16_finite(rng, 8, 22));
                xs.push(f16_finite(rng, 8, 22));
            }
            let pos = (rng.next() % lanes as u64) as usize;
            let sel = (idx / 6) % 6;
            let sg = ((rng.next() & 1) as u16) << 15;
            match sel {
                0 => ws[pos] = 0x7C00 | sg,
                1 => {
                    ws[pos] = 0x7C00 | sg;
                    xs[pos] = ((rng.next() & 1) as u16) << 15;
                }
                2 => xs[pos] = 0x7E00 | (rng.next() & 0x1FF) as u16,
                3 => {
                    ws[pos] = 0x7C00;
                    xs[pos] = 0x3C00;
                    ws[(pos + 1) % lanes] = 0x7C00;
                    xs[(pos + 1) % lanes] = 0xBC00;
                }
                4 => {
                    ws[pos] = 0x7BFF;
                    xs[pos] = 0x7BFF;
                }
                _ => {
                    ws[pos] = 0x7C00 | sg;
                    xs[pos] = 0x3C00;
                }
            }
        }
    }
    (ws, xs)
}

pub fn dot_f16_of(ws: &[u16], xs: &[u16]) -> u32 {
    let wlo: Vec<u8> = ws.iter().map(|&v| v as u8).collect();
    let whi: Vec<u8> = ws.iter().map(|&v| (v >> 8) as u8).collect();
    let xlo: Vec<u8> = xs.iter().map(|&v| v as u8).collect();
    let xhi: Vec<u8> = xs.iter().map(|&v| (v >> 8) as u8).collect();
    mxm::f16_dot(&wlo, &whi, &xlo, &xhi)
}

pub fn conv_types(op: u32) -> (EType, EType) {
    (lanes::conv_src(op), lanes::conv_dst(op))
}

/// Evaluate a job line ("kind args...", values in hex) with the Rust numerics.
/// `None` when the Ziv loop does not converge (printed as `NC` by the Sail driver).
pub fn eval_job(line: &str) -> Option<u32> {
    let f: Vec<&str> = line.split_whitespace().collect();
    let d = |i: usize| f[i].parse::<u32>().unwrap();
    let h = |i: usize| u32::from_str_radix(f[i], 16).unwrap();
    match f[0] {
        "tr" => transcendental::fp_tr_value(d(1), if d(2) == 16 { fp::FMT16 } else { fp::FMT32 }, h(3)),
        "i8" => Some(transcendental::i8_transcendental(d(1), h(2))),
        "bin" => Some(lanes::k_binary(d(1), ty(d(2)), h(3), h(4))),
        "un" => Some(lanes::apply(lanes::LaneOp::Un(d(1), ty(d(2))), h(3), 0, 0)),
        "cmp" => Some(lanes::k_cmp(d(1), ty(d(2)), h(3), h(4))),
        "cvt" => Some(lanes::k_convert_any(ty(d(1)), ty(d(2)), h(3))),
        "conv" => Some(lanes::k_convert(d(1), h(2))),
        "nz" => Some(lanes::k_nonzero(ty(d(1)), h(2)) as u32),
        k => panic!("unknown job kind {k}"),
    }
}

pub fn reduce_of(rop: u32) -> ReduceOp {
    match rop {
        0 => ReduceOp::Add,
        9 => ReduceOp::Max,
        10 => ReduceOp::Min,
        _ => panic!("reduce op"),
    }
}

/// `reference/` of the crate.
pub fn reference_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("reference")
}

/// Compare every `<job> = <result>` line of a file with the Rust numerics; returns the number of
/// lines and the mismatches (as text, at most 20 kept).
pub fn compare_file(path: &std::path::Path) -> (usize, usize, Vec<String>) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut n = 0;
    let mut bad = 0;
    let mut shown = Vec::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (job, want) = line.split_once(" = ").unwrap_or_else(|| panic!("bad line {line}"));
        let got = match eval_job(job) {
            Some(r) => format!("{r:08x}"),
            None => "NC".to_string(),
        };
        n += 1;
        if got != want {
            bad += 1;
            if shown.len() < 20 {
                shown.push(format!("{job}: rust {got}, reference {want}"));
            }
        }
    }
    (n, bad, shown)
}
