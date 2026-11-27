//! The integer lane kernels (`k_binary`, opcodes 0..13, integer element types) against the exact
//! i128 arithmetic of the model's functions: every opcode and type on edge values and random pairs.
mod common;

use common::*;
use tsp_numerics::lanes::{self, k_binary};
use tsp_numerics::EType;

const INT_TYPES: [EType; 6] = [EType::I8, EType::I16, EType::I32, EType::U8, EType::U16, EType::U32];

/// `lanes.sail k_binary` for the integer types, in i128: wrap, saturate and encode as the model does.
fn reference(op: u32, t: EType, x: u32, y: u32) -> u32 {
    let a = lanes::int_of(t, x);
    let b = lanes::int_of(t, y);
    let mask = lanes::all_ones(t);
    match op {
        0 | 2 => lanes::bits_t(t, lanes::wrap_t(t, a + b)),
        1 => lanes::bits_t(t, lanes::sat_t(t, a + b)),
        3 | 5 => lanes::bits_t(t, lanes::wrap_t(t, a - b)),
        4 => lanes::bits_t(t, lanes::sat_t(t, a - b)),
        6 | 8 => lanes::bits_t(t, lanes::wrap_t(t, a * b)),
        7 => lanes::bits_t(t, lanes::sat_t(t, a * b)),
        9 => lanes::bits_t(t, a.max(b)),
        10 => lanes::bits_t(t, a.min(b)),
        11 => (x & y) & mask,
        12 => (x | y) & mask,
        13 => (x ^ y) & mask,
        _ => 0,
    }
}

#[test]
fn integer_binary_kernels_match_the_exact_arithmetic() {
    let edges = [
        0u32,
        1,
        2,
        0x7f,
        0x80,
        0x81,
        0xff,
        0x100,
        0x7fff,
        0x8000,
        0x8001,
        0xffff,
        0x1_0000,
        0x7fff_ffff,
        0x8000_0000,
        0x8000_0001,
        0xffff_fffe,
        0xffff_ffff,
    ];
    let mut rng = Rng(7);
    for t in INT_TYPES {
        for op in 0..=13 {
            for &x in &edges {
                for &y in &edges {
                    assert_eq!(k_binary(op, t, x, y), reference(op, t, x, y), "{t:?} op {op} {x:#x} {y:#x}");
                }
            }
            for _ in 0..200_000 {
                let (x, y) = (rng.next() as u32, rng.next() as u32);
                assert_eq!(k_binary(op, t, x, y), reference(op, t, x, y), "{t:?} op {op} {x:#x} {y:#x}");
            }
        }
    }
}

/// The vector kernel of the 8-bit integer types, on every pair of elements.
#[test]
fn bin8_matches_k_binary_on_every_pair() {
    let a: Vec<u8> = (0..65536u32).map(|i| (i >> 8) as u8).collect();
    let b: Vec<u8> = (0..65536u32).map(|i| i as u8).collect();
    let mut out = vec![0u8; 65536];
    for (t, signed) in [(EType::I8, true), (EType::U8, false)] {
        for op in 0..=13 {
            lanes::bin8(op, signed, &a, &b, &mut out);
            for i in 0..65536 {
                assert_eq!(out[i] as u32, k_binary(op, t, a[i] as u32, b[i] as u32), "{t:?} op {op} {} {}", a[i], b[i]);
            }
        }
    }
}
