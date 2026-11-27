//! The host floating-point environment: f32 `fp_add`, `fp_sub` and `fp_mul` run on the host FPU, so the
//! start-up self-check (`tsp_numerics::hostfp`) must see an IEEE binary32, round-to-nearest-even
//! environment with gradual underflow, and these edge cases (subnormals, underflow, ties at the
//! overflow threshold) must equal the exact big-integer versions and the hand values below.
use tsp_numerics::bigfp::{fp_add_big, fp_mul_big, fp_sub_big};
use tsp_numerics::fp::{fp_add, fp_mul, fp_sub, FMT32};
use tsp_numerics::hostfp;

const MIN_SUB: u32 = 0x0000_0001; // 2^-149
const MIN_NORMAL: u32 = 0x0080_0000; // 2^-126
const MAX: u32 = 0x7f7f_ffff;
const INF: u32 = 0x7f80_0000;
const HALF: u32 = 0x3f00_0000;
const ONE: u32 = 0x3f80_0000;

#[test]
fn subnormal_edges_equal_the_exact_values() {
    // (x, y, expected): exact values worked out by hand
    let add: [(u32, u32, u32); 4] = [
        (MIN_SUB, MIN_SUB, 0x0000_0002),         // min subnormal + min subnormal
        (0x007f_ffff, MIN_SUB, MIN_NORMAL),      // largest subnormal + min subnormal carries into the normals
        (MIN_NORMAL, 0x8000_0001, 0x007f_ffff),  // min normal - min subnormal is the largest subnormal
        (0x8000_0001, 0x8000_0001, 0x8000_0002), // negative operands
    ];
    for (x, y, want) in add {
        assert_eq!(fp_add(FMT32, x, y), want, "{x:08x} + {y:08x}");
        assert_eq!(fp_add_big(FMT32, x, y), want, "big {x:08x} + {y:08x}");
    }
    // normal - normal giving a subnormal
    assert_eq!(fp_sub(FMT32, 0x0080_0001, MIN_NORMAL), MIN_SUB);
    assert_eq!(fp_sub_big(FMT32, 0x0080_0001, MIN_NORMAL), MIN_SUB);
    // equal operands: +0 in round-to-nearest, whatever their size
    assert_eq!(fp_sub(FMT32, 0x0080_0001, 0x0080_0001), 0);
    // products that underflow to a subnormal, to a tie, to the smallest subnormal and to zero
    let mul: [(u32, u32, u32); 8] = [
        (MIN_NORMAL, HALF, 0x0040_0000),         // 2^-126 * 1/2 = 2^-127 (subnormal)
        (MIN_SUB, ONE, MIN_SUB),                 // a subnormal operand is not flushed
        (MIN_SUB, HALF, 0),                      // 2^-150: the tie between 0 and 2^-149 goes to even (0)
        (MIN_SUB, 0x3f40_0000, MIN_SUB),         // 0.75 * 2^-149 rounds up to 2^-149
        (0x2000_0000, 0x2000_0000, MIN_NORMAL),  // 2^-63 * 2^-63 = 2^-126, the smallest normal
        (0x1f80_0000, 0x1f80_0000, 0x0020_0000), // 2^-64 * 2^-64 = 2^-128 (subnormal)
        (0x0c00_0000, 0x0c00_0000, 0),           // 2^-103 * 2^-103 underflows to zero
        (0x007f_ffff, 0x3f80_0001, MIN_NORMAL),  // the largest subnormal times 1 + 2^-23 rounds up into the normals
    ];
    for (i, (x, y, want)) in mul.into_iter().enumerate() {
        assert_eq!(fp_mul(FMT32, x, y), want, "case {i}: {x:08x} * {y:08x}");
        assert_eq!(fp_mul_big(FMT32, x, y), want, "big case {i}: {x:08x} * {y:08x}");
    }
    // negative zero from an underflow keeps its sign
    assert_eq!(fp_mul(FMT32, 0x8000_0001, HALF), 0x8000_0000);
}

#[test]
fn ties_at_the_overflow_threshold() {
    // max + 2^103 is exactly half an ulp above the largest finite value: the tie goes to the even
    // neighbour, 2^128 (which is Inf); just below the tie it stays at max
    let half_ulp = 0x7300_0000; // 2^103
    assert_eq!(fp_add(FMT32, MAX, half_ulp), INF);
    assert_eq!(fp_add_big(FMT32, MAX, half_ulp), INF);
    assert_eq!(fp_add(FMT32, MAX, half_ulp - 1), MAX);
    assert_eq!(fp_add_big(FMT32, MAX, half_ulp - 1), MAX);
    assert_eq!(fp_sub(FMT32, 0xff7f_ffff, 0x7300_0000), 0xff80_0000, "-max - 2^103 is -Inf");
    // max * (1 + 2^-23) is above the tie
    assert_eq!(fp_mul(FMT32, MAX, 0x3f80_0001), INF);
    assert_eq!(fp_mul_big(FMT32, MAX, 0x3f80_0001), INF);
    assert_eq!(fp_mul(FMT32, MAX, ONE), MAX);
}

#[test]
fn rounding_to_nearest_even_in_the_normal_range() {
    // 1 + 2^-24 is a tie (goes to 1); 1 + 2^-23 + 2^-24 is a tie (goes to 1 + 2^-22)
    assert_eq!(fp_add(FMT32, ONE, 0x3380_0000), ONE);
    assert_eq!(fp_add(FMT32, 0x3f80_0001, 0x3380_0000), 0x3f80_0002);
    // 1 + (2^-24 + 2^-47) is above the tie
    assert_eq!(fp_add(FMT32, ONE, 0x3380_0001), 0x3f80_0001);
}

#[test]
fn the_self_check_passes_in_the_default_environment() {
    assert_eq!(hostfp::check_host_fpu(), Ok(()));
}

/// Set flush-to-zero in the calling thread's floating-point control register; returns the old value.
#[cfg(target_arch = "aarch64")]
fn set_ftz(on: bool) -> u64 {
    let old: u64;
    // SAFETY: reads and writes FPCR of this thread only (bit 24 is FZ); no memory is touched.
    unsafe {
        std::arch::asm!("mrs {0}, fpcr", out(reg) old, options(nomem, nostack));
        let new = if on { old | (1 << 24) } else { old & !(1 << 24) };
        std::arch::asm!("msr fpcr, {0}", in(reg) new, options(nomem, nostack));
    }
    old
}

/// Set flush-to-zero and denormals-are-zero in MXCSR; returns the old value.
#[cfg(target_arch = "x86_64")]
fn set_ftz(on: bool) -> u64 {
    let mut old: u32 = 0;
    // SAFETY: reads and writes MXCSR of this thread only (FTZ is bit 15, DAZ bit 6).
    unsafe {
        std::arch::asm!("stmxcsr [{0}]", in(reg) &mut old, options(nostack));
        let new = if on { old | (1 << 15) | (1 << 6) } else { old & !((1 << 15) | (1 << 6)) };
        std::arch::asm!("ldmxcsr [{0}]", in(reg) &new, options(nostack, readonly));
    }
    old as u64
}

/// `x * y` as one multiply instruction that the compiler cannot move across the `set_ftz` asm blocks: floating-point operations are pure
/// to the optimiser and know nothing of FPCR/MXCSR, so a plain `*` can be hoisted above the write that switches flush-to-zero on (it was, in the
/// dev profile: the multiply ran before FZ was set and the test failed with "this host does not flush").
#[cfg(target_arch = "aarch64")]
fn mul_in_the_current_environment(x: f32, y: f32) -> f32 {
    let r: f32;
    // SAFETY: one register-to-register multiply; no memory is touched.
    unsafe {
        std::arch::asm!("fmul {0:s}, {1:s}, {2:s}", out(vreg) r, in(vreg) x, in(vreg) y, options(nomem, nostack))
    };
    r
}

#[cfg(target_arch = "x86_64")]
fn mul_in_the_current_environment(x: f32, y: f32) -> f32 {
    let mut r = x;
    // SAFETY: one register-to-register multiply; no memory is touched.
    unsafe { std::arch::asm!("mulss {0}, {1}", inout(xmm_reg) r, in(xmm_reg) y, options(nomem, nostack)) };
    r
}

#[test]
fn flush_to_zero_is_detected() {
    set_ftz(true);
    let r = hostfp::check_host_fpu();
    // the host really flushes, so the edge cases above would have been wrong
    let flushed = mul_in_the_current_environment(f32::from_bits(MIN_NORMAL), 0.5f32);
    set_ftz(false);
    assert_eq!(flushed.to_bits(), 0, "this host does not flush subnormals under FZ/FTZ: the test cannot show anything");
    let msg = r.expect_err("a flush-to-zero environment must fail the self-check");
    assert!(msg.contains("floating-point"), "{msg}");
    assert_eq!(hostfp::check_host_fpu(), Ok(()), "restored");
}
