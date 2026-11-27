//! The host floating-point environment that f32 `fp_add`, `fp_sub` and `fp_mul` rely on.
//!
//! Those three run on the host FPU, which equals the exact version of `fp.sail` only for IEEE
//! binary32 operations in round-to-nearest-even with gradual underflow: SSE2 on x86 and AArch64
//! in the default environment. The build refuses other targets; [`check_host_fpu`] refuses an
//! environment that a loaded library or the caller changed (flush-to-zero, denormals-are-zero,
//! another rounding mode). The control registers are per thread, so a thread that runs the model
//! checks for itself (`tsp_iss::Machine::run_until` does).

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", all(target_arch = "x86", target_feature = "sse2"))))]
compile_error!(
    "tsp-numerics runs f32 add, sub and mul on the host FPU: it supports aarch64 and x86_64 (x86 only with SSE2); \
     the x87 FPU rounds twice and has no f32 exponent range"
);

use std::hint::black_box;

/// One case: `x op y` must give `want`, all as f32 bit patterns.
struct Case {
    what: &'static str,
    op: fn(f32, f32) -> f32,
    x: u32,
    y: u32,
    want: u32,
}

fn add(x: f32, y: f32) -> f32 {
    x + y
}

fn sub(x: f32, y: f32) -> f32 {
    x - y
}

fn mul(x: f32, y: f32) -> f32 {
    x * y
}

const CASES: [Case; 9] = [
    Case { what: "subnormal + subnormal", op: add, x: 0x0000_0001, y: 0x0000_0001, want: 0x0000_0002 },
    Case { what: "normal - normal giving a subnormal", op: sub, x: 0x0080_0001, y: 0x0080_0000, want: 0x0000_0001 },
    Case { what: "underflow to a subnormal", op: mul, x: 0x0080_0000, y: 0x3f00_0000, want: 0x0040_0000 },
    Case { what: "subnormal * 1", op: mul, x: 0x0000_0001, y: 0x3f80_0000, want: 0x0000_0001 },
    Case { what: "underflow to zero (tie to even)", op: mul, x: 0x0000_0001, y: 0x3f00_0000, want: 0 },
    Case { what: "underflow rounding up to the smallest subnormal", op: mul, x: 0x0000_0001, y: 0x3f40_0000, want: 1 },
    Case { what: "tie at the overflow threshold", op: add, x: 0x7f7f_ffff, y: 0x7300_0000, want: 0x7f80_0000 },
    Case { what: "tie to even in the normal range", op: add, x: 0x3f80_0001, y: 0x3380_0000, want: 0x3f80_0002 },
    Case { what: "just above a tie", op: add, x: 0x3f80_0000, y: 0x3380_0001, want: 0x3f80_0001 },
];

/// Run the self-check: `Err` names the first case the host FPU gets wrong.
pub fn check_host_fpu() -> Result<(), String> {
    for c in &CASES {
        let got = (c.op)(black_box(f32::from_bits(c.x)), black_box(f32::from_bits(c.y))).to_bits();
        if got != c.want {
            return Err(format!(
                "host floating-point environment is not IEEE binary32 round-to-nearest with gradual underflow \
                 (flush-to-zero, denormals-are-zero or another rounding mode?): {} gives {got:08x}, expected {:08x}",
                c.what, c.want
            ));
        }
    }
    Ok(())
}
