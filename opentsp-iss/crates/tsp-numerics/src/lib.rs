//! Bit-exact numerics of the opentsp ISS for specification 0.1.
//!
//! A port of the Sail model's `core/numerics/{fp,lanes,transcendental}.sail`, function for
//! function: every ported function carries a `// model: core/numerics/fp.sail fp_add`
//! tag, and the Sail argument order is kept. Results equal the model bit for bit.
//!
//! | Module | Model source | Contents |
//! |---|---|---|
//! | [`types`] | `core/types.sail` | the element types |
//! | [`fp`] | `fp.sail` | exact f16/f32 arithmetic, comparison, conversion, rounding |
//! | [`lanes`] | `lanes.sail`, `vxm_insts.sail` | per-lane VXM kernels, conversions, `apply`, Reduce fold |
//! | [`transcendental`] | `transcendental.sail` | TanH/Exp/RSqrt/Log: f64 fast path, memoised tables |
//! | [`ziv`] | `transcendental.sail` | the straight big-integer port (reference and fallback) |
//! | [`bigfp`] | `fp.sail` | exact arithmetic on big integers (reference for the fast paths) |
//! | [`planes`] | (none) | the VXM lane kernels on byte planes, NEON on AArch64, equal to `lanes::apply` |
//! | [`mxm`] | `mxm_insts.sail` | the exact f16 dot product and the integer dot products |
//! | [`hostfp`] | (none) | the start-up check of the host floating-point environment |
//! | [`fault`] | `errors.sail` | the model fault (`internal_error`) raised below the lane kernels |
//!
//! # Platform assumptions
//!
//! Two shortcuts rely on the host, and each has a check that runs in the normal `cargo test`:
//!
//! - **f32 add, sub and mul run on the host FPU** (`fp_add`, `fp_sub`, `fp_mul`, and so the f32
//!   accumulate of the MXM and the VXM f32 operations). That equals the model on an IEEE binary32 FPU
//!   in its default environment: round to nearest even, gradual underflow, f32 exponent range. The
//!   build refuses targets other than aarch64, x86_64 and x86 with SSE2 (the x87 FPU has no f32 exponent
//!   range and rounds twice), and [`hostfp::check_host_fpu`] refuses a thread whose control register
//!   was changed (flush-to-zero, denormals-are-zero, another rounding mode). `tests/hostfp.rs` runs the
//!   subnormal, underflow and overflow-tie cases against the exact big-integer version.
//! - **The transcendental fast path calls the host libm** (`exp`, `ln`, `tanh`, `sqrt` in `f64`) and
//!   keeps its answer only when it is more than 2^-40 (relative) from every rounding boundary of the
//!   destination: it assumes the libm is within a few 2^-53 relative, which every libm that meets its
//!   documented accuracy does (the margin is about 2^11). i8 and f16 are checked exhaustively on the
//!   running host; for f32, `tests/f32_boundary.rs` runs up to 50 000 inputs per function closest to a
//!   boundary against the mpmath oracle (`tools/gen_f32_boundary.py` made `tests/data/f32_boundary.bin`),
//!   and `just iss-transcendental` sweeps all 2^32 inputs against a f32 screening work directory.

pub mod bigfp;
pub mod fault;
pub mod fp;
pub mod hostfp;
pub mod lanes;
pub mod mxm;
pub mod planes;
pub mod transcendental;
pub mod types;
pub mod ziv;

pub use types::EType;
