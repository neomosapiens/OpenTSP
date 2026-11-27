//! A byte copy for the vectors of the stream register file and the SRAM: lengths of one to a few hundred bytes, almost always
//! a multiple of 16, copied in 64-byte blocks with vector loads and stores instead of a call to the C library's `memmove`
//! (the call costs more than the 128 bytes it moves).

/// `dst.copy_from_slice(src)`: panics when the lengths differ.
#[inline(always)]
pub fn copy(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len(), "copy: lengths differ");
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::{vld1q_u8, vst1q_u8};
        let n = dst.len();
        let (d, s) = (dst.as_mut_ptr(), src.as_ptr());
        let mut i = 0;
        // SAFETY: both slices hold `n` bytes, every access below stays inside `i..i + 64` (resp. `i..i + 16`) with
        // `i + 64 <= n` (resp. `i + 16 <= n`), and `dst` (a `&mut`) cannot overlap `src`; NEON is part of the aarch64
        // baseline, and `vld1q_u8` / `vst1q_u8` have no alignment requirement.
        unsafe {
            while i + 64 <= n {
                let (a, b, c, e) =
                    (vld1q_u8(s.add(i)), vld1q_u8(s.add(i + 16)), vld1q_u8(s.add(i + 32)), vld1q_u8(s.add(i + 48)));
                vst1q_u8(d.add(i), a);
                vst1q_u8(d.add(i + 16), b);
                vst1q_u8(d.add(i + 32), c);
                vst1q_u8(d.add(i + 48), e);
                i += 64;
            }
            while i + 16 <= n {
                vst1q_u8(d.add(i), vld1q_u8(s.add(i)));
                i += 16;
            }
        }
        while i < n {
            dst[i] = src[i];
            i += 1;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    dst.copy_from_slice(src);
}
