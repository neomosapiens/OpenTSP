// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Obligations on the wide accumulator (extension wide_acc, ; this suite needs the 256-bit SMT integer width: a lane is 128 bits): the read-out of a lane rounds the exact sum once, as the f32 ACC does,
// and the lane addition keeps a special value sticky.

function wlane_of(sp : bits(2), nz : bool, v : bits(8)) -> wlane = struct { wv = signed(v), wsp = unsigned(sp), wnz = nz }

// A lane read out equals the f32 rounding of the same exact sum (§6.1: the wide read-out rounds once, like a replacing ACC f32); the sum is a signed 8-bit value here because a query over a
// symbolic significand costs minutes (the same cost note as numerics.sail).
$property
function p_wide_replace(nan : bool, pinf : bool, ninf : bool, nz : bool, v : bits(8)) -> bool = {
  let d : dsum = struct { dnan = nan, dpinf = pinf, dninf = ninf, dnz = nz, dsum = signed(v) };
  wlane_read(dsum_wide(d)) == dsum_round(d)
}

// Specials are sticky under addition: NaN, or +Inf with -Inf, gives NaN; an infinity stays an infinity beside a finite lane.
$property
function p_wide_sticky(sa : bits(2), sb : bits(2), na : bool, nb : bool, va : bits(8), vb : bits(8)) -> bool = {
  let a = wlane_of(sa, na, va);
  let b = wlane_of(sb, nb, vb);
  let r = wlane_add(a, b);
  let nan = unsigned(sa) == 3 | unsigned(sb) == 3 | (unsigned(sa) == 1 & unsigned(sb) == 2) | (unsigned(sa) == 2 & unsigned(sb) == 1);
  if nan then r.wsp == 3
  else if unsigned(sa) != 0 then r.wsp == unsigned(sa)
  else if unsigned(sb) != 0 then r.wsp == unsigned(sb)
  else r.wsp == 0 & r.wnz == (na & nb)
}

// Finite lanes wrap at 128 bits (two's complement); the specials of p_wide_sticky are excluded by wsp = 0.
$property
// Inputs are signed 8-bit like p_wide_replace (symbolic bits(128) times out under load); wrap_s still uses width 128.
function p_wide_wrap(va : bits(8), vb : bits(8), na : bool, nb : bool) -> bool = {
  let a : wlane = struct { wv = signed(va), wsp = 0, wnz = na };
  let b : wlane = struct { wv = signed(vb), wsp = 0, wnz = nb };
  let r = wlane_add(a, b);
  r.wsp == 0 & r.wv == wrap_s(signed(va) + signed(vb), 128) & r.wnz == (na & nb)
}
