// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// SXM instructions (ISCA 2020 sec. III-E; microarchitecture.md sec. 7). The distributor (DST)
// works in-row; the transposer (TSP) and permuter/shifter (PRM) read the outward row and
// write the inward one (core/layout.sail).
//
// 64-bit layout: slice 63..60 = 4, opcode 59..54, type 53..52 = i8 (0), dst 51..46,
// src 45..40, param 39..30, src2 29..24, reserved 23..0. Unused fields must be zero.

private function sx() -> range(0, 1) = issue_unit.sxm
// Shift and Permute move lanes across superlanes and have timing class T_SXM_CROSS; every
// other SXM instruction stays inside a superlane and has T_SXM. The issue's latency
// (its catalogue entry's row, core/timing.sail) gives the read and commit cycles.
private function sxm_rd_cycle() -> int = exec_cycle()
private function sxm_cm_cycle() -> nat = commit_at()

private function is_prm(u : unit_kind) -> bool = u == U_SXM_W_PRM | u == U_SXM_E_PRM
private function is_shf(u : unit_kind) -> bool = u == U_SXM_W_SHF | u == U_SXM_E_SHF
private function is_sel(u : unit_kind) -> bool = u == U_SXM_W_SEL | u == U_SXM_E_SEL
private function is_dst(u : unit_kind) -> bool = u == U_SXM_W_DST | u == U_SXM_E_DST
private function is_tsp(u : unit_kind) -> bool = u == U_SXM_W_TSP | u == U_SXM_E_TSP

private function sxm_lane(v : vec, l : int) -> bits(8) =
  if 0 <= l & l < vlanes then slice(v, 8 * l, 8) else 0x00

private function sxm_set_lane(v : vec, l : int, b : bits(8)) -> vec =
  if 0 <= l & l < vlanes then { var w = v; w[8 * l + 7 .. 8 * l] = b; w } else v

// Stream reference d + k on the same row (callers check the stream range first).
private function sxm_stream_plus(Streamref(d) : streamref, k : int) -> bits(6) =
  [d with 4 .. 0 = get_slice_int(5, unsigned(d[4 .. 0]) + k, 0)]

private function sxm_emit(d : streamref, v : vec) -> ExecutionResult =
  match write_planes(d, one_plane(v), 1, sxm_cm_cycle()) {
    Ok(()) => Issued(),
    Err(e) => Diagnostic(e),
  }

// ---------------------------------------------------------------------------------------
// Shift src, dir amount, dst: lanes move across the whole vector (crossing superlanes).
// up: out[l] = in[l - n]; down: out[l] = in[l + n]; zero fill. param = dir << 9 | n.

mapping encdec_shift_dir : shift_dir <-> bits(1) = { SHIFT_UP <-> 0b0, SHIFT_DOWN <-> 0b1 }

union clause instruction = SXM_SHIFT : (shift_dir, bits(9), streamref, streamref)

mapping clause encdec64 = SXM_SHIFT(dir, n, Streamref(s), Streamref(d))
  <-> 0x4 @ 0b000000 @ 0b00 @ d @ s @ encdec_shift_dir(dir) @ n @ 0b000000 @ 0x000000
  when unsigned(n) < vlanes

function clause execute SXM_SHIFT(dir, n, s, d) =
  match read_planes(s, 1, sxm_rd_cycle()) {
    Err(e) => Diagnostic(e),
    Ok(p)  => {
      ecc_dep_all = true;   // an output superlane takes lanes of its neighbours: one bad superlane poisons every output superlane (core/ecc.sail)
      let k = unsigned(n);
      var v : vec = zero_vec;
      foreach (l from 0 to (vlanes - 1)) {
        let j : int = match dir { SHIFT_UP => l - k, SHIFT_DOWN => l + k };
        v = sxm_set_lane(v, l, sxm_lane(p[0], j))
      };
      sxm_emit(d, v)
    },
  }

mapping shift_dir_name : shift_dir <-> string = { SHIFT_UP <-> "up", SHIFT_DOWN <-> "down" }

mapping clause assembly = SXM_SHIFT(dir, n, s, d)
  <-> "shift" ^ spc() ^ stream_name(s) ^ sep() ^ shift_dir_name(dir) ^ sep() ^ hex_bits_9(n) ^ sep() ^ stream_name(d)

// ---------------------------------------------------------------------------------------
// Permute src, map, dst: out[l] = m[l] < vlanes ? in[m[l]] : 0 (the map carries the mask);
// the entries below vlanes must be distinct. The check is made when the map is loaded (the map
// is then "bijective", or not); a map is ready for a Permute timing.sxm_map_ready cycles (ref: superlanes) after the
// execute cycle of its LoadMap (RES-001 before; SXM-001 if it is not bijective; either way the
// Permute has no effect). Distribute stays inside a superlane: its map is usable the next cycle.
// Distribute src, map, dst: per superlane g: e = m[16g + j]; out[16g + j] = e < 16 ? in[16g + e] : 0.

private mapping encdec_mapop : sxm_mapop <-> bits(6) = { MAP_PERMUTE <-> 0b000001, MAP_DISTRIBUTE <-> 0b000010 }

union clause instruction = SXM_MAP : (sxm_mapop, bits(4), streamref, streamref)

mapping clause encdec64 = SXM_MAP(op, m, Streamref(s), Streamref(d))
  <-> 0x4 @ encdec_mapop(op) @ 0b00 @ d @ s @ 0b000000 @ m @ 0b000000 @ 0x000000

private function sxm_map_distinct(m : perm_map) -> bool = {
  var seen : vector(vlanes, bool) = vector_init(false);
  var ok : bool = true;
  foreach (l from 0 to (vlanes - 1)) {
    let e = unsigned(m[l]);
    if e >= vlanes then ()
    else if seen[e] then ok = false
    else seen[e] = true
  };
  ok
}

function clause execute SXM_MAP(op, mi, s, d) = {
  let m = sxm_maps[sx()][unsigned(mi)];
  match read_planes(s, 1, sxm_rd_cycle()) {
    Err(e) => Diagnostic(e),
    Ok(p)  => match op {
      MAP_PERMUTE =>
        if exec_cycle() < sxm_map_ready[sx()][unsigned(mi)] then Diagnostic(RES_NOT_READY)
        else if not(sxm_map_bij[sx()][unsigned(mi)]) then Diagnostic(SXM_PERMUTE) else {
          ecc_dep_all = true;   // a lane may come from any superlane (core/ecc.sail)
          var v : vec = zero_vec;
          foreach (l from 0 to (vlanes - 1)) {
            let e = unsigned(m[l]);
            if e < vlanes then v = sxm_set_lane(v, l, sxm_lane(p[0], e))
          };
          sxm_emit(d, v)
        },
      MAP_DISTRIBUTE => {
        var v : vec = zero_vec;
        foreach (g from 0 to (superlanes - 1)) {
          foreach (j from 0 to 15) {
            let e = unsigned(m[16 * g + j]);
            if e < 16 then v = sxm_set_lane(v, 16 * g + j, sxm_lane(p[0], 16 * g + e))
          }
        };
        sxm_emit(d, v)
      },
    },
  }
}

mapping mapop_name : sxm_mapop <-> string = { MAP_PERMUTE <-> "permute", MAP_DISTRIBUTE <-> "distribute" }

mapping clause assembly = SXM_MAP(op, m, s, d)
  <-> mapop_name(op) ^ spc() ^ stream_name(s) ^ sep() ^ hex_bits_4(m) ^ sep() ^ stream_name(d)

// ---------------------------------------------------------------------------------------
// LoadMap map, src: m[l] = src[l]. LoadMapWide map, src, src2: m[l] = (src2[l] mod 4) * 256 + src[l].

union clause instruction = SXM_LOADMAP : (bits(4), streamref)
union clause instruction = SXM_LOADMAPWIDE : (bits(4), streamref, streamref)

mapping clause encdec64 = SXM_LOADMAP(m, Streamref(s))
  <-> 0x4 @ 0b000101 @ 0b00 @ 0b000000 @ s @ 0b000000 @ m @ 0b000000 @ 0x000000
mapping clause encdec64 = SXM_LOADMAPWIDE(m, Streamref(s), Streamref(s2))
  <-> 0x4 @ 0b000110 @ 0b00 @ 0b000000 @ s @ 0b000000 @ m @ s2 @ 0x000000

private function sxm_load_map(mi : bits(4), lo : vec, hi : vec) -> unit = {
  var m : perm_map = vector_init(0b0000000000);
  foreach (l from 0 to (vlanes - 1)) {
    m[l] = get_slice_int(10, (unsigned(sxm_lane(hi, l)) % 4) * 256 + unsigned(sxm_lane(lo, l)), 0)
  };
  sxm_stage_list = struct { q = issue_queue, sxm = sx(), map = unsigned(mi), value = m, bij = sxm_map_distinct(m),
                            ready = exec_cycle() + timing_sxm_map_ready } :: sxm_stage_list
}

function clause execute SXM_LOADMAP(mi, s) =
  match read_planes(s, 1, sxm_rd_cycle()) {
    Err(e) => Diagnostic(e),
    Ok(p)  => { sxm_load_map(mi, p[0], zero_vec); Issued() },
  }

function clause execute SXM_LOADMAPWIDE(mi, s, s2) =
  match (read_planes(s, 1, sxm_rd_cycle()), read_planes(s2, 1, sxm_rd_cycle())) {
    (Err(e), _)      => Diagnostic(e),
    (_, Err(e))      => Diagnostic(e),
    (Ok(lo), Ok(hi)) => { sxm_load_map(mi, lo[0], hi[0]); Issued() },
  }

mapping clause assembly = SXM_LOADMAP(m, s) <-> "loadmap" ^ spc() ^ hex_bits_4(m) ^ sep() ^ stream_name(s)
mapping clause assembly = SXM_LOADMAPWIDE(m, s, s2)
  <-> "loadmapwide" ^ spc() ^ hex_bits_4(m) ^ sep() ^ stream_name(s) ^ sep() ^ stream_name(s2)

// ---------------------------------------------------------------------------------------
// Rotate src, n, dst (n = 3 or 4): per superlane, lanes 0..n*n-1 form an n x n tile;
// output t = (tr, tc) goes to stream dst + t with out[r*n + c] = in[((r+tr) mod n)*n + (c+tc) mod n];
// other lanes are zero. dst + n*n - 1 must be a valid stream.

union clause instruction = SXM_ROTATE : (bits(10), streamref, streamref)

mapping clause encdec64 = SXM_ROTATE(n, Streamref(s), Streamref(d))
  <-> 0x4 @ 0b000011 @ 0b00 @ d @ s @ n @ 0b000000 @ 0x000000
  when n == 0b0000000011 | n == 0b0000000100

function clause execute SXM_ROTATE(nb, s, d) = {
  let n : {3, 4} = if nb == 0b0000000011 then 3 else 4;
  if stream_id(d) + n * n > 32 then Diagnostic(OPR_ALIGN)
  else match read_planes(s, 1, sxm_rd_cycle()) {
    Err(e) => Diagnostic(e),
    Ok(p)  => {
      var res : ExecutionResult = Issued();
      foreach (t from 0 to (n * n - 1)) {
        let tr = t / n;
        let tc = t % n;
        var v : vec = zero_vec;
        foreach (g from 0 to (superlanes - 1)) {
          foreach (r from 0 to (n - 1)) {
            foreach (c from 0 to (n - 1)) {
              v = sxm_set_lane(v, 16 * g + r * n + c, sxm_lane(p[0], 16 * g + ((r + tr) % n) * n + (c + tc) % n))
            }
          }
        };
        match write_planes(Streamref(sxm_stream_plus(d, t)), one_plane(v), 1, sxm_cm_cycle()) {
          Ok(()) => (),
          Err(e) => res = Diagnostic(e),
        }
      };
      res
    },
  }
}

mapping clause assembly = SXM_ROTATE(n, s, d)
  <-> "rotate" ^ spc() ^ stream_name(s) ^ sep() ^ hex_bits_10(n) ^ sep() ^ stream_name(d)

// ---------------------------------------------------------------------------------------
// Transpose src, dst: 16 input streams (base 0 or 16) -> 16 outputs; per superlane g:
// out_c[16g + r] = in_r[16g + c].

union clause instruction = SXM_TRANSPOSE : (streamref, streamref)

mapping clause encdec64 = SXM_TRANSPOSE(Streamref(s), Streamref(d))
  <-> 0x4 @ 0b000100 @ 0b00 @ d @ s @ 0b0000000000 @ 0b000000 @ 0x000000
  when s[3 .. 0] == 0x0 & d[3 .. 0] == 0x0

function clause execute SXM_TRANSPOSE(s, d) =
  match read_planes16(s, sxm_rd_cycle()) {
    Err(e)   => Diagnostic(e),
    Ok(rows) => {
      var res : ExecutionResult = Issued();
      foreach (c from 0 to 15) {
        var v : vec = zero_vec;
        foreach (g from 0 to (superlanes - 1)) {
          foreach (r from 0 to 15) { v = sxm_set_lane(v, 16 * g + r, sxm_lane(rows[r], 16 * g + c)) }
        };
        match write_planes(Streamref(sxm_stream_plus(d, c)), one_plane(v), 1, sxm_cm_cycle()) {
          Ok(()) => (),
          Err(e) => res = Diagnostic(e),
        }
      };
      res
    },
  }

mapping clause assembly = SXM_TRANSPOSE(s, d) <-> "transpose" ^ spc() ^ stream_name(s) ^ sep() ^ stream_name(d)

// ---------------------------------------------------------------------------------------
// SxSelect a, b, c, sel, dst: three-way select (ISCA 2020 Fig. 8: the select
// combines the unshifted vector with the north- and south-shifted ones). Per lane:
// sel[l] = 0 -> a[l] (unshifted), 1 -> b[l] (north), 2 -> c[l] (south); a lane with any other
// sel value is zero and the instruction reports one OTS-E-OPR-001 (the other lanes are written). The mnemonic is sxselect (VXM has its own select).
// Layout: opcode 7, dst 51..46, a 45..40, b 29..24, sel 23..18, c 17..12, reserved 11..0.

union clause instruction = SXM_SELECT : (streamref, streamref, streamref, streamref, streamref)

mapping clause encdec64 = SXM_SELECT(Streamref(a), Streamref(b), Streamref(c), Streamref(m), Streamref(d))
  <-> 0x4 @ 0b000111 @ 0b00 @ d @ a @ 0b0000000000 @ b @ m @ c @ 0b000000000000

function clause execute SXM_SELECT(a, b, c, m, d) =
  match (read_planes(a, 1, sxm_rd_cycle()), read_planes(b, 1, sxm_rd_cycle()), read_planes(c, 1, sxm_rd_cycle()), read_planes(m, 1, sxm_rd_cycle())) {
    (Err(e), _, _, _) => Diagnostic(e),
    (_, Err(e), _, _) => Diagnostic(e),
    (_, _, Err(e), _) => Diagnostic(e),
    (_, _, _, Err(e)) => Diagnostic(e),
    (Ok(pa), Ok(pb), Ok(pc), Ok(pm)) => {
      // DIAG-EFFECT: sxselect per-lane (an exception to "a diagnostic means no effect", microarchitecture.md section 8).
      // Per lane: a lane whose sel is above 2 is zero, the others are selected and written,
      // and the instruction reports one OPR-001 (an exception to "a diagnostic means no effect").
      var v : vec = zero_vec;
      var bad : bool = false;
      foreach (l from 0 to (vlanes - 1)) {
        let s = sxm_lane(pm[0], l);
        if s == 0x00 then v = sxm_set_lane(v, l, sxm_lane(pa[0], l))
        else if s == 0x01 then v = sxm_set_lane(v, l, sxm_lane(pb[0], l))
        else if s == 0x02 then v = sxm_set_lane(v, l, sxm_lane(pc[0], l))
        else bad = true
      };
      let r : ExecutionResult = sxm_emit(d, v);
      if bad then Diagnostic(OPR_RANGE) else r
    },
  }

mapping clause assembly = SXM_SELECT(a, b, c, m, d)
  <-> "sxselect" ^ spc() ^ stream_name(a) ^ sep() ^ stream_name(b) ^ sep() ^ stream_name(c) ^ sep() ^ stream_name(m) ^ sep() ^ stream_name(d)

function clause timing_of SXM_SELECT(_) = T_SXM
function clause issued_by(SXM_SELECT(_), u) = is_sel(u)

// ---------------------------------------------------------------------------------------

function clause timing_of SXM_ROTATE(_) = T_SXM
function clause timing_of SXM_TRANSPOSE(_) = T_SXM
function clause issued_by(SXM_ROTATE(_), u) = is_tsp(u)
function clause issued_by(SXM_TRANSPOSE(_), u) = is_tsp(u)

function clause timing_of SXM_SHIFT(_) = T_SXM_CROSS
function clause timing_of SXM_MAP(MAP_PERMUTE, _, _, _) = T_SXM_CROSS
function clause timing_of SXM_MAP(MAP_DISTRIBUTE, _, _, _) = T_SXM
function clause timing_of SXM_LOADMAP(_) = T_SXM
function clause timing_of SXM_LOADMAPWIDE(_) = T_SXM

function clause entry_of SXM_SELECT(_) = "SelectS"
function clause entry_of SXM_ROTATE(_) = "Rotate"
function clause entry_of SXM_TRANSPOSE(_) = "Transpose"
function clause entry_of SXM_SHIFT(_) = "Shift"
function clause entry_of SXM_MAP(MAP_PERMUTE, _, _, _) = "Permute"
function clause entry_of SXM_MAP(MAP_DISTRIBUTE, _, _, _) = "Distribute"
function clause entry_of SXM_LOADMAP(_) = "LoadMap"
function clause entry_of SXM_LOADMAPWIDE(_) = "LoadMapWide"

function clause issued_by(SXM_SHIFT(_), u) = is_shf(u)
function clause issued_by(SXM_MAP(MAP_PERMUTE, _, _, _), u) = is_prm(u)
function clause issued_by(SXM_MAP(MAP_DISTRIBUTE, _, _, _), u) = is_dst(u)
function clause issued_by(SXM_LOADMAP(_), u) = is_prm(u) | is_dst(u)
function clause issued_by(SXM_LOADMAPWIDE(_), u) = is_prm(u) | is_dst(u)
