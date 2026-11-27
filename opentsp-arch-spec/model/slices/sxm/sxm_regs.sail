// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// SXM maps: each of the two SXMs (0 = west, 1 = east) holds 16 maps of one 10-bit entry
// per lane. An unloaded map is the identity (microarchitecture.md sec. 7).
// Each map also records, from its latest LoadMap or LoadMapWide, whether it is bijective
// (the entries below vlanes are distinct) and the first cycle a Permute may use it: the
// execute cycle of the load + geometry.superlanes. The
// identity maps of reset are bijective and ready.

type perm_map = vector(vlanes, bits(10))

private function identity_map() -> perm_map = {
  var m : perm_map = vector_init(0b0000000000);
  foreach (l from 0 to (vlanes - 1)) { m[l] = get_slice_int(10, l, 0) };
  m
}

register sxm_maps : vector(2, vector(16, perm_map)) = vector_init(vector_init(vector_init(0b0000000000)))
register sxm_map_bij : vector(2, vector(16, bool)) = vector_init(vector_init(true))
register sxm_map_ready : vector(2, vector(16, nat)) = vector_init(vector_init(0))

// LoadMap changes are staged and applied at the end of the cycle (see mxm_regs.sail).
struct sxm_staged = { q : range(0, 144), sxm : range(0, 1), map : range(0, 15), value : perm_map, bij : bool, ready : nat }

register sxm_stage_list : list(sxm_staged) = [||]

function sxm_reset() -> unit = {
  let id = identity_map();
  sxm_maps = vector_init(vector_init(id));
  sxm_map_bij = vector_init(vector_init(true));
  sxm_map_ready = vector_init(vector_init(0));
  sxm_stage_list = [||]
}
