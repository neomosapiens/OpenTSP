"""The per-chip placement planner of the GPT-2 small SHIP-style pipeline ( and Appendix A).

It answers four questions for one tile width (the lanes of one vector the programs use: 256 lanes: four heads per vector, 128 lanes: two heads per vector):

1. words of one layer (matrices, f32 vectors, the K and V pool) and of the tied table and the position table;
2. the partition of the layers over N chips with the design's rule (aggregate SRAM, a reserve of 15 % for text, activations and scratch: an estimate),
   linear or ring, the smallest N that fits;
3. the stricter constraint of the generators in this repository: an MXM install reads its weights from 16 slices of one hemisphere
   (`opentsp_tools.programs.kernels.install.weight_planes`), so a matrix lives on the 16 slices of its hemisphere (`groups` such groups per hemisphere), and the K pool
   lives on the east plane's slices: `plane_fit` checks the words per slice of both hemispheres;
4. the host's map vectors of one step (K scatter: one per tile and slice pair, V scatter: one, shared by the tiles) and their cost in
   `host.cycles_per_vector` cycles.

Pure Python (standard library); `python3 gpt2s_plan.py [config.json]` prints the report. Figures are arithmetic on the layout; nothing here is measured."""
import json
import math
import os
import sys
from dataclasses import dataclass

LANES_MAX = 320
PAIRS = 8                    # slice pairs of the K plane (16 slices, 2 byte planes of an f16 tile each)
from opentsp_tools.paths import SPEC
DEFAULT_CONFIG = os.path.join(SPEC, "config", "ref.json")


@dataclass(frozen=True)
class Gpt2Dims:
    d: int = 768
    layers: int = 12
    ffn: int = 3072
    vocab: int = 50257
    npos: int = 1024


@dataclass(frozen=True)
class Layout:
    lanes: int                    # lanes of one tile (vector)
    v_planes: int = 2             # byte planes of a V element: 2 (f16, the pool of the programs) or 4 (f32, the design's script)

    def tiles(self, n):
        return math.ceil(n / self.lanes)

    def mat(self, din, dout):
        """Words of a d_in x d_out f16 matrix: d_out rows of ceil(d_in / lanes) f16 vectors of 2 byte planes."""
        return 2 * dout * self.tiles(din)

    def f32vec(self, n):
        return 4 * self.tiles(n)

    def matrix_words(self, m):
        return self.mat(m.d, 3 * m.d) + self.mat(m.d, m.d) + self.mat(m.d, m.ffn) + self.mat(m.ffn, m.d)

    def vector_words(self, m):
        return self.f32vec(3 * m.d) + 2 * self.f32vec(m.d) + self.f32vec(m.ffn) + 4 * self.f32vec(m.d)

    def layer_words(self, m):
        return self.matrix_words(m) + self.vector_words(m)

    def k_words(self, m, pool):
        return 2 * self.tiles(m.d) * pool

    def v_words(self, m, pool):
        return self.v_planes * self.tiles(m.d) * pool

    def kv_words(self, m, pool):
        return self.k_words(m, pool) + self.v_words(m, pool)

    def table_words(self, m):
        return 2 * m.vocab * self.tiles(m.d)

    def wpe_words(self, m):
        return 4 * m.npos * self.tiles(m.d)

    def map_vectors(self, m):
        """Host map vectors of one layer region and step: K scatter 1 per tile and slice pair, V scatter 1 (all the tiles write the same word)."""
        return PAIRS * self.tiles(m.d) + 1


def fits(loads0, per_layer, n_layers, n, cap):
    loads, layers, c = list(loads0), [0] * n, 0
    for _ in range(n_layers):
        while c < n and loads[c] + per_layer > cap:
            c += 1
        if c == n:
            return None
        loads[c] += per_layer
        layers[c] += 1
    return layers, loads


def partition(lay, m, pool, topology, words_chip, reserve=0.15):
    """The design's rule (Appendix A): the chips' aggregate SRAM at `1 - reserve`; chip 0 holds the table and the position table, the last chip of a
    linear chain a second table copy (the ring's head reads chip 0's). Returns (N, layers per chip, fullest chip's share of the SRAM)."""
    cap = int(words_chip * (1 - reserve))
    per_layer = lay.layer_words(m) + lay.kv_words(m, pool)
    for n in range(1, 64):
        pre = [0] * n
        pre[0] += lay.table_words(m) + lay.wpe_words(m)
        if topology == "linear":
            if n < 2:
                continue
            pre[-1] += lay.table_words(m)
        if fits(pre, per_layer, m.layers, n, cap) is None:
            continue
        lo, hi = max(pre), cap
        while lo < hi:
            mid = (lo + hi) // 2
            if fits(pre, per_layer, m.layers, n, mid):
                hi = mid
            else:
                lo = mid + 1
        layers, loads = fits(pre, per_layer, m.layers, n, lo)
        return n, layers, max(loads) / words_chip
    raise ValueError("no partition")


def plane_fit(lay, m, pool, layers, words_per_slice, groups=1, reserve=0.15):
    """The generators' constraint for `layers` layers on one chip: words per slice of the west matrix group, the east group (half the matrices plus the
    whole K pool), against the usable words of a slice. `groups` = groups of 16 slices a hemisphere offers the installs (today's generators use 1).
    Returns {west, east, usable, fits}: west and east in words per slice."""
    usable = int(words_per_slice * (1 - reserve))
    slices = 16 * groups
    half = lay.matrix_words(m) / 2
    west = layers * half / slices
    east = layers * (half + lay.k_words(m, pool)) / slices
    return {"west": west, "east": east, "usable": usable, "fits": max(west, east) <= usable}


def layers_per_chip(lay, m, pool, words_per_slice, groups=1, reserve=0.15):
    n = 0
    while plane_fit(lay, m, pool, n + 1, words_per_slice, groups, reserve)["fits"]:
        n += 1
    return n


def chips_for_layers(lay, m, pool, words_per_slice, groups=1, reserve=0.15):
    """Chips the layers alone need under `plane_fit` (the table and head not included)."""
    per = layers_per_chip(lay, m, pool, words_per_slice, groups, reserve)
    return math.ceil(m.layers / per) if per else None


def stage_regions(lay, m, pool, layers=1, words_per_slice=8192, wbase=0x200, k_base=0x1400, v_base=0x100, text_floor=6144):
    """The regions of one chip's stage program (`opentsp-vllm/src/opentsp_vllm/models/gpt2/layer.py`), as (slice class, region, first word, end word) with the bank of the
    first word: the weights of the layers on the 16 slices of each plane (bank 0 from `wbase`; the matrices of one layer are `matrix_words / 32` words of
    every plane slice), the K pool on the east plane (bank 1 from `k_base`, `k_words / 16` words of every slice), the V pool on east slices 76..87
    (bank 0 from `v_base`, `pool` words of every slice) and the instruction pager's text (from `text_floor` to the end of the slice). Returns
    (regions, overlaps): the list and the pairs of regions of one slice class that overlap or leave the slice."""
    w = layers * lay.matrix_words(m) // 32
    regions = [("west plane", "weights", wbase, wbase + w),
               ("east plane", "weights", wbase, wbase + w),
               ("east plane", "K pool", k_base, k_base + lay.k_words(m, pool) // 16),
               ("V slices", "V pool", v_base, v_base + pool),
               ("west plane", "text", text_floor, words_per_slice),
               ("east plane", "text", text_floor, words_per_slice),
               ("V slices", "text", text_floor, words_per_slice)]
    bad = [(a[:2], b[:2]) for i, a in enumerate(regions) for b in regions[i + 1:] if a[0] == b[0] and a[2] < b[3] and b[2] < a[3]]
    bad += [(r[:2], "beyond the slice") for r in regions if r[3] > words_per_slice]
    return regions, bad


def host_step(lay, m, layers_on_chip, cycles_per_vector):
    """(vectors, cycles) the host sends a chip for one step: the map vectors of its layer regions (the token, position and page-encoding vectors of a
    stage are a handful more and are not counted here)."""
    n = lay.map_vectors(m) * layers_on_chip
    return n, n * cycles_per_vector


def report(cfg_path=DEFAULT_CONFIG, lanes_list=(256, 128), pool=1024):
    cfg = json.load(open(cfg_path))
    wps = cfg["sram"]["words_per_slice"]
    words_chip = cfg["sram"]["slices"] * wps
    cpv = cfg["host"]["cycles_per_vector"]
    m = Gpt2Dims()
    out = [f"chip: {words_chip} words; host.cycles_per_vector {cpv}; pool {pool}; reserve 15 % (estimate)"]
    for lanes in lanes_list:
        for vp in (2, 4):
            lay = Layout(lanes, vp)
            out.append(f"\nlanes {lanes}, V planes {vp}: layer {lay.layer_words(m):,} words ({lay.matrix_words(m):,} matrices), "
                       f"KV {lay.kv_words(m, pool):,} per layer at the pool, table {lay.table_words(m):,}, wpe {lay.wpe_words(m):,}")
            for topo in ("ring", "linear"):
                n, layers, full = partition(lay, m, pool, topo, words_chip)
                out.append(f"  aggregate SRAM, {topo:6s}: N = {n} layers per chip {layers} fullest {full:.0%}")
            for g in (1, 2):
                per = layers_per_chip(lay, m, pool, wps, g)
                chips = chips_for_layers(lay, m, pool, wps, g)
                out.append(f"  install groups {g} per hemisphere: {per} layers per chip, {chips} chips for the 12 layers (table and head extra)")
            for nl in (1, 2):
                regions, bad = stage_regions(lay, m, pool, nl, wps)
                out.append(f"  stage program with {nl} layer(s) per chip: {'regions fit' if not bad else 'overlaps: ' + str(bad)}")
            n, c = host_step(lay, m, 1, cpv)
            out.append(f"  host map vectors: {n} per layer region and step, {c} cycles; 12 layers {12 * n} vectors, {12 * c} cycles")
    return "\n".join(out)


if __name__ == "__main__":
    print(report(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG))
