"""embed: `wte[token] + wpe[position]` on the chip . The packed table holds five vocabulary rows in the
five 64-lane segments of one word, cut into parts of at most one bank, each part on a slice pair (low byte plane, high
byte plane). Every read address is a patch site: the part that holds the token reads its packed row, the others a zero
word. Per byte plane the VXM ORs the three parts' bytes (OR with zero is the identity on every bit pattern), the permuter
applies the map of the token's segment (lane 64 k + j to lane j, the other lanes masked), the two planes wait in MEM for
the one-stream-per-cycle permuter (the pair is read together), `conv_f16_f32` widens the 64 lanes exactly and one f32
add of the `wpe` row (stored as f32) finishes."""
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from opentsp_tools import patch as tpatch
from opentsp_tools import stations
from opentsp_tools.stations import ScheduleError

PRM = 124                    # SXM.W.PRM
SEGMENTS = 5
SEG_LANES = 64
FIRST_MAP = 4                # map ids 4..8: the rotations by 64 k lanes (ids 1..3 are the reduce kernels')
STAGE = (16, 17)             # west slices: the two permuted byte planes wait here
STAGE_ADDR = 0x800           # word of lookup i: STAGE_ADDR + i
OR_ALUS = ((0, 1), (2, 3))   # the two chained OR ALUs of each byte plane
CONV_ALU, ADD_ALU = 4, 5


@dataclass(frozen=True)
class Part:
    """A table part: `rows` packed rows from word `addr` of the slice pair `slices` (low plane, high plane), and the
    word `zero` (no bits set) the lookups that do not belong to this part read."""
    slices: Tuple[int, int]
    addr: int
    zero: int
    rows: int


@dataclass(frozen=True)
class Tables:
    parts: Tuple[Part, ...]
    wpe_slices: Tuple[int, ...]     # the four byte planes of the f32 `wpe` rows
    wpe_addrs: Tuple[int, ...]      # word of position 0 on each
    positions: int
    map_slices: Tuple[int, int]     # low and high byte plane of the rotation maps
    map_addr: int                   # word of the first map on both
    segments: int = SEGMENTS        # vocabulary rows of a packed word: one segment map each


def segment_map(vlanes, seg):
    """The Permute map of segment `seg`: output lane j < 64 takes input lane 64 seg + j; the other lanes are masked
    (an entry of `vlanes` or more gives zero), so the entries below `vlanes` are distinct."""
    m = np.full(vlanes, vlanes, np.int64)
    m[:SEG_LANES] = SEG_LANES * seg + np.arange(SEG_LANES)
    return m


def load_maps(k, t_load, tb):
    """LoadMapWide of the segment maps (ids FIRST_MAP ..; five on `ref`, `tb.segments`) from the words `tb.map_addr` + i of the
    map slices, one per cycle from `t_load`, bound as externs. Returns the first cycle at which a Permute may use all of them."""
    p, b = k.p, k.b
    for i in range(tb.segments):
        m = segment_map(p.vlanes, i)
        planes = [(m & 255).astype(np.uint8), (m >> 8).astype(np.uint8)]
        names = []
        for j, pl in enumerate(planes):
            names.append(f"emm{i}{j}")
            k.extern_word(names[-1], tb.map_slices[j], tb.map_addr + i, bytes(pl))
        vs = [b.read_for(t_load + i, "loadmapwide", PRM, [tb.map_slices[j]], [tb.map_addr + i], "W", 24 + j,
                         addr_text=[names[j]]) for j in range(2)]
        b.unit(PRM, "loadmapwide", f"loadmapwide {FIRST_MAP + i:#x}, W:24, W:25", vs)
    return t_load + tb.segments + p.superlanes


def embed(f, tb, name, parts_addr, wpe_addrs, map_id, t0, stage_addr):
    """One lookup. The program carries the placeholders `parts_addr` (the three parts' word addresses: the token's row in
    its part, the zero word in the others), `wpe_addrs` (the four planes' words of the position) and `map_id`; the sites
    are `<name>.p<part><l|h>`, `<name>.wpe<plane>` and `<name>.map<plane>`. The first OR dispatches at `t0` (a later cycle
    when the tail finds a MEM slice busy). Returns the f32 result Value (row E, 4 planes, lanes 0..63 the hidden vector,
    the rest +0)."""
    b, p = f.b, f.p
    if len(tb.parts) != 3:
        raise ScheduleError("the embed kernel ORs three table parts")
    d_or, sk_or = p.timing("or.u8")
    staged = []
    for pl in range(2):
        a0, a1 = OR_ALUS[pl]
        ops = []
        for q in (0, 1):
            ops.append(_read_part(f, tb, name, q, pl, parts_addr[q], t0 + pl, a0))
        o0 = b.vxm(a0, "or.u8", "W", f.stream("W", 1), ops)
        t_or1 = o0.cycle + 1 - sk_or
        v2 = _read_part(f, tb, name, 2, pl, parts_addr[2], t_or1, a1)
        o1 = b.vxm(a1, "or.u8", "W", f.stream("W", 1), [o0, v2], chain=(0,))
        site = tpatch.Decl(f"{name}.map{pl}", "map", FIRST_MAP, FIRST_MAP + tb.segments - 1, map_id)
        s_out = f.stream("E", 1)
        _, vp = b.unit(PRM, "permute", f"permute W:{o1.stream}, {site.token}, E:{s_out}", [o1], dst=("E", s_out, 1),
                       site=site)
        staged.append(b.write(vp, [STAGE[pl]], [stage_addr]))
    t_w = max(staged)
    d_read = p.timing("read")[0]
    sk_conv, sk_add = p.timing("conv_f16_f32.f16")[1], p.timing("add.f32")[1]
    g_w = stations.need_gap(p, STAGE[0], "E", "write")
    g_c = stations.need_gap(p, 88 + CONV_ALU, "E", "read")
    g_a = stations.need_gap(p, tb.wpe_slices[0], "E", "write")
    g_ar = stations.need_gap(p, 88 + ADD_ALU, "E", "read")
    t_conv = t_w + 1 + (g_c - g_w) + d_read - sk_conv
    d_conv = p.timing("conv_f16_f32.f16")[0]
    while not _tail_free(f, tb, t_conv, d_conv, sk_add, (g_c, g_w, g_ar, g_a), d_read, sk_conv):
        t_conv += 1
    vs = b.read_for(t_conv, "conv_f16_f32.f16", 88 + CONV_ALU, list(STAGE), [stage_addr] * 2, "E", f.stream("E", 2))
    cv = b.vxm(CONV_ALU, "conv_f16_f32.f16", "E", f.stream("E", 4), [vs], width=4)
    t_add = cv.cycle + 1 - sk_add
    sites = [(f"{name}.wpe{j}", tb.wpe_addrs[j], tb.wpe_addrs[j] + tb.positions - 1) for j in range(4)]
    wp = b.read_for(t_add, "add.f32", 88 + ADD_ALU, list(tb.wpe_slices), list(wpe_addrs), "E", f.stream("E", 4), patch=sites)
    return b.vxm(ADD_ALU, "add.f32", "E", f.stream("E", 4), [cv, wp], chain=(0,))


def _read_part(f, tb, name, q, pl, addr, t, alu):
    """Byte plane `pl` of part q's word at `addr`, met by an `or` on VXM[alu] at cycle t; the address is a patch site over
    the part's rows and its zero word (two banks)."""
    part = tb.parts[q]
    site = [(f"{name}.p{q}{'lh'[pl]}", part.addr, part.zero, 1, 1, True)]
    return f.b.read_for(t, "or.u8", 88 + alu, [part.slices[pl]], [addr], "E", f.stream("E", 1), patch=site)


def _tail_free(f, tb, t_conv, d_conv, sk_add, gaps, d_read, sk_conv):
    """Whether the stage read, the `wpe` read and the two ALU slots of a tail starting at `t_conv` are free."""
    g_c, g_w, g_ar, g_a = gaps
    tr = f.b.t
    t_add = t_conv + d_conv + 1 - sk_add
    t_stage = t_conv + sk_conv - (g_c - g_w) - d_read
    t_wpe = t_add + sk_add - (g_ar - g_a) - d_read
    return (all(tr.free(s, t_stage, "R") for s in STAGE) and all(tr.free(s, t_wpe, "R") for s in tb.wpe_slices)
            and tr.free(88 + CONV_ALU, t_conv) and tr.free(88 + ADD_ALU, t_add))
