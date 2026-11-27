"""reduce_bcast: Reduce per superlane, a permute-and-combine tree over the superlanes of a segment of four, and
Distribute . A value of w byte planes is staged through MEM between the steps because the one permuter
of a hemisphere moves one stream per cycle."""
import numpy as np

from opentsp_tools.programs import data
from opentsp_tools.programs import numerics as nx
from opentsp_tools import stations

MAP_WORD = 0x60      # word of rotation map i on west slice 2 (+ 1 for the high byte of a wide map)


def rotations(rounds):
    """The rotations of the tree rounds, in order: 2, 1 for the two rounds of a segment of four superlanes, 4, 2, 1
    for the three of a segment of eight (the order is part of the arithmetic,  item 5)."""
    return tuple(1 << i for i in reversed(range(rounds)))


def load_maps(k, t_load, rounds=2, base=0):
    """LoadMap (LoadMapWide above 256 lanes) of the rotation maps (ids base + 1, base + 2 (base + 3): rotations of 2
    and 1, or 4, 2 and 1 superlanes within a segment of 2 ** rounds) on SXM.W.PRM, from west slices 2 and 3 at cycles
    t_load, t_load + 1 (t_load + 2). The words (MAP_WORD + base + i) and the extern names (m{base + i}{plane}) follow
    `base`, so that trees of different segments can share a program (two rounds at base 0, three at base 3: ids 1..2
    and 4..6, with the Distribute map 3 between them). Returns the first cycle at which a Permute may use them."""
    p, b = k.p, k.b
    wide = p.vlanes > 256
    for i, rot in enumerate(rotations(rounds)):
        m = nx.rotation_map(p.vlanes, rot, 1 << rounds)
        planes = [(m & 255).astype(np.uint8)] + ([(m >> 8).astype(np.uint8)] if wide else [])
        n = base + i
        for j, pl in enumerate(planes):
            k.extern_word(f"m{n}{j}", 2 + j, MAP_WORD + n, bytes(pl))
        vs = [b.read_for(t_load + i, "loadmap", 124, [2 + j], [MAP_WORD + n], "W", 24 + j, addr_text=[f"m{n}{j}"])
              for j in range(len(planes))]
        mnem = "loadmapwide" if wide else "loadmap"
        b.unit(124, mnem, f"{mnem} {n + 1:#x}, " + ", ".join(f"W:{24 + j}" for j in range(len(planes))), vs)
    return t_load + rounds - 1 + p.superlanes


def tree_round(k, t, op, w, words, map_id, alu, t_ready, t_src):
    """One round of the reduce-and-broadcast tree. The round's input is on east slices 44.. (w planes, word
    words[0], written at cycle t_src). Plane by plane, on consecutive cycles from the earliest cycle at which the
    words can be read, the input is read onto row W and permuted by map `map_id` on SXM.W.PRM (the unit moves one
    stream per cycle); each permuted plane is written to west slice `plane` (word words[2]) as it arrives; then the
    input (row W) and the permuted vector (row E) are read together by VXM[alu], which combines them onto row E,
    and east slices 44.. write the result to word words[1]. `t_ready`: the first cycle at which the map is ready.
    Returns the cycle of that last Write."""
    p, b = k.p, k.b
    src, dst, scratch = words
    d_read = p.timing("read")[0]
    g_w, g_r = stations.need_gap(p, 44, "W", "write"), stations.need_gap(p, 124, "W", "read")
    t_perm = max(t_ready, t_src + 1 + (g_r - g_w) + d_read)
    t_p = 0
    for pl in range(w):
        vp = b.read_for(t_perm + pl, "permute", 124, [44 + pl], [src], "W", 8 + pl)
        _, v = b.unit(124, "permute", f"permute W:{8 + pl}, {map_id:#x}, E:{4 + pl}", [vp], dst=("E", 4 + pl, 1))
        t_p = max(t_p, b.write(v, [pl], [scratch]))
    g_e = stations.need_gap(p, 0, "E", "write")
    t_add = t_p + 1 + (stations.need_gap(p, 88, "E", "read") - g_e) + d_read - p.timing(f"{op}.{t}")[1]
    sl = list(range(w))
    vr = b.read_for(t_add, f"{op}.{t}", 88, [44 + i for i in sl], [src] * w, "W", 16)
    vq = b.read_for(t_add, f"{op}.{t}", 88, sl, [scratch] * w, "E", 20)
    vs = b.vxm(alu, f"{op}.{t}", "E", 24, [vr, vq])
    return b.write(vs, [44 + i for i in sl], [dst] * w)



TREE_ALUS = (1, 2, 3)    # the VXM ALUs that combine the rounds
ZERO_WORD = 0x70         # a word no program writes: the SRAM resets to zero
DIST_MAP = 3             # map id of the Distribute maps (every entry 0: lane 16g of a superlane to all its lanes)


def dist_map(rounds):
    """The map id of the Distribute map: the one after the rotation maps."""
    return rounds + 1


def tree_words(a0, bank1, rounds):
    """(source, destination, scratch) words of each round of the tree at a0: a round reads from one bank and writes
    to the other, so a Read and a Write never share a bank; the scratch words are in bank 1."""
    a1, a2 = a0 + bank1, a0 + 1
    words = [(a0, a1, a0 + 2 + bank1), (a1, a2, a0 + 3 + bank1)]
    if rounds == 3:
        words.append((a2, a0 + 4 + bank1, a0 + 5 + bank1))
    return words


def reduce_tree(k, t, op, vx, a0, t_ready, chain=False, rounds=2, base=0):
    """`op` (add, max, min) of element type t over the superlanes of each segment (four, or eight with `rounds` = 3):
    Reduce of the Value vx (the previous VXM result when `chain`) on VXM[0], its planes written to east slices 44..
    at word a0, then the permute-and-combine rounds (tree_round; the words of tree_words hold the intermediate and
    the final totals). Returns (the cycle of the last Write, the word of the totals on east slices 44..): lane 16g of
    every superlane holds the segment's total, the other lanes are zero."""
    b, bank1 = k.b, 1 << k.p.bank_bit
    w = data.width(t)
    vr = b.vxm(0, f"reduce.{op}.{t}", "E", 8, [vx], width=w, chain=(0,) if chain else ())
    t_w = b.write(vr, [44 + i for i in range(w)], [a0] * w)
    words = tree_words(a0, bank1, rounds)
    for rnd, ws in enumerate(words):
        t_w = tree_round(k, t, op, w, ws, base + rnd + 1, TREE_ALUS[rnd], t_ready, t_w)
    return t_w, words[-1][1]


def load_distribute_map(k, t_load, map_id=DIST_MAP):
    """LoadMap of the all-zero map (id `map_id`, DIST_MAP by default; the maps of a hemisphere are shared by its SXM units) on west
    Distribute unit 0, from one zero word read at t_load. Returns the first cycle at which a Distribute may use it."""
    b = k.b
    v = b.read_for(t_load, "loadmap", 116, [6], [ZERO_WORD], "W", 28)
    b.unit(116, "loadmap", f"loadmap {map_id:#x}, W:28", [v])
    return t_load + 1 + k.p.timing("loadmap")[0]


def reduce_bcast(k, op, vx, a0, out_addr, t_ready, t_dist_ready, chain=False, rounds=2, base=0, dist=None):
    """reduce_tree of f32 followed by Distribute: plane k of the totals (east slice 44 + k) goes through the west
    Distribute unit k, one cycle after plane k - 1 (the select that turns the planes out of the SXM takes one stream
    per cycle), then through SXM.W.SEL with every lane selecting its own value, and is written to west slice k at
    word out_addr (bank 1). Every lane of a segment holds its total. Returns the (slices, address, first cycle the
    planes can be read) of the four planes. `base` and `dist` are the rotation maps' base and the Distribute map's id
    (default: the one after the rotation maps of base 0)."""
    p, b = k.p, k.b
    t_w, a2 = reduce_tree(k, "f32", op, vx, a0, t_ready, chain, rounds, base)
    d_read = p.timing("read")[0]
    g_w, g_r = stations.need_gap(p, 44, "W", "write"), stations.need_gap(p, 116, "W", "read")
    t_d = max(t_dist_ready, t_w + 1 + (g_r - g_w) + d_read)
    t_end = 0
    for u in range(4):
        vp = b.read_for(t_d + u, "distribute", 116 + u, [44 + u], [a2], "W", 8 + u)
        _, vd = b.unit(116 + u, "distribute", f"distribute W:{8 + u}, {(dist_map(rounds) if dist is None else dist):#x}, W:{12 + u}", [vp],
                       dst=("W", 12 + u, 1))
        t_sel = b.when(vd, "sxselect", 125)
        vz = b.read_for(t_sel, "sxselect", 125, [7], [ZERO_WORD], "W", 16 + u)
        _, vo = b.unit(125, "sxselect", f"sxselect W:{12 + u}, W:{12 + u}, W:{12 + u}, W:{16 + u}, E:{20 + u}",
                       [vd, vd, vd, vz], dst=("E", 20 + u, 1))
        t_end = max(t_end, b.write(vo, [u], [out_addr]))
    k.last = max(k.last, t_end)
    return list(range(4)), out_addr, t_end + 1
