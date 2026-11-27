"""gather: the embedding row `wte[token] + wpe[position]` read by Gather (the pipeline's S-6 choice: the row addresses come from map vectors the host
sends each step). A Gather moves one byte per tile, so one 320-byte plane of a table row costs 16 instructions (k = 0..15) on the table's
slice; the sixteen results each hold only lanes 16 g + k, so they are written to MEM and OR-ed in a chain of VXM ALUs, one word per cycle.
Every row is an f16 row of three 256-lane tiles = six byte planes, each on a slice of its own; the planes of all tiles share one map."""
from opentsp_tools import stations
from opentsp_tools.programs.kernels.flow import Vec

REF = 8              # the table slice whose timeline the repeated map reads are scheduled for
RESULT_BASE = 44     # the results of plane j are written to slices RESULT_BASE + 2 j and + 1 (alternating words: an OR reads two of them in a cycle)
PAIR_QUAD = 5        # the OR-ed byte planes (low, high) of tile n are on slices 44 + 4 (PAIR_QUAD + n) and the next
PLANES = 6


def gather_plane(b, slice_, map_vals, stream, res_addr, plane, result_base=RESULT_BASE):
    """16 Gathers (k = 0..15) of table slice `slice_` by the repeated map `map_vals`; each result (one byte per tile, zero elsewhere) is written to a
    west-to-east MEM slice RESULT_BASE + 2 plane + (k mod 2) at word res_addr + k. Returns the list of 16 (slice, addr, ready) of the results."""
    p = b.p
    d_mem = p.timing("gather")[0]
    g = stations.need_gap(p, slice_, "E", "write")
    out = []
    for k in range(16):
        t = b.when(map_vals[k], "gather", slice_)
        b._read(map_vals[k], slice_, "gather", t)
        b.raw(t, slice_, f"gather E:{stream}, E:{map_vals[k].stream}, {k:#x}, 0x0", slot="R")
        res = b._write("E", stream, 1, g, t + d_mem)
        s = result_base + 2 * plane + (k % 2)
        tw = b.write(res, [s], [res_addr + k])
        out.append((s, res_addr + k, tw + 1))
    return out


def or_words(f, words, alu0=0):
    """The OR of the 16 result words (a chain of 15 `or.u8` on VXM[alu0 + i], each taking the previous result chained and the next word from MEM).
    Returns the Value (row E)."""
    t_all = max(r for _, _, r in words)      # later results are still being written to the same slices, in the same bank
    vecs = [Vec([s], a, t_all, "E") for s, a, r in words]
    acc = f.vxm(alu0, "or.u8", [vecs[0], vecs[1]], to="E")
    for i in range(2, len(vecs)):
        acc = f.vxm(alu0 + i - 1, "or.u8", [acc, vecs[i]], to="E", chain=(0,))
    return acc


def embed_row(f, tables, map_word, t_ref, alu0=0, res_addr=0x300, result_base=RESULT_BASE, pair_quad=PAIR_QUAD, pair_addr=0x500):
    """The f16 row of a table (`tables` = the table slices of its six planes: plane 2 n + pl of tile n) at the row the map word addresses, as f32
    tiles: the sixteen Gathers of every plane, the OR chains, and `conv_f16_f32` of each tile's two planes. Returns the list of three tile Vecs.
    `map_word` = (slice, addr, text, stream) of the map in MEM."""
    b = f.b
    ms, ma, mtext, mstream = map_word
    mp = [b.read_for(t_ref + i, "gather", REF, [ms], [ma], "E", mstream, addr_text=[mtext], emit=(i == 0)) for i in range(16)]
    b.raw(mp[0].cycle - b.p.timing("read")[0] + 1, ms, "repeat 0xf, 0x0", slot=None)
    results = [gather_plane(b, tables[j], mp, mstream + 1 + j, res_addr + 16 * j, j, result_base) for j in range(PLANES)]
    tiles = []
    for n in range(PLANES // 2):
        base = 44 + 4 * (pair_quad + n)
        for pl in range(2):
            o = or_words(f, results[2 * n + pl], alu0)
            t = b.write(o, [base + pl], [pair_addr + n])
            f.k.last = max(f.k.last, t)
        v16 = Vec([base, base + 1], pair_addr + n, t + 1, "E")
        tiles.append(f.vxm(alu0, "conv_f16_f32.f16", [v16], to="E", width=4))
    return tiles
