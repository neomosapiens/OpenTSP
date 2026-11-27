"""kv_install: the K cache of a lane's context installed into an MXM plane block by block ( step 2 and
4.3). Block j of the context is two LW cycles (16 rows). Each of the 16 K slices reads the block's two words with one
patched Read and an AGT loop of two references (`step 1`), so a block costs one patch site per slice and a new block
a new reference."""
from opentsp_tools.programs.kernels import install as inst

LOOP = [("step", 1)]         # two references (the AGT's default iterations), one word apart, on one stream


def site(prefix, j, slice_):
    return f"{prefix}.b{j}.s{slice_}"


def install_k(k, table, addr_of, krange, t0=30, plane=2, prefix="kb"):
    """Install the K rows of the blocks `table` (the placeholders: block ids in block-table order, the null block 0 for
    a block beyond the request's) into `plane`, LW from cycle t0. `addr_of(block)` is the word of the block's rows
    0..7 in each of the 16 slices of the plane's hemisphere and `krange` = (lo, hi, stride, span) the range its patch sites take
    (the zero block's address to the last block's, all in one bank). The LW of block j are cycles t0 + 2 j and
    t0 + 2 j + 1. Returns the cycle of the first ABC the installed matrix serves."""
    b, p = k.b, k.p
    _, row, lw = inst.plane_queues(plane)
    slices = inst.group(p, plane)
    refs = []
    for j, blk in enumerate(table):
        refs.append([b.read_loop(b.read_cycle_for(t0 + 2 * j, "lw", lw, slices[s], row), slices[s], addr_of(blk), row, s,
                                 LOOP, patch=[(site(prefix, j, slices[s]), *krange)]) for s in range(16)])
    for i in range(2 * len(table)):
        b.unit(lw, "lw", f"lw.f16 {row}:0", [refs[i // 2][s][i % 2] for s in range(16)])
    t_iw = t0 + 2 * len(table) - 1 + p.timing("lw")[0]
    b.unit(lw, "iw", "iw.f16", cycle=t_iw)
    return t_iw + p.timing("iw")[0]
