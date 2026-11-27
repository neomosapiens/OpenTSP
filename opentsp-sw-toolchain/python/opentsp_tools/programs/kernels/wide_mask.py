"""wide_mask: the additive attention mask of a chunk of slots from the lane's run-time mask encoding (SHIP section 5.1 and appendix C: per page the
number of visible slots), on the chip. The encoding vector of a chunk of 8 pages (128 slots) is an f32 vector whose lane 16 g holds the
count of page g; Distribute copies it to the 16 lanes of the superlane, the VXM compares the lane's position in the page with it
(`pos < count`) and selects +0 or -inf: lane s of the result is the mask of slot s of the chunk (`ref_wide.mask_from_encoding`)."""
from opentsp_tools.programs.kernels import reduce_bcast as rb
from opentsp_tools.programs.kernels.flow import Vec

STREAM_LOW = {"W": 20, "E": 24}      # the Distribute sequence's fixed streams (W 8..19, E 20..23) are below the Flow's rotation


def distribute_counts(f, enc, out_addr, t_d, dist=None):
    """Distribute (map `dist` loaded by reduce_bcast.load_distribute_map) of the f32 Vec `enc` (east quad 0) from dispatch cycle t_d on: plane u
    through west Distribute unit u, one cycle after plane u - 1, then the select that turns the planes out of the SXM (the sequence of
    `reduce_bcast.reduce_bcast`); the result is written to west quad 0 at `out_addr`. Returns the Vec."""
    b = f.b
    dist = DIST if dist is None else dist
    t_end = 0
    for u in range(4):
        vp = b.read_for(t_d + u, "distribute", 116 + u, [44 + u], [enc.addr], "W", 8 + u, addr_text=[enc.names[u]])
        _, vd = b.unit(116 + u, "distribute", f"distribute W:{8 + u}, {dist:#x}, W:{12 + u}", [vp], dst=("W", 12 + u, 1))
        t_sel = b.when(vd, "sxselect", 125)
        vz = b.read_for(t_sel, "sxselect", 125, [7], [rb.ZERO_WORD], "W", 16 + u)
        _, vo = b.unit(125, "sxselect", f"sxselect W:{12 + u}, W:{12 + u}, W:{12 + u}, W:{16 + u}, E:{20 + u}", [vd, vd, vd, vz],
                       dst=("E", 20 + u, 1))
        t_end = max(t_end, b.write(vo, [u], [out_addr]))
    f.k.last = max(f.k.last, t_end)
    return Vec(list(range(4)), out_addr, t_end + 1, "W")


DIST = 1     # the id of the Distribute map


def chunk_mask(f, enc, pos, zero, neg_inf, out_addr, t_d, alu, dist=None):
    """The mask Value (f32, row E) of the chunk whose encoding is the Vec `enc`: distribute_counts, `cmp.lt` of the position Vec `pos` (lane l:
    l mod 16, f32) with the counts on ALU `alu`, and `select` of the Vecs `zero` and `neg_inf` by that result (chained) on the next ALU."""
    cnt = distribute_counts(f, enc, out_addr, t_d, dist)
    m = f.vxm(alu, "cmp.lt.f32", [pos, cnt])
    return f.vxm(alu + 1, "select.f32", [zero, neg_inf, m], chain=(2,))
