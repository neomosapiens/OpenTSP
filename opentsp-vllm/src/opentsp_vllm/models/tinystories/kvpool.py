"""The paged KV pool as the serve layout (opentsp-vllm/src/opentsp_vllm/models/tinystories/layout.py) places it: the words of
a bit-exact reference `BlockPool` in the slices and addresses the kernels read and write, and the host rules of a step's write set. The
layout module is imported on demand: it belongs to the showcase, which imports this library."""

import numpy as np

from opentsp_tools.programs import data
from opentsp_tools.stations import ScheduleError

K_BASE = 44          # first slice of the K pool: slice pair 2 (o mod 8) of row o, the low byte plane first


def serve():
    """(layout module, paged_reference module): the serve layout and the bit-exact reference pool."""
    from opentsp_vllm.models.tinystories import layout
    from opentsp_vllm.models.tinystories import paged_reference as reference_paged
    return layout, reference_paged


class Geometry:
    """Addresses of the pools of a serve plan (`layers` = 8 or 2, or the model `cfg`) on the profile `p`."""

    def __init__(self, p, layers=2, plan=None, cfg=None):
        L, _ = serve()
        self.L, self.p, self.layers = L, p, layers
        self.cfg = cfg or L.serve_model(L.load_profile(p.name), layers)
        self.plan = plan or L.plan_serve(self.cfg, L.load_profile(p.name))
        self.blocks = L.SERVE_BLOCKS
        self.sink = L.SINK_BLOCK

    def kaddr(self, layer, block):
        """Word, in each of the 16 K slices, of rows 0..7 of the block (rows 8..15 are the next word)."""
        return self.L.kblock_addr(self.plan, layer, block)

    def vaddr(self, layer, block):
        """Word, in each slice of the layer's V quad, of row 0 of the block."""
        return self.L.vblock_addr(self.plan, layer, block)

    def vslices(self, layer):
        q = 20 + 4 * (layer // 2)
        return list(range(q, q + 4))

    def krange(self, layer, write=False):
        """The patch range of a K block address, (lo, hi, stride, span): from the zero block to the last block vLLM can
        hand out, one value per block (stride 2 n words); of a write, from the first block vLLM allocates (the null block
        is never written) to the sink, a block's word or the next (rows 8..15). The grid refuses another layer's words."""
        n = self.cfg.layers
        if write:
            return self.kaddr(layer, 1), self.kaddr(layer, self.sink) + 1, 2 * n, 2
        return self.kaddr(layer, 0), self.kaddr(layer, self.blocks - 1), 2 * n, 1

    def vrange(self, layer, write=False):
        """As krange; a V row's word is the block's address plus the row, so a write has no grid."""
        if write:
            return self.vaddr(layer, 1), self.vaddr(layer, self.sink) + 15, 1, 1
        return self.vaddr(layer, 0), self.vaddr(layer, self.blocks - 1), 16, 1

    def kslices(self):
        return list(range(K_BASE, K_BASE + 16))


def k_words(geo, pool, layer, blocks):
    """{(slice, word): bytes} of the K rows of `blocks` (bit-exact reference `BlockPool.k`, f16) in the install layout."""
    out = {}
    vl = geo.p.vlanes
    for b in blocks:
        for o in range(16):
            row = np.zeros(vl, np.float16)
            row[:pool.k.shape[-1]] = pool.k[b, layer, o]
            for plane, pl in enumerate(data.to_planes(row)):
                out[(K_BASE + 2 * (o % 8) + plane, geo.kaddr(layer, b) + o // 8)] = pl
    return out


def v_words(geo, pool, layer, blocks):
    """{(slice, word): bytes} of the V rows of `blocks` (f32), one byte plane per slice of the layer's quad."""
    out = {}
    vl = geo.p.vlanes
    for b in blocks:
        for o in range(16):
            row = np.zeros(vl, np.float32)
            row[:pool.v.shape[-1]] = pool.v[b, layer, o]
            for plane, pl in enumerate(data.to_planes(row)):
                out[(geo.vslices(layer)[plane], geo.vaddr(layer, b) + o)] = pl
    return out


def region(words, slice_, lo, n, vlanes):
    """The bytes of `n` words of one slice from `lo`: the words in `words` (a dict as k_words), zero elsewhere."""
    zero = bytes(vlanes)
    return b"".join(words.get((slice_, lo + i), zero) for i in range(n))


def check_writes(lanes, geo, layer):
    """The host rule of  for the lanes' real writes (a lane is (block, offset), None for a padding lane that
    writes the sink): two lanes never write one word. Returns the (slice, word) set of the real K and V writes."""
    seen = {}
    for i, lane in enumerate(lanes):
        if lane is None:
            continue
        block, o = lane
        if not 0 < block < geo.blocks:
            raise ScheduleError(f"lane {i}: block {block} is the null block or the sink or outside the pool")
        words = [(K_BASE + 2 * (o % 8) + pl, geo.kaddr(layer, block) + o // 8) for pl in range(2)]
        words += [(s, geo.vaddr(layer, block) + o) for s in geo.vslices(layer)]
        for w in words:
            if w in seen:
                raise ScheduleError(f"lanes {seen[w]} and {i} write the word {w[1]:#x} of slice {w[0]}")
            seen[w] = i
    return set(seen)
