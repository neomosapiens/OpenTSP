"""The paged attention kernels as program cases : K install, the V reads of the
context and the K and V writes, each on the pool of the serve layout (`opentsp_vllm.models.tinystories.kvpool`: the 2-layer plan's addresses,
a reduced number of blocks preloaded) with a permuted block table and C = 64 or 128. Every expectation is bit-exact reference's
(`opentsp-vllm/src/opentsp_vllm/models/tinystories/paged_reference.py`): the K and V a lane installs are `paged_reference.install`, the scores and the
context its `ref` calls. The block ids are the placeholders of the patch sites; test_kv.py patches other tables."""
import numpy as np

from opentsp_tools.programs.case import Kit
from opentsp_tools.programs import data
from opentsp_vllm.models.tinystories import kvpool
from opentsp_tools.programs import ref
from opentsp_tools.programs.kernels import context as k_context, kv_install as k_kvi, scores as k_scores
from opentsp_tools.programs.kernels import kv_write as k_kvw
from opentsp_tools.programs.kernels.flow import Flow

PROFILES = ["ref"]
GENERIC = ["test/sl8", "test/sl32"]          # the pool of the serving model of the profile (the small model on 8 superlanes)
TAGS = ("sim", "ts-kernel")
NB = 10              # blocks of the pool the cases preload (ids 0..NB - 1; block 0 stays zero)
LAYER = 1            # of the 2-layer plan: the V pool of layer 1 is in bank 1, layer 0's in bank 0
HIDDEN, HEADS = 64, 16
TABLES = {64: [7, 2, 9, 4], 128: [7, 2, 9, 4, 1, 8, 3, 5]}
POS = {64: 50, 128: 100}


def padded(table, C):
    """The block table of a lane in bucket C: the null block for every block beyond the request's."""
    return list(table) + [0] * (C // 16 - len(table))


def kit(p, name, doc, seed):
    return Kit(p, f"model/ts_{name}", doc, seed, module="kv", tags=TAGS, source="numpy-twin")


def reference_pool(geo, rng):
    """A bit-exact reference BlockPool of NB blocks (ids 1..NB - 1 random, in both layers, block 0 zero)."""
    _, exact_ref = kvpool.serve()
    pool = exact_ref.BlockPool(geo.cfg, NB)
    pool.k[1:NB] = rng.standard_normal(pool.k[1:NB].shape).astype(np.float16)
    pool.v[1:NB] = rng.standard_normal(pool.v[1:NB].shape).astype(np.float32)
    return pool


def bind_region(k, geo, words, slices, name, lo, n):
    """One extern per slice over `n` words from `lo`, bound from the case's weights."""
    for j, s in enumerate(slices):
        k.b.prog.extern(f"{name}{j}", s, lo, n)
        k.weights[f"{name}{j}"] = ("raw", kvpool.region(words, s, lo, n, k.p.vlanes))


def bind_k_pool(k, geo, pool):
    words = {}
    for l in range(geo.cfg.layers):
        words.update(kvpool.k_words(geo, pool, l, range(NB)))
    lo = geo.kaddr(0, 0)
    bind_region(k, geo, words, geo.kslices(), "kpool", lo, geo.kaddr(geo.cfg.layers - 1, NB - 1) + 2 - lo)


def bind_v_pool(k, geo, pool, layer):
    words = kvpool.v_words(geo, pool, layer, range(NB))
    bind_region(k, geo, words, geo.vslices(layer), "vpool", geo.vaddr(layer, 0), NB * 16)


def kv_scores(p, C=64, seed=1, table=None, pos=None, kit_only=False):
    """Attention scores of 16 heads against the K cache installed block by block from the pool: the K of a lane's
    context is read through patched block addresses (one Read per slice and block, an AGT loop of two words), the
    context's position mask is 0 up to the position and -inf above."""
    k = kit(p, f"kv_scores{C}", f"q . K^T per head over a paged {C}-slot K cache, permuted block table", seed)
    geo = kvpool.Geometry(p)
    _, exact_ref = kvpool.serve()
    pool = reference_pool(geo, k.rng)
    table, pos = padded(table or TABLES[C], C), POS[C] if pos is None else pos
    bind_k_pool(k, geo, pool)
    q = np.zeros(p.vlanes, np.float32)
    q[:HIDDEN] = k.rng.standard_normal(HIDDEN) * 1.5
    add = np.zeros(p.vlanes, np.float32)
    add[:C] = exact_ref.position_mask(pos, C)
    f = Flow(k)
    masks_arr = ref.head_lane_masks(HEADS, HIDDEN)
    masks = []
    for h in range(HEADS):
        m = np.zeros(p.vlanes, np.float16)
        m[:HIDDEN] = masks_arr[h]
        masks.append(f.const(f"m{h}", m, "W", 0))
    install = lambda kk, t0: k_kvi.install_k(kk, table, lambda b: geo.kaddr(LAYER, b), geo.krange(LAYER), t0)
    out = k_scores.scores(f, C, f.const("q", q, "W", 1), masks, f.splat("h", 0.5, "W", 1), f.const("a", add, "W", 2), 30,
                          install=install)
    k16, _ = exact_ref.install(pool, LAYER, table, C)
    want = ref.scores(k16, q[:HIDDEN], add[:C], masks_arr, p.vlanes)
    for h, v in enumerate(out):
        w = np.zeros(p.vlanes, np.float32)
        w[:C] = want[h]
        f.store(v, h % 4, w)
    return k if kit_only else k.done("kv_scores")


def kv_context(p, C=64, seed=1, table=None, pos=None, kit_only=False):
    """The context of one layer over a paged V cache: the V rows of the lane's blocks are read through patched block
    addresses (one Read per block and slice of the layer's V quad, an AGT loop of 16 words, one per slot) into the
    Multiply of slot s at cycle 30 + s, with the probabilities P_s the softmax of the position mask gives (slots above
    the position are exactly +0), accumulated as `context` does."""
    k = kit(p, f"kv_context{C}", f"sum over {C} slots of p * V, V read by patched block addresses, f32, then f16", seed)
    geo = kvpool.Geometry(p)
    _, exact_ref = kvpool.serve()
    pool = reference_pool(geo, k.rng)
    table, pos = padded(table or TABLES[C], C), POS[C] if pos is None else pos
    bind_v_pool(k, geo, pool, LAYER)
    scores_ = (k.rng.standard_normal((HEADS, C)) * 2).astype(np.float32)
    scores_[:, pos + 1:] = -np.inf
    expanded = ref.expand(np.stack([exact_ref.softmax_full(s) for s in scores_]), HIDDEN // HEADS)
    pad = np.zeros((C, p.vlanes), np.float32)
    pad[:, :HIDDEN] = expanded
    f = Flow(k)
    ps = f.table("p", list(pad), "E", 2)
    vs = k_context.read_v_blocks(f, 30, table, lambda b: geo.vaddr(LAYER, b), geo.vslices(LAYER), geo.vrange(LAYER))
    z, zc = k_context.context(f, ps, vs, f.splat("z", 0.0, "W", 3), 30, (5, 6, 7, 8))
    _, v32 = exact_ref.install(pool, LAYER, table, C)
    values = np.zeros((C, p.vlanes), np.float32)
    values[:, :HIDDEN] = v32
    want = ref.context(pad, values)
    f.store(z, 0, want)
    f.store(zc, 1, ref.to_f16(want))
    return k if kit_only else k.done("kv_context")


WRITE_BLOCK, WRITE_OFFSET = 6, 11


def kv_write(p, seed=1, block=WRITE_BLOCK, offset=WRITE_OFFSET, kit_only=False):
    """The new token's K and V rows into the pool : the K row is narrowed to f16 and written by all eight
    slice pairs, the pair of the row (offset mod 8) at the block's word and the other seven at the sink block; the V row
    is copied to row `offset` of the block. `block` None is a padding lane: every K pair and the V row go to the sink.
    The pool holds other rows in the neighbouring blocks and in the rest of the block, so a Write at a wrong word shows."""
    where = "the sink" if block is None else f"row {offset} of block {block}"
    k = kit(p, "kv_write", f"K and V rows into {where}, the other K slice pairs to the sink", seed)
    geo = kvpool.Geometry(p)
    pool = reference_pool(geo, k.rng)
    bind_k_pool(k, geo, pool)
    bind_v_pool(k, geo, pool, LAYER)
    pair = offset % 8
    sink, v_sink = geo.kaddr(LAYER, geo.sink), geo.vaddr(LAYER, geo.sink)
    k_row, v_row = np.zeros(p.vlanes, np.float32), np.zeros(p.vlanes, np.float32)
    k_row[:HIDDEN], v_row[:HIDDEN] = k.rng.standard_normal((2, HIDDEN)) * 1.5
    f = Flow(k)
    real_k = None if block is None else geo.kaddr(LAYER, block) + offset // 8
    k_addrs = [real_k if r == pair and block is not None else sink for r in range(8)]
    v_addr = v_sink if block is None else geo.vaddr(LAYER, block) + offset
    t_k = k_kvw.write_k_paged(f, f.const("k", k_row, "E", 1), k_addrs, geo.krange(LAYER, True))
    t_v = k_kvw.write_v_paged(f, f.const("v", v_row, "E", 2), geo.vslices(LAYER), v_addr, geo.vrange(LAYER, True))
    k.last = max(t_k, t_v)
    k16 = np.zeros(p.vlanes, np.float16)
    k16[:HIDDEN] = ref.to_f16(k_row[:HIDDEN])
    k_planes = data.to_planes(k16)
    shown = [block if block is not None else 1]
    old_k = kvpool.k_words(geo, pool, LAYER, shown)
    for s in geo.kslices():
        r, plane = (s - kvpool.K_BASE) // 2, (s - kvpool.K_BASE) % 2
        for w in (geo.kaddr(LAYER, shown[0]), geo.kaddr(LAYER, shown[0]) + 1):
            k.expect(s, w, k_planes[plane] if w == k_addrs[r] else old_k[(s, w)])
        if k_addrs[r] == sink:
            k.expect(s, sink, k_planes[plane])
    v_planes = data.to_planes(v_row)
    old_v = kvpool.v_words(geo, pool, LAYER, shown)
    for plane, s in enumerate(geo.vslices(LAYER)):
        for row in range(16):
            w = geo.vaddr(LAYER, shown[0]) + row
            k.expect(s, w, v_planes[plane] if w == v_addr else old_v[(s, w)])
        if v_addr == v_sink:
            k.expect(s, v_sink, v_planes[plane])
    return k if kit_only else k.done("kv_write")


CASES = [
    ("model/ts_kv_scores64", kv_scores, PROFILES + GENERIC, {"C": 64}),
    ("model/ts_kv_scores128", kv_scores, PROFILES, {"C": 128}),
    ("model/ts_kv_context64", kv_context, PROFILES, {"C": 64}),
    ("model/ts_kv_context128", kv_context, PROFILES, {"C": 128}),
    ("model/ts_kv_write", kv_write, PROFILES, {}),
]
