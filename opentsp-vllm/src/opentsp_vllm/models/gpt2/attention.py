"""The full-pool attention of one layer region as one program ( section 4): per token step on one chip, the
mask of every chunk of 128 slots from the page-count encoding, the scores of every head pair (a tile of 128 lanes holds two heads of 64) against
the installed K chunks, the softmax of every head over the whole pool, the probabilities expanded to the lanes of their head and the context on the
VXM over the f16 V rows. Every expectation is the context of the twin `ref_wide.attention_pool` (tile 128, PV on the VXM). Text beyond the
instruction queue is paged (`opentsp_tools.pager`)."""
import os

import numpy as np

from opentsp_tools.programs.case import Kit
from opentsp_tools.programs import data
from opentsp_tools import pager
from opentsp_tools.programs import ref_wide as rw
from opentsp_tools.programs.kernels import expand as k_ex, install as k_install, reduce_bcast as k_rb, wide_context as k_wc, wide_mask as k_wm, wide_scores as k_wsc
from opentsp_tools.programs.kernels import wide_softmax as k_wsm
from opentsp_tools.programs.kernels.flow import Flow, Vec
from opentsp_vllm.models.gpt2.wide import pad_lanes

PROFILES = ["ref"]
TAGS = ("sim", "ts-kernel")
TILE = 128               # lanes of an attention vector: two heads of 64
CHUNK = rw.CHUNK
K_BASE = 0x800           # first word of the K pool on the K plane's slices (16 words per tile and chunk)
V_BASE = 0x100           # word of slot 0 of the V pool
V_FIRST = 76             # first east slice of the V pool: east quads 8 to 10
SCALE = 0.125


class PagedKit(Kit):
    """A Kit whose queues over the IQ become fetch-driven on `done` (see `pager.Case`)."""

    def done(self, fn, floor=None):
        view = pager.CycleView.of(self.b.prog)
        view2, blocks, externs = pager.page(view, self.p, self.b.t, floor=floor)
        self.b.t.check()
        self.b.prog = view2.program(self.p)
        for name, words in pager.tensor_words(blocks).items():
            self.weights[name] = ("raw", b"".join(bytes(w) for w in words))
        text, case = super().done(fn)
        case["expect"]["lints"] = []
        case["golden"].update(paged_queues=sorted({blk.queue for blk in blocks}), blocks=len(blocks))
        case["budget"]["cycles"] = self.last + 48
        return text, case


def bind_k_pool(k, p, k16, tiles, chunks, k_base=K_BASE):
    """The K pool as one extern per slice of the K plane: tile t chunk c installs from word `k_base + 16 (t * chunks + c)`."""
    base_k = k_install.plane_queues(k_wsc.PLANE)[0]
    per_slice = [[] for _ in range(16)]
    for t in range(tiles):
        for c in range(chunks):
            for j, ws in enumerate(k_install.weight_planes(k16[c * CHUNK:(c + 1) * CHUNK, t], p.vlanes)):
                per_slice[j] += ws
    for j in range(16):
        k.b.prog.extern(f"k{j}", base_k + j, k_base, len(per_slice[j]))
        k.weights[f"k{j}"] = ("raw", b"".join(per_slice[j]))


def v_pool(k, p, v16, tiles, base=V_BASE):
    """The V pool: tile t's rows (slots, f16) on the east slice pair (76 + 2 t, 77 + 2 t), slot s at word `base + s` of both: every tile has its row at
    the same word, so that one map vector scatters a new row into all the tiles. Returns, per tile, the Vec of each slot."""
    out, slots = [], len(v16)
    for t in range(tiles):
        planes = [data.to_planes(pad_lanes(r, p.vlanes).astype(np.float16)) for r in v16[:, t]]
        sl = [V_FIRST + 2 * t, V_FIRST + 2 * t + 1]
        for j, s_ in enumerate(sl):
            k.b.prog.extern(f"v{t}p{j}", s_, base, slots)
            k.weights[f"v{t}p{j}"] = ("raw", b"".join(pl[j] for pl in planes))
        out.append([Vec(sl, base + i, 0, "E", [f"v{t}p{j}+{i:#x}" for j in range(2)]) for i in range(slots)])
    return out


def enc_vectors(p, counts, page=16):
    """The encoding vector of each chunk (f32, one per chunk): the live count of every page of the chunk in lane 16 g of superlane g, `page` beyond."""
    pages, encs = CHUNK // page, []
    for c in range(len(counts)):
        e = np.zeros(p.vlanes, np.float32)
        e[16 * np.arange(pages)] = counts[c]
        e[16 * np.arange(pages, p.superlanes)] = page          # the superlanes beyond the chunk see every slot: their lanes stay finite (+0 scores)
        encs.append(e)
    return encs


def enc_weights(p, counts, page=16):
    """The host's encoding words ({extern name: one word per chunk}) of the page counts `counts` (chunks, pages): the planes of `enc_vectors`."""
    planes = [data.to_planes(e) for e in enc_vectors(p, counts, page)]
    return {f"encp{j}": b"".join(pl[j] for pl in planes) for j in range(len(planes[0]))}


def attend(p, k, f, qv, counts, vs, cursor, want=None, page=16, k_base=K_BASE, twin=None, maps=None):
    """The attention of `len(qv)` head pairs over the pool of `len(counts)` chunks: `qv` = the f32 query Vec of each tile, `counts` = the page counts
    (chunks, pages) of the encoding the host sends, `vs` = `v_tables`, `cursor` = the first cycle the phase may use, `maps` = (rotation maps ready, Distribute map ready, 3) when the program loaded them. `want` = the expected context
    of each tile (f32, TILE lanes) or None; `twin` = (scores, probabilities) of the twin, expected too when given. Returns (the Vec of each tile's
    context on east quad 0, the cycle after the last)."""
    tiles, chunks = len(qv), len(counts)
    heads = 2 * tiles
    f.low = {"E": max(k_wsc.RESULT_STREAMS, k_wm.STREAM_LOW["E"]), "W": k_wm.STREAM_LOW["W"]}
    dist_id = k_rb.dist_map(3)
    if maps is None:                                      # the program's first phase loads the rotation and Distribute maps
        maps = (k_rb.load_maps(k, 60, 3), k_rb.load_distribute_map(k, 80, dist_id), 3)
    t_dist = max(maps[1], cursor)
    cursor = t_dist + 20
    # the masks of the chunks
    encs = enc_vectors(p, counts, page)
    enc = f.table("enc", encs, "E", 0)
    pos = f.const("pos", (np.arange(p.vlanes) % 16).astype(np.float32), "W", 4)
    zero, ninf = f.splat("zero", 0.0, "W", 2), f.splat("ninf", -np.inf, "W", 3)
    add_masks = []
    for c in range(chunks):
        m = k_wm.chunk_mask(f, enc[c], pos, zero, ninf, (1 << p.bank_bit) + 0x40 + c, t_dist + 10 * c, 2 * (c % 8), dist=dist_id)
        add_masks.append(f.store(m, 10))
    cursor = max(cursor, max(m.ready for m in add_masks) + 20)
    # the scores of every tile and chunk, kept on the east quad 1 + chunk (bank 0)
    scale = f.splat("scale", SCALE, "W", 3)
    lane = rw.head_lane_mask_wide(2, TILE, TILE)                      # (2, 1, TILE): head 0 of the pair in lanes 0..63, head 1 in 64..127
    masks = [f.const(f"hm{h}", pad_lanes(lane[h, 0], p.vlanes), "E", 1 + h) for h in range(2)]
    score = {}
    for t in range(tiles):
        for c in range(chunks):
            out = k_wsc.tile_scores(f, CHUNK, qv[t], masks, scale, add_masks[c], cursor, k_base + 16 * (t * chunks + c),
                                    lambda j, i, o=16 * (t * chunks + c): f"k{j}+{o + i:#x}")
            for h, v in enumerate(out):
                w = None
                if twin:
                    w = np.zeros(p.vlanes, np.float32)
                    w[:CHUNK] = twin[0][2 * t + h, c * CHUNK:(c + 1) * CHUNK]
                score[2 * t + h, c] = f.store(v, 1 + c, bank=0, expect=w)
            cursor = max(v.cycle for v in out) + 40
    cursor = max(v.ready for v in score.values()) + 10
    # the softmax of every head over the pool
    prob = {}
    for h in range(heads):
        out = k_wsm.softmax_pool(f, [score[h, c] for c in range(chunks)], maps, 0x10 + 16 * (h % 4), (1 << p.bank_bit) + 0x30 + 2 * (h % 4),
                                 [1 + c for c in range(chunks)], at=cursor)
        for c, v in enumerate(out):
            prob[h, c] = v
            if twin:
                w = np.zeros(p.vlanes, np.float32)
                w[:CHUNK] = twin[1][h, c * CHUNK:(c + 1) * CHUNK]
                for sl, plane in zip(v.slices, data.to_planes(w)):
                    k.expect(sl, v.addr, plane)
        cursor = max(v.ready for v in out) + 60
        k.last = max(k.last, cursor)
    # per tile: the probabilities expanded to lanes, then the context over the slots
    # Alternating quads leave two consecutive R-path cycles for local text
    # fetches while ABC consumes one vector every two cycles.
    one_hot_values = k_ex.one_hot(CHUNK, p.vlanes)
    one_hot_tables = [f.table(f"e{i}", one_hot_values[i::2], "W", 2 + i) for i in range(2)]
    one_hots = [one_hot_tables[s % 2][s // 2] for s in range(CHUNK)]
    zero_acc = f.splat("zacc", 0.0, "W", 5)
    f.low = {"E": 0, "W": 0}
    probability_quads = (5, 6, 7)
    probability_bases = {q_: f.next_addr.get(("E", q_, 1), f.base) for q_ in probability_quads}
    zs = []
    for t in range(tiles):
        for q_, base in probability_bases.items():
            f.next_addr[("E", q_, 1)] = base
        ps = []
        for c in range(chunks):
            offset = len(ps) % len(probability_quads)
            quads = probability_quads[offset:] + probability_quads[:offset]
            ps += k_ex.expand(f, [prob[2 * t, c], prob[2 * t + 1, c]], one_hots, rw.DK, cursor, quads, hemi="E")
            cursor = ps[-1].ready + 10
        if twin:
            pe = np.repeat(rw.from_f16(rw.to_f16(twin[1][2 * t:2 * t + 2])).T, rw.DK, axis=1)
            for s_, v in enumerate(ps):
                for sl, plane in zip(v.slices, data.to_planes(pad_lanes(pe[s_], p.vlanes))):
                    k.expect(sl, v.addr, plane)
        z, zc = k_wc.context(f, ps, vs[t], zero_acc, cursor, (3, 4))
        zs.append(f.store(z, 0, None if want is None else pad_lanes(want[t], p.vlanes)))
        cursor = max(z.cycle, k.last) + 60
    return zs, cursor


def attention(p, seed=13, tiles=1, chunks=2, page=16, name=None):
    """`tiles` head pairs (2 * tiles heads) over a pool of `chunks` chunks of 128 slots."""
    heads, slots, n, pages = 2 * tiles, CHUNK * chunks, TILE * tiles, CHUNK // page
    debug = bool(os.environ.get("ATTN_DEBUG"))                    # also expect the scores and the probabilities (lanes beyond the chunk: -inf and +0)
    k = PagedKit(p, f"ship/ts_{name or f'attention{tiles}x{chunks}'}", f"full-pool attention: {heads} heads, {slots} slots in {chunks} chunks; 128-lane tiles",
                 seed, module="ship_attn", tags=TAGS, source="numpy-twin")
    rng = k.rng
    k16 = (rng.standard_normal((slots, tiles, TILE)) * 0.7).astype(np.float16)
    v16 = (rng.standard_normal((slots, tiles, TILE)) * 1.2).astype(np.float16)
    q = (rng.standard_normal(n) * 1.5).astype(np.float32)
    counts = rng.integers(0, page + 1, size=(chunks, pages))
    counts[0, 0], counts[0, 1] = page, 1
    mask = np.concatenate([rw.mask_from_encoding(counts[c], CHUNK, page) for c in range(chunks)])
    want = rw.attention_pool(q, k16, v16, mask, heads, SCALE, pv="vxm", tile=TILE)
    s_want = rw.scores_wide(k16, q, mask, heads, SCALE, TILE)
    twin = (s_want, rw.softmax_pool(s_want)) if debug else None
    bind_k_pool(k, p, k16, tiles, chunks)
    f = Flow(k)
    qt = rw.split(q, TILE)
    qv = [f.const(f"q{t}", pad_lanes(qt[t], p.vlanes), "W", 1) for t in range(tiles)]
    vs = v_pool(k, p, v16, tiles)
    attend(p, k, f, qv, counts, vs, 0, want=rw.split(want, TILE), page=page, twin=twin)
    return k.done(f"attention{tiles}x{chunks}", floor=6144)


ON_DEMAND = [
    ("ship/ts_attention1x8", attention, PROFILES, {"tiles": 1, "chunks": 8}),
    ("ship/ts_attention2x8", attention, PROFILES, {"tiles": 2, "chunks": 8}),
    ("ship/ts_attention6x8", attention, PROFILES, {"tiles": 6, "chunks": 8}),
]

CASES = [
    ("ship/ts_attention1x2", attention, PROFILES, {"tiles": 1, "chunks": 2}),
]
