"""The TinyStories kernels as program cases: each kernel of `opentsp_tools.programs.kernels` at the
showcase's sizes (64-lane vectors, 16 heads of 4 lanes, 64 cache slots, 128-lane FFN halves) on `ref`, with synthetic
data. Every expectation is the kernel's twin in `opentsp_tools.programs.ref`, whose operations `test_numerics.py` checks against the
Python oracle."""
import os
import sys

import numpy as np

from opentsp_tools.programs.case import Kit
from opentsp_tools.programs import data
from opentsp_tools.programs import ref
from opentsp_tools.programs.kernels import context as k_context, expand as k_expand, gelu as k_gelu, head_mask as k_head_mask, install as k_install, layernorm as k_layernorm
from opentsp_tools.programs.kernels import matvec_f16 as k_matvec, reduce_bcast as k_rb, residual as k_residual, scores as k_scores
from opentsp_tools.programs.kernels import softmax as k_softmax, write_row as k_write_row
from opentsp_tools.programs.kernels.flow import Flow

PROFILES = ["ref"]
GENERIC = ["test/sl8", "test/sl32"]          # other widths (8 and 32 superlanes): the kernels derive their sizes from the profile
FOLDED = ["test/ref_small"]                    # 8 MEM slices per hemisphere: the caller's quads fold onto the quads it has, an install reads both hemispheres
TAGS = ("sim", "ts-kernel")
HIDDEN, HEADS, SLOTS, HALF = 64, 16, 64, 128


def kit(p, name, doc, seed):
    return Kit(p, f"model/ts_{name}", doc, seed, module="model", tags=TAGS, source="numpy-twin")


def lanes(k, n, scale=1.0):
    """A vlanes-lane f32 vector: n random lanes, zero beyond (a vector shorter than the unit)."""
    v = np.zeros(k.p.vlanes, np.float32)
    v[:n] = k.rng.standard_normal(n) * scale
    return v


def residual(p, seed=1):
    """a + b in f32 on 64 lanes: random values and the special lanes (infinities of both signs, -0, a NaN)."""
    k = kit(p, "residual", "x + y, f32", seed)
    a, b = lanes(k, HIDDEN, 3), lanes(k, HIDDEN, 3)
    a[:4], b[:4] = (np.inf, np.inf, -0.0, np.nan), (-np.inf, 1.0, -0.0, 1.0)
    f = Flow(k)
    v = k_residual.residual(f, f.const("a", a, "W", 1), f.const("b", b, "W", 2))
    f.store(v, 0, ref.residual(a, b))
    return k.done("residual")


def gelu(p, seed=1):
    """GELU, tanh form, of a 128-lane FFN half (f32): nine chained VXM ALUs."""
    k = kit(p, "gelu", "GELU of one FFN half, f32", seed)
    x = lanes(k, HALF, 2)
    x[:2] = (-0.0, 0.0)
    v, _ = k_gelu.gelu_chain(k, "f32", k.lead("mul.f32"), x, 0, 1, 0x10)
    k.store(v, 0, 0x10, ref.gelu(x))
    return k.done("gelu")


def matvec_f16(p, seed=1):
    """y = W x + b: a 64 x 64 f16 matrix (LW 8 cycles, IW), an f32 activation narrowed to f16 once and an f32 bias."""
    k = kit(p, "matvec_f16", "W x + b, W 64 x 64 f16", seed)
    w = (k.rng.standard_normal((HIDDEN, HIDDEN)) * 0.25).astype(np.float16)
    x, bias = lanes(k, HIDDEN, 2), lanes(k, HIDDEN, 0.5)
    k_install.bind_weights(k, w)
    f = Flow(k)
    y = k_matvec.matvec_f16(f, HIDDEN, f.const("x", x, "W", 1), f.const("b", bias, "W", 2))
    want = np.zeros(p.vlanes, np.float32)
    want[:HIDDEN] = ref.matvec_f16(w, x[:HIDDEN], bias[:HIDDEN], p.vlanes)
    f.store(y, 0, want)
    return k.done("matvec_f16")


def reduce_bcast(p, seed=1):
    """reduce_bcast of f32, add then max, on every segment of four superlanes of a full vector: Reduce, two
    permute-and-combine rounds, Distribute and the select that turns the planes out of the SXM; every lane of a
    segment holds its total."""
    k = kit(p, "reduce_bcast", "reduce, butterfly over four superlanes, distribute; add and max", seed)
    t_ready, t_dist = k_rb.load_maps(k, 60), k_rb.load_distribute_map(k, 80)
    t0, bank1 = 30, 1 << p.bank_bit
    for j, op in enumerate(("add", "max")):
        x = lanes(k, p.vlanes, 4)
        a0 = 0x10 + 8 * j
        vx = k.feed(k.place(f"x{j}", x, 0, a0), a0, "E", 0, t0, f"reduce.{op}.f32")
        slices, addr, ready = k_rb.reduce_bcast(k, op, vx, a0, bank1 + 0x40 + j, t_ready, t_dist)
        for s_, plane in zip(slices, data.to_planes(ref.reduce_bcast(op, x))):
            k.expect(s_, addr, plane)
        t0 = ready + 20
    return k.done("reduce_bcast")


def layernorm(p, seed=1):
    """LayerNorm of a 64-lane f32 vector (epsilon 1e-5, 1/64 an exact f32): two reduce_bcasts, mean, centring,
    variance, RSqrt, scale and shift."""
    k = kit(p, "layernorm", "LayerNorm of 64 lanes, f32", seed)
    maps = (k_rb.load_maps(k, 60), k_rb.load_distribute_map(k, 80))
    x = lanes(k, HIDDEN, 3)
    gamma, beta = lanes(k, HIDDEN, 0.1), lanes(k, HIDDEN, 0.1)
    gamma[:HIDDEN] += 1
    f = Flow(k)
    y = k_layernorm.layernorm(f, f.const("x", x, "W", 1), f.const("g", gamma, "W", 2), f.const("b", beta, "W", 2),
                              f.splat("n", 1 / HIDDEN, "W", 3), f.splat("e", 1e-5, "W", 3), maps, 0x10,
                              (1 << p.bank_bit) + 0x40)
    want = np.zeros(p.vlanes, np.float32)
    want[:HIDDEN] = ref.layernorm(x[:HIDDEN], gamma[:HIDDEN], beta[:HIDDEN], 1e-5)
    f.store(y, 0, want)
    return k.done("layernorm")


def head_mask(p, seed=1):
    """The f16 query kept in the lanes of each of the 16 heads: the f32 query is narrowed to f16 on the VXM and
    kept in MEM, then mask.f16 with the head's lane mask (a constant f16 vector) on one ALU per head."""
    k = kit(p, "head_mask", "query masked to each of 16 heads of 4 lanes, f16", seed)
    q = lanes(k, HIDDEN, 2)
    f = Flow(k)
    q16 = f.store(f.vxm(0, "conv_f32_f16.f32", [f.const("q", q, "W", 1)], to="W", width=2), 3)
    masks = ref.head_lane_masks(HEADS, HIDDEN)
    want = ref.head_mask(ref.to_f16(q[:HIDDEN]), masks)
    for h in range(HEADS):
        m = np.zeros((p.vlanes,), np.float16)
        m[:HIDDEN] = masks[h]
        v = k_head_mask.head_mask(f, q16, f.const(f"m{h}", m, "EW"[h % 2], h // 2 % 3), 1 + h % 4)
        out = np.zeros(p.vlanes, np.float16)
        out[:HIDDEN] = want[h]
        f.store(v, h // 4, out)
    return k.done("head_mask")


def scores(p, seed=1):
    """Attention scores of 16 heads against a 64-slot K cache: slots 0..9 and the current slot 63 hold keys, the
    others are empty (zero rows) and masked to -inf. K is installed in east plane 2 (8 LW), then one masked-query
    ABC and one ACC per head, scale 0.5, mask add."""
    k = kit(p, "scores", "q . K^T per head, scale 0.5 and mask, 64 slots, 16 heads", seed)
    keys = np.zeros((SLOTS, HIDDEN), np.float16)
    live = list(range(10)) + [SLOTS - 1]
    keys[live] = (k.rng.standard_normal((len(live), HIDDEN)) * 0.5).astype(np.float16)
    q = lanes(k, HIDDEN, 1.5)
    add = np.full(p.vlanes, -np.inf, np.float32)
    add[live] = 0
    add[SLOTS:] = 0
    k_install.bind_weights(k, keys, plane=k_scores.PLANE, prefix="k")
    f = Flow(k)
    masks_arr = ref.head_lane_masks(HEADS, HIDDEN)
    masks = []
    for h in range(HEADS):
        m = np.zeros(p.vlanes, np.float16)
        m[:HIDDEN] = masks_arr[h]
        masks.append(f.const(f"m{h}", m, "W", 0))
    out = k_scores.scores(f, SLOTS, f.const("q", q, "W", 1), masks, f.splat("h", 0.5, "W", 1),
                          f.const("a", add, "W", 2), 30)
    want = ref.scores(keys, q[:HIDDEN], add[:SLOTS], masks_arr, p.vlanes)
    for h, v in enumerate(out):
        w = np.zeros(p.vlanes, np.float32)
        w[:SLOTS] = want[h]
        f.store(v, h % 4, w)
    return k.done("scores")


def softmax(p, seed=1):
    """Softmax of four packed vectors: each 320-lane f32 vector holds one head's slots in each of its five segments
    of four superlanes (the first vector with masked slots, -inf). 16 heads take ceil(16 / 5) = 4 vectors."""
    k = kit(p, "softmax", "softmax over 64-lane segments, 4 packed vectors, f32", seed)
    maps = (k_rb.load_maps(k, 60), k_rb.load_distribute_map(k, 80))
    f = Flow(k)
    bank1 = 1 << p.bank_bit
    for i in range(4):
        s = lanes(k, p.vlanes, 3)
        if i == 0:
            s[10:50] = -np.inf
            s[SLOTS + 5:2 * SLOTS - 5] = -np.inf
        v = k_softmax.softmax(f, f.const(f"s{i}", s, "W", 1), maps, 0x10 + 16 * i, bank1 + 0x40 + 2 * i,
                              e_quad=1 + i % 2, at=k.last)
        f.store(v, 3 + i // 2, ref.softmax(s))
    return k.done("softmax")


def softmax128(p, seed=1):
    """Softmax over 128-slot segments, the C = 128 bucket's three-round tree (rotations 4, 2, 1 of eight superlanes): four
    packed vectors of two heads each (lanes 0..127 and 128..255, the last 64 lanes a partial segment of four superlanes).
    Slots above 63 are live in every vector but the last; the expectation of each head is
    `paged_reference.softmax_full`."""
    showcase = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "showcase", "tinystories")
    if showcase not in sys.path:
        sys.path.insert(0, showcase)
    from opentsp_vllm.models.tinystories import paged_reference as reference_paged
    k = kit(p, "softmax128", "softmax over 128-slot segments, three-round tree, 4 packed vectors, f32", seed)
    maps = (k_rb.load_maps(k, 60, 3), k_rb.load_distribute_map(k, 80, k_rb.dist_map(3)), 3)
    f = Flow(k)
    bank1 = 1 << p.bank_bit
    for i in range(4):
        s = lanes(k, p.vlanes, 3)
        if i == 0:
            s[10:50] = -np.inf                      # a hole in the lower half
            s[128 + 90:2 * 128 - 5] = -np.inf       # the second head's upper quarter masked
        elif i == 1:
            s[100:128] = -np.inf                    # the first head's last slots
            s[128 + 65:2 * 128] = -np.inf
        elif i == 3:
            s[64:128] = -np.inf                     # an empty upper half: the tree of four superlanes
            s[128 + 64:2 * 128] = -np.inf
        v = k_softmax.softmax(f, f.const(f"s{i}", s, "W", 1), maps, 0x10 + 16 * i, bank1 + 0x40 + 2 * i,
                              e_quad=1 + i % 2, at=k.last)
        want = ref.softmax(s, 8)
        for seg in range(2):
            part = slice(128 * seg, 128 * seg + 128)
            want[part] = reference_paged.softmax_full(s[part])
        f.store(v, 3 + i // 2, want)
    return k.done("softmax128")


def trees(p, seed=1, kit_only=False):
    """Both tree shapes in one program ( fix 1, I-2): a LayerNorm (segment of four superlanes, rotation maps 1, 2) and a
    softmax over 128-slot segments (eight, rotation maps 4, 5, 6 loaded at base 3), sharing the all-zero Distribute map 3:
    six of the 16 map ids. Expectations: `ref.layernorm` and `paged_reference.softmax_full`."""
    showcase = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "showcase", "tinystories")
    if showcase not in sys.path:
        sys.path.insert(0, showcase)
    from opentsp_vllm.models.tinystories import paged_reference as reference_paged
    k = kit(p, "trees", "LayerNorm (segment 4) and softmax over 128 slots (segment 8) in one program, f32", seed)
    t_ln = k_rb.load_maps(k, 60)
    t_sm = k_rb.load_maps(k, min(70, t_ln - 4), 3, 3)       # before the first Permute of the LayerNorm tree, which is at t_ln or later
    t_dist = k_rb.load_distribute_map(k, 80, k_rb.DIST_MAP)
    f = Flow(k)
    bank1 = 1 << p.bank_bit
    x = lanes(k, HIDDEN, 3)
    gamma, beta = lanes(k, HIDDEN, 0.1), lanes(k, HIDDEN, 0.1)
    gamma[:HIDDEN] += 1
    y = k_layernorm.layernorm(f, f.const("x", x, "W", 1), f.const("g", gamma, "W", 2), f.const("b", beta, "W", 2),
                              f.splat("n", 1 / HIDDEN, "W", 3), f.splat("e", 1e-5, "W", 3), (t_ln, t_dist), 0x10,
                              bank1 + 0x40)
    want = np.zeros(p.vlanes, np.float32)
    want[:HIDDEN] = ref.layernorm(x[:HIDDEN], gamma[:HIDDEN], beta[:HIDDEN], 1e-5)
    f.store(y, 0, want)
    s = lanes(k, p.vlanes, 3)
    s[10:50] = -np.inf
    s[128 + 90:2 * 128 - 5] = -np.inf
    v = k_softmax.softmax(f, f.const("s", s, "W", 1), (t_sm, t_dist, 3, 3, k_rb.DIST_MAP), 0x30, bank1 + 0x50,
                          e_quad=1, at=k.last)
    want = ref.softmax(s, 8)
    for seg in range(2):
        part = slice(128 * seg, 128 * seg + 128)
        want[part] = reference_paged.softmax_full(s[part])
    f.store(v, 3, want)
    return k if kit_only else k.done("trees")


def write_row(p, seed=1):
    """The current token's K and V rows into slot 63 of the caches: the f32 K row is narrowed to f16 and written to
    the K cache in its install layout (east plane 2, row 63 is word 7 of slices 58 and 59), the f32 V row is copied to
    word 63 of the four slices of east quad 4. The caches are preloaded with other rows in every slot, so a Write
    that touched a neighbouring slot would show."""
    k = kit(p, "write_row", "K row (f16) and V row (f32) into slot 63 of the caches", seed)
    keys = (k.rng.standard_normal((SLOTS, HIDDEN)) * 0.5).astype(np.float16)
    values = [lanes(k, HIDDEN, 1) for _ in range(SLOTS)]
    k_row, v_row = lanes(k, HIDDEN, 1.5), lanes(k, HIDDEN, 1.5)
    k_install.bind_weights(k, keys, plane=k_write_row.K_PLANE, prefix="k")
    f = Flow(k)
    v_cache = f.table("c", values, "E", 4)[0]
    v_base, v_slices = v_cache.addr, v_cache.slices         # the V cache: slot s is word v_base + s of slices 60..63 (the east quad 4 of `ref`)
    k.last = max(k_write_row.write_k(f, f.const("k", k_row, "W", 1), SLOTS - 1),
                 k_write_row.write_v(f, f.const("v", v_row, "W", 2), SLOTS - 1, 4, v_base))
    padded = np.zeros(p.vlanes, np.float16)
    padded[:HIDDEN] = ref.to_f16(k_row[:HIDDEN])
    k_slices = k_install.group(p, k_write_row.K_PLANE)[14:16]
    for s_, plane, old in zip(k_slices, data.to_planes(padded), k_install.weight_planes(keys, p.vlanes)[14:16]):
        k.expect(s_, k_install.ADDR_W + 7, plane)                       # the new row 63
        k.expect(s_, k_install.ADDR_W + 6, old[6])                      # row 55 is untouched
    for pl, plane in enumerate(data.to_planes(v_row)):
        k.expect(v_slices[pl], v_base + SLOTS - 1, plane)
        k.expect(v_slices[pl], v_base + SLOTS - 2, data.to_planes(values[SLOTS - 2])[pl])
    return k.done("write_row")


def expand(p, seed=1):
    """Per-head probabilities (16 vectors of 64 slots, f32, softmax outputs with slots 10..62 masked) to the 64
    per-lane vectors P_s (lane l = p[l div 4][s]): W' installed from the VXM (8 LW), then one ABC with the f16
    one-hot vector e_s and one ACC per slot."""
    k = kit(p, "expand", "per-head probabilities to per-lane vectors by one install and 64 one-hot ABCs", seed)
    scores_ = (k.rng.standard_normal((HEADS, SLOTS)) * 2).astype(np.float32)
    scores_[:, 10:SLOTS - 1] = -np.inf
    probs = ref.softmax(scores_)
    f = Flow(k)
    heads = []
    for h in range(HEADS):
        v = np.zeros(p.vlanes, np.float32)
        v[:SLOTS] = probs[h]
        heads.append(f.const(f"p{h}", v, "W", h % 4))
    one_hots = f.table("e", k_expand.one_hot(SLOTS, p.vlanes), "W", 3)
    out = k_expand.expand(f, heads, one_hots, HIDDEN // HEADS, 40, 2)
    want = ref.expand(probs, HIDDEN // HEADS)
    for s, v in enumerate(out):
        w = np.zeros(p.vlanes, np.float32)
        w[:HIDDEN] = want[s]
        for s_, plane in zip(v.slices, data.to_planes(w)):
            k.expect(s_, v.addr, plane)
    return k.done("expand")


def context(p, seed=1):
    """The context of one layer: sum over 64 slots of P_s * V_s in four interleaved chains, combined as
    (c0 + c1) + (c2 + c3), then narrowed to f16. P_s is the expansion of softmax probabilities (slots 10..62 masked,
    so their products with V are zero or -0), V random."""
    k = kit(p, "context", "sum over 64 slots of p * V in four chains, f32, then f16", seed)
    scores_ = (k.rng.standard_normal((HEADS, SLOTS)) * 2).astype(np.float32)
    scores_[:, 10:SLOTS - 1] = -np.inf
    expanded = ref.expand(ref.softmax(scores_), HIDDEN // HEADS)
    pad = np.zeros((SLOTS, p.vlanes), np.float32)
    pad[:, :HIDDEN] = expanded
    values = np.zeros((SLOTS, p.vlanes), np.float32)
    values[:, :HIDDEN] = k.rng.standard_normal((SLOTS, HIDDEN)) * 2
    f = Flow(k)
    ps, vs = f.table("p", list(pad), "W", 2), f.table("v", list(values), "E", 4)
    z, zc = k_context.context(f, ps, vs, f.splat("z", 0.0, "W", 3), 30, (5, 6, 7, 8))
    want = ref.context(pad, values)
    f.store(z, 0, want)
    f.store(zc, 1, ref.to_f16(want))
    return k.done("context")


def block(p, pos=0, seed=1):
    """One transformer block (layer 0, one token, a preloaded K/V cache of `pos` positions), paged on `done`
    (opentsp-vllm/src/opentsp_vllm/models/tinystories/emit_block.py)."""
    here = os.path.dirname(os.path.abspath(__file__))
    showcase = os.path.join(here, "..", "..", "showcase", "tinystories")
    if showcase not in sys.path:
        sys.path.insert(0, showcase)
    from opentsp_vllm.models.tinystories import emit_block
    return emit_block.build(p, pos, seed)


def step(p, pos=0, seed=1, head_chunks=4):
    """One decode step (eight layers, the final LayerNorm, the output head over a window of the vocabulary), paged on
    `done` (opentsp-vllm/src/opentsp_vllm/models/tinystories/emit_step.py)."""
    here = os.path.dirname(os.path.abspath(__file__))
    showcase = os.path.join(here, "..", "..", "showcase", "tinystories")
    if showcase not in sys.path:
        sys.path.insert(0, showcase)
    from opentsp_vllm.models.tinystories import emit_step
    return emit_step.build(p, pos, seed, head_chunks=head_chunks)


# Built on request only (`build.py --out DIR NAME`, `just showcase-block STEP=1`): a step's case is about 11 MB (the
# weights of eight layers), too large to check in; test_step.py pins the digests of its text and expectation instead.
ON_DEMAND = [
    ("model/ts_step_p00", step, PROFILES, {"pos": 0}),
    ("model/ts_step_p05", step, PROFILES, {"pos": 5}),
    ("model/ts_step_p62", step, PROFILES, {"pos": 62}),
    ("model/ts_step_p63", step, PROFILES, {"pos": 63}),
]

CASES = [
    ("model/ts_residual", residual, PROFILES, {}),
    ("model/ts_gelu", gelu, PROFILES, {}),
    ("model/ts_matvec_f16", matvec_f16, PROFILES + GENERIC + FOLDED, {}),
    ("model/ts_reduce_bcast", reduce_bcast, PROFILES, {}),
    ("model/ts_layernorm", layernorm, PROFILES + GENERIC + FOLDED, {}),
    ("model/ts_head_mask", head_mask, PROFILES, {}),
    ("model/ts_scores", scores, PROFILES + GENERIC + FOLDED, {}),
    ("model/ts_softmax", softmax, PROFILES, {}),
    ("model/ts_softmax128", softmax128, PROFILES + GENERIC + FOLDED, {}),
    ("model/ts_trees", trees, PROFILES + GENERIC + FOLDED, {}),
    ("model/ts_write_row", write_row, PROFILES + GENERIC + FOLDED, {}),
    ("model/ts_expand", expand, PROFILES, {}),
    ("model/ts_context", context, PROFILES, {}),
    ("model/ts_block_p00", block, PROFILES, {"pos": 0}),
    ("model/ts_block_p15", block, PROFILES, {"pos": 15}),
    ("model/ts_block_p40", block, PROFILES, {"pos": 40}),
    ("model/ts_block_p62", block, PROFILES, {"pos": 62}),
]
