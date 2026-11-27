"""The reference twins of the kernels at a hidden width above 128 (GPT-2 small: width 768, 12 heads of 64):
the numerical references of `ref.py` for activations that no longer fit one vector, in the layout the SHIP-style pipeline uses
( sections 2 and 4). Every step is computed with `numerics`.

Layout (implementation choice, reasons below). A vector of `n` features is `ceil(n / TILE)` vectors of TILE = 256 lanes (16
superlanes); the last vector is padded with +0. A feature vector is an array (..., tiles, TILE).
 - 256 lanes is a power of two of superlanes, so a reduction over all features is the per-superlane fold, the elementwise sum
   of the vectors and the SXM butterfly of one segment of 16 superlanes (4 rounds); 320 lanes are 20 superlanes and have no
   such tree.
 - A head of 64 lanes (4 superlanes) lies inside one vector and 4 heads fill one vector exactly.
 - A weight matrix d_in x d_out is d_out MXM rows of ceil(d_in / 256) f16 vectors; for d_in = 768 that is the 3 tiles of
   the design's ceil(d_in / 320), for d_in = 3072 it is 12 tiles (the design counted 10).

Numerics. `matvec_tiled`: one exact f16 dot per tile rounded once to f32 (ABC and ACC), the tile results added in ascending
tile order in f32 (the ACC accumulate: the first tile replaces, every other adds), then the f32 bias add. `layernorm_wide`:
two-pass mean and variance as in `ref.layernorm`, the total over the vectors being the elementwise f32 sum in ascending
vector order of the per-superlane folds, then the 16-superlane butterfly. Attention runs over the whole KV pool of a stage:
scores per head over CHUNK = 128 slots per vector (8 superlanes), the mask generated from the page counts
(`mask_from_encoding`), a softmax whose maximum and sum combine the chunks elementwise before the 8-superlane tree, and a
context product `context_mxm` (probabilities rounded to f16, one exact f16 dot per chunk and head, chunks accumulated in f32
in ascending order) or `context_vxm` (the VXM chains of `ref.context` on f16 V widened to f32)."""
import numpy as np

from opentsp_tools.programs import numerics as nx
from opentsp_tools.programs import ref

TILE = 256           # lanes of one activation vector (16 superlanes)
CHUNK = 128          # cache slots of one score vector (8 superlanes)
PAGE = 16            # slots of one page (vLLM block size; SHIP page counts)
LANES = ref.LANES
DK = 64              # lanes of a head

_add, _sub, _mul = ref._add, ref._sub, ref._mul
to_f16, from_f16 = ref.to_f16, ref.from_f16


def tiles_of(n, tile=TILE):
    return -(-n // tile)


def split(x, tile=TILE):
    """(..., n) -> (..., tiles, tile), the last tile padded with +0 (of x's dtype)."""
    x = np.asarray(x)
    t = tiles_of(x.shape[-1], tile)
    out = np.zeros(x.shape[:-1] + (t * tile,), x.dtype)
    out[..., :x.shape[-1]] = x
    return out.reshape(x.shape[:-1] + (t, tile))


def join(x, n):
    """(..., tiles, tile) -> (..., n)."""
    return x.reshape(x.shape[:-2] + (-1,))[..., :n]


def matvec_tiled(w, x, b=None, tile=TILE, lanes=LANES):
    """y = W x (+ b) for W (rows, n_in) f16 and x (n_in) f32: x rounded to f16 once; per tile of `tile` input features the
    exact dot rounded once to f32; the tiles added in ascending order in f32; then the f32 bias."""
    wt, xt = split(w.astype(np.float16), tile), split(to_f16(x), tile)
    acc = None
    for t in range(wt.shape[-2]):
        p = nx.dot_f16(wt[:, t], xt[t], lanes)
        acc = p if acc is None else _add(acc, p)
    return acc if b is None else _add(acc, b)


def _vector_total(op, v, segment):
    """v (tiles, lanes): every lane of every segment of `segment` superlanes gets the total of the segment over all vectors:
    the per-superlane fold of each vector, the elementwise sum (max) of the vectors in ascending order, then the
    butterfly of `segment` superlanes (`numerics.reduce_bcast`'s rounds); Distribute fills the superlane."""
    r = nx.reduce(op, "f32", v)
    s = r[0]
    for t in range(1, len(r)):
        s = nx.binary(op, "f32", s, r[t])
    rot = segment // 2
    while rot:
        s = nx.binary(op, "f32", s, s[nx.rotation_map(s.shape[-1], rot, segment)])
        rot //= 2
    return nx.distribute_first(s)


def layernorm_wide(x, gamma, beta, eps, tile=TILE):
    """LayerNorm of an f32 feature vector x (n), n a multiple of `tile` (no pad lane: a pad lane would hold -mean after the
    centring); gamma, beta (n) f32; 1/n is rounded to f32. Returns (n)."""
    n = x.shape[-1]
    assert n % tile == 0, (n, tile)
    seg = tile // 16
    xt = split(x, tile)
    inv_n = np.float32(1.0 / n)
    mean = _mul(_vector_total("add", xt, seg), inv_n)
    d = _sub(xt, mean)
    var = _mul(_vector_total("add", _mul(d, d), seg), inv_n)
    r = nx.transcendental("rsqrt", "f32", _add(var, np.float32(eps)))
    out = _add(_mul(_mul(d, r), split(gamma, tile)), split(beta, tile))
    return join(out, n)


def mask_from_encoding(counts, pool, page=PAGE):
    """The additive mask (pool) f32 of a lane from its mask encoding (SHIP section 5.1, appendix C): `counts[g]` is the
    number of visible slots of page g (a count of `page` means the whole page); slot s of page g is visible when
    s mod page < counts[g]. Visible slots add +0, the others -inf."""
    counts = np.asarray(counts, np.int64)
    assert len(counts) * page == pool, (len(counts), page, pool)
    slot = np.arange(pool)
    visible = (slot % page) < counts[slot // page]
    return np.where(visible, np.float32(0.0), np.float32(-np.inf)).astype(np.float32)


def head_lane_mask_wide(heads, hidden, tile=TILE):
    """(heads, tiles, tile) f16: 1.0 in the lanes of head h, +0 elsewhere."""
    m = np.zeros((heads, hidden), np.float16)
    dk = hidden // heads
    for h in range(heads):
        m[h, h * dk:(h + 1) * dk] = 1.0
    return split(m, tile)


def scores_wide(k16, q, mask, heads, scale, tile=TILE, lanes=LANES):
    """k16 (slots, tiles, `tile`) f16, q (n) f32, mask (slots) f32. Per head h: the dot of every cache row's vector that holds
    head h with q (rounded to f16) masked to head h, times `scale` (a power of two: exact), plus the mask. (heads, slots)."""
    n = heads * DK
    qm = np.where(head_lane_mask_wide(heads, n, tile) != 0, split(to_f16(q), tile), np.float16(0.0)).astype(np.float16)
    out = np.empty((heads, k16.shape[0]), np.float32)
    per = tile // DK
    for h in range(heads):
        out[h] = nx.dot_f16(k16[:, h // per], qm[h, h // per], lanes)
    return np.where(mask == 0, _mul(out, np.float32(scale)), mask).astype(np.float32)


def softmax_pool(s, chunk=CHUNK, seg=None):
    """s (heads, slots) with slots a multiple of `chunk`; the softmax of each head over all slots, f32. The maximum and the
    sum are combined over the chunks elementwise (ascending) and then reduced by the tree of chunk / 16 superlanes; a masked
    slot (-inf) gives +0; 1/s is rsqrt(s * s) as in `ref.softmax`."""
    heads, slots = s.shape
    assert slots % chunk == 0
    sc = s.reshape(heads, slots // chunk, chunk)
    seg = seg or chunk // 16
    m = sc[:, 0]
    for c in range(1, sc.shape[1]):
        m = nx.binary("max", "f32", m, sc[:, c])
    m = nx.distribute_first(nx.reduce_bcast("max", "f32", m, seg))
    e = nx.transcendental("exp", "f32", _sub(sc, m[:, None, :]))
    tot = e[:, 0]
    for c in range(1, e.shape[1]):
        tot = _add(tot, e[:, c])
    tot = nx.distribute_first(nx.reduce_bcast("add", "f32", tot, seg))
    p = _mul(e, nx.transcendental("rsqrt", "f32", _mul(tot, tot))[:, None, :])
    return p.reshape(heads, slots)


def context_mxm(p, v16, chunk=CHUNK):
    """The PV product on the MXM (design S-7): p (heads, slots) f32, v16 (slots, tiles, TILE) f16. Per head and per chunk of
    slots one exact f16 dot (the probabilities rounded to f16) of the head's 64 feature columns of V against the chunk,
    rounded once to f32; the chunks added in ascending order in f32 (ACC accumulate). Returns (tiles * TILE) f32 with head h in
    lanes 64 h ..; the lanes of no head are +0."""
    heads, slots = p.shape
    per = TILE // DK
    p16 = to_f16(p).reshape(heads, slots // chunk, chunk)
    out = np.zeros((v16.shape[1], TILE), np.float32)
    for h in range(heads):
        vt, l0 = v16[:, h // per, (h % per) * DK:(h % per + 1) * DK].T, (h % per) * DK     # (DK, slots)
        acc = None
        for c in range(slots // chunk):
            part = nx.dot_f16(vt[:, c * chunk:(c + 1) * chunk], p16[h, c], LANES)
            acc = part if acc is None else _add(acc, part)
        out[h // per, l0:l0 + DK] = acc
    return out.reshape(-1)


def context_vxm(p, v16, chains=2, tile=TILE):
    """The PV product on the VXM chains with f16 V widened to f32: probabilities rounded to f16 and expanded to the lanes of their head, one f32
    product per slot, `chains` interleaved sequential f32 sums over the slots (s mod chains, each starting at +0; a VXM add takes four cycles, so
    one chain takes a slot every four cycles) combined by the ascending pairwise tree (c0 + c1) (+ (c2 + c3)). (tiles * TILE) f32."""
    heads, slots = p.shape
    pe = np.repeat(from_f16(to_f16(p)).T, DK, axis=1)                        # (slots, heads * DK)
    pe = split(pe, tile).reshape(slots, -1)
    v = from_f16(v16).reshape(slots, -1)
    pe_t = np.zeros_like(v)
    pe_t[:, :pe.shape[1]] = pe
    prod = _mul(pe_t, np.where(pe_t != 0, v, np.float32(0)))
    acc = []
    for j in range(chains):
        a = np.zeros(prod.shape[1:], np.float32)
        for row in prod[j::chains]:
            a = _add(a, row)
        acc.append(a)
    while len(acc) > 1:
        acc = [_add(acc[i], acc[i + 1]) for i in range(0, len(acc), 2)]
    return acc[0]


def attention_pool(q, k16, v16, mask, heads, scale=None, pv="mxm", tile=TILE):
    """q (n) f32; k16, v16 (slots, tiles, TILE) f16 (the stage's whole pool); mask (slots) f32. The context (n) f32."""
    scale = 1.0 / np.sqrt(DK) if scale is None else scale
    s = scores_wide(k16, q, mask, heads, scale, tile)
    p = softmax_pool(s)
    ctx = context_mxm(p, v16) if pv == "mxm" else context_vxm(p, v16, tile=tile)
    return ctx[:heads * DK]
