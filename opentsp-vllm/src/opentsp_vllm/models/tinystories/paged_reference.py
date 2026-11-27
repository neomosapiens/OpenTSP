"""Host-side bit-exact oracle for the paged TinyStories step, using arithmetic from `reference_model`.

The arithmetic of one lane is `reference_model.layer_step`'s, with two differences that the design fixes:
 - slot = position: the context is installed block by block in block-table order, the current token sits in slot `pos`
   and the mask is 0 at slots 0..pos, -inf above (reference_model puts the current token in the last slot);
 - the softmax reduces over the whole context vector: a butterfly of C/16 superlanes (the kernel and `opentsp_tools.programs.ref`
   stop at four), which for C = 64 is the same tree.
Lanes never interact except through the pool: within a layer every lane writes its K and V rows before any lane
reads its context, so a prefill chunk sees its own earlier tokens.
"""
import collections
from dataclasses import dataclass, replace

import numpy as np

from opentsp_tools.programs import numerics as nx
from opentsp_tools.programs import ref
from opentsp_vllm.models.tinystories import reference as exact_ref
from opentsp_vllm.models.tinystories.config import ModelConfig

BLOCK = 16                      # vLLM's block size 
POOL_BLOCKS = 128               # blocks the worker reports : 0 is the null block, vLLM uses 1..127
NULL = 0                        # never written: the zero source for padding block references


class BlockPool:
    """The host model of the chip's KV pool : `num_blocks` blocks of 16 positions per layer, K rows f16
    and V rows f32 as the attention kernels read them, the null block 0 (zero for ever), and the sink block
    `num_blocks`, which takes redirected writes and is never read. Blocks are allocated in FIFO order with reference
    counts, as vLLM's `FreeKVCacheBlockQueue` hands them out; freed blocks keep their stale words (the chip does not
    clear them: a masked slot must not depend on them)."""

    def __init__(self, cfg: ModelConfig, num_blocks=POOL_BLOCKS):
        self.cfg = cfg
        self.num_blocks = num_blocks
        self.sink = num_blocks
        self.k = np.zeros((num_blocks + 1, cfg.layers, BLOCK, cfg.hidden), dtype=np.float16)
        self.v = np.zeros((num_blocks + 1, cfg.layers, BLOCK, cfg.hidden), dtype=np.float32)
        self.free_queue = collections.deque(range(1, num_blocks))
        self.refs = {}

    def allocate(self):
        if not self.free_queue:
            raise MemoryError("block pool exhausted")
        b = self.free_queue.popleft()
        self.refs[b] = 1
        return b

    def share(self, b):
        """A second reader of a live block (prefix caching): no data moves."""
        self._live(b)
        self.refs[b] += 1
        return b

    def free(self, b):
        self._live(b)
        self.refs[b] -= 1
        if self.refs[b] == 0:
            del self.refs[b]
            self.free_queue.append(b)

    def copy(self, src, dst):
        """`kv_cache_block_copies`: every layer's K and V words of `src` into `dst`."""
        self._writable(dst)
        self.k[dst], self.v[dst] = self.k[src], self.v[src]

    def zero(self, b):
        """`new_block_ids_to_zero`."""
        self._writable(b)
        self.k[b], self.v[b] = 0, 0

    def write_row(self, b, layer, offset, k16, v32):
        self._writable(b)
        self.k[b, layer, offset] = k16
        self.v[b, layer, offset] = v32

    def _live(self, b):
        if b not in self.refs:
            raise ValueError(f"block {b} is not allocated")

    def _writable(self, b):
        if not 0 < b <= self.sink:
            raise ValueError(f"block {b} is the null block or outside the pool")


# -- embedding on the chip  -----------------------------------------------------------------------------
PACK = 5                        # vocabulary rows per SRAM word (5 segments of 64 lanes)
PART_ROWS = 4096                # packed rows per table part: a region may not cross a bank


def packed_table(w, pack=PACK):
    """The embedding table as the chip holds it: `(parts, PART_ROWS + 1, pack * H)` u16 words (the f16 bit patterns), row
    pack * w + k in lanes 64k.. of packed row w, the last word of every part the reserved zero word; padding rows are zero."""
    cache = w.__dict__.setdefault("_packed_wte", {})
    t = cache.get(pack)
    if t is None:
        H, V = w.cfg.hidden, w.cfg.vocab
        rows = -(-V // pack)
        flat = np.zeros((rows * pack, H), dtype=np.uint16)
        flat[:V] = w.wte.view(np.uint16)
        packed = flat.reshape(rows, pack * H)
        parts = -(-rows // PART_ROWS)
        t = np.zeros((parts, PART_ROWS + 1, pack * H), dtype=np.uint16)
        for p in range(parts):
            chunk = packed[p * PART_ROWS:(p + 1) * PART_ROWS]
            t[p, :len(chunk)] = chunk
        cache[pack] = t
    return t


def embed_chip(w, token, pos, pack=PACK):
    """The chip's `wte[token] + wpe[pos]` , step by step: each part's slice pair reads the packed row (or its zero
    word), the VXM ORs the parts, the permuter's map k = token mod 5 rotates lane 64k + j to lane j, `convert.f32.f16`
    widens the 64 lanes exactly and one f32 add of `wpe[pos]` (stored as f32) finishes. OR with a zero word is the
    identity on every bit pattern and the conversion is exact, so the result equals `reference_model.embed` bit for bit."""
    H = w.cfg.hidden
    tab = packed_table(w, pack)
    row, k = divmod(int(token), pack)
    part, local = divmod(row, PART_ROWS)
    word = np.zeros(pack * H, dtype=np.uint16)
    for p in range(tab.shape[0]):
        word |= tab[p, local if p == part else PART_ROWS]
    tok16 = word[k * H:(k + 1) * H].view(np.float16)
    return ref.residual(ref.from_f16(tok16), w.wpe[pos].astype(np.float32))


# -- one layer, one lane, on the slot = position layout ------------------------------------------------------------------
def bucket_cfg(cfg, C):
    """`cfg` with C context slots: bit-exact reference is always evaluated on the bucket's shape ."""
    return replace(cfg, slots=C).check()


def reduce_bcast_full(op, x):
    """`ref.reduce_bcast` over all superlanes of the vector (one segment of C/16, rotations C/32 .. 1): the same tree as
    `ref.reduce_bcast` for four superlanes, and for eight with an empty upper half the same additions."""
    s = nx.reduce(op, "f32", x)
    n = s.shape[-1]
    seg = n // 16
    assert seg & (seg - 1) == 0, seg
    rot = seg // 2
    while rot:
        s = nx.binary(op, "f32", s, s[..., nx.rotation_map(n, rot, seg)])
        rot //= 2
    return nx.distribute_first(s)


def softmax_full(s):
    """`ref.softmax` with `reduce_bcast_full`."""
    e = nx.transcendental("exp", "f32", ref._sub(s, reduce_bcast_full("max", s)))
    tot = reduce_bcast_full("add", e)
    return ref._mul(e, nx.transcendental("rsqrt", "f32", ref._mul(tot, tot)))


def attention(cfg, q, k16, v32, mask, parts=None, unit=ref.LANES):
    """`reference_model.attention` over cfg.slots slots with the full-vector softmax. `unit`: the lanes of the unit that sums an MXM dot product."""
    sc = ref.scores(k16, q, mask, ref.head_lane_masks(cfg.heads, cfg.hidden), unit)
    pr = softmax_full(sc)
    if parts is not None:
        parts["scores"], parts["probs"] = sc, pr
    return ref.context(ref.expand(pr, cfg.dk), v32)


def project(cfg, lw, x, unit=ref.LANES):
    """LayerNorm and the Q, K, V projections: returns (xn, q, k, v), k and v as the f32 projections."""
    xn = ref.layernorm(x, lw["ln1g"], lw["ln1b"], cfg.ln_eps)
    return (xn, ref.matvec_f16(lw["wq"], xn, lw["bq"], unit), ref.matvec_f16(lw["wk"], xn, lw["bk"], unit),
            ref.matvec_f16(lw["wv"], xn, lw["bv"], unit))


def finish_layer(cfg, lw, x, q, k16, v32, mask, parts=None, unit=ref.LANES):
    """Attention over the installed context, the output projection and the FFN: `reference_model.layer_step` after its K and V
    projections. Returns the layer output."""
    ctx = attention(cfg, q, k16, v32, mask, parts, unit)
    return finish_context(cfg, lw, x, ctx, parts, unit)


def finish_context(cfg, lw, x, ctx, parts=None, unit=ref.LANES):
    """The projection, residuals and MLP after attention."""
    half = cfg.half
    rec = parts.__setitem__ if parts is not None else (lambda n, a: None)
    ao = ref.matvec_f16(lw["wo"], ctx, lw["bo"], unit)
    h1 = ref.residual(x, ao)
    xn2 = ref.layernorm(h1, lw["ln2g"], lw["ln2b"], cfg.ln_eps)
    d, u = [], []
    for i in range(2):
        sl = slice(i * half, (i + 1) * half)
        u.append(ref.matvec_f16(lw["up_w"][sl], xn2, lw["up_b"][sl], unit))
        d.append(nx.dot_f16(lw["dn_w"][:, sl], ref.to_f16(ref.gelu(u[i])), unit))
    y = ref.residual(ref.residual(d[0], d[1]), lw["dn_b"])
    out = ref.residual(h1, y)
    for name, a in (("ctx", ctx), ("ao", ao), ("h1", h1), ("xn2", xn2), ("y", y)):
        rec(name, a)
    return out


def position_mask(pos, C):
    m = np.full(C, -np.inf, dtype=np.float32)
    m[:pos + 1] = 0
    return m


def install(pool, layer, table, C):
    """The context a lane installs : the C/16 blocks of its table in order, the null block for every block
    beyond the table. Returns (K16 (C, H) f16, V32 (C, H) f32); slot s is position s."""
    n = C // BLOCK
    ids = [table[j] if j < len(table) else NULL for j in range(n)]
    return pool.k[ids, layer].reshape(C, -1), pool.v[ids, layer].reshape(C, -1)


@dataclass
class Lane:
    """A token lane. `token` None is a padding lane : zero embedding, K and V writes to the sink, a mask that
    selects position 0 of the null block."""
    token: object
    pos: int
    table: list

    @property
    def real(self):
        return self.token is not None


PAD = Lane(None, 0, [])


def check_lane(pool, lane, C):
    if not lane.real:
        return
    if not 0 <= lane.pos < C:
        raise ValueError(f"position {lane.pos} outside the context bucket {C}")
    if len(lane.table) > C // BLOCK or lane.pos // BLOCK >= len(lane.table):
        raise ValueError(f"block table of {len(lane.table)} blocks does not cover position {lane.pos} in bucket {C}")
    for b in lane.table:
        if not 0 <= b < pool.num_blocks:
            raise ValueError(f"block id {b} outside the pool")
    if lane.table[lane.pos // BLOCK] == NULL:
        raise ValueError("the block of the written position is the null block")


def run_lanes(w, pool, lanes, C, trace=None, unit=ref.LANES, pack=PACK):
    """The layer-major pass over the lanes : per layer, every lane projects and writes its K and V
    rows, then every lane installs its context and finishes the layer. Returns per lane (hidden after the final
    LayerNorm (H) f32, K rows (layers, H) f16, V rows (layers, H) f32); `trace`, a list of dicts, one per lane,
    receives x0, x1.. and hidden."""
    cfg = bucket_cfg(w.cfg, C)
    H = cfg.hidden
    for lane in lanes:
        check_lane(pool, lane, C)
    written = [(ln.table[ln.pos // BLOCK], ln.pos % BLOCK) for ln in lanes if ln.real]
    if len(set(written)) != len(written):
        raise ValueError("two lanes write the same block row (design 4.3: no two lanes write one word)")
    xs = [embed_chip(w, ln.token, ln.pos, pack) if ln.real else np.zeros(H, dtype=np.float32) for ln in lanes]
    krows = [np.zeros((cfg.layers, H), dtype=np.float16) for _ in lanes]
    vrows = [np.zeros((cfg.layers, H), dtype=np.float32) for _ in lanes]
    rec = [(lambda n, a, t=t: t.__setitem__(n, np.array(a, copy=True))) if t is not None else (lambda n, a: None)
           for t in (trace if trace is not None else [None] * len(lanes))]
    for r, x in zip(rec, xs):
        r("x0", x)
    for l, lw in enumerate(w.layers):
        qs = []
        for i, (lane, x) in enumerate(zip(lanes, xs)):
            _, q, k, v = project(cfg, lw, x, unit)
            krows[i][l], vrows[i][l] = ref.to_f16(k), v
            if lane.real:
                pool.write_row(lane.table[lane.pos // BLOCK], l, lane.pos % BLOCK, krows[i][l], v)
            else:
                pool.write_row(pool.sink, l, 0, krows[i][l], v)
            qs.append(q)
        for i, (lane, x) in enumerate(zip(lanes, xs)):
            kc, vc = install(pool, l, lane.table if lane.real else [], C)
            xs[i] = finish_layer(cfg, lw, x, qs[i], kc, vc, position_mask(lane.pos, C), None, unit)
            rec[i](f"x{l + 1}", xs[i])
    out = []
    for i, x in enumerate(xs):
        h = ref.layernorm(x, w.lnf_g, w.lnf_b, cfg.ln_eps)
        rec[i]("hidden", h)
        out.append((h, krows[i], vrows[i]))
    return out


def decode_step_paged(w, pool, token, pos, block_table, C=None, logits=True, trace=None) -> exact_ref.StepResult:
    """One token at position `pos` of a sequence whose blocks are `block_table` (slot = position), its K and V rows written
    into the pool, on a context bucket of C slots (default the config's `slots`). Same result type as
    `reference_model.decode_step`."""
    C = C or w.cfg.slots
    t = {} if trace is not None else None
    (h, k, v), = run_lanes(w, pool, [Lane(token, pos, list(block_table))], C, None if t is None else [t])
    if trace is not None:
        trace.update(t)
    return exact_ref.StepResult(h, k, v, head_logits_chip(w, h) if logits else None, trace)


# -- the LM head on the chip  -------------------------------------------------------------------------------
def head_logits_chip(w, hidden, unit=ref.LANES, pack=None):
    """The chip's logits: `wte . h` with h rounded to f16 once, computed as the MXM computes it . The packed
    table has 5 vocabulary rows per 320-lane row (lanes 64 k .. 64 k + 63 hold entry 5 rho + k, +0 beyond the vocabulary);
    entry 5 rho + k is the exact f16 dot of the whole packed row with the segment vector h_k (h in lanes 64 k .., +0
    elsewhere), rounded once to f32. The other four segments therefore add products `w * (+0)`: they are zero, and they
    decide the sign of an exact zero sum (-0 only when all 320 products are -0, so only when the neighbouring weights are
    negative too) and carry a neighbour's infinity or NaN into the entry (infinity times zero is NaN). Unlike
    `reference_model.head_logits` (f64 sum) it is what the chip computes, bit for bit. The result is in vocabulary order."""
    vocab, d = w.wte.shape
    pack = unit // d if pack is None else pack
    rows = -(-vocab // pack)
    packed = np.zeros((rows * pack, d), np.float16)
    packed[:vocab] = w.wte
    packed = packed.reshape(rows, pack * d)
    h16 = ref.to_f16(hidden)
    out = np.zeros(rows * pack, np.float32)
    for k in range(pack):
        hk = np.zeros(pack * d, np.float16)
        hk[d * k:d * (k + 1)] = h16
        out[k::pack] = nx.dot_f16(packed, hk, unit)
    return out[:vocab]


# -- the batched step and its buckets  --------------------------------------------------------------------
GRID_T = (1, 2, 4, 8)           # token lanes
GRID_L = (1, 2, 4)              # logit rows, L <= T
GRID_C = (64, 128)              # context slots


def choose_bucket(n_tokens, n_logits, max_pos):
    """The smallest bucket P(T, L, C) of the grid with T >= tokens, L >= logit rows and C >= the largest position + 1."""
    def smallest(grid, n, what):
        for g in grid:
            if g >= n:
                return g
        raise ValueError(f"no bucket holds {n} {what}")
    return (smallest(GRID_T, n_tokens, "token lanes"), smallest(GRID_L, n_logits, "logit rows"),
            smallest(GRID_C, max_pos + 1, "context slots"))


@dataclass
class BatchResult:
    """`hidden` (T, H) f32, `k_rows` (T, layers, H) f16 and `v_rows` (T, layers, H) f32 are the rows of the T lanes in
    order (padding lanes last: their values are those of the padding lane and are never used); `logits` (n_logits, vocab)
    f32 are the head rows of lanes 0..n_logits-1."""
    hidden: np.ndarray
    k_rows: np.ndarray
    v_rows: np.ndarray
    logits: np.ndarray
    bucket: tuple
    n_real: int


def step_batch(w, pool, tokens, positions, block_tables, bucket, n_logits=None, unit=ref.LANES, pack=PACK) -> BatchResult:
    """One step on the bucket P(T, L, C): the real lanes `tokens[i]` at `positions[i]` with block table `block_tables[i]`,
    padded to T with padding lanes. Logit rows are the first `n_logits` lanes (default min(L, real lanes)); the host
    puts the lanes that need a next token first . Writes the real lanes' K and V rows into the pool (the
    padding lanes' go to the sink) and returns the rows of every lane."""
    T, L, C = bucket
    n = len(tokens)
    if T not in GRID_T or L not in GRID_L or C not in GRID_C or L > T:
        raise ValueError(f"bucket {bucket} is not in the grid")
    if not len(positions) == len(block_tables) == n or not 1 <= n <= T:
        raise ValueError(f"{n} lanes (positions {len(positions)}, tables {len(block_tables)}) in a bucket of {T}")
    n_logits = min(L, n) if n_logits is None else n_logits
    if not 0 <= n_logits <= min(L, n):
        raise ValueError(f"{n_logits} logit rows in a bucket of L = {L} with {n} real lanes")
    lanes = [Lane(int(t), int(p), list(b)) for t, p, b in zip(tokens, positions, block_tables)] + [PAD] * (T - n)
    out = run_lanes(w, pool, lanes, C, None, unit, pack)
    hidden = np.stack([h for h, _, _ in out])
    logits = np.stack([head_logits_chip(w, hidden[i], unit, pack) for i in range(n_logits)]) if n_logits else \
        np.zeros((0, w.cfg.vocab), np.float32)
    return BatchResult(hidden, np.stack([k for _, k, _ in out]), np.stack([v for _, _, v in out]), logits, bucket, n)


# -- greedy generation: the reference of serving ---------------------------------------------------------------
def pick_token(logits):
    """Greedy: the first maximum on ties."""
    return int(np.argmax(logits))


def generate_paged(w, prompt_ids, n_new, C=None) -> exact_ref.Generation:
    """`reference_model.generate` on the paged step: prefill token by token through `decode_step_paged` (logits only from the last
    prompt token on, by `head_logits_chip`), then greedy decoding. The TS layout's `generate` differs from this at about
    1 to 4 % of the steps (4 to 6 % on 8 layers) by an f16 rounding flip, so  and  compare with this function. Returns prompt
    + generated ids and every step's result."""
    C = C or w.cfg.slots
    tokens = [int(t) for t in prompt_ids]
    if len(tokens) + n_new > C:
        raise ValueError(f"{len(tokens)} prompt tokens and {n_new} new tokens (the last one is fed too) need more than {C} positions")
    pool = BlockPool(w.cfg, max(2, -(-C // BLOCK) + 1))
    table = [pool.allocate() for _ in range(-(-C // BLOCK))]
    steps = []
    pos = 0
    while True:
        last = pos >= len(tokens) - 1
        r = decode_step_paged(w, pool, tokens[pos], pos, table, C=C, logits=last)
        steps.append(r)
        if last:
            if len(tokens) - len(prompt_ids) >= n_new:
                break
            tokens.append(pick_token(r.logits))
        pos += 1
    return exact_ref.Generation(tokens, steps)
