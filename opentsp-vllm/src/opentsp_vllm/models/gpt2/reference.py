"""bit-exact reference of GPT-2 small on the SHIP-style pipeline: the bit-exact numpy twin of the decode step at width 768.

It follows the operation order of the kernels that the pipeline runs (
static text, per-step input, a full-position pool and f16 cache), every step computed with `opentsp_tools.programs.numerics`:
 - the embedding `wte[token] + wpe[position]` (the Gather of the token's table row and the position row, one f32 add);
 - per layer: LayerNorm over 3 vectors of 256 lanes, Q, K and V by the tiled matvec (one exact f16 dot per 256-lane tile
   rounded to f32, tiles added in f32, f32 bias), the new K and V rows narrowed to f16 and written to a slot of the pool (the
   Scatter), attention over the WHOLE pool with the mask generated from the page counts (12 heads of 64), the output
   projection, the residual, LayerNorm, the FFN (up, tanh-GELU, down; 12 tiles of 256 for the down projection), the residual;
 - the final LayerNorm and the tied head by the tiled matvec on the table.
The slot of the new token is a parameter. Slot order matters to the arithmetic (the sums of the softmax and of the context
run over fixed chunks of slots in lane order), so bit-exact reference runs on the slot layout the pool gives, not on positions."""
from dataclasses import dataclass

import numpy as np

from opentsp_tools.programs import ref
from opentsp_tools.programs import ref_wide as rw
from opentsp_vllm.models.gpt2.config import Gpt2Config


@dataclass
class Pool:
    """The KV pool of one stage: K and V of every slot, f16, in 256-lane vectors; `counts` is the page-count encoding of
    one sequence (visible slots of each page, the sequence's tokens in slot order)."""
    cfg: Gpt2Config
    layers: int
    k: np.ndarray = None
    v: np.ndarray = None
    used: int = 0

    def __post_init__(self):
        shape = (self.layers, self.cfg.pool, self.cfg.tiles, rw.TILE)
        self.k = np.zeros(shape, np.float16)
        self.v = np.zeros(shape, np.float16)

    def counts(self, n):
        """Page counts of a sequence of n tokens in slots 0..n-1."""
        c = np.zeros(self.cfg.pages, np.int64)
        full, rest = divmod(n, rw.PAGE)
        c[:full] = rw.PAGE
        if rest:
            c[full] = rest
        return c


@dataclass
class StepResult:
    hidden: np.ndarray                    # (H) f32, the final LayerNorm output
    logits: np.ndarray = None             # (vocab) f32
    trace: dict = None


def embed(w, token, pos):
    """x0 = wte[token] + wpe[pos] in f32 (both f16 tables widened exactly)."""
    return ref.residual(w.wte[token].astype(np.float32), w.wpe[pos].astype(np.float32))


def head_logits(w, hidden, rows=8192):
    """The tied head on the MXM: the table's rows as an f16 matrix, `hidden` rounded to f16, tiles of 256 features, f32 sums."""
    out = np.empty(w.cfg.vocab, np.float32)
    for a in range(0, w.cfg.vocab, rows):
        out[a:a + rows] = rw.matvec_tiled(w.wte[a:a + rows], hidden)
    return out


def layer_step(cfg, lw, x, kpool, vpool, slot, mask, pv="mxm", parts=None):
    """One layer for one token. kpool and vpool (pool, tiles, TILE) f16 are updated in place at `slot`; `mask` (pool) f32 is the
    lane's mask (it must show the slot, written before attention). Returns x out (H) f32."""
    rec = parts.__setitem__ if parts is not None else (lambda n, a: None)
    xn = rw.layernorm_wide(x, lw["ln1g"], lw["ln1b"], cfg.ln_eps)
    q = rw.matvec_tiled(lw["wq"], xn, lw["bq"])
    k = rw.matvec_tiled(lw["wk"], xn, lw["bk"])
    v = rw.matvec_tiled(lw["wv"], xn, lw["bv"])
    kpool[slot] = rw.split(rw.to_f16(k))                 # Scatter of the new K and V rows
    vpool[slot] = rw.split(rw.to_f16(v))
    ctx = rw.attention_pool(q, kpool, vpool, mask, cfg.heads, scale=np.float32(1.0 / np.sqrt(cfg.dk)), pv=pv)
    ao = rw.matvec_tiled(lw["wo"], ctx, lw["bo"])
    h1 = ref.residual(x, ao)
    xn2 = rw.layernorm_wide(h1, lw["ln2g"], lw["ln2b"], cfg.ln_eps)
    u = rw.matvec_tiled(lw["up_w"], xn2, lw["up_b"])
    g = ref.gelu(u)
    d = rw.matvec_tiled(lw["dn_w"], g, lw["dn_b"])
    out = ref.residual(h1, d)
    for name, a in (("xn", xn), ("q", q), ("k", k), ("v", v), ("ctx", ctx), ("ao", ao), ("h1", h1), ("xn2", xn2), ("u", u),
                    ("g", g), ("d", d)):
        rec(name, a)
    return out


def decode_step(w, pool: Pool, token, pos, slot, counts, logits=True, pv="mxm", trace=None) -> StepResult:
    """One token through all layers: `counts` is the page-count encoding that already includes the new token's slot."""
    cfg = w.cfg
    mask = rw.mask_from_encoding(counts, cfg.pool)
    assert mask[slot] == 0, "the new token's slot must be visible"
    rec = (lambda n, a: trace.__setitem__(n, np.array(a, copy=True))) if trace is not None else (lambda n, a: None)
    x = embed(w, token, pos)
    rec("x0", x)
    for l, lw in enumerate(w.layers):
        x = layer_step(cfg, lw, x, pool.k[l], pool.v[l], slot, mask, pv)
        rec(f"x{l + 1}", x)
    hidden = rw.layernorm_wide(x, w.lnf_g, w.lnf_b, cfg.ln_eps)
    rec("hidden", hidden)
    return StepResult(hidden, head_logits(w, hidden) if logits else None, trace)


def generate(w, prompt_ids, n_new, pv="mxm", slots=None, pool=None, on_step=None):
    """Prefill token by token (the logits only of the last prompt token), then greedy decoding (first maximum on ties). Returns
    (token ids, the logits of each generated step). `slots` maps a position to its pool slot (default: the identity)."""
    cfg = w.cfg
    pool = pool or Pool(cfg, cfg.layers)
    slots = slots or (lambda pos: pos)
    tokens, steps = [int(t) for t in prompt_ids], []
    pos = 0
    while True:
        last = pos >= len(tokens) - 1
        r = decode_step(w, pool, tokens[pos], pos, slots(pos), page_counts(cfg, slots, pos), logits=last, pv=pv)
        if on_step:
            on_step(pos, r)
        if last:
            steps.append(r.logits)
            if len(tokens) - len(prompt_ids) >= n_new:
                break
            tokens.append(int(np.argmax(r.logits)))
        pos += 1
    return tokens, steps


def page_counts(cfg, slots, pos):
    """The mask encoding of positions 0..pos under the map `slots`: the visible slots of a page are the first `count` of it, so
    a map that puts a position behind an empty slot of its page cannot be encoded and is refused."""
    c = np.zeros(cfg.pages, np.int64)
    for p in range(pos + 1):
        c[slots(p) // rw.PAGE] += 1
    for p in range(pos + 1):
        s = slots(p)
        if s % rw.PAGE >= c[s // rw.PAGE]:
            raise ValueError(f"slot {s} of position {p} is not in the visible prefix of its page: not encodable as a page count")
    return c
