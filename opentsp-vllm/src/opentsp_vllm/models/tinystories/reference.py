"""bit-exact reference: the bit-exact numpy twin of the TinyStories decode step on opentsp (tier 1 of the design's golden tiers).

It mirrors the operation order of the on-chip program of  (kernels of 4.4):
 - embedding `wte[id] + wpe[pos]` and the output head on the host; layers, final LayerNorm on the chip;
 - matrix products are f16 weights times f16 inputs (the activation rounded to f16 once), the exact sum rounded
   once to f32 (MXM ABC + ACC), then an f32 bias add (VXM); everything else is f32 VXM work;
 - LayerNorm and softmax reduce with `reduce_bcast`: per-superlane ascending fold, then the SXM butterfly;
 - attention runs over the host-managed cache with a fixed current slot (the last slot): K rows are f16 in the
   MXM install layout, V rows f32; scores = 16 head-masked ABCs, scaled by 0.5, plus the host mask; softmax in
   f32 with `rsqrt(s*s)` as the reciprocal; the context by the `expand` trick (probabilities rounded to f16, expanded
   exactly) and four interleaved f32 multiply-accumulate chains combined as ((c0 + c1) + (c2 + c3));
 - the FFN is split in two halves (one per hemisphere's MXM plane); the down projection's two partial sums are added
   in f32, then the bias.
Slot order matters to the arithmetic (sum orders), so bit-exact reference runs on the slot layout, not on positions.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from opentsp_tools.programs import numerics as nx
from opentsp_tools.programs import ref
from opentsp_vllm.models.tinystories.config import ModelConfig


def attention(cfg, q, k16, v32, mask, parts=None):
    """q (H) f32; k16 (slots, H) f16; v32 (slots, H) f32; mask (slots) f32. Returns the context (H) f32.
    The 16 heads are independent (the program runs them one after the other); they are batched here. `parts`
    receives the scores and the probabilities (heads, slots)."""
    sc = ref.scores(k16, q, mask, ref.head_lane_masks(cfg.heads, cfg.hidden))      # (heads, slots)
    pr = ref.softmax(sc)
    if parts is not None:
        parts["scores"], parts["probs"] = sc, pr
    return ref.context(ref.expand(pr, cfg.dk), v32)


@dataclass
class StepResult:
    hidden: np.ndarray           # (H) f32, the final LayerNorm output (what the chip dumps)
    k_cur: np.ndarray            # (L, H) f16, the current K rows (slot slots-1 after the step)
    v_cur: np.ndarray            # (L, H) f32
    logits: Optional[np.ndarray] = None   # (vocab) f32, host
    trace: Optional[dict] = None


class HostKV:
    """The host's K/V cache : positions 0..n-1 as the rows the chip wrote, preloaded every step into slots
    0..pos-1, the current token in the fixed last slot (zero until the step writes it)."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.k = np.zeros((cfg.layers, 0, cfg.hidden), dtype=np.float16)
        self.v = np.zeros((cfg.layers, 0, cfg.hidden), dtype=np.float32)

    @property
    def n(self):
        return self.k.shape[1]

    def slots(self, pos):
        c = self.cfg
        if pos != self.n:
            raise ValueError(f"position {pos} but {self.n} positions cached")
        if pos > c.slots - 1:
            raise ValueError(f"context limit: position {pos} needs more than {c.slots} slots")
        k = np.zeros((c.layers, c.slots, c.hidden), dtype=np.float16)
        v = np.zeros((c.layers, c.slots, c.hidden), dtype=np.float32)
        k[:, :pos], v[:, :pos] = self.k, self.v
        mask = np.full(c.slots, -np.inf, dtype=np.float32)
        mask[:pos] = 0
        mask[c.slots - 1] = 0
        return k, v, mask

    def append(self, k_cur, v_cur):
        self.k = np.concatenate([self.k, k_cur[:, None]], axis=1)
        self.v = np.concatenate([self.v, v_cur[:, None]], axis=1)


def embed(w, token, pos):
    """Host: x0 = wte[token] + wpe[pos] in f32."""
    return ref.residual(w.wte[token].astype(np.float32), w.wpe[pos].astype(np.float32))


def head_logits(w, hidden):
    """Host output head (tied embedding): logits = wte . h. f64 accumulation in ascending column order (a fixed
    order, so the result does not depend on a BLAS), rounded to f32."""
    acc = np.zeros(w.cfg.vocab, dtype=np.float64)
    wt = getattr(w, "_wte64", None)
    if wt is None:
        wt = w._wte64 = np.ascontiguousarray(w.wte.astype(np.float64).T)      # (H, vocab)
    for k in range(w.cfg.hidden):
        acc += wt[k] * float(hidden[k])
    return acc.astype(np.float32)


def layer_step(cfg, lw, x, k_slots, v_slots, mask, parts=None):
    """One transformer layer for one token on the slot layout: x (H) f32 in, the layer's K cache (slots, H) f16 and V
    cache (slots, H) f32 holding the cached positions in slots 0..pos-1 (the current slot is overwritten), the
    additive mask. Returns (x out (H) f32, the current K row (H) f16, the current V row (H) f32), the words the
    chip writes. `parts`, a dict, receives the intermediate vectors (the block program dumps them)."""
    S, half = cfg.slots, cfg.half
    rec = parts.__setitem__ if parts is not None else (lambda n, a: None)
    xn = ref.layernorm(x, lw["ln1g"], lw["ln1b"], cfg.ln_eps)
    q = ref.matvec_f16(lw["wq"], xn, lw["bq"])
    k = ref.matvec_f16(lw["wk"], xn, lw["bk"])
    v = ref.matvec_f16(lw["wv"], xn, lw["bv"])
    K16 = k_slots.copy()
    V32 = v_slots.copy()
    K16[S - 1] = ref.to_f16(k)                 # written on chip to the fixed current slot
    V32[S - 1] = v
    ctx = attention(cfg, q, K16, V32, mask, parts)
    ao = ref.matvec_f16(lw["wo"], ctx, lw["bo"])
    h1 = ref.residual(x, ao)
    xn2 = ref.layernorm(h1, lw["ln2g"], lw["ln2b"], cfg.ln_eps)
    d, u = [], []
    for i in range(2):                            # FFN halves: west plane rows 0..half-1, east plane the rest
        sl = slice(i * half, (i + 1) * half)
        u.append(ref.matvec_f16(lw["up_w"][sl], xn2, lw["up_b"][sl]))
        g = ref.gelu(u[i])
        d.append(nx.dot_f16(lw["dn_w"][:, sl], ref.to_f16(g), ref.LANES))
    y = ref.residual(ref.residual(d[0], d[1]), lw["dn_b"])
    out = ref.residual(h1, y)
    for name, a in (("xn", xn), ("q", q), ("k", k), ("v", v), ("ctx", ctx), ("ao", ao), ("h1", h1), ("xn2", xn2),
                    ("u0", u[0]), ("u1", u[1]), ("d0", d[0]), ("d1", d[1]), ("y", y)):
        rec(name, a)
    return out, K16[S - 1], V32[S - 1]


def decode_step_slots(w, token, pos, k_slots, v_slots, mask, logits=True, trace=None) -> StepResult:
    cfg = w.cfg
    H = cfg.hidden
    rec = (lambda n, a: trace.__setitem__(n, np.array(a, copy=True))) if trace is not None else (lambda n, a: None)
    x = embed(w, token, pos)
    rec("x0", x)
    k_cur = np.zeros((cfg.layers, H), dtype=np.float16)
    v_cur = np.zeros((cfg.layers, H), dtype=np.float32)
    for l, lw in enumerate(w.layers):
        x, k_cur[l], v_cur[l] = layer_step(cfg, lw, x, k_slots[l], v_slots[l], mask)
        rec(f"x{l + 1}", x)
    hidden = ref.layernorm(x, w.lnf_g, w.lnf_b, cfg.ln_eps)
    rec("hidden", hidden)
    return StepResult(hidden, k_cur, v_cur, head_logits(w, hidden) if logits else None, trace)


def decode_step(w, kv: HostKV, token, pos, logits=True, trace=None) -> StepResult:
    """One decode step as the host drives it: preload the slots, run, store the current K/V rows at `pos`."""
    k, v, mask = kv.slots(pos)
    r = decode_step_slots(w, token, pos, k, v, mask, logits=logits, trace=trace)
    kv.append(r.k_cur, r.v_cur)
    return r


@dataclass
class Generation:
    tokens: list
    steps: list = field(default_factory=list)


def generate(w, prompt_ids, n_new) -> Generation:
    """Prefill token by token through the decode step (logits discarded until the last prompt token), then greedy
    decoding (first maximum on ties). Returns prompt + generated ids and every step's result."""
    kv = HostKV(w.cfg)
    tokens = [int(t) for t in prompt_ids]
    steps = []
    pos = 0
    while True:
        t = tokens[pos]
        last = pos >= len(tokens) - 1
        r = decode_step(w, kv, t, pos, logits=last)
        steps.append(r)
        if last:
            if len(tokens) - len(prompt_ids) >= n_new:
                break
            tokens.append(int(np.argmax(r.logits)))
        pos += 1
    return Generation(tokens, steps)
