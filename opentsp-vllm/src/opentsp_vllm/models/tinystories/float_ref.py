"""A float64 forward of the same network in the natural (position) layout: the sanity reference of the bit-exact reference model (bit-exact reference must be
within 1e-2 relative). Weights are used as stored (f16 values), nothing is rounded. Not a golden of any tier: the
quantisation-matched and original PyTorch references are rounded PyTorch reference and FP32 reference (torch_ref.py, optional)."""
from dataclasses import dataclass

import numpy as np


@dataclass
class RefResult:
    hidden: np.ndarray       # (T, H) float64, after the final LayerNorm
    logits: np.ndarray       # (T, vocab)


def _ln(x, g, b, eps):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def forward(w, tokens):
    cfg = w.cfg
    T, H, A, dk = len(tokens), cfg.hidden, cfg.heads, cfg.dk
    f = lambda a: np.asarray(a, dtype=np.float64)
    x = f(w.wte[list(tokens)]) + f(w.wpe[:T])
    for lw in w.layers:
        n = _ln(x, f(lw["ln1g"]), f(lw["ln1b"]), cfg.ln_eps)
        q, k, v = (n @ f(lw[m]).T + f(lw[b]) for m, b in (("wq", "bq"), ("wk", "bk"), ("wv", "bv")))
        q, k, v = (a.reshape(T, A, dk).transpose(1, 0, 2) for a in (q, k, v))
        s = q @ k.transpose(0, 2, 1) / np.sqrt(dk)
        s = np.where(np.tril(np.ones((T, T), dtype=bool)), s, -np.inf)
        p = np.exp(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        ctx = (p @ v).transpose(1, 0, 2).reshape(T, H)
        x = x + ctx @ f(lw["wo"]).T + f(lw["bo"])
        n = _ln(x, f(lw["ln2g"]), f(lw["ln2b"]), cfg.ln_eps)
        x = x + _gelu(n @ f(lw["up_w"]).T + f(lw["up_b"])) @ f(lw["dn_w"]).T + f(lw["dn_b"])
    h = _ln(x, f(w.lnf_g), f(w.lnf_b), cfg.ln_eps)
    return RefResult(h, h @ f(w.wte).T)
