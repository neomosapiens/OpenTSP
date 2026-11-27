"""Optional tensor and float64 comparison functions; no test-runner dependency."""
import numpy as np

def layer_step_torch(cfg, lw, x0, k_slots, v_slots, mask):
    """The block's layer step in PyTorch: (x out (H), k (H), v (H)) as float32 numpy arrays. The rounding points are
    those of `reference_model.layer_step`: f16 inputs of every matmul, f16 q (head-masked ABC), f16 K cache, f16 probabilities
    (the expand trick); the V cache and everything else f32."""
    import torch                                                     # noqa: PLC0415 (optional dependency)
    t = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
    q16 = lambda x: x.half().float()
    H, A, dk, S = cfg.hidden, cfg.heads, cfg.dk, cfg.slots
    ln = lambda x, g, b: torch.nn.functional.layer_norm(x, (H,), t(g), t(b), cfg.ln_eps)
    x = t(x0)
    n = q16(ln(x, lw["ln1g"], lw["ln1b"]))
    q, k, v = (n @ t(lw[m]).T + t(lw[b]) for m, b in (("wq", "bq"), ("wk", "bk"), ("wv", "bv")))
    q, k16 = q16(q), q16(k)
    K, V = t(k_slots).clone(), t(v_slots).clone()
    K[S - 1], V[S - 1] = k16, v
    Kh, Vh = K.view(S, A, dk).permute(1, 0, 2), V.view(S, A, dk).permute(1, 0, 2)          # (A, S, dk)
    s = (Kh * q.view(A, 1, dk)).sum(-1) * 0.5 + t(mask)                                    # (A, S)
    p = torch.softmax(s, -1)
    ctx = (q16(p).unsqueeze(-1) * Vh).sum(1).reshape(H)
    x = x + q16(ctx) @ t(lw["wo"]).T + t(lw["bo"])
    n = q16(ln(x, lw["ln2g"], lw["ln2b"]))
    g = torch.nn.functional.gelu(n @ t(lw["up_w"]).T + t(lw["up_b"]), approximate="tanh")
    x = x + q16(g) @ t(lw["dn_w"]).T + t(lw["dn_b"])
    return x.numpy(), k.numpy(), v.numpy()

def layer_step_f64(cfg, lw, x0, k_slots, v_slots, mask):
    """The same layer step in float64 numpy with nothing rounded (weights as stored): the sanity reference that needs
    no PyTorch (bit-exact reference must be within 1e-2 of it)."""
    f = lambda a: np.asarray(a, dtype=np.float64)
    H, A, dk, S = cfg.hidden, cfg.heads, cfg.dk, cfg.slots
    ln = lambda x, g, b: (x - x.mean()) / np.sqrt(((x - x.mean()) ** 2).mean() + cfg.ln_eps) * f(g) + f(b)
    x = f(x0)
    n = ln(x, lw["ln1g"], lw["ln1b"])
    q, k, v = (f(lw[m]) @ n + f(lw[b]) for m, b in (("wq", "bq"), ("wk", "bk"), ("wv", "bv")))
    K, V = f(k_slots).copy(), f(v_slots).copy()
    K[S - 1], V[S - 1] = k, v
    Kh, Vh = K.reshape(S, A, dk).transpose(1, 0, 2), V.reshape(S, A, dk).transpose(1, 0, 2)
    s = (Kh * q.reshape(A, 1, dk)).sum(-1) * 0.5 + f(mask)
    e = np.exp(s - s.max(-1, keepdims=True))
    p = e / e.sum(-1, keepdims=True)
    ctx = (p[:, :, None] * Vh).sum(1).reshape(H)
    x = x + f(lw["wo"]) @ ctx + f(lw["bo"])
    n = ln(x, lw["ln2g"], lw["ln2b"])
    u = f(lw["up_w"]) @ n + f(lw["up_b"])
    g = 0.5 * u * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (u + 0.044715 * u ** 3)))
    return x + f(lw["dn_w"]) @ g + f(lw["dn_b"])
