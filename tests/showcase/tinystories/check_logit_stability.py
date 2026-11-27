#!/usr/bin/env python3
"""Why rounded PyTorch reference is per step: one f16 rounding flip, found stage by stage. Layer 0 of the real checkpoint at
position 1 (the cache bit-exact reference wrote at position 0) is run by `reference_model.layer_step` and by PyTorch (`check_block.
layer_step_torch`'s operations, with the stages exposed). Every stage agrees to 1e-7, but the f32 context vector of
PyTorch (another accumulation order than the chip's four interleaved chains) has an element on the other side of an
f16 rounding boundary: the next matmul gets an input one f16 step away, the difference grows to 1e-6 at the residual
and 3e-5 at the layer output, and the PyTorch twin's own K/V cache then differs, so a free-running reference drifts
(1e-3 of max |h|) while a per-step reference on the chip's cache stays at 2e-7 or at one such flip.

    python3 check_logit_stability.py [--checkpoint NAME]
"""
import argparse
import sys

import numpy as np


from opentsp_vllm.models.tinystories import reference as exact_ref
from opentsp_vllm.models.tinystories import comparison as comparison_ref
from opentsp_vllm.models.tinystories.torch_reference import layer_step_torch
from opentsp_vllm.models.tinystories.config import TINYSTORIES as CFG


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=comparison_ref.DEFAULT_CHECKPOINT)
    a = ap.parse_args(argv)
    w, why = comparison_ref.load_weights(a.checkpoint)
    why = why if w is None else comparison_ref.torch_missing()
    if w is None or why:
        print(f"SKIP check_logit_stability: {why}")
        return 0
    import torch                                                       # noqa: PLC0415
    cfg, lw = CFG, w.layers[0]
    kv = exact_ref.HostKV(cfg)
    exact_ref.decode_step(w, kv, 7454, 0)
    k_slots, v_slots, mask = kv.slots(1)
    x0 = exact_ref.embed(w, 2402, 1)
    parts = {}
    exact_ref.layer_step(cfg, lw, x0, k_slots[0], v_slots[0], mask, parts)
    t = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
    q16 = lambda x: x.half().float()
    H, A, dk, S = cfg.hidden, cfg.heads, cfg.dk, cfg.slots

    def show(name, mine, theirs, rounded=False):
        mine, theirs = np.asarray(mine), np.asarray(theirs)
        flips = int((mine.astype(np.float16) != theirs.astype(np.float16)).sum()) if rounded else 0
        print(f"{name:>4}: max |PyTorch - bit-exact reference| {np.abs(mine - theirs).max():.2e}"
              f"{f', f16 roundings that differ: {flips}' if rounded else ''}")

    n = q16(torch.nn.functional.layer_norm(t(x0), (H,), t(lw["ln1g"]), t(lw["ln1b"]), cfg.ln_eps))
    q, k, v = (n @ t(lw[m]).T + t(lw[b]) for m, b in (("wq", "bq"), ("wk", "bk"), ("wv", "bv")))
    show("q", q.numpy(), parts["q"], True)
    show("k", k.numpy(), parts["k"], True)
    q, k16 = q16(q), q16(k)
    K, V = t(k_slots[0]).clone(), t(v_slots[0]).clone()
    K[S - 1], V[S - 1] = k16, v
    Kh, Vh = K.view(S, A, dk).permute(1, 0, 2), V.view(S, A, dk).permute(1, 0, 2)
    p = torch.softmax((Kh * q.view(A, 1, dk)).sum(-1) * 0.5 + t(mask), -1)
    ctx = (q16(p).unsqueeze(-1) * Vh).sum(1).reshape(H)
    show("ctx", ctx.numpy(), parts["ctx"], True)
    x = t(x0) + q16(ctx) @ t(lw["wo"]).T + t(lw["bo"])
    show("h1", x.numpy(), parts["h1"])
    out, _, _ = layer_step_torch(cfg, lw, x0, k_slots[0], v_slots[0], mask)
    show("out", out, exact_ref.layer_step(cfg, lw, x0, k_slots[0], v_slots[0], mask)[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
