#!/usr/bin/env python3
"""Optional rounded PyTorch reference/FP32 reference reference (PyTorch). Not used by the tests; needs `torch` (and, for --checkpoint, `transformers`).

FP32 reference: the original fp32 model run in PyTorch. rounded PyTorch reference: the same forward with the weights and every matmul input rounded
to f16 at the points the chip program rounds . Writes a JSON golden: token ids and the final hidden
state per step. For synthetic weights the sanity reference of the bit-exact reference model is `float_ref.py` (numpy float64, no PyTorch).

    python3 torch_ref.py --synthetic 1 --tokens 6 --out golden.json
"""
import argparse
import json
import sys

import numpy as np

from opentsp_vllm.models.tinystories import weights as W
from opentsp_vllm.models.tinystories.config import TINYSTORIES, PROMPT_IDS


def forward_torch(w, ids, f16_inputs):
    import torch                                                     # noqa: PLC0415 (optional dependency)
    t = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
    q16 = (lambda x: x.half().float()) if f16_inputs else (lambda x: x)
    cfg = w.cfg
    T, H, A, dk = len(ids), cfg.hidden, cfg.heads, cfg.dk
    ln = lambda x, g, b: torch.nn.functional.layer_norm(x, (H,), t(g), t(b), cfg.ln_eps)
    x = t(w.wte[list(ids)]) + t(w.wpe[:T])
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    for lw in w.layers:
        n = q16(ln(x, lw["ln1g"], lw["ln1b"]))
        q, k, v = (n @ t(lw[m]).T + t(lw[b]) for m, b in (("wq", "bq"), ("wk", "bk"), ("wv", "bv")))
        q, k = q16(q), q16(k)
        q, k, v = (a.view(T, A, dk).transpose(0, 1) for a in (q, k, v))
        s = (q @ k.transpose(1, 2)) * 0.5
        p = torch.softmax(s.masked_fill(~mask, float("-inf")), -1)
        ctx = (q16(p) @ v).transpose(0, 1).reshape(T, H)
        x = x + q16(ctx) @ t(lw["wo"]).T + t(lw["bo"])
        n = q16(ln(x, lw["ln2g"], lw["ln2b"]))
        g = torch.nn.functional.gelu(n @ t(lw["up_w"]).T + t(lw["up_b"]), approximate="tanh")
        x = x + q16(g) @ t(lw["dn_w"]).T + t(lw["dn_b"])
    h = ln(x, w.lnf_g, w.lnf_b)
    return h.numpy(), (h @ t(w.wte).T).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--synthetic", type=int)
    g.add_argument("--checkpoint", nargs="?", const="calum/tinystories-gpt2-3M")
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    w = W.synthetic(TINYSTORIES, a.synthetic) if a.synthetic is not None else W.load_checkpoint(TINYSTORIES, a.checkpoint)
    out = {}
    for tag, f16 in (("rounded", True), ("fp32", False)):
        ids = list(PROMPT_IDS)
        for _ in range(a.tokens + 1):
            h, lg = forward_torch(w, ids, f16)
            if len(ids) == len(PROMPT_IDS) + a.tokens:
                break
            ids.append(int(np.argmax(lg[-1])))
        out[tag] = {"token_ids": ids, "hidden": h.tolist()}
    json.dump(out, open(a.out, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
