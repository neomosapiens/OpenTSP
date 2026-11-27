#!/usr/bin/env python3
"""Greedy tokens of bit-exact reference (the bit-exact twin of the SHIP-style GPT-2 small, `reference_model`) against Hugging Face GPT-2 small on the
f16-rounded checkpoint (matrices and both embedding tables rounded to f16, the rest as published; the same rounding the chip
weights have). For each prompt: the prompt's ids, the tokens of both, whether they are equal, the smallest top-1 margin of
the Hugging Face logits over the generated steps and the largest absolute difference of the two logit vectors.

    python3 tests/showcase/gpt2small/compare_hf.py [--checkpoint gpt2|PATH] [--prompt TEXT ...] [--new 8] [--pv mxm|vxm] [--pool 128]

With no --prompt the three standard prompts are used. Exit status 0 when every prompt's tokens are equal, 1 when one differs,
2 for a bad input."""
import argparse
import sys
import time

import numpy as np


from opentsp_vllm.models.gpt2.config import Gpt2Config
from opentsp_vllm.models.gpt2 import reference as G
from opentsp_vllm.models.gpt2 import weights as W

PROMPTS = ("The capital of France is", "Once upon a time, in a small village by the sea, there lived", "def add(a, b):\n    return")


def hf_model(path):
    """GPT2LMHeadModel with the checkpoint rounded to f16 where the chip holds f16, and the tokenizer."""
    import torch                                                   # noqa: PLC0415
    from safetensors.torch import load_file                        # noqa: PLC0415
    from transformers import GPT2Config, GPT2LMHeadModel           # noqa: PLC0415
    import os
    cfg = GPT2Config.from_pretrained(os.path.dirname(path))
    model = GPT2LMHeadModel(cfg).eval()
    sd = load_file(path)
    sd = {(k if k.startswith("transformer.") or k == "lm_head.weight" else "transformer." + k): v for k, v in sd.items()}
    for k, v in sd.items():
        if k.endswith(("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight", "wte.weight", "wpe.weight")):
            sd[k] = v.half().float()
    model.load_state_dict(sd, strict=False)
    model.tie_weights()
    return torch, model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="gpt2")
    ap.add_argument("--prompt", action="append")
    ap.add_argument("--new", type=int, default=8)
    ap.add_argument("--pv", choices=("mxm", "vxm"), default="mxm")
    ap.add_argument("--pool", type=int, default=128, help="KV slots of the pool (a multiple of 128; the chip's is 1024)")
    a = ap.parse_args(argv)
    if a.new < 1:
        print(f"compare_hf: error: --new {a.new} must be at least 1", file=sys.stderr)
        return 2
    if any(not t.strip() for t in a.prompt or ()):
        print("compare_hf: error: a prompt is empty", file=sys.stderr)
        return 2
    cfg = Gpt2Config(pool=a.pool).check() if a.pool % 128 == 0 else None
    if cfg is None:
        print(f"compare_hf: error: --pool {a.pool} is not a multiple of 128", file=sys.stderr)
        return 2
    try:
        path = a.checkpoint if a.checkpoint.endswith(".safetensors") else W.find_checkpoint(a.checkpoint)
        w = W.load_checkpoint(cfg, path)
    except (ValueError, FileNotFoundError) as e:
        print(f"compare_hf: error: {e}", file=sys.stderr)
        return 2
    from transformers import AutoTokenizer                          # noqa: PLC0415
    import os
    tok = AutoTokenizer.from_pretrained(os.path.dirname(path))
    torch, model = hf_model(path)
    bad = 0
    for text in a.prompt or PROMPTS:
        ids = tok(text)["input_ids"]
        if len(ids) + a.new > cfg.pool:
            print(f"compare_hf: error: prompt of {len(ids)} tokens and {a.new} new tokens exceed the pool of {cfg.pool}", file=sys.stderr)
            return 2
        t0 = time.time()
        reference_ids, reference_logits = G.generate(w, ids, a.new, pv=a.pv)
        t_reference = time.time() - t0
        hf_ids, hf_logits, margins = list(ids), [], []
        with torch.no_grad():
            for _ in range(a.new + 1):
                lg = model(torch.tensor([hf_ids])).logits[0, -1].numpy()
                hf_logits.append(lg)
                top2 = np.sort(lg)[-2:]
                margins.append(float(top2[1] - top2[0]))
                if len(hf_ids) - len(ids) < a.new:
                    hf_ids.append(int(np.argmax(lg)))
        # bit-exact reference's `steps` holds the logits of the generated steps (the last prompt token and each generated token but the last)
        diff = max(float(np.abs(g.astype(np.float64) - h.astype(np.float64)).max()) for g, h in zip(reference_logits, hf_logits[:len(reference_logits)]))
        same = reference_ids == hf_ids
        bad += not same
        print(f"prompt {text!r}: {len(ids)} ids {ids}")
        print(f"  bit-exact reference ({a.pv}, pool {cfg.pool}, {t_reference:.0f} s): {reference_ids[len(ids):]} {tok.decode(reference_ids[len(ids):])!r}")
        print(f"  HF f16-rounded checkpoint:        {hf_ids[len(ids):]} {tok.decode(hf_ids[len(ids):])!r}")
        print(f"  tokens equal: {same}; smallest top-1 margin {min(margins):.3f}; largest |logit difference| {diff:.4f}")
    print("all prompts equal" if not bad else f"{bad} prompt(s) differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
