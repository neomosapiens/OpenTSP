#!/usr/bin/env python3
"""Measure rounded PyTorch reference's numeric-closeness bounds on a set of independent prompts of the real checkpoint (SHOWCASE-2).

rounded PyTorch reference compares the chip with an fp32 PyTorch reference, so it measures numeric closeness; conformance is bit-exact reference (every dumped word bit-equal to the
numpy twin). The bounds of `driver_gates.py` are a stated multiple of the largest value this set shows. This prints one JSON line per prompt
and a summary line, and exits 1 when a measured maximum is above its bound.

usage: python3 calibrate_comparison.py [--tokens N] [--shard I/N]   (the real checkpoint and tokenizer must be in the local Hugging Face cache)
"""
import argparse
import json
import sys
import tempfile

from opentsp_vllm.models.tinystories import driver
from opentsp_vllm.models.tinystories import driver_gates as dg
from opentsp_vllm.models.tinystories import comparison as comparison_ref

PROMPTS = [
    "Once upon a time", "The rabbit", "Café ☕", "突然", "Why?", "I saw a fox.",
    "The old wizard opened the door and", "1, 2, 3,", "She whispered:", "A robot had no friends.",
    "ONCE UPON A TIME", "The moon is blue.", "Hello\n\n", "the the the the the the the",
    "Mr. Smith", "The pirate found a map", "  A cat", "It was raining when",
    "Can we go home?", "Lily said, \"I don't know.\"",
    "One day, a little bird", "Tom and Sue went to the park.", "The dragon was very hungry, so",
    "Mom said that it was time for bed.",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, default=56, help="tokens generated after each prompt (the prompt and the tokens share 64 steps)")
    ap.add_argument("--shard", default="0/1", help="I/N: run the prompts I, I+N, ... of the set (N processes cover it)")
    a = ap.parse_args(argv)
    index, count = (int(x) for x in a.shard.split("/"))
    prompts = PROMPTS[index::count]
    w, why = comparison_ref.load_weights()
    tok, why2 = driver.load_tokenizer() if w is not None else (None, "")
    if w is None or tok is None:
        print(f"SKIP calibrate_comparison: {why or why2}")
        return 0
    work = tempfile.mkdtemp(prefix="calibrate-rounded-")
    prog = driver.StepProgram(w, work)
    worst = dict(err=0.0, kv_err=0.0, free_err=0.0, min_cos=1.0, free_min_cos=1.0)
    for n, p in enumerate(prompts):
        ids = tok.encode(p).ids
        n_new = min(a.tokens, driver.MAX_STEPS - len(ids) + 1)
        r = driver.generate(w, ids, n_new, prog, {"exact"}, driver.EOS, say=lambda *_: None)
        h = r["hiddens"]
        g = dg.check_rounded(w, r["tokens"][:len(h)], h, r["kv"], lambda *_: None)
        worst["err"], worst["kv_err"] = max(worst["err"], g["max_err"]), max(worst["kv_err"], g["max_kv_err"])
        worst["free_err"] = max(worst["free_err"], g["free_max_err"])
        worst["min_cos"], worst["free_min_cos"] = min(worst["min_cos"], g["min_cos"]), min(worst["free_min_cos"], g["free_min_cos"])
        print(json.dumps(dict(n=n, prompt=p, steps=len(h), reference_words=r["reference_words"], hidden_err=g["max_err"], kv_err=g["max_kv_err"],
                              free_err=g["free_max_err"], min_cos=g["min_cos"], free_min_cos=g["free_min_cos"]), ensure_ascii=False), flush=True)
    prog.check_unchanged()
    bounds = dict(err=dg.ERR_ROUNDED, kv_err=dg.ERR_KV_ROUNDED, free_err=dg.FREE_ERR_ROUNDED)
    ok = all(worst[k] <= b for k, b in bounds.items()) and worst["min_cos"] >= comparison_ref.COSINE_MIN and worst["free_min_cos"] >= comparison_ref.COSINE_MIN
    print(json.dumps(dict(summary=True, prompts=len(prompts), worst=worst, bounds=bounds,
                          headroom={k: bounds[k] / worst[k] for k in bounds}, ok=ok)), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
