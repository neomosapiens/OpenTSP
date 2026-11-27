#!/usr/bin/env python3
"""Weight export for the TinyStories showcase: logical weights and a physical layout manifest.

    python3 export.py --synthetic SEED --out DIR              seeded synthetic weights of the checkpoint's shapes
    python3 export.py --checkpoint [NAME] --out DIR           real weights (optional: needs torch + transformers;
                                                              downloads or reads the Hugging Face cache; nothing
                                                              derived from a checkpoint is committed)
    python3 export.py ... --raw                               also one raw file per tensor (`wpack` input form)
    python3 export.py ... --mode serve [--layers 8|2]         the layout of the batched, paged, whole-model step (embedding parts, packed head, masks, pools, logits, arenas)
                                                              and DIR/layout_report.json (occupancy per slice and bank)

Writes logical tensors including embeddings to DIR/weights.safetensors and the physical placement to
DIR/layout.json. With --raw it also writes materialized physical words to DIR/raw/<tensor>.bin.
"""
import argparse
import hashlib
import json
import os
import sys

from opentsp_vllm.models.tinystories import layout as L
from opentsp_vllm.models.tinystories import weights as W
from opentsp_vllm.models.tinystories.config import TINYSTORIES, ModelConfig

from opentsp_tools import refasm
tsp_as = refasm.load()

SCHEMA = "opentsp.showcase-layout"


def build(cfg: ModelConfig, w: W.Weights, profile="ref", mode="step"):
    prof = L.load_profile(profile)
    if mode == "serve":
        plan = L.plan_serve(cfg, prof)
        tensors, objects = L.build_serve_images(cfg, plan, w)
    else:
        plan = L.plan_tinystories(cfg, prof)
        tensors, objects = L.build_images(cfg, plan, w)
    return prof, plan, tensors, objects


def package_bytes(prof, plan, tensors):
    """The physical Safetensors bytes, tensors in plan order (the order of the layout manifest), by the reference writer."""
    order = [r.name for r in plan.regions if r.exported]
    return tsp_as.write_weights(prof.name, [(n, tensors[n][0], tensors[n][1]) for n in order])


def manifest(cfg, prof, plan, tensors, objects, weights_digest, pkg_sha, mode="step"):
    ts = []
    for r in plan.regions:
        if r.exported:
            ts.append({"name": r.name, "slice": r.slices[0], "addr": r.addr, "words": r.words, "dtype": tensors[r.name][0],
                       "kind": r.kind, "bank": r.bank, "note": r.note})
    regions = [{"name": r.name, "kind": r.kind, "slices": [r.slices[0], r.slices[-1]] if len(r.slices) > 1 else [r.slices[0]],
                "words_per_slice": r.words, "addr": r.addr, "bank": r.bank, "rw": r.rw, "exported": r.exported}
               for r in plan.regions if not r.exported]
    m = {
        "schema": SCHEMA, "version": 1, "spec_version": "0.1", "profile": prof.name,
        "model": {"hidden": cfg.hidden, "heads": cfg.heads, "ffn": cfg.ffn, "layers": cfg.layers, "vocab": cfg.vocab,
                  "slots": cfg.slots, "current_slot": cfg.slots - 1},
        "sram": {"slices": prof.slices, "words_per_slice": prof.words_per_slice, "bank_bit": prof.bank_bit,
                 "vlanes": prof.vlanes},
        "weights_sha256": weights_digest, "package_sha256": pkg_sha,
        "externs": ts,                  # one `.extern name, slice=, addr=, size=` per tensor
        "regions": regions,             # runtime regions (host-written inputs, caches, activations, reserved)
        "objects": objects,
        "conflicts": [list(c) for c in plan.conflicts if set(plan.by_name[c[0]].slices) & set(plan.by_name[c[1]].slices)
                      and c[0].startswith("a_")],
        "usage": {k: v for k, v in plan.usage().items() if k != "per_slice"},
        "per_slice": plan.usage()["per_slice"],
    }
    if mode == "serve":
        m["mode"] = "serve"
        m["conflicts"] = []
        m["flow_arena"] = {"region": "fa<t>_<hemisphere><quad>", "lanes": L.LANES_MAX, "bank": 1, "scopes": list(L.FLOW_SCOPES),
                           "words": {f"{h}{q}": parts for (h, q), parts in sorted(L.FLOW_ARENA.items())},
                           "tables": {k: [v[1], list(v[0]), v[2]] for k, v in L.FLOW_TABLES.items()},
                           "note": "a lane's activation vectors: one word each, by (scope, hemisphere, quad, ordinal); the same words in every "
                                   "bucket and layer; `fhm` and `foh` are the Flow's f16 tables"}
        m["pool_conflicts"] = {"region": "kp", "against": ["w<l>s<slice>", "hd<slice>", "tx<s>"],
                               "note": "on the east slices 44..59 the K pool (bank 1) never shares a bank with weights, head or text (bank 0)"}
        m["serve"] = {"blocks": L.SERVE_BLOCKS, "sink_block": L.SINK_BLOCK, "block_tokens": L.BLOCK_TOKENS,
                      "contexts": list(L.MASK_CS), "lanes": L.LANES_MAX, "logit_rows": L.LOGIT_ROWS,
                      "head_installs": L.HEAD_INSTALLS, "pack": L.PACK, "layers": cfg.layers}
    return m


def write(out, cfg, w, profile="ref", raw=False, mode="step"):
    os.makedirs(out, exist_ok=True)
    prof, plan, tensors, objects = build(cfg, w, profile, mode)
    pkg = package_bytes(prof, plan, tensors)
    pkg_sha = hashlib.sha256(pkg).hexdigest()
    logical = W.encode_deployment(w)
    open(os.path.join(out, "weights.safetensors"), "wb").write(logical)
    m = manifest(cfg, prof, plan, tensors, objects, w.digest(), pkg_sha, mode)
    m["logical_sha256"] = hashlib.sha256(logical).hexdigest()
    with open(os.path.join(out, "layout.json"), "w") as f:
        json.dump(m, f, indent=1, sort_keys=True)
        f.write("\n")
    if mode == "serve":
        with open(os.path.join(out, "layout_report.json"), "w") as f:
            json.dump(L.report(plan, cfg), f, indent=1, sort_keys=True)
            f.write("\n")
    if raw:
        os.makedirs(os.path.join(out, "raw"), exist_ok=True)
        for r in plan.regions:
            if r.exported:
                open(os.path.join(out, "raw", r.name + ".bin"), "wb").write(b"".join(tensors[r.name][1]))
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--synthetic", type=int, metavar="SEED")
    g.add_argument("--checkpoint", nargs="?", const="calum/tinystories-gpt2-3M", metavar="NAME")
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default="ref")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--mode", choices=("step", "serve"), default="step")
    ap.add_argument("--layers", type=int, default=None, help="serve mode: 8 (default) or 2 (the test model, synthetic only)")
    a = ap.parse_args(argv)
    if a.layers is not None and a.mode != "serve":
        ap.error("--layers needs --mode serve")
    if a.layers not in (None, 8) and a.synthetic is None:
        ap.error("the 2-layer model has synthetic weights only")
    cfg = L.serve_config(a.layers or 8) if a.mode == "serve" else TINYSTORIES
    w = W.synthetic(cfg, a.synthetic) if a.synthetic is not None else W.load_checkpoint(cfg, a.checkpoint)
    m = write(a.out, cfg, w, a.profile, a.raw, a.mode)
    u = m["usage"]
    print(f"wrote {a.out}/weights.safetensors: {len(m['externs'])} tensors, sha256 {m['package_sha256'][:16]}; "
          f"{u['words_used']} of {u['words_capacity']} words in {u['slices_used']} slices "
          f"(at most {u['max_words_in_a_slice']} in one slice)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
