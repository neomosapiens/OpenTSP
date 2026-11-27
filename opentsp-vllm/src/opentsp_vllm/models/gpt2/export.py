#!/usr/bin/env python3
"""Export shared GPT-2 weights and optional per-stage SRAM arrays.

Run with python -m opentsp_vllm.models.gpt2.export --checkpoint NAME --out DIR.
Use --device-layout to also emit physical Safetensors for standalone runs.
The four-chip serving bundle uses just gpt2-programs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from opentsp_tools import refasm
tsp_as = refasm.load()
from opentsp_tools.programs import ref_wide as rw
from opentsp_vllm.models.gpt2.config import GPT2_SMALL, Gpt2Config
from opentsp_vllm.models.gpt2 import weights as W

VLANES = 320
SCHEMA = "opentsp.gpt2-small-tiled"


class ExportError(ValueError):
    pass


def matrix_words(w):
    """(rows, n_in) f16 -> (tiles * rows * 2, 320) uint8, word order [tile][row][plane]."""
    rows = w.shape[0]
    wt = rw.split(np.ascontiguousarray(w, np.float16))                  # (rows, tiles, 256)
    u8 = wt.view(np.uint8).reshape(rows, wt.shape[1], rw.TILE, 2)
    out = np.zeros((wt.shape[1], rows, 2, VLANES), np.uint8)
    for p in range(2):
        out[:, :, p, :rw.TILE] = u8[..., p].transpose(1, 0, 2)
    return out.reshape(-1, VLANES)


def matrix_from_words(words, rows, n_in):
    """The inverse of `matrix_words`: the (rows, n_in) f16 matrix."""
    tiles = rw.tiles_of(n_in)
    a = np.asarray(words, np.uint8).reshape(tiles, rows, 2, VLANES)
    u8 = np.zeros((rows, tiles, rw.TILE, 2), np.uint8)
    for p in range(2):
        u8[..., p] = a[:, :, p, :rw.TILE].transpose(1, 0, 2)
    return np.ascontiguousarray(u8.view(np.float16).reshape(rows, tiles * rw.TILE)[:, :n_in])


def vector_words(v):
    """(..., n) f32 -> (vectors * tiles * 4, 320) uint8: word [vector][tile][plane]."""
    vt = rw.split(np.ascontiguousarray(v, np.float32))                  # (..., tiles, 256)
    u8 = vt.view(np.uint8).reshape(vt.shape + (4,))
    out = np.zeros(vt.shape[:-1] + (4, VLANES), np.uint8)
    for p in range(4):
        out[..., p, :rw.TILE] = u8[..., p]
    return out.reshape(-1, VLANES)


def vector_from_words(words, n):
    tiles = rw.tiles_of(n)
    a = np.asarray(words, np.uint8).reshape(-1, tiles, 4, VLANES)
    u8 = np.zeros(a.shape[:2] + (rw.TILE, 4), np.uint8)
    for p in range(4):
        u8[..., p] = a[:, :, p, :rw.TILE]
    f = u8.view(np.float32).reshape(a.shape[0], tiles * rw.TILE)[:, :n]
    return np.ascontiguousarray(f)


def layer_tensors(lw):
    """{name: (kind, array of words)} of one layer: f16 matrices and f32 vectors, install order."""
    qkv = np.concatenate([lw["wq"], lw["wk"], lw["wv"]])
    bqkv = np.concatenate([lw["bq"], lw["bk"], lw["bv"]])
    return {
        "wqkv": matrix_words(qkv), "wo": matrix_words(lw["wo"]), "up": matrix_words(lw["up_w"]), "dn": matrix_words(lw["dn_w"]),
        "bqkv": vector_words(bqkv), "bo": vector_words(lw["bo"]), "upb": vector_words(lw["up_b"]), "dnb": vector_words(lw["dn_b"]),
        "ln1": vector_words(np.stack([lw["ln1g"], lw["ln1b"]])), "ln2": vector_words(np.stack([lw["ln2g"], lw["ln2b"]])),
    }


def stage_tensors(w, first, last, n_stages, index):
    """The tensors of stage `index`: layers first..last-1; the embedding tables on stage 0, the final LayerNorm and the head
    table on the last stage (a one-stage pipeline holds both)."""
    out = {}
    if index == 0:
        out["wte"] = matrix_words(w.wte)
        out["wpe"] = vector_words(w.wpe.astype(np.float32))
    for l in range(first, last):
        for name, words in layer_tensors(w.layers[l]).items():
            out[f"l{l:02d}_{name}"] = words
    if index == n_stages - 1:
        out["lnf"] = vector_words(np.stack([w.lnf_g, w.lnf_b]))
        if index != 0:
            out["wte"] = matrix_words(w.wte)
    return out


def package_bytes(tensors, profile="ref"):
    return tsp_as.write_weights(profile, [(n, None, [bytes(r) for r in words]) for n, words in tensors.items()])


def read_package(data, profile="ref"):
    """{name: (n words, (n, 320) uint8)} of a package, read with the reference reader."""
    return {n: (len(ws), np.frombuffer(b"".join(ws), np.uint8).reshape(-1, VLANES)) for n, (_, ws) in
            tsp_as.read_weights(data, profile).items()}


def weights_from_packages(cfg: Gpt2Config, packages):
    """The `Weights` of the stage packages (bytes, in stage order): the round trip of the export."""
    H, F = cfg.hidden, cfg.ffn
    tens = {}
    for pkg in packages:
        for n, (_, words) in read_package(pkg).items():
            tens.setdefault(n, words)
    mat = lambda n, r, k: matrix_from_words(tens[n], r, k)
    vec = lambda n, c: vector_from_words(tens[n], c)
    wpe = vec("wpe", H).astype(np.float16)
    w = W.Weights(cfg, mat("wte", cfg.vocab, H), wpe)
    for l in range(cfg.layers):
        p = f"l{l:02d}_"
        qkv, bqkv = mat(p + "wqkv", 3 * H, H), vec(p + "bqkv", 3 * H)[0]
        ln1, ln2 = vec(p + "ln1", H), vec(p + "ln2", H)
        w.layers.append(dict(
            wq=qkv[:H], wk=qkv[H:2 * H], wv=qkv[2 * H:], wo=mat(p + "wo", H, H), up_w=mat(p + "up", F, H), dn_w=mat(p + "dn", H, F),
            bq=bqkv[:H], bk=bqkv[H:2 * H], bv=bqkv[2 * H:], bo=vec(p + "bo", H)[0], up_b=vec(p + "upb", F)[0],
            dn_b=vec(p + "dnb", H)[0], ln1g=ln1[0], ln1b=ln1[1], ln2g=ln2[0], ln2b=ln2[1]))
    lnf = vec("lnf", H)
    w.lnf_g, w.lnf_b = lnf[0], lnf[1]
    return w.validate()


def export(w, out_dir, cfg=None, checkpoint=None, *, device_layout=False):
    """Write the shared tensors and manifest, optionally with physical stage images."""
    cfg = cfg or w.cfg
    os.makedirs(out_dir, exist_ok=True)
    stages = cfg.stages()
    W.save_deployment(os.path.join(out_dir, "weights.safetensors"), w)
    man = {"schema": SCHEMA, "version": 1, "spec_version": "0.1", "profile": "ref",
           "model": {"hidden": cfg.hidden, "heads": cfg.heads, "head_dim": cfg.dk, "ffn": cfg.ffn, "layers": cfg.layers,
                     "vocab": cfg.vocab, "max_pos": cfg.max_pos},
           "layout": {"tile_lanes": rw.TILE, "vlanes": VLANES, "score_chunk": rw.CHUNK, "page": rw.PAGE, "pool": cfg.pool,
                      "matrix": "d_out rows of ceil(d_in/256) f16 vectors; word (tile*rows + row)*2 + plane",
                      "vector": "ceil(n/256) f32 vectors; word tile*4 + plane", "stages_layers": list(cfg.stage_layers),
                      "kv": "f16 K and V, 256-lane vectors; 3 vectors per slot and layer"},
           "weights_sha256": w.digest(), "weights_file": "weights.safetensors", "checkpoint": checkpoint, "stages": []}
    for i, (a, b) in enumerate(stages):
        if not device_layout:
            man["stages"].append({"stage": i, "layers": [a, b]})
            continue
        tens = stage_tensors(w, a, b, len(stages), i)
        pkg = package_bytes(tens)
        path = os.path.join(out_dir, f"stage{i}.safetensors")
        with open(path, "wb") as f:
            f.write(pkg)
        man["stages"].append({
            "stage": i, "file": f"stage{i}.safetensors", "layers": [a, b], "package_sha256": hashlib.sha256(pkg).hexdigest(),
            "words": sum(len(t) for t in tens.values()), "bytes": len(pkg),
            "tensors": [{"name": n, "words": len(t), "sha256": hashlib.sha256(t.tobytes()).hexdigest()} for n, t in tens.items()]})
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
        f.write("\n")
    return man


def parse_stages(text, layers):
    try:
        st = tuple(int(x) for x in text.split(","))
    except ValueError:
        raise ExportError(f"--stages wants layer counts such as 2,7,3, not {text!r}") from None
    if sum(st) != layers or min(st) < 0:
        raise ExportError(f"--stages {text} holds {sum(st)} layers, the model has {layers}")
    return st


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="gpt2", help="a cached model name or a path to model.safetensors (default gpt2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default=",".join(map(str, GPT2_SMALL.stage_layers)), help="layers per stage (default 2,7,3)")
    ap.add_argument("--device-layout", action="store_true", help="also write physical SRAM word arrays")
    ap.add_argument("--verify", action="store_true", help="read the packages back and compare every array with the checkpoint's")
    a = ap.parse_args(argv)
    try:
        cfg = Gpt2Config(stage_layers=parse_stages(a.stages, GPT2_SMALL.layers)).check()
        w = W.load_checkpoint(cfg, a.checkpoint)
        man = export(w, a.out, cfg, a.checkpoint, device_layout=a.device_layout)
        if a.verify:
            back = W.load_deployment(cfg, Path(a.out, man["weights_file"]))
            if back.digest() != w.digest():
                raise ExportError("deployed tensors differ from the checkpoint")
            if a.device_layout:
                images = [Path(a.out, stage["file"]).read_bytes() for stage in man["stages"]]
                if weights_from_packages(cfg, images).digest() != back.digest():
                    raise ExportError("physical SRAM words differ from the logical tensors")
            print(f"verify: deployed tensors match the checkpoint (digest {back.digest()[:16]})")
    except (ValueError, FileNotFoundError) as e:
        print(f"gpt2s_export: error: {e}", file=sys.stderr)
        return 2
    print("wrote", Path(a.out, man["weights_file"]))
    for s in man["stages"] if a.device_layout else ():
        print(f"wrote {a.out}/{s['file']}: layers {s['layers'][0]}..{s['layers'][1] - 1}, {len(s['tensors'])} tensors, "
              f"{s['words']} words, {s['bytes'] / 1e6:.1f} MB, sha256 {s['package_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
