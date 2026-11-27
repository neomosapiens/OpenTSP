"""GPT-2 small layers on a chain of chips through the binding's `System` : one layer per chip, 128-lane tiles (`opentsp-vllm/src/opentsp_vllm/models/gpt2/layer.py`),
the weights of the real checkpoint in `.safetensors` packages next to the objects, the host's vectors and the pools written by the host, the output read back
as the PciSend vectors of the last chip.

    python3 ship_system.py --layers 2 --tokens 15496,11,995,0      # layers 0 and 1 on 2 chips, the last token through the System

The expectation is the twin (`ship_layer.twin`, `ref_wide` at tile 128): the context's K and V rows of every layer come from the twin run over the earlier
tokens of the prompt (they are the content of the pool before the step), the last token's x is the embedding `wte[t] + wpe[pos]` (the host's, f32 add
in the twin's arithmetic), and the output of the last layer's PciSend vectors must equal the twin's output tiles bit for bit.

Every stage's object is assembled by both assemblers (identical objects and packages are required, as in `tests/programs/run.py`), loaded with
`System.load_object`, and its weights package is bound to the SRAM by the host with `System.write_words` before the first run (the binding has no
weight loader of its own; the words are the ones `ots-asm objinput` binds from the `.safetensors`)."""
import argparse
import os
import shutil
import tempfile
import time

import numpy as np

from opentsp_vllm.models.gpt2 import paths as _gpaths
ROOT = _gpaths.ROOT
from opentsp_tools import build as R                                              # noqa: E402
from opentsp_tools import profile as tprofile
from opentsp_tools.programs import ref
from opentsp_tools.programs import ref_wide as rw
from opentsp_vllm.models.gpt2 import layer as sl
from opentsp_vllm.models.gpt2.attention import CHUNK
from opentsp_vllm.models.gpt2.config import GPT2_SMALL
from opentsp_vllm.models.gpt2 import weights as W

T = GPT2_SMALL.hidden // sl.TILE            # 6 tiles of 128 lanes
CHUNKS = 8                                  # pool 1024 slots
PAGE = 16
CHECKPOINT = os.path.expanduser("~/.cache/opentsp/checkpoints/gpt2/model.safetensors")


def layer_params(lw):
    """The parameters of a checkpoint layer in the form of `ship_layer.params` (Q, K, V stacked in one matrix)."""
    cat = np.concatenate
    return dict(ln1g=lw["ln1g"], ln1b=lw["ln1b"], wqkv=cat([lw["wq"], lw["wk"], lw["wv"]]), bqkv=cat([lw["bq"], lw["bk"], lw["bv"]]),
                wo=lw["wo"], bo=lw["bo"], ln2g=lw["ln2g"], ln2b=lw["ln2b"], wu=lw["up_w"], bu=lw["up_b"], wd=lw["dn_w"], bd=lw["dn_b"])


def counts_for(live, chunks=CHUNKS, page=PAGE):
    """The page counts (chunks, pages per chunk) of a sequence in slots 0 .. live - 1."""
    pages = CHUNK // page
    c = np.clip(live - page * np.arange(chunks * pages), 0, page)
    return c.reshape(chunks, pages).astype(np.int64)


def mask_of(counts):
    return np.concatenate([rw.mask_from_encoding(counts[c], CHUNK, PAGE) for c in range(len(counts))])


def embed(w, token, pos):
    return ref.residual(w.wte[token].astype(np.float32), w.wpe[pos].astype(np.float32))


def twin_context(w, ids, nlayers, chunks=CHUNKS):
    """The twin over the tokens `ids[:-1]` through `nlayers` layers: the pools before the last token's step (per layer (k16, v16, counts of the live
    slots including the new one)), the last token's input x and the twin's output of every layer for it."""
    slots = CHUNK * chunks
    Ps = [layer_params(w.layers[l]) for l in range(nlayers)]
    k = [np.zeros((slots, T, sl.TILE), np.float16) for _ in range(nlayers)]
    v = [np.zeros((slots, T, sl.TILE), np.float16) for _ in range(nlayers)]
    last = len(ids) - 1
    for pos, tok in enumerate(ids):
        x = x_in = embed(w, tok, pos)
        counts = counts_for(pos + 1, chunks)
        mask = mask_of(counts)
        outs = []
        for l in range(nlayers):
            r = sl.twin(x, Ps[l], k[l], v[l], pos, mask, GPT2_SMALL.heads)
            if pos < last:
                k[l], v[l] = r["k16"], r["v16"]
            x = r["out"]
            outs.append(x)
    return Ps, [(k[l], v[l], counts) for l in range(nlayers)], x_in, outs


def build_stages(p, n, Ps, pools, x0, slot, work, tiles=T, chunks=CHUNKS, route=None, chips=None, fan=None, *, harness_input=True):
    """Build each chip's object and weights. Differential callers also get text-harness input."""
    stages, infos = sl.chain(p, n, tiles, chunks, slot=slot, params=Ps, pools=pools, x0=x0, route=route, chips=chips, fan=fan)
    runner = R.CaseBuilder(argparse_ns())
    out = []
    for i, (ots, case) in enumerate(stages):
        cdir = os.path.join(work, f"c{i}")
        os.makedirs(cdir)
        open(os.path.join(cdir, case["program"]), "w").write(ots)
        if harness_input:
            text, (obj, weights) = runner.build_input(case, cdir, cdir)
            out.append(dict(case=case, elf=obj, sram_weights=weights, text=text))
        else:
            obj, weights = runner.build_artifacts(case, cdir, cdir)
            out.append(dict(case=case, elf=obj, sram_weights=weights))
    return out, infos


def argparse_ns():
    return argparse.Namespace(asm=os.environ.get("OTS_ASM", os.path.join(ROOT, "target", "release", "ots-asm")), sim=None, iss=None, tag=None, jobs=1,
                              timeout=1800)


def split_input(text, vlanes=320):
    """The SRAM words (`S slice addr bytes`) and the host's DMA vectors (`H dma cycle bytes`) of an objinput text."""
    sram, dma = [], []
    for ln in text.split("\n"):
        if ln.startswith("S "):
            f = ln.split()
            data = bytes(int(x, 16) for x in f[3:])
            assert len(data) % vlanes == 0, ln[:40]
            sram.append((int(f[1], 16), int(f[2], 16), data))
        elif ln.startswith("H dma "):
            f = ln.split()
            dma.append((int(f[2], 16), bytes(int(x, 16) for x in f[3:])))
    return sram, dma


def run_system(profile, stages, max_cycles, chips=None, route=None, mutate=None):
    """Build the System, load the objects and write the weights, push the host's vectors, run to the end. Returns (system, cycles, seconds)."""
    from opentsp_vllm.system import System
    t0 = time.time()
    route = list(range(len(stages))) if route is None else route
    s = System(profile, chips or len(stages))
    parts = [split_input(st["text"]) for st in stages]
    for c, st in zip(route, stages):
        s.load_object(c, st["elf"])
    for c, (sram, _) in zip(route, parts):
        for sli, addr, data in sram:
            s.write_words(c, sli, addr, data)
    for c, (_, dma) in zip(route, parts):
        for cycle, v in dma:
            s.dma_write(c, v, cycle)
    if mutate:
        mutate(s, route)
    t1 = time.time()
    _, res = s.run_to_done(max_cycles=max_cycles)
    s.finish()
    return s, res["cycle"], t1 - t0, time.time() - t1


def check(s, stages, route=None):
    """The last stage's PciSend vectors against the case's expectation (the twin's output planes); no diagnostic on any chip."""
    route = list(range(len(stages))) if route is None else route
    last = route[-1]
    for c in route:
        d = s.diagnostics(c)
        if d:
            raise SystemExit(f"FAIL chip {c}: diagnostics {d[:3]}")
    got = [v for _, v in s.egress(last)]
    want = [bytes.fromhex(h) for h in stages[-1]["case"]["expect"]["egress"]]
    if len(got) != len(want):
        raise SystemExit(f"FAIL: {len(got)} vectors PciSent, {len(want)} expected")
    for i, (g, wv) in enumerate(zip(got, want)):
        if g[:len(wv)] != wv:
            raise SystemExit(f"FAIL: PciSend vector {i} differs from the twin")
    return len(got)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layers", type=int, default=2, help="layers 0 .. LAYERS-1 on LAYERS chips")
    ap.add_argument("--tokens", default="15496,11,995,0", help="comma separated token ids; the last one is the step's token")
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--keep", help="keep the programs and packages in this directory")
    ap.add_argument("--max-cycles", type=int, default=0)
    a = ap.parse_args(argv)
    ids = [int(t) for t in a.tokens.split(",")]
    n = a.layers
    t0 = time.time()
    w = W.load_checkpoint(GPT2_SMALL, a.checkpoint)
    Ps, pools, x0, outs = twin_context(w, ids, n)
    print(f"twin over {len(ids)} tokens x {n} layers: {time.time() - t0:.1f} s", flush=True)
    p = tprofile.load("ref")
    work = tempfile.mkdtemp(prefix="ship-sys-")
    try:
        t1 = time.time()
        stages, infos = build_stages(p, n, Ps, pools, x0, len(ids) - 1, work)
        print(f"programs: {time.time() - t1:.1f} s; twin output of the chain equals the context twin: "
              f"{np.array_equal(infos[-1]['out'], outs[-1])}", flush=True)
        for i, inf in enumerate(infos):
            print(f"  chip {i}: stage ends at cycle {inf['end']}", flush=True)
        budget = a.max_cycles or max(st["case"]["budget"]["cycles"] for st in stages)
        s, cycles, t_load, t_run = run_system("ref", stages, budget)
        v = check(s, stages)
        print(f"PASS: {n} chips through System, {v} PciSend vectors equal the twin; {cycles} cycles; load {t_load:.1f} s, run {t_run:.1f} s")
    finally:
        if a.keep:
            shutil.rmtree(a.keep, ignore_errors=True)
            shutil.copytree(work, a.keep)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
