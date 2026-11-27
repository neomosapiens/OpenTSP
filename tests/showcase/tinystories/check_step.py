#!/usr/bin/env python3
"""rounded PyTorch reference check of the decode step for the real checkpoint . Optional (not part of `just test`): it needs
the checkpoint in the local Hugging Face cache (its `model.safetensors` is read without PyTorch); without it the script
prints a SKIP line with the reason and exits 0. The PyTorch comparison needs PyTorch; without it the bit-exact reference part still runs
and its lines start with `SKIP-PARTIAL (bit-exact reference only passed):`, never `PASS`.

For each position (default 5: the sixth prompt token, with the cache bit-exact reference's own steps wrote for the five before it) it
  1. builds the step with the real weights bound as the program's weight packages and the head window holding bit-exact reference's
     token, and runs it through `tests/programs/run.py` on tsp_sim (`--sim`), the ISS and the native ISS: every dumped
     word (each layer's output, its K and V rows, the final hidden state, the logits of the window) equals bit-exact reference bit for
     bit;
  2. runs the same step in PyTorch from the same K/V slots (weights and every matmul input rounded to f16 at the
     points the chip rounds, everything else f32) and requires bit-exact reference's final hidden state within 1e-3 of it (relative to
     max |h|) and cosine >= 0.99999, the logits' cosine >= 0.99999, and the same greedy token.

    python3 check_step.py [--pos 5 [--pos 62 ...]] [--sim PATH | --no-sim] [--checkpoint NAME] [--online]
"""
import argparse
import os
import sys

import numpy as np

from opentsp_vllm.models.tinystories import paths as _paths

from opentsp_tools import profile as tprofile
from opentsp_vllm.models.tinystories import emit_step as es
from opentsp_vllm.models.tinystories import reference as exact_ref
from opentsp_vllm.models.tinystories import comparison as comparison_ref
import case_runner as runner_util
from opentsp_vllm.models.tinystories.torch_reference import layer_step_torch

ROOT = _paths.ROOT


def step_torch(inp):
    """The step in PyTorch from the slot layout: (final hidden (H), logits (vocab)) as float32 numpy arrays."""
    import torch                                                     # noqa: PLC0415 (optional dependency)
    cfg, w = inp.cfg, inp.w
    x = inp.x0
    for l, lw in enumerate(w.layers):
        x = layer_step_torch(cfg, lw, x, inp.k_slots[l], inp.v_slots[l], inp.mask)[0]
    t = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
    h = torch.nn.functional.layer_norm(t(x), (cfg.hidden,), t(w.lnf_g), t(w.lnf_b), cfg.ln_eps)
    return h.numpy(), (h @ t(w.wte).T).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pos", type=int, action="append")
    ap.add_argument("--checkpoint", default=comparison_ref.DEFAULT_CHECKPOINT)
    ap.add_argument("--online", action="store_true", help="allow a download (default: the local cache only)")
    ap.add_argument("--tolerance", type=float, default=1e-3, help="rounded PyTorch reference against bit-exact reference, relative to max |h|")
    ap.add_argument("--sim", default=os.environ.get("TSP_SIM", os.path.join(ROOT, "target", "spec", "tsp_sim")))
    ap.add_argument("--no-sim", action="store_true", help="the ISS legs only (about a minute faster per position)")
    a = ap.parse_args(argv)
    w, why = comparison_ref.load_weights(a.checkpoint, a.online)
    if w is None:
        print(f"SKIP check_step: checkpoint {a.checkpoint!r} is not available ({why}); bit-exact reference is checked by "
              "`just showcase-block STEP=1`")
        return 0
    no_torch = comparison_ref.torch_missing()
    p = tprofile.load("ref")
    sim = None if a.no_sim else a.sim
    bad = 0
    for pos in a.pos or [5]:
        text, case = es.build(p, pos, w=w)
        rc, out = runner_util.run_on_iss(text, case, sim=sim)
        inp = es.step_inputs(pos, w=w)
        res = exact_ref.decode_step_slots(inp.w, inp.token, pos, inp.k_slots, inp.v_slots, inp.mask)
        legs = "the ISS and the native ISS" if sim is None else "tsp_sim, the ISS and the native ISS"
        line = f"pos {pos}: step == bit-exact reference on {legs} ({why}): {'yes' if rc == 0 else 'NO'}; bit-exact reference token {inp.argmax}"
        ok = rc == 0
        if no_torch:
            line += f"; SKIP the PyTorch comparison: {no_torch}"
        else:
            h1, lg1 = step_torch(inp)
            err, cos = comparison_ref.agreement(res.hidden, h1)
            lcos = comparison_ref.cosine(res.logits, lg1)
            tok = int(np.argmax(lg1))
            ok = ok and err <= a.tolerance and cos >= comparison_ref.COSINE_MIN and lcos >= comparison_ref.COSINE_MIN and tok == inp.argmax
            line += (f"; rounded PyTorch reference vs bit-exact reference: hidden max rel err {err:.2e} (<= {a.tolerance:g}), cosine {cos:.7f}, logits cosine "
                     f"{lcos:.7f} (>= {comparison_ref.COSINE_MIN}), PyTorch token {tok}")
        bad += not ok
        print(f"{comparison_ref.verdict(ok, no_torch)} {line}", flush=True)
        if rc != 0:
            print(out[-2000:])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
