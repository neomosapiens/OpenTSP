#!/usr/bin/env python3
"""rounded PyTorch reference check of the block for the real checkpoint's layer 0 . Optional (not part of `just test`): it
needs the checkpoint in the local Hugging Face cache (its `model.safetensors` is read without PyTorch); without it
the script prints a SKIP line with the reason and exits 0. The PyTorch comparison needs PyTorch; without it the bit-exact reference
part still runs and its lines start with `SKIP-PARTIAL (bit-exact reference only passed):`, never `PASS`.

For each position (0, 15, 40, 62) it
  1. builds the block with the checkpoint's layer-0 weights bound as the program's weight package (the program is the
     same for any weights: model weights are `.safetensors` packages, never embedded) and runs it through
     `tests/programs/run.py` on tsp_sim, the ISS and the native ISS: every dumped word equals bit-exact reference, bit for bit;
  2. runs the same layer step in PyTorch (weights and every matmul input rounded to f16 at the points the chip rounds,
     everything else f32) and requires bit-exact reference's block output within 1e-3 of it (relative to max |x|) and cosine >= 0.99999.

    python3 check_block.py [--checkpoint NAME] [--online] [--tolerance 1e-3]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from opentsp_vllm.models.tinystories import paths as _paths
from opentsp_tools import profile as tprofile
from opentsp_vllm.models.tinystories import emit_block as eb
from opentsp_vllm.models.tinystories import reference as exact_ref
from opentsp_vllm.models.tinystories import comparison as comparison_ref
from opentsp_vllm.models.tinystories.comparison import DEFAULT_CHECKPOINT

ROOT = _paths.ROOT


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


def run_case(p, pos, w, iss, sim):
    """Build the block at `pos` with weights `w` and run it through run.py: tsp_sim, the ISS and the native ISS (the
    case is tagged `sim`), every dumped word equal to bit-exact reference. Returns None, or the failure text."""
    text, case = eb.build(p, pos, w=w)
    with tempfile.TemporaryDirectory(prefix="ts2-rounded-") as d:
        name = f"ts_block_real_p{pos:02d}.ref"
        case["program"] = name + ".ots"
        open(os.path.join(d, name + ".ots"), "w").write(text)
        path = os.path.join(d, name + ".case.json")
        json.dump(case, open(path, "w"))
        env = dict(os.environ, TSP_SIM=sim)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "programs", "run.py"), "--iss", iss, path],
                           capture_output=True, text=True, env=env)
        return None if r.returncode == 0 else (r.stdout + r.stderr)[-2000:]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--online", action="store_true", help="allow a download (default: the local cache only)")
    ap.add_argument("--tolerance", type=float, default=1e-3, help="rounded PyTorch reference against bit-exact reference, relative to max |x|")
    ap.add_argument("--iss", default=os.path.join(ROOT, "target", "release", "tsp-iss"))
    ap.add_argument("--sim", default=os.environ.get("TSP_SIM", os.path.join(ROOT, "target", "spec", "tsp_sim")))
    a = ap.parse_args(argv)
    w, why = comparison_ref.load_weights(a.checkpoint, a.online)
    if w is None:
        print(f"SKIP check_block: checkpoint {a.checkpoint!r} is not available ({why}); "
              "bit-exact reference is checked by `just showcase-block`")
        return 0
    no_torch = comparison_ref.torch_missing()
    if not os.access(a.iss, os.X_OK):
        print(f"check_block: tsp-iss not built at {a.iss} (cargo build --release -p tsp-iss -p ots-asm)", file=sys.stderr)
        return 1
    p = tprofile.load("ref")
    bad = 0
    for pos in eb.POSITIONS:
        fail = run_case(p, pos, w, a.iss, a.sim)
        inp = eb.block_inputs(pos, w=w)
        out0 = exact_ref.layer_step(inp.cfg, inp.layer, inp.x0, inp.k_slots, inp.v_slots, inp.mask)[0]
        line = (f"pos {pos}: block == bit-exact reference on tsp_sim, the ISS and the native ISS ({why}): "
                f"{'yes' if fail is None else 'NO'}")
        ok = fail is None
        if no_torch:
            line += f"; SKIP the PyTorch comparison: {no_torch}"
        else:
            out1 = layer_step_torch(inp.cfg, inp.layer, inp.x0, inp.k_slots, inp.v_slots, inp.mask)[0]
            err, cos = comparison_ref.agreement(out0, out1)
            ok = ok and err <= a.tolerance and cos >= comparison_ref.COSINE_MIN
            line += f"; rounded PyTorch reference vs bit-exact reference: max rel err {err:.2e} (<= {a.tolerance:g}), cosine {cos:.7f} (>= {comparison_ref.COSINE_MIN})"
        bad += not ok
        print(f"{comparison_ref.verdict(ok, no_torch)} {line}")
        if fail:
            print(fail)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
