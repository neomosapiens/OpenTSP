"""The PyTorch tiers of the generation driver : rounded PyTorch reference and FP32 reference against the hidden states the chip produced.

rounded PyTorch reference: PyTorch with the weights and every matmul input rounded to f16 at the points the program rounds, one step at a time
on the K/V rows the chip wrote (`check_step.step_torch`): per step the final hidden state's cosine >= 0.99999 and max
difference <= ERR_ROUNDED max |h|, the step's new K and V rows within ERR_KV_ROUNDED of the row's maximum.
rounded PyTorch reference and FP32 reference measure numeric closeness to an independent reference; conformance is bit-exact reference (every dumped word bit-equal to the numpy twin). Their
bounds are a stated multiple of the largest value that `calibrate_comparison.py` measures on a set of independent prompts (see the constants).
FP32 reference: the original fp32 model (`transformers` GPT-2 on the checkpoint's own float32 weights), run on the ids the chip
consumed: per step the hidden-state cosine >= 0.999; the greedy token of every generated position (the first 20)
equal to the fp32 model's on the same prefix, a mismatch accepted only when that model's top-1/top-2 logit margin is
below 1e-3 (reported with the margin); the free-running greedy sequence of the fp32 model is reported next to the
chip's.
A requested comparison fails when its reference dependencies are unavailable.
"""
import os
import types

import numpy as np

from opentsp_vllm.models.tinystories import reference as exact_ref
from opentsp_vllm.models.tinystories import comparison as comparison_ref

COS_FP32 = 0.999
MARGIN_FP32 = 1e-3
TOKENS_FP32 = 20
# rounded PyTorch reference bounds, each about 1.5 times the largest value measured by `calibrate_comparison.py` on 24 independent prompts (650 steps; 6 of them again at 58 to 64 steps) of calum/tinystories-gpt2-3M
# ("rounded PyTorch reference calibration"): hidden state 1.04e-3 -> 1.6e-3; new K/V rows 1.66e-3 -> 2.5e-3;
# free-running sequence 2.16e-3 (58 steps) -> 3.3e-3. Each is far below the 1e-1 that a wrong embedding or cache placement gives.
ERR_ROUNDED = 1.6e-3
ERR_KV_ROUNDED = 2.5e-3
FREE_ERR_ROUNDED = 3.3e-3


def margin(logits):
    top = np.partition(np.asarray(logits, np.float64), -2)[-2:]
    return float(top[1] - top[0])


def fp32_model(name):
    """(model, None) of the original checkpoint in float32, or (None, the reason)."""
    why = comparison_ref.torch_missing()
    if why:
        return None, why
    try:
        from transformers import GPT2LMHeadModel                       # noqa: PLC0415 (optional dependency)
    except ImportError as e:
        return None, f"{e.name} is not installed (pip install transformers into the toolchain venv)"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return GPT2LMHeadModel.from_pretrained(name, torch_dtype="float32").eval(), None
    except Exception as e:                                             # not cached, offline
        return None, f"{type(e).__name__}: {str(e)[:160]}"


def fp32_forward(model, ids):
    """(final hidden states (T, H), logits (T, vocab)) of the fp32 model for the sequence `ids`."""
    import torch                                                       # noqa: PLC0415
    with torch.no_grad():
        out = model(torch.tensor([list(ids)]), output_hidden_states=True)
    return out.hidden_states[-1][0].numpy().astype(np.float32), out.logits[0].numpy()


def fp32_greedy(model, ids, n_new, eos=None):
    ids = list(ids)
    for _ in range(n_new):
        nxt = int(np.argmax(fp32_forward(model, ids)[1][-1]))
        ids.append(nxt)
        if eos is not None and nxt == eos:
            break
    return ids


def fp32_reference(name):
    """(reference, None) or (None, the reason): `forward(ids)` -> (hidden (T, H), logits (T, vocab)) and
    `greedy(ids, n_new, eos)` -> ids, of the original fp32 model; `fp32` takes any object with these two."""
    model, why = fp32_model(name)
    if model is None:
        return None, why
    return types.SimpleNamespace(forward=lambda ids: fp32_forward(model, ids),
                                 greedy=lambda ids, n_new, eos=None: fp32_greedy(model, ids, n_new, eos)), None


def step_torch_rows(inp):
    """The step in PyTorch from the slot layout: (final hidden (H), K rows (L, H) rounded to f16, V rows (L, H)) as
    float32 numpy arrays, the rows the chip writes to the current slot."""
    import torch                                                       # noqa: PLC0415
    from opentsp_vllm.models.tinystories.torch_reference import layer_step_torch
    cfg, w = inp.cfg, inp.w
    x, ks, vs = inp.x0, [], []
    for l, lw in enumerate(w.layers):
        x, k, v = layer_step_torch(cfg, lw, x, inp.k_slots[l], inp.v_slots[l], inp.mask)
        ks.append(k.astype(np.float16).astype(np.float32))
        vs.append(v)
    t = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
    h = torch.nn.functional.layer_norm(t(x), (cfg.hidden,), t(w.lnf_g), t(w.lnf_b), cfg.ln_eps)
    return h.numpy(), np.stack(ks), np.stack(vs)


def rows_err(chip, ref):
    """Largest max |chip - ref| / max |ref| over the layers' rows (the rounded PyTorch reference measure of a vector, per layer)."""
    chip, ref = np.asarray(chip, np.float64), np.asarray(ref, np.float64)
    return float(max(np.max(np.abs(c - r)) / (np.max(np.abs(r)) + 1e-12) for c, r in zip(chip, ref)))


def check_rounded(w, fed, hiddens, kv, say, step_ref=None, free_ref=None):
    """rounded PyTorch reference against PyTorch with f16-rounded weights and matmul inputs, per step on the K/V rows the chip wrote at the
    earlier positions (the same inputs as the chip, as `check_step.py`): the final hidden state (cosine >= 0.99999, max
    difference <= ERR_ROUNDED max |h|) and the step's new K and V rows of every layer (max difference <= ERR_KV_ROUNDED of the row's
    maximum). The free-running sequence (`forward_torch`, its own cache) drifts further: an f16 rounding that differs
    in the last f32 bit changes one matmul input by one f16 step and the difference then carries through the cache; it
    is gated at a bound calibrated on this checkpoint. `step_ref(inp)` -> (hidden, K rows, V rows) and
    `free_ref(w, fed)` -> hidden (T, H) default to PyTorch; tests inject their own."""
    if step_ref is None or free_ref is None:
        why = comparison_ref.torch_missing()
        if why:
            say(f"SKIP rounded PyTorch reference: {why}")
            return None
    if free_ref is None:
        from opentsp_vllm.models.tinystories.torch_ref import forward_torch
        free_ref = lambda w_, f: forward_torch(w_, f, True)[0]
    step_ref = step_ref or step_torch_rows
    cfg = w.cfg
    free = free_ref(w, fed)
    rows = []
    for pos, h in enumerate(hiddens):
        k = np.zeros((cfg.layers, cfg.slots, cfg.hidden), np.float16)
        v = np.zeros((cfg.layers, cfg.slots, cfg.hidden), np.float32)
        k[:, :pos], v[:, :pos] = kv.k[:, :pos], kv.v[:, :pos]
        mask = np.full(cfg.slots, -np.inf, np.float32)
        mask[:pos] = 0
        mask[cfg.slots - 1] = 0
        inp = types.SimpleNamespace(cfg=cfg, w=w, pos=pos, x0=exact_ref.embed(w, fed[pos], pos), k_slots=k, v_slots=v,
                                    mask=mask)
        ref_h, ref_k, ref_v = step_ref(inp)
        err, cos = comparison_ref.agreement(h, ref_h)
        kv_err = max(rows_err(kv.k[:, pos], ref_k), rows_err(kv.v[:, pos], ref_v))
        ferr, fcos = comparison_ref.agreement(h, free[pos])
        rows.append(dict(pos=pos, err=err, cos=cos, kv_err=kv_err, free_err=ferr, free_cos=fcos))
    worst = max(rows, key=lambda r: r["err"])
    fw = max(rows, key=lambda r: r["free_err"])
    kw = max(rows, key=lambda r: r["kv_err"])
    ok_step = all(r["err"] <= ERR_ROUNDED and r["cos"] >= comparison_ref.COSINE_MIN and r["kv_err"] <= ERR_KV_ROUNDED for r in rows)
    ok_free = all(r["free_err"] <= FREE_ERR_ROUNDED and r["free_cos"] >= comparison_ref.COSINE_MIN for r in rows)
    ok = ok_step and ok_free
    say(f"GATE rounded PyTorch reference {'PASS' if ok else 'FAIL'}: {len(rows)} steps against PyTorch with f16-rounded weights and matmul "
        f"inputs (per step, on the chip's K/V rows): hidden max |dh| / max |h| {worst['err']:.2e} at position "
        f"{worst['pos']} (<= {ERR_ROUNDED:g}), min cosine {min(r['cos'] for r in rows):.8f} (>= {comparison_ref.COSINE_MIN}); new K/V "
        f"rows max difference {kw['kv_err']:.2e} at position {kw['pos']} (<= {ERR_KV_ROUNDED:g}); "
        f"{sum(r['err'] < 1e-6 for r in rows)} steps agree to 1e-6")
    say(f"rounded PyTorch reference free-running PyTorch sequence (numeric closeness, gated at the calibrated {FREE_ERR_ROUNDED:g}; conformance is bit-exact reference): max |dh| / max |h| "
        f"{fw['free_err']:.2e} at position {fw['pos']}, min cosine {min(r['free_cos'] for r in rows):.8f}")
    return dict(ok=ok, ok_step=ok_step, ok_free=ok_free, rows=rows, max_err=worst["err"],
                min_cos=min(r["cos"] for r in rows), max_kv_err=kw["kv_err"], free_max_err=fw["free_err"],
                free_min_cos=min(r["free_cos"] for r in rows))


def check_fp32(w, a, ids, toks, hiddens, say, eos, reference=None):
    if reference is None:
        reference, why = fp32_reference(a.checkpoint)
        if reference is None:
            say(f"SKIP FP32 reference: {why}")
            return None
    steps = len(hiddens)
    fed = toks[:steps]
    h2, lg2 = reference.forward(fed)
    rows = []
    for pos in range(steps):
        rows.append(dict(pos=pos, cos_h=comparison_ref.cosine(hiddens[pos], h2[pos]),
                         cos_logits=comparison_ref.cosine(exact_ref.head_logits(w, hiddens[pos]), lg2[pos]),
                         err=comparison_ref.agreement(hiddens[pos], h2[pos])[0]))
    n_new = len(toks) - len(ids)
    flips, accepted = [], 0
    for i in range(min(n_new, TOKENS_FP32)):
        pos = len(ids) - 1 + i
        want, got, m = int(np.argmax(lg2[pos])), toks[pos + 1], margin(lg2[pos])
        rows[pos].update(token=got, fp32_token=want, margin=m)
        if want != got:
            ok_flip = m < MARGIN_FP32
            accepted += ok_flip
            flips.append(dict(index=i, pos=pos, chip=got, fp32=want, margin=m, accepted=ok_flip))
    free = reference.greedy(ids, n_new, eos)
    div = next((i for i, (x, y) in enumerate(zip(free[len(ids):], toks[len(ids):])) if x != y), None)
    if div is None and len(free) != len(toks):                  # one sequence ends first (the end token)
        div = min(len(free), len(toks)) - len(ids)
    cos_ok = all(r["cos_h"] >= COS_FP32 for r in rows)
    tok_ok = all(f["accepted"] for f in flips)
    ok = cos_ok and tok_ok
    say(f"GATE FP32 reference {'PASS' if ok else 'FAIL'}: {steps} steps against the fp32 model: hidden cosine min "
        f"{min(r['cos_h'] for r in rows):.6f} (>= {COS_FP32}), logits cosine min "
        f"{min(r['cos_logits'] for r in rows):.6f}; "
        f"the first {min(n_new, TOKENS_FP32)} greedy tokens on the same prefix: {len(flips)} mismatch(es)"
        f"{', all with margin < ' + format(MARGIN_FP32, 'g') if flips and tok_ok else ''}")
    for f in flips:
        say(f"  mismatch at generated token {f['index']} (position {f['pos']}): chip {f['chip']}, fp32 model "
            f"{f['fp32']}, fp32 top-1/top-2 margin {f['margin']:.3e} "
            f"({'accepted' if f['accepted'] else 'NOT accepted'})")
    say(f"FP32 reference free-running greedy ids: {' '.join(map(str, free[len(ids):]))}")
    say(f"chip ids:                  {' '.join(map(str, toks[len(ids):]))}")
    say("free-running sequences: " + ("identical" if div is None else
                                      f"first divergence at generated token {div}"))
    return dict(ok=ok, rows=rows, flips=flips, free_ids=free[len(ids):], first_divergence=div,
                min_cos_h=min(r["cos_h"] for r in rows))


def run(w, a, toks, ids, res, golden, summary, tok, eos=None, say=print):
    """Run each requested comparison and count failures, including unavailable references."""
    steps = len(res["rows"])
    hiddens = res["hiddens"]
    results = summary["gates"]
    if "rounded" in golden:
        result = check_rounded(w, toks[:steps], hiddens, res["kv"], say)
        results["rounded"] = result if result is not None else {
            "ok": False, "reason": "rounded PyTorch reference unavailable"
        }
    if "fp32" in golden:
        if a.weights == "real":
            result = check_fp32(w, a, ids, toks, hiddens, say, eos)
            results["fp32"] = result if result is not None else {
                "ok": False, "reason": "FP32 reference unavailable"
            }
        else:
            message = "FP32 reference requires --weights real"
            say(f"FAIL: {message}")
            results["fp32"] = {"ok": False, "reason": message}
    return sum(not results[tier]["ok"] for tier in {"rounded", "fp32"} & golden)
