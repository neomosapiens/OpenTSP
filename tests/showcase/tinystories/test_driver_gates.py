"""Unit tests of the rounded PyTorch reference and FP32 reference decisions of `driver_gates` on small synthetic inputs (no checkpoint, no PyTorch): the
references are injected, so what is tested is the verdict logic: which perturbation fails which tier, the margin clause,
the combination of the cosine and token criteria, and the first-divergence report."""
import types

import numpy as np
import pytest


from opentsp_vllm.models.tinystories import driver_gates as dg
from opentsp_vllm.models.tinystories import reference as exact_ref
from opentsp_vllm.models.tinystories import weights as W
from opentsp_vllm.models.tinystories.config import SMALL

T = 4
FED = [3, 5, 7, 9]


@pytest.fixture(scope="module")
def w():
    return W.synthetic(SMALL, 1)


def rng(seed):
    return np.random.default_rng(seed)


def chip(w):
    """Rows and hidden states as a chip run of T steps would hand them over (random but fixed)."""
    r = rng(1)
    kv = exact_ref.HostKV(SMALL)
    for _ in range(T):
        kv.append(r.standard_normal((SMALL.layers, SMALL.hidden)).astype(np.float16),
                  r.standard_normal((SMALL.layers, SMALL.hidden)).astype(np.float32))
    return kv, r.standard_normal((T, SMALL.hidden)).astype(np.float32)


def run_rounded_comparison(w, kv, hiddens, ref_h=None, ref_free=None, ref_kv=None):
    ref_h = hiddens if ref_h is None else ref_h
    ref_free = hiddens if ref_free is None else ref_free
    ref_kv = kv if ref_kv is None else ref_kv
    lines = []
    step = lambda inp: (ref_h[inp.pos], ref_kv.k[:, inp.pos].astype(np.float32), ref_kv.v[:, inp.pos])
    r = dg.check_rounded(w, FED, hiddens, kv, lines.append, step_ref=step, free_ref=lambda w_, fed: ref_free)
    return r, lines


def test_rounded_reference_passes_when_the_reference_agrees(w):
    kv, h = chip(w)
    r, lines = run_rounded_comparison(w, kv, h)
    assert r["ok"] and any(ln.startswith("GATE rounded PyTorch reference PASS") for ln in lines)
    assert r["max_err"] == 0 and r["min_cos"] > 0.99999


def test_rounded_reference_fails_on_a_perturbed_hidden_state(w):
    """One element moved by 1e-2 of max |h| is above the calibrated per-step bound and fails."""
    kv, h = chip(w)
    bad = h.copy()
    bad[2, 5] += 1e-2 * np.abs(h[2]).max()
    r, lines = run_rounded_comparison(w, kv, bad, ref_h=h, ref_free=h)
    assert not r["ok"] and r["rows"][2]["err"] > dg.ERR_ROUNDED and not r["ok_step"]
    assert any(ln.startswith("GATE rounded PyTorch reference FAIL") for ln in lines)


def test_rounded_reference_fails_on_a_wrong_direction(w):
    kv, h = chip(w)
    r, _ = run_rounded_comparison(w, kv, h, ref_h=np.roll(h, 1, axis=1), ref_free=h)
    assert not r["ok"] and min(x["cos"] for x in r["rows"]) < 0.99999


def test_rounded_reference_fails_on_a_perturbed_kv_row(w):
    """The chip's K and V rows are compared with the reference's too: a wrong V row fails with a correct hidden
    state."""
    kv, h = chip(w)
    ref = exact_ref.HostKV(SMALL)
    ref.k, ref.v = kv.k.copy(), kv.v.copy()
    ref.v[1, 1, 4] += 0.1 * np.abs(ref.v[1, 1]).max()
    r, lines = run_rounded_comparison(w, kv, h, ref_kv=ref)
    assert not r["ok"] and r["rows"][1]["kv_err"] > dg.ERR_ROUNDED
    assert all(x["kv_err"] == 0 for i, x in enumerate(r["rows"]) if i != 1)


# the largest values `calibrate_comparison.py` measured on 24 independent prompts (hidden, K/V rows; free-running at 58 steps)
MEASURED = dict(hidden=1.04e-3, kv=1.66e-3, free=2.16e-3)


def rounded_result_with_errors(w, hidden, kv_rows, free):
    kv, h = chip(w)
    rowmax = np.abs(h).max(axis=1)
    ref_h = h.copy()
    ref_h[2, 5] += hidden * rowmax[2]
    ref = exact_ref.HostKV(SMALL)
    ref.k, ref.v = kv.k.copy(), kv.v.copy()
    ref.v[1, 1, 4] += kv_rows * np.abs(ref.v[1, 1]).max()
    ref_free = h.copy()
    ref_free[1, 0] += free * rowmax[1]
    return run_rounded_comparison(w, kv, h, ref_h=ref_h, ref_free=ref_free, ref_kv=ref)[0]


def test_rounded_reference_accepts_the_largest_errors_measured_on_independent_prompts(w):
    r = rounded_result_with_errors(w, MEASURED["hidden"], MEASURED["kv"], MEASURED["free"])
    assert r["ok"], (r["max_err"], r["max_kv_err"], r["free_max_err"])
    assert r["max_err"] > 0.9 * MEASURED["hidden"] and r["max_kv_err"] > 0.9 * MEASURED["kv"]


def test_rounded_reference_bounds_keep_headroom_over_the_measured_maxima_and_still_fail_a_larger_error(w):
    for key, bound in (("hidden", dg.ERR_ROUNDED), ("kv", dg.ERR_KV_ROUNDED), ("free", dg.FREE_ERR_ROUNDED)):
        assert 1.4 * MEASURED[key] <= bound <= 2.0 * MEASURED[key], key
    assert not rounded_result_with_errors(w, 1.5 * dg.ERR_ROUNDED, 0, 0)["ok_step"]
    assert not rounded_result_with_errors(w, 0, 1.5 * dg.ERR_KV_ROUNDED, 0)["ok_step"]
    assert not rounded_result_with_errors(w, 0, 0, 1.5 * dg.FREE_ERR_ROUNDED)["ok_free"]


def test_rounded_reference_free_running_drift_has_a_calibrated_bound(w):
    kv, h = chip(w)
    rowmax = np.abs(h).max(axis=1)
    ok = h.copy()
    ok[:, 0] += 2.4e-3 * rowmax                       # inside the calibrated bound, outside the per-step one
    r, _ = run_rounded_comparison(w, kv, h, ref_free=ok)
    assert r["ok"] and dg.ERR_ROUNDED < r["free_max_err"] < dg.FREE_ERR_ROUNDED
    bad = h.copy()
    bad[1, 0] += 5e-3 * rowmax[1]
    r, lines = run_rounded_comparison(w, kv, h, ref_free=bad)
    assert not r["ok"] and r["free_max_err"] > dg.FREE_ERR_ROUNDED
    assert any("free-running" in ln for ln in lines)


# -- FP32 reference ----------------------------------------------------------------------------------------------------------
IDS = [3, 5]                 # the prompt; the chip then generated two tokens: toks = IDS + NEW
NEW = [11, 13]


def reference(w, toks, h2, lg2, free):
    """The injected fp32 model: its hidden states and logits per position, and its free-running greedy ids."""
    return types.SimpleNamespace(forward=lambda ids: (h2[:len(ids)], lg2[:len(ids)]),
                                 greedy=lambda ids, n_new, eos=None: list(ids) + list(free))


def scenario(w, margin=None, flip=False, cos_break=False, free=None):
    """toks = prompt + 2 generated; hiddens of 3 steps (positions 0..2); the fp32 logits at positions 1 and 2 choose the
    chip's tokens 11 and 13 unless `flip`, which makes position 1 prefer token 20 by `margin`."""
    toks = IDS + NEW
    hid = rng(2).standard_normal((3, SMALL.hidden)).astype(np.float32)
    lg = np.zeros((3, SMALL.vocab), np.float32)
    lg[0, 1], lg[1, 11], lg[2, 13] = 0.5, 1.0, 1.0
    if flip:
        lg[1, 11], lg[1, 20] = 1.0, 1.0 + margin
    h2 = np.roll(hid, 7, axis=1) if cos_break else hid.copy()
    lines = []
    ref = reference(w, toks, h2, lg, NEW if free is None else free)
    a = types.SimpleNamespace(checkpoint="unused")
    return dg.check_fp32(w, a, IDS, toks, hid, lines.append, None, reference=ref), lines


def test_fp32_reference_passes_when_the_model_agrees(w):
    r, lines = scenario(w)
    assert r["ok"] and r["flips"] == [] and r["first_divergence"] is None
    assert any(ln.startswith("GATE FP32 reference PASS") for ln in lines)


def test_fp32_reference_fails_on_a_token_flip_with_a_large_margin(w):
    r, lines = scenario(w, margin=0.5, flip=True)
    assert not r["ok"]
    (f,) = r["flips"]
    assert f["chip"] == 11 and f["fp32"] == 20 and f["margin"] == pytest.approx(0.5) and not f["accepted"]
    assert any("NOT accepted" in ln for ln in lines) and any(ln.startswith("GATE FP32 reference FAIL") for ln in lines)


def test_fp32_reference_accepts_a_token_flip_with_a_tiny_margin(w):
    r, lines = scenario(w, margin=5e-4, flip=True)
    assert r["ok"]
    (f,) = r["flips"]
    assert f["accepted"] and f["margin"] < dg.MARGIN_FP32
    assert any("accepted" in ln and "NOT" not in ln for ln in lines)


def test_fp32_reference_margin_clause_boundary(w):
    r, _ = scenario(w, margin=2e-3, flip=True)
    assert not r["ok"]


def test_fp32_reference_fails_on_the_hidden_cosine_even_when_every_token_agrees(w):
    r, _ = scenario(w, cos_break=True)
    assert not r["ok"] and r["flips"] == [] and r["min_cos_h"] < dg.COS_FP32


def test_fp32_reference_fails_when_both_criteria_fail_and_reports_the_first_divergence(w):
    r, lines = scenario(w, margin=0.5, flip=True, cos_break=True, free=[11, 99])
    assert not r["ok"] and r["first_divergence"] == 1
    assert any("first divergence at generated token 1" in ln for ln in lines)


def test_fp32_reference_free_running_divergence_alone_does_not_fail_the_gate(w):
    """Free-running sequences are reported; the gate is the teacher-forced agreement and the cosine."""
    r, lines = scenario(w, free=[11, 77])
    assert r["ok"] and r["first_divergence"] == 1
    assert any("first divergence at generated token 1" in ln for ln in lines)


def test_fp32_reference_skips_visibly_without_a_reference(w, monkeypatch):
    monkeypatch.setattr(dg, "fp32_reference", lambda name: (None, "transformers is not installed"))
    lines = []
    r = dg.check_fp32(w, types.SimpleNamespace(checkpoint="x"), IDS, IDS + NEW, np.zeros((3, SMALL.hidden), np.float32),
              lines.append, None)
    assert r is None and lines == ["SKIP FP32 reference: transformers is not installed"]
