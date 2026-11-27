"""Tests of the bit-exact reference model (tier-1 golden) of the TinyStories network.

bit-exact reference's matvec equals the oracle's dot_f16 on random rows; the forward with synthetic weights is deterministic and
within 1e-2 (relative) of a float64 forward; the host-managed cache with a fixed current slot; the attention slot
layout; a pinned digest of the synthetic weights and of bit-exact reference's output. The lane arithmetic bit-exact reference is built from is checked
against the oracle in tests/programs/gen/test_numerics.py.
"""
import hashlib

import numpy as np
import pytest


from opentsp_reference import numerics as on
from opentsp_tools.programs import ref
from opentsp_vllm.models.tinystories import reference as exact_ref
from opentsp_vllm.models.tinystories import float_ref
from opentsp_vllm.models.tinystories import weights as W
from opentsp_vllm.models.tinystories.config import SMALL, TINYSTORIES, PROMPT_IDS


def bits32(a):
    return np.asarray(a, dtype=np.float32).view(np.uint32)


def test_matvec_equals_oracle_dot_f16_on_random_rows():
    rng = np.random.default_rng(11)
    Wm = (rng.standard_normal((64, 64)) * 0.1).astype(np.float16)
    x = (rng.standard_normal(64) * 2).astype(np.float32)
    b = (rng.standard_normal(64) * 0.1).astype(np.float32)
    got = ref.matvec_f16(Wm, x, b)
    x16 = x.astype(np.float16)
    for r in range(64):
        dot = on.dot_f16([int(v) for v in Wm[r].view(np.uint16)] + [0] * 256, [int(v) for v in x16.view(np.uint16)] + [0] * 256)
        want = on.k_binary(0, 'f32', dot, int(b[r:r + 1].view(np.uint32)[0]))
        assert int(bits32(got[r:r + 1])[0]) == want, r


def test_synthetic_weights_deterministic_and_pinned():
    a = W.synthetic(SMALL, seed=1)
    b = W.synthetic(SMALL, seed=1)
    c = W.synthetic(SMALL, seed=2)
    assert a.digest() == b.digest() != c.digest()
    assert a.digest() == "f33013afa5237d061b14a4e032bb14e9baeb64caa3040c06855d5952d340c5cd"


def test_gpt2_state_dict_mapping_transposes_and_splits():
    cfg = SMALL
    H, F = cfg.hidden, cfg.ffn
    rng = np.random.default_rng(3)
    sd = {"transformer.wte.weight": rng.standard_normal((cfg.vocab, H)), "transformer.wpe.weight": rng.standard_normal((cfg.max_pos, H)),
          "transformer.ln_f.weight": rng.standard_normal(H), "transformer.ln_f.bias": rng.standard_normal(H)}
    for l in range(cfg.layers):
        p = f"transformer.h.{l}."
        sd[p + "attn.c_attn.weight"] = rng.standard_normal((H, 3 * H))
        sd[p + "attn.c_attn.bias"] = rng.standard_normal(3 * H)
        sd[p + "attn.c_proj.weight"] = rng.standard_normal((H, H))
        sd[p + "attn.c_proj.bias"] = rng.standard_normal(H)
        sd[p + "mlp.c_fc.weight"] = rng.standard_normal((H, F))
        sd[p + "mlp.c_fc.bias"] = rng.standard_normal(F)
        sd[p + "mlp.c_proj.weight"] = rng.standard_normal((F, H))
        sd[p + "mlp.c_proj.bias"] = rng.standard_normal(H)
        for n in ("ln_1", "ln_2"):
            sd[p + n + ".weight"] = rng.standard_normal(H)
            sd[p + n + ".bias"] = rng.standard_normal(H)
    w = W.from_gpt2_state_dict(cfg, sd)
    qkv = sd["transformer.h.1.attn.c_attn.weight"]
    assert np.array_equal(w.layers[1]["wk"], qkv[:, H:2 * H].T.astype(np.float16))     # column block k of the Conv1D
    assert np.array_equal(w.layers[1]["wv"], qkv[:, 2 * H:].T.astype(np.float16))
    assert np.array_equal(w.layers[0]["dn_w"], sd["transformer.h.0.mlp.c_proj.weight"].T.astype(np.float16))
    assert w.layers[0]["dn_w"].shape == (H, F) and w.layers[0]["up_w"].shape == (F, H)
    assert w.layers[0]["bv"].dtype == np.float32 and w.wte.dtype == np.float16


def test_decode_step_deterministic_on_the_real_size_network():
    w = W.synthetic(TINYSTORIES, seed=1)
    runs = []
    for _ in range(2):
        kv = exact_ref.HostKV(TINYSTORIES)
        runs.append([exact_ref.decode_step(w, kv, t, p).hidden for p, t in enumerate(PROMPT_IDS[:3])])
    for a, b in zip(*runs):
        assert np.array_equal(bits32(a), bits32(b))


def test_forward_within_1e2_of_float64_forward():
    w = W.synthetic(TINYSTORIES, seed=1)
    kv = exact_ref.HostKV(TINYSTORIES)
    ref = float_ref.forward(w, PROMPT_IDS)
    for pos, t in enumerate(PROMPT_IDS):
        r = exact_ref.decode_step(w, kv, t, pos)
        h = r.hidden.astype(np.float64)
        assert np.max(np.abs(h - ref.hidden[pos])) <= 1e-2 * np.max(np.abs(ref.hidden[pos])), pos
        cos = h @ ref.hidden[pos] / (np.linalg.norm(h) * np.linalg.norm(ref.hidden[pos]))
        assert cos > 0.99999, (pos, cos)
        assert np.max(np.abs(r.logits.astype(np.float64) - ref.logits[pos])) <= 1e-2 * np.max(np.abs(ref.logits[pos]))


def test_fixed_current_slot_cache_layout():
    w = W.synthetic(SMALL, seed=7)
    kv = exact_ref.HostKV(SMALL)
    r0 = exact_ref.decode_step(w, kv, 5, 0)
    r1 = exact_ref.decode_step(w, kv, 17, 1)
    k, v, mask = kv.slots(2)                 # what the host preloads for position 2
    S = SMALL.slots
    assert k.shape == (SMALL.layers, S, SMALL.hidden) and k.dtype == np.float16 and v.dtype == np.float32
    assert np.array_equal(k[:, 0], r0.k_cur) and np.array_equal(k[:, 1], r1.k_cur)
    assert np.array_equal(v[:, 0], r0.v_cur) and np.array_equal(v[:, 1], r1.v_cur)
    assert not k[:, 2:].any() and not v[:, 2:].any()          # the current slot and the empty slots are zero
    assert mask.dtype == np.float32 and mask.shape == (S,)
    assert np.all(mask[:2] == 0) and mask[S - 1] == 0 and np.all(mask[2:S - 1] == -np.inf)


def test_empty_slots_are_masked():
    # a step must not depend on the content of the empty slots (the mask gives them probability exactly 0)
    w = W.synthetic(SMALL, seed=7)
    kv = exact_ref.HostKV(SMALL)
    exact_ref.decode_step(w, kv, 5, 0)
    k, v, mask = kv.slots(1)
    r = exact_ref.decode_step_slots(w, 17, 1, k, v, mask)
    k2, v2 = k.copy(), v.copy()
    k2[:, 1:SMALL.slots - 1] = np.float16(3.0)
    v2[:, 1:SMALL.slots - 1] = np.float32(-7.0)
    r2 = exact_ref.decode_step_slots(w, 17, 1, k2, v2, mask)
    assert np.array_equal(bits32(r.hidden), bits32(r2.hidden))
    # and it does depend on a cached slot
    k3 = k.copy()
    k3[:, 0] = np.float16(2.0)
    r3 = exact_ref.decode_step_slots(w, 17, 1, k3, v, mask)
    assert not np.array_equal(bits32(r.hidden), bits32(r3.hidden))


def test_context_limit():
    w = W.synthetic(SMALL, seed=7)
    kv = exact_ref.HostKV(SMALL)
    for pos in range(SMALL.slots):
        exact_ref.decode_step(w, kv, pos % 7, pos)
    with pytest.raises(ValueError):
        exact_ref.decode_step(w, kv, 1, SMALL.slots)


def test_greedy_generation_is_reproducible_and_pinned():
    w = W.synthetic(TINYSTORIES, seed=1)
    a = exact_ref.generate(w, PROMPT_IDS, 6)
    b = exact_ref.generate(w, PROMPT_IDS, 6)
    assert a.tokens == b.tokens and len(a.tokens) == len(PROMPT_IDS) + 6
    h = hashlib.sha256()
    for s in a.steps:
        h.update(s.hidden.tobytes())
    assert [int(t) for t in a.tokens] == PINNED_TOKENS
    assert h.hexdigest() == PINNED_HIDDEN_SHA


PINNED_TOKENS = [7454, 2402, 257, 640, 612, 373, 26434, 26434, 26434, 26434, 26434, 26434]
PINNED_HIDDEN_SHA = "156187d93e268b23d90b906b06bd17abc0de962980bc6bda2536823ca6de97a4"
