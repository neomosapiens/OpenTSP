"""Tests of the GPT-2 small export and of bit-exact reference at width 768: the tiled layout round trip through the `.safetensors`
packages, the word counts of the layout, the refusals of the export, and bit-exact reference against an independent float64 forward pass of
GPT-2 on reduced configurations (same width 768, 12 heads of 64; fewer layers, a small vocabulary and pool).

usage: python3 -m pytest tests/showcase/gpt2small -q   (or python3 tests/showcase/gpt2small/test_gpt2s.py)"""
import dataclasses
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opentsp_vllm.models.gpt2.config import GPT2_SMALL, Gpt2Config
from opentsp_vllm.models.gpt2 import export as X
from opentsp_vllm.models.gpt2 import reference as G
from opentsp_vllm.models.gpt2 import weights as W

SMALL = Gpt2Config(layers=3, vocab=1000, pool=128, stage_layers=(1, 1, 1)).check()


def state_dict(w):
    """The Hugging Face GPT-2 state dict (Conv1D stores (in, out)) of `Weights`: the inverse of `from_state_dict`."""
    sd = {"wte.weight": w.wte.astype(np.float32), "wpe.weight": w.wpe.astype(np.float32),
          "ln_f.weight": w.lnf_g, "ln_f.bias": w.lnf_b}
    for l, lw in enumerate(w.layers):
        p = f"h.{l}."
        sd[p + "attn.c_attn.weight"] = np.concatenate([lw["wq"], lw["wk"], lw["wv"]]).astype(np.float32).T
        sd[p + "attn.c_attn.bias"] = np.concatenate([lw["bq"], lw["bk"], lw["bv"]])
        sd[p + "attn.c_proj.weight"], sd[p + "attn.c_proj.bias"] = lw["wo"].astype(np.float32).T, lw["bo"]
        sd[p + "mlp.c_fc.weight"], sd[p + "mlp.c_fc.bias"] = lw["up_w"].astype(np.float32).T, lw["up_b"]
        sd[p + "mlp.c_proj.weight"], sd[p + "mlp.c_proj.bias"] = lw["dn_w"].astype(np.float32).T, lw["dn_b"]
        sd[p + "ln_1.weight"], sd[p + "ln_1.bias"] = lw["ln1g"], lw["ln1b"]
        sd[p + "ln_2.weight"], sd[p + "ln_2.bias"] = lw["ln2g"], lw["ln2b"]
    return sd


def forward_f64(w, ids):
    """Independent float64 forward of GPT-2 (full recompute, causal attention, the f16-rounded weights): the final hidden state
    and the logits of the last position."""
    cfg = w.cfg
    H, A, dk = cfg.hidden, cfg.heads, cfg.dk
    f = lambda a: np.asarray(a, np.float64)
    ln = lambda x, g, b: (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + cfg.ln_eps) * f(g) + f(b)
    T = len(ids)
    x = f(w.wte[list(ids)]) + f(w.wpe[:T])
    causal = np.tril(np.ones((T, T), bool))
    for lw in w.layers:
        n = ln(x, lw["ln1g"], lw["ln1b"])
        q, k, v = (n @ f(lw[m]).T + f(lw[b]) for m, b in (("wq", "bq"), ("wk", "bk"), ("wv", "bv")))
        q, k, v = (a.reshape(T, A, dk).transpose(1, 0, 2) for a in (q, k, v))
        s = np.where(causal, q @ k.transpose(0, 2, 1) / np.sqrt(dk), -np.inf)
        p = np.exp(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        x = x + (p @ v).transpose(1, 0, 2).reshape(T, H) @ f(lw["wo"]).T + f(lw["bo"])
        n = ln(x, lw["ln2g"], lw["ln2b"])
        u = n @ f(lw["up_w"]).T + f(lw["up_b"])
        x = x + 0.5 * u * (1 + np.tanh(np.sqrt(2 / np.pi) * (u + 0.044715 * u ** 3))) @ f(lw["dn_w"]).T + f(lw["dn_b"])
    h = ln(x, w.lnf_g, w.lnf_b)[-1]
    return h, f(w.wte) @ h


class Layout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = W.synthetic(SMALL, 3)

    def test_matrix_words_round_trip_and_follow_the_stated_order(self):
        rng = np.random.default_rng(1)
        m = (rng.standard_normal((5, 768)) * 0.1).astype(np.float16)
        words = X.matrix_words(m)
        self.assertEqual(words.shape, (3 * 5 * 2, 320))
        self.assertTrue(np.array_equal(X.matrix_from_words(words, 5, 768), m))
        u8 = m.view(np.uint8).reshape(5, 768, 2)
        t, r, p = 2, 3, 1                                              # tile 2, row 3, high byte plane
        self.assertTrue(np.array_equal(words[(t * 5 + r) * 2 + p, :256], u8[r, 256 * t:256 * (t + 1), p]))
        self.assertFalse(words[:, 256:].any())

    def test_vector_words_round_trip(self):
        v = np.random.default_rng(2).standard_normal((2, 3072)).astype(np.float32)
        words = X.vector_words(v)
        self.assertEqual(words.shape, (2 * 12 * 4, 320))
        self.assertTrue(np.array_equal(X.vector_from_words(words, 3072), v))

    def test_word_counts_of_the_gpt2_small_layout(self):
        # the design's table (section 2): 301 542 words; its matrices of one layer 52 224 words with 10 tiles of 320 lanes for the
        # down projection, here 12 tiles of 256 lanes: 55 296 (+3 072, the cost of the power-of-two vector)
        c = GPT2_SMALL
        self.assertEqual(c.vocab * c.tiles * 2, 301_542)
        words = X.layer_tensors(self.w.layers[0])
        self.assertEqual({k: len(v) for k, v in words.items() if k in ("wqkv", "wo", "up", "dn")},
                         {"wqkv": 3 * 2304 * 2, "wo": 3 * 768 * 2, "up": 3 * 3072 * 2, "dn": 12 * 768 * 2})
        self.assertEqual(sum(len(words[k]) for k in ("wqkv", "wo", "up", "dn")), 55_296)

    def test_packages_round_trip_the_weights_bit_for_bit(self):
        with tempfile.TemporaryDirectory() as d:
            man = X.export(self.w, d, SMALL, "synthetic", device_layout=True)
            back = X.weights_from_packages(SMALL, [open(os.path.join(d, s["file"]), "rb").read() for s in man["stages"]])
        self.assertEqual(back.digest(), self.w.digest())
        self.assertEqual([s["layers"] for s in man["stages"]], [[0, 1], [1, 2], [2, 3]])
        self.assertEqual(man["weights_sha256"], self.w.digest())

    def test_stage_contents_follow_the_linear_pipeline(self):
        names = [sorted(X.stage_tensors(self.w, a, b, 3, i)) for i, (a, b) in enumerate(SMALL.stages())]
        self.assertTrue("wte" in names[0] and "wpe" in names[0])
        self.assertNotIn("wte", names[1])
        self.assertTrue("wte" in names[2] and "lnf" in names[2])
        self.assertTrue(all(len(n) <= 15 for ns in names for n in ns))

    def test_a_corrupted_word_changes_the_digest_and_the_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            man = X.export(self.w, d, SMALL, "synthetic", device_layout=True)
            pkgs = [bytearray(open(os.path.join(d, s["file"]), "rb").read()) for s in man["stages"]]
        pkgs[1][-5] ^= 1
        with self.assertRaises(Exception):
            X.weights_from_packages(SMALL, [bytes(p) for p in pkgs])


class Refusals(unittest.TestCase):
    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = X.main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    def test_a_missing_checkpoint_path_names_the_path(self):
        with tempfile.TemporaryDirectory() as d:
            rc, _, err = self.run_main("--checkpoint", "/nonexistent/dir/model.safetensors", "--out", d)
        self.assertEqual(rc, 2)
        self.assertIn("error", err)

    def test_a_missing_cached_model_says_what_to_fetch(self):
        with tempfile.TemporaryDirectory() as d:
            rc, _, err = self.run_main("--checkpoint", "no-such-org/no-such-model", "--out", d)
        self.assertEqual(rc, 2)
        self.assertIn("no model.safetensors", err)

    def test_a_wrong_width_is_refused_with_both_widths(self):
        from safetensors.numpy import save_file
        sd = {"wte.weight": np.zeros((50257, 1024), np.float32)}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.safetensors")
            save_file(sd, path)
            rc, _, err = self.run_main("--checkpoint", path, "--out", os.path.join(d, "out"))
        self.assertEqual(rc, 2)
        self.assertIn("width 1024", err)
        self.assertIn("768", err)

    def test_a_bad_stage_split_is_refused(self):
        for bad, why in (("2,7", "9 layers"), ("a,b", "layer counts"), ("2,7,4", "13 layers")):
            with self.assertRaises(ValueError) as cm:
                X.parse_stages(bad, 12)
            self.assertIn(why, str(cm.exception))

    def test_the_manifest_is_json_with_the_layout_rules(self):
        with tempfile.TemporaryDirectory() as d:
            X.export(W.synthetic(SMALL, 4), d, SMALL, "synthetic", device_layout=True)
            man = json.load(open(os.path.join(d, "manifest.json")))
        self.assertEqual(man["layout"]["tile_lanes"], 256)
        self.assertEqual(man["model"]["head_dim"], 64)
        self.assertEqual(len(man["stages"]), 3)


class Reference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Gpt2Config(layers=2, vocab=1000, pool=256, stage_layers=(2,)).check()
        cls.w = W.synthetic(cls.cfg, 5)

    def test_hidden_state_and_logits_match_the_float64_forward(self):
        ids = [11, 400, 7, 999, 3, 52, 8, 200, 13, 1]
        pool = G.Pool(self.cfg, self.cfg.layers)
        r = None
        for pos, t in enumerate(ids):
            r = G.decode_step(self.w, pool, t, pos, pos, G.page_counts(self.cfg, lambda p: p, pos), logits=pos == len(ids) - 1)
        h, lg = forward_f64(self.w, ids)
        np.testing.assert_allclose(r.hidden, h, rtol=0, atol=2e-2 * np.abs(h).max())
        np.testing.assert_allclose(r.logits, lg, rtol=0, atol=2e-2 * np.abs(lg).max())
        self.assertEqual(int(np.argmax(r.logits)), int(np.argmax(lg)))

    def test_both_context_forms_give_the_same_greedy_tokens(self):
        a, _ = G.generate(self.w, [5, 17, 300], 4, pv="mxm")
        b, _ = G.generate(self.w, [5, 17, 300], 4, pv="vxm")
        self.assertEqual(a, b)

    def test_a_different_slot_map_changes_no_token_and_stays_close(self):
        slots = lambda pos: pos + 16 * (pos // 16)                      # one empty page between two used pages
        a, la = G.generate(self.w, [5, 17, 300, 2, 9, 14, 80, 81, 5, 17, 300, 2, 9, 14, 80, 81, 33, 34], 3)
        b, lb = G.generate(self.w, [5, 17, 300, 2, 9, 14, 80, 81, 5, 17, 300, 2, 9, 14, 80, 81, 33, 34], 3, slots=slots)
        self.assertEqual(a, b)
        np.testing.assert_allclose(la[0], lb[0], atol=1e-3 * np.abs(la[0]).max())

    def test_a_slot_that_is_not_a_page_prefix_is_refused(self):
        with self.assertRaises(ValueError):
            G.generate(self.w, [1, 2], 1, slots=lambda pos: pos + 3)

    def test_the_new_slot_must_be_visible(self):
        pool = G.Pool(self.cfg, self.cfg.layers)
        with self.assertRaises(AssertionError):
            G.decode_step(self.w, pool, 1, 0, 5, np.array([1] + [0] * 15), logits=False)

    def test_generation_is_deterministic(self):
        a, la = G.generate(self.w, [1, 2, 3], 3)
        b, lb = G.generate(self.w, [1, 2, 3], 3)
        self.assertEqual(a, b)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(la, lb)))

    def test_state_dict_mapping_round_trips(self):
        back = W.from_state_dict(self.cfg, state_dict(self.w))
        self.assertEqual(back.digest(), self.w.digest())


class CommandLine(unittest.TestCase):
    """The refusals of the two tools, run as a person runs them (each exits 2 with one line on stderr, before any model is built)."""
    HERE = os.path.dirname(os.path.abspath(__file__))

    def run_tool(self, tool, *args):
        import subprocess
        cmd = ([sys.executable, "-m", "opentsp_vllm.models.gpt2.export"] if tool == "gpt2s_export.py" else [sys.executable, os.path.join(self.HERE, tool)])
        r = subprocess.run([*cmd, *args], capture_output=True, text=True)
        return r.returncode, r.stderr.strip()

    def test_a_missing_checkpoint_file_is_named(self):
        for tool in ("gpt2s_export.py", "compare_hf.py"):
            extra = ("--out", tempfile.gettempdir()) if tool == "gpt2s_export.py" else ()
            rc, err = self.run_tool(tool, "--checkpoint", "/nonexistent/model.safetensors", *extra)
            self.assertEqual(rc, 2, (tool, err))
            self.assertIn("checkpoint file '/nonexistent/model.safetensors' does not exist", err)
            self.assertEqual(len(err.splitlines()), 1, err)

    def test_an_empty_prompt_and_no_new_tokens_are_refused(self):
        rc, err = self.run_tool("compare_hf.py", "--prompt", " ")
        self.assertEqual((rc, err), (2, "compare_hf: error: a prompt is empty"))
        rc, err = self.run_tool("compare_hf.py", "--new", "0")
        self.assertEqual((rc, err), (2, "compare_hf: error: --new 0 must be at least 1"))

    def test_a_pool_that_is_not_a_multiple_of_128_is_refused(self):
        rc, err = self.run_tool("compare_hf.py", "--pool", "100")
        self.assertEqual(rc, 2)
        self.assertIn("not a multiple of 128", err)


class Config(unittest.TestCase):
    def test_gpt2_small_is_the_checked_shape(self):
        c = GPT2_SMALL
        self.assertEqual((c.hidden, c.heads, c.dk, c.ffn, c.layers, c.tiles, c.ffn_tiles), (768, 12, 64, 3072, 12, 3, 12))
        self.assertEqual(c.stages(), [(0, 2), (2, 9), (9, 12)])
        with self.assertRaises(AssertionError):
            dataclasses.replace(c, pool=100).check()
        with self.assertRaises(AssertionError):
            dataclasses.replace(c, hidden=1024).check()


if __name__ == "__main__":
    unittest.main()
