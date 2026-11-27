"""Tests of `ship_system.py`: the helpers, and a 2-chip chain of small layers (hidden 256) through the binding's `System` that must equal the twin, with
mutants that must not (a flipped weight bit, a wrong host vector, a missing weight). The chain needs the binding (`just vllm-binding`) and `ots-asm`."""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

from opentsp_vllm.models.gpt2 import paths as _gpaths

sys.path.insert(0, os.path.join(_gpaths.ROOT, "opentsp-vllm", "src"))
from opentsp_vllm.models.gpt2 import system as S
from opentsp_vllm.models.gpt2 import layer as sl
from opentsp_tools import profile as tprofile

import importlib.util                      # noqa: E402
HAVE_BINDING = importlib.util.find_spec("opentsp_vllm._tsp_iss") is not None
ASM = os.path.join(_gpaths.ROOT, "target", "release", "ots-asm")


def flip_words(system, chip, text, lane, bit):
    """Flip one bit of `lane` in the first 64 words of the largest SRAM run of the stage (a weight plane's slice: every word of it is used)."""
    sli, addr, data = max(S.split_input(text)[0], key=lambda r: len(r[2]))
    n = min(64, len(data) // 320)
    w = bytearray(system.read_words(chip, sli, addr, n))
    for i in range(n):
        w[320 * i + lane] ^= bit
    system.write_words(chip, sli, addr, bytes(w))


class Helpers(unittest.TestCase):
    def test_counts(self):
        c = S.counts_for(37, 2)
        self.assertEqual(c.shape, (2, 8))
        self.assertEqual(c.sum(), 37)
        self.assertEqual(list(c[0, :3]), [16, 16, 5])

    def test_split_input(self):
        w = bytes(range(256)) + bytes(64)
        text = "Q 3c 00000001\nS 2c 10 " + " ".join(f"{b:x}" for b in w) + "\nH dma 1f " + " ".join(f"{b:x}" for b in w[:320]) + "\n"
        sram, dma = S.split_input(text)
        self.assertEqual(sram, [(0x2c, 0x10, w)])
        self.assertEqual(dma, [(0x1f, w[:320])])

    def test_layer_params_names(self):
        from opentsp_vllm.models.gpt2 import weights as W
        from opentsp_vllm.models.gpt2.config import GPT2_SMALL
        lw = W.synthetic(GPT2_SMALL, 1).layers[0]
        P = S.layer_params(lw)
        self.assertEqual(set(P), set(sl.params(np.random.default_rng(1), 2)))
        self.assertEqual(P["wqkv"].shape, (3 * 768, 768))


@unittest.skipUnless(HAVE_BINDING and os.access(ASM, os.X_OK), "needs the binding (just vllm-binding) and ots-asm")
class SmallChain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="ship-sys-test-")
        cls.p = tprofile.load("ref")
        cls.stages, cls.infos = S.build_stages(cls.p, 2, None, None, None, 37, cls.work, tiles=2, chunks=2)
        cls.budget = max(st["case"]["budget"]["cycles"] for st in cls.stages)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def run_chain(self, mutate=None):
        s, cycles, _, _ = S.run_system("ref", self.stages, self.budget, mutate=mutate)
        return s, S.check(s, self.stages)

    def test_equals_twin(self):
        s, n = self.run_chain()
        self.assertEqual(n, 8)

    def test_flipped_weight_bit_is_caught(self):
        def flip(s, route):
            flip_words(s, route[0], self.stages[0]["text"], 3, 0x40)
        with self.assertRaises(SystemExit) as e:
            self.run_chain(flip)
        self.assertIn("FAIL", str(e.exception))

    def test_flipped_bit_in_the_second_chip_is_caught(self):
        def flip(s, route):
            flip_words(s, route[1], self.stages[1]["text"], 7, 0x40)
        with self.assertRaises(SystemExit):
            self.run_chain(flip)

    def test_wrong_host_vector_is_caught(self):
        text = self.stages[0]["text"]
        k = text.index("H dma ")
        f = text[k:].split("\n")[0].split()
        f[3] = format(int(f[3], 16) ^ 0x40, "x")           # the first byte of the first vector the host pushes
        bad = text[:k] + " ".join(f) + text[k + len(" ".join(text[k:].split("\n")[0].split())):]
        stages = [dict(self.stages[0], text=bad)] + self.stages[1:]
        s, _, _, _ = S.run_system("ref", stages, self.budget)
        with self.assertRaises(SystemExit):
            S.check(s, self.stages)


@unittest.skipUnless(HAVE_BINDING and os.access(ASM, os.X_OK), "needs the binding (just vllm-binding) and ots-asm")
class FanToHeads(unittest.TestCase):
    """Two layer chips whose last chip Sends its output to two head chips over the links (`chain(fan=...)`, `ship_head.head(io=...)`): every head chip's logits equal the
    twin of the chain's output, bit for bit (hidden 256, 2 logits tiles per head chip)."""

    @classmethod
    def setUpClass(cls):
        from opentsp_vllm.models.gpt2 import head as H
        cls.work = tempfile.mkdtemp(prefix="ship-fan-test-")
        p = cls.p = tprofile.load("ref")
        cls.stages, cls.infos = S.build_stages(p, 2, None, None, None, 37, cls.work, tiles=2, chunks=2, route=[0, 1], chips=4, fan=[2, 3])
        cls.heads = []
        runner = S.R.CaseBuilder(S.argparse_ns())
        rng = np.random.default_rng(5)
        for h, chip in enumerate((2, 3)):
            recv = [c + sl.LINK_RX for c in cls.infos[-1]["sends_all"][h]]
            W = (rng.standard_normal((256, 256)) * 0.05).astype(np.float16)
            text, case, _ = H.head(p, T=2, vtiles=2, name=f"fan{h}", x=cls.infos[-1]["out"], W=W, io=dict(src=1, link=sl.hop(4, 16, 1, chip)[1], recv=recv))
            cdir = os.path.join(cls.work, f"h{h}")
            os.makedirs(cdir)
            open(os.path.join(cdir, case["program"]), "w").write(text)
            txt, (oa, _pkg) = runner.build_input(case, cdir, cdir)
            cls.heads.append(dict(case=case, elf=oa, text=txt))
        cls.budget = max(st["case"]["budget"]["cycles"] for st in cls.stages + cls.heads)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def run_all(self, mutate=None):
        stages = self.stages + self.heads
        s, cycles, _, _ = S.run_system("ref", stages, self.budget, chips=4, route=[0, 1, 2, 3], mutate=mutate)
        return s, [S.check(s, [hs], [c]) for hs, c in zip(self.heads, (2, 3))]

    def test_the_last_layer_chip_has_no_host_output(self):
        self.assertNotIn("egress", self.stages[-1]["case"]["expect"])
        self.assertEqual(len(self.infos[-1]["sends_all"]), 2)
        self.assertLess(self.infos[-1]["sends_all"][0][-1], self.infos[-1]["sends_all"][1][0])      # the destinations are served one after the other

    def test_head_chips_equal_the_twin(self):
        s, n = self.run_all()
        self.assertEqual(n, [8, 8])

    def test_a_flipped_head_weight_is_caught(self):
        def flip(s, route):
            flip_words(s, 3, self.heads[1]["text"], 5, 0x40)
        with self.assertRaises(SystemExit):
            self.run_all(flip)


if __name__ == "__main__":
    unittest.main()
