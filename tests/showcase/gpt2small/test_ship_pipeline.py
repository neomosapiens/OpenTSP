"""Tests of `ship_pipeline.py` and the host-side words it writes: the step's maps and mask encoding equal what the generator bakes (`ship_layer.layer`), the
route follows the link wiring, the pool extern names are the K and V pools. The steps on a System are in `test_ship_system.py` (a chain of small layers)
and, on the real checkpoint, `ship_pipeline.py --layers N`."""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

from opentsp_vllm.models.gpt2 import paths as _gpaths

sys.path.insert(0, os.path.join(_gpaths.ROOT, "opentsp-vllm", "src"))
from opentsp_vllm.models.gpt2 import pipeline as P
from opentsp_vllm.models.gpt2 import layer as sl
from opentsp_vllm.models.gpt2 import attention as sa
from opentsp_tools import profile as tprofile


class HostWords(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = tprofile.load("ref")
        cls.T, cls.chunks, cls.slot, cls.seed = 2, 2, 37, 5
        cls.counts = P.S.counts_for(cls.slot + 1, cls.chunks)
        zero = np.zeros((sl.CHUNK * cls.chunks, cls.T, sl.TILE), np.float16)
        cls.text, cls.case = sl.layer(cls.p, seed=cls.seed, T=cls.T, chunks=cls.chunks, slot=cls.slot, pools=(zero, zero, cls.counts))
        cls.weights = cls.case["weights"]

    def baked(self, name):
        w = self.weights[name]
        return bytes.fromhex(w["hex"])

    def test_maps_equal_the_generators(self):
        mw = sl.map_weights(self.p, self.T, self.chunks, self.slot, self.seed)
        self.assertEqual(len(mw), 8 * self.T + 1)
        for name, blob in mw.items():
            self.assertEqual(blob, self.baked(name), name)

    def test_maps_follow_the_slot(self):
        a = sl.map_weights(self.p, self.T, self.chunks, 37, 1)
        b = sl.map_weights(self.p, self.T, self.chunks, 38, 1)
        self.assertNotEqual(a["mv"], b["mv"])

    def test_encoding_equals_the_generators(self):
        enc = sa.enc_weights(self.p, self.counts)
        self.assertEqual(sorted(enc), [f"encp{j}" for j in range(4)])
        for name, blob in enc.items():
            self.assertEqual(blob, self.baked(name), name)

    def test_encoding_vectors_hold_the_counts(self):
        counts = P.S.counts_for(37, 2)
        e = sa.enc_vectors(self.p, counts)
        self.assertEqual([float(e[0][16 * k]) for k in range(8)], [16, 16, 5, 0, 0, 0, 0, 0])
        self.assertEqual(float(e[1][16 * 7]), 0.0)
        self.assertEqual(float(e[0][16 * 8]), 16.0)


class Routes(unittest.TestCase):
    def test_reduced_fixture_routes(self):
        self.assertEqual(P.default_route(2), ([0, 1], 2))
        route, chips = P.default_route(8)
        for source, destination in zip(route, route[1:]):
            sl.hop(chips, 16, source, destination)
        for layers in (0, 9, 12, 15):
            with self.assertRaisesRegex(ValueError, "one to eight layers"):
                P.default_route(layers)

    def test_pool_names(self):
        for n in ("k0", "k15", "v0p3", "v5p15"):
            self.assertTrue(P.POOL_NAME.fullmatch(n), n)
        for n in ("encp0", "mk0_1", "mv", "lw0s3", "ep0", "hm0p1"):
            self.assertFalse(P.POOL_NAME.fullmatch(n), n)

    def test_node_wiring(self):
        self.assertEqual(sl.link_peer(16, 16, 0, 7), (8, 7))
        self.assertEqual(sl.link_peer(16, 16, 7, 7), None)
        self.assertEqual(sl.link_peer(3, 16, 0, 0), (1, 1))
        with self.assertRaises(ValueError):
            sl.hop(16, 16, 7, 8)                  # chips 7 and 8 are in different nodes: only chip 0 and chip 8 are joined


class HeadLayout(unittest.TestCase):
    """The head chips' weight layout (`ship_head.bind_blocks2`): where the blocks lie in the two banks."""

    @classmethod
    def setUpClass(cls):
        from opentsp_vllm.models.gpt2 import head as H
        cls.H, cls.p = H, tprofile.load("ref")

    def layout(self, vtiles, two=True):
        class K:                                     # the externs and weights the binder declares, without a program
            class b:
                class prog:
                    externs = []

                    @classmethod
                    def extern(cls, name, sl_, addr, words):
                        cls.externs.append((name, sl_, addr, words))
            weights = {}
        K.b.prog.externs = []
        W = np.zeros((128 * vtiles, 128 * 6), np.float16)
        out = self.H.bind_blocks2(K, self.p, W, 6, two)
        return out, K.b.prog.externs

    def test_tiled_head_fit_depends_on_banks(self):
        self.assertEqual(self.H.capacity(self.p, 6, True), 133)
        self.assertEqual(self.H.capacity(self.p, 6, False), 74)

    def test_no_block_touches_the_layernorm_scratch(self):
        out, ext = self.layout(131)
        self.assertEqual(len(out), 131 * 6)
        for (o, i), (pl, addr, _) in out.items():
            self.assertTrue(addr + 16 <= 0x1000 or self.H.B1_BASE <= addr, (o, i, addr))
            self.assertGreaterEqual(addr, sl.WBASE)
            self.assertLessEqual(addr + 16, self.H.B1_END)
        # the second extern of every slice starts above the scratch words 0x1000 .. 0x1050 of the LayerNorm
        self.assertTrue(all(a >= 0x1050 for n, s_, a, w in ext if n.startswith("lx")))
        self.assertTrue(any(n.startswith("lx") for n, s_, a, w in ext))

    def test_bank_zero_only_without_two_banks(self):
        self.layout(74, two=False)
        with self.assertRaises(ValueError):
            self.layout(75, two=False)

    def test_too_many_tiles_are_refused(self):
        with self.assertRaises(ValueError):
            self.layout(self.H.capacity(self.p, 6, True) + 1)


class BuildCleanup(unittest.TestCase):
    def test_an_interrupted_build_leaves_no_directory(self):
        import tempfile
        from opentsp_vllm.models.gpt2 import weights as W
        from opentsp_vllm.models.gpt2.config import GPT2_SMALL
        w = W.synthetic(GPT2_SMALL)
        saved = P.build_cache

        def boom(w_, layers, route, chips, d, log=print, kv_chunks=P.CHUNKS):
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "half"), "w").write("x")
            raise KeyboardInterrupt
        P.build_cache = boom
        try:
            with tempfile.TemporaryDirectory() as cache:
                with self.assertRaises(KeyboardInterrupt):
                    P.Pipeline(w, 2, cache=cache, log=lambda m: None)
                self.assertTrue(all(name.endswith(".lock") for name in os.listdir(cache)))
        finally:
            P.build_cache = saved


class CacheKey(unittest.TestCase):
    """The artifact key depends on build inputs and is portable across source directories."""

    def test_covers_the_encoding_and_the_assembler(self):
        names = P.digest_files(P.ROOT)
        self.assertIn("opentsp-arch-spec/isa-db/generated/encoding.json", names)
        self.assertIn("opentsp-arch-spec/tools/tsp_as.py", names)
        self.assertIn("opentsp-arch-spec/config/ref.json", names)

    def test_covers_the_files_that_pack_the_weights_and_save_the_stages(self):
        names = P.digest_files(P.ROOT)
        for f in ("pipeline.py", "system.py", "config.py", "weights.py"):
            self.assertIn("opentsp-vllm/src/opentsp_vllm/models/gpt2/" + f, names)

    class _W:
        def __init__(self, d):
            self.d = d

        def digest(self):
            return self.d

    def _copy_tree(self, dst):
        for n in P.digest_files(P.ROOT):
            os.makedirs(os.path.dirname(os.path.join(dst, n)), exist_ok=True)
            shutil.copy(os.path.join(P.ROOT, n), os.path.join(dst, n))

    def test_the_key_changes_with_the_content_of_every_covered_file(self):
        """Each covered file is changed in a copy of the tree, one at a time: the key must differ from the key of the unchanged copy."""
        with tempfile.TemporaryDirectory() as tmp:
            self._copy_tree(tmp)
            w = self._W("w1")
            base = P.source_digest(w, 2, [0, 1], 2, root=tmp)
            self.assertEqual(base, P.source_digest(w, 2, [0, 1], 2, root=tmp))
            for n in P.digest_files(P.ROOT):
                f = os.path.join(tmp, n)
                old = open(f, "rb").read()
                open(f, "wb").write(old + b"\n")
                try:
                    self.assertNotEqual(P.source_digest(w, 2, [0, 1], 2, root=tmp), base, n)
                finally:
                    open(f, "wb").write(old)

    def test_the_key_follows_the_weights_and_the_shape_of_the_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._copy_tree(tmp)
            k = lambda w="w1", layers=2, route=(0, 1), chips=2: P.source_digest(self._W(w), layers, list(route), chips, root=tmp)
            self.assertNotEqual(k(), k(w="w2"))
            self.assertNotEqual(k(), k(layers=3))
            self.assertNotEqual(k(), k(route=(1, 0)))
            self.assertNotEqual(k(), k(chips=3))

    def test_the_key_does_not_depend_on_where_the_tree_is(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self._copy_tree(a)
            self._copy_tree(b)
            self.assertEqual(P.source_digest(self._W("w"), 2, [0], 1, root=a), P.source_digest(self._W("w"), 2, [0], 1, root=b))

    def test_names_are_relative_to_the_tree(self):
        for n in P.digest_files(P.ROOT):
            self.assertFalse(os.path.isabs(n), n)
            self.assertTrue(os.path.isfile(os.path.join(P.ROOT, n)), n)


class CacheLoadOwnership(unittest.TestCase):
    def test_eviction_cannot_remove_an_entry_while_pipeline_is_opening_its_stages(self):
        self._load_race(False)

    def test_direct_managed_package_is_locked_through_its_stage_reads_too(self):
        self._load_race(True)

    def _load_race(self, direct):
        import json
        import threading
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from opentsp_vllm.models.gpt2 import cache as ship_cache
        entered, release = threading.Event(), threading.Event()
        failures, loaded = [], []
        with tempfile.TemporaryDirectory() as cache:
            for i, name in enumerate(["active", "newer0", "newer1", "newer2"]):
                d = Path(cache) / name
                (d / "s0").mkdir(parents=True)
                (d / "manifest.json").write_text(json.dumps({"model": "test", "units_sha256": {"s0": "unit"}}))
                (d / "s0/program.elf").write_bytes(b"owned-stage")
                os.utime(d, (1000 + i, 1000 + i))
            if direct:
                (Path(cache) / "active.lock").touch()

            def stage(directory, *args):
                entered.set()
                if not release.wait(10):
                    raise AssertionError("test did not release stage reader")
                return (Path(directory) / "s0/program.elf").read_bytes()

            def consumer():
                try:
                    entry = os.path.join(cache, "active") if direct else cache
                    loaded.append(P.Pipeline(SimpleNamespace(layers=[None]), layers=1, cache=entry, log=lambda _: None))
                except BaseException as error:
                    failures.append(error)

            with patch.object(P, "source_digest", return_value="active"), \
                    patch.object(P, "stage_manifest", return_value={"model": "test"}), \
                    patch.object(P.S, "layer_params", return_value=None), patch.object(P, "Stage", side_effect=stage):
                worker = threading.Thread(target=consumer)
                worker.start()
                try:
                    self.assertTrue(entered.wait(10), failures)
                    removed = ship_cache.evict(cache, keep=3, retain=os.path.join(cache, "newer2"))
                finally:
                    release.set()
                    worker.join(10)
                self.assertFalse(worker.is_alive(), "no lock-order deadlock")
                self.assertEqual(failures, [], removed)
                self.assertEqual(loaded[0].stages, [b"owned-stage"])


    def test_a_failed_consumer_releases_the_digest_lock_and_keeps_the_published_entry(self):
        import fcntl
        from pathlib import Path
        with tempfile.TemporaryDirectory() as cache:
            directory = os.path.join(cache, "active")
            def build(tmp):
                (Path(tmp) / "manifest.json").write_text("{}")
            def interrupted(_):
                raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                P.ensure_cache(directory, build, interrupted)
            self.assertTrue((Path(directory) / "manifest.json").is_file())
            self.assertFalse(any(".tmp" in name for name in os.listdir(cache)))
            with open(directory + ".lock", "a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(P.ensure_cache(directory, lambda _: self.fail("must not rebuild"), lambda _: "loaded"), "loaded")


if __name__ == "__main__":
    unittest.main()
