"""Profile identity survives each binary format without truncation."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import tsp_as as asm

CONFIG = str(Path(__file__).resolve().parents[2] / "config/mini.json")


class Names(unittest.TestCase):
    def test_sixteen_bytes_roundtrip_all_formats(self):
        name = "abcdefghijklmnop"
        obj = asm.write_object(SimpleNamespace(profile=name, images={}, externs={}), CONFIG)
        weights = asm.write_weights(name, [], CONFIG)
        manifest = asm.write_manifest(name, [obj], CONFIG)
        import json
        header_size = int.from_bytes(weights[:8], "little")
        weight_header = json.loads(weights[8:8 + header_size])
        self.assertEqual(weight_header["__metadata__"]["opentsp.profile"], name)
        self.assertEqual(asm.read_weights(weights, name, CONFIG), {})
        for data, reader in [(obj, asm.read_object), (manifest, asm.read_manifest)]:
            self.assertEqual(reader(data, name, CONFIG)[0], name)

    def test_seventeen_bytes_rejected_before_writing(self):
        name = "abcdefghijklmnopq"
        prog = SimpleNamespace(profile=name, images={}, externs={})
        for action in [lambda: asm.load_profile(name, CONFIG),
                       lambda: asm.write_object(prog, CONFIG),
                       lambda: asm.write_weights(name, [], CONFIG),
                       lambda: asm.write_manifest(name, [b""], CONFIG)]:
            with self.assertRaises(asm.AsmError):
                action()


SPEC = Path(__file__).resolve().parents[2]


def find_sim():
    """TSP_SIM, else the harness of this tree (target/spec/tsp_sim at the repository root)."""
    import os
    for p in (os.environ.get("TSP_SIM"), str(SPEC.parent / "target/spec/tsp_sim")):
        if p and Path(p).exists():
            return p
    return None


class FetchIntegrityVector(unittest.TestCase):
    """: the ecc-901 vector (an uncorrectable fault in the second sample of a stream Ifetch) is derived from the
    profile's timing and layout, not from a literal valid at ref latencies only."""

    def vector(self, profile):
        sim = find_sim()
        self.assertIsNotNone(sim, "no tsp_sim: set TSP_SIM or build target/spec/tsp_sim (opentsp-arch-spec/tools/build_sim.sh)")
        import gen_model_vectors as g
        cfg = str(SPEC / "config" / f"{profile}.json")
        return g.fetch_integrity_vector(sim, cfg, g.Catalogue(cfg))

    def test_ref_slow_generates_and_contains_the_fault(self):
        v = self.vector("test/ref_slow")
        self.assertEqual(v["name"], "ecc-901")
        self.assertEqual(len(v["host"]), 2)
        self.assertTrue([e for e in v["expect"]["ext_events"] if e.startswith("irq ")], "the fault raised no interrupt")

    def test_shipped_profiles_keep_their_golden_vector(self):
        import json
        for p in ("ref", "mini"):
            gold = [x for x in json.load(open(SPEC / "conformance/model" / f"{p}.json"))["vectors"] if x["name"] == "ecc-901"][0]
            self.assertEqual(self.vector(p), gold, p)

    def test_every_variant_profile_generates(self):
        for p in ("test/mid_min", "test/fetch_alt", "test/link_tight", "test/c2c_tx_delayed", "test/ref_small", "test/mini_small"):
            with self.subTest(profile=p):
                self.assertEqual(self.vector(p)["name"], "ecc-901")


class ExtensionVectorCoverage(unittest.TestCase):
    def test_mini_extension_vectors_include_the_directed_extension_programs(self):
        import json
        import tempfile
        import gen_model_vectors as g
        sim = find_sim()
        self.assertIsNotNone(sim, "build target/spec/tsp_sim first")
        cfg = str(SPEC / "config/test/mini_ext.json")
        cat = g.Catalogue(cfg)
        programs = g.wide_programs(sim, cfg)
        self.assertTrue(programs)
        expected = set()
        for program in programs:
            recs, _, _, units = g.fit(program, cfg, cat, False)
            expected.add(json.dumps(g.text_json(g.lowered(recs, units)), sort_keys=True))
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "mini_ext.json")
            self.assertEqual(g.main(["gen_model_vectors.py", sim, cfg, output, "0", "7"]), 0)
            vectors = json.loads(Path(output).read_text())["vectors"]
        actual = {json.dumps(v["text"], sort_keys=True) for v in vectors}
        self.assertEqual(expected - actual, set(), "directed extension programs missing from mini_ext vectors")


if __name__ == "__main__":
    unittest.main()
