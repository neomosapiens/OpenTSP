"""Validate a profile with bf16, wide_acc, block_fp8, fp8_rate 2 and acc_depth 8."""
from pathlib import Path
import json
import sys
import unittest

SPEC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SPEC / "tools"))


class AllExtProfile(unittest.TestCase):
    PATH = SPEC / "config/test/all_ext.json"

    def test_profile_exists_with_the_combined_keys(self):
        self.assertTrue(self.PATH.is_file(), "missing config/test/all_ext.json")
        cfg = json.loads(self.PATH.read_text())
        self.assertEqual(sorted(cfg["extensions"]), ["bf16", "block_fp8", "wide_acc"])
        self.assertEqual(cfg["mxm"]["fp8_rate"], 2)
        self.assertEqual(cfg["mxm"]["acc_depth"], 8)
        self.assertEqual(cfg["mxm"]["acc_bits"], 128)

    def test_schema_invariants_accept(self):
        from opentsp_reference import schema as si
        errs = si.check(json.loads(self.PATH.read_text()))
        self.assertEqual(errs, [])

    def test_model_test_variants_list_includes_it(self):
        text = (SPEC / "tools/model_test.sh").read_text()
        self.assertRegex(text, r'VARIANTS="[^"]*test/all_ext')



    def test_default_check_refuses_an_orphaned_all_ext_sidecar(self):
        import tempfile
        from unittest.mock import patch
        import gen_rtl_obligations as rtl

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "opentsp-arch-spec"
            side = spec / "conformance/rtl"
            side.mkdir(parents=True)
            for name in ("ref", "mini", "all_ext"):
                (side / f"{name}.json").write_text("{}\n")
            original_build = rtl.build

            def build(iss, profile, jobs):
                if profile == "all_ext":
                    return original_build(iss, profile, jobs)
                return {}

            with patch.object(rtl, "ROOT", root), patch.object(rtl, "SPEC", spec), patch.object(rtl, "build", build):
                with self.assertRaises(FileNotFoundError):
                    rtl.main(["--iss", "unused", "--check"])

    def test_rtl_sidecar_exists(self):
        """: an RTL obligation sidecar covers the all_ext golden vectors (x128 / bfp8)."""
        side = SPEC / "conformance/rtl/all_ext.json"
        self.assertTrue(side.is_file(), "missing conformance/rtl/all_ext.json")
        doc = json.loads(side.read_text())
        self.assertEqual(doc["profile"], "all_ext")
        self.assertGreaterEqual(doc["counts"]["pass"] + doc["counts"]["none"] + doc["counts"]["undecided"], 1)
        self.assertEqual(len(doc["vectors"]), sum(doc["counts"].values()))

if __name__ == "__main__":
    unittest.main()
