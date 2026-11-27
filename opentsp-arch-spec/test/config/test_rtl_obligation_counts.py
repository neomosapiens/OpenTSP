"""Bring-up documentation and RTL sidecars agree on verification counts."""
from pathlib import Path
import json
import unittest

SPEC = Path(__file__).resolve().parents[2]
ROOT = SPEC.parent
BRING_UP = ROOT / "opentsp-arch-spec/docs/bring-up.md"


class RtlObligationCounts(unittest.TestCase):
    EXPECTED = {
        "ref": {"pass": 396, "none": 517, "undecided": 74, "vectors": 987},
        "mini": {"pass": 394, "none": 512, "undecided": 68, "vectors": 974},
        # x128+bfp8 dump under model-extra (outside freeze); RTL sidecar beside it.
        "all_ext": {"pass": 6, "none": 8, "undecided": 3, "vectors": 17},
    }

    def test_sidecars_match_expected(self):
        for name, exp in self.EXPECTED.items():
            side = json.loads((SPEC / "conformance/rtl" / f"{name}.json").read_text())
            self.assertEqual(side["counts"], {k: exp[k] for k in ("pass", "none", "undecided")}, name)
            self.assertEqual(len(side["vectors"]), exp["vectors"], name)

    def test_bring_up_table_matches_sidecars(self):
        text = BRING_UP.read_text()
        for name, exp in self.EXPECTED.items():
            pat = rf"\|\s*`{name}`\s*\|\s*{exp['pass']}\s*\|\s*{exp['none']}\s*\|\s*{exp['undecided']}\s*\|\s*{exp['vectors']}\s*\|"
            self.assertRegex(text, pat, "bring-up.md table row for %s" % name)



    def test_bring_up_names_all_ext(self):
        """bring-up.md records the all_ext RTL sidecar counts (model-extra vectors)."""
        text = BRING_UP.read_text()
        exp = self.EXPECTED["all_ext"]
        pat = rf"\|\s*`all_ext`\s*\|\s*{exp['pass']}\s*\|\s*{exp['none']}\s*\|\s*{exp['undecided']}\s*\|\s*{exp['vectors']}\s*\|"
        self.assertRegex(text, pat, "bring-up.md table row for all_ext")
        self.assertIn("model-extra", text)


if __name__ == "__main__":
    unittest.main()
