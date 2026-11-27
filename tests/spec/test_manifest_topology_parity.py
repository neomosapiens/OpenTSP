"""diff_tsp_as.py names the widened-key profiles and an mpack --topology phase."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
DIFF = (ROOT / "opentsp-sw-toolchain/tests/diff_tsp_as.py").read_text()


class Iaf04DiffKeys(unittest.TestCase):
    def test_diff_names_the_four_profiles(self):
        i = DIFF.index("# Profile-variable bounds under --config.")
        block = DIFF[i : i + 400]
        for name in ("w16", "acc8", "one_plane", "c2c_wide"):
            self.assertIn(f'"{name}"', block)

    def test_diff_has_mpack_topology_phase(self):
        self.assertIn("iaf04-topo", DIFF)
        self.assertIn('["mpack", "--profile", "ref", "--topology", topo', DIFF)


if __name__ == "__main__":
    unittest.main()
