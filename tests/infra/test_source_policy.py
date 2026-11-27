"""The source scan covers historical folders, filenames and UTF-16 bytes too."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from check_source_policy import scan


class SourcePolicy(unittest.TestCase):
    def test_all_inventory_paths(self):
        identifier = "US" + "12345678"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = ["docs/reviews/old.txt", "docs/research/old.bin", identifier + ".txt", "LICENSE"]
            for name in paths:
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"Patent license grant")
            (root / paths[0]).write_text(identifier)
            (root / paths[1]).write_bytes(identifier.encode("utf-16-le"))
            self.assertEqual(scan(root, paths), sorted(paths[:3]))


if __name__ == "__main__":
    unittest.main()
