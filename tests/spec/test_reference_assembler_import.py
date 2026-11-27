"""Toolchain patching modules use the shared reference-assembler loader."""
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[2]
class ReferenceAssemblerImport(unittest.TestCase):
    def test_pager_and_patch_use_shared_loader(self):
        tools = ROOT / 'opentsp-sw-toolchain/python/opentsp_tools'
        refasm = (tools / 'refasm.py').read_text()
        self.assertIn('importlib.util.spec_from_file_location', refasm)
        self.assertNotIn('sys.path.insert', refasm)
        for name in ('pager.py','patch.py'):
            source = (tools / name).read_text()
            self.assertNotIn('sys.path', source)
            self.assertIn('refasm.load()', source)
if __name__ == '__main__':
    unittest.main()
