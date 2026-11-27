"""Public assurance documentation distinguishes tests and proofs from whole-machine assurance."""
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[2]
class AssuranceBoundary(unittest.TestCase):
    def test_whole_machine_limitation_is_explicit(self):
        text = (ROOT / 'docs/assurance.md').read_text().lower()
        self.assertIn('whole-machine', text)
        self.assertRegex(text, r'not (?:fully )?(?:proved|proven|verified)|unproved|unverified|incomplete')
    def test_mutation_ratchet_keeps_explicit_expectations(self):
        rows = (ROOT / 'opentsp-arch-spec/test/formal/mutants_expected.txt').read_text().splitlines()
        entries = [r.split() for r in rows if r.strip() and not r.lstrip().startswith('#')]
        self.assertTrue(entries)
        self.assertEqual(len({r[0] for r in entries}),len(entries))
        self.assertTrue(all(len(r)>=2 and r[1] in ('kill','survive') for r in entries))
if __name__ == '__main__':
    unittest.main()
