"""Host counters reject missing, unavailable, duplicate and nonpositive measurements."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('host_counters', ROOT / 'opentsp-sw-toolchain/python/opentsp_tools/host_counters.py')
counters = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(counters)


class HostCounters(unittest.TestCase):
    def test_linux_counter(self):
        self.assertEqual(counters.count('Linux', '# header\n760761,,instructions,266468,100.00,,\n'), 760761)

    def test_macos_counter(self):
        self.assertEqual(counters.count('Darwin', '  123456 instructions retired\n'), 123456)

    def test_unavailable_or_ambiguous_is_not_zero(self):
        for text in ('', '<not supported>,,instructions,0,100,,\n', '0,,instructions,0,100,,\n', '-1,,instructions,0,100,,\n', '100,,instructions,0,100,,\n200,,instructions,0,100,,\n'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                counters.count('Linux', text)

    def test_unsupported_platform(self):
        with self.assertRaises(ValueError):
            counters.count('unsupported', '100,,instructions,0,100,,\n')


if __name__ == '__main__':
    unittest.main()
