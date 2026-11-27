"""A changed screening value must fail even when double and long double agree."""
import math
import unittest

from mpmath import mp
from transcendental_filter_audit import relative_error


class FilterAudit(unittest.TestCase):
    def test_independent_precision(self):
        value = math.exp(1)
        self.assertLess(relative_error(22, 0x3f800000, value), mp.mpf(2) ** -48)
        # Equal-width host types can agree on the same incorrect value.
        changed = value * (1 + 2 ** -40)
        self.assertGreater(relative_error(22, 0x3f800000, changed), mp.mpf(2) ** -48)


if __name__ == "__main__":
    unittest.main()
