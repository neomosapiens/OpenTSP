"""The test-machine oracle uses the architecture's exact numerical reference."""
import sys

from opentsp_reference import numerics as _numerics

sys.modules[__name__] = _numerics
