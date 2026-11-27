"""Source/data locations for this model package; imports never change global search paths."""
from pathlib import Path
from opentsp_tools import paths
ROOT, SPEC = paths.ROOT, paths.SPEC
HERE = str(Path(__file__).resolve().parent)
