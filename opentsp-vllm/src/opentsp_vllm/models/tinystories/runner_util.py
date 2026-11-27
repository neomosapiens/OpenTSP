"""Running a generated case through `tests/programs/run.py` from a test (the ISS legs only: the case's `sim` tag is
dropped, tsp_sim is run by the recipes). The binaries are built when absent, so the test runs rather than skips."""
import os
import subprocess

from opentsp_vllm.models.tinystories import paths as _paths

ROOT = _paths.ROOT
ISS = os.path.join(ROOT, "target", "release", "tsp-iss")
ASM = os.path.join(ROOT, "target", "release", "ots-asm")


def ensure_binaries():
    if not (os.access(ISS, os.X_OK) and os.access(ASM, os.X_OK)):
        subprocess.run(["cargo", "build", "--release", "-q", "-p", "ots-asm", "-p", "tsp-iss"], cwd=ROOT, check=True)


