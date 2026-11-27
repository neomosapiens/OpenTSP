"""Running a generated case through `tests/programs/run.py` from a test (the ISS legs only: the case's `sim` tag is
dropped, tsp_sim is run by the recipes). The binaries are built when absent, so the test runs rather than skips."""
import json
import os
import subprocess
import sys
import tempfile

from opentsp_vllm.models.tinystories import paths as _paths

ROOT = _paths.ROOT
ISS = os.path.join(ROOT, "target", "release", "tsp-iss")
ASM = os.path.join(ROOT, "target", "release", "ots-asm")


def ensure_binaries():
    if not (os.access(ISS, os.X_OK) and os.access(ASM, os.X_OK)):
        subprocess.run(["cargo", "build", "--release", "-q", "-p", "ots-asm", "-p", "tsp-iss"], cwd=ROOT, check=True)


def run_on_iss(text, case, flip=None, timeout=900, sim=None):
    """(returncode, output) of run.py on the case with its `sim` tag dropped (kept when `sim`, the path of tsp_sim, is
    given); `flip` = a dump key whose expected word gets its lowest bit inverted (a planted defect, which run.py must
    report)."""
    ensure_binaries()
    case = json.loads(json.dumps(case))
    if sim is None:
        case["tags"] = [t for t in case["tags"] if t != "sim"]
    if flip is not None:
        w = bytearray.fromhex(case["expect"]["words"][flip])
        w[0] ^= 1
        case["expect"]["words"][flip] = w.hex()
    with tempfile.TemporaryDirectory(prefix="ts3-case-") as d:
        name = os.path.basename(case["program"])
        open(os.path.join(d, name), "w").write(text)
        path = os.path.join(d, name.replace(".ots", ".case.json"))
        json.dump(case, open(path, "w"))
        run_py = os.path.join(ROOT, "tests", "programs", "run.py")
        r = subprocess.run([sys.executable, run_py, "--sim", sim or sys.executable, "--iss", ISS, "--asm", ASM, path],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
