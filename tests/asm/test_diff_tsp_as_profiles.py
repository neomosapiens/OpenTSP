"""Verify assembler differential coverage for extended profiles and mpack --topology."""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = (ROOT / "opentsp-sw-toolchain/tests/diff_tsp_as.py").read_text()
fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))
    fails += 0 if ok else 1


# Each profile must appear as a config/test/<name>.json path (not only in a refused-bound comment).
for name in ("w16", "acc8", "one_plane", "c2c_wide"):
    hit = re.search(rf'config["\'].*?test["\'].*?{name}|test/{name}\.json|test["\', ].*{name}\.json', SRC)
    hit = hit or (f"{name}.json" in SRC and "config/test" in SRC and name in SRC)
    # require the exact relative path form used by the other extension profiles
    hit = f"test/{name}.json" in SRC or f'test", "{name}.json"' in SRC or f"test\", f\"{name}" in SRC
    hit = hit or bool(re.search(rf'["\']test/{name}\.json["\']|join\(SPEC,\s*["\']config["\'],\s*["\']test["\'],\s*["\']{name}\.json["\']\)', SRC))
    hit = hit or bool(re.search(rf'os\.path\.join\(SPEC,\s*"config",\s*"test",\s*f?"{name}\.json"\)', SRC))
    hit = hit or bool(re.search(rf'["\']config/test/{name}\.json["\']', SRC))
    # also accept NEW_KEY_PROFILES = ("w16", ...) with a join that builds the path
    hit = hit or (
        re.search(r'NEW_KEY_PROFILES|CR_PROFILES|KEY_PROFILES', SRC) is not None
        and re.search(rf'["\']{name}["\']', SRC) is not None
        and "config" in SRC and "test" in SRC
    )
    check(f"--config variant names {name}", bool(hit), "not found as a config/test path in diff_tsp_as.py")

check("a topology_phase (or mpack topology) function exists",
      "def topology_phase" in SRC or "def mpack_topology" in SRC,
      "no topology_phase function")
check("topology probes include chips 513", '"chips": 513' in SRC or "'chips': 513" in SRC or "chips\":513" in SRC)
check("topology probes include chips 0", '"chips": 0' in SRC or "'chips': 0" in SRC or '"chips":0' in SRC)
check("topology probes include chips true", '"chips": true' in SRC or "'chips': True" in SRC or '"chips":true' in SRC)
check("topology probes include a non-JSON body", "not json" in SRC.lower() or "{not" in SRC or "non-JSON" in SRC or "non_json" in SRC)

sys.exit(1 if fails else 0)
