#!/usr/bin/env python3
"""JSON key spelling must not change effective configuration defaults or behavior."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SIM = sys.argv[1]
CFG = json.loads((ROOT / "config/mini.json").read_text())
SCHEMA = json.loads((ROOT / "config/schema.json").read_text())
CFG.setdefault("ras", {})["diag_depth"] = 1
# Ten invalid NOPs expose the diagnostic FIFO capacity and overflow flag.
PROGRAM = "Q 0 12000000 00010000 " + " ".join(["00000001 10000002 00000000"] * 10) + "\nR 400\nR 402\n"
PROGRAM = PROGRAM.replace("\\n", "\n")


def escaped_key(text, key):
    return text.replace(json.dumps(key) + ":", '"\\u%04x%s":' % (ord(key[0]), key[1:]))


def run(directory, name, text):
    path = Path(directory) / (name + ".json")
    path.write_text(text)
    return subprocess.run([SIM, str(path), "--trace"], input=PROGRAM,
                          capture_output=True, text=True, timeout=10)


def main():
    with tempfile.TemporaryDirectory(prefix="escaped-defaults-") as directory:
        reference = run(directory, "plain", json.dumps(CFG))
        assert reference.returncode == 0, reference.stderr
        assert "X 400 1\nX 402 1\n" in reference.stdout, reference.stdout
        filled = copy.deepcopy(CFG)
        omitted = copy.deepcopy(CFG)
        default_keys = []
        for section, spec in SCHEMA["properties"].items():
            for key, prop in spec.get("properties", {}).items():
                if "default" not in prop:
                    continue
                default_keys.append(key)
                filled.setdefault(section, {}).setdefault(key, prop["default"])
                if key != "diag_depth":
                    omitted.setdefault(section, {}).pop(key, None)
        cases = {"all-explicit": json.dumps(filled), "optional-omitted": json.dumps(omitted)}
        for section in CFG:
            cases["section-" + section] = escaped_key(json.dumps(omitted), section)
        for key in sorted(set(default_keys)):
            cases["member-" + key] = escaped_key(json.dumps(filled), key)
        cases["all-keys-escaped"] = json.dumps(filled)
        for key in set(CFG).union(default_keys):
            cases["all-keys-escaped"] = escaped_key(cases["all-keys-escaped"], key)
        fully_omitted = copy.deepcopy(omitted)
        for section, spec in SCHEMA["properties"].items():
            for key, prop in spec.get("properties", {}).items():
                if "default" in prop:
                    fully_omitted.setdefault(section, {}).pop(key, None)
        default_reference = run(directory, "default-reference", json.dumps(fully_omitted))
        assert default_reference.returncode == 0, default_reference.stderr
        assert "X 400 a\nX 402 0\n" in default_reference.stdout
        for section in CFG:
            cases["default-section-" + section] = escaped_key(json.dumps(fully_omitted), section)
        failures = []
        for name, text in cases.items():
            actual = run(directory, name, text)
            expected = default_reference.stdout if name.startswith("default-section-") else reference.stdout
            if (actual.returncode, actual.stdout) != (0, expected):
                failures.append((name, actual.returncode, actual.stderr[-200:], actual.stdout[-100:]))
        # Semantically duplicate and NUL keys remain invalid before defaults.
        duplicate = json.dumps(CFG).replace('"diag_depth": 1', '"diag_depth": 1, "diag_\\u0064epth": 2')
        nul = json.dumps(CFG).replace('"ras":', '"ras\\u0000":')
        for name, text in (("duplicate-decoded", duplicate), ("nul-key", nul)):
            actual = run(directory, name, text)
            if actual.returncode != 2 or actual.stdout:
                failures.append((name, actual.returncode, actual.stderr[-200:], actual.stdout[-100:]))
        assert not failures, failures
        print(f"PASS: {len(cases)} equivalent configuration spellings and 2 invalid controls")


if __name__ == "__main__":
    main()
