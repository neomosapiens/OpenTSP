#!/usr/bin/env python3
"""Exercise compiled catalogue matchers at actual profile-dependent boundaries."""
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import tsp_as


class CatalogueProfileBoundaries(unittest.TestCase):
    def test_compiled_matchers_follow_each_profile(self):
        catalogue = json.loads((ROOT / "isa-db/generated/encoding.json").read_text())
        entries = {entry["name"]: entry for entry in catalogue["instructions"]}

        def word(name, **fields):
            entry = entries[name]
            result = int(entry["match"].replace("-", "0"), 2)
            for variable in entry["variables"]:
                value = fields.get(variable["name"], 0)
                shift = 0
                for key in ("location", "hi_location", "top_location"):
                    if key not in variable:
                        continue
                    loc = variable[key].split("-")
                    high, low = int(loc[0]), int(loc[-1])
                    size = high - low + 1
                    result |= ((value >> shift) & ((1 << size) - 1)) << low
                    shift += size
            return result

        with tempfile.TemporaryDirectory(prefix="catalogue-profile-boundary-") as tmp:
            work = Path(tmp)
            for bits, extensions, depth in ((13, [], 2), (16, ["bf16", "wide_acc", "block_fp8"], 8)):
                config = json.loads((ROOT / "config/ref.json").read_text())
                config["sram"]["addr_bits"] = bits
                config["mxm"]["acc_depth"] = depth
                config["extensions"] = extensions
                if "wide_acc" in extensions:
                    config["mxm"]["acc_bits"] = 128
                profile = work / "profile.json"
                profile.write_text(json.dumps(config))
                header = work / "tables.h"
                subprocess.run(["python3", str(ROOT / "test/isa_db/gen_exhaustive_tables.py"),
                                str(ROOT / "isa-db/generated/encoding.json"), str(header), str(profile)], check=True)
                source = work / "probe.c"
                source.write_text('#define VLANES 320ULL\n#define ACC_DEPTH 2ULL\n'
                                  '#include "tables.h"\n'
                                  'int match32(unsigned long long w) { return catalogue_matches32(w); }\n'
                                  'int match64(unsigned long long w) { return catalogue_matches64(w); }\n')
                library = work / ("probe%d.so" % bits)
                subprocess.run(["cc", "-shared", "-fPIC", "-O0", str(source), "-o", str(library)], check=True)
                native = ctypes.CDLL(str(library))
                for function in (native.match32, native.match64):
                    function.argtypes = [ctypes.c_uint64]
                    function.restype = ctypes.c_int
                cases = [
                    ("IfetchM", {"addr": 8191}, 1),
                    ("IfetchM", {"addr": 8192}, int(bits == 16)),
                    ("IfetchM", {"addr": 65535}, int(bits == 16)),
                    ("Read", {"addr": 8192}, int(bits == 16)),
                    ("Write", {"addr": 65535}, int(bits == 16)),
                    ("SetSegment", {"n": 1 << bits}, 1),
                    ("SetSegment", {"n": (1 << bits) + 1}, 0),
                    ("Step", {"a": 8191}, 1),
                    ("Step", {"a": 8192}, int(bits == 16)),
                    ("Step", {"a": (-8192) & 65535}, int(bits == 16)),
                    ("Step", {"a": 32768}, 0),
                    ("IW", {"type": 4}, int("bf16" in extensions)),
                    ("IW", {"type": 5}, int("block_fp8" in extensions)),
                    ("ACC", {"type": 0}, int("wide_acc" in extensions)),
                    ("ACC", {"type": 2, "acc": depth - 1}, 1),
                    ("ACC", {"type": 2, "acc": depth}, 0),
                ]
                for name, fields, expected in cases:
                    with self.subTest(addr_bits=bits, name=name, fields=fields):
                        function = native.match32 if entries[name]["width"] == 32 else native.match64
                        self.assertEqual(function(word(name, **fields)), expected)
                # Literal words are independent of the test's field builder. The
                # reference assembler separately decodes every split-field control.
                reference = tsp_as.ISA(config=str(profile))
                split_cases = [
                    ("Add", 0x2000000000000000, 1),
                    ("Add", 0x2000000000020000, int("bf16" in extensions)),
                    ("Add_sat", 0x2040000000020000, 0),
                    ("Convert", 0x2880000000010000, int("bf16" in extensions)),
                ]
                for name, encoded, expected in split_cases:
                    with self.subTest(addr_bits=bits, name=name, word=hex(encoded)):
                        self.assertEqual(native.match64(encoded), expected)
                        self.assertEqual(reference.decode(64, encoded) is not None, bool(expected))
                for entry in catalogue["instructions"]:
                    for variable in entry["variables"]:
                        if "top_location" not in variable:
                            continue
                        for code in range(32):
                            fields = {field["name"]: 0 for field in entry["variables"]}
                            fields[variable["name"]] = code
                            encoded = reference.encode(entry, fields)
                            expected = int(reference.decode(entry["width"], encoded) is not None)
                            with self.subTest(addr_bits=bits, name=entry["name"],
                                              field=variable["name"], code=code):
                                self.assertEqual(word(entry["name"], **fields), encoded)
                                self.assertEqual(native.match64(encoded), expected)


if __name__ == "__main__":
    unittest.main()
