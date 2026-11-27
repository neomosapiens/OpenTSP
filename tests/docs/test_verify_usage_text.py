#!/usr/bin/env python3
"""the usage line of `tsp-iss verify` in assembly.md is the one the tool prints (USAGE in verify.rs), so a flag added to the tool
(--topology) is in the document."""
import os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src = open(os.path.join(ROOT, "opentsp-iss", "crates", "tsp-iss", "src", "verify.rs"), encoding="utf-8").read()
usage = re.search(r'pub const USAGE: &str = "usage: (tsp-iss verify[^"]*)";', src).group(1)
doc = open(os.path.join(ROOT, "opentsp-arch-spec", "docs", "assembly.md"), encoding="utf-8").read()
if f"`{usage}`" not in doc:
    print("FAIL: assembly.md does not give the usage line of verify.rs:\n  " + usage)
    sys.exit(1)
print("verify usage line: all pass")
