#!/usr/bin/env python3
"""The number of `while` loops in the generated Lean of the model may only fall.

usage: lean_while.py GENERATED_DIR RECORDED_FILE

Lean elaborates a Sail `while` as `Loop.forIn`, a `partial def`: a function that contains or calls one is opaque to a proof, with no equation
lemma. A function that a proof about `tick` has to unfold cannot contain one. RECORDED_FILE holds the count of the day (and nothing else); a
higher count fails, a lower one fails until the file is lowered to it in the same change, so that the number is a ratchet.
Counts every `while` statement of every `.lean` file under GENERATED_DIR and prints the count per file (largest first).
"""
from pathlib import Path
import re
import sys


def count(directory):
    """file name -> number of `while` loops, for the files that have any."""
    found = {}
    for path in sorted(Path(directory).rglob("*.lean")):
        loops = len(re.findall(r"^\s*while\s", path.read_text(), re.M))
        if loops:
            found[path.name] = loops
    return found


def verdict(total, recorded):
    """(ok, message)."""
    if total > recorded:
        return False, "%d while loops in the generated Lean, %d recorded: a loop was added (Lean cannot unfold a function with one)" % (total, recorded)
    if total < recorded:
        return False, "%d while loops in the generated Lean, %d recorded: lower the record to %d" % (total, recorded, total)
    return True, "%d while loops in the generated Lean, as recorded" % total


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    per_file = count(argv[0])
    recorded = int(Path(argv[1]).read_text().split()[0])
    ok, message = verdict(sum(per_file.values()), recorded)
    print(("ok: " if ok else "FAIL: ") + message)
    print("  " + ", ".join("%s %d" % (name, loops) for name, loops in sorted(per_file.items(), key=lambda kv: (-kv[1], kv[0]))))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
