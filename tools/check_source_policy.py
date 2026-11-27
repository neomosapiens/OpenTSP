#!/usr/bin/env python3
"""Check distributable paths and bytes for disallowed source identifiers.

No historical-document exemption. Standard license terms are unaffected.
This scans the current source directory and is
not a legal opinion or a proof of independent provenance.
"""
from pathlib import Path
import re
import sys
import unicodedata

IDENTIFIER = re.compile(r"\b(?:US|EP|WO)[\s-]*\d[\d, -]{6,}(?:[AB]\d)?\b", re.I)
SOURCE_HOST = re.compile("patents" + r"\.google\.com|patentscope" + r"\.wipo\.int", re.I)


def disallowed(text):
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    return bool(IDENTIFIER.search(text) or SOURCE_HOST.search(text))


def scan(root, paths):
    bad = []
    for name in sorted(set(paths)):
        p = root / name
        if not p.is_file():
            continue
        data = p.read_bytes()
        forms = [name, data.decode("utf-8", "ignore")]
        if b"\0" in data:
            forms.extend([data.decode("utf-16-le", "ignore"), data.decode("utf-16-be", "ignore")])
        if any(disallowed(form) for form in forms):
            bad.append(name)
    return bad


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    from source_files import inventory
    paths = inventory(root)
    bad = scan(root, paths)
    if bad:
        sys.exit("source policy: disallowed references in\n" + "\n".join(bad))
    print("source policy: current source directory clean")
