#!/usr/bin/env bash
# Check maintained source and prose for historical development identifiers and source naming.
# CLEAN_TEXT_ROOT/PATTERNS/ALLOW support isolated regression fixtures.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLEAN_TEXT_TOOL_DIR="$here"
export CLEAN_TEXT_ROOT="${CLEAN_TEXT_ROOT:-$(cd "$here/.." && pwd)}"
export CLEAN_TEXT_PATTERNS="${CLEAN_TEXT_PATTERNS:-$here/clean_text_patterns.txt}"
export CLEAN_TEXT_ALLOW="${CLEAN_TEXT_ALLOW:-$here/clean_text_allow.txt}"

exec python3 - "$@" <<'PY'
import collections, fnmatch, os, re, sys
sys.path.insert(0, os.environ["CLEAN_TEXT_TOOL_DIR"])
from source_files import inventory

root = os.environ["CLEAN_TEXT_ROOT"]
mode, use_allow = "check", True
for a in sys.argv[1:]:
    if a == "--count":
        mode = "count"
    elif a == "--count-files":
        mode = "files"
    elif a == "--no-allow":
        use_allow = False
    else:
        sys.exit("check_clean_text: unknown option " + a)

CODE = {".rs", ".py", ".sh", ".toml", ".yaml", ".yml", ".sail", ".c", ".h"}
patterns = []
for n, line in enumerate(open(os.environ["CLEAN_TEXT_PATTERNS"], encoding="utf-8"), 1):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    parts = [p.strip() for p in line.split("|", 2)]
    if len(parts) != 3 or parts[0] not in ("all", "code"):
        sys.exit("check_clean_text: bad pattern line %d: %s" % (n, line))
    patterns.append((parts[0], parts[1], re.compile(parts[2])))

allow = []
if use_allow:
    for n, line in enumerate(open(os.environ["CLEAN_TEXT_ALLOW"], encoding="utf-8"), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        glob, bar, reason = line.partition("|")
        if not bar or not reason.strip():
            sys.exit("check_clean_text: allowlist line %d has no reason: %s" % (n, line))
        allow.append(glob.strip())


def exempt(path):
    return any(path.startswith(g) if g.endswith("/") else fnmatch.fnmatch(path, g) for g in allow if "::" not in g)


line_allow = [g.split("::", 1) for g in allow if "::" in g]


def line_exempt(path, line):
    return any(fnmatch.fnmatch(path, g) and text in line for g, text in line_allow)


def files():
    return inventory(root)



def group(path):
    p = path.split("/")
    return "/".join(p[:2]) if len(p) > 2 else p[0] if len(p) == 1 else "/".join(p[:2])


hits = []
for path in files():
    full = os.path.join(root, path)
    if exempt(path) or not os.path.isfile(full) or os.path.islink(full):
        continue
    data = open(full, "rb").read()
    if b"\0" in data:
        continue
    code = os.path.splitext(path)[1] in CODE or os.path.basename(path) == "justfile"
    for n, line in enumerate(data.decode("utf-8", "replace").split("\n"), 1):
        for scope, name, rx in patterns:
            if (scope == "all" or code) and rx.search(line) and not line_exempt(path, line):
                hits.append((path, n, name, line.strip()))
                break

if mode == "check":
    for path, n, name, text in hits:
        print("%s:%d: [%s] %s" % (path, n, name, text[:160]))
    if hits:
        print("check_clean_text: %d line(s) match tools/clean_text_patterns.txt (reword them; see the header of tools/check_clean_text.sh)" % len(hits), file=sys.stderr)
        sys.exit(1)
    print("check_clean_text: clean")
else:
    count = collections.Counter()
    names = sorted({h[2] for h in hits})
    for path, n, name, text in hits:
        count[(path if mode == "files" else group(path), name)] += 1
    keys = sorted({k[0] for k in count})
    print("%-60s %6s  %s" % ("path", "lines", "  ".join(names)))
    for k in keys:
        print("%-60s %6d  %s" % (k, sum(count[(k, nm)] for nm in names), "  ".join(str(count[(k, nm)]) for nm in names)))
    print("%-60s %6d" % ("TOTAL", len(hits)))
PY
