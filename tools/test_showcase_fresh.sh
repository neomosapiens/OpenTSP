#!/usr/bin/env bash
# Test of the freshness of the ISS binding in tools/showcase.sh (run by `just vllm-binding-test`): a source of a Cargo
# dependency of the binding that is not under the ISS or binding directories (here `ots-profile`) is newer than the
# extension; the launcher must rebuild the extension. The edit is a `touch` of one source (no content changes) that is
# set back to its old modification time at the end. Needs the binding to be built (`just vllm-binding`).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
so=opentsp-vllm/native/target/$(uname -s)-$(uname -m)/_tsp_iss.abi3.so
dep=opentsp-sw-toolchain/crates/ots-profile/src/lib.rs
[ -e "$so" ] || { echo "FAIL the binding is not built; run 'just vllm-binding' first" >&2; exit 1; }
[ -e "$dep" ] || { echo "FAIL $dep does not exist" >&2; exit 1; }

mtime() { python3 -c 'import os, sys; print(os.stat(sys.argv[1]).st_mtime_ns)' "$1"; }
old="$(mtime "$dep")"
restore() { python3 -c 'import os, sys; os.utime(sys.argv[1], ns=(int(sys.argv[2]), int(sys.argv[2])))' "$dep" "$old"; }
trap restore EXIT

# A warm launch must preserve the installed binding; a changed dependency must refresh it.
bash tools/showcase.sh chat --help >/dev/null 2>&1
warm="$(mtime "$so")"
bash tools/showcase.sh chat --help >/dev/null 2>&1
[ "$(mtime "$so")" = "$warm" ] || { echo "FAIL a warm launch republished the unchanged binding" >&2; exit 1; }
echo "PASS a warm launch reused the unchanged binding"
sleep 1.1
touch "$dep"
before="$(mtime "$so")"
[ "$(mtime "$dep")" -gt "$before" ] || { echo "FAIL the dependency is not newer than the extension" >&2; exit 1; }
bash tools/showcase.sh chat --help >/dev/null 2>&1
after="$(mtime "$so")"
if [ "$after" -gt "$before" ]; then
    echo "PASS the launcher rebuilt the extension after a dependency changed"
else
    echo "FAIL the extension is older than its dependency $dep and the launcher did not rebuild it" >&2
    exit 1
fi
