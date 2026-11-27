#!/usr/bin/env bash
# Run the $[test] functions of one Sail file of test/formal/ on the model (C backend, the model's prelude, core and numerics modules only).
#   c_test.sh FILE.sail [profile ...]      default profiles: mini ref
# SRF_RING_KEEP=dir keeps the build there.
# The model directory is the tree's (OPENTSP_MODEL_DIR points it at a copy: the mutation ratchet runs this test on a mutated copy).
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
model="${OPENTSP_MODEL_DIR:-$here/model}"
testfile="${1:?usage: c_test.sh FILE.sail [profile ...]}"; shift
profiles="${*:-mini ref}"
command -v sail >/dev/null || { echo "c_test: sail not found (source the toolchain env.sh)" >&2; exit 2; }
if [ -n "${SRF_RING_KEEP:-}" ]; then work="$SRF_RING_KEEP"; mkdir -p "$work"; else work="$(mktemp -d)"; fi
[ -n "${SRF_RING_KEEP:-}" ] || trap 'find "$work" -mindepth 1 -delete 2>/dev/null; rmdir "$work" 2>/dev/null || true' EXIT
lib="$(cd "$(dirname "$(command -v sail)")/../share/sail/lib" && pwd)"
ln -s "$model/prelude" "$work/prelude"
ln -s "$model/core" "$work/core"
cp "$here/test/formal/$testfile" "$work/test_under_run.sail"
# the modules of the model's project file, in its order
python3 - "$model/tsp.sail_project" "$work/ring.sail_project" <<'PY'
import re, sys
src = re.sub(r"//[^\n]*", "", open(sys.argv[1]).read())
blocks = dict(re.findall(r"(\w+)\s*\{(.*?)\n\}", src, re.S))
out = ["%s {%s\n}\n" % (name, blocks[name]) for name in ("prelude", "core", "numerics")]
out.append("ring {\n  requires prelude, core, numerics\n  files test_under_run.sail\n}\n")
open(sys.argv[2], "w").write("\n".join(out))
PY
(cd "$work" && sail -c --strict-var --strict-bitvector --strict-exponentials --all-modules --c-no-main --c-include tsp_model_if.h -o "$work/model" ring.sail_project) \
  > "$work/sail.log" 2>&1 || { cat "$work/sail.log" >&2; exit 1; }
inc=(-I"$lib" -I"$here/c_emulator" -I"$work"); link=(-lgmp -lz)
if [ -n "${GMP_PREFIX:-}" ]; then
  inc+=(-I"$GMP_PREFIX/include" -I"$GMP_PREFIX/include/x86_64-linux-gnu")
  link=("${GMP_LIBS:-$GMP_PREFIX/lib/libgmp.so}" "${Z_LIBS:--lz}")
fi
cc -O1 -w "${inc[@]}" "$work/model.c" "$here/test/formal/c_test_main.c" "$here/test/unit_tests/stubs.c" "$lib/sail.c" "$lib/rts.c" \
  "$lib/elf.c" "$lib/sail_failure.c" "$lib/sail_config.c" "$lib/cJSON.c" -o "$work/c_test" "${link[@]}"
for profile in $profiles; do
  "$work/c_test" "$here/config/$profile.json"
  echo "c_test ok: $testfile profile=$profile"
done
