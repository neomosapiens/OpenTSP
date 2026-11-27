#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Compile the production numerics and finite-domain cap entry point, using the
# same native C lifecycle/flags as tools/check_transcendental_f32.sh.
set -euo pipefail
spec="$(cd "$(dirname "$0")/../.." && pwd)"
work="${1:?usage: build_numeric_cap.sh WORK [MODEL_DIR]}"
model="${2:-$spec/model}"
model="$(cd "$model" && pwd)"
mkdir -p "$work"
work="$(cd "$work" && pwd)"
lib="$(cd "$(dirname "$(command -v sail)")/../share/sail/lib" && pwd)"
receipt="$spec/test/formal/numeric_cap_build_receipt.py"
"${PYTHON:-python3}" "$receipt" before "$spec" "$model" "$lib" "$work"
ln -sfn "$model" "$work/model"
ln -sfn "$spec/test/formal" "$work/drv"
"${PYTHON:-python3}" - "$model/tsp.sail_project" "$work/cap.sail_project" <<'PY'
import re
import sys
source = re.sub(r"//[^\n]*", "", open(sys.argv[1]).read())
blocks = dict(re.findall(r"(\w+)\s*\{(.*?)\n\}", source, re.S))
out = []
for name in ("prelude", "core", "numerics", "main"):
    body = re.sub(r"^(\s+)([\w/]+\.sail)", r"\1model/\2", blocks[name], flags=re.M)
    out.append("%s {%s\n}\n" % (name, body))
out.append("driver {\n  requires prelude, core, numerics\n  files drv/numeric_cap.sail\n}\n")
open(sys.argv[2], "w").write("\n".join(out))
PY
(cd "$work" && sail -c --strict-var --strict-bitvector --strict-exponentials --all-modules --c-no-main \
  --c-preserve numeric_cap_eval -o "$work/model" cap.sail_project) > "$work/sail.log" 2>&1
inc=(-I"$lib" -I"$spec/c_emulator" -I"$work"); link=(-lgmp -lz)
if [ -n "${GMP_PREFIX:-}" ]; then
  inc+=(-I"$GMP_PREFIX/include" -I"$GMP_PREFIX/include/x86_64-linux-gnu")
  link=("${GMP_LIBS:-$GMP_PREFIX/lib/libgmp.so}" "${Z_LIBS:--lz}")
fi
cc -O2 -w "${inc[@]}" "$work/model.c" "$spec/test/formal/numeric_cap_driver.c" "$spec/test/unit_tests/stubs.c" \
  "$lib/sail.c" "$lib/rts.c" "$lib/elf.c" "$lib/sail_failure.c" "$lib/sail_config.c" "$lib/cJSON.c" \
  -o "$work/numeric_cap_driver.partial" "${link[@]}"
cc --version > "$work/compiler.txt"
sail --version > "$work/generator.txt"
"${PYTHON:-python3}" "$receipt" finish "$spec" "$model" "$lib" "$work" -- \
  cc -O2 -w "${inc[@]}" "$work/model.c" "$spec/test/formal/numeric_cap_driver.c" "$spec/test/unit_tests/stubs.c" \
  "$lib/sail.c" "$lib/rts.c" "$lib/elf.c" "$lib/sail_failure.c" "$lib/sail_config.c" "$lib/cJSON.c" \
  -o "$work/numeric_cap_driver.partial" "${link[@]}"
mv "$work/numeric_cap_driver.partial" "$work/numeric_cap_driver"
echo "numeric_cap native driver: $work/numeric_cap_driver"
