#!/usr/bin/env bash
# Mini $[test] run for the proof-gate mutation ratchet.
# Profiles (arrays only; no backslash continuations):
#   config/mini.json            - kills m_s3 and m_t1 (Permute bare-key assert)
#   config/test/mini_ext.json   - kills m_ws1 / m_be1 / m_bf1 (bfp8)
#   config/test/mini_dot.json   - mini + legal Exp.f16 dotted row (ratchet profile)
# Hold SMT binds model postlude/tick.sail hold_decide via prove_extra hold (postlude project).
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
out="$(mktemp -d)"
cleanup() { find "$out" -mindepth 1 -delete 2>/dev/null || true; rmdir "$out" 2>/dev/null || true; }
trap cleanup EXIT
lib="$(cd "$(dirname "$(command -v sail)")/../share/sail/lib" && pwd)"
sail_log="$out/sail.log"
sail_failed() { echo "unit_mini: sail failed; its output follows" >&2; cat "$sail_log" >&2; exit 1; }
cd "$here/model"
sail_args=(
  -c --strict-var --strict-bitvector --strict-exponentials --all-modules --c-no-main
  --c-include tsp_model_if.h --c-preserve timing_config_error
  -o "$out/model" tsp.sail_project
)
sail "${sail_args[@]}" >"$sail_log" 2>&1 || sail_failed
"${PYTHON:-python3}" "$here/tools/schema_header.py" "$here/config/schema.json" "$out/tsp_config_schema.h"
inc=(-I"$lib" -I"$here/c_emulator" -I"$out"); link=(-lgmp -lz)
if [ -n "${GMP_PREFIX:-}" ]; then
  inc+=(-I"$GMP_PREFIX/include" -I"$GMP_PREFIX/include/x86_64-linux-gnu")
  link=("${GMP_LIBS:-$GMP_PREFIX/lib/libgmp.so}" "${Z_LIBS:--lz}")
fi
cc_srcs=(
  "$out/model.c"
  "$here/test/unit_tests/main_unit_tests.c"
  "$here/c_emulator/tsp_config_check.c"
  "$here/test/unit_tests/stubs.c"
  "$lib/sail.c"
  "$lib/rts.c"
  "$lib/elf.c"
  "$lib/sail_failure.c"
  "$lib/sail_config.c"
  "$lib/cJSON.c"
)
cc -O1 -w "${inc[@]}" "${cc_srcs[@]}" -o "$out/unit_tests" "${link[@]}"
"$out/unit_tests" "$here/config/mini.json"
echo "unit_mini ok: mini"
"$out/unit_tests" "$here/config/test/mini_ext.json"
echo "unit_mini ok: mini_ext"
"$out/unit_tests" "$here/config/test/mini_dot.json"
echo "unit_mini ok: mini_dot"
