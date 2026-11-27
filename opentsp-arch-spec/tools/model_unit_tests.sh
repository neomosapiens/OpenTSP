#!/usr/bin/env bash
# Build the Sail model's C output with its $[test] functions and run them on every
# profile. Needs sail (0.20.x) on PATH, a C compiler and GMP (set GMP_PREFIX if GMP is
# not installed system-wide).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
lib="$(cd "$(dirname "$(command -v sail)")/../share/sail/lib" && pwd)"
# Sail's diagnostics go to a log that is printed when the build fails (never discarded).
sail_log="$out/sail.log"
sail_failed() { echo "$(basename "$0"): sail failed; its output follows" >&2; cat "$sail_log" >&2; exit 1; }
cd "$here/model"
sail -c --strict-var --strict-bitvector --strict-exponentials --all-modules --c-no-main \
  --c-include tsp_model_if.h --c-preserve timing_config_error -o "$out/model" tsp.sail_project > "$sail_log" 2>&1 || sail_failed
# Generate the runtime config schema used by tsp_config_load.
"${PYTHON:-python3}" "$here/tools/schema_header.py" "$here/config/schema.json" "$out/tsp_config_schema.h"
inc=(-I"$lib" -I"$here/c_emulator" -I"$out"); link=(-lgmp -lz)
if [ -n "${GMP_PREFIX:-}" ]; then
  inc+=(-I"$GMP_PREFIX/include" -I"$GMP_PREFIX/include/x86_64-linux-gnu")
  link=("${GMP_LIBS:-$GMP_PREFIX/lib/libgmp.so}" "${Z_LIBS:--lz}")
fi
cc -O1 -w "${inc[@]}" "$out/model.c" "$here/test/unit_tests/main_unit_tests.c" "$here/c_emulator/tsp_config_check.c" "$here/test/unit_tests/stubs.c" "$lib/sail.c" "$lib/rts.c" \
  "$lib/elf.c" "$lib/sail_failure.c" "$lib/sail_config.c" "$lib/cJSON.c" -o "$out/unit_tests" "${link[@]}"
# The shipped profiles and the test profiles (config/test/): the
# profile-generic tests (test_profile.sail and every test that reads the geometry from the
# configuration) must pass on all of them.
for cfg in "$here"/config/ref.json "$here"/config/mini.json "$here"/config/test/*.json; do
  "$out/unit_tests" "$cfg"
done
# The runner refuses an invalid configuration like tsp_sim.
"${PYTHON:-python3}" "$here/test/config/harness_reject.py" "$out/unit_tests" "$here/config/ref.json" | tail -1
