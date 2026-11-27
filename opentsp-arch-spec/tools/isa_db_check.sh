#!/usr/bin/env bash
# Build test/isa_db/isa_query (the Sail model's encdec/assembly behind a line protocol) and
# run the ISA catalogue cross-check and the reference assembler checks (tools/tsp_as.py:
# round trip in the accept, reject and decode directions; the assembly.md examples and
# program-file rules) on every profile. TSP_SIM=<tsp_sim> also runs the assembly.md sec. 3
# example program on the emulator (tools/model_test.sh does).
# usage: isa_db_check.sh [samples-per-instruction] [seed]
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
lib="$(cd "$(dirname "$(command -v sail)")/../share/sail/lib" && pwd)"
# Sail's diagnostics go to a log that is printed when the build fails (never discarded).
sail_log="$out/sail.log"
sail_failed() { echo "$(basename "$0"): sail failed; its output follows" >&2; cat "$sail_log" >&2; exit 1; }
cd "$here/model"
sail -c -O --Oconstant-fold --strict-var --strict-bitvector --strict-exponentials --all-modules --c-no-main \
  --c-include tsp_model_if.h --c-preserve isa_disasm32 --c-preserve isa_disasm64 \
  --c-preserve isa_roundtrip32 --c-preserve isa_roundtrip64 --c-preserve timing_config_error --c-preserve isa_entry32 --c-preserve isa_entry64 --c-preserve isa_timing32 --c-preserve isa_timing64 --c-preserve isa_entry_class --c-preserve isa_timing_type32 --c-preserve isa_timing_type64 --c-preserve isa_entry_takes_type --c-preserve isa_station --c-preserve isa_decode_text \
  -o "$out/model" tsp.sail_project > "$sail_log" 2>&1 || sail_failed
# config/schema.json for tsp_config_load (every harness checks its config first, ).
"${PYTHON:-python3}" "$here/tools/schema_header.py" "$here/config/schema.json" "$out/tsp_config_schema.h"
inc=(-I"$lib" -I"$here/c_emulator" -I"$out"); link=(-lgmp -lz)
if [ -n "${GMP_PREFIX:-}" ]; then
  inc+=(-I"$GMP_PREFIX/include" -I"$GMP_PREFIX/include/x86_64-linux-gnu")
  link=("${GMP_LIBS:-$GMP_PREFIX/lib/libgmp.so}" "${Z_LIBS:--lz}")
fi
cc -O1 -w "${inc[@]}" "$out/model.c" "$here/test/isa_db/isa_query.c" "$here/c_emulator/tsp_config_check.c" "$here/test/unit_tests/stubs.c" "$lib/sail.c" "$lib/rts.c" \
  "$lib/elf.c" "$lib/sail_failure.c" "$lib/sail_config.c" "$lib/cJSON.c" -o "$out/isa_query" "${link[@]}"
# ISA_QUERY_OUT=<path> keeps the query binary (for the docs example check and the ISS).
if [ -n "${ISA_QUERY_OUT:-}" ]; then cp "$out/isa_query" "$ISA_QUERY_OUT"; fi
# isa_query refuses an invalid configuration like tsp_sim.
"${PYTHON:-python3}" "$here/test/config/harness_reject.py" "$out/isa_query" "$here/config/ref.json" | tail -1
for profile in ref mini test/ext_bf16; do
  echo "profile $profile:"
  "${PYTHON:-python3}" "$here/test/isa_db/check_isa_db.py" "$out/isa_query" "$here/config/$profile.json" "${1:-40}" "${2:-1}"
  "${PYTHON:-python3}" "$here/test/isa_db/check_footprints.py" "$here/config/$profile.json"
  "${PYTHON:-python3}" "$here/test/isa_db/asm_roundtrip.py" "$out/isa_query" "$here/config/$profile.json" "${1:-40}" "${2:-1}"
  "${PYTHON:-python3}" "$here/test/isa_db/check_asm_examples.py" "$out/isa_query" "$here/config/$profile.json" ${TSP_SIM:+"$TSP_SIM"}
done
