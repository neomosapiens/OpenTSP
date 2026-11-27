#!/usr/bin/env bash
# The encoding masks as the Sail model has them (about 3 min: the model is built with the C backend, then probed).
# Builds test/isa_db/isa_query (the model's encdec behind a line protocol, as tools/isa_db_check.sh does) and runs
# test/formal/encoding_from_model.py on it: for every catalogue entry, the bits the model treats as fixed are derived from the
# model by single-bit probes and compared with the match patterns of isa-db/generated/encoding.json.
#   encoding_model.sh [profile]          profile: ref (default) or any config/<profile>.json
# ISA_QUERY=path uses an isa_query that is already built; ISA_QUERY_OUT=path keeps the one this run builds.
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
profile="${1:-ref}"
py="${PYTHON:-python3}"
if [ -n "${ISA_QUERY:-}" ]; then
  exe="$ISA_QUERY"
else
  out="$(mktemp -d)"
  trap 'find "$out" -mindepth 1 -delete 2>/dev/null; rmdir "$out" 2>/dev/null || true' EXIT
  lib="$(cd "$(dirname "$(command -v sail)")/../share/sail/lib" && pwd)"
  sail_log="$out/sail.log"
  (cd "$here/model" && sail -c -O --Oconstant-fold --strict-var --strict-bitvector --strict-exponentials --all-modules --c-no-main \
    --c-include tsp_model_if.h --c-preserve isa_entry32 --c-preserve isa_entry64 --c-preserve isa_timing32 --c-preserve isa_timing64 \
    --c-preserve isa_disasm32 --c-preserve isa_disasm64 --c-preserve isa_roundtrip32 --c-preserve isa_roundtrip64 \
    --c-preserve isa_timing_type32 --c-preserve isa_timing_type64 --c-preserve isa_entry_takes_type \
    --c-preserve timing_config_error --c-preserve isa_entry_class --c-preserve isa_station --c-preserve isa_decode_text \
    -o "$out/model" tsp.sail_project) > "$sail_log" 2>&1 || { echo "encoding_model: sail failed; its output follows" >&2; cat "$sail_log" >&2; exit 1; }
  "$py" "$here/tools/schema_header.py" "$here/config/schema.json" "$out/tsp_config_schema.h"
  inc=(-I"$lib" -I"$here/c_emulator" -I"$out"); link=(-lgmp -lz)
  if [ -n "${GMP_PREFIX:-}" ]; then
    inc+=(-I"$GMP_PREFIX/include" -I"$GMP_PREFIX/include/x86_64-linux-gnu")
    link=("${GMP_LIBS:-$GMP_PREFIX/lib/libgmp.so}" "${Z_LIBS:--lz}")
  fi
  cc -O1 -w "${inc[@]}" "$out/model.c" "$here/test/isa_db/isa_query.c" "$here/c_emulator/tsp_config_check.c" "$here/test/unit_tests/stubs.c" \
    "$lib/sail.c" "$lib/rts.c" "$lib/elf.c" "$lib/sail_failure.c" "$lib/sail_config.c" "$lib/cJSON.c" -o "$out/isa_query" "${link[@]}"
  exe="$out/isa_query"
  [ -z "${ISA_QUERY_OUT:-}" ] || cp "$exe" "$ISA_QUERY_OUT"
fi
"$py" "$here/test/formal/encoding_from_model.py" "$exe" "$here/config/$profile.json"
