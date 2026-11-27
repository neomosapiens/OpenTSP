#!/usr/bin/env bash
# Exhaustive 32-bit encoding check: every one of the 2^32 words through the Sail model's
# encdec32 (C backend) against the catalogue matchers (test/isa_db/exhaustive32.c).
# Runs in parallel over ${JOBS:-available CPUs} ranges on the ref profile (about 1.2 us per word).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
lib="$(cd "$(dirname "$(command -v sail)")/../share/sail/lib" && pwd)"
"${PYTHON:-python3}" "$here/test/isa_db/test_exhaustive_tables.py"
# Sail's diagnostics go to a log that is printed when the build fails (never discarded).
sail_log="$out/sail.log"
sail_failed() { echo "$(basename "$0"): sail failed; its output follows" >&2; cat "$sail_log" >&2; exit 1; }
"${PYTHON:-python3}" "$here/test/isa_db/gen_exhaustive_tables.py" "$here/isa-db/generated/encoding.json" "$out/exhaustive_tables.h" "$here/config/ref.json"
cd "$here/model"
sail -c -O --Oconstant-fold --strict-var --strict-bitvector --strict-exponentials --all-modules --c-no-main \
  --c-include tsp_model_if.h --c-preserve isa_roundtrip32 --c-preserve isa_decodes32 --c-preserve timing_config_error -o "$out/model" tsp.sail_project > "$sail_log" 2>&1 || sail_failed
# config/schema.json for tsp_config_load before model initialization.
"${PYTHON:-python3}" "$here/tools/schema_header.py" "$here/config/schema.json" "$out/tsp_config_schema.h"
inc=(-I"$lib" -I"$here/c_emulator" -I"$out"); link=(-lgmp -lz)
if [ -n "${GMP_PREFIX:-}" ]; then
  inc+=(-I"$GMP_PREFIX/include" -I"$GMP_PREFIX/include/x86_64-linux-gnu")
  link=("${GMP_LIBS:-$GMP_PREFIX/lib/libgmp.so}" "${Z_LIBS:--lz}")
fi
cc -O2 -w -DVLANES=320ULL -DACC_DEPTH=2ULL "${inc[@]}" "$out/model.c" "$here/test/isa_db/exhaustive32.c" "$here/c_emulator/tsp_config_check.c" "$here/test/unit_tests/stubs.c" "$lib/sail.c" "$lib/rts.c" \
  "$lib/elf.c" "$lib/sail_failure.c" "$lib/sail_config.c" "$lib/cJSON.c" -o "$out/exhaustive32" "${link[@]}"
# exhaustive32 refuses an invalid configuration like tsp_sim; count 0 words.
"${PYTHON:-python3}" "$here/test/config/harness_reject.py" "$out/exhaustive32" "$here/config/ref.json" 0 0 | tail -1
jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN)}"
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "exhaustive32: JOBS must be a positive integer" >&2; exit 2; }
total=$((1 << 32))
step=$(( (total + jobs - 1) / jobs ))
workers=()
counts=()
for ((j = 0; j < jobs; j++)); do
  first=$((j * step))
  [ "$first" -lt "$total" ] || break
  n=$(( first + step > total ? total - first : step ))
  "$out/exhaustive32" "$here/config/ref.json" "$first" "$n" > "$out/part$j" &
  workers+=("$!")
  counts+=("$n")
done
failed=0
dec=0; err=0; examined=0
for j in "${!workers[@]}"; do
  summary="$out/part$j"
  if ! wait "${workers[$j]}"; then
    echo "exhaustive32: worker $j failed" >&2
    failed=1
  fi
  grep -v "^ok\|^errors" "$summary" | head -5 || true
  if ! read -r st d e n extra < <(tail -n 1 "$summary") ||
    [[ ! $st =~ ^(ok|errors)$ || ! $d =~ ^[0-9]+$ || ! $e =~ ^[0-9]+$ || ! $n =~ ^[0-9]+$ || -n $extra ]]; then
    echo "exhaustive32: worker $j produced an invalid summary" >&2
    failed=1
    continue
  fi
  if [ "$n" -ne "${counts[$j]}" ] || [ "$d" -gt "$n" ]; then
    echo "exhaustive32: worker $j examined $n of ${counts[$j]} required words" >&2
    failed=1
  fi
  [ "$st" = ok ] || failed=1
  dec=$((dec + d)); err=$((err + e)); examined=$((examined + n))
done
echo "exhaustive32: $examined words, $dec decode, $err errors"
[ "$failed" -eq 0 ] && [ "$examined" -eq "$total" ] && [ "$err" -eq 0 ]
