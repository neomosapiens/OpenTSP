#!/usr/bin/env bash
# Exhaustive binary16 add, subtract and multiply: all 2^32 operand pairs of each operation, the model's fp_add / fp_sub / fp_mul (Sail,
# C backend, built from the model's prelude and core/numerics) against the host's _Float16 arithmetic. NaN results are compared by
# class (the model's must be the canonical 0x7E00). Every other result is compared bit for bit.
#   f16_exhaustive.sh [add|sub|mul ...]      default: all three, one after the other
# A full operation took about 7,000 CPU-seconds in the reference measurement.
# The cost measured with the eager large-integer ilog2 implementation is obsolete. Run one operation at a time, in the background.
# Environment: JOBS (default: 8) processes per operation; F16_X_MAX=n checks the first n values of x only (a smoke run; default 65536 = exhaustive)
#              OPENTSP_MODEL_DIR  a copy of the model directory (the mutation check points it at a mutated copy)
#              F16_X_INSN=1       record retired instructions using time on macOS or perf on Linux
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
model="${OPENTSP_MODEL_DIR:-$here/model}"
ops="${*:-add sub mul}"
jobs="${JOBS:-8}"
command -v sail >/dev/null || { echo "f16_exhaustive: sail not found (source the toolchain env.sh)" >&2; exit 2; }
work="$(mktemp -d)"
trap 'find "$work" -mindepth 1 -delete 2>/dev/null; rmdir "$work" 2>/dev/null || true' EXIT
lib="$(cd "$(dirname "$(command -v sail)")/../share/sail/lib" && pwd)"
ln -s "$model/prelude" "$work/prelude"
ln -s "$model/core/numerics" "$work/numerics"
cp "$here/test/numerics/f16_exhaustive.sail" "$work/f16_exhaustive.sail"
cat > "$work/f16.sail_project" <<'PROJ'
prelude {
  files
    prelude/prelude.sail,
    prelude/errors.sail,
}
numerics {
  requires prelude
  files numerics/fp.sail
}
driver {
  requires prelude, numerics
  files f16_exhaustive.sail
}
PROJ
(cd "$work" && sail -c -O --Oconstant-fold --strict-var --strict-bitvector --strict-exponentials --all-modules --c-no-main \
  --c-preserve f16_add --c-preserve f16_sub --c-preserve f16_mul -o "$work/model" f16.sail_project) > "$work/sail.log" 2>&1 \
  || { cat "$work/sail.log" >&2; exit 1; }
inc=(-I"$lib" -I"$work"); link=(-lgmp -lz)
if [ -n "${GMP_PREFIX:-}" ]; then
  inc+=(-I"$GMP_PREFIX/include" -I"$GMP_PREFIX/include/x86_64-linux-gnu")
  link=("${GMP_LIBS:-$GMP_PREFIX/lib/libgmp.so}" "${Z_LIBS:--lz}")
fi
cc -O2 -w "${inc[@]}" "$work/model.c" "$here/test/numerics/f16_exhaustive.c" "$lib/sail.c" "$lib/rts.c" \
  "$lib/elf.c" "$lib/sail_failure.c" "$lib/sail_config.c" "$lib/cJSON.c" -o "$work/f16_exhaustive" "${link[@]}"
status=0
xmax="${F16_X_MAX:-65536}"
for op in $ops; do
  start=$SECONDS
  workers=()
  step=$(( (xmax + jobs - 1) / jobs ))
  for ((j = 0; j < jobs; j++)); do
    first=$((j * step)); [ "$first" -lt "$xmax" ] || break
    n=$step; [ $((first + n)) -le "$xmax" ] || n=$((xmax - first))
    if [ -n "${F16_X_INSN:-}" ]; then
      case "$(uname -s)" in
        Darwin)
          nice -n 10 /usr/bin/time -l -o "$work/$op.$j.time" "$work/f16_exhaustive" "$op" "$first" "$n" > "$work/$op.$j" &
          ;;
        Linux)
          LC_ALL=C nice -n 10 perf stat --no-big-num -x, -e instructions -o "$work/$op.$j.time" -- "$work/f16_exhaustive" "$op" "$first" "$n" > "$work/$op.$j" &
          ;;
        *)
          echo "f16_exhaustive: no retired-instruction backend for $(uname -s)" >&2
          exit 1
          ;;
      esac
    else
      nice -n 10 "$work/f16_exhaustive" "$op" "$first" "$n" > "$work/$op.$j" &
    fi
    workers+=("$!")
  done
  for worker in "${workers[@]}"; do
    if ! wait "$worker"; then
      echo "f16_exhaustive: $op: worker $worker failed" >&2
      status=1
    fi
  done
  pairs=0; nans=0; errs=0
  for f in "$work/$op".*[0-9]; do
    if ! read -r st p e extra < <(tail -n 1 "$f") ||
      [[ ! $st =~ ^(ok|errors)$ || ! $p =~ ^[0-9]+$ || ! $e =~ ^[0-9]+$ || -n $extra ]]; then
      echo "f16_exhaustive: $op: invalid worker summary: $f" >&2
      status=1
      continue
    fi
    pairs=$((pairs + p))
    if [ "$st" = ok ]; then
      nans=$((nans + e))
    else
      errs=$((errs + e))
      status=1
    fi
    /usr/bin/grep '^MISMATCH' "$f" | head -3 || true
  done
  insn=""
  if [ -n "${F16_X_INSN:-}" ]; then
    insn=", $("${PYTHON:-python3}" "$here/test/numerics/retired_instructions.py" "$(uname -s)" "$work/$op".*.time) instructions retired"
  fi
  expected=$((xmax * 65536))
  if [ "$pairs" -ne "$expected" ] || [ "$expected" -le 0 ]; then
    echo "f16_exhaustive: $op: $pairs of $expected required pairs completed" >&2
    status=1
  fi
  echo "f16_exhaustive: $op: $pairs pairs, $nans NaN results compared by class, $errs mismatches, $((SECONDS - start)) s$insn"
  [ "$errs" -eq 0 ] || status=1
done
exit $status
