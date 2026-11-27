#!/usr/bin/env bash
# Sampled binary32 add, subtract and multiply: the model's fp_add / fp_sub / fp_mul (Sail, C backend, built from the model's prelude and
# core/numerics) against the host's `float` arithmetic on pairs drawn from four strata (uniform patterns, nearly equal exponents,
# subnormal and tiny operands, near-overflow operands; see f32_sample.c). NaN results are compared by class (the model's must be the
# canonical 0x7FC00000); every other result bit for bit. 2^64 pairs cannot be run: this is a sample, not a proof.
#  .
#   f32_sample.sh [add|sub|mul ...]      default: all three, one after the other
# Environment: JOBS (default: 8) processes per operation, each with its own seed (F32_SEED + job number);
#              F32_PAIRS pairs per stratum and process (default 5000000: 160 million pairs per operation with 8 processes)
#              OPENTSP_MODEL_DIR  a copy of the model directory (a mutation check points it at a mutated copy)
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
model="${OPENTSP_MODEL_DIR:-$here/model}"
ops="${*:-add sub mul}"
jobs="${JOBS:-8}"
command -v sail >/dev/null || { echo "f32_sample: sail not found (source the toolchain env.sh)" >&2; exit 2; }
work="$(mktemp -d)"
trap 'find "$work" -mindepth 1 -delete 2>/dev/null; rmdir "$work" 2>/dev/null || true' EXIT
lib="$(cd "$(dirname "$(command -v sail)")/../share/sail/lib" && pwd)"
ln -s "$model/prelude" "$work/prelude"
ln -s "$model/core/numerics" "$work/numerics"
cp "$here/test/numerics/f32_sample.sail" "$work/f32_sample.sail"
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
  files f32_sample.sail
}
PROJ
(cd "$work" && sail -c -O --Oconstant-fold --strict-var --strict-bitvector --strict-exponentials --all-modules --c-no-main \
  --c-preserve f32_add --c-preserve f32_sub --c-preserve f32_mul -o "$work/model" f16.sail_project) > "$work/sail.log" 2>&1 \
  || { cat "$work/sail.log" >&2; exit 1; }
inc=(-I"$lib" -I"$work"); link=(-lgmp -lz)
if [ -n "${GMP_PREFIX:-}" ]; then
  inc+=(-I"$GMP_PREFIX/include" -I"$GMP_PREFIX/include/x86_64-linux-gnu")
  link=("${GMP_LIBS:-$GMP_PREFIX/lib/libgmp.so}" "${Z_LIBS:--lz}")
fi
cc -O2 -w "${inc[@]}" "$work/model.c" "$here/test/numerics/f32_sample.c" "$lib/sail.c" "$lib/rts.c" \
  "$lib/elf.c" "$lib/sail_failure.c" "$lib/sail_config.c" "$lib/cJSON.c" -o "$work/f32_sample" "${link[@]}"
status=0
count="${F32_PAIRS:-5000000}"      # pairs per stratum and process; four strata
seed="${F32_SEED:-20261119}"
for op in $ops; do
  start=$SECONDS
  workers=()
  for ((j = 0; j < jobs; j++)); do
    nice -n 10 "$work/f32_sample" "$op" $((seed + j)) "$count" > "$work/$op.$j" &
    workers+=("$!")
  done
  for worker in "${workers[@]}"; do
    if ! wait "$worker"; then
      echo "f32_sample: $op: worker $worker failed" >&2
      status=1
    fi
  done
  pairs=0; nans=0; errs=0
  for f in "$work/$op".*; do
    if ! read -r st p e extra < <(tail -n 1 "$f") ||
      [[ ! $st =~ ^(ok|errors)$ || ! $p =~ ^[0-9]+$ || ! $e =~ ^[0-9]+$ || -n $extra ]]; then
      echo "f32_sample: $op: invalid worker summary: $f" >&2
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
  expected=$((4 * jobs * count))
  if [ "$pairs" -ne "$expected" ] || [ "$expected" -le 0 ]; then
    echo "f32_sample: $op: $pairs of $expected required pairs completed" >&2
    status=1
  fi
  echo "f32_sample: $op: $pairs pairs in 4 strata, $nans NaN results compared by class, $errs mismatches, $((SECONDS - start)) s"
  [ "$errs" -eq 0 ] || status=1
done
exit $status
