#!/usr/bin/env bash
# Full differential run of the Rust numerics against the Sail model's own functions.
#
#   tools/run_differential.sh [--threads N] [--screen-work DIR] [SET ...]
#
# SET: i8 (all i8 transcendentals), f16tr (all f16 inputs of the four transcendentals), f32tr
# (edge classes and a random sample of f32 inputs; with --screen-work DIR also every input of the
# exhaustive f32 check's work directory, opentsp-arch-spec/tools/check_transcendental_f32.sh
# --work DIR), kernels (about 13.5 million inputs of the binary, unary, comparison and
# conversion kernels: all 8-bit operand pairs, all 8- and 16-bit conversion sources, random and
# edge inputs of the wider types). Default: i8 f16tr f32tr kernels.
#
# For each set, tools/gen_jobs.py writes the job file, tools/sail_dump.sh has the Sail model
# evaluate it into target/tsp-numerics/sail/SET.txt, and `cargo test --release -p tsp-numerics
# --test differential sail_dumps_match -- --ignored` compares every line with the Rust code. Needs the
# toolchain of tools/sail_dump.sh. Takes a few minutes (kernels: the longest).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../../.." && pwd)"
threads=12
screen_work=""
sets=()
while [ $# -gt 0 ]; do
  case "$1" in
    --threads) threads="$2"; shift ;;
    --screen-work) screen_work="$2"; shift ;;
    -h|--help) sed -n 2,17p "$0"; exit 0 ;;
    *) sets+=("$1") ;;
  esac
  shift
done
[ ${#sets[@]} -gt 0 ] || sets=(i8 f16tr f32tr kernels)
py="${PYTHON:-python3}"
out="$root/target/tsp-numerics/sail"
mkdir -p "$out"
rm -f "$out"/*.txt
for s in "${sets[@]}"; do
  jobs="$(mktemp)"
  if [ "$s" = f32tr ] && [ -n "$screen_work" ]; then "$py" "$here/gen_jobs.py" "$s" "$jobs" --screen-work "$screen_work"
  else "$py" "$here/gen_jobs.py" "$s" "$jobs"; fi
  "$here/sail_dump.sh" --jobs "$jobs" --out "$out/$s.txt" --threads "$threads"
  rm -f "$jobs"
done
cd "$root"
TSP_NUMERICS_SAIL_DUMP="$out" cargo test --release -p tsp-numerics --test differential sail_dumps_match -- --ignored --nocapture
