#!/usr/bin/env bash
# The output of `tsp-iss run --trace --lint` on the benchmark inputs of `tools/iss_bench_prep.py`, as a sha256 per
# bucket with the cycle and tick counts: an optimisation of the run loop must leave every line unchanged.
# Usage: tools/iss_bench_check.sh [-d DIR] [-b BIN] [BUCKET ...]
set -euo pipefail
dir="${ISS_BENCH_DIR:-/tmp/iss-bench}"
bin=target/release/tsp-iss
while getopts "d:b:" o; do
    case "$o" in
        d) dir=$OPTARG ;;
        b) bin=$OPTARG ;;
        *) exit 2 ;;
    esac
done
shift $((OPTIND - 1))
buckets=("$@")
[ ${#buckets[@]} -gt 0 ] || buckets=(P1_1_64 P8_4_128)
stats=$(mktemp)
trap 'rm -f "$stats"' EXIT
for b in "${buckets[@]}"; do
    : > "$stats"
    sum=$("$bin" run --profile ref --trace --lint --weights "$dir/$b.safetensors" --stats="$stats" "$dir/$b.elf" | shasum -a 256 | cut -d' ' -f1)
    python3 -c "
import json
d = json.load(open('$stats'))
print(f\"$b: trace sha256 $sum, {d['last_cycle']} cycles, {d['ticks']} ticks, {d['dispatches']} dispatches, {d['diagnostics']} diagnostics\")"
done
