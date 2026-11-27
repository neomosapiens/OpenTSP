#!/usr/bin/env bash
# Exhaustive double-precision screening of f32 inputs, followed by selected
# high-precision/model comparisons. This is not exhaustive Sail evaluation or a proof
# that libm never misses a hard input. NOT part of model-test: a full run
# takes about 8 minutes on a 12-thread machine (see the numerical-screening section of validation.md).
#
#   1. build   the model's numerics (prelude, core, core/numerics) to C with sail and link
#              the Sail-evaluator driver (test/numerics/transcendental_f32_driver.{sail,c}),
#   2. sweep   every f32 bit pattern for the four functions in double precision
#              (test/numerics/transcendental_f32_sweep.c) and flag the "hard" inputs: the
#              double value lies within 2^-40 (relative) of an f32 rounding boundary,
#   3. recheck the hard inputs, every special/edge class and a seeded random sample with
#              mpmath (200 bits, more where needed): correctly rounded result and distance
#              of the exact value to the nearest rounding boundary
#              (test/numerics/transcendental_f32_recheck.py),
#   4. compare the model's fp_tr_value(op, FMT32, x) with the mpmath result on all of them,
#   5. report  per function: hard inputs, closest input and its distance, Sail evaluations
#              and mismatches. Exit status non-zero on any failure.
#
# usage: tools/check_transcendental_f32.sh [--quick] [--tree DIR] [--work DIR] [--threads N]
#                                          [--samples N] [--seed N] [--keep]
#   --quick      sweep every 4096th bit pattern (plus all specials) and a small sample: ~2 min
#   --tree DIR   the opentsp-arch-spec directory whose model/ is built (default: this one);
#                the check programs themselves always come from this script's checkout
#   --work DIR   keep all intermediate files and results in DIR (default: a temp dir, removed)
#   --threads N  sweep threads, mpmath processes and Sail processes, at most 12 (default 12)
#   --samples N  random inputs per function and kind, uniform and stratified (full: 200000)
#   --seed N     seed of the random sample (default 20261112)
# Needs sail 0.20.x on PATH, a C compiler, GMP (set GMP_PREFIX / GMP_LIBS / Z_LIBS as for
# model_unit_tests.sh if not installed system-wide), python3 with mpmath. Heavy steps run at
# nice 19.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
tree="$here"
work=""
keep=0
quick=0
threads=12
samples=""
seed=20261112
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) quick=1 ;;
    --tree) tree="$(cd "$2" && pwd)"; shift ;;
    --work) work="$2"; keep=1; shift ;;
    --threads) threads="$2"; shift ;;
    --samples) samples="$2"; shift ;;
    --seed) seed="$2"; shift ;;
    --keep) keep=1 ;;
    -h|--help) sed -n 2,35p "$0"; exit 0 ;;
    *) echo "check_transcendental_f32.sh: unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[ "$threads" -ge 1 ] && [ "$threads" -le 12 ] || { echo "--threads must be 1..12" >&2; exit 2; }
if [ "$quick" = 1 ]; then stride=4096; : "${samples:=3000}"; else stride=1; : "${samples:=200000}"; fi
py="${PYTHON:-python3}"
"$py" -c 'import mpmath' 2>/dev/null || { echo "check_transcendental_f32.sh: python3 with mpmath is required" >&2; exit 2; }
command -v sail >/dev/null || { echo "check_transcendental_f32.sh: sail not on PATH" >&2; exit 2; }
num="$here/test/numerics"
if [ -z "$work" ]; then
  work="$(mktemp -d)"
  trap '[ "$keep" = 1 ] || rm -rf "$work"' EXIT
else
  mkdir -p "$work"
fi
work="$(cd "$work" && pwd)"
lib="$(cd "$(dirname "$(command -v sail)")/../share/sail/lib" && pwd)"
start=$SECONDS
stage() { echo "[$(( SECONDS - start ))s] $*"; }

# ---- 1. build ---------------------------------------------------------------------------
stage "build: model numerics from $tree/model + Sail evaluator driver"
# Sail resolves project paths relative to the project file and rejects '-' in them: link the
# model and the check's driver under short names in the work directory.
ln -sfn "$tree/model" "$work/model"
ln -sfn "$num" "$work/drv"
"$py" - "$tree/model/tsp.sail_project" "$work/f1.sail_project" <<'EOF'
import re, sys
src = open(sys.argv[1]).read()
src = re.sub(r"//[^\n]*", "", src)
blocks = dict(re.findall(r"(\w+)\s*\{(.*?)\n\}", src, re.S))
out = []
for name in ("prelude", "core", "numerics", "main"):
    body = blocks[name]
    body = re.sub(r"^(\s+)([\w/]+\.sail)", r"\1model/\2", body, flags=re.M)
    out.append("%s {%s\n}\n" % (name, body))
out.append("driver {\n  requires prelude, core, numerics\n  files\n    drv/transcendental_f32_driver.sail,\n}\n")
open(sys.argv[2], "w").write("\n".join(out))
EOF
( cd "$work" && sail -c --strict-var --strict-bitvector --strict-exponentials --all-modules --c-no-main \
    --c-preserve tf32_eval --c-preserve tf32_level --c-preserve tf32_acc -o "$work/model" f1.sail_project 2>"$work/sail.log" ) \
  || { cat "$work/sail.log" >&2; echo "sail build failed" >&2; exit 1; }
inc=(-I"$lib" -I"$tree/c_emulator" -I"$work"); link=(-lgmp -lz)
if [ -n "${GMP_PREFIX:-}" ]; then
  inc+=(-I"$GMP_PREFIX/include" -I"$GMP_PREFIX/include/x86_64-linux-gnu")
  link=("${GMP_LIBS:-$GMP_PREFIX/lib/libgmp.so}" "${Z_LIBS:--lz}")
fi
cc -O2 -w "${inc[@]}" "$work/model.c" "$num/transcendental_f32_driver.c" "$tree/test/unit_tests/stubs.c" \
  "$lib/sail.c" "$lib/rts.c" "$lib/elf.c" "$lib/sail_failure.c" "$lib/sail_config.c" "$lib/cJSON.c" \
  -o "$work/sail_driver" "${link[@]}"
cc -O2 -ffp-contract=off -fno-fast-math -pthread -Wall -o "$work/sweep" "$num/transcendental_f32_sweep.c" -lm

# Independent sampled audit of the exact C functions used by the screen.
"$py" "$num/transcendental_filter_audit.py" "$work/sweep" "$work/filter_audit.json"

# ---- 2. sweep ---------------------------------------------------------------------------
stage "sweep: 4 functions x 2^32 patterns / stride $stride in double precision, $threads threads"
nice -n 19 "$work/sweep" "$threads" "$stride" "$work/hard.txt" "$work/sweep_stats.txt"
cat "$work/sweep_stats.txt"

# ---- 3. recheck -------------------------------------------------------------------------
stage "recheck: hard inputs, special/edge classes and a random sample with mpmath"
nice -n 19 "$py" "$num/transcendental_f32_recheck.py" jobs --hard "$work/hard.txt" --stride "$stride" \
  --seed "$seed" --samples "$samples" --out "$work" --procs "$threads"

# ---- 4. Sail evaluator ------------------------------------------------------------------
stage "compare: the model's fp_tr_value on every job, $threads processes"
grep '^A ' "$work/jobs.txt" > "$work/jobs_acc.txt"
grep '^E ' "$work/jobs.txt" > "$work/jobs_eval.txt"
rm -f "$work"/shard_*
split -n "l/$threads" -d "$work/jobs_eval.txt" "$work/shard_"
pids=(); shards=()
for s in "$work"/shard_*; do
  [ -s "$s" ] || continue
  shards+=("$s")
  nice -n 19 "$work/sail_driver" "$tree/config/ref.json" "$s" > "$s.out" 2>&1 &
  pids+=($!)
done
rc=0
for i in "${!pids[@]}"; do
  st=0; wait "${pids[$i]}" || st=$?
  # exit 1 = mismatches (reported below); anything else is a crash
  if [ "$st" -gt 1 ]; then echo "Sail driver crashed on ${shards[$i]} (status $st)" >&2; tail -5 "${shards[$i]}.out" >&2; rc=1; fi
done
nice -n 19 "$work/sail_driver" "$tree/config/ref.json" "$work/jobs_acc.txt" > "$work/acc.out" 2>&1 \
  || { echo "Sail driver crashed on the accuracy probes" >&2; tail -5 "$work/acc.out" >&2; rc=1; }

# ---- 5. report --------------------------------------------------------------------------
stage "report"
outs=("$work"/shard_*.out "$work/acc.out")
st=0
"$py" "$num/transcendental_f32_recheck.py" report --dir "$work" --sail "${outs[@]}" \
  --sweep-stats "$work/sweep_stats.txt" | tee "$work/report.txt" || st=$?
stage "done"
[ "$keep" = 1 ] && echo "results kept in $work"
[ "$rc" = 0 ] && [ "$st" = 0 ]
