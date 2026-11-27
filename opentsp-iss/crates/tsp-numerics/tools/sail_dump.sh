#!/usr/bin/env bash
# Evaluate the Sail model's numerics (core/numerics) on a job list, for the Rust differential tests.
#
#   tools/sail_dump.sh --jobs JOBS.txt --out OUT.txt [--work DIR] [--threads N] [--spec DIR]
#
# Builds the model's numerics to C with sail (once per DIR; rebuilt when a model file changes), links
# the driver tools/sail_dump_driver.c (wrappers in tools/sail_dump.sail), runs it on the job list in
# N shards and writes "<job> = <result>" lines to OUT.txt in job order. Job format: see
# tools/sail_dump_driver.c; tools/gen_jobs.py writes job files. The same recipe as
# opentsp-arch-spec/tools/check_transcendental_f32.sh (the f32 screening check); nothing under
# opentsp-arch-spec is modified.
#
# Needs sail 0.20.x on PATH, a C compiler, GMP (GMP_PREFIX / GMP_LIBS / Z_LIBS as for the model
# checks if not installed system-wide) and python3.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../../.." && pwd)"
spec="$root/opentsp-arch-spec"
work="${TMPDIR:-/tmp}/tsp-numerics-sail"
threads=8
jobs=""; out=""
while [ $# -gt 0 ]; do
  case "$1" in
    --jobs) jobs="$2"; shift ;;
    --out) out="$2"; shift ;;
    --work) work="$2"; shift ;;
    --threads) threads="$2"; shift ;;
    --spec) spec="$(cd "$2" && pwd)"; shift ;;
    -h|--help) sed -n 2,16p "$0"; exit 0 ;;
    *) echo "sail_dump.sh: unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[ -n "$jobs" ] && [ -n "$out" ] || { echo "sail_dump.sh: --jobs and --out are required" >&2; exit 2; }
command -v sail >/dev/null || { echo "sail_dump.sh: sail not on PATH" >&2; exit 2; }
py="${PYTHON:-python3}"
mkdir -p "$work"; work="$(cd "$work" && pwd)"
lib="$(cd "$(dirname "$(command -v sail)")/../share/sail/lib" && pwd)"

# ---- build (skipped when the model's files and the driver are unchanged) ---------------------------
stamp="$(cat "$spec"/model/prelude/*.sail "$spec"/model/core/*.sail "$spec"/model/core/numerics/*.sail \
  "$spec/model/tsp.sail_project" "$here/sail_dump.sail" "$here/sail_dump_driver.c" | sha256sum | cut -d' ' -f1)"
if [ ! -x "$work/sail_dump_driver" ] || [ "$(cat "$work/stamp" 2>/dev/null)" != "$stamp" ]; then
  echo "sail_dump.sh: building the model's numerics with sail" >&2
  ln -sfn "$spec/model" "$work/model"
  ln -sfn "$here" "$work/drv"
  "$py" - "$spec/model/tsp.sail_project" "$work/dump.sail_project" <<'PYEOF'
import re, sys
src = open(sys.argv[1]).read()
src = re.sub(r"//[^\n]*", "", src)
blocks = dict(re.findall(r"(\w+)\s*\{(.*?)\n\}", src, re.S))
out = []
for name in ("prelude", "core", "numerics", "main"):
    body = blocks[name]
    body = re.sub(r"^(\s+)([\w/]+\.sail)", r"\1model/\2", body, flags=re.M)
    out.append("%s {%s\n}\n" % (name, body))
out.append("driver {\n  requires prelude, core, numerics\n  files\n    drv/sail_dump.sail,\n}\n")
open(sys.argv[2], "w").write("\n".join(out))
PYEOF
  keep=""
  for f in dmp_tr dmp_i8 dmp_bin dmp_un dmp_cmp dmp_cvt dmp_conv dmp_anybit; do keep="$keep --c-preserve $f"; done
  ( cd "$work" && sail -c --strict-var --strict-bitvector --strict-exponentials --all-modules --c-no-main \
      $keep -o "$work/model" dump.sail_project 2>"$work/sail.log" ) \
    || { cat "$work/sail.log" >&2; echo "sail build failed" >&2; exit 1; }
  inc=(-I"$lib" -I"$spec/c_emulator" -I"$work"); link=(-lgmp -lz)
  if [ -n "${GMP_PREFIX:-}" ]; then
    inc+=(-I"$GMP_PREFIX/include" -I"$GMP_PREFIX/include/x86_64-linux-gnu")
    link=("${GMP_LIBS:-$GMP_PREFIX/lib/libgmp.so}" "${Z_LIBS:--lz}")
  fi
  cc -O2 -w "${inc[@]}" "$work/model.c" "$here/sail_dump_driver.c" "$spec/test/unit_tests/stubs.c" \
    "$lib/sail.c" "$lib/rts.c" "$lib/elf.c" "$lib/sail_failure.c" "$lib/sail_config.c" "$lib/cJSON.c" \
    -o "$work/sail_dump_driver" "${link[@]}"
  echo "$stamp" > "$work/stamp"
fi

# ---- run -----------------------------------------------------------------------------------------
rm -f "$work"/shard_*
split -n "l/$threads" -d "$jobs" "$work/shard_"
pids=(); shards=()
for s in "$work"/shard_*; do
  [ -s "$s" ] || continue
  shards+=("$s")
  nice -n 10 "$work/sail_dump_driver" "$spec/config/ref.json" "$s" > "$s.out" 2> "$s.err" &
  pids+=($!)
done
rc=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "sail driver failed on ${shards[$i]}" >&2; tail -3 "${shards[$i]}.err" >&2; rc=1; }
done
[ "$rc" = 0 ] || exit 1
: > "$out"
for s in "${shards[@]}"; do cat "$s.out" >> "$out"; done
rm -f "$work"/shard_*
echo "sail_dump.sh: $(wc -l < "$out") results in $out" >&2
