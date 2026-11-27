#!/usr/bin/env bash
# The equivalence check of the transcendental helpers that lost their `while` loops (model/core/numerics/transcendental.sail): ln2_iv, exp_small, log_iv
# and isqrt of the model against the loops they replaced (kept in loops_equiv.sail), in the C backend, on edge values and random arguments of every
# size up to the 4096-bit Ziv cap. Also prints the most Newton steps isqrt takes over the arguments checked, which stays below the bound of the
# model's loop (the test fails when a step count exceeds it).
# usage: loops_equiv.sh [N]   (N random arguments of each kind, default 3000; set OPENTSP_MODEL_DIR to check a copy of the model directory)
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
model="${OPENTSP_MODEL_DIR:-$here/model}"
iters="${1:-3000}"
work="$(mktemp -d)"
trap 'find "$work" -mindepth 1 -delete; rmdir "$work"' EXIT
lib="$(cd "$(dirname "$(command -v sail)")/../share/sail/lib" && pwd)"
ln -s "$model/prelude" "$work/prelude"
mkdir -p "$work/core"
ln -s "$model/core/numerics" "$work/numerics"
for f in geometry types; do ln -s "$model/core/$f.sail" "$work/core/$f.sail"; done
sed "s/ITERATIONS/$iters/" "$here/test/export/loops_equiv.sail" > "$work/loops_equiv.sail"
cat > "$work/loops.sail_project" <<'PROJ'
prelude {
  files
    prelude/prelude.sail,
    prelude/errors.sail,
}
core {
  requires prelude
  files
    core/geometry.sail,
    core/types.sail,
}
numerics {
  requires prelude, core
  files
    numerics/fp.sail,
    numerics/lanes.sail,
    numerics/transcendental.sail,
}
equiv {
  requires prelude, core, numerics
  files
    loops_equiv.sail,
}
PROJ
cd "$work"
sail -c --strict-var --strict-bitvector --strict-exponentials --all-modules --c-no-main --c-preserve loops_equiv_run -o equiv loops.sail_project > sail.log 2>&1 || { cat sail.log >&2; exit 1; }
inc=(-I"$lib"); link=(-lgmp -lz)
if [ -n "${GMP_PREFIX:-}" ]; then
  inc+=(-I"$GMP_PREFIX/include" -I"$GMP_PREFIX/include/x86_64-linux-gnu")
  link=("${GMP_LIBS:-$GMP_PREFIX/lib/libgmp.so}" "${Z_LIBS:--lz}")
fi
cc -O1 -w "${inc[@]}" -I"$work" equiv.c "$here/test/export/loops_equiv.c" "$lib/sail.c" "$lib/rts.c" "$lib/elf.c" "$lib/sail_failure.c" "$lib/sail_config.c" "$lib/cJSON.c" -o equiv_bin "${link[@]}"
time ./equiv_bin "$here/config/mini.json"
