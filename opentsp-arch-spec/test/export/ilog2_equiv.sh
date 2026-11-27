#!/usr/bin/env bash
# The equivalence check of ilog2: both forms of the model's loop-free ilog2 (model/core/numerics/fp.sail; the nested one of every
# build and the flat one of the proof build, -D OPENTSP_PROOF_BUILD) against the loop it replaced (kept in ilog2_equiv.sail), in the C backend. Checks every power of two from 2^0 to 2^8300 with its four neighbours and
# N random integers (default 1000000; the recorded run used 100000000).
# usage: ilog2_equiv.sh [N]   (set OPENTSP_MODEL_DIR to check a copy of the model directory)
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
model="${OPENTSP_MODEL_DIR:-$here/model}"
iters="${1:-1000000}"
work="$(mktemp -d)"
trap 'find "$work" -mindepth 1 -delete; rmdir "$work"' EXIT
lib="$(cd "$(dirname "$(command -v sail)")/../share/sail/lib" && pwd)"
for dir in prelude; do ln -s "$model/$dir" "$work/$dir"; done
ln -s "$model/core/numerics" "$work/numerics"
sed "s/ITERATIONS/$iters/" "$here/test/export/ilog2_equiv.sail" > "$work/ilog2_equiv.sail"
cat > "$work/ilog2.sail_project" <<'PROJ'
prelude {
  files
    prelude/prelude.sail,
    prelude/errors.sail,
}
numerics {
  requires prelude
  files
    numerics/fp.sail,
}
equiv {
  requires prelude, numerics
  files
    ilog2_equiv.sail,
}
PROJ
cd "$work"
inc=(-I"$lib"); link=(-lgmp -lz)
if [ -n "${GMP_PREFIX:-}" ]; then
  inc+=(-I"$GMP_PREFIX/include" -I"$GMP_PREFIX/include/x86_64-linux-gnu")
  link=("${GMP_LIBS:-$GMP_PREFIX/lib/libgmp.so}" "${Z_LIBS:--lz}")
fi
for form in nested flat; do
  define=(); [ "$form" = flat ] && define=(-D OPENTSP_PROOF_BUILD)
  echo "ilog2_equiv: the $form form"
  sail -c --strict-var --strict-bitvector --strict-exponentials --all-modules ${define[@]+"${define[@]}"} -o equiv ilog2.sail_project > sail.log 2>&1 || { cat sail.log >&2; exit 1; }
  cc -O1 -w "${inc[@]}" equiv.c "$lib/sail.c" "$lib/rts.c" "$lib/elf.c" "$lib/sail_failure.c" "$lib/sail_config.c" "$lib/cJSON.c" -o equiv_bin "${link[@]}"
  time ./equiv_bin
done
