#!/usr/bin/env bash
# Build the Sail-generated TSP emulator (c_emulator/tsp_sim.c around the model's C code).
# usage: build_sim.sh [output-binary] (default: target/spec/tsp_sim at the repository root)
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
bin="${1:-$here/../target/spec/tsp_sim}"
mkdir -p "$(dirname "$bin")"
bin="$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
lib="$(cd "$(dirname "$(command -v sail)")/../share/sail/lib" && pwd)"
cd "$here/model"
sail -c -O --Oconstant-fold --strict-var --strict-bitvector --strict-exponentials --all-modules --c-no-main \
  --c-include tsp_model_if.h --c-preserve tick --c-preserve reset --c-preserve model_quiescent --c-preserve model_next_event --c-preserve model_finish \
  --c-preserve model_csr_read --c-preserve model_csr_write --c-preserve model_inject_unit --c-preserve model_set_start --c-preserve model_ptype_queued --c-preserve model_packet_cycles --c-preserve model_flip_sram --c-preserve model_flip_srf --c-preserve model_flip_iq --c-preserve model_poll_event --c-preserve model_scrub_before --c-preserve model_link_fec --c-preserve model_link_flits --c-preserve model_link_flit_mask --c-preserve model_inject_mxm --c-preserve model_mxm_pending \
  --c-preserve timing_config_error \
  -o "$out/model" tsp.sail_project >"$out/sail.log" 2>&1 || { cat "$out/sail.log" >&2; exit 1; }
# config/schema.json as a C string literal (tsp_config_load checks every config against it).
"${PYTHON:-python3}" "$here/tools/schema_header.py" "$here/config/schema.json" "$out/tsp_config_schema.h"
inc=(-I"$lib" -I"$here/c_emulator" -I"$out"); link=(-lgmp -lz)
if [ -n "${GMP_PREFIX:-}" ]; then
  inc+=(-I"$GMP_PREFIX/include" -I"$GMP_PREFIX/include/x86_64-linux-gnu")
  link=("${GMP_LIBS:-$GMP_PREFIX/lib/libgmp.so}" "${Z_LIBS:--lz}")
fi
cc -O2 -w -D_GNU_SOURCE "${inc[@]}" "$out/model.c" "$here/c_emulator/tsp_sim.c" "$here/c_emulator/tsp_config_check.c" "$lib/sail.c" "$lib/rts.c" "$lib/elf.c" \
  "$lib/sail_failure.c" "$lib/sail_config.c" "$lib/cJSON.c" -o "$bin" "${link[@]}"
