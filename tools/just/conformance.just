# Conformance checks for the assembler and simulator.

# Golden-vector mutation differential: 400 mutated vectors run on tsp_sim and tsp-iss, exit code, stdout and stderr equal
# (timeouts counted and bounded). The runs take the start cycles of `starts` in turn (hex, comma separated: 0, 2^62, 2^63 and
# 2^64 - 2^20 by default), so the fault paths above 2^63 are compared too. About 25 s for 400 runs on 3 jobs (500 runs: 30 s, load 6).
# `just iss-vecmut 3 800` runs seed 3 with 800 runs; `just iss-vecmut 3 800 0` keeps every run at start cycle 0.
iss-vecmut seed="1" count="400" starts="0,4000000000000000,8000000000000000,fffffffffff00000":
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    bash tools/ensure_sim.sh "${TSP_SIM:-target/spec/tsp_sim}"
    cargo build --release -q -p tsp-iss
    python3 opentsp-sw-toolchain/tests/iss_vecmut.py --seed {{seed}} --count {{count}} --starts {{starts}} --sim "${TSP_SIM:-target/spec/tsp_sim}" --iss target/release/tsp-iss

# The golden vectors of both profiles compared with site observation (exit code, stdout, stderr; timeouts counted).
iss-sites:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    bash tools/ensure_sim.sh "${TSP_SIM:-target/spec/tsp_sim}"
    cargo build --release -q -p tsp-iss
    python3 tools/iss_compare_sites.py --sail "${TSP_SIM:-target/spec/tsp_sim}" --iss target/release/tsp-iss

# The golden vectors of the test profiles (ref_slow ref_hw ref_small mid_min link_tight fetch_alt c2c_tx_delayed mini_small ref_patrol convert_src, zz_typed, …):
# generated with tsp_sim for each profile (tools/gen_model_vectors.py, into a scratch directory) and replayed in-process with
# `tsp-iss vectors <file> --config <profile>`, so the ISS is compared with the model on the variants and not only on `ref` and `mini`.
# One verdict line at the end. Takes about 10 minutes (measured: 9 min 54 s from the start of `just iss-vectors-variants` at load 9 to 16, 5,519 vectors, the build included); run it in the background with a log.
iss-vectors-variants:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    sim="${TSP_SIM:-target/spec/tsp_sim}"
    bash tools/ensure_sim.sh "$sim"
    cargo build --release -q -p tsp-iss
    cfg=opentsp-arch-spec/config
    scratch="$(mktemp -d)"
    trap 'python3 -c "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)" "$scratch"' EXIT
    total=0
    profiles=0
    for p in ref_slow ref_hw ref_small mid_min link_tight fetch_alt c2c_tx_delayed mini_small ref_patrol ext_bf16 mini_ext ref_ext convert_src zz_typed all_ext mini_dot; do
        python3 opentsp-arch-spec/tools/gen_model_vectors.py "$sim" "$cfg/test/$p.json" "$scratch/$p.json" | grep '^golden vectors' 
        line="$(target/release/tsp-iss vectors "$scratch/$p.json" --config "$cfg/test/$p.json" | tail -1)"
        echo "$p: $line"
        total=$((total + $(echo "$line" | sed -E 's/^([0-9]+) vectors.*/\1/')))
        profiles=$((profiles + 1))
    done
    echo "iss-vectors-variants: $profiles profiles, $total vectors replayed by the ISS, all pass"
