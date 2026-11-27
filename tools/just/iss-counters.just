# ISS activity counters and the post-hoc energy model.

# The counters against the reference simulator's trace on five programs, and the unit tests of the counters and the energy table.
iss-counters:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    bash tools/ensure_sim.sh "${TSP_SIM:-target/spec/tsp_sim}"
    cargo build --release -q -p ots-asm -p tsp-iss
    cargo test --release -p tsp-iss --test counters
    python3 tests/perf/test_counters_xcheck.py --sim "${TSP_SIM:-target/spec/tsp_sim}"
    python3 tests/perf/test_counters_types.py
    python3 tests/perf/test_counters_acc_price.py
