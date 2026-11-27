# Chip numbers above 63, link numbers above 15 and the topology file (opentsp-arch-spec/docs/topology.md section 7).

# The wide chip and link numbers, the topology file and --chips K through tsp_sim and the ISS (identical outputs), a system of 264 chips on the ISS.
iss-wide:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    sim="${TSP_SIM:-target/spec/tsp_sim}"
    bash tools/ensure_sim.sh "$sim"
    cargo build --release -q -p ots-asm -p tsp-iss
    python3 opentsp-arch-spec/test/c2c/wide_directed.py "$sim" target/release/tsp-iss opentsp-arch-spec/config/mini.json opentsp-arch-spec/config/test/c2c_wide.json
