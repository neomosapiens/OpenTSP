# Serving checks for alternative profiles.

# The serving programs on variant profiles: the capability check (`ProfileNeeds`: a profile that
# cannot hold the programs is refused with the first unmet requirement), then the 2-layer serving step emitted for `test/ref_slow` (the
# geometry of `ref`, other timing) and for derived timing and gap variants, run on the ISS (and once on tsp_sim) against the host
# bit-exact reference. About 5 minutes (measured: 5 min 21 s at load 6); `serving-profiles-full` adds the 18 buckets of the grid (about 9 minutes more).
serving-profiles:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    bash tools/ensure_sim.sh "${TSP_SIM:-target/spec/tsp_sim}"
    cargo build --release -q -p ots-asm -p tsp-iss
    $PYTHON -m pytest -q -m "not slow" tests/showcase/tinystories/test_profile_needs.py tests/showcase/tinystories/test_profile_timing.py tests/showcase/tinystories/test_serve_geometry.py

serving-profiles-full:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    bash tools/ensure_sim.sh "${TSP_SIM:-target/spec/tsp_sim}"
    cargo build --release -q -p ots-asm -p tsp-iss
    $PYTHON -m pytest -q tests/showcase/tinystories/test_profile_needs.py tests/showcase/tinystories/test_profile_timing.py tests/showcase/tinystories/test_serve_geometry.py

# The kernel cookbook (docs/kernel-cookbook.md): the division idioms measured on the ISS
# (the sweep and the edge searches, about 10 s), the golden programs that pin their edges on tsp_sim and the ISS, and the regeneration of every
# listing and table of the document from the kernels (a block that differs fails), and the whole division sweep once more on tsp_sim (the Sail
# model) with the same result. About 3 minutes.
kernel-cookbook:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    bash tools/ensure_sim.sh "${TSP_SIM:-target/spec/tsp_sim}"
    cargo build --release -q -p ots-asm -p tsp-iss
    $PYTHON tests/programs/gen/test_division.py
    $PYTHON tests/programs/gen/test_cookbook.py
    $PYTHON tests/programs/gen/division_domains.py --check --sim "${TSP_SIM:-target/spec/tsp_sim}"
    $PYTHON tests/programs/gen/build.py --check vxm/division_edges
    $PYTHON tests/programs/run.py --sim "${TSP_SIM:-target/spec/tsp_sim}" --iss target/release/tsp-iss vxm/division_edges
