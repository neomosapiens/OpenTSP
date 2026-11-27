# Chip performance model: derived peaks and roofline, derived profiles, model mapping. Imported by the root justfile
# with `import 'tools/just/perf-model.just'`. The install-bound agreement tests run published_bounds.py on tsp_sim, and the activity-counter
# test runs tsp-iss, so the recipe builds both when they are missing or older than their sources.

# The tests of the performance model (derive_peaks, derive_profile, model_map, the activity counters' field names) on the simulator and the
# ISS. The recipe checks the profile, timing and counter contracts.
perf-model:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    export TSP_SIM="${TSP_SIM:-$PWD/target/spec/tsp_sim}"
    bash tools/ensure_sim.sh "$TSP_SIM"
    cargo build --release -q -p ots-asm -p tsp-iss
    python3 tests/perf/make_profiles.py --check
    "${PYTHON:-python3}" tests/perf/test_derive_peaks.py
    "${PYTHON:-python3}" tests/perf/test_derive_profile.py
    "${PYTHON:-python3}" tests/perf/test_set_typed_row.py
    "${PYTHON:-python3}" tests/perf/test_model_map.py

# Peaks of a profile at an assumed clock: `just perf-peaks opentsp-arch-spec/config/ref.json 900`.
perf-peaks profile="opentsp-arch-spec/config/ref.json" clock="900":
    ${PYTHON:-python3} opentsp-arch-spec/tools/perf/derive_peaks.py {{profile}} --clock-mhz {{clock}}
