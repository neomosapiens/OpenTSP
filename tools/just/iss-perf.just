# ISS benchmarking and platform checks.
# without the import, run them as `just --justfile tools/just/iss-perf.just --working-directory . <recipe>`.

# The schedule cache and the replay tests need a replay build (`--features replay`, off by default); the cache unit tests
# run without it. The list is the replay tests of the ISS, the platform and the `--stats` document, the parametrised replay (its
# schedule, the shapes of a platform with their budget, the SRAM journal) and the high-start-cycle ECC test. `just test` runs it
# after the workspace tests, which compile the feature-gated files to zero tests; `tools/check_gates_wired.py` fails for a
# feature-gated test file that no recipe runs with its feature.
iss-replay-test:
    cargo test --release -p tsp-iss --features replay --no-fail-fast --lib --test replay_cache --test replay --test replay_compiled --test replay_default --test replay_param --test replay_platform --test sram_journal --test platform_replay --test platform --test stats --test ecc_high_cycle --test time_boundaries

# The real bucket (default P8_4_128; inputs from `python3 tools/iss_bench_prep.py`, directory ISS_BENCH_DIR, default
# /tmp/iss-bench): the clean run served from the cache, then 40 single-bit faults of its words at cycle 0, each compared
# with the normal driver; the ones that change the fetched text fall back to the normal driver. Takes about two minutes.
iss-replay-bucket:
    cargo test -p tsp-iss --features replay --test replay_cache -- --ignored --nocapture

# Interleaved A/B against ISS_PERF_BASE_DIR, a distinct source directory.
# Its ISS_PERF_BASE_PYTHON must import that baseline's own editable packages.
# The baseline build cache is content/toolchain bound; fixture generation uses the canonical model cache.
# ISS_PERF_PAIRS and ISS_PERF_GATE/SMALL/INSN_GATE select repeats and thresholds.
iss-perf-gate: iss-perf-test
    bash tools/iss_perf_gate.sh

# The instruction leg of the gate alone: retired instructions, the least of three runs per program and side, the tree over the baseline (limit
# ISS_PERF_INSN_GATE, default 1.01). Load moves the count by a few tenths of a percent (the least of three drops most of it), so a ratio within 0.5 %
# of the limit is read together with interleaved repeats; run by `iss-diff`.
iss-perf-insn:
    ISS_PERF_INSN_ONLY=1 bash tools/iss_perf_gate.sh

# The unit test of the gate's verdict (synthetic timings, no binary): a regression in any one program must fail it.
iss-perf-test:
    python3 tests/iss_perf/test_iss_ab_overhead.py

# A PGO build of the ISS trained on the coverage programs and the P1 and P8 buckets (target/iss-pgo/tsp-iss; ). Install the Rust LLVM
# tools first (`rustup component add llvm-tools-preview`). The binary is for the CPU it was built on; `tools/iss_build_pgo.sh --path`
# prints the one a serving recipe should run (the PGO binary when it is newer than the ISS sources, else the release one).
iss-pgo:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    tools/iss_build_pgo.sh --corpus "${ISS_BENCH_DIR:-/tmp/iss-bench}"

# The stored benchmark baseline that `tests/benchmark_baseline.rs` compares B4 with: run it on a quiet host and commit the file with the change that
# moves it. It records the commit, the load and the host next to the numbers.
iss-perf-baseline:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    cargo build --release -q -p tsp-iss
    out="$(target/release/tsp-iss bench --repeat 3 --elements)"
    OUT="$out" python3 - <<'PY'
    import hashlib, json, os, platform
    from pathlib import Path
    rows = {}
    for line in os.environ["OUT"].splitlines():
        name, *kv = line.split()
        rows[name] = dict(p.split("=", 1) for p in kv)
    keep = ("cycles", "ticks", "seconds", "dispatches", "ns_per_dispatch", "ns_per_visited_tick", "elements", "ns_per_element")
    doc = {
        "source_directory": os.getcwd(),
        "binary_sha256": hashlib.sha256(Path("target/release/tsp-iss").read_bytes()).hexdigest(),
        "load_average": os.getloadavg(),
        "host": platform.platform(),
        "machine": platform.machine(),
        "command": "tsp-iss bench --repeat 3 --elements (release build)",
        "programs": {n: {k: float(r[k]) for k in keep if k in r} for n, r in rows.items()},
    }
    path = "opentsp-iss/crates/tsp-iss/benches/baseline.json"
    open(path, "w").write(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {path}: load {doc['load_average']}")
    PY

# A PGO build of the in-process binding that serving runs (`opentsp-vllm/native`; target/iss-pgo/_tsp_iss.abi3.so), trained on the P8 bucket
# of the gate's fixture (ISS_BENCH_DIR, else the fixture `just iss-perf-gate` writes under ~/.cache/opentsp/iss-perf). The serving
# recipes run the binding, not the command-line `tsp-iss`, so only this build can speed serving up; copy the file over the binding of
# `just vllm-binding` (opentsp-vllm/native/target/<os>-<arch>/_tsp_iss.abi3.so) to use it. Optional: the gain is measured and stated in
# opentsp-iss/README.md. Needs the Rust LLVM tools (`rustup component add llvm-tools-preview`) and a Python 3.11 or newer.
iss-pgo-binding:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    dir="${ISS_BENCH_DIR:-}"
    if [ -z "$dir" ]; then
        dir="$(ls -dt "${ISS_PERF_CACHE:-$HOME/.cache/opentsp/iss-perf}"/fixture-* 2> /dev/null | head -n 1 || true)"
    fi
    if [ -z "$dir" ] || [ ! -f "$dir/P8_4_128.elf" ]; then
        echo "iss-pgo-binding: no P8_4_128 training input: run 'just iss-perf-gate' once, or 'python3 tools/iss_bench_prep.py --out DIR P8_4_128' and set ISS_BENCH_DIR=DIR" >&2
        exit 2
    fi
    tools/iss_build_pgo.sh --binding "$dir"
