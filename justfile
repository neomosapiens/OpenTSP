import 'tools/just/perf-model.just'
import 'tools/just/conformance.just'
import 'tools/just/iss-perf.just'
import 'tools/just/formal.just'
import 'tools/just/spec-verification.just'
import 'tools/just/iss-counters.just'
import 'tools/just/replay.just'
import 'tools/just/serving-generic.just'
import 'tools/just/formal-proofs.just'
import 'tools/just/chips-links.just'

# Install this checkout's Python reference and toolchain packages in the active toolchain environment.
python-setup:
    ${PYTHON:-python3} -m pip install -e opentsp-arch-spec -e opentsp-sw-toolchain -e opentsp-vllm

# Workspace tests: docs location, freeze, lint, the performance model's tests and the Rust workspace (ISS and toolchain).
test: check-docs check-freeze lint perf-model
    cargo test --workspace
    {{just_executable()}} --justfile {{justfile()}} iss-replay-test
    cargo test --release -p tsp-iss --features phases --test phase_report

# The code-quality gate: rustfmt, clippy (warnings denied), pyflakes, shell syntax (tools/lint.sh).
lint:
    bash tools/lint.sh

check-docs:
    ${PYTHON:-python3} tests/spec/test_encoding_type_codes.py
    ${PYTHON:-python3} tests/spec/test_encoding_extensions_table.py
    ${PYTHON:-python3} tests/spec/test_manifest_topology_parity.py
    ${PYTHON:-python3} tests/docs/test_bringup_iq_vectors.py
    ${PYTHON:-python3} tests/spec/test_reference_assembler_import.py
    bash tools/check_docs_location.sh
    bash tools/check_clean_text.sh
    ${PYTHON:-python3} tools/check_source_policy.py
    ${PYTHON:-python3} tests/infra/test_source_policy.py
    ${PYTHON:-python3} tests/architecture/test_python_packages.py
    ${PYTHON:-python3} tests/infra/test_python_checkout.py
    ${PYTHON:-python3} tests/infra/test_ensure_sim.py
    ${PYTHON:-python3} tests/infra/test_host_counters.py
    ${PYTHON:-python3} tests/infra/test_devcontainer_identity.py
    ${PYTHON:-python3} tests/infra/test_rust_coverage.py
    ${PYTHON:-python3} opentsp-sw-toolchain/tests/test_schedule.py
    ${PYTHON:-python3} tests/infra/test_observed_comparison.py
    ${PYTHON:-python3} tests/infra/test_check_freeze.py
    ${PYTHON:-python3} tests/infra/test_check_perf_base.py
    ${PYTHON:-python3} opentsp-arch-spec/test/isa_db/test_generator_cli.py
    ${PYTHON:-python3} opentsp-arch-spec/test/config/test_schema_cli.py
    ${PYTHON:-python3} tests/infra/test_check_clean_text.py
    ${PYTHON:-python3} tests/infra/test_check_docs_location.py
    ${PYTHON:-python3} opentsp-arch-spec/tools/gen_hazard_table.py --check
    ${PYTHON:-python3} tests/docs/test_rtl_rule_text.py
    ${PYTHON:-python3} tests/docs/test_rtl_obligations.py
    ${PYTHON:-python3} tests/docs/test_csr_link_map.py
    ${PYTHON:-python3} tests/docs/test_assembly_acc_index.py
    ${PYTHON:-python3} tests/asm/test_diff_tsp_as_profiles.py
    ${PYTHON:-python3} tests/asm/test_diff_tsp_as_mutant_ratchet.py
    ${PYTHON:-python3} tests/docs/test_specification_contracts.py
    ${PYTHON:-python3} tests/docs/test_kernel_cookbook_location.py
    ${PYTHON:-python3} tests/spec/test_no_instruction_128.py
    ${PYTHON:-python3} tests/spec/test_refusal_names_config.py
    ${PYTHON:-python3} tests/spec/test_wpack_no_truncate.py
    ${PYTHON:-python3} tests/spec/test_diff_oracle_ext_random.py
    ${PYTHON:-python3} tests/docs/test_mxm_rows_advisory.py
    ${PYTHON:-python3} tests/docs/test_relay_recipe.py
    ${PYTHON:-python3} tests/docs/test_verify_usage_text.py
    ${PYTHON:-python3} opentsp-arch-spec/test/config/test_all_ext_profile.py
    ${PYTHON:-python3} opentsp-arch-spec/test/config/test_rtl_obligation_counts.py
    ${PYTHON:-python3} tests/spec/test_mutation_survivor_limits.py
    ${PYTHON:-python3} tools/check_gates_wired.py
    ${PYTHON:-python3} tests/infra/test_check_gates_wired.py
    ${PYTHON:-python3} tests/infra/test_profile_replay_lists.py
    ${PYTHON:-python3} tests/infra/test_model_vector_parallel.py
    ${PYTHON:-python3} tests/infra/test_arch_doc_currency.py
    just formal-unit
    ${PYTHON:-python3} tests/iss_perf/test_iss_ab_overhead.py
    ${PYTHON:-python3} tests/iss_perf/test_comparison_inputs.py

# Verify the public source baseline for specification 0.1.
check-freeze:
    bash tools/check_freeze.sh
    ${PYTHON:-python3} tests/infra/test_check_freeze.py

# Architecture specification v0.1 (opentsp-arch-spec). Needs sail 0.20.3, z3, a C compiler, GMP, python3 + pyyaml + mpmath.
# Every model check except the SMT proofs and the exhaustive encoding check.
model-test:
    bash opentsp-arch-spec/tools/model_test.sh
    ${PYTHON:-python3} opentsp-arch-spec/test/numerics/gen_fp8_digests.py --check

# Compare configuration, stdin and same-cycle link contracts across the reference and downstream tools.
harness-contract:
    cargo build --release -p tsp-iss
    TSP_SIM="${TSP_SIM:-target/spec/tsp_sim}" TSP_ISS=target/release/tsp-iss python3 tests/spec/test_harness_contract.py
    TSP_SIM="${TSP_SIM:-target/spec/tsp_sim}" TSP_ISS=target/release/tsp-iss python3 tests/spec/test_config_key_sweep.py

model-prove:
    bash opentsp-arch-spec/tools/model_prove.sh

# Compare every 32-bit word decoded by the Sail model with the catalogue.
model-exhaustive:
    bash opentsp-arch-spec/tools/isa_exhaustive.sh

# Full double-precision f32 screening and selected model/mpmath comparisons (about 8 min on 12 threads; the f32 screening check).
model-transcendental:
    bash opentsp-arch-spec/tools/check_transcendental_f32.sh

fmt:
    cargo fmt --all

# Assembler crates (ots-profile, ots-isa, ots-format, ots-asm): fails below 90 % lines or branches,
# per crate and in total. Needs nightly with llvm-tools-preview and cargo-llvm-cov (branch data).
coverage-asm:
    bash tools/coverage_asm.sh

# ISS crates (tsp-numerics, tsp-iss), with the differential tests against tsp_sim (TSP_SIM or target/spec/tsp_sim):
# fails below 90 % lines or branches, per crate and in total (all tests are under tests/, none in src/). Needs
# nightly with llvm-tools-preview and cargo-llvm-cov (branch data); takes about 15 minutes.
coverage-iss:
    bash tools/coverage_iss.sh

# End-to-end programs: the
# cases tagged `e2e` of tests/programs/ (vec_add, shift_select, barrier_pipeline, park_waits, repeat, matmul_i8,
# weighted_matmul) on `ref`, `mini` and, where generated, `test/ref_slow`. Each is assembled by ots-asm and tsp_as.py (byte-identical objects and weight
# packages), run on tsp_sim (TSP_SIM, default target/spec/tsp_sim, built by build_sim.sh when absent) and on tsp-iss
# (the harness-compatible mode, and natively with `tsp-iss run <object> --weights <package>`), with the E line, the
# diagnostics and the dumped words asserted and the canonical traces of all runs equal.
e2e: check-docs
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    sim="${TSP_SIM:-target/spec/tsp_sim}"
    bash tools/ensure_sim.sh "$sim"
    cargo build --release -q -p ots-asm -p tsp-iss
    python3 opentsp-sw-toolchain/tests/test_long_decimals.py "${CARGO_TARGET_DIR:-target}/release/ots-asm"
    python3 tests/programs/gen/test_program_generators.py
    TSP_SIM="$sim" python3 tests/programs/gen/test_pager.py
    python3 tests/programs/gen/build.py --check
    python3 tests/programs/run.py --sim "$sim" --iss target/release/tsp-iss --tag e2e --jobs "${JOBS:-4}"

# Program cases in tests/programs/: the library
# tests, `build.py --check`, then every case through both assemblers on tsp_sim (TSP_SIM, default target/spec/tsp_sim)
# and on the ISS when one is built (TSP_ISS or target/release/tsp-iss; otherwise the ISS runs are reported as skipped).
programs: kernel-cookbook programs-numerics ship-chain
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    cargo build --release -q -p ots-asm -p tsp-iss
    python3 tests/programs/gen/test_program_generators.py
    python3 tests/programs/gen/test_ref_wide.py
    python3 tests/programs/gen/test_geometry.py
    python3 tests/programs/gen/test_run_selection.py
    python3 opentsp-sw-toolchain/tests/test_import_surface.py
    TSP_SIM="${TSP_SIM:-target/spec/tsp_sim}" python3 tests/programs/gen/test_pager.py
    TSP_SIM="${TSP_SIM:-target/spec/tsp_sim}" python3 tests/programs/gen/test_patch.py
    TSP_SIM="${TSP_SIM:-target/spec/tsp_sim}" python3 tests/programs/gen/test_kv.py
    TSP_SIM="${TSP_SIM:-target/spec/tsp_sim}" python3 tests/programs/gen/test_serve_kernels.py
    python3 tests/programs/gen/test_masked_kv.py "${TSP_SIM:-target/spec/tsp_sim}" target/release/tsp-iss
    python3 tests/programs/gen/test_division.py
    python3 tests/programs/gen/test_cookbook.py
    python3 tests/showcase/gpt2small/test_plan.py
    python3 tests/showcase/gpt2small/test_ship_cache.py
    python3 tests/showcase/gpt2small/test_gpt2s.py
    python3 tests/programs/gen/build.py --check
    python3 tests/programs/run.py --sim "${TSP_SIM:-target/spec/tsp_sim}" --iss "${CARGO_TARGET_DIR:-target}/release/tsp-iss" --jobs "${JOBS:-4}"

# Seeds run in parallel; the default is seeds 1 and 2 at 500 cases, the long run is `just asm-diff "1 2 3 4 5" 2000`.
# Differential of ots-asm against tools/tsp_as.py over every command (tests/diff_tsp_as.py): `cases` mutations per seed.
asm-diff seeds="1 2" cases="500":
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    cargo build --release -q -p ots-asm
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    cp "${CARGO_TARGET_DIR:-target}/release/ots-asm" "$tmp/ots-asm"
    pids=()
    for s in {{seeds}}; do
        python3 opentsp-sw-toolchain/tests/diff_tsp_as.py "$tmp/ots-asm" --fuzz {{cases}} --seed "$s" --objects > "$tmp/$s.log" 2>&1 &
        pids+=($!)
    done
    rc=0
    i=0
    for s in {{seeds}}; do
        if wait "${pids[$i]}"; then tail -n 1 "$tmp/$s.log"; else rc=1; cat "$tmp/$s.log"; fi
        i=$((i + 1))
    done
    exit $rc

# The 3-chip chain of stage programs (one layer per chip, C2C between chips, the host at both ends) on tsp_sim --chips 3 and the ISS (about 2 minutes).
ship-chain:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    cargo build --release -q -p ots-asm -p tsp-iss
    python3 tests/programs/chain_run.py --sim "${TSP_SIM:-target/spec/tsp_sim}" --chips 3

# GPT-2 small on chips through the binding's System (needs `just vllm-binding`): the tests of the host words, the routes and a 2-chip chain with mutants (about 20 s).
ship-system-test:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    cargo build --release -q -p ots-asm -p tsp-iss
    python3 tests/showcase/gpt2small/test_ship_pipeline.py
    python3 tests/showcase/gpt2small/test_ship_system.py

# The same, restricted to the cases tagged `sim`, in parallel.
programs-sim:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    cargo build --release -q -p ots-asm -p tsp-iss
    python3 tests/programs/run.py --sim "${TSP_SIM:-target/spec/tsp_sim}" --tag sim --jobs "${JOBS:-8}"

# The numpy twin of the lane arithmetic (opentsp_tools.programs.numerics) against the Python oracle; the oracle runs
# in a process pool, so the wall time depends on the host load (about 3 CPU minutes).
programs-numerics:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    python3 tests/programs/gen/test_numerics.py

# ISS numerics: exhaustive i8 and f16, 8-bit pairs, oracle hashes, f16 dot cases (release build).
iss-numerics:
    cargo test --release -p tsp-numerics

# ISS numerics against the Sail model's own functions: all i8 and f16 transcendentals, f32 samples (and
# with F32_SCREEN_WORK=DIR every input of a check_transcendental_f32.sh --work DIR), ~13.5 M kernel inputs.
iss-numerics-sail:
    bash opentsp-iss/crates/tsp-numerics/tools/run_differential.sh ${F32_SCREEN_WORK:+--screen-work $F32_SCREEN_WORK}

# f32 TanH, Exp, RSqrt, Log of the ISS against the mpmath results of the f32 screening check (set up with
# `just model-transcendental`'s work directory: F32_SCREEN_WORK=DIR) and against the Ziv port on a stride.
iss-transcendental:
    TSP_F32_WORK=$F32_SCREEN_WORK cargo test --release -p tsp-numerics --test differential -- --ignored --skip sail_dumps_match --nocapture

# ISS harness: the drop-in configuration and argument checks of tsp_sim, run on the ISS (needs the python venv of the
# specification toolchain: source ~/.local/opt/opentsp-toolchain/env.sh).
iss-config:
    cargo build --release -p tsp-iss
    python3 opentsp-arch-spec/test/config/config_negative.py target/release/tsp-iss opentsp-arch-spec/config/ref.json
    python3 opentsp-arch-spec/test/config/config_negative.py target/release/tsp-iss opentsp-arch-spec/config/mini.json
    python3 opentsp-arch-spec/test/config/harness_reject.py target/release/tsp-iss opentsp-arch-spec/config/ref.json
    python3 opentsp-arch-spec/test/config/harness_reject.py target/release/tsp-iss opentsp-arch-spec/config/mini.json

# Replay shipped and generated profile vectors through the ISS. This suite also
# checks RTL obligations and compares the numerical implementations.

iss-vectors: iss-vectors-extra iss-vectors-variants iss-vectors-superlanes iss-rtl-obligations iss-numerics
    cargo build --release -p tsp-iss
    target/release/tsp-iss vectors opentsp-arch-spec/conformance/model/ref.json --config opentsp-arch-spec/config/ref.json
    target/release/tsp-iss vectors opentsp-arch-spec/conformance/model/mini.json --config opentsp-arch-spec/config/mini.json

# ISS drop-in scripts: the directed tests of the specification run with the ISS in place of tsp_sim: the front end,
# the whole directed suite on ref, and the C2C and multi-chip suite.
iss-scripts:
    cargo build --release -p tsp-iss
    python3 opentsp-arch-spec/test/directed/frontend.py target/release/tsp-iss opentsp-arch-spec/config/ref.json
    python3 opentsp-arch-spec/test/directed/frontend.py target/release/tsp-iss opentsp-arch-spec/config/mini.json
    python3 opentsp-arch-spec/test/directed/directed.py target/release/tsp-iss opentsp-arch-spec/config/ref.json
    python3 opentsp-arch-spec/test/directed/fp8.py target/release/tsp-iss opentsp-arch-spec/config/ref.json
    python3 opentsp-arch-spec/test/directed/ext_types.py target/release/tsp-iss opentsp-arch-spec/config/ref.json
    python3 opentsp-arch-spec/test/directed/ext_types.py target/release/tsp-iss opentsp-arch-spec/config/test/mini_ext.json
    python3 opentsp-arch-spec/test/directed/ext_types.py target/release/tsp-iss opentsp-arch-spec/config/test/ref_ext.json
    python3 opentsp-arch-spec/test/directed/bf16.py target/release/tsp-iss opentsp-arch-spec/config/ref.json
    python3 opentsp-arch-spec/test/directed/bf16.py target/release/tsp-iss opentsp-arch-spec/config/test/ext_bf16.json
    python3 opentsp-arch-spec/test/directed/ras_diag.py target/release/tsp-iss opentsp-arch-spec/config/ref.json
    python3 opentsp-arch-spec/test/directed/superlanes.py target/release/tsp-iss opentsp-arch-spec/config/ref.json
    python3 opentsp-arch-spec/test/directed/superlanes.py target/release/tsp-iss opentsp-arch-spec/config/test/sl8.json
    python3 opentsp-arch-spec/test/directed/superlanes.py target/release/tsp-iss opentsp-arch-spec/config/test/sl32.json
    python3 opentsp-arch-spec/test/directed/mxm_keys.py target/release/tsp-iss opentsp-arch-spec/config/ref.json
    python3 opentsp-arch-spec/test/directed/mxm_keys.py target/release/tsp-iss opentsp-arch-spec/config/mini.json
    python3 opentsp-arch-spec/test/directed/mxm_keys.py target/release/tsp-iss opentsp-arch-spec/config/test/acc8.json
    python3 opentsp-arch-spec/test/directed/mxm_keys.py target/release/tsp-iss opentsp-arch-spec/config/test/f16_tandem.json
    python3 opentsp-arch-spec/test/directed/mxm_keys.py target/release/tsp-iss opentsp-arch-spec/config/test/fp8_rate2.json
    python3 opentsp-arch-spec/test/directed/fp8_pair.py target/release/tsp-iss opentsp-arch-spec/config/test/fp8_rate2.json
    python3 opentsp-arch-spec/test/directed/mxm_keys.py target/release/tsp-iss opentsp-arch-spec/config/test/one_plane.json
    python3 opentsp-arch-spec/test/directed/mxm_keys.py target/release/tsp-iss opentsp-arch-spec/config/test/mxm_wide.json
    python3 opentsp-arch-spec/test/directed/ras_diag.py target/release/tsp-iss opentsp-arch-spec/config/mini.json
    python3 opentsp-arch-spec/test/directed/ras_spare.py target/release/tsp-iss opentsp-arch-spec/config/ref.json
    python3 opentsp-arch-spec/test/c2c/c2c_directed.py target/release/tsp-iss opentsp-arch-spec/config/ref.json
    python3 opentsp-arch-spec/test/c2c/c2c_directed.py target/release/tsp-iss opentsp-arch-spec/config/mini.json
    python3 opentsp-arch-spec/test/c2c/deskewrecv_directed.py target/release/tsp-iss opentsp-arch-spec/config/ref.json
    python3 opentsp-arch-spec/test/c2c/deskewrecv_directed.py target/release/tsp-iss opentsp-arch-spec/config/mini.json
    python3 opentsp-arch-spec/test/c2c/relay_directed.py target/release/tsp-iss opentsp-arch-spec/config/ref.json
    python3 opentsp-arch-spec/test/c2c/relay_directed.py target/release/tsp-iss opentsp-arch-spec/config/mini.json
    python3 opentsp-arch-spec/test/c2c/relay_directed.py target/release/tsp-iss opentsp-arch-spec/config/test/c2c_wide.json
    python3 opentsp-arch-spec/test/c2c/skew_directed.py target/release/tsp-iss opentsp-arch-spec/config/ref.json
    python3 opentsp-arch-spec/test/c2c/skew_directed.py target/release/tsp-iss opentsp-arch-spec/config/mini.json
    python3 opentsp-arch-spec/test/c2c/skew_directed.py target/release/tsp-iss opentsp-arch-spec/config/test/c2c_wide.json

# ISS differential gate (oracle and fuzzing): `diff_oracle.py --impl` with the ISS against tsp_sim on the profiles of 20
# superlanes: `ref`, `test/ref_slow`, `test/link_tight`, `test/ref_small`
# (the oracle runs each of its templates ISS_DIFF_PROGRAMS times, seed 1; default 300; the old name `N` still works),
# then `fuzz_sim.py` with the ISS in place of
# tsp_sim (FUZZ seconds, default 120, with ISS_DIFF_JOBS workers, default 8, the old name `JOBS` still works; the script
# reads the program name `tsp_sim:` in messages, so the ISS runs through tools/iss_as_sim.sh). TSP_SIM, default target/spec/tsp_sim, built by build_sim.sh when absent.
# Then the parity gates of the faulted and high-start-cycle paths: `iss-vecmut` (400 mutated golden vectors from four start cycles),
# `iss-sites` (the golden vectors with site observation), `iss-diff-variants` (the oracle on the 12 profiles `mini`, `test/mid_min`, `test/sl8`,
# `test/sl32`, `test/acc8`, `test/f16_tandem`, `test/fp8_rate2`, `test/one_plane`, `test/mxm_wide`, `test/ext_bf16`, `test/ref_ext` and `test/mini_ext`) and
# the configuration and directed checks, including the shared harness CLI and topology fixtures.
# Each step must pass before the next begins. The optional ISS timing and instruction benchmarks have separate recipes.
iss-diff: iss-wide
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    sim="${TSP_SIM:-target/spec/tsp_sim}"
    bash tools/ensure_sim.sh "$sim"
    cargo build --release -q -p tsp-iss
    cfg=opentsp-arch-spec/config
    for c in "$cfg/ref.json" "$cfg/test/ref_slow.json" "$cfg/test/link_tight.json" "$cfg/test/ref_small.json"; do
        python3 opentsp-arch-spec/test/oracle/diff_oracle.py "$sim" "$c" "${ISS_DIFF_PROGRAMS:-${N:-300}}" 1 --impl target/release/tsp-iss
    done
    out="$(mktemp -d)"
    TSP_ISS=target/release/tsp-iss python3 opentsp-arch-spec/test/config/fuzz_sim.py tools/iss_as_sim.sh "$cfg" \
        opentsp-arch-spec/isa-db/generated/encoding.json "${FUZZ:-120}" 1 "${ISS_DIFF_JOBS:-${JOBS:-8}}" "$out"
    rmdir "$out" 2>/dev/null || true
    for step in iss-vecmut iss-sites iss-diff-variants iss-config iss-scripts harness-contract; do
        echo "== $step"
        {{just_executable()}} --justfile {{justfile()}} "$step"
        echo "== $step: passed"
    done

# ISS performance: the benchmark programs B0..B4 and the idle run with `tsp-iss bench`, then the
# targets as an `#[ignore]`d timing test (tests/benchmark_baseline.rs, run here with `--ignored`; not part of `just test`; release builds only,
# a quiet machine and the reference host): B1..B3 at least 1000 times faster than tsp_sim on the reference
# host, B4 within the tolerance of its recorded rate in benches/baseline.json, 10^8 idle cycles under 1 s. Takes about two
# minutes (B4 runs 17 s a time).
iss-bench:
    cargo run --release -q -p tsp-iss -- bench --repeat 3
    cargo test --release -p tsp-iss --test benchmark_baseline -- --ignored

# The Rust fuzzer of the ISS: random text images on random queues of every unit, run for
# SECONDS (default 60) of wall time, release build; it asserts no panic (exit 4), determinism, `--tick-every-cycle` equality
# and the exit code and stdout of tsp_sim. The seed is $ISS_FUZZ_SEED (default 1); a failing input is written under
# $ISS_FUZZ_OUT or the temp directory, with its arguments in a `.args` file next to it. On several chips it also
# asserts that 1 and 3 chip threads (`--jobs`) give the same bytes as the default.
iss-fuzz SECONDS="60":
    ISS_FUZZ_SECONDS={{SECONDS}} cargo test --release -p tsp-iss --test fuzz -- --nocapture

# `--tick-every-cycle` equality: the golden vectors, the gap, Repeat, fetch and link programs with
# and without the option, then every differential test against tsp_sim with the ISS ticking every cycle (release build;
# an idle tick costs about 19 us, so the c2c_directed scripts, which run programs of 20 million cycles, are left out).
iss-tick:
    cargo test --release -p tsp-iss --test tick_every_cycle
    ISS_TICK_EVERY_CYCLE=1 cargo test --release -p tsp-iss --test c2c_diff --test decode_diff --test directed_diff --test fetch_diff --test frontend_diff --test mem_diff --test mxm_diff --test sxm_diff --test vxm_diff --test mixed_diff -- --skip c2c_directed

# TinyStories showcase (pure Python + numpy; needs the specification toolchain environment): model, layout and export.
showcase-reference:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    sim="${TSP_SIM:-target/spec/tsp_sim}"
    bash tools/ensure_sim.sh "$sim"
    cargo build --release -q -p ots-asm -p tsp-iss
    $PYTHON -m pytest -q -m "not slow" tests/showcase/tinystories

# The same with the `slow` tests: the full 8-layer text-sizing grid and the  `paged_iss` tests. A cold program cache adds generation time.
showcase-reference-full:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    sim="${TSP_SIM:-target/spec/tsp_sim}"
    bash tools/ensure_sim.sh "$sim"
    cargo build --release -q -p ots-asm -p tsp-iss
    $PYTHON -m pytest -q tests/showcase/tinystories

# , the host driver without vLLM (opentsp_vllm.models.tinystories.driver_paged): the toy scheduler (block pool, prefix caching, chunked
# prefill, continuous batching, preemption by recompute) on the numerical reference, then on the ISS with the 2-layer cached bucket programs and the K/V
# pool carried in SRAM across steps (4 requests sharing a prefix, a preemption, a random schedule with block copy and zeroing, staged against
# paged on the 8-layer model). The programs are built on demand into $OPENTSP_VLLM_CACHE (default a temporary directory, so a cold run builds about
# a dozen buckets: 10 minutes; keep the buckets between runs by setting it). `just showcase-paged-full`: the full-model scenario.
showcase-paged:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    cargo build --release -q -p ots-asm -p tsp-iss
    $PYTHON -m pytest -q tests/showcase/tinystories/test_driver_paged.py

# The full 8-layer model, two requests sharing a prefix, on the ISS (the real checkpoint when the Hugging Face cache has it, else synthetic weights):
# builds three 8-layer buckets on the first run (about 10 minutes with the jobs of build_programs.py in parallel, then cached) and records cycles and
# wall time. `driver_paged.py full --json FILE` prints the table of steps.
showcase-paged-full:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    cargo build --release -q -p ots-asm -p tsp-iss
    $PYTHON -m pytest -q -s -m paged_full tests/showcase/tinystories/test_driver_paged.py

# TinyStories kernels: the kernel twins' tests, then the cases tagged `ts-kernel` of
# tests/programs/model/ through both assemblers on tsp_sim (TSP_SIM, default target/spec/tsp_sim, built when absent)
# and on tsp-iss, with the E line, the diagnostics, the dumped words and the canonical traces compared.
showcase-kernels:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    sim="${TSP_SIM:-target/spec/tsp_sim}"
    bash tools/ensure_sim.sh "$sim"
    cargo build --release -q -p ots-asm -p tsp-iss
    python3 tests/programs/gen/test_ref.py
    python3 tests/programs/gen/build.py --check
    python3 tests/programs/run.py --sim "$sim" --iss target/release/tsp-iss --tag ts-kernel --jobs "${JOBS:-4}"

# One TinyStories transformer block: layer 0 of the synthetic weights, one
# token, a preloaded K/V cache at positions 0, 15, 40 and 62, emitted as paged hand-scheduled generators. The expectation
# tests (the numerical reference's layer step, the pinned digests, queue fit), the cases tagged `ts-block` through both assemblers on tsp_sim
# and on tsp-iss (the E line, no diagnostics, no IQ lints, every dumped word equal to the numerical reference, the canonical traces equal),
# then the reference check of the real checkpoint's layer 0 (a visible SKIP when the checkpoint is not cached; the PyTorch
# comparison has its own SKIP line when PyTorch is absent).
# `just showcase-block STEP=1` is the decode step: eight layers, the final LayerNorm and the
# output head window at positions 0, 5, 62 and 63. A step's case is about 11 MB, so it is not checked in: the cases are
# built into target/ts-step (build.py --out) and run through tsp_sim, the ISS and the native ISS (about 2.5 minutes with
# three jobs), then the reference check of the step on the real checkpoint (tests/showcase/tinystories/check_step.py).
showcase-block step="":
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    sim="${TSP_SIM:-target/spec/tsp_sim}"
    bash tools/ensure_sim.sh "$sim"
    cargo build --release -q -p ots-asm -p tsp-iss
    step="{{step}}"; step="${step#STEP=}"
    if [ "$step" = 1 ]; then
        python3 -m pytest -q tests/showcase/tinystories/test_step.py
        out=target/ts-step
        rm -rf "$out"
        python3 tests/programs/gen/build.py --out "$out" model/ts_step_p00 model/ts_step_p05 model/ts_step_p62 \
            model/ts_step_p63
        python3 tests/programs/run.py --sim "$sim" --iss target/release/tsp-iss --dir "$out" --tag ts-step \
            --jobs "${JOBS:-3}"
        TSP_SIM="$sim" python3 tests/showcase/tinystories/check_step.py
    else
        python3 -m pytest -q tests/showcase/tinystories/test_block.py
        python3 tests/programs/gen/build.py --check
        python3 tests/programs/run.py --sim "$sim" --iss target/release/tsp-iss --tag ts-block --jobs "${JOBS:-4}"
        TSP_SIM="$sim" python3 tests/showcase/tinystories/check_block.py
    fi

# The vLLM environment: a venv outside the
# repository (OPENTSP_VLLM_VENV, default ~/.local/opt/opentsp-vllm-venv) with the pinned CPU vLLM distribution
# and the editable OpenTSP plugin. A matching installation is reused.
vllm-env:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    bash tools/setup_vllm_env.sh

# The vLLM surface gate: the pinned vLLM has every name, signature and RPC the design depends on, and the package
# skeleton matches. Runs in the vLLM venv and fails, never skips, when the venv does not exist.
vllm-surface:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    venv="${OPENTSP_VLLM_VENV:-$HOME/.local/opt/opentsp-vllm-venv}"
    if [ ! -x "$venv/bin/python" ]; then
        echo "vllm-surface: the vLLM environment $venv does not exist; run 'just vllm-env' first" >&2
        exit 1
    fi
    here="$("$venv/bin/python" tools/check_python_checkout.py "$PWD")"
    echo "vllm-surface: testing the tree $PWD (opentsp_vllm from $here)"
    "$venv/bin/python" -m pytest -q opentsp-vllm/tests/test_vllm_surface.py

# The vLLM unit tests: TspPlatform, StepPlan, TspModelRunner and TspWorker on ReferenceChip (the numpy twin of the chip, bit-exact
# against opentsp_vllm.models.tinystories.paged_reference) with synthetic SchedulerOutputs. Seconds. Fails, never skips, when the venv is missing.
vllm-unit:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    cargo build --release -q -p ots-asm -p tsp-iss   # the release assembler carries the profile digest: a stale binary makes every assembled fixture fail with OTS-E-ASM-003
    venv="${OPENTSP_VLLM_VENV:-$HOME/.local/opt/opentsp-vllm-venv}"
    if [ ! -x "$venv/bin/python" ]; then
        echo "vllm-unit: the vLLM environment $venv does not exist; run 'just vllm-env' first" >&2
        exit 1
    fi
    here="$("$venv/bin/python" tools/check_python_checkout.py "$PWD")"
    echo "vllm-unit: testing the tree $PWD (opentsp_vllm from $here)"
    "$venv/bin/python" -m pytest -q opentsp-vllm/tests/unit opentsp-vllm/tests/test_serving_artifacts.py opentsp-vllm/tests/test_demo_observe.py opentsp-vllm/tests/bench/test_host_cost_summary.py

# End to end on the real vLLM engine: `LLM.generate` and `vllm serve` on ReferenceChip and IssChip, shadow mode, the 2-layer test
# model and the full model, token ids equal `paged_reference.generate_paged`. It needs the binding (`just vllm-binding`) and the bucket programs of both
# models (`LAYERS=2 just vllm-programs` and `LAYERS=8 just vllm-programs`; a missing set is built by the tests, which takes minutes). The engine
# runs in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0) except in the `vllm serve` test. `tests/serving/measure.py` prints the start-up and
# per-step host overhead of the serving path.
[positional-arguments]
vllm-e2e cache="shared":
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    venv="${OPENTSP_VLLM_VENV:-$HOME/.local/opt/opentsp-vllm-venv}"
    if [ ! -x "$venv/bin/python" ]; then
        echo "vllm-e2e: the vLLM environment $venv does not exist; run 'just vllm-env' first" >&2
        exit 1
    fi
    case "$1" in
        shared) export OPENTSP_VLLM_CACHE="${OPENTSP_VLLM_CACHE:-$HOME/.cache/opentsp/vllm}" ;;
        isolated) unset OPENTSP_VLLM_CACHE ;;  # the pytest session owns and removes its scratch cache
        *) echo "vllm-e2e: cache must be shared or isolated" >&2; exit 2 ;;
    esac
    here="$("$venv/bin/python" tools/check_python_checkout.py "$PWD")"
    echo "vllm-e2e: testing the tree $PWD (opentsp_vllm from $here), cache ${OPENTSP_VLLM_CACHE:-isolated}"
    "$venv/bin/python" -m pytest -q opentsp-vllm/tests/serving/test_e2e.py

# GPT-2 small truncated to 2 layers on 2 chips of the SHIP-style pipeline behind the real vLLM engine: tokens equal the stand-alone form, every step equal to the numpy
# twin bit for bit; one request and two at once. Needs `just vllm-binding` and the GPT-2 small checkpoint in ~/.cache/opentsp/checkpoints/gpt2; the stage programs of the 2 layers are
# built into ~/.cache/opentsp/ship the first time.
vllm-ship-e2e:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    venv="${OPENTSP_VLLM_VENV:-$HOME/.local/opt/opentsp-vllm-venv}"
    if [ ! -x "$venv/bin/python" ]; then
        echo "vllm-ship-e2e: the vLLM environment $venv does not exist; run 'just vllm-env' first" >&2
        exit 1
    fi
    here="$("$venv/bin/python" tools/check_python_checkout.py "$PWD")"
    echo "vllm-ship-e2e: testing the tree $PWD (opentsp_vllm from $here)"
    "$venv/bin/python" -c 'from pathlib import Path; from opentsp_vllm.ship_chip import DEFAULT_CHECKPOINT; import sys; path = Path(DEFAULT_CHECKPOINT, "model.safetensors"); sys.exit(f"vllm-ship-e2e: missing required checkpoint {path}" if not path.is_file() else 0)'
    cargo build --release -q -p ots-asm
    "$venv/bin/python" -m pytest -q -ra opentsp-vllm/tests/serving/test_ship_e2e.py

# The PyO3 binding of the ISS: opentsp-vllm/native/tsp_iss_py, module opentsp_vllm._tsp_iss. It is its
# own Cargo workspace (the root one never needs Python) and uses PyO3 0.28 with the limited API (abi3-py311),
# PYO3_BUILD_EXTENSION_MODULE=1 and PYO3_NO_PYTHON=1, so the build needs no Python interpreter, no headers and links no
# libpython; build.rs adds the macOS linker argument. cargo builds the cdylib and the recipe copies it to
# opentsp-vllm/native/target/$(uname -s)-$(uname -m)/_tsp_iss.abi3.so (untracked; on macOS its install name is made relative). The import check
# uses the vLLM venv's Python, else python3 (3.11 or newer); without either it says so and the build still stands.
[positional-arguments]
vllm-binding mode="force":
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    venv="${OPENTSP_VLLM_VENV:-$HOME/.local/opt/opentsp-vllm-venv}"
    build_args=()
    case "$1" in
        force) ;;
        if-stale) build_args=(--quiet) ;;
        *) echo "vllm-binding: expected force or if-stale" >&2; exit 2 ;;
    esac
    export PYO3_NO_PYTHON=1 PYO3_BUILD_EXTENSION_MODULE=1
    # the opt-in parametrised schedule replay is compiled in (it runs only when a chip asks: `Chip.param_replay(True)`, the `replay`
    # serving option or OPENTSP_VLLM_REPLAY=1); OPENTSP_BINDING_REPLAY=0 builds a binding without it
    features=(--features native-mem-prefetch)
    if [ "${OPENTSP_BINDING_REPLAY:-1}" != 0 ]; then features+=(--features replay); fi
    cargo build --release --manifest-path opentsp-vllm/native/Cargo.toml "${features[@]}" ${build_args[@]+"${build_args[@]}"}
    case "$(uname -s)" in Darwin) lib=lib_tsp_iss.dylib ;; *) lib=lib_tsp_iss.so ;; esac
    out=opentsp-vllm/native/target/$(uname -s)-$(uname -m)/_tsp_iss.abi3.so
    built="${CARGO_TARGET_DIR:-opentsp-vllm/native/target}/release/$lib"
    if [ "$1" = if-stale ] && [ -f "$out" ] && [ ! "$built" -nt "$out" ]; then exit 0; fi
    mkdir -p "$(dirname "$out")"
    staged="$(mktemp "${out}.XXXXXX")"
    trap 'rm -f "$staged"' EXIT
    cp "$built" "$staged"
    if [ "$(uname -s)" = Darwin ]; then install_name_tool -id @loader_path/_tsp_iss.abi3.so "$staged"; fi
    mv -f "$staged" "$out"
    if [ -x "$venv/bin/python" ]; then py="$venv/bin/python"; else py="$(command -v python3 || true)"; fi
    if [ -z "$py" ]; then echo "vllm-binding: built $out; no Python to import it with"; exit 0; fi
    PYTHONPATH="$PWD/opentsp-vllm/src" "$py" -c 'from opentsp_vllm import _tsp_iss; import os, sys; assert os.path.samefile(_tsp_iss.__file__, sys.argv[1]); print("binding:", _tsp_iss.__file__)' "$out"

# The binding's tests: Chip (programs, patches, warm restart against replay, aperture, failures, statistics) and IssChip; the `System` of N chips (`test_system.py`, compared with `tsp_sim --chips N`, which `tools/ensure_sim.sh` builds when stale); and the
# patching of paged text blocks, the `ProgramSet` reader, its site values and logits decoding, and the pool copy and zero, on the real cached buckets of
# the 2-layer model (built into $OPENTSP_VLLM_CACHE, else a temporary directory: about 2 minutes the first time).
# The tests require an editable install of all OpenTSP packages from this checkout. Run `just vllm-binding` first.
# (a missing binding is a failure).
vllm-binding-test:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    cargo build --release -q -p ots-asm -p tsp-iss   # the release assembler carries the profile digest: a stale binary makes every assembled fixture fail with OTS-E-ASM-003
    venv="${OPENTSP_VLLM_VENV:-$HOME/.local/opt/opentsp-vllm-venv}"
    if [ ! -x "$venv/bin/python" ]; then
        echo "vllm-binding-test: the vLLM environment $venv does not exist; run 'just vllm-env' first" >&2
        exit 1
    fi
    here="$("$venv/bin/python" tools/check_python_checkout.py "$PWD")"
    echo "vllm-binding-test: testing the tree $PWD (opentsp_vllm from $here)"
    "$venv/bin/python" -m pytest -q opentsp-vllm/tests/test_binding.py opentsp-vllm/tests/test_text_blocks.py opentsp-vllm/tests/test_tiny_fixed.py opentsp-vllm/tests/test_programs.py opentsp-vllm/tests/test_pool_blocks.py opentsp-vllm/tests/test_sram_record.py opentsp-vllm/tests/test_system.py opentsp-vllm/tests/test_ship_runtime.py opentsp-vllm/tests/test_recovery_messages.py opentsp-vllm/tests/test_program_content.py opentsp-vllm/tests/test_ship_overlap.py
    # the launcher of the showcase rebuilds a stale binding: its test touches a dependency of the binding
    bash tools/test_showcase_fresh.sh

# The randomised multi-request schedules: 5 seeded schedules of 12 steps on the 2-layer test model, each run step by step on a
# warm-restart chip, a replay-restart chip and a shadow chip rebuilt from the host's record of the SRAM only; after every step the logits, the diagnostics,
# the cycle counts and the SRAM digests are equal and the logits, the tokens and the lanes equal the numerical reference's; the failure reproducer end to end.
# The 9 buckets of C = 64 are built first if the cache lacks them (`OPENTSP_VLLM_CACHE` reuses a cache between runs; else a temporary one: about 1 minute with
# 6 processes). Measured on an 18-core Apple Silicon MacBook: about 2.6 minutes with the buckets built, 3.3 without (the tests print the per-step times). Needs `just vllm-binding` first.
vllm-schedules:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    venv="${OPENTSP_VLLM_VENV:-$HOME/.local/opt/opentsp-vllm-venv}"
    if [ ! -x "$venv/bin/python" ]; then
        echo "vllm-schedules: the vLLM environment $venv does not exist; run 'just vllm-env' first" >&2
        exit 1
    fi
    cargo build --release -q -p ots-asm -p tsp-iss
    export OTS_ASM="${OTS_ASM:-$PWD/target/release/ots-asm}" TSP_ISS="${TSP_ISS:-$PWD/target/release/tsp-iss}"
    here="$("$venv/bin/python" tools/check_python_checkout.py "$PWD")"
    echo "vllm-schedules: testing the tree $PWD (opentsp_vllm from $here)"
    "$venv/bin/python" -m pytest -q -s -m "not slow" opentsp-vllm/tests/test_restart_schedules.py

# The full-model schedule: one 6-step schedule on the 8-layer model (the real checkpoint when it is cached under ~/.cache/huggingface, else synthetic
# seed 1), warm against replay against the shadow, and the logits against the numerical reference. It builds the three 8-layer buckets it needs (P(8,1,64), P(8,4,64),
# P(4,4,64): about 2.5 minutes with 3 processes; set `OPENTSP_VLLM_CACHE` to keep them) and runs about 1 minute (3.3 s a step on each chip). Run it in the background.
vllm-schedules-full:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    venv="${OPENTSP_VLLM_VENV:-$HOME/.local/opt/opentsp-vllm-venv}"
    if [ ! -x "$venv/bin/python" ]; then
        echo "vllm-schedules-full: the vLLM environment $venv does not exist; run 'just vllm-env' first" >&2
        exit 1
    fi
    cargo build --release -q -p ots-asm -p tsp-iss
    export OTS_ASM="${OTS_ASM:-$PWD/target/release/ots-asm}" TSP_ISS="${TSP_ISS:-$PWD/target/release/tsp-iss}"
    here="$("$venv/bin/python" tools/check_python_checkout.py "$PWD")"
    echo "vllm-schedules-full: testing the tree $PWD (opentsp_vllm from $here)"
    "$venv/bin/python" -m pytest -q -s -m slow opentsp-vllm/tests/test_restart_schedules.py

# The bucket programs of the vLLM integration: the batched step emitter, the pager, `ots-asm asm`, the text
# package, the patch table and the output regions of each bucket, under ~/.cache/opentsp/vllm/<digest>/P<T>_<L>_<C>/
# ($OPENTSP_VLLM_CACHE overrides the root), the patch-table self-check, one run on the ISS (a padding plan, then real lanes
# against the numerical reference), and <cache>/cycles.json. A bucket whose digest is already in the cache is skipped. These are environment variables, not just arguments:
# `LAYERS=2|8` (default 8; the 2-layer test model has all 18 buckets too), `C=64|128`, `T=1,2,4,8` restrict the grid, e.g.
# `LAYERS=2 C=64 just vllm-programs` builds the 9 buckets of the test model's C = 64 half; `JOBS=n` builds n buckets at once (one process each). The 8-layer grid takes about 22 minutes in
# sequence (almost all in the pager); run long builds in chunks, the cache makes each chunk restartable. `TIME=1` also times every
# built bucket on the ISS (best of three, one at a time) into <cache>/<digest>/timing.json; <cache>/cycles.json merges it. The digest
# covers the generator sources, the profile and the layer count, not the `ots-asm` and `tsp-iss` binaries (their crate version,
# sha256 and source commit are recorded in each program.json and manifest.json): after a change of the assembler or the simulator run
# with `--force` (`python3 -m opentsp_vllm.models.tinystories.build_programs --layers N --force`).
vllm-programs:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    cargo build --release -q -p ots-asm -p tsp-iss
    export OTS_ASM="${OTS_ASM:-$PWD/target/release/ots-asm}" TSP_ISS="${TSP_ISS:-$PWD/target/release/tsp-iss}"
    args=(--layers "${LAYERS:-8}" --jobs "${JOBS:-1}")
    if [ -n "${C:-}" ]; then args+=(--C "$C"); fi
    if [ -n "${T:-}" ]; then args+=(--T "$T"); fi
    python3 -m opentsp_vllm.models.tinystories.build_programs "${args[@]}"
    if [ -n "${TIME:-}" ]; then python3 tests/showcase/tinystories/cycle_report.py --time --layers "${LAYERS:-8}"; fi
    python3 tests/showcase/tinystories/cycle_report.py --cycles

# TinyStories generation with the built-in host driver: the host driver (tests/showcase/tinystories/
# driver.py) runs the decode step of `showcase-block STEP=1` on the ISS once per token (prefill and generation) with
# the K/V rows and the embedding kept by the host and the output head on the host. `just showcase-decode N=20` generates
# 20 tokens: first with the synthetic weights (token IDs, dumped words and logits match the NumPy reference), then with the real checkpoint
# (compare the reference, PyTorch with f16-rounded inputs, and the original fp32
# model; a visible SKIP when the checkpoint, PyTorch or transformers is absent) and prints the story. `WEIGHTS=synthetic`
# or `WEIGHTS=real` runs one of the two. About 1 s per token on the ISS (the step is 91 080 cycles).
showcase-decode n="20" weights="both" prompt="":
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    cargo build --release -q -p ots-asm -p tsp-iss
    n="{{n}}"; n="${n#N=}"
    w="{{weights}}"; w="${w#WEIGHTS=}"
    pr="{{prompt}}"; pr="${pr#PROMPT=}"
    drv=opentsp_vllm.models.tinystories.driver
    extra=(--stream)
    if [ -n "$pr" ]; then extra+=(--prompt "$pr"); fi
    if [ "$w" != real ]; then python3 -m "$drv" --weights synthetic --tokens "$n" --golden exact "${extra[@]}"; fi
    if [ "$w" != synthetic ]; then python3 -m "$drv" --weights real --tokens "$n" --golden exact,rounded,fp32 "${extra[@]}"; fi

# The TinyStories showcase (docs/demos.md#single-chip-tinystories): the real checkpoint through real vLLM (the TspPlatform plugin) on the ISS, the
# KV cache paged in the chip's SRAM, tokens streamed as they are produced. `just showcase-prepare` builds, once and under
# `$OPENTSP_SHOWCASE_DIR` (default ~/.cache/opentsp/showcase), the checkpoint weights and immutable stages (`JOBS=n` concurrent builders). It needs the vLLM environment (`just vllm-env`) and the binding (`just vllm-binding`).
showcase-prepare:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    cargo build --release -q -p ots-asm -p tsp-iss
    venv="${OPENTSP_VLLM_VENV:-$HOME/.local/opt/opentsp-vllm-venv}"
    "$venv/bin/python" tools/check_python_checkout.py "$PWD" >/dev/null
    HF_HUB_OFFLINE=1 "$venv/bin/python" -m opentsp_vllm.showcase.prepare

# An interactive session: `just showcase` loads once, then asks for prompts (/help lists the commands). `PROMPT="Once upon a time" just showcase`
# is one shot; MAX_TOKENS, TEMPERATURE, TOP_K, SEED and VERBOSE=1 set the sampler and the per-step view.
[no-exit-message, positional-arguments]
showcase *args:
    @tools/showcase.sh chat "$@"

# Several prompts at once, continuously batched (`N=6` for another count; more than 4 queue); each stream's text is printed when it completes,
# with how much of its prompt came from the prefix cache.
[no-exit-message, positional-arguments]
showcase-batch *args:
    @tools/showcase.sh batch "$@"

# `vllm serve` with the plugin and the real weights on http://127.0.0.1:$PORT (default 8000), model name `tinystories`; the README has the curl.
[no-exit-message, positional-arguments]
showcase-serve *args:
    @tools/showcase.sh serve "$@"

# The showcase's own tests in the vLLM environment: the session logic (a scripted backend), the session on a real engine over ReferenceChip, and once on
# the ISS (the 2-layer test model; its programs are built into a scratch cache when absent: minutes).
showcase-test:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    venv="${OPENTSP_VLLM_VENV:-$HOME/.local/opt/opentsp-vllm-venv}"
    if [ ! -x "$venv/bin/python" ]; then
        echo "showcase-test: the vLLM environment $venv does not exist; run 'just vllm-env' first" >&2
        exit 1
    fi
    here="$("$venv/bin/python" tools/check_python_checkout.py "$PWD")"
    echo "showcase-test: testing the tree $PWD (opentsp_vllm from $here)"
    "$venv/bin/python" -m pytest -q opentsp-vllm/tests/showcase

# The Hugging Face GPT-2 checkpoint (fp32) of the f16-rounded TSP weights, for vLLM's stock CPU backend (
# tools/vllm_ref_checkpoint.py). `LAYERS=2` (default) or 8, `WEIGHTS=synthetic:1` (default) or `real` (the calum/tinystories-gpt2-3M
# checkpoint of the local Hugging Face cache); written to `$OUT`, default <cache>/ref-checkpoints/l<LAYERS>-<WEIGHTS>.
vllm-ref-checkpoint:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    venv="${OPENTSP_VLLM_VENV:-$HOME/.local/opt/opentsp-vllm-venv}"
    if [ ! -x "$venv/bin/python" ]; then
        echo "vllm-ref-checkpoint: the vLLM environment $venv does not exist; run 'just vllm-env' first" >&2
        exit 1
    fi
    layers="${LAYERS:-2}"; weights="${WEIGHTS:-synthetic:1}"
    out="${OUT:-${OPENTSP_VLLM_CACHE:-$HOME/.cache/opentsp/vllm}/ref-checkpoints/l$layers-${weights//:/}}"
    "$venv/bin/python" tools/check_python_checkout.py "$PWD" >/dev/null
    "$venv/bin/python" tools/vllm_ref_checkpoint.py --layers "$layers" --weights "$weights" --out "$out"

# The serving tests: every test of opentsp-vllm/tests/serving on ReferenceChip and IssChip, through a real vLLM
# `LLM` on the TSP platform (continuous batching, stops, streaming, prefix caching, chunked prefill, preemption, sampling, logprobs,
# penalties, `n`, two full-model scenarios, the comparison with vLLM's CPU backend). It builds missing bucket programs into
# $OPENTSP_VLLM_CACHE (default ~/.cache/opentsp/vllm; 9 test-model buckets take minutes, the full model's 18 about 22 minutes, once).
# Extra pytest arguments follow in ARGS, e.g. `ARGS="-k sampling" just vllm-serve-test`. Fails, never skips, when the venv is missing.
vllm-serve-test:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    cargo build --release -q -p ots-asm -p tsp-iss   # the release assembler carries the profile digest: a stale binary makes every assembled fixture fail with OTS-E-ASM-003
    venv="${OPENTSP_VLLM_VENV:-$HOME/.local/opt/opentsp-vllm-venv}"
    if [ ! -x "$venv/bin/python" ]; then
        echo "vllm-serve-test: the vLLM environment $venv does not exist; run 'just vllm-env' first" >&2
        exit 1
    fi
    here="$("$venv/bin/python" tools/check_python_checkout.py "$PWD")"
    cargo build --release -q -p ots-asm -p tsp-iss
    export OTS_ASM="${OTS_ASM:-$PWD/target/release/ots-asm}" TSP_ISS="${TSP_ISS:-$PWD/target/release/tsp-iss}"
    export OPENTSP_VLLM_CACHE="${OPENTSP_VLLM_CACHE:-$HOME/.cache/opentsp/vllm}"
    echo "vllm-serve-test: testing the tree $PWD (opentsp_vllm from $here), program cache $OPENTSP_VLLM_CACHE"
    # shellcheck disable=SC2086
    "$venv/bin/python" -m pytest -q -ra opentsp-vllm/tests/serving ${ARGS:-}

# Synthesize and prove the management-domain result and DMA gate.
ras-prove:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    out="$PWD/target/ras-formal"
    spec="$PWD/opentsp-arch-spec"
    mkdir -p "$out/sail"
    (cd "$out/sail" && sail -smt "$spec/model/prelude/prelude.sail" "$spec/platform/ras/commit.sail")
    cd opentsp-arch-spec/platform/ras
    yosys -Q -q -p 'read_verilog -sv ras_commit.sv; synth -top ras_commit; check -assert'
    yosys -Q -q -p "read_verilog -sv ras_commit.sv; prep -top ras_commit; write_smt2 -wires \"$out/gate.smt2\""
    PYTHONPATH="$spec/tools" python3 check_refinement.py "$out"
    cat "$out/gate.smt2" state_upset.smt2 | z3 -T:60 -in > "$out/state-upset.log"
    diff -u <(printf '%s\n' sat unsat unsat) "$out/state-upset.log"
    echo "PASS single management-state upset blocks publication and requires service until cold reset"
    sby -f -d "$out/prove" commit.sby prove
    sby -f -d "$out/cover" commit.sby cover

# Exercise detected faults in the actual single-chip and full SHIP serving packages.
ras-qualify:
    #!/usr/bin/env bash
    set -euo pipefail
    export OPENTSP_VLLM=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
    py="${OPENTSP_VLLM_VENV:-$HOME/.local/opt/opentsp-vllm-venv}/bin/python"
    "$py" tools/check_python_checkout.py "$PWD" >/dev/null
    "$py" -m pytest -q opentsp-vllm/tests/test_ras_policy.py opentsp-vllm/tests/qualification/test_ras_recovery.py opentsp-vllm/tests/qualification/test_ras_http.py

# Assemble the released four-chip GPT-2 schedules against the upstream checkpoint.
[positional-arguments]
gpt2-programs *args:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    venv="${OPENTSP_VLLM_VENV:-$HOME/.local/opt/opentsp-vllm-venv}"
    if [[ ! -x "$venv/bin/python" ]]; then
        echo "gpt2-programs: run 'just vllm-env' first." >&2
        exit 2
    fi
    "$venv/bin/python" tools/check_python_checkout.py "$PWD" >/dev/null
    exec "$venv/bin/python" opentsp-vllm/tools/build_gpt2_programs.py "$@"

# Start the four-chip GPT-2 browser and its dedicated vLLM process.
gpt2-demo:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    venv="${OPENTSP_VLLM_VENV:-$HOME/.local/opt/opentsp-vllm-venv}"
    programs="${OPENTSP_GPT2_4_PROGRAMS:-$HOME/.cache/opentsp/gpt2-four-chip}"
    if [[ ! -x "$venv/bin/python" || ! -r "$programs/single-page-pair/manifest.json" ]]; then
        echo "gpt2-demo: prepare the serving environment and four-chip program bundle as described in docs/demos.md#four-chip-gpt-2-small." >&2
        exit 2
    fi
    export PYTHONPATH="$PWD/opentsp-vllm/src:$PWD/opentsp-sw-toolchain/python:$PWD/opentsp-arch-spec/reference"
    export PYTHONDONTWRITEBYTECODE=1
    export OPENTSP_ISS_CHIP_WORKERS="${OPENTSP_ISS_CHIP_WORKERS:-4}"
    exec "$venv/bin/python" opentsp-vllm/demo/serve.py --launch --programs "$programs" --port "${DEMO_PORT:-18765}" --vllm-port "${VLLM_PORT:-8000}"
