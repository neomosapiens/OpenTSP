#!/usr/bin/env bash
# Local validation entry point for the specification toolchain.
#   ci_tests.sh                 every step below (what the original script did, plus the proof-gate unit tests)
#   ci_tests.sh rust            build the reference simulator and run the Rust workspace tests
#   ci_tests.sh proof           the SMT proof gate (opentsp-arch-spec/tools/model_prove.sh, mini and ref)
#   ci_tests.sh formal-unit     the unit tests of the proof infrastructure (no solver run, about 40 s: 36 s measured at a load of 16)
#   ci_tests.sh formal          the extra formal checks: Lean theorems, extra SMT obligations, SRF ring and lane tests, exact encodings
#                               (the same checks `just verify-all` runs after its first four steps)
#   ci_tests.sh nightly         the long gates: the mutation ratchet, the proof on every profile, the extra obligations at 256-bit integers
#                               (the same checks `just prove-nightly` runs)
#   ci_tests.sh --dry-run       check the tools the steps need and print the steps; run nothing
# `just check-docs` runs formal-unit. Use `just verify-all` for the proof, model checks and extra formal checks.
set -euo pipefail
cd "$(dirname "$0")/.."
py="${PYTHON:-python3}"

need_tools() {
  command -v sail
  command -v z3
  "$py" -c 'import yaml, mpmath'
}

step_rust() {
  mkdir -p target/spec
  bash opentsp-arch-spec/tools/build_sim.sh target/spec/tsp_sim
  export TSP_SIM="$PWD/target/spec/tsp_sim"
  cargo test --release --workspace
  # Replay is deliberately opt-in; its integration tests must execute too.
  cargo test --release -p tsp-iss --features replay --test replay --test replay_compiled --test platform_replay \
    --test replay_param --test replay_platform --test sram_journal --test ecc_high_cycle
}

step_proof() {
  bash opentsp-arch-spec/tools/model_prove.sh
}

# Unit tests of the proof infrastructure: the vacuity check and solver banner, the port-proof substitution counts, the
# exact encoding checks, the mutation ratchet's table and comparison, the reach and coverage reports.
step_formal_unit() {
  spec=opentsp-arch-spec
  "$py" $spec/test/config/test_smt_obligations.py
  "$py" $spec/test/formal/test_fetch_port_proof.py
  "$py" $spec/test/formal/test_encoding_exact.py
  "$py" $spec/test/formal/encoding_exact.py
  "$py" $spec/test/formal/test_prove_mutants.py
  "$py" $spec/test/formal/test_parallel_extra.py
  "$py" $spec/test/formal/test_reach.py
  "$py" $spec/test/formal/test_coverage_report.py
  "$py" $spec/test/formal/mutants.py --check-table
  "$py" $spec/test/formal/test_scripts.py
  "$py" $spec/test/formal/test_gen_callbacks.py
  "$py" $spec/test/formal/test_patch_generated.py
  "$py" $spec/test/formal/test_verify_all.py
  "$py" $spec/test/formal/test_check_axioms.py
  "$py" $spec/test/formal/test_lean_while.py
  "$py" $spec/test/formal/test_lean_driver.py
  "$py" $spec/test/formal/test_solver_protocol.py
  "$py" $spec/test/formal/test_isqrt_certificate.py
  "$py" $spec/test/formal/test_numeric_cap_certificate.py
  "$py" $spec/test/formal/test_numeric_cap_build.py
  "$py" $spec/test/formal/lean/test_resource_mutations.py
  "$py" $spec/test/formal/test_encoding_from_model.py
}

# The extra formal checks, as `just verify-all` runs them (recipes lean-check, prove-extra, srf-ring-test, lanes-test, encoding-exact, encoding-model).
step_formal() {
  spec=opentsp-arch-spec
  bash $spec/test/formal/lean/lean_check.sh mini
  bash $spec/test/formal/lean/lean_check.sh ref
  bash $spec/test/formal/prove_extra.sh
  bash $spec/test/formal/srf_ring_test.sh
  bash $spec/test/formal/lanes_test.sh
  "$py" $spec/test/formal/encoding_exact.py
  bash $spec/test/formal/encoding_model.sh
}

# The nightly gates, as `just prove-nightly` runs them (recipes prove-mutants, prove-profiles, prove-extra-wide).
step_nightly() {
  spec=opentsp-arch-spec
  bash $spec/tools/prove_mutants.sh --expect-current --jobs "${JOBS:-3}"
  PROVE_PROFILES=all bash $spec/tools/model_prove.sh
  PROVE_EXTRA_WIDE=1 bash $spec/test/formal/prove_extra.sh
}

steps=("$@")
dry=0
if [ "${steps[0]:-}" = "--dry-run" ]; then dry=1; steps=("${steps[@]:1}"); fi
[ "${#steps[@]}" -gt 0 ] || steps=(rust proof formal-unit)
for s in "${steps[@]}"; do
  case "$s" in rust | proof | formal-unit | formal | nightly) ;; *) echo "ci_tests.sh: unknown step $s" >&2; exit 2 ;; esac
done
need_tools
if [ "$dry" -eq 1 ]; then
  for s in "${steps[@]}"; do echo "would run: $s"; done
  exit 0
fi
for s in "${steps[@]}"; do
  echo "== ci_tests: $s"
  "step_${s//-/_}"
done
