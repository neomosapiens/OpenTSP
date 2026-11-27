# Local proof, coverage and export recipes. The longer campaigns can run individually.

# About 3 min per profile and pass, plus the --smt-int-size 256 pass: run it in the background with a log.
# Prove the obligations on every profile: mini, ref and each config/test/*.json.
prove-profiles:
    PROVE_PROFILES=all bash opentsp-arch-spec/tools/model_prove.sh

# The obligations once on the profiles with the widened keys (addr_bits 16, one MXM plane, acc_depth 8, 128 C2C links), default integer width
# only: about 21 minutes. Run it before a release and when a key of these profiles changes.
prove-widened:
    PROVE_PROFILES="test/w16 test/one_plane test/acc8 test/c2c_wide" PROVE_WIDE=0 PROVE_REACH=0 bash opentsp-arch-spec/tools/model_prove.sh

# A mutant that nothing kills runs every gate (about 35 min); a kill stops at the first gate that fails; JOBS mutants run together (the
# whole table takes about two hours at 3). A new survivor, or a survivor that is now killed, fails the run. Prints the three numbers
# `proofs: a/N, tests: b/N, survivors: c/N` (and `lean: l/N` when a Lean statement kills a mutant).
# The proof-gate mutation ratchet: one-line mutants of the model against test/formal/mutants_expected.txt.
prove-mutants:
    bash opentsp-arch-spec/tools/prove_mutants.sh --expect-current --jobs "${JOBS:-3}"

# About 40 s. Unit tests of the proof infrastructure (vacuity check, solver pin, port-proof counts, exact encodings, ratchet); run by `check-docs`.
formal-unit:
    bash tools/ci_tests.sh formal-unit

# Exact structural checks of the instruction encodings (every entry, every pair).
encoding-exact:
    ${PYTHON:-python3} opentsp-arch-spec/test/formal/encoding_exact.py

# About 5 min (the model is built with the C backend). The encoding masks as the model has them: bits the model treats as fixed, found by
# single-bit probes of words that the model decodes, against the match patterns of the catalogue (every entry, both widths).
encoding-model:
    bash opentsp-arch-spec/test/formal/encoding_model.sh

# The generated config schema equals config/schema.json (config/schema.json is never rewritten).
schema-check:
    bash opentsp-arch-spec/tools/model_check.sh --check

# Check the Sail exports used for proof and simulation.
export-check:
    #!/usr/bin/env bash
    set -euo pipefail
    bash opentsp-arch-spec/test/export/export_check.sh


# Static reach of the obligations from Sail's type-checked AST, then functions and branch targets the suites execute.
# What the proofs and the test suites reach in the Sail model (about 30 min; COVERAGE_SUITES narrows it).
model-coverage:
    bash opentsp-arch-spec/test/formal/model_coverage.sh

# Prints a SHA-256 over verdict lines, excluding timings and paths.
# Every formal gate in one run: proof gate (both integer widths), exhaustive encodings, model checks, prover export, the extra
# obligations, the Lean theorems, the SRF ring and lane tests and the exact encodings (long: about an hour more than the first four).
verify-all:
    #!/usr/bin/env bash
    set -euo pipefail
    scope=full
    if [ "${VERIFY_ALL_STEPS+x}" ]; then
        scope=selected
        if [[ ${VERIFY_ALL_STEPS} =~ ^[[:space:]]*$ ]]; then
            echo "no checks selected" >&2
            exit 2
        fi
    fi
    set -f
    checks=()
    log="${VERIFY_ALL_LOG:-$PWD/target/verify-all.log}"
    mkdir -p "$(dirname "$log")"
    : > "$log"
    timings=""
    # VERIFY_ALL_STEPS narrows the run (for example to export-check) when a step is being debugged.
    for step in ${VERIFY_ALL_STEPS-model-prove model-exhaustive model-test export-check prove-extra lean-check srf-ring-test lanes-test encoding-exact encoding-model f16-smoke}; do
        checks+=("$step")
        echo "== verify-all: $step" | tee -a "$log"
        started=$(date +%s)
        just --justfile "{{ justfile() }}" "$step" 2>&1 | tee -a "$log"
        took=$(( $(date +%s) - started ))
        echo "-- verify-all: $step took $took s" | tee -a "$log"
        timings="$timings $step=${took}s"
    done
    echo "-- verify-all: step times:$timings"
    verdicts="$(grep -E '^(== verify-all:|PASS |typecheck ok|schema check ok|exhaustive32:|score:|reach:|theorem|capture|complete-or-discard|host-injection|deliver-padding-overflow|dispatch-or-hold|reset-epoch|fifo-order|ordered-dma-grant|unsat$|prove_extra: all suites proved|lean_check: ok |c_test ok: |encoding_exact: [0-9]|encoding_model: [0-9]+ entries|fround_fp_theory: |f16_exhaustive: |model-test: all passed|export_check: all checks passed|ok: [0-9]+ while loops)' "$log" || true)"
    echo "verify-all: PASS source-directory=$PWD verdict-sha256=$(printf '%s\n' "$verdicts" | shasum -a 256 | cut -d' ' -f1) scope=$scope checks=${checks[*]} log=$log"

# The nightly gates in one run: the mutation ratchet, the proof on all nine profiles and the extra obligations at 256-bit integers.
# Run it in the background with a log (hours); it prints one line per gate with its time and a hash over the verdict lines.
prove-nightly:
    #!/usr/bin/env bash
    set -euo pipefail
    scope=full
    if [ "${PROVE_NIGHTLY_STEPS+x}" ]; then
        scope=selected
        if [[ ${PROVE_NIGHTLY_STEPS} =~ ^[[:space:]]*$ ]]; then
            echo "no checks selected" >&2
            exit 2
        fi
    fi
    set -f
    checks=()
    log="${PROVE_NIGHTLY_LOG:-$PWD/target/prove-nightly.log}"
    mkdir -p "$(dirname "$log")"
    : > "$log"
    for step in ${PROVE_NIGHTLY_STEPS-prove-mutants prove-profiles prove-extra-wide}; do
        checks+=("$step")
        echo "== prove-nightly: $step" | tee -a "$log"
        started=$(date +%s)
        just --justfile "{{ justfile() }}" "$step" 2>&1 | tee -a "$log"
        echo "-- prove-nightly: $step took $(( $(date +%s) - started )) s" | tee -a "$log"
    done
    verdicts="$(grep -E '^(== prove-nightly:|PASS |proofs: |prove_extra: all suites proved|mutants: )' "$log" || true)"
    echo "prove-nightly: PASS source-directory=$PWD verdict-sha256=$(printf '%s\n' "$verdicts" | shasum -a 256 | cut -d' ' -f1) scope=$scope checks=${checks[*]} log=$log"
