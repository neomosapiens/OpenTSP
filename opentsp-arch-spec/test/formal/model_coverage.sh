#!/usr/bin/env bash
# What the proofs and the test suites reach in the Sail model . Three measurements:
#   1. static: the model functions the proof obligations call, from Sail's type-checked AST (reach.py; an upper bound);
#   2. dynamic: functions entered and branch targets taken by Sail unit tests on every profile,
#      directed tests, oracle differential and golden vectors, on a model built with `sail --c-coverage`;
#   3. the merge of the dynamic runs against the static list of the model (coverage_report.py).
# The instrumented builds reuse the unmodified build scripts: a `sail` wrapper adds --c-coverage and a `cc` wrapper adds
# the recorder (coverage_shim.c, which replaces the Rust runtime of the Sail distribution that is not installed here).
#   model_coverage.sh                  all suites (about 30 min on a loaded machine)
#   COVERAGE_SUITES="unit"             only the named suites: unit directed vectors oracle
#   COVERAGE_WORK=DIR                  keep the builds, coverage files and reports in DIR
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
py="${PYTHON:-python3}"
suites="${COVERAGE_SUITES:-unit directed vectors oracle}"
if [ -n "${COVERAGE_WORK:-}" ]; then work="$COVERAGE_WORK"; mkdir -p "$work"; else work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT; fi
real_sail="$(command -v sail)"; real_cc="$(command -v cc)"

echo "== static: model functions the obligations call (profile mini)"
bash "$here/test/formal/props_project.sh" "$work/proj"
mkdir -p "$work/proj/mini"
(cd "$work/proj/mini" && sail --no-warn --ddump-tc-ast --just-check --all-modules --config "$here/config/mini.json" \
  "$work/proj/props.sail_project") > "$work/ast.txt"
# The extra SMT suites have projects of their own (prove_extra.sh dumps their ASTs); the Lean proofs name model functions in their files.
PROVE_EXTRA_AST_DIR="$work/extra_ast" PROVE_EXTRA_AST_ONLY=1 bash "$here/test/formal/prove_extra.sh" ties frontend numerics > /dev/null
sources=(); for suite in ties frontend numerics; do sources+=(--source "$suite=$work/extra_ast/$suite.ast:$here/test/formal/props/$suite.txt"); done
"$py" "$here/test/formal/reach.py" "$work/ast.txt" --functions "$work/reach_functions.tsv" "${sources[@]}" --lean "$here/test/formal/lean/proofs"

# Wrappers: every `sail -c` gets --c-coverage; every `cc` that links a model gets the recorder.
# The build scripts find Sail's library at ../share/sail/lib relative to the sail they run, so the wrapper directory
# has a share link next to its bin.
mkdir -p "$work/tool/bin" "$work/runs"
ln -sfn "$(dirname "$real_sail")/../share" "$work/tool/share"
bin="$work/tool/bin"
cat > "$bin/sail" <<WRAP
#!/usr/bin/env bash
for a in "\$@"; do
  if [ "\$a" = -c ]; then exec "$real_sail" "\$@" --c-coverage "\${COVERAGE_STATIC:?}"; fi
done
exec "$real_sail" "\$@"
WRAP
cat > "$bin/cc" <<WRAP
#!/usr/bin/env bash
exec "$real_cc" "\$@" "$here/test/formal/coverage_shim.c"
WRAP
chmod +x "$bin/sail" "$bin/cc"
export SAIL_COVERAGE_DIR="$work/runs"

for suite in $suites; do
  case "$suite" in
    unit)
      echo "== Sail unit tests, instrumented"
      COVERAGE_STATIC="$work/static_unit.cov" PATH="$bin:$PATH" bash "$here/tools/model_unit_tests.sh" | grep -v "^Testing\|^Pass$"
      ;;
    directed | vectors | oracle)
      if [ ! -x "$work/tsp_sim_cov" ]; then
        echo "== instrumented simulator"
        COVERAGE_STATIC="$work/static_sim.cov" PATH="$bin:$PATH" bash "$here/tools/build_sim.sh" "$work/tsp_sim_cov"
      fi
      sim="$work/tsp_sim_cov"
      case "$suite" in
        directed)
          echo "== directed tests, instrumented"
          for c in ref test/ref_slow; do "$py" "$here/test/directed/directed.py" "$sim" "$here/config/$c.json" | tail -1; done
          "$py" "$here/test/c2c/c2c_directed.py" "$sim" "$here/config/ref.json" | tail -1
          for c in ref mini; do "$py" "$here/test/directed/frontend.py" "$sim" "$here/config/$c.json" | tail -1; done
          ;;
        vectors)
          echo "== golden vectors, instrumented"
          for c in ref mini; do "$py" "$here/tools/gen_model_vectors.py" --check "$sim" "$here/config/$c.json" "$here/conformance/model/$c.json"; done
          ;;
        oracle)
          echo "== Python oracle differential, instrumented"
          "$py" "$here/test/oracle/diff_oracle.py" "$sim" "$here/config/ref.json" "${ORACLE_N:-100}" 1
          ;;
      esac
      ;;
    *) echo "model_coverage: unknown suite $suite" >&2; exit 2 ;;
  esac
done

echo "== merge"
cat "$work"/static_*.cov > "$work/static_all.cov"
"$py" "$here/test/formal/coverage_report.py" "$work/static_all.cov" "$work"/runs/cov.* --missed "$work/missed_functions.tsv"
[ -z "${COVERAGE_WORK:-}" ] || echo "coverage: model functions no run entered: $work/missed_functions.tsv"
