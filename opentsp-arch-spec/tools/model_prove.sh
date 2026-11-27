#!/usr/bin/env bash
# Prove the model's $property obligations (model/props/props.sail) with Sail's SMT backend
# and z3, on the mini and ref profiles. Every $property must be proved and every $counterexample
# refuted. The properties use the model's own functions, so the whole project is loaded.
#
# Environment:
#   PROVE_PROFILES   profiles to prove, default "mini ref"; a name is config/<name>.json, so "test/ref_small"
#                    is a test profile; "all" is mini, ref and every config/test/*.json (about 3 min each,
#                    measured: 374 s for mini and ref on a loaded machine)
#   PROVE_WIDE       1 (default): after the default pass, repeat each profile with `--smt-int-size 256` and
#                    require the same verdicts (Sail's default 128-bit integers hide a counterexample that
#                    needs a wider one); 0 skips the second pass
#   PROVE_REACH      1 (default): after the first profile print which model functions the obligations call, from
#                    Sail's type-checked AST (test/formal/reach.py); 0 skips it
#   PROVE_TWINS      1 (default): build the twin reachability query of every property and require it to be satisfiable (a property
#                    that always raises, or that is true on no input that completes, proves nothing); 0 skips the twins
#   PROVE_KEEP       a directory to keep the generated queries in (default: a scratch directory, removed)
#   PROVE_ALLOW_Z3   1 accepts a z3 other than the pinned one (test/formal/z3_version.txt)
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
py="${PYTHON:-python3}"
if [ -n "${PROVE_KEEP:-}" ]; then
  work="$PROVE_KEEP"; mkdir -p "$work"
else
  work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
fi
"$py" "$here/tools/check_smt_obligations.py" --solver-info "$here/test/formal/z3_version.txt"
profiles="${PROVE_PROFILES:-mini ref}"
if [ "$profiles" = all ]; then
  profiles="mini ref"
  for cfg in "$here"/config/test/*.json; do profiles="$profiles test/$(basename "$cfg" .json)"; done
fi
# The proof project: links to the model, the port-proof instantiation, the obligations text, the project file.
PROPS_TWINS="${PROVE_TWINS:-1}" bash "$here/test/formal/props_project.sh" "$work"
twin_flag=(); [ "${PROVE_TWINS:-1}" != 1 ] || twin_flag=(--twins)
# Generation must succeed; do not mask compiler failures. Solve each named file
# independently: aggregate sat/unsat counts can conceal two opposite regressions.
# Sail's backends reject a `config` key that the profile file leaves out (the SMT backend: "Runtime configuration is not supported"), so
# Sail reads the profile with its omitted defaulted keys set (ras.diag_depth, schema_invariants.with_defaults): .
normalised_config() { # profile -> path of the normalised copy
  local out="$work/config_${1//\//_}.json"
  [ -f "$out" ] || "$py" "$here/tools/schema_invariants.py" --normalise "$here/config/$1.json" "$out"
  echo "$out"
}
prove_profile() { # profile directory-suffix extra-sail-flags...
  local profile="$1" tag="$2"; shift 2
  local dir="$work/${profile//\//_}$tag"
  # The relations between keys first: Sail reports a profile that breaks one with a message about another key.
  "$py" "$here/tools/schema_invariants.py" --check "$here/config/$profile.json" || { echo "model-prove: profile $profile breaks a schema invariant" >&2; return 1; }
  mkdir -p "$dir"
  echo "Proving profile $profile${tag:+ ($tag)}"
  # -D OPENTSP_PROOF_BUILD: internal_error is translatable under the proof build symbol (model/prelude/errors.sail), so an
  # obligation that reaches a failure site gets a verdict instead of crashing the generator.
  (cd "$dir" && sail -smt --all-modules -D OPENTSP_PROOF_BUILD "$@" --config "$(normalised_config "$profile")" "$work/props.sail_project")
  "$py" "$here/tools/check_smt_obligations.py" "$work/obligations.txt" "$dir" ${twin_flag[@]+"${twin_flag[@]}"}
}
# What the obligations reach in the model (an upper bound on coverage), from Sail's own type-checked AST; the dump reuses
# the typechecker cache of the profile's proof directory.
reach_report() { # profile
  local dir="$work/${1//\//_}"
  (cd "$dir" && sail --no-warn --ddump-tc-ast --just-check --all-modules --config "$(normalised_config "$1")" \
    "$work/props.sail_project") > "$work/ast.txt"
  # The extra SMT suites have projects of their own (prove_extra.sh dumps their ASTs), so that "union of all sources" is one.
  PROVE_EXTRA_AST_DIR="$work/extra_ast" PROVE_EXTRA_AST_ONLY=1 bash "$here/test/formal/prove_extra.sh" ties frontend numerics > /dev/null
  local sources=() suite
  for suite in ties frontend numerics; do sources+=(--source "$suite=$work/extra_ast/$suite.ast:$here/test/formal/props/$suite.txt"); done
  "$py" "$here/test/formal/reach.py" "$work/ast.txt" --functions "$work/reach_functions.tsv" "${sources[@]}" --lean "$here/test/formal/lean/proofs"
  [ -z "${PROVE_KEEP:-}" ] || echo "reach: functions and the obligations calling them: $work/reach_functions.tsv"
}
reached=0
for profile in $profiles; do
  prove_profile "$profile" ""
  if [ "$reached" = 0 ] && [ "${PROVE_REACH:-1}" = 1 ]; then reached=1; reach_report "$profile"; fi
  if [ "${PROVE_WIDE:-1}" = 1 ]; then
    prove_profile "$profile" ".int256" --smt-int-size 256
  fi
done

# Independent mathematical transition model. Check each named theorem and
# reject solver errors, timeouts, missing theorems and unexpected output.
z3 -T:60 "$here/test/formal/frontend_induction.smt2" > "$work/induction.log"
for theorem in capture complete-or-discard host-injection deliver-padding-overflow dispatch-or-hold reset-epoch fifo-order ordered-dma-grant; do
  printf '%s\nunsat\n' "$theorem"
done > "$work/expected"
diff -u "$work/expected" "$work/induction.log"
cat "$work/induction.log"
