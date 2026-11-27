#!/usr/bin/env bash
# Extra SMT obligations of the formal-proofs package: suites of `$property` / `$counterexample` functions
# under test/formal/props/ that the model's own props/props.sail does not carry (the model directory is frozen).
# Each suite is a reduced Sail project (only the modules its obligations need), generated with `sail -smt -D OPENTSP_PROOF_BUILD`
# and solved by z3 like tools/model_prove.sh does: every named query is checked, the inventory file must list exactly the
# obligations of the suite, and a body that can raise is rejected (vacuity).
#   prove_extra.sh [suite ...]     suites: domains, ties, frontend, wide, hold, packing, numerics, fptheory (default: all, in this order: the cheapest first)
# Environment: OPENTSP_SPEC_DIR  an alternative arch-spec directory (the mutation ratchet points it at a mutated copy)
#              PROVE_KEEP=dir     keep the generated queries there (default: a scratch directory, removed)
#              PROVE_FAIL_FAST=1  stop at the first obligation that fails (the mutation ratchet does)
#              PROVE_ALLOW_Z3=1   accept a z3 other than the pinned one
#              PROVE_EXTRA_AST_DIR=dir   also write Sail's type-checked AST of each suite there (<suite>.ast), for test/formal/reach.py;
#              PROVE_EXTRA_AST_ONLY=1    with it, stop after the AST (no SMT generation, no solving)
#              PROVE_EXTRA_WIDE=1       after the default pass over every suite, prove them all again with `--smt-int-size 256` and require the same
#                                 verdicts (the nightly run does: about 1.8 times the default run and up to 22 GB for one query, so not per push)
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
spec="${OPENTSP_SPEC_DIR:-$here}"
py="${PYTHON:-python3}"
suites="${*:-domains ties frontend wide hold packing numerics fptheory}"
if [ "${PROVE_EXTRA_JOBS:-1}" != 1 ]; then
  exec "$py" "$here/test/formal/parallel_extra.py" "${BASH_SOURCE[0]}" "$spec" "${PROVE_EXTRA_JOBS}" "$suites"
fi
if [ -n "${PROVE_KEEP:-}" ]; then work="$PROVE_KEEP"; mkdir -p "$work"; else work="$(mktemp -d)"; fi
[ -n "${PROVE_KEEP:-}" ] || trap 'find "$work" -mindepth 1 -delete 2>/dev/null; rmdir "$work" 2>/dev/null || true' EXIT
"$py" "$spec/tools/check_smt_obligations.py" --solver-info "$spec/test/formal/z3_version.txt" 240 305
passes="default"
[ "${PROVE_EXTRA_WIDE:-0}" != 1 ] || passes="default 256"
for width_pass in $passes; do
for suite in $suites; do
  pass=""; width=()
  if [ "$width_pass" != default ]; then pass="$width_pass"; width=(--smt-int-size "$pass"); fi
  [ "$suite" != fptheory ] || [ -z "$pass" ] || continue    # one pass is enough for the cross-check against the solver's theory
  [ "$suite" != wide ] || [ -z "$pass" ] || continue        # the wide accumulator suite always runs at 256-bit integers (a lane is 128 bits)
  [ "$suite" != wide ] || width=(--smt-int-size 256)
  src="$spec/test/formal/props/$suite.sail"
  inv="$spec/test/formal/props/$suite.txt"
  [ -f "$src" ] && [ -f "$inv" ] || { echo "prove_extra: unknown suite $suite (no $src or $inv)" >&2; exit 2; }
  dir="$work/$suite${pass:+.int$pass}"
  mkdir -p "$dir/out"
  ln -s "$spec/model/prelude" "$dir/prelude"
  cp "$src" "$dir/props.sail"
  # The twin reachability query of every property (check_smt_obligations.twin_source), in the default pass: it must be satisfiable.
  twins=0
  if [ -z "$pass" ] && [ "$suite" != fptheory ]; then      # the fptheory property is checked against the solver's own theory instead
    twins=1
    "$py" -c 'import sys; sys.path.insert(0, sys.argv[1] + "/tools"); import check_smt_obligations as c; print(c.twin_source(open(sys.argv[2]).read()))' \
      "$spec" "$src" >> "$dir/props.sail"
  fi
  case "$suite" in
    numerics|ties|fptheory|domains|packing)    # the rounding core alone: prelude and core/numerics/fp.sail
      ln -s "$spec/model/core/numerics" "$dir/numerics"
      {
        printf 'prelude {\n  files\n    prelude/prelude.sail,\n    prelude/errors.sail,\n}\n\n'
        printf 'numerics {\n  requires prelude\n  files numerics/fp.sail\n}\n\n'
        printf 'props {\n  requires prelude, numerics\n  files props.sail\n}\n'
      } > "$dir/props.sail_project" ;;
    frontend)    # the instruction-queue transitions: the model's prelude, core, numerics, slice types and sys modules, and cu_iq.sail alone
      ln -s "$spec/model/core" "$dir/core"; ln -s "$spec/model/slices" "$dir/slices"; ln -s "$spec/model/sys" "$dir/sys"
      "$py" - "$spec/model/tsp.sail_project" "$dir/props.sail_project" <<'PY'
import re, sys
src = re.sub(r"//[^\n]*", "", open(sys.argv[1]).read())
blocks = dict(re.findall(r"(\w+)\s*\{(.*?)\n\}", src, re.S))
out = ["%s {%s\n}\n" % (name, blocks[name]) for name in ("prelude", "core", "numerics", "slice_types", "sys")]
out.append("cu {\n  requires prelude, core, numerics, slice_types, sys\n  files slices/cu/cu_iq.sail\n}\n")
out.append("props {\n  requires prelude, core, numerics, slice_types, sys, cu\n  files props.sail\n}\n")
open(sys.argv[2], "w").write("\n".join(out))
PY
      ;;
    wide)    # the wide accumulator functions of mxm_insts.sail: the whole slice layer is needed to type them
      ln -s "$spec/model/core" "$dir/core"; ln -s "$spec/model/slices" "$dir/slices"; ln -s "$spec/model/sys" "$dir/sys"
      "$py" - "$spec/model/tsp.sail_project" "$dir/props.sail_project" <<'PY'
import re, sys
src = re.sub(r"//[^\n]*", "", open(sys.argv[1]).read())
blocks = dict(re.findall(r"(\w+)\s*\{(.*?)\n\}", src, re.S))
out = ["%s {%s\n}\n" % (name, blocks[name]) for name in ("prelude", "core", "numerics", "slice_types", "sys", "slices")]
out.append("props {\n  requires prelude, core, numerics, slice_types, sys, slices\n  files props.sail\n}\n")
open(sys.argv[2], "w").write("\n".join(out))
PY
      ;;
    hold)    # bind model postlude/tick.sail hold_decide / view_free (no props twin)
      ln -s "$spec/model/core" "$dir/core"; ln -s "$spec/model/slices" "$dir/slices"; ln -s "$spec/model/sys" "$dir/sys"
      ln -s "$spec/model/postlude" "$dir/postlude"
      "$py" - "$spec/model/tsp.sail_project" "$dir/props.sail_project" <<'PY'
import re, sys
src = re.sub(r"//[^\n]*", "", open(sys.argv[1]).read())
blocks = dict(re.findall(r"(\w+)\s*\{(.*?)\n\}", src, re.S))
out = ["%s {%s\n}\n" % (name, blocks[name]) for name in ("prelude", "core", "numerics", "slice_types", "sys", "slices", "postlude")]
out.append("props {\n  requires prelude, core, numerics, slice_types, sys, slices, postlude\n  files props.sail\n}\n")
open(sys.argv[2], "w").write("\n".join(out))
PY
      ;;
    *) echo "prove_extra: unknown suite $suite" >&2; exit 2 ;;
  esac
  echo "Proving suite $suite${pass:+ (integer width $pass)}"
  config=()
  if [ "$suite" != numerics ] && [ "$suite" != fptheory ] && [ "$suite" != domains ] && [ "$suite" != packing ]; then
    # Sail's backends reject a `config` key the profile leaves out: the normalised copy carries the defaults
    "$py" "$spec/tools/schema_invariants.py" --normalise "$spec/config/mini.json" "$work/mini.normalised.json"
    config=(--config "$work/mini.normalised.json")
  fi   # core's geometry is a profile key
  if [ -z "$pass" ] && { [ -n "${PROVE_EXTRA_AST_DIR:-}" ] || [ "${PROVE_EXTRA_AST_ONLY:-0}" = 1 ]; }; then
    # AST dump for reach.py, or a throwaway check when AST_ONLY has no AST_DIR
    ast_out="${PROVE_EXTRA_AST_DIR:-$dir}/$suite.ast"
    [ -n "${PROVE_EXTRA_AST_DIR:-}" ] && mkdir -p "$PROVE_EXTRA_AST_DIR"
    (cd "$dir/out" && sail --no-warn --ddump-tc-ast --just-check --all-modules -D OPENTSP_PROOF_BUILD ${config[@]+"${config[@]}"} ../props.sail_project) > "$ast_out"
  fi
  # AST_ONLY must skip SMT even when PROVE_EXTRA_AST_DIR is unset
  [ "${PROVE_EXTRA_AST_ONLY:-0}" != 1 ] || continue
  # hold: props call model tick.sail hold_decide (postlude project); vector-literal views avoid path_views.
  # discharged by sail -smt like the other suites.
  proof_define=(-D OPENTSP_PROOF_BUILD)
  if [ "$suite" = domains ] || [ "$suite" = packing ]; then proof_define=(); width=(--smt-int-size 256); fi
  (cd "$dir/out" && sail -smt --all-modules ${proof_define[@]+"${proof_define[@]}"} ${width[@]+"${width[@]}"} ${config[@]+"${config[@]}"} ../props.sail_project)
  if [ "$suite" = fptheory ]; then
    # binary16 fround against the solver's floating-point theory (the query of the property is extended; see the script)
    "$py" "$spec/test/formal/fround_fp_theory.py" "$src" "$dir/out" "$inv" || { echo "prove_extra: the fptheory suite failed" >&2; exit 1; }
    continue
  fi
  "$py" - "$spec" "$src" "$dir/out" "$inv" "$twins" <<'PY'
import sys
sys.path.insert(0, sys.argv[1] + "/tools")
import os
import re
import subprocess
import check_smt_obligations as c
from pathlib import Path
# decoding a symbolic bit pattern costs tens of seconds: a longer limit than the model's own obligations (60 s)
c.TIMEOUT, c.KILL_AFTER = 240, 305
def check_fail_fast(source, directory):
    """c.check, stopping at the first failing obligation (same verdict rule, same output)."""
    expected = c.obligations(source)
    files = {p.stem: p for p in Path(directory).glob("*.smt2")}
    if files.keys() != expected.keys():
        raise ValueError("SMT inventory mismatch: missing=%s unexpected=%s" %
                         (sorted(expected.keys() - files.keys()), sorted(files.keys() - expected.keys())))
    for name, wanted in expected.items():
        query = re.sub(r"(?m)^\(get-model\)\s*$", "", files[name].read_text())
        result = subprocess.run(["z3", "-in", "-T:%d" % c.TIMEOUT], input=query, text=True, capture_output=True, timeout=c.KILL_AFTER)
        status = c.report_solver_result(name, wanted, result)
        if status == "error":
            raise c.SolverProtocolError("solver protocol failure: " + name)
        if status != "pass":
            raise c.UnprovedObligations("unproved obligation: " + name)


try:
    source = Path(sys.argv[2]).read_text()
    c.check_vacuity(source)
    c.check_inventory(source, Path(sys.argv[4]).read_text())
    checked = source + ("\n" + c.twin_source(source) if sys.argv[5] == "1" else "")
    # WIDE_QFBV_NORMALIZER_BEGIN
    # The 256-bit wide query is quantifier-free bitvector arithmetic. Its
    # automatic solver strategy times out on p_wide_replace. Its dedicated
    # qfbv strategy proves the identical assertions within the same limits, keeping
    # every declaration/assertion and all existing verdict/timeout checks.
    if Path(sys.argv[2]).stem == "wide":
        for path in Path(sys.argv[3]).glob("*.smt2"):
            original = path.read_text()
            query, count = re.subn(r"(?m)^\(check-sat\)[ \t]*$", "(check-sat-using qfbv)", original)
            if count != 1:
                raise ValueError("wide solver strategy requires exactly one check-sat: " + str(path))
            path.with_suffix(".smt2.original").write_text(original)
            path.write_text(query)
    # WIDE_QFBV_NORMALIZER_END
    if os.environ.get("PROVE_FAIL_FAST") == "1":
        check_fail_fast(checked, sys.argv[3])
    else:
        c.check(checked, sys.argv[3])
except c.UnprovedObligations as error:
    print("prove_extra: " + str(error), file=sys.stderr)
    sys.exit(1)
except (ValueError, OSError, c.subprocess.SubprocessError) as error:
    print("ERROR SMT prove_extra: " + str(error), file=sys.stderr)
    sys.exit(1)
PY
done
done
if [ "${PROVE_EXTRA_AST_ONLY:-0}" = 1 ]; then
  echo "prove_extra: ast-only: suites type-checked ($suites)"
elif [ "$passes" = default ]; then
  echo "prove_extra: all suites proved ($suites)"
else
  echo "prove_extra: all suites proved ($suites at both integer widths)"
fi
