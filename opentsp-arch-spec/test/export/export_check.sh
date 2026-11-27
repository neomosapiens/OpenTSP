#!/usr/bin/env bash
# Export check: the production Sail model must be accepted by the three theorem-prover backends,
# Lean 4 (`sail --lean`), Rocq (`sail --coq`) and Lem (`sail --lem`), for one profile.
#
# What this checks: Sail's code generators run to the end on every production module (proof.sail_project:
# tsp.sail_project without unit_tests and main). It does NOT check that the generated Lean or Rocq elaborates:
# the callbacks of core/callbacks.sail are still undefined in the export.
#
# usage: [EXPORT_CHECK_BACKENDS="lean coq lem"] export_check.sh [profile]   (profile: mini (default), ref, or any config/<profile>.json)
# exit 0: every check passed; 1: a check failed (the failing logs are printed).
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
test_dir="$here/test/export"
model="${OPENTSP_MODEL_DIR:-$here/model}"   # a copy of the model directory, for mutation runs
profile="${1:-mini}"
cfg="$here/config/$profile.json"
[ -f "$cfg" ] || { echo "export_check: no profile $profile ($cfg)" >&2; exit 2; }
selection="${EXPORT_CHECK_BACKENDS-lean coq lem}"
read -r -a backends <<< "${selection//$'\n'/ }"
if [ "${#backends[@]}" -eq 0 ]; then
  echo "export_check: no export backends selected" >&2
  exit 2
fi
seen=" "
for be in "${backends[@]}"; do
  case "$seen" in
    *" $be "*) echo "export_check: repeated backend $be" >&2; exit 2 ;;
  esac
  seen+="$be "
  case "$be" in
    lean|coq|lem) ;;
    *) echo "export_check: unknown backend $be" >&2; exit 2 ;;
  esac
done
command -v sail >/dev/null || { echo "export_check: sail not found (source the toolchain env.sh)" >&2; exit 2; }
work="$(mktemp -d)"
# The scratch directory holds symbolic links to the model directories: find without -L deletes the links, not their targets.
trap 'find "$work" -mindepth 1 -delete; rmdir "$work"' EXIT
# Sail's backends reject a `config` key that the profile file leaves out: they read the profile with its omitted defaulted keys set.
"${PYTHON:-python3}" "$here/tools/schema_invariants.py" --normalise "$cfg" "$work/profile.json"
cfg="$work/profile.json"
fail=0
bad() { echo "FAIL: $*"; fail=1; }

# 1. proof.sail_project lists the production files of the model project and nothing else.
files_of() { grep -vE '^[[:space:]]*//' "$1" | tr ', \t' '\n\n\n' | grep -E '\.sail$' | sort; }
files_of "$model/tsp.sail_project" | grep -v '^unit_tests/' | grep -v '^main/' > "$work/expect.txt"
files_of "$test_dir/proof.sail_project" > "$work/have.txt"
if diff -u "$work/expect.txt" "$work/have.txt" > "$work/drift.txt"; then
  echo "ok: proof.sail_project lists the $(wc -l < "$work/have.txt" | tr -d ' ') production files of tsp.sail_project"
else
  cat "$work/drift.txt"; bad "proof.sail_project differs from the production files of model/tsp.sail_project"
fi

# 2. No `while (match ...)` condition anywhere in the model: the Lean and Rocq backends stop with an internal error on it
#    (a.sail below is the reproducer). Use a helper function or a flag variable instead.
if grep -rnE 'while[[:space:]]*\([[:space:]]*match' "$model" --include='*.sail'; then
  bad "a while condition that is a match expression (use a helper function, see model/slices/mxm/mxm_insts.sail nonempty_list)"
else
  echo "ok: no while (match ...) condition in model/"
fi

# 2b. No range-cast escape in the matrix unit: the indices are typed arithmetic, so the checker proves them and no
#     internal_error stands between a type and its bound.
if grep -nE 'internal_error' "$model"/slices/mxm/*.sail; then
  bad "an internal_error in model/slices/mxm: a range cast that typed arithmetic (min, a range-typed parameter) removes"
else
  echo "ok: no internal_error in model/slices/mxm"
fi

# 3. The 8-line reproducer a.sail: a negative control. While Sail has the defect, Lean and Rocq must reject it with the
#    type-rewrite internal error; b.sail (the same function with a helper in the condition) must export on both.
#    If a.sail starts to pass, the compiler is fixed: the message says so and the check in step 2 may be relaxed.
repro() { # name backend
  local d="$work/repro_$1_$2"; mkdir -p "$d/out"
  case "$2" in
    lean) (cd "$d" && sail --lean --lean-output-dir "$d/out" --lean-force-output --all-modules "$test_dir/$1.sail") > "$d/log" 2>&1 ;;
    coq)  (cd "$d" && sail --coq -o r --all-modules "$test_dir/$1.sail") > "$d/log" 2>&1 ;;
  esac
}
for be in lean coq; do
  if repro a "$be"; then
    echo "NOTE: a.sail exports on $be: the Sail defect is fixed; step 2 may be relaxed"
  elif grep -q 'Cannot infer type of: match' "$work/repro_a_$be/log"; then
    echo "ok: a.sail still fails on $be with the known internal error (negative control)"
  else
    cat "$work/repro_a_$be/log"; bad "a.sail failed on $be for another reason"
  fi
  if repro b "$be"; then echo "ok: b.sail exports on $be"; else cat "$work/repro_b_$be/log"; bad "b.sail (helper in the while condition) does not export on $be"; fi
done

# 4. The model, three backends in parallel (each in its own directory; project paths resolve through links).
for dir in "$model"/*/; do ln -s "$dir" "$work/$(basename "$dir")"; done
cp "$test_dir/proof.sail_project" "$work/proof.sail_project"
run_backend() { # backend; writes $work/<backend>/{log,status,secs,lines}
  local be="$1" d="$work/$1" t0 t1
  mkdir -p "$d/out"
  t0=$(date +%s)
  case "$be" in
    lean) (cd "$d" && sail --lean --lean-output-dir "$d/out" --lean-force-output --all-modules --config "$cfg" "$work/proof.sail_project") > "$d/log" 2>&1 ;;
    coq)  (cd "$d" && sail --coq -o tsp --all-modules --config "$cfg" "$work/proof.sail_project") > "$d/log" 2>&1 ;;
    lem)  (cd "$d" && sail --lem --lem-output-dir "$d" -o tsp --all-modules --config "$cfg" "$work/proof.sail_project") > "$d/log" 2>&1 ;;
  esac && echo ok > "$d/status" || echo FAIL > "$d/status"
  t1=$(date +%s)
  echo $((t1 - t0)) > "$d/secs"
  case "$be" in
    lean) find "$d/out" -name '*.lean' -print0 | xargs -0 cat | wc -l ;;
    coq)  cat "$d"/*.v 2>/dev/null | wc -l ;;
    lem)  cat "$d"/*.lem 2>/dev/null | wc -l ;;
  esac > "$d/lines"
}
workers=()
for be in "${backends[@]}"; do
  mkdir -p "$work/$be"
  run_backend "$be" &
  workers+=("$!")
done
for j in "${!workers[@]}"; do
  wait "${workers[$j]}" || bad "export worker for ${backends[$j]} failed"
done
# 5. internal_error must be SMT-translatable in a proof build (-D OPENTSP_PROOF_BUILD) and stays unchanged in the default build,
#    where the SMT backend stops at dec_str: the second run is the negative control.
smt_project() { # directory, file: a project of the model's prelude and one file
  mkdir -p "$1"; ln -s "$model/prelude" "$1/prelude"; cp "$test_dir/$2" "$1/$2"
  printf 'prelude {\n  files\n    prelude/prelude.sail,\n    prelude/errors.sail,\n}\n\nt {\n  requires prelude\n  files %s\n}\n' "$2" > "$1/t.sail_project"
}
smt_project "$work/errors" errors_smt.sail
mkdir -p "$work/errors/proof" "$work/errors/plain"
if (cd "$work/errors/proof" && sail -smt --all-modules -D OPENTSP_PROOF_BUILD ../t.sail_project) > "$work/errors/proof.log" 2>&1 && [ -s "$work/errors/proof/p_pick.smt2" ]; then
  echo "ok: internal_error translates to SMT in a proof build"
else
  cat "$work/errors/proof.log"; bad "internal_error does not translate to SMT with -D OPENTSP_PROOF_BUILD"
fi
if (cd "$work/errors/plain" && sail -smt --all-modules ../t.sail_project) > "$work/errors/plain.log" 2>&1; then
  bad "the default build of internal_error translated to SMT: the proof-build switch is not needed any more (or the check is wrong)"
elif grep -q 'dec_str' "$work/errors/plain.log"; then
  echo "ok: the default build still stops at dec_str (negative control)"
else
  cat "$work/errors/plain.log"; bad "the default build failed for another reason"
fi

# 6. ilog2 on 20-bit inputs: 2^k <= n < 2^(k+1) holds for k = ilog2(n) (z3: unsat), for the nested form of every build and for the flat form
#    of the proof build (-D OPENTSP_PROOF_BUILD, the one every SMT obligation uses; ). Needs z3.
if command -v z3 >/dev/null; then
  for form in nested flat; do
    d="$work/ilog2_$form"; define=(); [ "$form" = flat ] && define=(-D OPENTSP_PROOF_BUILD)
    mkdir -p "$d"; ln -s "$model/prelude" "$d/prelude"; ln -s "$model/core/numerics" "$d/numerics"
    cp "$test_dir/ilog2_smt.sail" "$d/ilog2_smt.sail"
    printf 'prelude {\n  files\n    prelude/prelude.sail,\n    prelude/errors.sail,\n}\n\nnumerics {\n  requires prelude\n  files numerics/fp.sail\n}\n\nt {\n  requires prelude, numerics\n  files ilog2_smt.sail\n}\n' > "$d/t.sail_project"
    if (cd "$d" && sail -smt --all-modules ${define[@]+"${define[@]}"} t.sail_project) > "$d/sail.log" 2>&1 && [ "$(z3 -T:60 "$d/p_ilog2.smt2")" = unsat ]; then
      echo "ok: ilog2 ($form form) satisfies 2^k <= n < 2^(k+1) on every 20-bit input (z3: unsat)"
    else
      cat "$d/sail.log"; bad "the ilog2 lemma on 20-bit inputs does not hold (or does not translate) in the $form form"
    fi
  done
else
  bad "z3 not found (source the toolchain env.sh)"
fi

# 6b. The two forms of ilog2 agree with the loop they replaced (C backend: every power of two to 2^8300 with its neighbours, and 1000
#     random integers per form); the recorded long run uses 100,000,000.
if OPENTSP_MODEL_DIR="$model" bash "$test_dir/ilog2_equiv.sh" 1000 > "$work/ilog2_equiv.log" 2>&1 && [ "$(grep -c 'no mismatch' "$work/ilog2_equiv.log")" = 2 ]; then
  echo "ok: ilog2_equiv.sh: the nested and the flat form agree with the loop"
else
  cat "$work/ilog2_equiv.log"; bad "ilog2_equiv.sh: a form of ilog2 differs from the loop (or the check did not run)"
fi

# 7. The number of `while` loops in the generated Lean may only fall (test/export/lean_while.py, recorded in lean_while_count.txt): a loop is
#    a Lean `partial` definition, which no proof can unfold. Needs the Lean backend of step 4.
case " ${backends[*]} " in
  *" lean "*)
    if python3 "$test_dir/lean_while.py" "$work/lean/out" "$test_dir/lean_while_count.txt" > "$work/lean_while.log"; then
      head -n 1 "$work/lean_while.log"
    else
      cat "$work/lean_while.log"; bad "the generated Lean has a different number of while loops than test/export/lean_while_count.txt"
    fi ;;
  *) echo "skipped: the while-loop count needs the Lean backend" ;;
esac

echo "profile $profile:"
printf '  %-6s %-6s %8s %12s\n' backend status seconds output-lines
for be in "${backends[@]}"; do
  st="$(cat "$work/$be/status")"
  lines="$(tr -d ' ' < "$work/$be/lines")"
  printf '  %-6s %-6s %8s %12s\n' "$be" "$st" "$(cat "$work/$be/secs")" "$lines"
  if [ "$lines" -eq 0 ]; then bad "export backend $be wrote no output"; fi
  if [ "$st" != ok ]; then tail -n 30 "$work/$be/log"; bad "sail --$be does not accept the model on $profile"; fi
done
echo "note: the generated Lean is elaborated, with the harness callbacks defined, by test/formal/lean/lean_check.sh; the Rocq output is not elaborated"

if [ "$fail" -ne 0 ]; then echo "export_check: FAILED"; exit 1; fi
echo "export_check: all checks passed ($profile)"
