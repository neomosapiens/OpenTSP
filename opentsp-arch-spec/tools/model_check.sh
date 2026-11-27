#!/usr/bin/env bash
# Typecheck the Sail model for every profile, with sail-riscv's strictness flags, and
# write the generated config schema (completed by tools/schema_invariants.py). --all-modules is required: without it a
# .sail_project selects no modules and "typechecks" nothing.
#   model_check.sh          typecheck, then replace config/schema.json when the generated schema differs
#   model_check.sh --check  typecheck and diff the generated schema against config/schema.json without writing
#                           it (config/schema.json is a frozen file: a check must never rewrite it)
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
check=0
case "${1:-}" in
  "") ;;
  --check) check=1 ;;
  *) echo "usage: model_check.sh [--check]" >&2; exit 2 ;;
esac
flags=(--strict-var --strict-bitvector --strict-exponentials --all-modules --require-version 0.20.3)
cd "$here/model"
if [ "$check" -eq 1 ]; then schema_tmp="$(mktemp)"; else schema_tmp="$(mktemp "$here/config/.schema.XXXXXX")"; fi
trap 'rm -f "$schema_tmp"' EXIT
# The test profiles (config/test/) vary words_per_slice, mem_slices_per_hemisphere and mxm.rows.
for cfg in "$here"/config/ref.json "$here"/config/mini.json "$here"/config/test/*.json; do
  name="$(basename "$cfg" .json)"
  sail --just-check "${flags[@]}" --config "$cfg" --output-schema "$schema_tmp" tsp.sail_project
  echo "typecheck ok: $name ($(sail tsp.sail_project --all-modules --list-files-separated ' ' | wc -w) files)"
done
# Sail emits only per-key type bounds; add the cross-field invariants.
"${PYTHON:-python3}" "$here/tools/schema_invariants.py" "$schema_tmp"
# Never expose Sail's partial schema to concurrent loaders or freeze checks.
if ! cmp -s "$schema_tmp" "$here/config/schema.json"; then
  if [ "$check" -eq 1 ]; then
    echo "model_check: the generated schema differs from config/schema.json (not written):" >&2
    diff -u "$here/config/schema.json" "$schema_tmp" >&2 || true
    exit 1
  fi
  mv "$schema_tmp" "$here/config/schema.json"
fi
[ "$check" -eq 0 ] || echo "schema check ok: config/schema.json is the generated schema"
"${PYTHON:-python3}" "$here/tools/schema_invariants.py" --check "$here"/config/ref.json "$here"/config/mini.json "$here"/config/test/*.json
