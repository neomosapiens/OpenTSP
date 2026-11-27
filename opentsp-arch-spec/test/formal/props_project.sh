#!/usr/bin/env bash
# Assemble the proof project in a work directory: read-only links to the model's directories, the port-proof
# instantiation of the production transitions (tools/fetch_port_proof.py), the obligations text and the Sail project
# file that adds the props module to the model's modules. Used by tools/model_prove.sh and test/formal/model_coverage.sh.
#   props_project.sh WORKDIR
# PROPS_TWINS=1 adds twins.sail to the props module: the twin reachability query of every property (check_smt_obligations.twin_source).
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
work="${1:?usage: props_project.sh WORKDIR}"
mkdir -p "$work"
# Project-relative Sail sources resolve through these read-only directory links.
for dir in "$here"/model/*/; do ln -sfn "$dir" "$work/$(basename "$dir")"; done
"${PYTHON:-python3}" "$here/tools/fetch_port_proof.py" "$work/ports.sail"
cat "$here/model/props/props.sail" "$work/ports.sail" > "$work/obligations.txt"
twins=""
if [ "${PROPS_TWINS:-0}" = 1 ]; then
  "${PYTHON:-python3}" -c 'import sys; sys.path.insert(0, sys.argv[1] + "/tools"); import check_smt_obligations as c; print(c.twin_source(open(sys.argv[2]).read()))' \
    "$here" "$work/obligations.txt" > "$work/twins.sail"
  twins="    twins.sail,
"
fi
# A project = the model plus a props module that requires every model module.
{
  cat "$here/model/tsp.sail_project"
  printf '\nprops {\n  requires prelude, core, numerics, slice_types, sys, slices, postlude\n\n  files\n    props/props.sail,\n    ports.sail,\n%s}\n' "$twins"
} > "$work/props.sail_project"
