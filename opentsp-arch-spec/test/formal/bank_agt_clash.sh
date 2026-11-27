#!/usr/bin/env bash
#  mut_bank kill: build tsp_sim from this tree and run the two-AGT-loop MEM-001 case.
# Invoked as: bash bank_agt_clash.sh <tree-root>
set -euo pipefail
tree="${1:?tree root}"
here="$(cd "$(dirname "$0")" && pwd)"
out="/tmp/opentsp-bank-agt-$$"
mkdir -p "$out"
bash "$tree/tools/build_sim.sh" "$out/tsp_sim"
"${PYTHON:-python3}" "$here/bank_agt_clash.py" "$out/tsp_sim" "$tree/config/ref.json"
