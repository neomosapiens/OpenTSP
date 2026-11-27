#!/usr/bin/env bash
# The ISS in place of tsp_sim for the scripts that read the program name in its messages
# (`tsp_sim: input line N: ...`): runs $TSP_ISS (default target/release/tsp-iss) with the same
# arguments and standard input, and names itself tsp_sim on standard error. Exit code and
# standard output are the ISS's.
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
iss="${TSP_ISS:-$root/target/release/tsp-iss}"
err="$(mktemp)"
trap 'rm -f "$err"' EXIT
"$iss" "$@" 2>"$err"
rc=$?
sed -e 's/^tsp-iss:/tsp_sim:/' "$err" >&2
exit "$rc"
