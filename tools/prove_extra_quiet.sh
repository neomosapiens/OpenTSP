#!/usr/bin/env bash
# Run prove_extra.sh only when the host meets the selected load budget.
# Owner gate for  wide close: refuse when 1-minute load >= LOAD_MAX (default 20) or when
# any f16_exhaustive worker is live (those jobs consume the numerical proof budget).
# Usage: tools/prove_extra_quiet.sh [suite ...]
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
LOAD_MAX="${LOAD_MAX:-20}"
load1="$(uptime | awk -F'load averages?: ' '{print $2}' | awk -F', ' '{print $1}')"
# awk comparison as float
too_busy="$(awk -v l="$load1" -v m="$LOAD_MAX" 'BEGIN { print (l+0 >= m+0) ? 1 : 0 }')"
if [ "$too_busy" = 1 ]; then
  echo "prove_extra_quiet: refuse: 1-minute load $load1 >= LOAD_MAX=$LOAD_MAX" >&2
  exit 3
fi
if pgrep -f 'f16_exhaustive' >/dev/null 2>&1; then
  echo "prove_extra_quiet: refuse: f16_exhaustive worker(s) still live" >&2
  exit 3
fi
if [ "$#" -lt 1 ]; then
  echo "prove_extra_quiet: usage: tools/prove_extra_quiet.sh <suite> [suite ...]" >&2
  echo "prove_extra_quiet: refuse: refusing default all-suites (explicit suites required)" >&2
  exit 2
fi
echo "prove_extra_quiet: load $load1 < $LOAD_MAX and no f16_exhaustive; running prove_extra $*"
exec bash "$here/opentsp-arch-spec/test/formal/prove_extra.sh" "$@"
