#!/usr/bin/env bash
# The exhaustive test of the integer VXM lane kernels (lanes_exhaustive.sail), on the mini profile (a few minutes).
exec bash "$(dirname "$0")/c_test.sh" lanes_exhaustive.sail "${@:-mini}"
