#!/usr/bin/env bash
# The bounded-exhaustive test of the stream register file ring (srf_ring.sail; about 1 minute per profile).
#   srf_ring_test.sh [profile ...]      default: mini ref
exec bash "$(dirname "$0")/c_test.sh" srf_ring.sail "$@"
