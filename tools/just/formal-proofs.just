# Component proofs, generated-model checks and numerical campaigns.

# The extra SMT obligations of test/formal/props/ (ties to even, the instruction
# queue refinement, exactness and rounding of the four formats): not part of `model-prove`, run by the mutation ratchet in front of it.
prove-extra:
    bash opentsp-arch-spec/test/formal/prove_extra.sh

# About 25 min and up to 22 GB for one query: the nightly gate, not a per-push one. The same obligations again with `--smt-int-size 256`
# (the default width of Sail's SMT generator is 128 bits, which hides a counterexample that needs a wider integer).
prove-extra-wide:
    PROVE_EXTRA_WIDE=1 bash opentsp-arch-spec/test/formal/prove_extra.sh

# A cold run fetches the pinned Lean library and builds the generated model.
# Needs `elan` with leanprover/lean4:v4.29.0. The production model elaborates in Lean 4 with the harness callbacks of gen_callbacks.py on
# `mini` and `ref`, and the proofs of test/formal/lean/proofs about it check on both (no sorry, no axiom beyond propext, Classical.choice,
# Quot.sound: check_axioms.py compares every `#print axioms` line).
lean-check:
    bash opentsp-arch-spec/test/formal/lean/lean_check.sh mini
    bash opentsp-arch-spec/test/formal/lean/lean_check.sh ref

# About 1 min per profile. The test of the stream register file ring over the real srf_write and srf_read: exhaustive over gaps, diagonal
# differences and orders of the writes (the first commit cycle and the values are fixed).
srf-ring-test:
    bash opentsp-arch-spec/test/formal/srf_ring_test.sh

# About 1 min. All 14 integer VXM lane operations (add, saturating add, subtract, saturating subtract, multiply, saturating multiply, max, min,
# and, or, xor) against an independent specification: every 8-bit pair, a 16-bit and a 32-bit lattice. The float arms are not covered.
lanes-test:
    bash opentsp-arch-spec/test/formal/lanes_test.sh

# Every binary16 add, sub and mul pair against the host's _Float16 arithmetic (the model's C backend).
# Run one operation at a time in the background (`just f16-exhaustive add`); F16_X_MAX=n runs the first n x values only (a smoke run).
f16-exhaustive *ops:
    JOBS="${JOBS:-16}" bash opentsp-arch-spec/test/numerics/f16_exhaustive.sh {{ops}}

# Smoke: a bounded slice of the same host `_Float16` differential (independent of encoding.json and the Python oracle).
# About 1 min (Sail C backend build dominates); F16_X_MAX=2 is 2 × 65536 pairs of binary16 add.
f16-smoke:
    F16_X_MAX="${F16_X_MAX:-2}" JOBS="${JOBS:-1}" bash opentsp-arch-spec/test/numerics/f16_exhaustive.sh add

# About 4 min per operation on 8 cores. 160 million binary32 add, sub and mul pairs per operation from four strata against the host's
# `float` arithmetic (a sample: 2^64 pairs cannot be run).
f32-sample:
    JOBS="${JOBS:-8}" bash opentsp-arch-spec/test/numerics/f32_sample.sh

# Regenerate the combined-extension vectors and RTL obligations outside the freeze set.
# Requires built tsp_sim and tsp-iss; the current dump has 17 directed extension programs.
all-ext-vectors:
    python3 opentsp-arch-spec/tools/gen_all_ext_vectors.py target/spec/tsp_sim opentsp-arch-spec/conformance/model-extra/all_ext.json
    python3 opentsp-arch-spec/tools/gen_rtl_obligations.py --iss target/release/tsp-iss --profile all_ext -j 8
