#!/usr/bin/env bash
# All model checks except the SMT proofs (tools/model_prove.sh). Needs sail 0.20.3, a C
# compiler, GMP and python3 with pyyaml. See opentsp-arch-spec/docs/validation.md.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
py="${PYTHON:-python3}"
scratch="$(mktemp -d)"
sim="$scratch/tsp_sim"
trap 'rm -rf "$scratch"' EXIT
# The profiles whose golden vectors are generated and replayed (the test profiles; ref and mini have their frozen files), ref_slow last.
VARIANTS="test/mid_min test/fetch_alt test/link_tight test/c2c_tx_delayed test/ref_small test/mini_small test/ref_hw test/sl8 test/sl32 test/ref_patrol test/ref_slow test/w14 test/w15 test/w16 test/acc8 test/f16_tandem test/fp8_rate2 test/one_plane test/mxm_wide test/c2c_wide test/ext_bf16 test/mini_ext test/ref_ext test/convert_src test/zz_typed test/all_ext test/mini_dot"
vectors_variants() {
  local vec; vec="$scratch/vectors"; mkdir -p "$vec"
  local jobs="${MODEL_VECTOR_JOBS:-1}"
  [[ "$jobs" =~ ^([1-9]|[12][0-9]|3[0-2])$ ]] || { echo "MODEL_VECTOR_JOBS must be 1..32" >&2; return 2; }
  if [ "$jobs" -eq 1 ]; then
    for c in $VARIANTS; do
      "$py" "$here/tools/gen_model_vectors.py" "$sim" "$here/config/$c.json" "$vec/$(basename "$c").json" | tail -2
      "$py" "$here/tools/gen_model_vectors.py" --check "$sim" "$here/config/$c.json" "$vec/$(basename "$c").json" | tail -3
    done
    return
  fi
  # Each profile has its own destination/log; reap every child and retain every failure.
  command -v setsid >/dev/null || { echo "parallel vector profiles require setsid" >&2; return 2; }
  local pids=() logs=() rc=0 log pid
  # Each child owns a process group, including its generator and simulator descendants.
  vector_children_stop() {
    local child
    for child in "${pids[@]}"; do kill -TERM -- "-$child" "$child" 2>/dev/null || true; done
    sleep 0.1
    for child in "${pids[@]}"; do kill -KILL -- "-$child" "$child" 2>/dev/null || true; done
    for child in "${pids[@]}"; do wait "$child" 2>/dev/null || true; done
    pids=()
  }
  trap 'vector_children_stop; exit 130' INT
  trap 'vector_children_stop; exit 143' TERM
  for c in $VARIANTS; do
    log="$vec/$(basename "$c").log"; logs+=("$log")
    setsid bash -c '
      set -euo pipefail
      "$1" "$2/tools/gen_model_vectors.py" "$3" "$4" "$5" | tail -2 &&
        "$1" "$2/tools/gen_model_vectors.py" --check "$3" "$4" "$5" | tail -3
    ' vector-profile "$py" "$here" "$sim" "$here/config/$c.json" "$vec/$(basename "$c").json" >"$log" 2>&1 &
    pids+=("$!")
    if [ "${#pids[@]}" -ge "$jobs" ]; then
      wait "${pids[0]}" || rc=1
      pids=("${pids[@]:1}")
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  trap - INT TERM
  unset -f vector_children_stop
  for log in "${logs[@]}"; do cat "$log"; done
  return "$rc"
}

if [ "${1:-}" = "vectors-variants" ]; then       # just model-vectors-variants: only this step, on TSP_SIM or a fresh build
  if [ -n "${TSP_SIM:-}" ]; then sim="$TSP_SIM"; else bash "$here/tools/build_sim.sh" "$sim"; fi
  echo "== golden vectors of the variant profiles: generated, then replayed"; vectors_variants
  echo "model-test vectors-variants: all passed"; exit 0
fi
echo "== filter audit regressions"; "$py" "$here/test/numerics/test_filter_audit.py"
echo "== reference assembler API refuses a profile beyond the caps"; "$py" "$here/../tests/spec/test_tsp_as_api.py"
echo "== transcendental interval enclosure: actual Sail against exact rational bounds"
"$py" "$here/test/numerics/exp_enclosure.py"
echo "== transcendental helpers: the bounded loops against the while loops they replaced (about 1 minute)"; bash "$here/test/export/loops_equiv.sh"
"$py" "$here/test/directed/numerics_speed.py" --selftest
echo "== integer log speed: the cascade scans within 1.5 times of the one-bit loop (two compilations, about 2 minutes)"; "$py" "$here/test/directed/numerics_speed.py"
echo "== schema advisories"; "$py" "$here/test/config/test_schema_advisories.py"
echo "== a typed timing row of an extension type needs the extension"; "$py" "$here/test/config/test_typed_row_extension.py"
echo "== proof gate regressions"; "$py" "$here/test/config/test_smt_obligations.py"
echo "== typecheck";            bash "$here/tools/model_check.sh" --check
echo "== unit tests";           bash "$here/tools/model_unit_tests.sh" | grep -v "^Testing\|^Pass$"
echo "== provenance";           "$py" "$here/test/isa_db/check_provenance.py"
echo "== public contract text (RSqrt and cycle-number limits)"; "$py" "$here/test/isa_db/check_public_contract_text.py"
echo "== generated docs";       "$py" "$here/tools/gen_isa_doc.py" --check
echo "== FP8 digests: the unit test of the model equals the independent oracle"; "$py" "$here/test/numerics/gen_fp8_digests.py" --check
echo "== simulator-lint registry"; "$py" "$here/tools/check_lints.py"
echo "== emulator";                bash "$here/tools/build_sim.sh" "$sim"
echo "== escaped configuration defaults"; "$py" "$here/test/config/escaped_defaults.py" "$sim"
echo "== multi-chip coordinator resource ownership"; "$py" "$here/test/config/chip_resources.py" "$sim"
echo "== profile identity and vector-generator regressions"; TSP_SIM="$sim" "$py" "$here/test/config/test_profile_names.py"
echo "== catalogue, reference assembler"; TSP_SIM="$sim" bash "$here/tools/isa_db_check.sh" "${SAMPLES:-40}" "${SEED:-1}"
echo "== directed tests (every latency, the superlane count, words per slice and rows from the profile: ref, the slower test profile ref_slow and mini; )"
for c in ref test/ref_slow test/ref_hw mini; do "$py" "$here/test/directed/directed.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== C2C and harness tests";  "$py" "$here/test/c2c/c2c_directed.py" "$sim" "$here/config/ref.json" | tail -1
echo "== C2C and harness tests (mini)"; "$py" "$here/test/c2c/c2c_directed.py" "$sim" "$here/config/mini.json" | tail -1
echo "== DeskewRecv: the polling loop at the HAC epoch, one chip and two chips, on every profile it is run on"
for c in ref mini test/ref_small test/mini_small test/mid_min test/ref_slow test/link_tight test/fetch_alt test/c2c_tx_delayed test/sl8 test/sl32 test/c2c_wide; do "$py" "$here/test/c2c/deskewrecv_directed.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== multi-chip start skew: fixed offsets break past the link margin, the Deskew pattern does not"
for c in ref mini test/ref_small test/mini_small test/mid_min test/ref_slow test/ref_hw test/link_tight test/fetch_alt test/c2c_tx_delayed test/sl8 test/sl32 test/c2c_wide; do "$py" "$here/test/c2c/skew_directed.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== relay over three chips: A -> B -> C, the cost of a hop, LNK-001 and LNK-003 on the relay's link"
for c in ref mini test/ref_small test/mini_small test/mid_min test/ref_slow test/ref_hw test/link_tight test/fetch_alt test/c2c_tx_delayed test/sl8 test/sl32 test/c2c_wide; do "$py" "$here/test/c2c/relay_directed.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== wide chip and link numbers, topology files and --chips K for any K: tsp_sim alone here, tsp-iss in just iss-wide"
"$py" "$here/test/c2c/wide_directed.py" "$sim" - "$here/config/mini.json" "$here/config/test/c2c_wide.json" | tail -1
echo "== published weight-install bound: the first ABC that sees every weight, against the derived bound"
for c in ref mini test/ref_slow test/ref_hw test/sl8; do "$py" "$here/test/directed/published_bounds.py" "$sim" "$here/config/$c.json" | tail -1; done
"$py" "$here/test/directed/published_bounds.py" "$sim" "$here/../tests/perf/profiles/iw3mem3.json" | tail -1
echo "== instruction front end"; for c in ref mini; do "$py" "$here/test/directed/frontend.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== superlane count 1..32: SetVL 1..32 and its clamp, Config, the last superlane, the error record"; for c in ref mini test/ref_small test/mini_small test/mid_min test/ref_slow test/sl8 test/sl32; do "$py" "$here/test/directed/superlanes.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== SRAM address width 13..16: the last word, the bank bit, IfetchM, Gather, SetSegment, Step"; for c in ref mini test/ref_small test/mini_small test/mid_min test/ref_slow test/w14 test/w15 test/w16; do "$py" "$here/test/directed/sram_width.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== profile parameters"; for c in ref mini test/ref_small test/mini_small test/mid_min test/ref_slow test/ref_hw test/sl8 test/sl32 test/w14 test/w16; do "$py" "$here/test/directed/profile_params.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== timing rows: override rows, the profile keys, f16 rate, Repeat spacing"; for c in ref mini; do "$py" "$here/test/directed/timing_rows.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== FP8 element types and matrix products: every operand pair against the oracle"; for c in ref mini; do "$py" "$here/test/directed/fp8.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== optional extensions: the decoder gate on every profile, the wide accumulator and bfp8 on the profile that declares them"; for c in ref mini test/mini_ext test/ref_ext; do "$py" "$here/test/directed/ext_types.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== bf16: refused without the extension (ref, mini), every operation, conversion and the matrix product against the oracle with it (test/ext_bf16)"; for c in ref mini test/ext_bf16; do "$py" "$here/test/directed/bf16.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== profile-generic directed tests on the test profiles"
for c in test/ref_small test/mini_small test/mid_min test/link_tight; do
  "$py" "$here/test/c2c/c2c_directed.py" "$sim" "$here/config/$c.json" | tail -1
  "$py" "$here/test/directed/frontend.py" "$sim" "$here/config/$c.json" | tail -1
done
# ref_slow: the front-end tests are timed from the profile and pass on it; c2c_directed.py is not (its link latencies and
# MEM/SXM timing are those of ref), so it runs on ref and the test profiles above only.
"$py" "$here/test/directed/frontend.py" "$sim" "$here/config/test/ref_slow.json" | tail -1
echo "== manifests"; for c in ref mini test/ref_small test/mini_small test/mid_min test/link_tight test/sl8 test/sl32 test/c2c_wide; do "$py" "$here/test/c2c/manifest_test.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== host path"; for c in ref mini test/ref_small test/mini_small test/mid_min test/ref_slow test/ref_hw test/link_tight test/sl8 test/sl32 test/c2c_wide; do "$py" "$here/test/host/host_directed.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== SECDED ECC"; for c in ref mini test/ref_small test/mini_small test/mid_min test/ref_slow test/ref_hw test/link_tight test/fetch_alt test/c2c_tx_delayed test/sl8 test/sl32; do "$py" "$here/test/directed/ras.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== error CSRs and error interrupt"; for c in ref mini test/ref_small test/mini_small test/mid_min test/ref_slow test/ref_hw test/link_tight test/fetch_alt test/c2c_tx_delayed test/sl8 test/sl32 test/c2c_wide; do "$py" "$here/test/directed/ras_csr.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== chip diagnostic log of the CSR space (microarchitecture.md sec. 16.3)"; for c in ref mini test/ref_small test/mini_small test/mid_min test/ref_slow test/ref_hw test/link_tight test/fetch_alt test/c2c_tx_delayed test/sl8 test/sl32; do "$py" "$here/test/directed/ras_diag.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== C2C link FEC"; for c in ref mini test/ref_small test/mini_small test/mid_min test/ref_slow test/ref_hw test/link_tight test/fetch_alt test/c2c_tx_delayed test/sl8 test/sl32 test/c2c_wide; do "$py" "$here/test/directed/link_fec.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== matrix-unit error detection"; for c in ref mini test/ref_slow test/ref_hw test/c2c_tx_delayed test/sl8 test/sl32; do "$py" "$here/test/directed/mxm_check.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== replay policy"; for c in ref mini test/ref_small test/mini_small test/mid_min test/ref_slow test/ref_hw test/link_tight test/fetch_alt test/c2c_tx_delayed test/sl8 test/sl32; do "$py" "$here/test/directed/ras_replay.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== MXM and C2C poison, on the model and on the oracle"; for c in ref mini test/ref_small test/mini_small test/mid_min test/ref_slow test/ref_hw test/link_tight test/fetch_alt test/c2c_tx_delayed test/sl8 test/sl32; do "$py" "$here/test/directed/ras_poison.py" "$sim" "$here/config/$c.json" | tail -1; done; "$py" "$here/test/directed/ras_poison.py" oracle "$here/config/ref.json" | tail -1
echo "== replay marks of F link and F mxm"; for c in ref mini test/ref_slow test/ref_hw test/c2c_tx_delayed test/sl8 test/sl32; do "$py" "$here/test/directed/ras_replay_marks.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== software SRAM scrub"; for c in ref mini test/ref_small test/mini_small test/mid_min test/ref_slow test/ref_hw test/link_tight test/fetch_alt test/c2c_tx_delayed test/sl8 test/sl32; do "$py" "$here/test/directed/ras_scrub.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== spare-bit error reports, on the model and on the oracle"; for c in ref mini test/ref_small test/mini_small test/mid_min test/ref_slow test/ref_hw test/link_tight test/fetch_alt test/c2c_tx_delayed test/sl8 test/sl32; do "$py" "$here/test/directed/ras_spare.py" "$sim" "$here/config/$c.json" | tail -1; "$py" "$here/test/directed/ras_spare.py" oracle "$here/config/$c.json" | tail -1; done
echo "== patrol scrubber"; for c in ref mini test/ref_small test/mini_small test/mid_min test/ref_slow test/ref_hw test/link_tight test/fetch_alt test/c2c_tx_delayed test/sl8 test/sl32; do "$py" "$here/test/directed/ras_patrol.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== bank-conflict report in the spare bits, on the model and on the oracle"; for c in ref mini test/ref_small test/mini_small test/mid_min test/ref_slow test/ref_hw test/link_tight test/fetch_alt test/c2c_tx_delayed test/sl8 test/sl32; do "$py" "$here/test/directed/ras_bank.py" "$sim" "$here/config/$c.json" | tail -1; "$py" "$here/test/directed/ras_bank.py" oracle "$here/config/$c.json" | tail -1; done
echo "== patrol explicit-check-bit oracle"; for c in ref mini test/ref_slow test/ref_hw; do "$py" "$here/test/oracle/patrol_diff.py" "$sim" "$here/config/$c.json"; done
echo "== reader contracts"; for c in ref mini; do "$py" "$here/test/directed/reader_contract.py" "$sim" "$here/config/$c.json"; done
echo "== release architecture contract"; "$py" "$here/test/directed/execution_contract.py" "$sim" "$here/config/ref.json"
echo "== MXM structure keys (accumulator depth, f16 capacity, FP8 rate, planes per hemisphere)"; for c in ref mini test/acc8 test/f16_tandem test/fp8_rate2 test/one_plane test/mxm_wide; do "$py" "$here/test/directed/mxm_keys.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== FP8 pair rule (an ABC in the pair's commit cycle keeps both products; the Python oracle and the model agree)"; "$py" "$here/test/directed/fp8_pair.py" "$sim" "$here/config/test/fp8_rate2.json" | tail -1
echo "== invalid configurations"; for c in ref mini test/ref_small test/mini_small test/mid_min test/ref_slow test/ref_hw test/link_tight test/sl8 test/sl32 test/w14 test/w15 test/w16 test/acc8 test/f16_tandem test/one_plane test/mxm_wide test/c2c_wide; do "$py" "$here/test/config/config_negative.py" "$sim" "$here/config/$c.json" | tail -1; done
echo "== golden vectors and catalogue coverage"; for c in ref mini; do "$py" "$here/tools/gen_model_vectors.py" --check "$sim" "$here/config/$c.json" "$here/conformance/model/$c.json"; done
echo "== golden vectors of the variant profiles (the generator on every schema-legal timing and layout variant; generated, then replayed)"; vectors_variants
echo "== side vector sets: replayed on the model; every VXM (entry, type) pair, IQ-002 and IQ-003 (on ref and on mini), patrol and the F kinds are in them"
"$py" "$here/test/config/test_extra_vectors.py"
for c in ref mini test/ref_hw test/ref_patrol; do "$py" "$here/tools/gen_model_vectors.py" --extra --check "$sim" "$here/config/$c.json" "$here/conformance/model-extra/$(basename "$c").json"; done
echo "== patrol error-interrupt stop ordering"; for c in mini ref; do "$py" "$here/test/directed/ras_patrol_stop.py" "$sim" "$here/config/$c.json"; done
echo "== oracle differential"; "$py" "$here/test/oracle/diff_oracle.py" "$sim" "$here/config/ref.json" ${ORACLE_N:-300} 1
echo "== oracle differential (test profile ref_small: 1024 words, 8 MEM slices, 160 MXM rows)"; "$py" "$here/test/oracle/diff_oracle.py" "$sim" "$here/config/test/ref_small.json" ${ORACLE_N_SMALL:-100} 2
echo "== oracle differential (test profile ref_slow: MEM 7, VXM 8, SXM 7/24 latencies)"; "$py" "$here/test/oracle/diff_oracle.py" "$sim" "$here/config/test/ref_slow.json" ${ORACLE_N_SLOW:-100} 4
echo "== oracle differential (test profile link_tight: cycles_per_vector 3, rx_depth 2; link limits)"; "$py" "$here/test/oracle/diff_oracle.py" "$sim" "$here/config/test/link_tight.json" ${ORACLE_N_LINK:-100} 4 --only c2c_limits --no-directed
echo "== oracle differential (a whole-queue record meets an open AGT configuration)"; "$py" "$here/test/oracle/diff_oracle.py" "$sim" "$here/config/ref.json" ${ORACLE_N_WHOLE:-120} 3 --only agt_whole --no-directed
echo "== oracle differential (Ifetch on one stream operand: two consecutive vectors)"; "$py" "$here/test/oracle/diff_oracle.py" "$sim" "$here/config/ref.json" ${ORACLE_N_IFETCH:-150} 5 --only front_ifetch --no-directed
echo "== oracle differential (test profile ref_hw: multi-cycle Exp/Log/RSqrt/TanH/Mul/Reduce/Permute rows)"; "$py" "$here/test/oracle/diff_oracle.py" "$sim" "$here/config/test/ref_hw.json" ${ORACLE_N_HW:-100} 6
echo "== oracle differential (profiles of other geometry: mini 1 superlane, mid_min 4, sl8 8 superlanes, sl32 32 superlanes = 512 lanes)"
for c in mini test/mid_min test/sl8 test/sl32; do "$py" "$here/test/oracle/diff_oracle.py" "$sim" "$here/config/$c.json" ${ORACLE_N_GEO:-60} 7; done
echo "== oracle differential (profiles of other MXM structure: acc8 8 accumulators, f16_tandem 320 f16 rows, fp8_rate2, one_plane, mxm_wide 32 accumulators on 16 rows)"
for c in test/acc8 test/f16_tandem test/fp8_rate2 test/one_plane test/mxm_wide; do "$py" "$here/test/oracle/diff_oracle.py" "$sim" "$here/config/$c.json" ${ORACLE_N_MXM:-60} 8; done
echo "model-test: all passed"
