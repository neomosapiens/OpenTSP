#!/usr/bin/env bash
# Elaborate the production model in Lean 4 (Lean is the proof assistant of the repository): `sail --lean` on every production module with the harness
# callbacks given as Lean declarations (gen_callbacks.py), then `lake build` of the generated project with the pinned toolchain
# (lean4 v4.29.0) and library (lean-sail v6). Prints the Lean line count and the phase times.
#   lean_check.sh [profile]      profile: mini (default) or any config/<profile>.json (the gate runs mini and ref)
# Environment: LEAN_CHECK_DIR  work directory (default /tmp/opentsp-lean-check-<profile>), kept: the Lean build cache lives in it
#              LEAN_CHECK_GEN_ONLY=1   stop after generating the Lean sources
#              OPENTSP_MODEL_DIR       a copy of the model directory (the mutation runs use it)
# Needs elan with the pinned toolchain (`elan toolchain install leanprover/lean4:v4.29.0`), about 2.6 GB, outside the repository.
set -euo pipefail
here="$(cd "$(dirname "$0")/../../.." && pwd)"
profile="${1:-mini}"
model="${OPENTSP_MODEL_DIR:-$here/model}"   # a copy of the model directory, for mutation runs
cfg="$here/config/$profile.json"
[ -f "$cfg" ] || { echo "lean_check: no profile $profile" >&2; exit 2; }
export PATH="$HOME/.elan/bin:$PATH"
command -v sail >/dev/null || { echo "lean_check: sail not found (source the toolchain env.sh)" >&2; exit 2; }
command -v lake >/dev/null || { echo "lean_check: lake not found (install elan)" >&2; exit 2; }
work="${LEAN_CHECK_DIR:-/tmp/opentsp-lean-check-$profile}"
mkdir -p "$work/gen"
# Sail's backends reject a `config` key that the profile file leaves out: they read the profile with its omitted defaulted keys set.
python3 "$here/tools/schema_invariants.py" --normalise "$cfg" "$work/profile.json"
cfg="$work/profile.json"
for dir in "$model"/*/; do ln -sfn "$dir" "$work/$(basename "$dir")"; done
cp "$here/test/export/proof.sail_project" "$work/proof.sail_project"   # the production modules of the model
python3 "$here/test/formal/lean/gen_callbacks.py" "$model" "$work/Env.lean"
t0=$(date +%s)
(cd "$work" && sail --lean --lean-output-dir "$work/gen" --lean-force-output --lean-import-file "$work/Env.lean" --lean-non-beq-type parcel \
  --all-modules --config "$cfg" "$work/proof.sail_project") > "$work/sail.log" 2>&1 || { tail -n 30 "$work/sail.log"; exit 1; }
t1=$(date +%s)
proj="$work/gen/out"
# Layout defects of Sail 0.20.3's Lean generator, patched by exact rules with expected counts (see patch_generated.py).
python3 "$here/test/formal/lean/patch_generated.py" "$proj/Out"
echo "lean_check: sail --lean $((t1 - t0)) s, $(cat "$proj"/Out/*.lean | wc -l | tr -d ' ') Lean lines, toolchain $(cat "$proj/lean-toolchain")"
[ "${LEAN_CHECK_GEN_ONLY:-0}" = 1 ] && exit 0
cd "$proj"
# The generated lakefile names the library by its tag (lean-sail v6). The committed manifest (test/formal/lean/lake-manifest.json) pins the
# revision that tag resolved to when the proofs were written, so that a later move of the tag cannot change what is checked.
cp "$here/test/formal/lean/lake-manifest.json" "$proj/lake-manifest.json"
pinned="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["packages"][0]["rev"])' "$proj/lake-manifest.json")"
lake build Out 2>&1 | tee "$work/lake.log" | tail -n 40
t2=$(date +%s)
actual="$(git -C "$proj/.lake/packages/Sail" rev-parse HEAD 2>/dev/null || echo none)"
[ "$actual" = "$pinned" ] || { echo "lean_check: lean-sail is at $actual, the manifest pins $pinned"; exit 1; }
echo "lean_check: lean-sail $pinned (pinned by lake-manifest.json)"
echo "lean_check: lake build Out $((t2 - t1)) s: $(/usr/bin/grep -c 'error' "$work/lake.log" || true) lines mention an error"
/usr/bin/grep -q "error" "$work/lake.log" && exit 1
echo "lean_check: the generated model elaborates ($profile)"
# The proofs about the generated model (test/formal/lean/proofs): a second library of the generated project. No `sorry`, and no
# axiom beyond the three of Lean's core (propext, Classical.choice, Quot.sound): native_decide (ofReduceBool) and any axiom a proof
# file declares are refused (check_axioms.py compares every `#print axioms` line of proofs/Axioms.lean with that set).
mkdir -p "$proj/Proofs"
cp -R "$here/test/formal/lean/proofs/." "$proj/Proofs/"
printf '\n[[lean_lib]]\nname = "Proofs"\nroots = ["Proofs.Common", "Proofs.IqPop", "Proofs.IqPush", "Proofs.Barrier", "Proofs.Resources", "Proofs.Numerics", "Proofs.Intervals", "Proofs.Enclosure", "Proofs.RangeReduction", "Proofs.Ln2Certificates", "Proofs.Frontend", "Proofs.Patrol", "Proofs.DispatchSlots", "Proofs.Rounding", "Proofs.SrfState", "Proofs.Capacity", "Proofs.Positivity", "Proofs.Reciprocal", "Proofs.Logarithm", "Proofs.LogSeries", "Proofs.Reset", "Proofs.Storage", "Proofs.RootEnclosure", "Proofs.ZivDomains", "Proofs.Binade", "Proofs.GridBounds", "Proofs.SignedRounding", "Proofs.Events", "Proofs.StorageRep", "Proofs.Initialized", "Proofs.StorageProduct", "Proofs.CallerBounds", "Proofs.DotBounds", "Proofs.EventFrames", "Proofs.FormatNearest", "Proofs.EndpointBounds", "Proofs.CarrierDomains", "Proofs.ScalarSemantics", "Proofs.LaneArithmetic", "Proofs.LanePolicies", "Proofs.FpSigns", "Proofs.ParcelFrame", "Proofs.ParcelBound", "Proofs.TextCheckProduct", "Proofs.DotSemantics", "Proofs.ReduceSemantics", "Proofs.PublicNumerics", "Proofs.InputDecode", "Proofs.Wide128Bounds", "Proofs.LanePlacement", "Proofs.EventQueries", "Proofs.OrderedImage", "Proofs.MachineInit", "Proofs.ScrubCatchup", "Proofs.OrderedDelivery", "Proofs.ResourceAdmission", "Proofs.GrantConsumption", "Proofs.ResourceFrames", "Proofs.ResourceRasFrames", "Proofs.PatrolFrames", "Proofs.PatrolCycleFrames", "Proofs.ResourceInvariant", "Proofs.PatrolOwnership", "Proofs.PatrolProtected", "Proofs.PatrolOwnedCycle", "Proofs.ResourceGeometry", "Proofs.SramStorage", "Proofs.PatrolMemory", "Proofs.PatrolValidity", "Proofs.IqReach", "Proofs.EventCensus", "Proofs.ResourcePairCases", "Proofs.ResourceExecuteFrames", "Proofs.OrderedFetch", "Proofs.ResourceReach", "Proofs.ResourcePresence", "Proofs.ResourceReady", "Proofs.BankEvaluator", "Proofs.BankDomain", "Proofs.TagCoverage", "Proofs.TaggedCheck", "Proofs.ControlConsumption", "Proofs.QueueConsumption","Proofs.AllocationHistory","Proofs.HistoryFrames","Proofs.HistoryComposition","Proofs.CombinedMachineInvariant","Proofs.HistoryContent","Proofs.RepairCell","Proofs.RepairView","Proofs.OverlayHistory","Proofs.EventUnparked","Proofs.EventQuiet","Proofs.EventScrubPrefix","Proofs.EventTimedView","Proofs.EventBarrierLazy","Proofs.EventBoundaries","Proofs.EventStageQuiet","Proofs.EventParcelMonotone","Proofs.EventAgtNext","Proofs.EventSearch","Proofs.EventBarrierHistory","Proofs.EventDispatch","Proofs.MxmChecksum","Proofs.MxmIntegrity","Proofs.MxmRowClassifier","Proofs.FailureEffects","Proofs.FailureAdmission","Proofs.MxmNominalRows","Proofs.Axioms", "Proofs.AgtExecutionFrames", "Proofs.AgtOriginReset", "Proofs.AgtOriginState", "Proofs.AgtOrigins", "Proofs.AgtTimeBound", "Proofs.AgtTimeClosure", "Proofs.AgtTimedHistory", "Proofs.DistinctWriterSeen", "Proofs.ExecuteProducer", "Proofs.ExecuteRingFrames", "Proofs.ExecutionHistory", "Proofs.ExecutionTimingBounds", "Proofs.HoldAdmission", "Proofs.HoldExecutionFrames", "Proofs.HoldQueryRel", "Proofs.ResourceComplete", "Proofs.ResourceReadLoops", "Proofs.ResourceScan", "Proofs.SlotAdmission", "Proofs.SrfBitPrimitives", "Proofs.SrfCommitDue","Proofs.SrfCommitPolicy","Proofs.SrfPolicyHistory","Proofs.SrfPairPolicy", "Proofs.SrfCommitJournal", "Proofs.SrfCommittedHistory", "Proofs.SrfFaultHistory", "Proofs.SrfHistoryFrames", "Proofs.SrfLifetime", "Proofs.SrfLoopTable", "Proofs.SrfMerge", "Proofs.SrfMergeSemantics", "Proofs.SrfNearest", "Proofs.SrfPoisonFlow", "Proofs.SrfReportFlow", "Proofs.SrfSpareSemantics", "Proofs.SrfUpstream", "Proofs.SrfValueFrames"]\n' >> "$proj/lakefile.toml"
# Capture every command's status: pipefail alone loses whether a status 1 came
# from the prover or from an output/log consumer (e.g. a failed tee write).
set +e
lake build Proofs 2>&1 | tee "$work/proofs.log" | /usr/bin/grep -v '^trace' | tail -n 30
proofs_pipeline=("${PIPESTATUS[@]}")
set -e
proofs_status=${proofs_pipeline[0]}
t3=$(date +%s)
if [ "${proofs_pipeline[1]}" -ne 0 ] || [ "${proofs_pipeline[2]}" -ne 0 ] || [ "${proofs_pipeline[3]}" -ne 0 ]; then
  echo "lean_check: proof output pipeline failed (statuses ${proofs_pipeline[*]})" >&2
  exit 1
fi
if [ "$proofs_status" -ne 0 ]; then
  # Errexit must not bypass the verdict. Mixed/unrecognized error diagnostics
  # fail closed: a theorem rejection cannot mask a dependency/build failure.
  theorem_error='^error: Proofs/(AgtExecutionFrames|AgtOriginReset|AgtOriginState|AgtOrigins|AgtTimeBound|AgtTimeClosure|AgtTimedHistory|AllocationHistory|BankDomain|Barrier|Binade|CallerBounds|Capacity|CarrierDomains|Common|CombinedMachineInvariant|ControlConsumption|DispatchSlots|DistinctWriterSeen|DotBounds|DotSemantics|Enclosure|EndpointBounds|EventAgtNext|EventBarrierHistory|EventBarrierLazy|EventBoundaries|EventCensus|EventDispatch|EventFrames|EventParcelMonotone|EventQueries|EventQuiet|EventScrubPrefix|EventSearch|EventStageQuiet|EventTimedView|EventUnparked|Events|ExecuteProducer|ExecuteRingFrames|ExecutionHistory|ExecutionTimingBounds|FailureAdmission|FailureEffects|FormatNearest|FpSigns|Frontend|GrantConsumption|GridBounds|HistoryComposition|HistoryContent|HistoryFrames|HoldAdmission|HoldExecutionFrames|HoldQueryRel|Initialized|InputDecode|Intervals|IqPop|IqPush|IqReach|LaneArithmetic|LanePlacement|LanePolicies|Ln2Certificates|LogSeries|Logarithm|MachineInit|MxmChecksum|MxmIntegrity|MxmNominalRows|MxmRowClassifier|Numerics|OrderedDelivery|OrderedFetch|OrderedImage|OverlayHistory|ParcelBound|ParcelFrame|Patrol|PatrolCycleFrames|PatrolFrames|PatrolMemory|PatrolOwnedCycle|PatrolOwnership|PatrolProtected|PatrolValidity|Positivity|PublicNumerics|QueueConsumption|RangeReduction|Reciprocal|ReduceSemantics|RepairCell|RepairView|Reset|ResourceAdmission|ResourceComplete|ResourceExecuteFrames|ResourceFrames|ResourceGeometry|ResourceInvariant|ResourcePresence|ResourceRasFrames|ResourceReach|ResourceReadLoops|ResourceReady|ResourceScan|Resources|RootEnclosure|Rounding|ScalarSemantics|ScrubCatchup|SignedRounding|SlotAdmission|SramStorage|SrfBitPrimitives|SrfCommitDue|SrfCommitJournal|SrfCommitPolicy|SrfPolicyHistory|SrfPairPolicy|SrfCommittedHistory|SrfFaultHistory|SrfHistoryFrames|SrfLifetime|SrfLoopTable|SrfMerge|SrfMergeSemantics|SrfNearest|SrfPoisonFlow|SrfReportFlow|SrfSpareSemantics|SrfState|SrfUpstream|SrfValueFrames|Storage|StorageProduct|StorageRep|TagCoverage|TaggedCheck|TextCheckProduct|Wide128Bounds|ZivDomains)\.lean:[0-9]+:[0-9]+: (Application type mismatch|Type mismatch|unsolved goals|[Tt]actic .*failed|omega could not prove the goal:|Tactic `decide` proved that the proposition)'
  allowed_error="$theorem_error|^error: Lean exited with code 1$|^error: build failed$"
  unclassified_errors=0
  # A nested-error wrapper does not establish what its inner failure means.
  if /usr/bin/grep -qE '^error: .*: [Tt]actic .*failed with a nested error:' "$work/proofs.log"; then
    unclassified_errors=1
  fi
  # Nested tactic failures can wrap resource exhaustion without an error: prefix.
  # They are infrastructure failures, never semantic theorem rejections.
  if /usr/bin/grep -qE '^[[:space:]]*(maximum recursion depth has been reached|maximum number of heartbeats|(\(deterministic\) )?timeout at .*maximum number of heartbeats)' "$work/proofs.log"; then
    unclassified_errors=1
  fi
  if /usr/bin/grep -Ei 'error:|fatal:|exception:' "$work/proofs.log" | /usr/bin/grep -Ev "$allowed_error" >/dev/null; then
    unclassified_errors=1
  fi
  if [ "$proofs_status" -eq 1 ] && [ "$unclassified_errors" -eq 0 ] && /usr/bin/grep -qE "$theorem_error" "$work/proofs.log"; then
    echo "lean_check: a proof does not check"
  else
    echo "lean_check: proof build failed without a theorem rejection (exit $proofs_status)" >&2
  fi
  exit "$proofs_status"
fi
/usr/bin/grep -qEi "error:|fatal:|exception:" "$work/proofs.log" && { echo "lean_check: proof build reported errors without a failure status" >&2; exit 1; }
/usr/bin/grep -qE "sorryAx|ofReduceBool|declaration uses .sorry" "$work/proofs.log" && { echo "lean_check: a proof uses sorry or native_decide"; exit 1; }
python3 "$here/test/formal/lean/check_axioms.py" "$work/proofs.log" "$proj/Proofs/Axioms.lean" "$proj/Proofs" || { echo "lean_check: an axiom outside the allowed set, or a theorem without its axiom line"; exit 1; }
theorems="$(python3 "$here/test/formal/lean/check_axioms.py" "$work/proofs.log" "$proj/Proofs/Axioms.lean" "$proj/Proofs" | sed 's/.*theorems=\([0-9]*\).*/\1/')"
echo "lean_check: lake build Proofs $((t3 - t2)) s: $theorems theorems checked, none uses sorry, native_decide or an axiom of its own"
echo "lean_check: ok profile=$profile theorems=$theorems"
