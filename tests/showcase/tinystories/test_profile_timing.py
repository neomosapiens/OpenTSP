"""The scheduler derives its timing from the profile. The serving step is emitted for `test/ref_slow`
(the geometry of `ref`, slower MEM, SXM and VXM, a longer chained add, slower MXM ABC occupancy, no timing overrides) and run on
the ISS against bit-exact reference (the host reference): every pool word, the final hidden state and the logits equal bit for bit, with no diagnostic
and no lint, for the single-lane step at three contexts and for the sample of lanes of `test_step_buckets.py` (several lanes, logit
rows, shared prefix blocks, padding lanes, C = 64 and 128); once on `tsp_sim`. Every bucket of the grid builds and pages
(slow). `ref` stays byte-identical (`test_step_batch.py`, the digests of the report).
`test/fetch_alt`, `test/ref_small`, `test/mid_min` and `mini` have another geometry: `test_profile_needs.py` pins their refusal."""
import copy
import json
import os

import pytest


from opentsp_vllm.models.tinystories import emit_step_batch as E
import test_step_batch as TS
import test_step_buckets as SBK
from opentsp_vllm.models.tinystories import text_sizing as TX
from opentsp_tools import profile as tprofile
from opentsp_tools.stations import ScheduleError

PROFILE = "test/ref_slow"
_built = {}


def build(bucket, layers=2, name=PROFILE):
    """The batch of `bucket` on profile `name`, built once per test run. The text regions of the serve plan are the sizes measured
    on `ref`: a bucket whose text on this profile does not fit them is built on a plan measured for it (`text_sizing.measured_plan`)."""
    key = (name, tuple(bucket), layers)
    if key not in _built:
        p = tprofile.load(name)
        try:
            _built[key] = E.emit_step_batch(*bucket, p, layers=layers)
        except ScheduleError as e:
            assert "no free word for text" in str(e) and "text_sizing.measured_plan" in str(e), e
            _built[key] = E.emit_step_batch(*bucket, p, layers=layers, plan=TX.measured_plan(name, layers, bucket))
    return _built[key]


@pytest.mark.parametrize("ctx", sorted(TS.CONTEXTS))
def test_the_two_layer_step_is_reference_exact_on_the_iss_with_other_timing(ctx):
    out = TS.check_step(build((1, 1, 64)), ctx)
    assert out["diags"] == [] and out["lints"] == []


@pytest.mark.parametrize("case", SBK.SAMPLE, ids=SBK.sample_id)
def test_the_two_layer_sample_is_reference_exact_on_the_iss_with_other_timing(case):
    bucket, lanes, n = case
    SBK.check_lanes(build(bucket), lanes, n)


def test_a_two_lane_bucket_is_reference_exact_on_tsp_sim_with_other_timing():
    assert os.path.exists(TS.SIM), f"tsp_sim is not built at {TS.SIM}: `just showcase-reference` builds it"
    SBK.check_lanes(build((2, 2, 128)), [SBK.SHARED_A, SBK.SHARED_B], 2, exe=TS.SIM)


def test_the_slower_profile_makes_a_longer_program():
    ref = E.emit_step_batch(1, 1, 64, tprofile.load("ref"), layers=2)
    assert build((1, 1, 64)).kit.last > ref.kit.last


@pytest.mark.slow
@pytest.mark.parametrize("bucket", SBK.GRID, ids=lambda b: "P%d_%d_%d" % b)
def test_every_bucket_builds_and_pages_at_two_layers_with_other_timing(bucket):
    assert build(bucket).kit.last > 0


# -- derived variants: other latencies and station gaps than any shipped profile, written to a temporary file ---------------------
VARIANTS = {
    "longer_vxm_add": dict(timing=dict(vxm=dict(d_func=7, d_skew=1))),                  # a chained add every 7 cycles
    "slow_mem_and_sxm": dict(timing=dict(mem=dict(d_func=6, d_skew=0), sxm=dict(d_func=6, d_skew=0))),
    "vxm_gap_22": dict(layout=dict(vxm_gap=22)),
    "shifted_gaps": dict(layout=dict(vxm_gap=20, mem_up_gap_base=17)),
}


def variant(tmp_path, timing=None, layout=None):
    """A profile loaded as `test/ref_slow`, with `timing` rows and `layout` gaps replaced and its own config file."""
    cfg = copy.deepcopy(json.load(open(tprofile.Profile(PROFILE).path)))
    cfg["timing"].update(timing or {})
    cfg["layout"].update(layout or {})
    p = tprofile.Profile(PROFILE)
    p.cfg, p.layout, p.path = cfg, cfg["layout"], str(tmp_path / "variant.json")
    json.dump(cfg, open(p.path, "w"))
    return p


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_a_variant_of_other_latencies_or_gaps_is_reference_exact_on_the_iss(name, tmp_path):
    """No latency, gap or offset is wired into the schedule: every variant builds and the chip equals bit-exact reference."""
    batch = E.emit_step_batch(1, 1, 64, variant(tmp_path, **VARIANTS[name]), layers=2)
    out = TS.check_step(batch, 20)
    assert out["diags"] == [] and out["lints"] == []


def test_a_chained_add_faster_than_four_cycles_is_refused_with_its_numbers(tmp_path):
    p = variant(tmp_path, timing=dict(vxm=dict(d_func=4, d_skew=2)))
    with pytest.raises(ScheduleError, match=r"chained add follows its predecessor after 3 cycles.*d_func 4, d_skew 2.*at least 4"):
        E.emit_step_batch(1, 1, 64, p, layers=2, measure=True)
