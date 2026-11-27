"""The serving generators test a profile by what it offers (`opentsp_tools.programs.needs`), not by its name.
The batched serving step and its planner (`needs.SERVE`) take any profile that offers at least their lower bounds and refuse the
others with the first unmet requirement and both numbers, before any scheduling (`test_serve_geometry.py` runs them on
other geometries). The fixed programs of the TinyStories showcase (the one-token step, the block, the fixed package: `needs.SHOWCASE`)
are written for the geometry of `ref`: a profile of that geometry with other timing is accepted, every other one is refused with
the words "written for exactly" and both numbers, a larger one included."""
import json
import os

import pytest

from opentsp_vllm.models.tinystories import paths as _paths

from opentsp_vllm.models.tinystories import emit_block as eb
from opentsp_vllm.models.tinystories import emit_step as es
from opentsp_vllm.models.tinystories import emit_step_batch as esb
from opentsp_vllm.models.tinystories import layout as L
from opentsp_tools.programs import needs
from opentsp_tools import profile as tprofile
from opentsp_tools.stations import ScheduleError

SPEC_CONFIG = os.path.join(_paths.SPEC, "config")
# (profile, requirement named first, needed, has): the order of the checks is superlanes, MXM rows, slices, words
UNFIT = [
    ("mini", "superlanes", 20, 1),
    ("test/fetch_alt", "superlanes", 20, 4),
    ("test/mid_min", "superlanes", 20, 4),
    ("test/ref_small", "mxm.rows", 320, 160),
    ("test/sl8", "superlanes", 20, 8),
    ("test/sl32", "superlanes", 20, 32),
]
def verb(name):
    """How the refusal of an unfit profile starts: a smaller profile does not fit, a larger one is not the geometry the programs are written for."""
    need, has = next((n, h) for p, _, n, h in UNFIT if p == name)
    return "does not fit" if has < need else "is not the geometry for"


# the batched step: the lower bounds of `needs.SERVE`; a profile above them fits
SERVE_UNFIT = [
    ("mini", "superlanes", 8, 1),
    ("test/fetch_alt", "superlanes", 8, 4),
    ("test/mid_min", "superlanes", 8, 4),
    ("test/ref_small", "slices per hemisphere", 44, 8),
]
FIT = ["ref", "test/ref_slow"]


def all_profiles():
    names = [f[:-5] for f in os.listdir(SPEC_CONFIG) if f.endswith(".json") and f != "schema.json"]
    names += ["test/" + f[:-5] for f in os.listdir(os.path.join(SPEC_CONFIG, "test")) if f.endswith(".json")]
    return sorted(names)


@pytest.mark.parametrize("name", FIT)
def test_a_profile_with_the_geometry_of_ref_fits(name):
    assert needs.SHOWCASE.check(tprofile.load(name)) is not None
    assert needs.SHOWCASE.unmet(L.load_profile(name)) is None


@pytest.mark.parametrize("name, req, need, has", UNFIT)
def test_a_profile_that_cannot_hold_the_programs_names_the_first_unmet_requirement(name, req, need, has):
    with pytest.raises(needs.ProfileUnfit) as e:
        needs.SHOWCASE.check(tprofile.load(name))
    msg = str(e.value)
    assert f"profile {name} {verb(name)}" in msg
    assert f"written for exactly {need}" in msg and f"profile has {has}" in msg and f"first unmet requirement: {req}" in msg
    assert "ref profile" not in msg


def test_the_check_reads_both_profile_kinds_alike():
    for name in all_profiles():
        a, b = needs.facts(tprofile.load(name)), needs.facts(L.load_profile(name))
        assert a == b, name
        cfg = json.load(open(os.path.join(SPEC_CONFIG, name + ".json")))
        assert a == (cfg["geometry"]["superlanes"], cfg["mxm"]["rows"], cfg["geometry"]["mem_slices_per_hemisphere"],
                     cfg["sram"]["words_per_slice"])


def test_the_verdict_follows_the_numbers_not_the_name():
    """A copy of ref under another name fits; ref with one superlane fewer does not, whatever it is called."""
    p = tprofile.load("ref")
    renamed = tprofile.Profile("ref")
    renamed.name = "my_ref_copy"
    assert needs.SHOWCASE.check(renamed) is renamed
    fewer = tprofile.Profile("ref")
    fewer.name, fewer.superlanes = "ref", p.superlanes - 1
    with pytest.raises(needs.ProfileUnfit, match="written for exactly 20 superlanes, profile has 19"):
        needs.SHOWCASE.check(fewer)


def refused(calls):
    """The distinct messages of the calls, each of which must raise `ProfileUnfit`."""
    msgs = set()
    for call in calls:
        with pytest.raises(ScheduleError) as e:
            call()
        assert isinstance(e.value, needs.ProfileUnfit)
        msgs.add(str(e.value))
    return msgs


@pytest.mark.parametrize("name", [n for n, *_ in UNFIT])
def test_every_fixed_generator_refuses_an_unfit_profile_with_the_same_message(name):
    p, lp = tprofile.load(name), L.load_profile(name)
    msgs = refused([lambda: es.build(p, 5), lambda: eb.build(p, 5), lambda: L.plan_tinystories(L.TINYSTORIES_L2, lp, 4)])
    assert len(msgs) == 1 and f"profile {name} {verb(name)}" in msgs.pop()


@pytest.mark.parametrize("name, req, need, has", SERVE_UNFIT)
def test_the_batched_step_and_its_planner_refuse_a_profile_below_their_bounds_with_the_same_message(name, req, need, has):
    p, lp = tprofile.load(name), L.load_profile(name)
    msgs = refused([lambda: esb.emit_step_batch(1, 1, 64, p, layers=2, measure=True), lambda: L.plan_serve(L.serve_config(2), lp)])
    assert len(msgs) == 1
    msg = msgs.pop()
    assert f"profile {name} does not fit the serving programs: needs {need}" in msg and f"profile has {has}" in msg
    assert f"first unmet requirement: {req}" in msg


def test_the_serving_planner_runs_on_a_profile_of_the_same_geometry():
    plan = L.plan_serve(L.serve_config(2), L.load_profile("test/ref_slow"))
    assert plan.p.name == "test/ref_slow"


@pytest.mark.parametrize("name", [n for n, *_ in UNFIT])
def test_the_driver_and_the_fixed_package_refuse_an_unfit_profile_before_building(name, tmp_path):
    from opentsp_vllm.models.tinystories import driver
    from opentsp_vllm.models.tinystories import fixed_package
    from opentsp_vllm.models.tinystories import fixed_step
    from opentsp_vllm.models.tinystories import weights as W
    with pytest.raises(needs.ProfileUnfit, match=f"profile {name} {verb(name)}"):
        driver.StepProgram(W.synthetic(L.serve_config(2), 1), str(tmp_path), profile=name)
    with pytest.raises(needs.ProfileUnfit, match=f"profile {name} {verb(name)}"):
        fixed_package.build(tmp_path / "cache", layers=2, profile_name=name)
    with pytest.raises(needs.ProfileUnfit, match=f"profile {name} {verb(name)}"):
        fixed_step.Model(tprofile.load(name), W.synthetic(L.serve_config(2), 1))
