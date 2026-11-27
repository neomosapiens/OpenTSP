"""The batched serving step is derived from the geometry of the profile (`layout.Shape`, `opentsp_tools.programs.geometry`), not from the numbers of `ref`.

The step is emitted for profiles of other widths and run on the ISS against bit-exact reference (the host reference, computed with the profile's own
lanes and packing): every pool word, the final hidden state and the logits equal bit for bit, with no diagnostic and no lint.
 - `test/sl32` (32 superlanes, 512 lanes): the 2-layer test model of `ref`; a larger profile than the programs were written for fits.
 - `test/sl8` (8 superlanes, 128 lanes, 64 f16 rows per MXM plane): a model whose FFN halves, K block and head installs fit one plane
   (`config.SERVE_SMALL`), with the head packed two vocabulary rows per word and eight installs.
A profile that cannot hold a program or a model is refused before any scheduling with the first unmet requirement and both numbers.
`ref` keeps its sizes (`Shape.of` equals the constants the programs were written with)."""
import pytest


from opentsp_vllm.models.tinystories import config
from opentsp_vllm.models.tinystories import emit_step_batch as E
from opentsp_vllm.models.tinystories import layout as L
import test_step_batch as TS
import test_step_buckets as SBK
from opentsp_vllm.models.tinystories import text_sizing as TX
from opentsp_tools.programs import needs
from opentsp_tools import profile as tprofile
from opentsp_tools.stations import ScheduleError

_built = {}


def build(name, bucket=(1, 1, 64), cfg=None):
    """The batch of `bucket` on profile `name` for the model `cfg` (default the 2-layer test model), built once per test run.
    The pinned text regions are measured on `ref`: a bucket whose text does not fit them is built on a measured plan."""
    cfg = cfg or L.serve_config(2)
    key = (name, tuple(bucket), cfg)
    if key not in _built:
        p = tprofile.load(name)
        try:
            _built[key] = E.emit_step_batch(*bucket, p, layers=cfg.layers, cfg=cfg)
        except ScheduleError as e:
            assert "no free word for text" in str(e) and "text_sizing.measured_plan" in str(e), e
            plan = TX.measured_plan(name, cfg.layers, bucket, cfg=cfg)
            _built[key] = E.emit_step_batch(*bucket, p, layers=cfg.layers, cfg=cfg, plan=plan)
    return _built[key]


def ref_shape(cfg):
    return L.Shape.of(L.load_profile("ref"), cfg)


# -- the shape of each profile -------------------------------------------------------------------------------------------
def test_ref_keeps_the_sizes_the_programs_were_written_with():
    for layers in (2, 8):
        s = ref_shape(L.serve_config(layers))
        assert (s.lanes, s.pack, s.head_rows, s.head_installs, s.ctxs) == (320, 5, 160, 63, (64, 128))
        assert (s.nlw_head, s.logit_stride) == (20, 315)


def test_the_shape_follows_the_lanes_the_matrix_and_the_vocabulary():
    cfg = config.SERVE_SMALL
    sl8, sl32 = L.Shape.of(L.load_profile("test/sl8"), cfg), L.Shape.of(L.load_profile("test/sl32"), L.serve_config(2))
    assert (sl8.lanes, sl8.pack, sl8.head_rows, sl8.head_installs, sl8.ctxs) == (128, 2, 64, 8, (64,))
    assert (sl32.lanes, sl32.pack, sl32.head_rows, sl32.head_installs, sl32.ctxs) == (512, 5, 160, 63, (64, 128))
    assert (sl8.nlw_head, sl8.logit_stride) == (8, 16)


# -- the capability check names what is missing --------------------------------------------------------------------------
@pytest.mark.parametrize("name, req, need, has", [
    ("mini", "superlanes", 8, 1),
    ("test/mid_min", "superlanes", 8, 4),
    ("test/ref_small", "slices per hemisphere", 44, 8),
])
def test_a_profile_below_a_lower_bound_is_refused_with_both_numbers(name, req, need, has):
    with pytest.raises(needs.ProfileUnfit) as e:
        E.emit_step_batch(1, 1, 64, tprofile.load(name), layers=2)
    msg = str(e.value)
    assert f"profile {name} does not fit the serving programs: needs {need}" in msg and f"profile has {has}" in msg
    assert f"first unmet requirement: {req}" in msg


def test_a_model_that_a_plane_cannot_hold_is_refused_with_the_rows_it_needs():
    with pytest.raises(needs.ProfileUnfit, match=r"does not fit the model of 2 layers \(hidden 64, FFN 256\): needs 256 MXM rows, "
                                                  r"profile has 128 .*first unmet requirement: mxm.rows"):
        E.emit_step_batch(1, 1, 64, tprofile.load("test/sl8"), layers=2)


def test_a_context_that_the_k_install_cannot_hold_is_refused_with_its_numbers():
    with pytest.raises(needs.ProfileUnfit, match=r"does not fit the bucket P\(1, 1, 128\): needs 256 MXM rows for the 128-slot K "
                                                  r"install, profile has 128"):
        E.emit_step_batch(1, 1, 128, tprofile.load("test/sl8"), cfg=config.SERVE_SMALL)


def test_a_larger_profile_than_the_programs_need_fits():
    for name in ("ref", "test/ref_slow", "test/sl32"):
        assert needs.SERVE.check(tprofile.load(name)) is not None


def test_the_fixed_programs_say_they_are_written_for_exactly_the_geometry_of_ref():
    with pytest.raises(needs.ProfileUnfit, match=r"written for exactly 20 superlanes, profile has 32"):
        needs.SHOWCASE.check(tprofile.load("test/sl32"))


def test_a_larger_profile_is_not_said_to_not_fit(): # a bigger chip holds the model; the programs are written for one geometry
    with pytest.raises(needs.ProfileUnfit) as e:
        needs.SHOWCASE.check(tprofile.load("test/sl32"))
    assert "does not fit" not in str(e.value)
    msg = str(e.value)
    assert msg.startswith("profile test/sl32 is not the geometry for the serving programs: written for exactly 20 superlanes, profile has 32"), msg
    assert msg.count("serving programs") == 1, msg


def test_a_planner_that_runs_out_of_a_bank_says_how_many_words_it_needs():
    p = L.load_profile("test/ref_small")                       # 1024 words per slice, banks of 512
    pl = L.Planner(p)
    pl.add("big", "weight", (0,), 600, rw=False)
    with pytest.raises(ValueError, match=r"region big \(weight\) does not fit bank \d: it needs 600 words in slice\(s\) \[0\], the bank of the "
                                          r"profile has 512 words \(1024 words per slice\)"):
        pl.place()


def test_a_vocabulary_that_the_embedding_parts_cannot_hold_says_what_the_bank_must_offer():
    with pytest.raises(ValueError, match=r"the vocabulary of 50257 entries needs 10 embedding parts of at most 1024 packed rows "
                                          r"\(5 entries per row\) and the layout has 3 slice pairs for them: it needs 8192 words per "
                                          r"slice .*the profile has 2048"):
        L.emb_part_rows(50257, 1024, 5)


def test_a_profile_with_fewer_words_per_slice_than_the_model_needs_says_how_many_it_needs():
    """The planner of the serving layout on a profile of the geometry of `ref` with a quarter of the words: the first thing that
    does not fit is named with the words per slice it needs (the vocabulary of the full model, then the pools)."""
    quarter = L.Profile("quarter", 88, 2048, 10, 320, 44, 320)
    with pytest.raises(ValueError, match=r"needs 8192 words per slice .*the profile has 2048"):
        L.plan_serve(L.serve_config(2), quarter)
    small = config.SERVE_SMALL
    with pytest.raises(ValueError, match=r"does not fit bank \d: it needs \d+ words in slice\(s\) .*the bank of the profile has 512 words"):
        L.plan_serve(small, L.Profile("tiny", 88, 1024, 9, 320, 44, 320))


# -- the step on a wider profile -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("ctx", sorted(TS.CONTEXTS))
def test_the_two_layer_step_is_reference_exact_on_32_superlanes(ctx):
    out = TS.check_step(build("test/sl32"), ctx)
    assert out["diags"] == [] and out["lints"] == []


def test_two_lanes_with_a_shared_prefix_are_reference_exact_on_32_superlanes():
    SBK.check_lanes(build("test/sl32", (2, 2, 128)), [SBK.SHARED_A, SBK.SHARED_B], 2)


# -- the step on a narrower profile, with a model that fits it -----------------------------------------------------------------
@pytest.mark.parametrize("ctx", [1, 20, 64])
def test_the_small_model_step_is_reference_exact_on_8_superlanes(ctx):
    out = TS.check_step(build("test/sl8", cfg=config.SERVE_SMALL), ctx)
    assert out["diags"] == [] and out["lints"] == []


def test_several_lanes_and_logit_rows_are_reference_exact_on_8_superlanes():
    b = build("test/sl8", (2, 2, 64), config.SERVE_SMALL)
    SBK.check_lanes(b, [SBK.SHARED_A, SBK.SHARED_B], 2)


def test_the_host_twin_takes_the_lanes_of_the_unit_and_the_packing_of_the_profile():
    """An exact zero sum is -0 only when every product of the unit's lanes is -0: a packed row of -0 weights against a +0 hidden vector gives
    -0 on a unit that has exactly the lanes of the row (128 on 8 superlanes) and +0 on a wider one, where lanes above the row add +0."""
    import numpy as np
    from opentsp_vllm.models.tinystories import paged_reference as G
    wte = np.full((4, 64), -0.0, np.float16)
    w = type("W", (), {"wte": wte, "cfg": type("C", (), {"vocab": 4, "hidden": 64})()})()
    zero = np.zeros(64, np.float32)
    on_128 = G.head_logits_chip(w, zero, 128, 2)
    on_320 = G.head_logits_chip(w, zero, 320, 2)
    assert np.signbit(on_128).all() and not np.signbit(on_320).any()
    assert G.head_logits_chip(w, zero, 320, 5).shape == (4,) and np.array_equal(G.packed_table(w, 2).shape, [1, 4097, 128])


@pytest.mark.parametrize("name, pack", [("ref", 5), ("test/sl8", 2), ("test/sl32", 5)])
def test_the_embedding_loads_one_segment_map_per_vocabulary_row_of_a_word(name, pack, serving_program_cases):
    """One rotation map per segment of a packed word: five on `ref`, two on 8 superlanes (a map of a segment that no word has would be a map
    id spent and a LoadMapWide of time)."""
    text, _ = serving_program_cases.embed(tprofile.load(name))
    assert text.count("loadmapwide") == pack
