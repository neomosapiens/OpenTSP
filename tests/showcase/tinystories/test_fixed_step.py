from opentsp_vllm.models.tinystories import fixed_step as fixed


def test_static_stages_have_no_instruction_sites():
    from opentsp_tools import profile
    from opentsp_vllm.models.tinystories import weights
    from opentsp_vllm.models.tinystories import layout

    model = fixed.Model(profile.load("ref"), weights.synthetic(layout.serve_config(2), 1))
    for name in ("copy9", "pre0", "score0_15", "normalize", "single15", "context0_15", "post0", "head"):
        kit = model.emit(name)
        assert not kit.view2.sites
