"""What a generator needs of a profile, checked by capability instead of by name.

Two kinds of generator. The geometry-generic ones (the batched serving step, the wide kernels) derive their sizes from the profile
(`opentsp_tools.programs.geometry`) and state what they need as lower bounds, `geometry.Needs`: `SERVE` below. A profile that offers more than the bound
fits (a bigger chip holds the same program); one that offers less is refused with the first unmet requirement and both numbers.

The others (the one-token decode step, the fixed TinyStories programs) are written for the geometry of `ref`: `ProfileNeeds` lists
superlanes, MXM rows, MEM slices per hemisphere and words per slice, and `check(profile)` raises `ProfileUnfit` naming the first
requirement the profile does not equal, with both numbers and the words "written for exactly": such a program fits a profile of the
geometry of `ref` (`test/ref_slow`) and no other, a larger one included.

Timing (latencies, gaps, occupancies) is never a need: every generator derives it from the profile (`Profile.timing`, `stations`)."""
from dataclasses import dataclass

from opentsp_tools.programs.geometry import Need, Needs, ProfileUnfit


def facts(p):
    """(superlanes, MXM rows, MEM slices per hemisphere, words per slice) of a profile object of either kind
    (`opentsp_tools.profile.Profile` or `layout.Profile`)."""
    if hasattr(p, "superlanes"):
        superlanes = p.superlanes
    else:
        superlanes, rem = divmod(p.vlanes, 16)
        assert rem == 0, f"{p.name}: {p.vlanes} lanes are not a whole number of superlanes"
    slices = p.slices_per_hemisphere if hasattr(p, "slices_per_hemisphere") else p.mem_per_hemisphere
    words = p.words if hasattr(p, "words") else p.words_per_slice
    return superlanes, p.mxm_rows, slices, words


@dataclass(frozen=True)
class ProfileNeeds:
    what: str                   # the program or kernel set, for the message
    superlanes: int
    mxm_rows: int
    slices_per_hemisphere: int
    words: int                  # words per slice

    def requirements(self, p):
        """[(name, needed, has, unit, why)] of every requirement, in the order they are checked."""
        sl, rows, per, words = facts(p)
        return [
            ("superlanes", self.superlanes, sl, "superlanes", f"{16 * self.superlanes} lanes: the tile, chunk and lane constants"),
            ("mxm.rows", self.mxm_rows, rows, "MXM rows", "the weight tiles and the install schedule"),
            ("slices per hemisphere", self.slices_per_hemisphere, per, "MEM slices per hemisphere",
             "the queue numbers of the quads and the shifter pairs"),
            ("words per slice", self.words, words, "words per slice", "the fixed word addresses and the bank bit"),
        ]

    def unmet(self, p):
        """The first unmet requirement as (name, needed, has, unit, why), or None."""
        for req in self.requirements(p):
            if req[2] != req[1]:
                return req
        return None

    def check(self, p):
        """Raise `ProfileUnfit` for the first requirement `p` does not meet; returns `p`."""
        req = self.unmet(p)
        if req is not None:
            name, need, has, unit, why = req
            # a smaller profile does not fit the programs; a larger one is not the geometry they are written for
            verb = "does not fit" if has < need else "is not the geometry for"
            raise ProfileUnfit(f"profile {p.name} {verb} {self.what}: written for exactly {need} {unit}, profile has {has} "
                               f"({why}); first unmet requirement: {name}")
        return p


# the serving programs of the TinyStories showcase and the kernels they use, as written for the geometry of `ref`
SHOWCASE = ProfileNeeds("the serving programs", superlanes=20, mxm_rows=320, slices_per_hemisphere=44, words=8192)


# The batched serving step (`emit_step_batch`, `layout.plan_serve`): lower bounds. Sizes that depend on the profile are derived
# (`layout.Shape`); what is needed is the width of the packed softmax (two heads of 64 slots in a vector), one plane's worth of
# f16 rows for a 64-slot K install, and the MEM floor plan of the layout (eleven quads per hemisphere). The model's own needs
# (its FFN half and hidden width against the MXM rows) are `model_needs`; the words per slice the plan needs are named by the planner
# when a region does not fit a bank.
SERVE = Needs("the serving programs", [
    (Need("superlanes", 8, "superlanes", "128 lanes: the packed softmax holds two heads of 64 slots, a context of 64 slots fills half"),
     lambda g: g.superlanes),
    (Need("mxm.rows", 128, "MXM rows", "a 64-slot K install per plane (f16 rows are two stream values)"), lambda g: g.mxm_rows),
    (Need("slices per hemisphere", 44, "MEM slices per hemisphere", "the floor plan names quads 0 to 10 of each hemisphere"),
     lambda g: g.slices_per_hemisphere),
])


def model_needs(cfg):
    """The needs of a model on top of `SERVE`: the f16 rows of one MXM plane must hold its largest installed matrix (the FFN half,
    the hidden width) and the heads' 64-slot K block."""
    rows = max(cfg.half, cfg.hidden, 64)
    return Needs(f"the model of {cfg.layers} layers (hidden {cfg.hidden}, FFN {cfg.ffn})", [
        (Need("mxm.rows", 2 * rows, "MXM rows", f"a plane holds {rows} f16 rows: the largest matrix of the model"), lambda g: g.mxm_rows),
        (Need("superlanes", cfg.hidden // 16, "superlanes", f"{cfg.hidden} hidden features in one vector"), lambda g: g.superlanes)])
