"""TinyStories embedding-table geometry shared by runtime emitters and fixture generators."""
from opentsp_tools.programs.kernels import embed as k_embed

class EmbedSizes:
    """The geometry of the embedding table: the rows of each part (a bank holds 4 096; the reduced cases cut smaller)."""

    def __init__(self, part_rows, vocab, positions=128, pack=5):
        self.part_rows, self.vocab, self.positions, self.pack = tuple(part_rows), vocab, positions, pack
        self.rows = sum(part_rows)
        assert -(-vocab // pack) == self.rows, "the parts hold exactly the packed rows of the vocabulary"

    def locate(self, token):
        """(part, word in the part, segment) of a token."""
        rho, seg = divmod(token, self.pack)
        for q, n in enumerate(self.part_rows):
            if rho < n:
                return q, rho, seg
            rho -= n
        raise ValueError(f"token {token} is outside the vocabulary")

def tables(layout, plan, sizes):
    return k_embed.Tables(
        parts=tuple(k_embed.Part(layout.EMB_PAIRS[q], plan.by_name[f"em{layout.EMB_PAIRS[q][0]}"].addr,
                                 plan.by_name[f"ez{layout.EMB_PAIRS[q][0]}"].addr, n)
                    for q, n in enumerate(sizes.part_rows)),
        wpe_slices=layout.WPE_SLICES, wpe_addrs=tuple(plan.by_name[f"wp{s}"].addr for s in layout.WPE_SLICES),
        positions=sizes.positions, map_slices=layout.MAPS_QUAD[:2], map_addr=plan.by_name["maps"].addr,
        segments=getattr(sizes, "pack", k_embed.SEGMENTS))
