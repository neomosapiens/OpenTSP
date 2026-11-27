"""GPT-2 small (124M parameters) as the SHIP-style pipeline runs it ( section 2):
12 layers of width 768 and 12 heads of 64, a 3 072-wide tanh-GELU FFN, 1 024 positions, the tied 50 257-token table."""
from dataclasses import dataclass

from opentsp_tools.programs import ref_wide as rw


@dataclass(frozen=True)
class Gpt2Config:
    hidden: int = 768
    heads: int = 12
    ffn: int = 3072
    layers: int = 12
    vocab: int = 50257
    max_pos: int = 1024
    pool: int = 1024                 # KV slots of a stage (64 pages of 16)
    ln_eps: float = 1e-5
    # layer ranges of the pipeline stages, the design's balanced split at pool 1 024 (linear form, 3 chips: [2, 7, 3])
    stage_layers: tuple = (2, 7, 3)

    @property
    def dk(self):
        return self.hidden // self.heads

    @property
    def tiles(self):
        return rw.tiles_of(self.hidden)

    @property
    def ffn_tiles(self):
        return rw.tiles_of(self.ffn)

    @property
    def pages(self):
        return self.pool // rw.PAGE

    def check(self):
        assert self.hidden % self.heads == 0 and self.dk == rw.DK, "heads of 64 lanes"
        assert self.hidden % rw.TILE == 0, "the hidden vector is whole 256-lane vectors"
        assert self.pool % rw.CHUNK == 0, "the pool is whole score chunks"
        assert sum(self.stage_layers) == self.layers
        return self

    def stages(self):
        """[(first layer, one past the last)] of each stage."""
        out, a = [], 0
        for n in self.stage_layers:
            out.append((a, a + n))
            a += n
        return out


GPT2_SMALL = Gpt2Config().check()
