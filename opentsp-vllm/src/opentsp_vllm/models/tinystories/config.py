"""TinyStories model dimensions and OpenTSP mapping constraints."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """GPT-2 decoder: pre-LayerNorm, causal attention with biases, tanh-GELU FFN, learned positions, tied head.

    `slots` is the K/V cache length of the static decode program: slots 0 .. pos-1 hold the cached positions
    and the last slot (`slots - 1`) is the current token. `half` is the FFN half width (the FFN
    is split over the two hemispheres' MXM planes). The mapping needs H, slots, half to be multiples of 16
    (a superlane) and H/16 and slots/16 powers of two (the SXM butterfly)."""
    hidden: int = 64
    heads: int = 16
    ffn: int = 256
    layers: int = 8
    vocab: int = 50257
    max_pos: int = 1024
    slots: int = 64
    ln_eps: float = 1e-5

    @property
    def dk(self):
        return self.hidden // self.heads

    @property
    def half(self):
        return self.ffn // 2

    def check(self):
        assert self.hidden % self.heads == 0
        for n in (self.hidden, self.slots, self.half):
            assert n % 16 == 0, n
        for n in (self.hidden // 16, self.slots // 16):
            assert n & (n - 1) == 0, n
        assert self.ffn % 2 == 0 and self.hidden % 8 == 0 and self.half % 8 == 0
        return self


TINYSTORIES = ModelConfig().check()          # calum/tinystories-gpt2-3M
TINYSTORIES_L2 = ModelConfig(layers=2).check()   # the 2-layer serving test model (vLLM )
# the serving test model of a profile whose MXM holds 64 f16 rows per plane (`test/sl8`): the FFN halves and the K block fit one plane,
# the vocabulary is small so that the packed head needs several installs of 64 rows on both planes and the embedding one part
SERVE_SMALL = ModelConfig(hidden=64, heads=16, ffn=128, layers=2, vocab=1000, max_pos=128, slots=64).check()
# a small configuration with the same structure, for tests that run every element through the oracle
SMALL = ModelConfig(hidden=32, heads=8, ffn=64, layers=2, vocab=97, max_pos=64, slots=32).check()

PROMPT = "Once upon a time there was"
PROMPT_IDS = (7454, 2402, 257, 640, 612, 373)   # GPT-2 BPE ids of PROMPT
