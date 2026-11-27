"""Test device helpers that retain the plans and results for assertions."""

from opentsp_vllm.models.tinystories.config import ModelConfig
from opentsp_vllm.models.tinystories.reference_chip import ReferenceChip as NumericalChip, make_serving_chip as serving_chip


class ReferenceChip(NumericalChip):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.plans = []
        self.results = []

    def step(self, plan):
        result = super().step(plan)
        self.plans.append(plan)
        self.results.append(result)
        return result


def make_config():
    """The shape of the unit tests' model: the TinyStories GPT-2 with 2 layers and a 2000-word vocabulary."""
    return ModelConfig(layers=2, vocab=2000, max_pos=128)


def make_chip(vllm_config):
    """The chip factory of the worker tests (`additional_config opentsp.chip = "reference_chip:make_chip"`)."""
    context = vllm_config.additional_config.get("opentsp", {}).get("max_context", 128)
    return ReferenceChip(make_config(), grid_c=(64,) if context == 64 else (64, 128))


def make_chip_64(vllm_config):
    """A chip whose programs stop at context 64 whatever `max_context` says (the chip, not the options, limits the worker)."""
    return ReferenceChip(make_config(), grid_c=(64,))


def make_serving_chip(vllm_config):
    return serving_chip(vllm_config, chip_class=ReferenceChip)
