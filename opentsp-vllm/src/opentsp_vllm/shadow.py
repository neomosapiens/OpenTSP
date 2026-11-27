"""Shadow mode (test only): a `Chip` that runs every operation on two chips and requires the logits of each step to be
bit-identical. The worker wraps its chip in it when `OPENTSP_VLLM_SHADOW` names the twin."""

import numpy as np

from opentsp_vllm.contracts import TspChipError


class ShadowChip:
    """`chip` answers; `shadow` (ReferenceChip, normally) sees the same weights, block operations and plans and is only compared against."""

    def __init__(self, chip, shadow, shadow_weights):
        self.chip = chip
        self.shadow = shadow
        self.shadow_weights = shadow_weights
        self._steps = 0
        self._failure = None

    @property
    def info(self):
        return self.chip.info

    @property
    def plan_step(self):
        # The planner is optional. Preserve its absence for chips using the runner's default.
        return self.chip.plan_step

    @property
    def failure(self):
        return self._failure or self.chip.failure or self.shadow.failure

    def load_model(self, source):
        self.chip.load_model(source)
        self.shadow.load_model(self.shadow_weights)

    def warm_up(self):
        self.chip.warm_up()
        self.shadow.warm_up()

    def zero_pool(self):
        self.chip.zero_pool()
        self.shadow.zero_pool()

    def zero_blocks(self, block_ids):
        ids = list(block_ids)
        self.chip.zero_blocks(ids)
        self.shadow.zero_blocks(ids)

    def copy_blocks(self, copies):
        pairs = list(copies)
        self.chip.copy_blocks(pairs)
        self.shadow.copy_blocks(pairs)

    def step(self, plan):
        result = self.chip.step(plan)
        twin = self.shadow.step(plan)
        a, b = np.asarray(result.logits), np.asarray(twin.logits)
        if a.shape != b.shape:
            bad = f"shape {a.shape} against {b.shape}"
        elif a.dtype != b.dtype:
            bad = f"dtype {a.dtype} against {b.dtype}"
        elif a.tobytes() != b.tobytes():
            rows = [i for i, (primary, expected) in enumerate(zip(a, b)) if primary.tobytes() != expected.tobytes()]
            bad = f"logit row(s) {rows}"
        else:
            bad = None
        if bad:
            self._failure = f"shadow mismatch at step {self._steps}, bucket {tuple(plan.bucket)}: {bad} differ from the twin's"
            raise TspChipError(self._failure)
        self._steps += 1
        return result

    def close(self):
        try:
            self.chip.close()
        finally:
            self.shadow.close()
