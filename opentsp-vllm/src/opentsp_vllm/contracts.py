"""Dependency-leaf serving contracts shared by planners, loaders and device adapters."""
import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, runtime_checkable

class TspChipError(RuntimeError):
    """A run raised a diagnostic, or the chip faulted or did not finish : a defect, never a recoverable condition.

    The binding fills in `cycle`, `queue` and `code` (the first diagnostic) and `diagnostics` (all of them, as
    `(cycle, queue, code)`); they stay `None` and empty for a failure that is not a diagnostic.
    """

    cycle = None
    queue = None
    code = None
    diagnostics = ()
    reproducer = None  # the directory `IssChip` wrote for the failure , when it could


@dataclass(frozen=True)
class ChipInfo:
    """What the worker needs of the loaded model and programs: the KV specs, the pool size and the context buckets."""

    layers: int
    vocab: int
    hidden: int
    heads: int
    num_blocks: int  # blocks of the pool including the null block 0; the sink block comes after
    grid_c: tuple  # the context slots of the programs the chip holds
    kv_heads: int = 16  # the KV shape vLLM is told (one KV cache layer per model layer): heads and head size
    head_size: int = 4


@dataclass
class ChipResult:
    logits: object  # (n_logits, vocab) float32, rows in logit-lane order; the runner copies it before sampling (read-only is fine)
    cycles: object = None  # simulated cycles of the run (None where there are none)
    diagnostics: tuple = ()


class StepPlanError(TspChipError, ValueError):
    """The scheduled step cannot be mapped onto a bucket, or a patch value is out of range. Raised before the chip is touched."""


@dataclass(frozen=True)
class PlanRequest:
    """One scheduled request: every token it knows (prompt and sampled), its block ids, the positions already in the KV cache and
    how many tokens this step computes."""

    req_id: str
    token_ids: object
    block_ids: object
    num_computed: int
    num_scheduled: int


@dataclass(frozen=True)
class Lane:
    """`req_id` None is a padding lane."""

    req_id: object
    token: object
    position: int
    block_table: tuple
    logits: bool


PAD = Lane(None, None, 0, (), False)


@dataclass
class StepPlan:
    bucket: tuple
    lanes: tuple  # T lanes: the logit lanes first, then the other real lanes, then padding lanes
    n_real: int
    n_logits: int
    logit_req_ids: tuple  # request of logit lane i, in InputBatch order
    patches: dict = field(default_factory=dict)
    demo_step: int | None = None

    @property
    def real_lanes(self):
        return self.lanes[: self.n_real]

    @property
    def padding_lanes(self):
        return self.bucket[0] - self.n_real

    @property
    def context_padding(self):
        last = max((lane.position for lane in self.real_lanes), default=-1)
        return self.bucket[2] - 1 - last

    @property
    def patch_count(self):
        return sum(int(v.size) for v in self.patches.values())


@runtime_checkable
class Chip(Protocol):
    @property
    def info(self) -> ChipInfo: ...

    @property
    def failure(self) -> "str | None":
        """None while the chip is healthy; else what failed."""

    def load_model(self, source: str) -> None:
        """Load the weights: `synthetic:SEED` or a directory of the exported package."""

    def warm_up(self) -> None: ...

    def zero_pool(self) -> None: ...

    def zero_blocks(self, block_ids) -> None:
        """`new_block_ids_to_zero`: the K and V words of each block of every layer."""

    def copy_blocks(self, copies) -> None:
        """`kv_cache_block_copies`: pairs `(src, dst)`, applied in order."""

    def step(self, plan) -> ChipResult:
        """One run of the plan's bucket: writes the real lanes' K and V rows, returns the logit rows."""

    def close(self) -> None: ...


def patch_columns(table):
    """The columns of a patch table (`opentsp_tools.patch` `Table`, or the path of `bindings.npz`) as the dict of
    lists that `Chip.load_program(patch_table=...)` takes."""
    if isinstance(table, (str, os.PathLike)):
        import numpy as np

        with np.load(table) as z:
            table = {k: z[k] for k in z.files}
    get = table.get if isinstance(table, dict) else lambda k: getattr(table, k)
    cols = {k: get(k) for k in ("names", "queue", "unit", "nunits", "lo", "width", "vmin", "vmax", "placeholder", "fixed", "stride", "span")}
    out = {k: [str(x) for x in v] if k == "names" else [int(x) for x in v] for k, v in cols.items()}
    if isinstance(table, dict) and "block_units" in table:  # a paged program: where its text blocks start (serving bindings)
        out["image_queues"] = [int(x) for x in table["image_queues"]]
        out["image_units"] = [int(x) for x in table["image_units"]]
        out["block_units"] = int(table["block_units"])
    return out


def block_records(blocks):
    """The text blocks of a paged program (the block list from `program.json`, or a path to a JSON list) as `Chip.load_program(blocks=...)` takes
    them: `[(queue, index, [(tensor, slice, address, words)])]`."""
    if isinstance(blocks, (str, os.PathLike)):
        import json

        with open(blocks) as f:
            blocks = json.load(f)
    return [(int(b["queue"]), int(b["index"]), [(str(n), int(s), int(a), int(w)) for n, s, a, w in b["tensors"]]) for b in blocks]


@dataclass
class BucketProgram:
    """One bucket program as the program cache  hands it over.

    `elf` and `text_package` are bytes or paths; `patch_table` the columns of the patch table (`patch_columns`); `outputs` the
    regions `(slice, addr, words)` a step returns; `max_cycles` the cycle limit (twice the recorded cycles); `cycles`
    the cycle count a run reports (`Platform::cycle`: one more than the ISS's recorded `last_cycle`; checked by `warm_up`). `blocks` are
    the text blocks of a paged program, `site_names` the site names that the int32 bytes of `site_values` follow. `site_values(plan)` is the patch values of a plan, `{site: int}` or int32
    bytes in table order, and `decode(words, plan)` the `(n_logits, vocab)` float32 array of the output words.
    """

    name: str
    elf: object
    site_values: Callable
    decode: Callable
    patch_table: Optional[dict] = None
    text_package: object = None
    outputs: Optional[list] = None
    max_cycles: int = 1_000_000_000
    cycles: Optional[int] = None
    blocks: Optional[list] = None  # a paged program's text blocks (`block_records`)
    site_names: Optional[list] = None  # the names of the sites in table order, when `site_values` gives int32 bytes


class DomainFault(TspChipError):
    def __init__(self, message, reports=(), *, localized=True):
        super().__init__(message)
        self.reports = tuple(reports)
        self.target = None
        unsafe = [(chip, r) for chip, r in reports if r["fatal_count"]]
        if localized and len(unsafe) == 1 and not any(r["overflow"] for _, r in reports):
            chip, r = unsafe[0]
            if r["sram_word"] is not None:
                self.target = (chip, *r["sram_word"])
