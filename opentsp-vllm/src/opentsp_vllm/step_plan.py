"""Plan TinyStories bucket sizes and instruction patches for scheduled requests.

Pure data and numpy: no vLLM and no chip. The runner turns a `SchedulerOutput` into `PlanRequest`s; the chip consumes the
`StepPlan`. The numerical comparison uses `paged_reference`.
"""

import numpy as np

from opentsp_vllm.contracts import Lane as Lane
from opentsp_vllm.contracts import PAD as PAD
from opentsp_vllm.contracts import PlanRequest
from opentsp_vllm.contracts import StepPlan as StepPlan
from opentsp_vllm.contracts import StepPlanError as StepPlanError

BLOCK = 16  # positions per block
POOL_BLOCKS = 128  # blocks the worker reports: 0 is the null block, vLLM uses 1..127
NULL = 0  # never written: the zero source of padding block references
SINK = POOL_BLOCKS  # takes redirected writes, never read; not a block id vLLM hands out
GRID_T = (1, 2, 4, 8)  # token lanes
GRID_L = (1, 2, 4)  # logit rows
GRID_C = (64, 128)  # context slots
PACK = 5  # vocabulary rows per embedding word
PART_ROWS = 4096  # words per embedding part; word PART_ROWS of a part is its zero word
KWRITE_PAIRS = 8  # K row o lives in slice pair o mod 8

# the patch kinds of the TinyStories programs; a patch table names the ones its programs use
PATCH_KINDS = ("kblock", "vblock", "kwrite", "vwrite", "emb_part", "emb_map", "wpe", "mask")


def choose_bucket(n_tokens, n_logits, max_pos, grid_c=GRID_C):
    """The smallest bucket (T, L, C) of the grid with T >= tokens, L >= logit rows and C >= the largest position + 1."""

    def smallest(grid, n, what):
        for g in grid:
            if g >= n:
                return g
        raise ValueError(f"no bucket holds {n} {what}")

    return (
        smallest(GRID_T, n_tokens, "token lanes"),
        smallest(GRID_L, n_logits, "logit rows"),
        smallest(grid_c, max_pos + 1, "context slots"),
    )


def embedding_parts(vocab):
    return -(-(-(-vocab // PACK)) // PART_ROWS)


def _check_request(r: PlanRequest, vocab):
    toks, blocks = list(r.token_ids), list(r.block_ids)
    c, n = r.num_computed, r.num_scheduled
    if n < 1:
        raise StepPlanError(f"request {r.req_id}: {n} tokens scheduled")
    if not 0 <= c <= len(toks):
        raise StepPlanError(f"request {r.req_id}: {c} tokens computed of {len(toks)} known")
    if c + n > len(toks):
        raise StepPlanError(f"request {r.req_id}: {n} tokens scheduled after {c} computed exceed the {len(toks)} known")
    for t in toks[c : c + n]:
        if not 0 <= t < vocab:
            raise StepPlanError(f"request {r.req_id}: token {t} outside the vocabulary of {vocab}")
    for b in blocks:
        if not 0 <= b < POOL_BLOCKS:
            raise StepPlanError(f"request {r.req_id}: block id {b} outside the pool of {POOL_BLOCKS} blocks")
    last = c + n - 1
    if last // BLOCK >= len(blocks):
        raise StepPlanError(
            f"request {r.req_id}: block table of {len(blocks)} blocks does not cover position {last}"
        )
    # the lanes read every block of the table up to their position: a null block anywhere in that prefix would read zeros as context
    # (a hole, as of a sliding window) and the emitter's host check refuses it too
    for j, b in enumerate(blocks[: last // BLOCK + 1]):
        if b == NULL:
            raise StepPlanError(f"request {r.req_id}: block {j} of the table (positions {j * BLOCK}..{j * BLOCK + BLOCK - 1}) is the null block")
    return toks, blocks, c, n


def plan_step(requests, *, vocab, grid_c=GRID_C, patch_kinds=PATCH_KINDS):
    """Lanes, bucket and patch values of one step. `requests` are in InputBatch order. Lane i of request r is the token at
    position c + i; the last lane needs logits iff the chunk reaches the request's last known token ."""
    unknown = [k for k in patch_kinds if k not in PATCH_KINDS]
    if unknown:
        raise StepPlanError(f"unknown patch kind {unknown[0]!r}")
    if not requests:
        raise StepPlanError("no request is scheduled")
    logit_lanes, other_lanes = [], []
    for r in requests:
        toks, blocks, c, n = _check_request(r, vocab)
        for pos in range(c, c + n):
            lane = Lane(r.req_id, int(toks[pos]), pos, tuple(blocks[: pos // BLOCK + 1]), pos == len(toks) - 1)
            (logit_lanes if lane.logits else other_lanes).append(lane)
    n_real = len(logit_lanes) + len(other_lanes)
    if n_real > GRID_T[-1]:
        raise StepPlanError(f"{n_real} token lanes exceed the largest bucket of {GRID_T[-1]} lanes")
    if len(logit_lanes) > GRID_L[-1]:
        raise StepPlanError(f"{len(logit_lanes)} logit rows exceed the largest bucket of {GRID_L[-1]}")
    real = logit_lanes + other_lanes
    try:
        bucket = choose_bucket(n_real, len(logit_lanes), max(ln.position for ln in real), grid_c)
    except ValueError as e:
        raise StepPlanError(f"{e}: the programs cover the context slots {tuple(grid_c)}") from e
    lanes = tuple(real) + (PAD,) * (bucket[0] - n_real)
    plan = StepPlan(bucket, lanes, n_real, len(logit_lanes), tuple(ln.req_id for ln in logit_lanes))
    plan.patches = patch_values(plan, vocab, patch_kinds)
    return plan


def patch_values(plan, vocab, kinds=PATCH_KINDS):
    """The int32 value arrays of the patch kinds, one row per lane (padding lanes: the zero block, the zero word, the sink).

    - `kblock`, `vblock` (T, C/16): the block of each context block, NULL beyond the lane's table;
    - `kwrite` (T, 8, 2): per K slice pair the block (the real one on pair o mod 8 of position row o, the sink on the other
      seven) and the word half o div 8; `vwrite` (T, 2): block and row o;
    - `emb_part` (T, parts): per embedding part the word of the token's row, the zero word on the parts that do not hold it;
      `emb_map` (T): token mod 5, the segment the permuter rotates to lane 0;
    - `wpe`, `mask` (T): the position, which selects the `wpe` row and the mask.
    """
    T, _, C = plan.bucket
    parts = embedding_parts(vocab)
    out = {}
    for kind in kinds:
        if kind in ("kblock", "vblock"):
            a = np.full((T, C // BLOCK), NULL, dtype=np.int32)
            for i, ln in enumerate(plan.lanes):
                a[i, : len(ln.block_table)] = ln.block_table
        elif kind == "kwrite":
            a = np.zeros((T, KWRITE_PAIRS, 2), dtype=np.int32)
            a[:, :, 0] = SINK
            for i, ln in enumerate(plan.lanes[: plan.n_real]):
                row = ln.position % BLOCK
                a[i, row % KWRITE_PAIRS, 0] = ln.block_table[-1]
                a[i, :, 1] = row // KWRITE_PAIRS
        elif kind == "vwrite":
            a = np.zeros((T, 2), dtype=np.int32)
            a[:, 0] = SINK
            for i, ln in enumerate(plan.lanes[: plan.n_real]):
                a[i] = (ln.block_table[-1], ln.position % BLOCK)
        elif kind == "emb_part":
            a = np.full((T, parts), PART_ROWS, dtype=np.int32)
            for i, ln in enumerate(plan.lanes[: plan.n_real]):
                part, local = divmod(ln.token // PACK, PART_ROWS)
                a[i, part] = local
        elif kind == "emb_map":
            a = np.zeros(T, dtype=np.int32)
            for i, ln in enumerate(plan.lanes[: plan.n_real]):
                a[i] = ln.token % PACK
        else:  # wpe, mask
            a = np.zeros(T, dtype=np.int32)
            for i, ln in enumerate(plan.lanes[: plan.n_real]):
                a[i] = ln.position
        out[kind] = a
    return out


def padding_plan(bucket, *, vocab):
    """Build the fixed bucket's plan with no live tokens or requests."""
    plan = StepPlan(bucket, (PAD,) * bucket[0], 0, 0, ())
    plan.patches = patch_values(plan, vocab)
    return plan
