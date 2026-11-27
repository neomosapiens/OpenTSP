"""The step plan of the SHIP-style GPT-2 pipeline (design `` sections 4 and 6): scheduled requests and
request state to lanes and the per-lane input vectors of the pipeline. Pure data and numpy: no vLLM and no chip.

One lane is one token through the pipeline (one run of the chips of `ship_pipeline.Pipeline`, 128-lane tiles). The inputs that vary from lane to lane are
the SHIP form, never a patch of the program text (SHIP section 5.1, section 7.2, appendix C):

- `ids`, `positions`: the token and its position (the host embeds `wte[id] + wpe[position]`: the embedding is not on the chips, see the report's open question);
- `slots`: the pool slot of the new K and V rows (page * 16 + position mod 16), from which the host writes the Scatter maps (`ship_layer.map_weights`);
- `counts`: the mask encoding of the lane, one count per page of the pool (visible slots of the page: 16 for a full page, a partial count for the page of the
  new token, 0 for a page the request does not hold), from which the chips generate the additive mask (`ship_attn.enc_weights`).

vLLM's block of 16 positions is the page. The default pool has 64 pages (1,024 slots, 8 chunks of 8 pages), and the short pool has 16 pages; vLLM's block 0 is its null block, so block `b` is page `b - 1`.
Every stage has the same page layout, so one block table drives all chips (SHIP section 5.1, "symmetric memory allocation"). The mask encoding of a lane
lists the request's own pages only, which is what keeps requests apart; shared (prefix cached) pages are in the encodings of every request that holds them.
"""

import numpy as np

from opentsp_vllm.contracts import Lane, StepPlan, StepPlanError

KV_CHUNK_CHOICES = (2, 8)
FULL_KV_CHUNKS = 8

PAGE = 16  # slots per page = vLLM's block size
PAGES = 64  # pages of the pool: 8 chunks of 128 slots (`ship_system.CHUNKS`, `ship_attn.CHUNK`)
POOL_SLOTS = PAGE * PAGES
PAGES_PER_CHUNK = 8  # a chunk is 128 slots (`ship_attn.CHUNK`)
NUM_BLOCKS = PAGES + 1  # vLLM's block ids 0 .. 64; block 0 is the null block


def kv_limits(chunks):
    """Return usable slots and vLLM blocks including the null block."""
    if type(chunks) is not int or chunks not in KV_CHUNK_CHOICES:
        raise ValueError(f"KV chunks must be one of {KV_CHUNK_CHOICES}, not {chunks!r}")
    pages = chunks * PAGES_PER_CHUNK
    return PAGE * pages, pages + 1


def page_of(block_id):
    return block_id - 1


def slot_of(block_table, position):
    """The pool slot of `position` of a request whose block table is `block_table`."""
    return page_of(block_table[position // PAGE]) * PAGE + position % PAGE


def encoding(block_table, position, pages=PAGES):
    """The page counts (PAGES) of a lane at `position`: the visible slots of each page the request holds, positions 0 .. `position` (the lane's own new
    row included, as `ship_system.counts_for` with live = position + 1)."""
    counts = np.zeros(pages, np.int32)
    for j in range(position // PAGE + 1):
        counts[page_of(block_table[j])] = min(PAGE, position + 1 - j * PAGE)
    return counts


def _check_request(r, vocab, pool_slots, num_blocks):
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
        if not 0 <= b < num_blocks:
            raise StepPlanError(f"request {r.req_id}: block id {b} outside the pool of {num_blocks} blocks")
    last = c + n - 1
    if last >= pool_slots:
        raise StepPlanError(f"request {r.req_id}: position {last} is beyond the pool of {pool_slots} slots")
    if last // PAGE >= len(blocks):
        raise StepPlanError(f"request {r.req_id}: block table of {len(blocks)} blocks does not cover position {last}")
    for j, b in enumerate(blocks[: last // PAGE + 1]):
        if b == 0:
            raise StepPlanError(f"request {r.req_id}: block {j} of the table (positions {j * PAGE}..{j * PAGE + PAGE - 1}) is the null block")
    if len(set(blocks[: last // PAGE + 1])) != last // PAGE + 1:
        raise StepPlanError(f"request {r.req_id}: a block appears twice in the table")
    return toks, blocks, c, n


def plan_step(requests, *, vocab, kv_chunks=FULL_KV_CHUNKS):
    """Lanes and input vectors of one step. `requests` are in InputBatch order; lane i of request r is the token at position c + i, and the lanes run in this
    order (a chunk's positions ascend, so every lane finds its predecessors' K and V rows in the pool). The last lane of a request needs logits iff
    the chunk reaches the request's last known token (as `step_plan.plan_step`). The bucket is (lanes, logit rows, pool slots)."""
    if not requests:
        raise StepPlanError("no request is scheduled")
    pool_slots, num_blocks = kv_limits(kv_chunks)
    lanes = []
    for r in requests:
        toks, blocks, c, n = _check_request(r, vocab, pool_slots, num_blocks)
        for pos in range(c, c + n):
            lanes.append(Lane(r.req_id, int(toks[pos]), pos, tuple(blocks[: pos // PAGE + 1]), pos == len(toks) - 1))
    logit = [ln for ln in lanes if ln.logits]
    plan = StepPlan((len(lanes), len(logit), pool_slots), tuple(lanes), len(lanes), len(logit), tuple(ln.req_id for ln in logit))
    plan.patches = inputs(plan)
    return plan


def inputs(plan):
    """The input vectors of the real lanes of `plan`, a dict of int32 arrays (so that `StepPlan.patch_count` counts them): ids, positions, slots, counts."""
    lanes = plan.real_lanes
    pages = plan.bucket[2] // PAGE
    return dict(
        ids=np.array([ln.token for ln in lanes], np.int32),
        positions=np.array([ln.position for ln in lanes], np.int32),
        slots=np.array([slot_of(ln.block_table, ln.position) for ln in lanes], np.int32),
        counts=np.array([encoding(ln.block_table, ln.position, pages) for ln in lanes], np.int32),
    )
