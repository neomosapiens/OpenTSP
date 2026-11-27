#!/usr/bin/env python3
"""Run paged and batched TinyStories inference without vLLM.

`Scheduler` is a toy of vLLM's V1 scheduler in the aspects the chip sees: a block pool with the null block 0 (and the sink block
`num_blocks`, the chip's, never handed out), block size 16, reference counts and FIFO reuse of freed blocks; prefix caching
(a full block is hashed by its tokens chained to its predecessor's hash; a hit takes the cached block, no copy); chunked prefill
under a token budget; continuous batching (requests join and finish between steps, a step takes up to 8 lanes and up to 4
requests); preemption by recompute (the youngest running request loses its blocks when the pool is empty, re-enters the queue
at the front and re-prefills, hitting the cache where its full blocks survive); greedy sampling with a stop on the end token or
the length. Like vLLM, a block is entered into the cache as soon as the step being scheduled will fill it, so a request
admitted in the same pass can already hit it: the chip writes every lane's K and V rows of a layer before any lane reads the shared pool.

    scheduler = Scheduler(SchedulerConfig(num_blocks=6, max_model_len=64))
    scheduler.submit("a", prompt_ids, max_new)         # at any time between steps
    step = scheduler.step                            # None when nothing is scheduled; else a `Step`
    ...run the step on a chip...
    finished = scheduler.finish_step({req_id: next_token})      # one entry per logit lane (`step.logit_req_ids`)

`Step.lanes` are the real lanes of the step in the order of `opentsp_vllm.step_plan.plan_step` (the lanes that need a next token
first, then the others, each request's lanes by position); `Step.requests` are its `PlanRequest` fields (req_id, token ids,
block ids, positions computed, tokens scheduled), so `PlanRequest(*r) for r in step.requests` gives `plan_step` the same step;
`Step.bucket` is `paged_reference.choose_bucket`'s. `random_schedule(seed, steps)` is a seeded multi-request schedule (staggered joins,
finishes, prefix hits, at least one preemption, a block copy and a zeroing between runs) recorded without any chip: the
schedule source of .

`Driver` runs a scheduler on a chip, `kv_mode` (the second argument) one of
  `exact`      `paged_reference.step_batch` over a host `BlockPool` (the oracle, no ISS);
  `iss`     the bucket programs of the program cache (`build_programs`, built on demand) on `tsp-iss` through the harness input
            (`tsp-iss <config>`, as `build_programs.compat_run`): per step, the patched images and text blocks, the model
            tensors, and the K/V words of the pool; `tsp-iss run` cannot keep SRAM across runs, so the host carries the pool: it
            dumps the words the step wrote (per real lane and layer 2 K words and 4 V words) and preloads every word written so far
            into the next run (warm restart, , is 's);
  `staged`  the gate driver's one-lane step (`driver.StepProgram`, the TS layout, 8 layers, positions below 64) per token with the K/V history
            written by the host from a host block store ( B); logits from the host head.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace

import numpy as np

from opentsp_vllm.models.tinystories import paths as _paths
from opentsp_vllm.models.tinystories import paged_reference as G
from opentsp_vllm.models.tinystories.config import SMALL

BLOCK = G.BLOCK
EOS = 50256


class SchedulerError(ValueError):
    pass


class DriverError(RuntimeError):
    pass


# -- the block pool with prefix caching -------------------------------------------------------------------------------------
@dataclass
class SchedulerConfig:
    num_blocks: int = G.POOL_BLOCKS          # blocks 1..num_blocks - 1 are handed out; the chip's sink is block `num_blocks`
    token_budget: int = G.GRID_T[-1]         # tokens per step (max_num_batched_tokens)
    max_seqs: int = G.GRID_L[-1]             # running requests (max_num_seqs): at most the largest L, one logit row each
    max_model_len: int = 128

    def check(self):
        if not 2 <= self.num_blocks <= G.POOL_BLOCKS:
            raise SchedulerError(f"num_blocks {self.num_blocks} outside 2..{G.POOL_BLOCKS}")
        if not 1 <= self.token_budget <= G.GRID_T[-1]:
            raise SchedulerError(f"token_budget {self.token_budget} outside 1..{G.GRID_T[-1]}")
        if not 1 <= self.max_seqs <= G.GRID_L[-1]:
            raise SchedulerError(f"max_seqs {self.max_seqs} outside 1..{G.GRID_L[-1]}")
        if (self.num_blocks - 1) * BLOCK < self.max_model_len:
            need = -(-self.max_model_len // BLOCK) + 1
            raise SchedulerError(f"num_blocks {self.num_blocks}: at least {need} blocks are needed to admit a request of "
                                 f"max_model_len {self.max_model_len} (vLLM's admission check)")
        return self


def block_hash(parent, tokens):
    """The hash of a full block: its tokens chained to the hash of the block before it (vLLM's `hash_block_tokens`)."""
    h = hashlib.sha256()
    h.update(parent or b"\0")
    h.update(np.asarray(tokens, dtype=np.int32).tobytes())
    return h.digest()


class BlockManager:
    """`paged_reference.BlockPool`'s free queue and reference counts plus the prefix cache: a freed block keeps its hash until it is
    handed out again (the evictable cache), a hit on such a block takes it back from the queue."""

    def __init__(self, config):
        self.config = config
        self.pool = G.BlockPool(SMALL, config.num_blocks)            # bookkeeping only: the K/V words live in the chip's pool
        self.cache = {}                                              # hash -> block id
        self.hash_of = {}                                            # block id -> hash
        self.replicas = set()                                        # cached blocks that are copies (`Scheduler.register_copy`)

    @property
    def refs(self):
        return self.pool.refs

    @property
    def num_free(self):
        return len(self.pool.free_queue)

    def allocate(self):
        b = self.pool.allocate()
        h = self.hash_of.pop(b, None)
        if h is not None:
            del self.cache[h]
        self.replicas.discard(b)
        return b

    def touch(self, b):
        """A new reader of the cached block `b`, live or on the free queue."""
        if b in self.pool.refs:
            self.pool.share(b)
        else:
            self.pool.free_queue.remove(b)
            self.pool.refs[b] = 1
        return b

    def free(self, blocks):
        """Release a request's blocks, the last first (vLLM's reverse order: the tail of a sequence is evicted first)."""
        for b in reversed(blocks):
            self.pool.free(b)

    def longest_hit(self, tokens, max_tokens):
        """(blocks, hashes) of the longest chain of cached full blocks at the start of `tokens`, at most `max_tokens` tokens."""
        blocks, hashes, parent = [], [], None
        for j in range(max_tokens // BLOCK):
            h = block_hash(parent, tokens[j * BLOCK:(j + 1) * BLOCK])
            b = self.cache.get(h)
            if b is None:
                break
            blocks.append(b)
            hashes.append(h)
            parent = h
        return blocks, hashes

    def cache_full_blocks(self, req, upto):
        """Enter the blocks of `req` that hold `upto` known tokens in full into the cache (the first block of a content wins)."""
        for j in range(len(req.hashes), upto // BLOCK):
            h = block_hash(req.hashes[-1] if req.hashes else None, req.tokens[j * BLOCK:(j + 1) * BLOCK])
            req.hashes.append(h)
            if h not in self.cache:
                self.cache[h] = req.blocks[j]
                self.hash_of[req.blocks[j]] = h

    def check(self, holders=()):
        """The accounting: every block is free or referenced, never both; the references are the holders' blocks; the cache
        maps are inverse; block 0 and the sink are never handed out."""
        free, refs = list(self.pool.free_queue), self.pool.refs
        if len(set(free)) != len(free) or set(free) & set(refs) or set(free) | set(refs) != set(range(1, self.config.num_blocks)):
            raise AssertionError(f"free {sorted(free)} and live {sorted(refs)} blocks are not a partition of the pool")
        if any(n < 1 for n in refs.values()):
            raise AssertionError(f"reference counts {refs}")
        if holders:
            held = collections.Counter(b for blocks in holders for b in blocks)
            if dict(held) != refs:
                raise AssertionError(f"references {refs} differ from the holders' blocks {dict(held)}")
        if {b: h for h, b in self.cache.items()} != self.hash_of:
            raise AssertionError("the prefix cache maps are not inverse")


# -- requests and steps --------------------------------------------------------------------------------------------------------
@dataclass
class Request:
    req_id: str
    prompt: list
    max_new: int
    eos: object = None
    tokens: list = field(default_factory=list)       # prompt, then the sampled tokens
    computed: int = 0                                # positions whose K and V are in the pool
    blocks: list = field(default_factory=list)
    hashes: list = field(default_factory=list)       # hashes of the full blocks entered into the cache
    status: str = "waiting"
    preemptions: int = 0
    cached_tokens: int = 0                           # tokens of the prefix hit at the last admission

    @property
    def output(self):
        return self.tokens[len(self.prompt):]


@dataclass(frozen=True)
class Lane:
    req_id: str
    token: int
    position: int
    block_table: tuple
    logits: bool


@dataclass
class Step:
    index: int
    lanes: tuple
    bucket: tuple
    n_logits: int
    requests: tuple                                  # (req_id, token_ids, block_ids, num_computed, num_scheduled) per request
    preempted: tuple = ()                            # requests preempted while this step was scheduled
    hits: tuple = ()                                 # (req_id, tokens) of the requests admitted with a prefix hit
    block_copies: tuple = ()                         # (src, dst) pairs the host applies before the run 
    zero_blocks: tuple = ()

    @property
    def tokens(self):
        return [ln.token for ln in self.lanes]

    @property
    def positions(self):
        return [ln.position for ln in self.lanes]

    @property
    def tables(self):
        return [list(ln.block_table) for ln in self.lanes]

    @property
    def logit_req_ids(self):
        return tuple(ln.req_id for ln in self.lanes if ln.logits)


class Scheduler:
    def __init__(self, config=None):
        self.config = (config or SchedulerConfig()).check()
        self.blocks = BlockManager(self.config)
        self.requests = {}
        self.waiting = collections.deque()
        self.running = []
        self.steps = 0
        self.preemptions = 0
        self.replica_hits = 0                                       # prefix hits that took a copied block
        self._pending = None

    def has_work(self):
        return bool(self.waiting or self.running)

    def submit(self, req_id, prompt, max_new, eos=None):
        if req_id in self.requests:
            raise SchedulerError(f"request {req_id!r} exists")
        if not prompt or max_new < 1:
            raise SchedulerError(f"request {req_id!r}: an empty prompt or max_new < 1")
        if len(prompt) + max_new > self.config.max_model_len:
            raise SchedulerError(f"request {req_id!r}: {len(prompt)} prompt tokens and {max_new} new tokens exceed "
                                 f"max_model_len {self.config.max_model_len}")
        req = Request(req_id, [int(t) for t in prompt], max_new, eos, [int(t) for t in prompt])
        self.requests[req_id] = req
        self.waiting.append(req)
        return req

    # -- one scheduling pass (vLLM's `Scheduler.schedule`) --------------------------------------------------------------------
    def step(self):
        if self._pending is not None:
            raise SchedulerError("finish_step has not been called for the previous step")
        cfg, m = self.config, self.blocks
        budget = cfg.token_budget
        scheduled, preempted, hits = [], [], []
        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            n = min(len(req.tokens) - req.computed, budget)
            while True:
                need = -(-(req.computed + n) // BLOCK) - len(req.blocks)
                if need <= m.num_free:
                    break
                victim = self.running.pop()                                   # the youngest
                self._preempt(victim)
                preempted.append(victim.req_id)
                if victim is req:
                    n = 0
                    break
            if n == 0:
                break
            req.blocks += [m.allocate() for _ in range(need)]
            m.cache_full_blocks(req, min(req.computed + n, len(req.tokens)))
            scheduled.append((req, n))
            budget -= n
            i += 1
        if not preempted:
            while self.waiting and budget > 0 and len(self.running) < cfg.max_seqs:
                req = self.waiting[0]
                blocks, hashes = m.longest_hit(req.tokens, len(req.tokens) - 1)
                hit = len(blocks) * BLOCK
                n = min(len(req.tokens) - hit, budget)
                need = -(-(hit + n) // BLOCK) - len(blocks)
                if need > m.num_free - sum(b not in m.refs for b in blocks):
                    break
                self.waiting.popleft()
                req.blocks = [m.touch(b) for b in blocks]
                self.replica_hits += sum(b in m.replicas for b in blocks)
                req.hashes, req.computed, req.cached_tokens = list(hashes), hit, hit
                req.blocks += [m.allocate() for _ in range(need)]
                m.cache_full_blocks(req, min(hit + n, len(req.tokens)))
                req.status = "running"
                self.running.append(req)
                scheduled.append((req, n))
                budget -= n
                if hit:
                    hits.append((req.req_id, hit))
        if not scheduled:
            if self.waiting and not self.running:
                raise SchedulerError("the pool cannot hold the next request (num_blocks too small for it)")
            return None
        logit, other, reqs = [], [], []
        for req, n in scheduled:
            reqs.append((req.req_id, tuple(req.tokens), tuple(req.blocks), req.computed, n))
            for pos in range(req.computed, req.computed + n):
                lane = Lane(req.req_id, req.tokens[pos], pos, tuple(req.blocks[:pos // BLOCK + 1]), pos == len(req.tokens) - 1)
                (logit if lane.logits else other).append(lane)
        lanes = tuple(logit + other)
        bucket = G.choose_bucket(len(lanes), len(logit), max(ln.position for ln in lanes))
        step = Step(self.steps, lanes, bucket, len(logit), tuple(reqs), tuple(preempted), tuple(hits))
        self.steps += 1
        self._pending = (step, scheduled)
        return step

    def _preempt(self, req):
        self.blocks.free(req.blocks)
        req.blocks, req.hashes, req.computed, req.status = [], [], 0, "waiting"
        req.preemptions += 1
        self.preemptions += 1
        self.waiting.appendleft(req)

    def finish_step(self, sampled):
        """Advance the requests of the step with the sampled tokens ({req_id: id}, one per logit lane); returns the ids of the
        requests that finished (end token or length), whose blocks are released."""
        if self._pending is None:
            raise SchedulerError("no step is pending")
        step, scheduled = self._pending
        if set(sampled) != set(step.logit_req_ids):
            raise SchedulerError(f"sampled tokens for {sorted(sampled)}, the logit lanes are {sorted(step.logit_req_ids)}")
        self._pending = None
        finished = []
        for req, n in scheduled:
            req.computed += n
            if req.req_id in sampled:
                tok = int(sampled[req.req_id])
                req.tokens.append(tok)
                if tok == req.eos or len(req.output) >= req.max_new:
                    req.status = "finished"
                    self.blocks.free(req.blocks)
                    req.blocks = []
                    self.running.remove(req)
                    finished.append(req.req_id)
        return finished

    def register_copy(self, src, dst):
        """The host copied block `src` into the free block `dst` : when `src` is a cached full block, `dst` becomes the
        block the cache maps its hash to (a replica: a later hit takes `dst`, and reads the copied words)."""
        m = self.blocks
        if dst in m.refs or dst in m.hash_of:
            raise SchedulerError(f"block {dst} is live or cached: it cannot take a copy")
        h = m.hash_of.pop(src, None)
        if h is not None:
            m.cache[h] = dst
            m.hash_of[dst] = h
            m.replicas.add(dst)

    def check(self):
        self.blocks.check([r.blocks for r in self.requests.values()])


def sample(step, logits):
    """{req_id: greedy token} of the logit lanes of `step` from their rows `logits` (n_logits, vocab)."""
    return {rid: G.pick_token(logits[i]) for i, rid in enumerate(step.logit_req_ids)}


# -- the random schedule of  ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Submission:
    step: int                       # the number of steps run before the request is visible to the scheduler
    req_id: str
    prompt: tuple
    max_new: int
    eos: object = None


@dataclass
class Schedule:
    seed: int
    config: SchedulerConfig
    submissions: list
    steps: list
    preemptions: int
    prefix_hit_tokens: int
    finished_before_end: int

    def hook(self):
        """The `Driver` hook that applies this schedule's block copies and zeroings (the lanes need no hook)."""
        def hook(sched, step, n):
            rec = self.steps[n]
            for src, dst in rec.block_copies:
                sched.register_copy(src, dst)
            return replace(step, block_copies=rec.block_copies, zero_blocks=rec.zero_blocks)
        return hook


VOCAB = 50257


def random_schedule(seed, steps=12, vocab=VOCAB, max_tries=300, max_model_len=64):
    """A seeded multi-request schedule of exactly `steps` steps (4 to 200) without any chip. Requests are drawn until the schedule
    is that long (a first four that own the shared block, fill a small pool, and join late on the shared prefix, then more
    with staggered joins). Coverage of the first `steps` steps, retried over derived seeds until it holds: a join after step 0
    (steps >= 3), a prefix hit (>= 4), a finish (>= 8), a block copy and a zeroing (>= 6), a preemption (>= 12), and, from 12 steps
    on, a copy whose destination a later lane reads (see below). The pool has 6 or 7 blocks, so C = 64 buckets only.
    `max_model_len=128` (steps >= 40) draws a 56 to 62 token request that decodes past position 63 in a pool of 9 or 10 blocks
    (C = 128 buckets, a preemption in 8 or 9 usable blocks).

    The hooks are those of , applied between two runs (`Step.block_copies`, `Step.zero_blocks`): the zeroed block is a free
    block outside the prefix cache that no lane reads or writes; the copy goes into such a block too, from a live block, and when
    the source is a cached full block the copy becomes its cache replica (`Scheduler.register_copy`: the cache then maps the
    block's hash to the copy), so a later request that hits the prefix reads the copied words and the logits depend on the copy
    being right. The sampled tokens are the schedule's own (seeded); the shape of a schedule does not depend on the values a
    chip would sample, so `Driver(w, s.config, mode, hook=s.hook).run(s.submissions, max_steps=steps)` on any chip runs the
    same lanes (`s.steps`). The same (seed, steps, max_model_len) gives the same schedule; `Schedule.steps` are `Step`s as
    `Scheduler.step` returns them."""
    if not 4 <= steps <= 200:
        raise ValueError(f"steps {steps}: random_schedule makes schedules of 4 to 200 steps")
    if max_model_len not in (64, 128) or (max_model_len == 128 and steps < 40):
        raise ValueError(f"max_model_len {max_model_len}: 64, or 128 with at least 40 steps")
    long = max_model_len == 128
    for attempt in range(max_tries):
        rng = random.Random((seed * 7919 + attempt) * 1009 + steps)
        cfg = SchedulerConfig(num_blocks=rng.choice((9, 10) if long else (6, 6, 7)), token_budget=rng.choice((4, 6, 8)),
                              max_seqs=rng.choice((3, 4)), max_model_len=max_model_len)
        prefix = [rng.randrange(vocab) for _ in range(BLOCK)]
        tok = lambda n: [rng.randrange(vocab) for _ in range(n)]
        # r0 owns the shared block and keeps decoding, r1 and r3 fill the pool with blocks of their own, r2 joins late on the
        # shared prefix (a hit) and is short; the pool is too small for all of them at once
        if long:
            shapes = [(prefix + tok(rng.randint(40, 46)), rng.randint(20, 40), 0), (tok(rng.randint(10, 15)), rng.randint(20, 32), rng.randint(0, 1)),
                      (prefix + tok(rng.randint(1, 6)), rng.randint(2, 4), rng.randint(3, 6)),
                      (tok(rng.randint(10, 15)), rng.randint(20, 32), rng.randint(0, 3))]
        else:
            shapes = [(prefix + tok(rng.randint(2, 6)), rng.randint(6, 14), 0), (tok(rng.randint(10, 15)), rng.randint(6, 14), rng.randint(0, 1)),
                      (prefix + tok(rng.randint(1, 6)), rng.randint(2, 4), rng.randint(2, max(2, min(6, steps - 2)))),
                      (tok(rng.randint(10, 15)), rng.randint(6, 14), rng.randint(0, 3))]
        subs = [Submission(at, f"s{seed}r{i}", tuple(p), n) for i, (p, n, at) in enumerate(shapes[:rng.randint(3, 4)])]
        at, extra = max(s.step for s in subs), 0
        hook_at = rng.randint(2, max(2, steps // 2))
        while True:
            state = {"done": False}
            drv = Driver(None, cfg, "none", rng=random.Random(rng.random()), vocab=vocab,
                         hook=lambda sched, step, n, st=state, hr=random.Random(rng.random()): _hooks(sched, step, hr, st, hook_at))
            drv.run(subs, max_steps=steps)
            if len(drv.steps) >= steps:
                break
            for _ in range(2 + extra // 2):                                   # the work ran out: more requests
                at += rng.randint(1, 5)
                p = prefix + tok(rng.randint(1, 8)) if rng.random() < 0.5 else tok(rng.randint(6, 20))
                subs.append(Submission(at, f"s{seed}r{len(subs)}", tuple(p), rng.randint(2, 14)))
            extra += 1
        sched = drv.scheduler
        finished = sum(r.status == "finished" for r in sched.requests.values())
        st = drv.steps
        ok = ((steps < 3 or any(x.step > 0 for x in subs)) and (steps < 4 or any(x.hits for x in st)) and (steps < 8 or finished >= 1)
              and (steps < 6 or any(x.block_copies for x in st) and any(x.zero_blocks for x in st))
              and (steps < 12 or sched.preemptions >= 1 and sched.replica_hits >= 1) and (not long or any(x.bucket[2] == 128 for x in st)))
        if ok:
            return Schedule(seed, cfg, [x for x in subs if x.req_id in sched.requests], list(st), sched.preemptions,
                            sum(h for x in st for _, h in x.hits), finished)
    raise SchedulerError(f"no schedule of {steps} steps with the required coverage found for seed {seed} (max_model_len {max_model_len})")


def _hooks(sched, step, rng, state, hook_at):
    """The first step from `hook_at` on that has two free uncached blocks no lane uses gets one copy (a live cached block's replica
    when there is one) and one zeroing; later steps get none."""
    m = sched.blocks
    if state["done"] or step.index < hook_at:
        return step
    used = {b for ln in step.lanes for b in ln.block_table}
    free = [b for b in m.pool.free_queue if b not in m.hash_of and b not in used]
    live = sorted(m.refs)
    if len(free) < 2 or not live:
        return step
    cached = [b for b in live if b in m.hash_of]
    dst, zero = rng.sample(free, 2)
    src = rng.choice(cached or live)
    sched.register_copy(src, dst)
    state["done"] = True
    return replace(step, block_copies=((src, dst),), zero_blocks=(zero,))


# -- the chips -------------------------------------------------------------------------------------------------------------------
class NullChip:
    """No arithmetic: one-hot logit rows of seeded random tokens (the schedule generator's chip)."""

    name = "none"

    def __init__(self, rng, vocab=VOCAB):
        self.rng, self.vocab, self.last = rng, vocab, {}

    def run(self, step):
        rows = np.zeros((step.n_logits, self.vocab), np.float32)
        for r in rows:
            r[self.rng.randrange(self.vocab)] = 1
        return rows

    def copy_block(self, src, dst):
        pass

    def zero_block(self, b):
        pass


class ReferenceChip:
    """`kv_mode=exact`: the oracle over a host `BlockPool`."""

    name = "exact"

    def __init__(self, w, num_blocks=G.POOL_BLOCKS):
        self.w = w
        self.pool = G.BlockPool(w.cfg, num_blocks)
        self.last = {}

    def run(self, step):
        r = G.step_batch(self.w, self.pool, step.tokens, step.positions, step.tables, step.bucket, step.n_logits)
        return r.logits[:step.n_logits]

    def copy_block(self, src, dst):
        self.pool.copy(src, dst)

    def zero_block(self, b):
        self.pool.zero(b)


class IssChip:
    """`kv_mode=iss`: the cached bucket programs on `tsp-iss`. The pool is carried by the host as `shadow`, the words every step
    wrote ({(slice, word): 320 bytes}; the chip's SRAM is zero elsewhere), dumped after the step (`D` lines) and preloaded in
    full into the next run (`S` lines). `verify` runs `paged_reference.step_batch` beside the chip on a second pool and requires the
    logits and every K and V word the step wrote to equal the oracle's bit for bit."""

    name = "iss"

    def __init__(self, model, cache=None, verify=True):
        from opentsp_vllm.models.tinystories import build_programs as BP
        from opentsp_vllm.models.tinystories import program_cache as PC
        from opentsp_vllm.models.tinystories import kvpool
        from opentsp_tools import profile as tprofile
        from opentsp_vllm import step_plan as SP
        self.BP, self.PC, self.SP, self.kvpool = BP, PC, SP, kvpool
        self.model, self.cache, self.verify = model, cache, verify
        self.iss = BP.tsp_iss()
        self.geo = kvpool.Geometry(tprofile.load("ref"), model.layers)
        self.vlanes = self.geo.p.vlanes
        self.config = os.path.join(_paths.SPEC, "config", f"{model.profile}.json")
        self.tsp_as = BP.tsp_as
        self.tensors = self.tsp_as.read_weights(model.package, model.profile, self.config)
        self.arts, self.static, self.names = {}, {}, {}
        self.shadow = {}
        self.oracle = G.BlockPool(model.cfg, G.POOL_BLOCKS) if verify else None
        self.last = {}
        self.build_seconds = 0.0

    # -- the programs ------------------------------------------------------------------------------------------------------------
    def art(self, bucket):
        if bucket not in self.arts:
            t0 = time.perf_counter()
            self.BP.build_bucket(self.model, bucket, self.cache)
            self.arts[bucket] = self.BP.Artifacts(self.PC.bucket_dir(self.cache, self.model, bucket), self.model.profile)
            self.build_seconds += time.perf_counter() - t0
            art = self.arts[bucket]
            self.names[bucket] = [str(n) for n in art.paged.table.names]
            fixed = [x for x in art.externs if x["name"] in self.tensors]
            self.static[bucket] = self.tsp_as.preload_text(self.tsp_as.bind_weights(fixed, self.tensors))
        return self.arts[bucket]

    def plan(self, step):
        SP = self.SP
        real = [SP.Lane(ln.req_id, ln.token, ln.position, ln.block_table, ln.logits) for ln in step.lanes]
        plan = SP.StepPlan(step.bucket, tuple(real) + (SP.PAD,) * (step.bucket[0] - len(real)), len(real), step.n_logits,
                           step.logit_req_ids)
        plan.patches = SP.patch_values(plan, self.model.cfg.vocab)
        return plan

    # -- the words of a K/V row --------------------------------------------------------------------------------------------------
    def row_words(self, layer, block, offset):
        """The (slice, word) keys of row `offset` of `block` in `layer`: the two K byte planes, the four V byte planes."""
        g, kv = self.geo, self.kvpool
        return ([(kv.K_BASE + 2 * (offset % 8) + p, g.kaddr(layer, block) + offset // 8) for p in range(2)] +
                [(s, g.vaddr(layer, block) + offset) for s in g.vslices(layer)])

    def written(self, step):
        return sorted({k for ln in step.lanes for l in range(self.model.layers)
                       for k in self.row_words(l, ln.block_table[-1], ln.position % G.BLOCK)})

    # -- one step ----------------------------------------------------------------------------------------------------------------
    def run(self, step):
        t_start = time.perf_counter()
        art = self.art(step.bucket)
        values = self.PC.plan_site_values(art.record["geometry"], self.names[step.bucket], self.plan(step))
        images, tensors = art.patched(values)
        dyn = [x for x in art.externs if x["name"] in tensors]
        ta = self.tsp_as
        text = self.static[step.bucket] + ta.preload_text(ta.bind_weights(dyn, tensors)) + ta.input_text(images)
        pool_in = len(self.shadow)
        text += ta.preload_text([(s, a, w) for (s, a), w in self.shadow.items()])
        writes = self.written(step)
        dumps = writes + [(s, a + k) for s, a, n in art.regions for k in range(n)]
        text += "".join(f"D {s:x} {a:x}\n" for s, a in dumps)
        t0 = time.perf_counter()
        r = subprocess.run([self.iss, self.config, "--lint", "--max-cycles", str(2 * art.record["schedule_cycles"])],
                           input=text.encode(), capture_output=True, timeout=1500)
        t_iss = time.perf_counter() - t0
        out = r.stdout.decode()
        if r.returncode:
            raise DriverError(f"step {step.index}: tsp-iss exited {r.returncode}: {r.stderr.decode()[:300]}")
        diags = [ln for ln in out.splitlines() if (ln.startswith("E ") and not ln.startswith("E - ")) or ln.startswith("T lint")]
        if diags:
            raise DriverError(f"step {step.index} in bucket P{step.bucket}: diagnostics {diags[:3]}")
        words = {}
        for ln in out.splitlines():
            if ln.startswith("W "):
                f = ln.split()
                words[(int(f[1], 16), int(f[2], 16))] = bytes(int(x, 16) for x in f[3:])
        rows = b"".join(words[(s, a + k)] for s, a, n in art.regions for k in range(n))
        logits = self.PC.decode_logits(rows, step.bucket[1], self.model.cfg.vocab, art.index)[:step.n_logits]
        for key in writes:
            self.shadow[key] = words[key]
        if self.verify:
            self._verify(step, logits)
        # the harness prints no cycle count (its `E` line carries the cycle of the first diagnostic, -1 when there is none): the
        # step's cycles are the program's schedule length, the same whatever way the cache entry was built
        self.last = {"cycles": art.record["schedule_cycles"], "iss_seconds": t_iss,
                     "step_seconds": time.perf_counter() - t_start, "pool_words_in": pool_in, "pool_words_out": len(writes),
                     "input_bytes": len(text)}
        return logits

    def _verify(self, step, logits):
        want = G.step_batch(self.model.w, self.oracle, step.tokens, step.positions, step.tables, step.bucket, step.n_logits)
        for r in range(step.n_logits):
            if logits[r].tobytes() != want.logits[r].tobytes():
                bad = np.nonzero(logits[r] != want.logits[r])[0]
                raise DriverError(f"step {step.index} P{step.bucket}: logit row {r} differs from bit-exact reference in {len(bad)} entries, "
                                  f"first {bad[:3].tolist()}")
        expect = {}
        for ln in step.lanes:
            for l in range(self.model.layers):
                b = ln.block_table[-1]
                if (l, b) not in expect:
                    expect[(l, b)] = {**self.kvpool.k_words(self.geo, self.oracle, l, [b]),
                                      **self.kvpool.v_words(self.geo, self.oracle, l, [b])}
                for key in self.row_words(l, b, ln.position % G.BLOCK):
                    if key not in self.shadow:
                        raise DriverError(f"step {step.index} P{step.bucket}: the K/V word {key} of layer {l}, block {b}, position "
                                          f"{ln.position} was not carried into the pool (not read back from the chip)")
                    if self.shadow[key][:len(expect[(l, b)][key])] != expect[(l, b)][key]:
                        raise DriverError(f"step {step.index} P{step.bucket}: the K/V word {key} of layer {l}, block {b}, "
                                          f"position {ln.position} differs from bit-exact reference")

    # -- the hooks of , between runs ---------------------------------------------------------------------------------
    def copy_block(self, src, dst):
        for l in range(self.model.layers):
            for o in range(G.BLOCK):
                for ks, kd in zip(self.row_words(l, src, o), self.row_words(l, dst, o)):
                    if ks in self.shadow:
                        self.shadow[kd] = self.shadow[ks]
                    else:
                        self.shadow.pop(kd, None)
        if self.oracle is not None:
            self.oracle.copy(src, dst)

    def zero_block(self, b):
        for l in range(self.model.layers):
            for o in range(G.BLOCK):
                for k in self.row_words(l, b, o):
                    self.shadow.pop(k, None)
        if self.oracle is not None:
            self.oracle.zero(b)


class StagedChip:
    """`kv_mode=staged`: one lane per run of the gate driver's step program (the TS layout: 64 slots, the current token in the last one); the
    host keeps the K/V rows of every block in a store and writes each lane's history into the program's SRAM tensors."""

    name = "staged"

    def __init__(self, w, work=None, interface="native"):
        from opentsp_vllm.models.tinystories import driver as TS4
        from opentsp_vllm.models.tinystories import reference as exact_ref
        from opentsp_vllm.models.tinystories.config import TINYSTORIES
        if w.cfg != TINYSTORIES:
            raise DriverError("the staged step is the 8-layer TS-3 program: kv_mode=staged needs the 8-layer model")
        self.TS4, self.exact_ref, self.w, self.cfg = TS4, exact_ref, w, TINYSTORIES
        self.work = work or tempfile.mkdtemp(prefix="v1h-staged-")
        self.prog = TS4.StepProgram(w, self.work, interface, say=lambda *a: None)
        self.store = {}
        self.last = {}

    def _block(self, b):
        if b not in self.store:
            c = self.cfg
            self.store[b] = (np.zeros((c.layers, G.BLOCK, c.hidden), np.float16), np.zeros((c.layers, G.BLOCK, c.hidden), np.float32))
        return self.store[b]

    def run(self, step):
        TS4, exact_ref, w, c = self.TS4, self.exact_ref, self.w, self.cfg
        logits, cycles, t0 = {}, 0, time.perf_counter()
        for i in sorted(range(len(step.lanes)), key=lambda i: step.lanes[i].position):     # earlier positions are written first
            ln = step.lanes[i]
            if ln.position > c.slots - 1:
                raise DriverError(f"position {ln.position}: the staged step holds {c.slots} slots")
            kv = exact_ref.HostKV(c)
            if ln.position:
                missing = [b for b in ln.block_table[:(ln.position - 1) // G.BLOCK + 1] if b not in self.store]
                if missing:
                    raise DriverError(f"staged: blocks {missing} of the table were never written")
                kv.k = np.concatenate([self.store[b][0] for b in ln.block_table], axis=1)[:, :ln.position]
                kv.v = np.concatenate([self.store[b][1] for b in ln.block_table], axis=1)[:, :ln.position]
            k_slots, v_slots, mask = kv.slots(ln.position)
            words, cyc, _ = self.prog.step(exact_ref.embed(w, ln.token, ln.position), k_slots, v_slots, mask)
            cycles += cyc or 0
            blk = self._block(ln.block_table[-1])
            for l in range(c.layers):
                blk[0][l, ln.position % G.BLOCK] = TS4.plane_array(words[f"k{l}"], "f16")
                blk[1][l, ln.position % G.BLOCK] = TS4.plane_array(words[f"v{l}"], "f32")
            if ln.logits:
                logits[i] = exact_ref.head_logits(w, TS4.plane_array(words["hidden"], "f32"))
        self.last = {"cycles": cycles, "step_seconds": time.perf_counter() - t0}
        return np.stack([logits[i] for i in range(step.n_logits)]) if step.n_logits else np.zeros((0, c.vocab), np.float32)

    def copy_block(self, src, dst):
        k, v = self._block(src)
        self.store[dst] = (k.copy(), v.copy())

    def zero_block(self, b):
        self.store.pop(b, None)


# -- the driver ---------------------------------------------------------------------------------------------------------------------
@dataclass
class StepRecord:
    step: Step
    logits: np.ndarray                  # (n_logits, vocab) float32, the rows of the logit lanes in lane order
    seconds: float
    cycles: object = None
    info: dict = field(default_factory=dict)


class Driver:
    """A `Scheduler` on a chip. `kv_mode`: "exact", "iss" (needs `model`, a `program_cache.Model`), "staged" or "none" (the schedule
    generator's, seeded by `rng`). `hook(scheduler, step, n)` may return the step with block copies and zeroings to apply."""

    def __init__(self, w, config=None, kv_mode="exact", *, model=None, cache=None, verify=True, work=None, eos=None, rng=None,
                 vocab=VOCAB, hook=None):
        self.config = (config or SchedulerConfig()).check()
        self.scheduler = Scheduler(self.config)
        self.kv_mode, self.eos, self.hook = kv_mode, eos, hook
        if kv_mode == "exact":
            self.chip = ReferenceChip(w, self.config.num_blocks)
        elif kv_mode == "iss":
            if model is None:
                raise DriverError("kv_mode=iss needs the program_cache.Model of the cached programs (model=...)")
            self.chip = IssChip(model, cache, verify)
        elif kv_mode == "staged":
            self.chip = StagedChip(w, work)
        elif kv_mode == "none":
            self.chip = NullChip(rng or random.Random(0), vocab)
        else:
            raise DriverError(f"kv_mode {kv_mode!r}: exact, iss, staged or none")
        self.steps, self.records = [], []

    def run(self, submissions, max_steps=None):
        """Run until every request has finished (or `max_steps` steps ran); a request is visible once `step` steps have run (an
        idle scheduler takes the next one at once). Returns {req_id: the new tokens}."""
        pending = sorted(submissions, key=lambda s: s.step)
        sched = self.scheduler
        while True:
            if max_steps is not None and len(self.steps) >= max_steps:
                break
            if not sched.has_work() and pending and pending[0].step > len(self.steps):
                pending[0] = replace(pending[0], step=len(self.steps))
            for s in [s for s in pending if s.step <= len(self.steps)]:
                sched.submit(s.req_id, s.prompt, s.max_new, self.eos if s.eos is None else s.eos)
                pending.remove(s)
            if not sched.has_work() and not pending:
                break
            step = sched.step()
            if step is None:
                raise DriverError("the scheduler has work but schedules nothing")
            if self.hook is not None:
                step = self.hook(sched, step, len(self.steps))
                sched._pending = (step, sched._pending[1])
            for src, dst in step.block_copies:
                self.chip.copy_block(src, dst)
            for b in step.zero_blocks:
                self.chip.zero_block(b)
            t0 = time.perf_counter()
            logits = self.chip.run(step)
            self.steps.append(step)
            self.records.append(StepRecord(step, np.array(logits, copy=True), time.perf_counter() - t0, self.chip.last.get("cycles"),
                                           dict(self.chip.last)))
            sched.finish_step(sample(step, logits))
        sched.check()
        return {rid: list(r.output) for rid, r in sched.requests.items()}


# -- the scenarios of the design (6.6) and the command line --------------------------------------------------------------------------
PREFIX16 = [7454, 2402, 257, 640, 612, 373, 257, 1310, 2576, 3706, 11, 543, 373, 257, 922, 2576]     # one full block
SCENARIOS = {
    # four requests sharing the block of PREFIX16, staggered joins, on the 2-layer test model
    "shared4": [Submission(0, "a", tuple(PREFIX16 + [11, 12, 13]), 4), Submission(1, "b", tuple(PREFIX16 + [21, 22, 23, 24]), 3),
                Submission(4, "c", tuple(PREFIX16 + [31, 32]), 4), Submission(5, "d", tuple(PREFIX16 + [41, 42, 43, 44, 45]), 3)],
    # the full-model scenario: two requests sharing one full block, the second joining after the first has computed it
    "full": [Submission(0, "a", tuple(PREFIX16 + [198, 198]), 4), Submission(2, "b", tuple(PREFIX16 + [464, 3290, 373]), 4)],
}


def run_scenario(name, layers, checkpoint=None, cache=None, mode="iss", say=print, work=None):
    """Run a scenario on `mode`; every request's tokens are compared with `paged_reference.generate_paged` of the request alone. Returns
    the summary dict (`ok`, per-step rows, tokens, seconds)."""
    from opentsp_vllm.models.tinystories import program_cache as PC
    if checkpoint:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model = PC.Model(layers, 1, checkpoint)
    w = model.w
    drv = Driver(w, SchedulerConfig(), mode, model=model, cache=cache, work=work)
    subs = SCENARIOS[name]
    t0 = time.perf_counter()
    out = drv.run(subs)
    wall = time.perf_counter() - t0
    build = getattr(drv.chip, "build_seconds", 0.0)
    say(f"{'step':>4} {'bucket':>10} {'lanes':>5} {'logits':>6} {'cycles':>8} {'run s':>7} {'step s':>7} {'pool in':>7} {'out':>4}  requests")
    rows = []
    for rec in drv.records:
        st = rec.step
        row = dict(step=st.index, bucket=list(st.bucket), lanes=len(st.lanes), logits=st.n_logits, cycles=rec.cycles,
                   iss_s=rec.info.get("iss_seconds"), step_s=rec.seconds, pool_in=rec.info.get("pool_words_in"),
                   pool_out=rec.info.get("pool_words_out"), input_bytes=rec.info.get("input_bytes"), preempted=list(st.preempted), hits=[list(h) for h in st.hits])
        rows.append(row)
        say(f"{st.index:>4} {'P%d_%d_%d' % st.bucket:>10} {row['lanes']:>5} {row['logits']:>6} "
            f"{rec.cycles if rec.cycles is not None else '-':>8} {row['iss_s'] if row['iss_s'] is not None else 0:>7.2f} "
            f"{rec.seconds:>7.2f} {row['pool_in'] or 0:>7} {row['pool_out'] or 0:>4}  {','.join(sorted({ln.req_id for ln in st.lanes}))}"
            + (f"  hit {st.hits}" if st.hits else "") + (f"  preempted {list(st.preempted)}" if st.preempted else ""))
    ok = True
    for s in subs:
        ref = G.generate_paged(w, list(s.prompt), s.max_new, C=64 if len(s.prompt) + s.max_new <= 64 else 128)
        same = out[s.req_id] == ref.tokens[len(s.prompt):]
        ok &= same
        say(f"{'PASS' if same else 'FAIL'} request {s.req_id}: {len(s.prompt)} prompt tokens, new tokens {out[s.req_id]} "
            f"{'equal to' if same else 'DIFFER from'} generate_paged alone")
    cycles = [r["cycles"] for r in rows if r["cycles"]]
    summary = dict(scenario=name, mode=mode, layers=layers, weights=model.source, ok=bool(ok), steps=len(rows), wall_s=wall,
                   build_s=build, cycles_total=sum(cycles) if cycles else None, lanes_total=sum(r["lanes"] for r in rows),
                   tokens=out, rows=rows, digest=model.key)
    say(f"{name} on {mode} ({layers} layers, weights {model.source}, programs {model.key}): {len(rows)} steps, "
        f"{summary['lanes_total']} lanes, "
        + (f"{sum(cycles):,} simulated cycles, " if cycles else "") + f"wall {wall:.1f} s (of which program builds {build:.1f} s)")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", choices=sorted(SCENARIOS))
    ap.add_argument("--layers", type=int, choices=(2, 8), default=None, help="default: 2 for shared4, 8 for full")
    ap.add_argument("--mode", choices=("exact", "iss", "staged"), default="iss")
    ap.add_argument("--checkpoint", nargs="?", const="calum/tinystories-gpt2-3M", help="the real checkpoint (8 layers; the cached copy)")
    ap.add_argument("--cache", default=None, help="the program cache (default $OPENTSP_VLLM_CACHE or ~/.cache/opentsp/vllm)")
    ap.add_argument("--json", help="write the summary here")
    a = ap.parse_args(argv)
    layers = a.layers or (8 if a.scenario == "full" else 2)
    try:
        s = run_scenario(a.scenario, layers, a.checkpoint, a.cache, a.mode)
    except (DriverError, SchedulerError) as e:
        print(f"FAIL driver_paged: {e}")
        return 1
    if a.json:
        with open(a.json, "w") as f:
            json.dump(s, f, indent=1, default=str)
    return 0 if s["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
