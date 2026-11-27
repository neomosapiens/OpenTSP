"""Opt-in local demo observations of vLLM scheduling and SHIP execution."""
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time

_TRACE_LIMIT = 16 * 1024 * 1024
_TAG = re.compile(r"cmpl-opentsp-(demo|wave)-([a-f0-9]{32})-(\d+)(?:-.*)?\Z")
_LOG = logging.getLogger(__name__)


def request_label(request_id):
    match = _TAG.fullmatch(request_id)
    if not match:
        return hashlib.sha256(request_id.encode()).hexdigest()[:12]
    kind, tag, index = match.groups()
    return f"{tag}/{index}" if kind == "wave" else tag


def configured_trace():
    path = os.environ.get("OPENTSP_DEMO_TRACE")
    return Path(path) if path else None


def emit(path, record):
    """Append a whole record from one engine process. I/O cannot abort serving."""
    try:
        with path.open("ab", buffering=0) as stream:
            if os.fstat(stream.fileno()).st_size < _TRACE_LIMIT:
                stream.write(json.dumps({**record, "observed_ns": time.monotonic_ns()}, separators=(",", ":")).encode() + b"\n")
    except OSError as error:
        _LOG.warning("Cannot record the demo event: %s", error)


class Trace:
    """Snapshot admitted rows and the allocator's actual block table and refcounts."""

    def __init__(self, path):
        self.path = path
        self.step = 0
        self.ordinals = {}

    def schedule(self, scheduler, output):
        if not output.num_scheduled_tokens:
            return
        self.step += 1
        output.opentsp_demo_step = self.step
        prefixes = {req.req_id: req.num_computed_tokens for req in output.scheduled_new_reqs}
        self.snapshot(scheduler, "scheduled", output.num_scheduled_tokens, prefixes)

    def output(self, scheduled, outputs):
        step = getattr(scheduled, "opentsp_demo_step", None)
        tokens = []
        for client in outputs.values():
            for row in client.outputs:
                request_id = row.request_id
                if not _TAG.fullmatch(request_id):
                    continue
                if step is not None and row.new_token_ids:
                    ordinal = self.ordinals.get(request_id, 0) + 1
                    tokens.append(dict(id=request_label(request_id), ordinal=ordinal, ids=list(row.new_token_ids)))
                    self.ordinals[request_id] = ordinal + len(row.new_token_ids) - 1
                if row.finish_reason is not None:
                    self.ordinals.pop(request_id, None)
            for request_id in getattr(client, "finished_requests", None) or ():
                self.ordinals.pop(request_id, None)
        if tokens:
            emit(self.path, dict(phase="tokens", step=step, tokens=tokens))

    def settled(self, scheduler):
        self.snapshot(scheduler, "settled", {}, {})

    def snapshot(self, scheduler, phase, counts, prefixes):
        cache = scheduler.kv_cache_manager
        requests = []
        for request_id, request in scheduler.requests.items():
            blocks = cache.get_block_ids(request_id)
            amount = counts.get(request_id, 0)
            if not amount and not any(blocks):
                continue
            filled = max(0, request.num_computed_tokens - amount)
            prompt = request.num_prompt_tokens
            prefill = min(amount, max(0, prompt - filled))
            row = dict(id=request_label(request_id), prompt=prompt, filled=filled, prefill=prefill,
                       decode=amount - prefill, blocks=blocks[0] if blocks else [])
            if request_id in prefixes:
                row["prefix"] = prefixes[request_id]
            requests.append(row)
        pages = [[b.block_id, b.ref_cnt, b.block_hash is not None] for b in cache.block_pool.blocks
                 if b.block_id and (b.ref_cnt or b.block_hash is not None)]
        emit(self.path, dict(phase=phase, step=self.step, requests=requests, pages=pages,
                             capacity=len(cache.block_pool.blocks) - 1))
