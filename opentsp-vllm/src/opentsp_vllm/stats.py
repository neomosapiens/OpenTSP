"""per-step statistics.  (the aggregates `TspWorker.get_tsp_stats` returns)."""

import time

SUMS = ("real_lanes", "padding_lanes", "context_padding", "cycles", "patch_count", "host_prep_s", "chip_s", "sampler_s")


class Stats:
    """Counters and sums over the steps. `last` is the most recent step's record. Plain Python values only: the aggregates
    cross the RPC boundary."""

    def __init__(self):
        self.steps = 0
        self.sums = dict.fromkeys(SUMS, 0)
        self.buckets = {}
        self.blocks_zeroed = 0
        self.blocks_copied = 0
        self.last = None
        self.started = time.monotonic()

    def record(self, *, bucket, real_lanes, padding_lanes, context_padding, cycles, patch_count, host_prep_s, chip_s, sampler_s):
        self.last = dict(
            bucket=tuple(bucket),
            real_lanes=real_lanes,
            padding_lanes=padding_lanes,
            context_padding=context_padding,
            cycles=cycles,
            patch_count=patch_count,
            host_prep_s=host_prep_s,
            chip_s=chip_s,
            sampler_s=sampler_s,
        )
        self.steps += 1
        for k in SUMS:
            self.sums[k] += self.last[k] or 0
        key = "x".join(str(v) for v in bucket)
        self.buckets[key] = self.buckets.get(key, 0) + 1

    def snapshot(self):
        return dict(
            steps=self.steps,
            **self.sums,
            buckets=dict(self.buckets),
            blocks_zeroed=self.blocks_zeroed,
            blocks_copied=self.blocks_copied,
            uptime_s=time.monotonic() - self.started,
        )
