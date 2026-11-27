"""Execute the sealed four-chip GPT-2 layout and decode its output."""

import time

import numpy as np

from opentsp_tools import profile, refasm
from opentsp_tools.programs import data, ref_wide as rw
from opentsp_vllm.ras import check_completion, probe_word

from . import attention, layer, system
from .config import GPT2_SMALL
from .wide import pad_lanes

NCHIPS = 4
NTRANSFORMER = 3
LAYERS_PER_STAGE = 4
TILES = 393
SHARDS = (TILES,)
WINDOW = 16


def stripe_maps(p, chunks, owner, slot):
    """Bind one append address per physical slice pair in the resident KV layout."""
    if chunks != 2:
        raise ValueError("four-chip layout requires 256 shared KV positions")
    k_base = 0x1400 + 0x100 * owner
    v_base = 0x100 + 256 * owner
    sink = 0x17c0
    maps = layer.map_weights(p, 6, chunks, slot, 99, k_base=k_base, v_base=v_base, sink=sink)
    append = maps.pop("mv")
    other = np.frombuffer(append, dtype=np.uint8).copy()
    other[0::16], other[1::16] = sink & 255, sink >> 8
    pair = slot & 1
    maps[f"mv{pair}"] = append
    maps[f"mv{pair ^ 1}"] = other.tobytes()
    return maps


def update_controls(sim, stage, chip, p, chunks, row, mask_chunk=None):
    _, _, slot, counts = row
    if mask_chunk is not None:
        if slot // 128 != mask_chunk or np.any(counts[:mask_chunk]) or np.any(counts[mask_chunk + 1:]):
            raise ValueError("request has KV in a chunk excluded from the scheduled attention")
        counts = counts[mask_chunk:mask_chunk + 1]
    masks = attention.enc_weights(p, counts)
    for i in range(LAYERS_PER_STAGE):
        controls = stripe_maps(p, chunks, i, slot)
        controls.update(masks)
        for name, bits in controls.items():
            sl, addr, size = stage.externs[f"l{i}{name}"]
            assert len(bits) == size * p.vlanes
            sim.write_words(chip, sl, addr, bits)


def unpack_sequential(sim, count):
    result = [[] for _ in range(count)]
    latest = [0 for _ in range(count)]
    for shard, tiles in enumerate(SHARDS):
        stream = sim.egress(NTRANSFORMER + shard)
        stride = 4 * tiles
        assert len(stream) == count * stride
        for row in range(count):
            egress = stream[row*stride:(row+1)*stride]
            latest[row] = max(latest[row], egress[-1][0])
            values = np.concatenate([
                data.from_planes([word[:128] for _, word in egress[t:t+4]], "f32")
                for t in range(0, stride, 4)
            ])
            result[row].append(values)
    return [np.concatenate(v)[:GPT2_SMALL.vocab] for v in result], latest


def first_images(p, stages, shift=128):
    """Build initial host Enqueue events for a resident SRAM program switch."""
    cap = p.icu["iq_depth"] // 4
    rate = p.icu["fetch_bytes_per_cycle"] // 4
    assert shift >= (cap + rate - 1) // rate + 2
    asm = refasm.load()
    events = []
    for stage in stages:
        by_cycle = {0: []}
        for q, units in stage.images.items():
            first_instruction = next(i for i, unit in enumerate(units) if not asm.is_nop_word(unit))
            first = sum(units[:first_instruction]) + shift
            prefix = [65535] * (first // 65535)
            if first % 65535:
                prefix.append(first % 65535)
            image = prefix + list(units[first_instruction:])
            by_cycle[0].append((q, image[:cap]))
            if len(image) > cap:
                by_cycle.setdefault(first + 1, []).append((q, image[cap:]))
        events.append(sorted(by_cycle.items()))
    return shift, events


def observe_plan(observe, stages, shift, period, launches, width, limit):
    if observe is None:
        return
    handoffs = []
    for index in range(len(launches)):
        group, row = divmod(index, width)
        offset = shift + group * period
        handoffs.append([offset + stage.meta["sends"][0][24 * (row + 1) - 1] for stage in stages[:3]])
    observe("wave-plan", limit=limit, ingress_cycles=launches, handoffs=handoffs)


class WaveReceives:
    """Observe sealed-route rows as their 24th vector enters each destination.

    FIFO order on outgoing link 0 / incoming link 2 is the sealed four-chip
    package's row order. Clock values here come from actual DMA and Receive.
    """

    def __init__(self, sim, count, emit):
        self.sim, self.count, self.emit = sim, count, emit
        self.received = [[], [], [], []]
        self.last = [0] * count
        self.available = emit is not None and hasattr(sim, "observe_link_receives")
        if self.available:
            sim.observe_link_receives()
        elif emit is not None:
            emit("trace-gap", reason="The native binding has no link receive observer")

    def horizon(self, cycle, done=False):
        if not self.available:
            return
        self.received[0] = [at for at, kind in self.sim.dma_events(0) if kind == "in"]
        for chip in range(1, 4):
            self.received[chip].extend(at for at, link in self.sim.drain_link_receives(chip) if link == 2)
        sizes = [len(data) for data in self.received]
        if any(size > self.count * 24 for size in sizes) or (done and sizes != [24 * self.count] * 4):
            self.emit("trace-gap", reason="Observed vectors disagree with the sealed row layout", counts=sizes)
            self.available = False
            return
        stages = [sum(size >= 24 * (row + 1) for size in sizes) for row in range(self.count)]
        moves = [[row, stage, self.received[stage - 1][24 * (row + 1) - 1]]
                 for row, current in enumerate(stages) for stage in range(self.last[row] + 1, current + 1)]
        if moves:
            self.emit("wave-positions", cycle=cycle, stages=stages, moves=moves)
        self.last = stages


def drive_sequential(sim, model, chunks, stages, period, waves, rows, *, health_target=None, flips=(), boot=None, mask_chunk=None, observe=None):
    n = len(rows)
    p = profile.load("ref")
    start = time.perf_counter()
    sim.restart()
    telemetry = WaveReceives(sim, len(rows), observe)
    shift, initial = boot if boot is not None else (0, ())
    text_bytes = 0
    for chip, injections in enumerate(initial):
        for at, parts in injections:
            sim.inject_text(chip, at, parts)
            text_bytes += sum(4 * len(units) for _, units in parts)
    for i in range(NTRANSFORMER):
        update_controls(sim, stages[i], i, p, chunks, rows[0], mask_chunk)
    for j, (token, pos, _, _) in enumerate(rows):
        hidden = system.embed(model, token, pos)
        words = (v for tile in rw.split(hidden, 128) for v in data.to_planes(pad_lanes(tile, p.vlanes)))
        for at, bits in zip(stages[0].dma_cycles, words, strict=True):
            sim.dma_write(0, bits, shift + j*period + at)
    updates = []
    for row in range(1, n):
        for chip, (boundary, injections) in enumerate(waves):
            for cycle, parts in injections:
                sim.inject_text(chip, shift + row*period + cycle, parts)
                text_bytes += sum(4 * len(units) for _, units in parts)
            if chip < NTRANSFORMER:
                updates.append((shift + row*period + boundary, chip, row))
    if health_target is not None:
        chip, sl, addr = health_target
        if not 0 <= chip < NCHIPS:
            raise RuntimeError("health check targets a chip outside the fence")
        for member in range(NCHIPS):
            probe_word(lambda: sim.read_words(member, sl, addr, 1),
                       lambda bits: sim.write_words(member, sl, addr, bits), p.vlanes)
    reports = lambda: [(chip, sim.ras_snapshot(chip)) for chip in range(NCHIPS)]
    limit = shift + max(s.budget for s in stages) + period * (n-1) + 300
    launches = [shift + j * period + stages[0].dma_cycles[0] for j in range(n)]
    observe_plan(observe, stages, shift, period, launches, 1, limit)
    run_start = time.perf_counter()
    try:
        for at, chip, row in sorted(updates):
            _, outcome = sim.run_to_done(max_cycles=at - 1)
            if outcome["reason"] != "limit":
                raise RuntimeError(f"expected SHIP control boundary at {at}, got {outcome}")
            if observe:
                observe("wave-horizon", cycle=outcome["cycle"])
                telemetry.horizon(outcome["cycle"])
            update_controls(sim, stages[chip], chip, p, chunks, rows[row], mask_chunk)
        _, result = sim.run_to_done(max_cycles=limit)
        if result["reason"] != "done":
            raise RuntimeError(f"pipeline stopped at {result['reason']} before completion")
        if observe:
            observe("wave-horizon", cycle=result["cycle"])
            telemetry.horizon(result["cycle"], done=True)
        sim.finish()
    except RuntimeError as error:
        check_completion(reports(), error)
        raise
    for flip in flips:
        sim.inject_sram(*flip)
    sim.service_scrub()
    check_completion(reports())
    native_s = time.perf_counter() - run_start
    logits, arrive = unpack_sequential(sim, n)
    total_s = time.perf_counter() - start
    return logits, dict(native_seconds=native_s, pipeline_seconds=total_s, cycles=result["cycle"],
                        load_s=run_start - start, collect_s=total_s - (run_start - start) - native_s,
                        period=period, arrival_cycles=arrive, launch_cycles=launches, text_bytes=text_bytes,
                        stage_ticks=[sim.active_ticks(chip) for chip in range(NCHIPS)])


def update_panel_controls(sim, stage, chip, rows, p):
    for owner in range(4):
        for member, (_, _, slot, counts) in enumerate(rows):
            assert slot < 128 and not np.any(counts[1])
            words = stripe_maps(p, 2, owner, slot)
            if f"l{owner}r{member}encp0" in stage.externs:
                encoded_counts = counts[:1]
                if getattr(stage, "meta", {}).get("compact_qk") is True:
                    if np.count_nonzero(encoded_counts) != 1 or counts[0, slot // 16] <= 0:
                        raise ValueError("compact QK requires one visible physical page")
                    encoded_counts = np.zeros_like(encoded_counts)
                    encoded_counts[0, 0] = counts[0, slot // 16]
                words.update(attention.enc_weights(p, encoded_counts))
            else:
                assert slot % 16 == 0 and counts[0, slot // 16] == 1 and np.count_nonzero(counts) == 1
            for name, payload in words.items():
                sl, addr, size = stage.externs[f"l{owner}r{member}{name}"]
                assert len(payload) == size * p.vlanes
                sim.write_words(chip, sl, addr, payload)


def output_rows(sim, groups, width, window, tiles_per_vector=1):
    if tiles_per_vector not in (1, 2):
        raise ValueError(f"unsupported vocabulary vector layout: {tiles_per_vector}")
    stream = sim.egress(3)
    windows = [min(window, TILES - start) for start in range(0, TILES, window)]
    vectors_per_row = sum((n + tiles_per_vector - 1) // tiles_per_vector for n in windows) * 4
    expected = groups * width * vectors_per_row
    if len(stream) != expected:
        raise RuntimeError(f"vocabulary stream has {len(stream)} vectors, expected {expected}")
    cursor, logits, latest = 0, [], []
    for _ in range(groups):
        row = [[] for _ in range(width)]
        times = [0] * width
        for n in windows:
            for i in range(width):
                for start in range(0, n, tiles_per_vector):
                    planes = stream[cursor:cursor+4]
                    count = min(n - start, tiles_per_vector)
                    for tile in range(count):
                        start_lane = tile * 128
                        row[i].append(data.from_planes(
                            [b[start_lane:start_lane+128] for _, b in planes], "f32",
                        ))
                    times[i] = planes[-1][0]
                    cursor += 4
        logits.extend(np.concatenate(values)[:GPT2_SMALL.vocab] for values in row)
        latest.extend(times)
    return logits, latest, expected


def drive_panel(sim, model, stages, rows, p, period, waves, *, boot=None, health_target=None, flips=(), prepare_pages=None, observe=None):
    width, extra = divmod(len(stages[0].dma_cycles), 24)
    assert not extra and width in (1, 2) and len(rows) % width == 0
    groups = [rows[i:i+width] for i in range(0, len(rows), width)]
    start = time.perf_counter()
    sim.restart()
    linked_panel = prepare_pages is not None and health_target is None and not flips
    if linked_panel:
        sim._prepare_linked_panel()

    linked_panel_windows = 0

    def run_panel(limit):
        nonlocal linked_panel, linked_panel_windows
        if linked_panel:
            result = sim._run_linked_panel(limit)
            if result is not None:
                linked_panel_windows += 1
                return result
            linked_panel = False
        return sim.run_to_done(max_cycles=limit)[1]

    telemetry = WaveReceives(sim, len(rows), observe)
    shift, initial = boot if boot is not None else (0, ())
    transfers, minimum_slack, commit = [], None, None
    repeats = [waves] * len(groups)
    if prepare_pages is not None:
        shift, layouts, transfers, minimum_slack, commit = prepare_pages(groups, period)
        initial = layouts[0][0]
        repeats = [repeat for _, repeat in layouts]
    text_bytes = 0
    for chip, injections in enumerate(initial):
        for cycle, parts in injections:
            sim.inject_text(chip, cycle, parts)
            text_bytes += sum(len(words)*4 for _, words in parts)
    launches = []
    for chip, stage in enumerate(stages[:3]):
        update_panel_controls(sim, stage, chip, groups[0], p)
    for group, members in enumerate(groups):
        data_in = []
        for token, position, _, _ in members:
            hidden = system.embed(model, token, position)
            data_in.extend(b for v in rw.split(hidden, 128) for b in data.to_planes(pad_lanes(v, p.vlanes)))
        group_start = shift + group*period
        launches.extend(group_start + at for at in stages[0].dma_cycles[::24])
        for at, bits in zip(stages[0].dma_cycles, data_in, strict=True):
            sim.dma_write(0, bits, at + group_start)
    updates = []
    for at, chip, words in transfers:
        if at:
            updates.append((at, chip, None, words))
        else:
            for sl, addr, blob, _ in words:
                sim.write_words(chip, sl, addr, blob)
    for group in range(1, len(groups)):
        for chip, (boundary, injections) in enumerate(repeats[group]):
            for cycle, parts in injections:
                sim.inject_text(chip, shift+group*period+cycle, parts)
                text_bytes += sum(len(words)*4 for _, words in parts)
            if chip < 3:
                updates.append((shift+group*period+boundary, chip, group, ()))
    if health_target is not None:
        chip, sl, addr = health_target
        if not 0 <= chip < 4:
            raise RuntimeError("health check targets a chip outside the fence")
        for member in range(4):
            probe_word(lambda: sim.read_words(member, sl, addr, 1),
                       lambda bits: sim.write_words(member, sl, addr, bits), p.vlanes)
    reports = lambda: [(chip, sim.ras_snapshot(chip)) for chip in range(4)]
    limit = shift + max(s.budget for s in stages) + (len(groups)-1)*period + 300
    observe_plan(observe, stages, shift, period, launches, width, limit)
    native_start = time.perf_counter()
    try:
        for at, chip, group, words in sorted(updates, key=lambda item: item[:2]):
            result = run_panel(at-1)
            if result["reason"] != "limit":
                raise RuntimeError(f"expected panel control boundary at {at}, got {result}")
            if observe:
                observe("wave-horizon", cycle=result["cycle"])
                telemetry.horizon(result["cycle"])
            if group is None:
                for sl, addr, blob, _ in words:
                    sim.write_words(chip, sl, addr, blob)
            else:
                update_panel_controls(sim, stages[chip], chip, groups[group], p)
        result = run_panel(limit)
        if result["reason"] != "done":
            raise RuntimeError(f"panel pipeline stopped at {result['reason']} before completion")
        if observe:
            observe("wave-horizon", cycle=result["cycle"])
            telemetry.horizon(result["cycle"], done=True)
        sim.finish()
    except RuntimeError as error:
        check_completion(reports(), error)
        raise
    for flip in flips:
        sim.inject_sram(*flip)
    sim.service_scrub()
    check_completion(reports())
    if commit is not None:
        commit()
    native_end = time.perf_counter()
    head = stages[3].meta
    out, completion, host_vectors_out = output_rows(
        sim, len(groups), width, head.get("head_window", WINDOW), head.get("head_tiles_per_vector", 1),
    )
    end = time.perf_counter()
    return out, dict(host_vectors_out=host_vectors_out, seconds=end-start, cycles=result["cycle"], period=period,
                     cycles_per_token=period/width, completion_cycles=completion, launch_cycles=launches,
                     native_seconds=native_end-native_start, load_s=native_start-start,
                     collect_s=end-native_end, arrival_cycles=completion, text_bytes=text_bytes,
                     stage_ticks=[sim.active_ticks(i) for i in range(4)],
                     page_text_words=sum(len(words) for _, _, words in transfers),
                     minimum_page_text_slack=minimum_slack, linked_panel_windows=linked_panel_windows)
