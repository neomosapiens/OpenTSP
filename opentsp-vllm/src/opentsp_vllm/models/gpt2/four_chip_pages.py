"""Bind the selected resident KV page into sealed instruction pages."""

from functools import lru_cache
import functools
import hashlib
import io
from types import SimpleNamespace

import numpy as np

from opentsp_tools import pager, refasm
from opentsp_tools import patch as operand_patch
from opentsp_vllm.ship_plan import PAGE

from . import pipeline
from .four_chip_execution import first_images
from .four_chip_packages import admits


def opener(p, stage, bindings):
    with np.load(io.BytesIO(bindings), allow_pickle=False) as d:
        sites = operand_patch.Table(list(d["page_names"]), *[d["page_" + k] for k in operand_patch.COLUMNS],
                                    stride=d["page_stride"], span=d["page_span"])
        table = operand_patch.Paged(sites, {int(q): int(n) for q, n in d["page_image_lengths"]},
                                    int(d["page_block_units"]))
        data = dict(zip(stage.externs, stage.runs, strict=True))
        names = iter(d["page_block_names"])
        blocks = []
        for q, i, at, length in d["page_block_rows"]:
            assert length in (1, 2)
            entries = [(str(n), *data[n]) for n in (next(names) for _ in range(length))]
            blocks.append(pager.Block(int(q), int(i), int(at), "ifetchm" if length == 1 else "ifetch",
                                      entries, p.vlanes))
        assert next(names, None) is None

    @lru_cache(maxsize=16)
    def bound(pages):
        current = bind_images(p, table, stage.images, blocks, pages)
        current.budget = stage.budget
        current.schedule = functools.partial(pipeline.Stage.schedule, current)
        return current

    baseline = bind_images(p, table, stage.images, blocks, None)
    return baseline, bound


def bind_images(p, table, image, blocks, pages):
    sites = table.table
    if pages is None:
        images, text = image, {}
    else:
        assert len(pages) == len(set(pages)) and all(0 <= page < 8 for page in pages)
        values = {str(name): int(low) + pages[int(str(name).split(".")[1][1:])] * int(stride)
                  for name, low, stride in zip(sites.names, sites.vmin, sites.stride, strict=True)}
        assembled = {(b.queue, b.index): b"".join(bits for _, _, _, bits in b.tensors) for b in blocks}
        images, text = operand_patch.apply_paged(operand_patch.to_image(image), assembled, table, values)
        images = operand_patch.to_units(images)
    stage = SimpleNamespace(images=images, text_pages={}, runs={})

    for b in pager.with_block_bytes(blocks, text):
        halves = ("",) if b.kind == "ifetchm" else ("a", "b")
        for (name, sl, address, bits), half in zip(b.tensors, halves, strict=True):
            stage.runs[name] = (sl, address, bits, b.cycle)
            stage.text_pages.setdefault(b.queue, {})[b.index, half] = bits
    assert next(iter(stage.runs.values()))[2]
    assert all(len(bits) % p.vlanes == 0 for _, _, bits, _ in stage.runs.values())
    return stage


def delta(p, before, after):
    """List changed physical text words and the first next-use deadline."""
    for name, (sl, addr, blob, deadline) in after.runs.items():
        old_sl, old_addr, old, _ = before.runs[name]
        assert (sl, addr, len(blob)) == (old_sl, old_addr, len(old))
        for j in range(0, len(blob), p.vlanes):
            word = blob[j:j + p.vlanes]
            if word != old[j:j + p.vlanes]:
                yield sl, addr + j // p.vlanes, word, deadline


def packet_splices(packets, stage):
    """Locate each packet's suffix of the initial image once."""
    asm = refasm.load()
    joined = {}
    for _, items in packets:
        for q, words in items:
            joined.setdefault(q, []).extend(words)
    cuts = {}
    for q, words in joined.items():
        source = stage.images[q]
        skip = next(i for i, unit in enumerate(source) if not asm.is_nop_word(unit))
        prefix = len(words) - len(source) + skip
        assert prefix >= 0 and words[prefix:] == list(source[skip:])
        cuts[q] = prefix, skip
    positions = dict.fromkeys(joined, 0)
    mapped = []
    for at, items in packets:
        locations = []
        for q, words in items:
            off = positions[q]
            limit = off + len(words)
            prefix, skip = cuts[q]
            count = max(0, min(len(words), prefix - off))
            first = skip + max(0, off - prefix)
            locations.append((q, list(words[:count]), first, first + len(words) - count))
            positions[q] = limit
        mapped.append((at, locations))
    return mapped


def materialize(packets, images):
    return [(at, [(q, prefix + images[q][first:last]) for q, prefix, first, last in places])
            for at, places in packets]


class PageTransferBudget(RuntimeError):
    """An otherwise admissible wave cannot meet its text transfer deadline."""


def open_wave(p, root, stages, period, waves, boot=None):
    """Prepare per-page-pair IQ text and writable fetch-page windows."""
    shift, init = boot if boot is not None else first_images(p, stages)
    opened = []
    for chip, stage in enumerate(stages[:3]):
        directory = root / f"s{chip}"
        bindings = (directory / "bindings.npz").read_bytes()
        if hashlib.sha256(bindings).hexdigest() != stage.meta["files_sha256"]["bindings.npz"]:
            raise ValueError(f"{directory}: page-binding file differs from its seal")
        opened.append(opener(p, stage, bindings))
    original = [entry[0] for entry in opened]
    previous = list(original)
    start_locs = [packet_splices(init[i], stage) for i, stage in enumerate(stages[:3])]
    loop_locs = [packet_splices(waves[i][1], stage) for i, stage in enumerate(stages[:3])]

    @lru_cache(maxsize=16)
    def layout(pages):
        bound = [entry[1](pages) for entry in opened]
        boot = [materialize(start_locs[i], b.images) for i, b in enumerate(bound)] + [init[3]]
        repeat = [(waves[i][0], materialize(loop_locs[i], b.images))
                  for i, b in enumerate(bound)] + [waves[3]]
        return bound, boot, repeat

    def prepare(groups, expected_period):
        if period != expected_period:
            raise ValueError("page binding modified the stage cycle envelope")
        changes = []
        minimum_slack = None
        pace = p.cfg["host"]["cycles_per_vector"]
        expected = []
        current = list(previous)
        for i, rows in enumerate(groups):
            if not admits(rows):
                raise ValueError("page specialization requires distinct, nonempty one-page masks")
            key = tuple(slot // PAGE for _, _, slot, _ in rows)
            bound, boot, repeat = layout(key)
            expected.append((boot, repeat))
            for chip, (stage, needed) in enumerate(zip(stages[:3], bound, strict=True)):
                # stage.budget includes the previous group's drain, so
                # all old text readers are past these SRAM locations.
                at = shift + (i - 1) * period + stage.budget if i else 0
                words = sorted(delta(p, current[chip], needed), key=lambda w: w[3])
                if words:
                    slack = min(shift + i * period + need - (at + (j + 1) * pace)
                                for j, (_, _, _, need) in enumerate(words))
                    if slack < 0:
                        raise PageTransferBudget(f"chip {chip} group {i} lacks {abs(slack)} cycles for changed text")
                    minimum_slack = slack if minimum_slack is None else min(minimum_slack, slack)
                    changes.append((at, chip, words))
                current[chip] = needed

        def commit():
            previous[:] = current

        return shift, expected, changes, minimum_slack, commit

    def reset():
        previous[:] = original

    return prepare, reset
