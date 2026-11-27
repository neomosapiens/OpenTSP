"""Inspect explicit issue cycles in assembled objects and sealed stage packages."""

import argparse
import base64
from collections import defaultdict
from functools import lru_cache
import hashlib
import heapq
import json
from pathlib import Path
import re
import struct

from . import refasm


GROUPS = ("MEM", "VXM", "MXM", "SXM", "C2C", "CU")
PAGE = re.compile(r"tx(\d+)b(\d+)([ab]?)")
PAYLOAD = ("program.elf", "bindings.npz")
SAFE_CYCLE = (1 << 53) - 1


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pack(values):
    """Pack nonnegative numbers into unsigned LEB128 for the standalone viewer."""
    out = bytearray()
    for value in values:
        while value >= 128:
            out.append((value & 127) | 128)
            value >>= 7
        out.append(value)
    return base64.b64encode(out).decode("ascii")


def pages(asm, externs, tensors):
    text_externs = [(PAGE.fullmatch(ext["name"]), ext) for ext in externs]
    declared = [(m, ext) for m, ext in text_externs if m is not None]
    if not declared or not tensors:
        return {}
    chunks = defaultdict(dict)
    for match, ext in declared:
        queue, block, half = match.groups()
        tensor = tensors.get(ext["name"])
        if tensor is None or len(tensor[1]) != ext["size"]:
            raise ValueError(f"{ext['name']}: missing or wrong-size instruction text payload")
        chunks[int(queue), int(block)][half] = b"".join(tensor[1])
    out = defaultdict(list)
    for (queue, block), parts in sorted(chunks.items()):
        if (block != len(out[queue])
                or set(parts) not in ({""}, {"a", "b"})):
            raise ValueError(f"queue {queue}: incomplete or noncontiguous text block {block}")
        out[queue].append([
            word for (word,) in struct.iter_unpack("<I", b"".join(parts[k] for k in sorted(parts)))
            if word != asm.PADDING
        ])
    return out


def stage(asm, object_bytes, weight_bytes, profile, physical_chip, window):
    profile, images, externs = asm.read_object(object_bytes, profile)
    tensors = asm.object_preloads(object_bytes, profile, externs)
    if weight_bytes is not None:
        asm.merge_preloads(tensors, asm.read_weights(weight_bytes, profile))
    fetched = pages(asm, externs, tensors)
    unknown_queues = fetched.keys() - images.keys()
    if unknown_queues:
        raise ValueError(f"text blocks with no initial image: {sorted(unknown_queues)}")
    isa = asm.ISA(profile)

    @lru_cache(maxsize=65536)
    def opcode(queue, word, width):
        err, entry, _ = isa.decode_word(queue, word, width)
        if err or not entry:
            raise ValueError(f"{asm.queue_name(queue)}: undecodable instruction {word:#x}")
        return entry["name"]

    tracks = []
    incomplete = False
    for queue, image in sorted(images.items()):
        blocks = fetched.get(queue, ())
        text = image + [unit for block in blocks for unit in block]
        by_cycle = defaultdict(list)
        fetches = 0
        for cycle, _, word, width in asm.raise_images({queue: text}):
            if cycle > SAFE_CYCLE:
                raise ValueError(f"cycle exceeds the viewer's exact integer range: {cycle}")
            name = opcode(queue, word, width)
            fetches += name in ("Ifetch", "IfetchM")
            if window is None or window[0] <= cycle <= window[1]:
                by_cycle[cycle].append(name)
        if blocks and len(blocks) != fetches:
            raise ValueError(f"{asm.queue_name(queue)}: {len(blocks)} text blocks for {fetches} fetches")
        incomplete |= fetches > 0 and not blocks
        if by_cycle:
            name = asm.queue_name(queue)
            group = re.split(r"[.\[]", name, maxsplit=1)[0]
            tracks.append(dict(queue=queue, name=name, unit=group, issues=[
                (cycle, " + ".join(names)) for cycle, names in by_cycle.items()
            ]))
    budget = max((q["issues"][-1][0]+1 for q in tracks), default=1)
    return dict(physical_chip=physical_chip, budget=budget, tracks=tracks), profile, incomplete


def unit_payload(asm, directory, expected, chip):
    record = json.loads((directory / "program.json").read_bytes())
    if record.get("chip") != chip:
        raise ValueError(f"{directory}: metadata targets a different chip")
    files = record.get("files_sha256", {})
    if set(files) != set(PAYLOAD):
        raise ValueError(f"{directory}: incomplete payload inventory")
    body = {k: v for k, v in record.items() if k != "metadata_sha256"}
    if sha(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()) != record.get("metadata_sha256"):
        raise ValueError(f"{directory}: invalid program metadata seal")
    body["files_sha256"] = dict(files, **{"program.elf": record["elf_sha256"]})
    if sha(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()) != expected:
        raise ValueError(f"{directory}: program metadata differs from the manifest")
    object_bytes = (directory / "program.elf").read_bytes()
    with (directory / "bindings.npz").open("rb") as stream:
        bindings = hashlib.file_digest(stream, "sha256").hexdigest()
    if sha(object_bytes) != files["program.elf"] or bindings != files["bindings.npz"]:
        raise ValueError(f"{directory}: payload digest mismatch")
    if asm.elf_identity(object_bytes) != record["elf_sha256"]:
        raise ValueError(f"{directory}: decoded ELF differs from release identity")
    return object_bytes


def sealed_route(asm, label, root, window):
    raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    route, units = manifest["route"], manifest["units_sha256"]
    if (manifest.get("schema") != "opentsp.serving" or manifest.get("version") != 1
            or not 1 <= len(route) <= 512 or len(set(route)) != len(route)
            or any(type(chip) is not int or not 0 <= chip < 512 for chip in route)
            or manifest.get("chips") != len(route) or set(units) != {f"s{i}" for i in range(len(route))}):
        raise ValueError(f"{root}: expected a complete sealed serving route")
    panel = manifest.get("panel_width", 1)
    if type(panel) is not int or panel < 1:
        raise ValueError(f"{root}: invalid panel width")
    stages = []
    incomplete = False
    for i, chip in enumerate(route):
        name = f"s{i}"
        object_bytes = unit_payload(asm, root/name, units[name], chip)
        unit, profile, partial = stage(asm, object_bytes, None, manifest["profile"], chip, window)
        stages.append(unit)
        incomplete |= partial
    return dict(label=label, manifest=sha(raw), profile=profile, panel=panel,
                chips=stages, incomplete=incomplete)


def raw_object(asm, label, path, weights, window):
    body = path.read_bytes()
    weight_bytes = weights.read_bytes() if weights else None
    unit, profile, incomplete = stage(asm, body, weight_bytes, None, 0, window)
    # Timing is for an object and its paged text, when supplied.
    digest = sha(body) if weight_bytes is None else sha(bytes.fromhex(sha(body)) + bytes.fromhex(sha(weight_bytes)))
    return dict(label=label, manifest=digest, profile=profile, panel=1,
                chips=[unit], incomplete=incomplete)


def checked_digest(receipt, route):
    if not isinstance(receipt, dict):
        raise ValueError("evidence must be a JSON object")
    if receipt.get("source_sha256") != route["manifest"]:
        raise ValueError(f"{route['label']}: evidence source digest does not match the input")


def timing(receipt, route):
    checked_digest(receipt, route)
    fields = ("period", "cycles", "rows", "boot")
    period, cycles, rows, boot = (receipt[k] for k in fields)
    if any(type(x) is not int or not 0 <= x <= SAFE_CYCLE for x in (period, cycles, rows, boot)):
        raise ValueError("timing needs exact nonnegative period, cycles, rows and boot")
    if not period or not cycles or not rows or rows % route["panel"] or rows // route["panel"] > 64:
        raise ValueError("timing needs one through 64 complete panels and a positive period and cycle count")
    if boot + (rows // route["panel"] - 1) * period >= cycles:
        raise ValueError("the last panel starts outside the measured wave")
    return {k: receipt[k] for k in fields}


def trace(receipt, route):
    checked_digest(receipt, route)
    chip, extent, queues, marks = (receipt[k] for k in ("chip", "extent", "queues", "waits"))
    if (type(chip) is not int or not 0 <= chip < len(route["chips"]) or len(extent) != 2
            or not all(type(v) is int and 0 <= v <= SAFE_CYCLE for v in extent) or extent[0] >= extent[1]):
        raise ValueError("trace needs a valid stage index and a half-open cycle extent")
    queues = sorted(set(queues))
    if any(type(q) is not int or not 0 <= q <= 65535 for q in queues):
        raise ValueError("invalid trace queue IDs")
    for cycle, queue, reason, due in marks:
        if (type(cycle) is not int or not extent[0] <= cycle < extent[1] or queue not in queues
                or type(due) is not int or not 0 <= due <= SAFE_CYCLE or not isinstance(reason, str)):
            raise ValueError("invalid wait sample")
    return dict(label=route["label"], chip=chip, extent=extent, queues=queues, waits=marks)


def report_queue(track, top, stage_span):
    issues = track["issues"]
    span = issues[-1][0] - issues[0][0] + 1

    def gaps():
        for (left, before), (right, after) in zip(issues, issues[1:]):
            count = right-left-1
            if count:
                yield dict(start=left+1, end=right-1, cycles=count, before=before, after=after,
                           share_of_stage_span_pct=round(100*count/stage_span, 6))

    idle = span - len(issues)
    return dict(queue=track["queue"], name=track["name"], unit=track["unit"], first=issues[0][0],
                last=issues[-1][0], issue_cycles=len(issues), span_cycles=span, gap_cycles=idle,
                gap_share_pct=round(100*idle/span, 6),
                largest=heapq.nsmallest(top, gaps(), key=lambda g: (-g["cycles"], g["start"])))


def report_stage(stage, top):
    tracks = stage["tracks"]
    first = min((track["issues"][0][0] for track in tracks), default=0)
    last = max((track["issues"][-1][0] for track in tracks), default=0)
    span = last-first+1 if tracks else 0
    queues = [report_queue(q, top, span) for q in tracks]
    totals = defaultdict(lambda: dict(issue_cycles=0, gap_cycles=0, span_cycles=0))
    for q in queues:
        for key in totals[q["unit"]]:
            totals[q["unit"]][key] += q[key]
    for counts in totals.values():
        counts["gap_share_pct"] = round(100*counts["gap_cycles"]/counts["span_cycles"], 6)
    largest = heapq.nsmallest(top, (dict(g, queue=q["queue"], unit=q["unit"]) for q in queues for g in q["largest"]),
                              key=lambda g: (-g["cycles"], g["queue"], g["start"]))
    return dict(physical_chip=stage["physical_chip"], span_cycles=span, units=dict(totals), queues=queues, largest=largest)


def viewer_route(route):
    names, codes = [], {}
    stages = []
    for unit in route["chips"]:
        rows = []
        for track in unit["tracks"]:
            prev, deltas, entries = 0, [], []
            for cycle, text in track["issues"]:
                deltas.append(cycle-prev)
                prev = cycle
                if text not in codes:
                    codes[text] = len(names)
                    names.append(text)
                entries.append(codes[text])
            rows.append([track["queue"], GROUPS.index(track["unit"]), track["name"], pack(deltas), pack(entries)])
        stages.append(dict(physical_chip=unit["physical_chip"], budget=unit["budget"], queues=rows))
    return dict(route, chips=stages, opcodes=names)


def inputs_by_label(args, option):
    result = {}
    for label, path in args:
        if label in result:
            raise ValueError(f"{option}: {label!r} appears more than once")
        result[label] = Path(path)
    return result


def generate(args, asm):
    objects = inputs_by_label(args.object, "--object")
    sealed = inputs_by_label(args.route, "--route")
    weights = inputs_by_label(args.weights, "--weights")
    timings = inputs_by_label(args.timing, "--timing")
    traces = inputs_by_label(args.trace, "--trace")
    labels = objects.keys() | sealed.keys()
    if not 1 <= len(labels) <= 2 or objects.keys() & sealed.keys():
        raise ValueError("supply one or two distinctly labelled objects or routes")
    if weights.keys() - objects.keys() or (timings.keys() | traces.keys()) - labels:
        raise ValueError("weights must name an object and evidence must name an input")
    if args.window and not 0 <= args.window[0] <= args.window[1] < SAFE_CYCLE:
        raise ValueError("--window needs an inclusive start and end in the exact cycle range")
    if not 1 <= args.top <= 1000:
        raise ValueError("--top must be between 1 and 1000")
    if args.html and args.json and args.html.resolve() == args.json.resolve():
        raise ValueError("HTML and JSON outputs must be separate files")
    outputs = {p.resolve() for p in (args.html, args.json) if p}
    if any(p.resolve() in outputs for group in (objects, weights, timings, traces) for p in group.values()):
        raise ValueError("an output would overwrite an input")
    if any(p.resolve().is_relative_to(root.resolve()) for p in outputs for root in sealed.values()):
        raise ValueError("keep outputs outside sealed route directories")

    routes = [raw_object(asm, label, file, weights.get(label), args.window) for label, file in objects.items()]
    routes += [sealed_route(asm, label, path, args.window) for label, path in sealed.items()]
    overlays = []
    for route in routes:
        label = route["label"]
        route["observed"] = timing(json.loads(timings[label].read_text()), route) if label in timings else None
        if label in traces:
            overlays.append(trace(json.loads(traces[label].read_text()), route))
    if args.html:
        template = Path(__file__).with_name("schedule_viewer.html").read_text()
        data = dict(groups=GROUPS, routes=[viewer_route(r) for r in routes], traces=overlays, window=args.window)
        blob = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
        args.html.write_text(template.replace("/*__SCHEDULE_DATA__*/", blob))
    if args.json:
        summaries = [dict({k: v for k, v in r.items() if k not in {"chips", "manifest"}},
                          source_sha256=r["manifest"], stages=[report_stage(s, args.top) for s in r["chips"]])
                     for r in routes]
        args.json.write_text(json.dumps(dict(schema="opentsp.schedule.v1", window=args.window, routes=summaries),
                                        indent=2) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--object", action="append", nargs=2, default=[], metavar=("LABEL", "FILE"))
    ap.add_argument("--route", action="append", nargs=2, default=[], metavar=("LABEL", "DIRECTORY"))
    ap.add_argument("--weights", action="append", nargs=2, default=[], metavar=("LABEL", "SAFETENSORS"))
    ap.add_argument("--timing", action="append", nargs=2, default=[], metavar=("LABEL", "JSON"))
    ap.add_argument("--trace", action="append", nargs=2, default=[], metavar=("LABEL", "JSON"))
    ap.add_argument("--window", type=int, nargs=2, metavar=("FIRST", "LAST"), help="inclusive static cycle window")
    ap.add_argument("--top", type=int, default=20, help="number of ranked gaps per queue and stage in JSON")
    ap.add_argument("--html", type=Path, help="write the interactive offline comparison")
    ap.add_argument("--json", type=Path, help="write per-stage, per-unit and per-queue bubble statistics")
    args = ap.parse_args(argv)
    if not args.html and not args.json:
        ap.error("supply --html, --json, or both")
    try:
        asm = refasm.load()
    except (OSError, ImportError) as error:
        ap.error(str(error))
    try:
        generate(args, asm)
    except (OSError, ValueError, TypeError, KeyError, asm.AsmError) as error:
        ap.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
