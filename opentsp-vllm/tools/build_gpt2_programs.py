#!/usr/bin/env python3
"""Assemble the released four-chip schedules and pack weights from GPT-2 small."""

import argparse
from collections import OrderedDict
from hashlib import file_digest, sha256
import json
import os
import shutil
from pathlib import Path
import tempfile
from types import SimpleNamespace
import zipfile

from opentsp_tools import refasm
from opentsp_vllm import serving_artifacts as artifacts
from opentsp_vllm.models.gpt2 import weights
from opentsp_vllm.models.gpt2.config import GPT2_SMALL
from opentsp_vllm.models.gpt2.four_chip_packages import SEALS
from opentsp_vllm.models.gpt2.weight_layout import LANES, Layout, RESIDENT

SOURCES = Path(__file__).resolve().parents[1] / "demo" / "gpt2-program-sources.zip"
SOURCE_DIGEST = "ff002deb32f0e93bb4ed6b697c9705eccf0d10c33700e5798a84d89c2b376f4b"
CHECKPOINT_DIGEST = "dfb14c72e4c5ba1288e3c6fd6dd4e664e8d192f924c942dde755b6ee42576aaa"


MODEL_FILES = ("config.json", "generation_config.json", "merges.txt", "tokenizer.json",
               "tokenizer_config.json", "vocab.json")
ARCHIVE_VARIANTS = ("full", "sparse", "paired", "single-page", "single-page-pair")


def compile_unit(assembler, schedule, constants):
    """Resolve program-owned sources and leave logical coefficient bindings in the ELF."""
    images = OrderedDict((int(queue), words) for queue, words in schedule["initial"])
    externs = OrderedDict((item["name"], item) for item in schedule["externs"])
    program = SimpleNamespace(profile="ref", images=images, externs=externs)
    text, literals, recipe, zeroes = [], [], [], {}
    covered = set()
    for name, count, kind, source in schedule["preloads"]:
        if name in covered or externs[name]["size"] != count:
            raise ValueError(f"invalid scheduled load for {name}")
        covered.add(name)
        if kind == "zero":
            zeroes[name] = 1 if RESIDENT.fullmatch(name) else 3
        elif kind == "literal":
            length = count * LANES
            if not 0 <= source <= source + length <= len(constants):
                raise ValueError(f"literal {name} falls outside the source image")
            words = [constants[i:i + LANES] for i in range(source, source + length, LANES)]
            (text if name.startswith("tx") else literals).append((name, None, words))
        elif kind in ("matrix", "vector"):
            recipe.append([name, count, kind, source])
        else:
            raise ValueError(f"unknown scheduled source {kind!r}")
    if covered != externs.keys():
        raise ValueError("scheduled load sources do not cover placements")
    return assembler.write_object(
        program, text=assembler.write_weights("ref", text) if text else b"",
        constants=assembler.write_weights("ref", literals) if literals else b"",
        zeroes=zeroes, recipe=json.dumps(recipe, separators=(",", ":")).encode(),
    )


def write_bundle(archive, root, checkpoint, model, assembler):
    weights.save_deployment(root / "weights.safetensors", model)
    model_dir = root / "model"
    model_dir.mkdir()
    for name in MODEL_FILES:
        shutil.copyfile(checkpoint / name, model_dir / name)
    for variant in ARCHIVE_VARIANTS:
        family = root / variant
        family.mkdir()
        manifest = json.loads(archive.read(f"{variant}/manifest.json"))
        manifest["units_sha256"] = units = {}
        for chip in range(4):
            label = f"{variant}/s{chip}"
            unit = root / label
            unit.mkdir()
            descriptor = json.loads(archive.read(f"{label}/program.json"))
            for key in ("files_sha256", "metadata_sha256", "generation_seconds", "pack_seconds"):
                descriptor.pop(key, None)
            schedule = json.loads(archive.read(f"{label}/schedule.json"))
            program = compile_unit(assembler, schedule, archive.read(f"{label}/constants.bin"))
            (unit / "program.elf").write_bytes(program)
            (unit / "bindings.npz").write_bytes(archive.read(f"{label}/bindings.npz"))
            descriptor["elf_sha256"] = assembler.elf_identity(program)
            artifacts.seal(unit, descriptor, payload=artifacts.ELF_PAYLOAD)
            units[f"s{chip}"] = artifacts.unit_digest(json.loads((unit / "program.json").read_bytes()))
            print(f"{label}: packed", flush=True)
        raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
        (family / "manifest.json").write_bytes(raw)


def build(source, checkpoint, destination, *, prebuild=None):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"{destination} already exists; choose an empty output path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != SOURCE_DIGEST:
            raise ValueError(f"{source}: schedule source archive digest differs from the release pin")
    model = weights.load_checkpoint(GPT2_SMALL, str(checkpoint / "model.safetensors"))
    if model.digest() != CHECKPOINT_DIGEST:
        raise ValueError("checkpoint tensor digest does not match the four-chip GPT-2 schedule")
    assembler = refasm.load()
    with zipfile.ZipFile(source) as archive, tempfile.TemporaryDirectory(
            dir=destination.parent, prefix=".gpt2-programs-") as temporary:
        root = Path(temporary) / "programs"
        root.mkdir()
        write_bundle(archive, root, checkpoint, model, assembler)
        for variant in ARCHIVE_VARIANTS:
            if artifacts.file_digest(root / variant / "manifest.json") != SEALS[variant]:
                raise ValueError(f"{variant}: rebuilt bytes differ from the sealed ELF package")
        if prebuild:
            release = sha256(":".join(SEALS.values()).encode()).hexdigest()
            layout = Layout(model, release, directory=prebuild)
            for variant in ARCHIVE_VARIANTS:
                for chip in range(4):
                    unit = root / variant / f"s{chip}"
                    program = (unit / "program.elf").read_bytes()
                    meta = json.loads((unit / "program.json").read_bytes())
                    layout.resolve(chip, assembler._read_elf(program)[b".tsp.recipe"], meta["elf_sha256"])
        root.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cache = Path.home() / ".cache" / "opentsp"
    parser.add_argument("--checkpoint", type=Path, default=cache / "checkpoints" / "gpt2",
                        help="GPT-2 small checkpoint directory")
    parser.add_argument("--output", type=Path,
                        default=Path(os.environ.get("OPENTSP_GPT2_4_PROGRAMS") or cache / "gpt2-four-chip"),
                        help="new destination for the assembled bundle")
    parser.add_argument("--prebuild-layout", type=Path, help="prepare derived device layouts in a private cache directory")
    args = parser.parse_args()
    try:
        build(SOURCES, args.checkpoint.expanduser(), args.output.expanduser(), prebuild=args.prebuild_layout)
    except (OSError, ValueError) as error:
        parser.exit(2, f"gpt2-programs: {error}\n")


if __name__ == "__main__":
    main()
