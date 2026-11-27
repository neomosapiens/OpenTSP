"""Build and atomically publish TinyStories' fixed programs and shared weights."""

import multiprocessing
import os
from pathlib import Path
import tempfile

import numpy as np

from opentsp_vllm.models.tinystories import fixed_step as fixed
from opentsp_vllm.models.tinystories import program_cache as pc
from opentsp_tools.programs import needs
from opentsp_tools import pager
from opentsp_tools import profile


def stages(layers):
    for quad in fixed.COPY_QUADS:
        yield f"copy{quad}"
    yield "reset"
    yield "normalize"
    for chunk in range(fixed.CHUNKS):
        yield f"single{chunk}"
    yield "head"
    for layer in range(layers):
        yield f"pre{layer}"
        yield f"post{layer}"
        for chunk in range(fixed.CHUNKS):
            yield f"score{layer}_{chunk}"
            yield f"context{layer}_{chunk}"


def write_stage(model, root, name):
    directory = root / name
    directory.mkdir()
    k = model.emit(name)
    if k.view2.sites:
        raise ValueError(f"{name} contains instruction patch sites")
    asm = pager.tsp_as
    cfg = k.p.path
    _, obj = asm.assemble(k.b.prog.text(), config=cfg)
    own = [
        (n, None, [v[i : i + k.p.vlanes] for i in range(0, len(v), k.p.vlanes)])
        for n, (_, v) in k.weights.items()
        if n not in model.tensors
    ]
    literals = asm.write_weights(k.p.asm_name, own, cfg) if own else b""
    (directory / "program.elf").write_bytes(asm.write_object(obj, cfg, constants=literals))
    outputs = pc.output_regions(model.plan, 1) if name == "head" else []
    np.savez(
        directory / "bindings.npz",
        outputs=np.asarray(outputs, dtype=np.int32).reshape(-1, 3),
        **(pc.output_index() if outputs else {}),
    )
    pc.artifacts.seal(directory, dict(stage=name, max_cycles=k.last + 1024), payload=pc.artifacts.ELF_PAYLOAD)


def _initialize_worker(weights, directory, profile_name="ref"):
    global _model, _root
    _model = fixed.Model(profile.load(profile_name), weights)
    _root = directory


def _build(name):
    write_stage(_model, _root, name)


def build(cache, checkpoint=None, layers=8, jobs=1, seed=1, profile_name="ref"):
    paths = pc.generator_files() + [__file__, fixed.__file__]
    needs.SHOWCASE.check(profile.load(profile_name))
    source = pc.Model(layers, checkpoint=checkpoint, seed=seed, profile=profile_name, generator=pc.generator_digest(paths))
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / source.key
    names = list(stages(layers))
    with pc.bucket_lock(str(target)):
        if (target / "manifest.json").is_file():
            try:
                existing = pc.read_json(target / "manifest.json")
                if (
                    existing["metadata_sha256"] == pc.artifacts.metadata_digest(existing)
                    and existing["stages"] == names
                    and pc.artifacts.file_digest(target / "weights.safetensors") == source.manifest()["package_sha256"]
                ):
                    for name in names:
                        rec = pc.read_json(target / name / "program.json")
                        pc.artifacts.verify(target / name, rec, payload=pc.artifacts.ELF_PAYLOAD)
                        if rec["metadata_sha256"] != existing["stage_descriptors"][name]:
                            raise ValueError(f"stage {name} belongs to another build")
                    return target
            except (OSError, ValueError, KeyError):
                pass
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            directory = Path(tmp) / "package"
            directory.mkdir()
            model = fixed.Model(profile.load(profile_name), source.w)
            if jobs == 1:
                for n in names:
                    write_stage(model, directory, n)
            else:
                with multiprocessing.get_context("spawn").Pool(jobs, _initialize_worker, (source.w, directory, profile_name)) as pool:
                    for _ in pool.imap_unordered(_build, names):
                        pass
            (directory / "weights.safetensors").write_bytes(source.logical)
            manifest = dict(
                source.manifest(),
                execution="physical-kv",
                stages=names,
                vocab=source.cfg.vocab,
                hidden=source.cfg.hidden,
                heads=source.cfg.heads,
            )
            manifest["regions"] = {
                r.name: [list(r.slices), r.addr, r.words] for r in model.plan.regions if r.kind in ("pool", "data")
            }
            manifest["stage_descriptors"] = {
                n: pc.read_json(directory / n / "program.json")["metadata_sha256"] for n in names
            }
            manifest["metadata_sha256"] = pc.artifacts.metadata_digest(manifest)
            pc.write_json(directory / "manifest.json", manifest)
            if target.exists():
                pc.swap_directories(directory, target)
            else:
                os.replace(directory, target)
    return target
