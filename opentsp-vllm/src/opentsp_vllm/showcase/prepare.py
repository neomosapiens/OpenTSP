"""`just showcase-prepare`: everything the TinyStories showcase loads, built once under `$OPENTSP_SHOWCASE_DIR`
(default `~/.cache/opentsp/showcase`):

    model/      config.json and generation_config.json of the checkpoint, and the tokenizer files (vLLM tokenizes on the host)
    programs/   the shared weight preload and fixed programs for one chip

It reads the checkpoint `calum/tinystories-gpt2-3M` and the GPT-2 tokenizer from the local Hugging Face cache only (no download), and
is idempotent: an unchanged, verified package is reused.
Run with the serving environment's Python, which supplies the checkpoint loader and tokenizer.
"""

import argparse
import os
import pathlib
import shutil
import sys
import tempfile

from opentsp_vllm.serving_artifacts import file_digest

CHECKPOINT = "calum/tinystories-gpt2-3M"
TOKENIZER_REPO = "gpt2"
TOKENIZER_FILES = ["vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"]


def default_root():
    return pathlib.Path(os.environ.get("OPENTSP_SHOWCASE_DIR") or os.path.expanduser("~/.cache/opentsp/showcase"))


def _cached(repo, name):
    from huggingface_hub import hf_hub_download

    try:
        return pathlib.Path(hf_hub_download(repo, name, local_files_only=True))
    except Exception as e:  # noqa: BLE001 - a missing file is reported once, with the remedy
        raise SystemExit(f"showcase-prepare: {repo}/{name} is not in the local Hugging Face cache ({type(e).__name__}); "
                         f"fetch it with `hf download {repo} {name}`")


def write_model_dir(model):
    model.mkdir(parents=True, exist_ok=True)
    inputs = [(CHECKPOINT, "config.json"), (CHECKPOINT, "generation_config.json")]
    inputs.extend((TOKENIZER_REPO, name) for name in TOKENIZER_FILES)
    for repo, name in inputs:
        source, target = _cached(repo, name), model / name
        if target.is_file() and file_digest(target) == file_digest(source):
            continue
        with tempfile.TemporaryDirectory(dir=model) as tmp:
            staged = pathlib.Path(tmp) / name
            shutil.copyfile(source, staged)
            os.replace(staged, target)


def build_programs(root, checkpoint, jobs):
    from opentsp_vllm.models.tinystories import fixed_package

    return fixed_package.build(root / "programs", checkpoint=checkpoint, jobs=jobs)


def select_package(root, key):
    """Pin the showcase to the exact package returned by its builder."""
    target = pathlib.Path("programs") / key
    if not (root / target / "manifest.json").is_file():
        raise ValueError(f"missing published serving package {root / target}")
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        link = pathlib.Path(tmp) / "serving"
        link.symlink_to(target, target_is_directory=True)
        os.replace(link, root / "serving")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="the showcase directory (default $OPENTSP_SHOWCASE_DIR or ~/.cache/opentsp/showcase)")
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS") or 6), help="fixed stages built concurrently")
    a = ap.parse_args(argv)
    root = pathlib.Path(a.root) if a.root else default_root()
    write_model_dir(root / "model")
    print(f"showcase-prepare: model files in {root / 'model'}")
    package = build_programs(root, a.checkpoint, a.jobs)
    select_package(root, package.name)
    print(f"showcase-prepare: ready: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
