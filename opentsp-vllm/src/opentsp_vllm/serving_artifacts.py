"""On-disk serving programs. The model manifest selects the host execution contract."""
import hashlib
import json
from pathlib import Path

SCHEMA = "opentsp.serving"
VERSION = 1
PAYLOAD = ("program.elf", "sram.safetensors", "bindings.npz")
ELF_PAYLOAD = ("program.elf", "bindings.npz")
FILES = (*PAYLOAD, "program.json")


def manifest(model, chips, shared_preload=None, **metadata):
    return dict(metadata, schema=SCHEMA, version=VERSION, model=model, chips=chips, shared_preload=shared_preload)


def file_digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def metadata_digest(record):
    body = json.dumps({k: v for k, v in record.items() if k != "metadata_sha256"}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()



def unit_digest(record):
    """The release identity excludes only representation-dependent transport fields."""
    transport = record["files_sha256"]
    canonical = dict(record, files_sha256=dict(transport, **{"program.elf": record["elf_sha256"]}))
    return metadata_digest(canonical)


def seal(directory, record, *, payload=PAYLOAD):
    directory = Path(directory)
    record = dict(record, files_sha256={name: file_digest(directory / name) for name in payload})
    record["metadata_sha256"] = metadata_digest(record)
    (directory / "program.json").write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def _payload_digests(record, payload=PAYLOAD):
    files = record.get("files_sha256")
    if not isinstance(files, dict) or set(files) != set(payload):
        raise ValueError("program.json has an incomplete payload inventory")
    if metadata_digest(record) != record.get("metadata_sha256"):
        raise ValueError("program.json metadata sha256 differs from publication")
    return files


def verified_payload(directory, record, *, payload=PAYLOAD):
    """Return the bytes actually hashed for execution."""
    files = _payload_digests(record, payload)
    contents = {}
    for name in payload:
        contents[name] = (Path(directory) / name).read_bytes()
        if hashlib.sha256(contents[name]).hexdigest() != files[name]:
            raise ValueError(f"{name} sha256 differs from publication")
    return contents


def verify(directory, record, *, payload=PAYLOAD):
    for name, expected in _payload_digests(record, payload).items():
        if file_digest(Path(directory) / name) != expected:
            raise ValueError(f"{name} sha256 differs from publication")
