"""Architectural observations through the real host, dispatcher and slice interface."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sim, cfg = sys.argv[1:3]
config = json.loads(Path(cfg).read_text())
lanes = 16 * config["geometry"]["superlanes"]


def run(text, *flags, profile=cfg):
    return subprocess.run([sim, profile, "--trace", *flags], input=text, text=True,
                          capture_output=True, check=True).stdout.splitlines()


def check_dma():
    # Every subset of the chip's four RX issuers, with distinct destinations.
    # Five head counts cover empty, every partial win, and all four succeeding.
    qs = (128, 129, 142, 143)
    for mask in range(16):
        clients = [(q, 24 + j + (32 if q >= 142 else 0)) for j, q in enumerate(qs) if mask & 1 << j]
        text = "".join(f"Q {q:x} 32 {0x51400000 | s << 10:08x}\n" for q, s in clients)
        for depth in range(5):
            dma = "".join("H dma 0 " + " ".join([f"{i+1:x}"] * lanes) + "\n" for i in range(depth))
            lines = run(text + dma)
            errors = [(int(t[4]), t[5]) for t in map(str.split, lines) if t[:2] == ["T", "diag"]]
            assert errors == [(q, "OTS-E-TIM-001") for q, _ in clients[depth:]], (mask, depth, errors)
            writes = {t[3]: t[6:] for t in map(str.split, lines) if t[:2] == ["T", "srf"]}
            targets = {("W:" if s >= 32 else "E:") + str(s % 32): i + 1
                       for i, (_, s) in enumerate(clients[:depth])}
            assert writes.keys() == targets.keys(), (mask, depth, writes.keys(), targets)
            for stream, value in targets.items():
                assert bytes.fromhex("".join(writes[stream])) == bytes([value]) * lanes


def check_idle_intervals():
    # Writes to unassigned CSR 0x7f0 have no chip effect. The harness marks
    # each write as a mutation and therefore calls tick even on an idle edge.
    snapshots = "R 10\nR 12\nR 13\nR 14\nR 30\nR 31\nR 32\nD 0 0\n"
    def compare(text, horizon, profile=cfg):
        # Equal observation horizons matter if autonomous patrol is enabled.
        end = f"H csr {horizon:x} w 7f0 0\n" + snapshots
        force = "".join(f"H csr {i:x} w 7f0 0\n" for i in range(horizon))
        assert run(text + end, profile=profile) == run(text + end + force, profile=profile), text[:110]

    p = Path(__file__).resolve().parents[2] / "conformance/model" / Path(cfg).name
    vectors = json.loads(p.read_text())["vectors"]

    def image(v):
        ram = "".join("S " + " ".join(f"{int(k):x}" for k in key.split(":")) + " " +
                      bytes.fromhex(val).hex(" ") + "\n" for key, val in v["sram"].items())
        return ram + "".join(f"Q {int(q):x} {us}\n" for q, us in v["text"].items()) + "\n".join(v["host"]) + "\n"

    fetched = image(next(v for v in vectors if v["name"] == "ecc-901"))
    link = image(next(v for v in vectors if any(h.startswith("F link ") for h in v.get("host", []))))
    for text in ["Q 58 51c00000 01800001\n", "Q 58 51c00000 01800001\nH db 7 0\n",
                 "Q 58 00400000 52000001\nQ 59 8 00800000\n",
                 "Q 58 3 01800001 00c00031\n", fetched]:
        compare(text, 64)
    compare(link, 330)
    with tempfile.TemporaryDirectory() as directory:
        patrol = Path(directory) / "patrol.json"
        patrol.write_text(json.dumps({**config, "scrub": {"period": 2}}))
        compare("F sram 5 0 3 0 80\n", 64, str(patrol))



def check_fp8_groups():
    # Every 6-bit VXM opcode, both FP8 formats, with and without the group
    # bit. Other fields are zero (aligned streams, default modes). Execute
    # each bank on one queue so the cycle locates the rejected input exactly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from oracle.isa import ISA
    catalogue = ISA(vlanes=lanes)
    grouped = {0, 3, 6, 9, 10, 15, 16, 17, 18, 19, 20}
    for typ in (11, 12):
        for sg4 in (0, 1):
            legal = grouped if sg4 else grouped | {33, 34}
            units = []
            for opcode in range(64):
                word = (2 << 60 | opcode << 54 | (typ & 3) << 52 | (typ >> 2) << 22 | sg4 << 18)
                assert bool(catalogue.decode64(word)) == (opcode in legal), (opcode, typ, sg4)
                units.extend([word >> 32, word & 0xffffffff])
            text = "Q 58 " + " ".join(f"{u:x}" for u in units) + "\n"
            rejected = {int(t[2]) for t in map(str.split, run(text))
                        if t[:2] == ["T", "diag"] and t[-1] == "OTS-E-DEC-003"}
            assert rejected == set(range(64)) - legal, (typ, sg4, sorted(rejected ^ (set(range(64)) - legal)))


def check_observer():
    text = "Q 58 51c00000 01800001\n"
    for release in (False, True):
        program = text + ("H db 7 0\n" if release else "")
        side = run(program, "--observe")
        assert run(program) == [line for line in side if not line.startswith("O ")]
        assert ("O 0 8 88 dispatch 1 image:1" in side) == release
        assert ("O 0 8 88 SetVL 8 0" in side) == release
    assert run(text) == run(text + "H db 7 0\n")


check_dma()
check_idle_intervals()
check_fp8_groups()
check_observer()
print("reader contract: 80 ingress states, 7 literal intervals, 256 FP8 opcode/group cases and parked/released observer cases passed")
