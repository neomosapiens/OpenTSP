#!/usr/bin/env python3
"""Stop on a patrol IRQ before later effects; compare sparse/dense schedules."""
import json
import pathlib
import subprocess
import sys
import tempfile

sim, config = sys.argv[1:3]
base = json.loads(pathlib.Path(config).read_text())
faults = "F sram 1 0 1 0 5\nF sram 1 0 1 0 9\n"
cases = [
    ("stop", 8, 40, False, None),
    ("stop", 16, 40, False, None),
    ("stop", 8, 80, False, None),
    ("stop", 8, 40, True, None),
    ("stop", 8, 40, False, 8),
    ("stop", 8, None, False, None),
    ("stop", 0, 40, True, None),
    ("continue", 8, 40, True, None),
    ("replay", 8, 40, True, None),
]
with tempfile.TemporaryDirectory(prefix="opentsp-patrol-stop-") as tmp:
    for policy, period, write, two, mask_at in cases:
        cfg = json.loads(json.dumps(base))
        cfg["scrub"] = {"period": period}
        path = pathlib.Path(tmp) / "profile.json"
        path.write_text(json.dumps(cfg))
        text = faults
        if two:
            text += "F sram 1 0 2 0 5\nF sram 1 0 2 0 9\n"
        if write is not None:
            text += f"H csr {write:x} w 16 0\n"
        if mask_at is not None:
            text += f"H csr {mask_at:x} w 16 0\n"
        text += "R 16\nR 13\n"
        dense = text + "".join(f"H csr {t:x} w 7f0 0\n" for t in range(write or 2))
        rows = []
        for data in (text, dense):
            p = subprocess.run([sim, str(path), "--trace", "--on-irq", policy],
                               input=data, text=True, capture_output=True, timeout=30)
            rows.append((p.returncode, p.stdout, p.stderr))
        assert rows[0] == rows[1], (policy, period, write, two, mask_at, rows)
        stopped = policy == "stop" and period > 0 and write is not None and mask_at is None
        if stopped:
            assert rows[0][0] == 5 and "X 016 aa\n" in rows[0][1], rows
            assert "X 013 1\n" in rows[0][1], rows
        elif policy == "stop":
            assert rows[0][0] == 0, rows
print("patrol IRQ stop: 9 sparse/dense case families passed")
