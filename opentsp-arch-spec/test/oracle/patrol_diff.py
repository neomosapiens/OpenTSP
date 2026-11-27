"""Patrol writeback, arbitration and cancellation against the explicit-check-bit oracle."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import diff_oracle as diff

sim = str(Path(sys.argv[1]).resolve())
cfg = json.loads(Path(sys.argv[2]).read_text())
cfg['scrub']['period'] = 32
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / 'profile.json'
    path.write_text(json.dumps(cfg))
    for end, later, busy in [(1, None, False), (33, None, False), (33, 1, False),
                             (33, 130, False), (33, None, True), (65, None, True)]:
        m = diff.mc.Machine(cfg)
        recs = [(end, diff.mc.CTRL_ID, 1, m.isa.encode('Notify'), 0)]
        if busy:
            recs.append((32 - m.lat('Read')[1], 0, 1, m.isa.encode('Read', addr=5, dst=7, thread=0), 0))
        faults = [('sram', 0, 0, 0, 0, 0)]
        if later is not None:
            faults.append(('sram', 2, 0, 0, 0, later))
        p = SimpleNamespace(recs=recs, units=None, pre={(0, 0): bytes(m.vl)}, faults=faults, dump=[(0, 0)])
        want = diff.oracle_output(m, p)
        text = '\n'.join(diff.tsp_as.q_lines(diff.images_of(recs))) + '\nD 0 0\n'
        text += ''.join('F sram %x %x %x %x %x\n' % f[1:] for f in faults)
        result = subprocess.run([sim, str(path), '--trace'], input=text, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        got = result.stdout.splitlines()
        assert got == want, (end, later, busy, got, want)
        print('PASS patrol oracle end=%d later=%s busy=%s' % (end, later, busy))
