"""Static scheduling reports and standalone HTML for exact assembled text."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from opentsp_tools import refasm, schedule


class ScheduleTest(unittest.TestCase):
    def object(self, directory):
        asm = refasm.load()
        program = asm.assemble("""
.arch opentsp-tsa
.profile mini
.section text
bundle @cycle=3 {
  MEM[0]: read 0x0, E:0, 0x0
}
bundle @cycle=13 {
  MEM[0]: read 0x0, E:0, 0x0
}
bundle @cycle=17 {
  MEM[0]: read 0x0, E:0, 0x0
}
""")[1]
        path = Path(directory) / "three-issues.elf"
        path.write_bytes(asm.write_object(program))
        return path

    def test_cli_reports_exact_closed_interval_and_escapes_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj = self.object(tmp)
            html = Path(tmp) / "schedule.html"
            report = Path(tmp) / "bubbles.json"
            label = "</script><script>alert(1)</script>"
            p = subprocess.run([
                sys.executable, "-m", "opentsp_tools.schedule", "--object", label, str(obj),
                "--html", str(html), "--json", str(report), "--top", "2",
            ], text=True, capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertNotIn(label, html.read_text())
            self.assertIn("OpenTSP Schedule Explorer", html.read_text())
            q, = json.loads(report.read_text())["routes"][0]["stages"][0]["queues"]
            self.assertEqual((q["issue_cycles"], q["span_cycles"], q["gap_cycles"]), (3, 15, 12))
            self.assertEqual([(g["start"], g["end"], g["cycles"]) for g in q["largest"]], [(4, 12, 9), (14, 16, 3)])

    def test_pinned_timing_and_claimed_panel_geometry(self):
        receipt = {"source_sha256": "a" * 64, "period": 19, "cycles": 59, "rows": 4, "boot": 1}
        route = dict(label="Sample", manifest="a" * 64, panel=2)
        self.assertEqual(schedule.timing(receipt, route)["period"], 19)
        with self.assertRaisesRegex(ValueError, "source digest"):
            schedule.timing(dict(receipt, source_sha256="b" * 64), route)
        with self.assertRaisesRegex(ValueError, "panel"):
            schedule.timing(dict(receipt, rows=3), route)


    def test_partial_window_only_counts_gaps_between_visible_issues(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj = self.object(tmp)
            route = schedule.raw_object(refasm.load(), "Window", obj, None, (5, 17))
            stage, = route["chips"]
            q, = schedule.report_stage(stage, 20)["queues"]
            self.assertEqual((q["first"], q["last"], q["gap_cycles"]), (13, 17, 3))

    def test_tampered_package_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            unit = Path(tmp)
            asm = refasm.load()
            elf = self.object(tmp).read_bytes()
            payload = {"program.elf": elf, "bindings.npz": b"bindings"}
            for name, data in payload.items():
                (unit / name).write_bytes(data)
            files = {name: schedule.sha(data) for name, data in payload.items()}
            identity = asm.elf_identity(elf)
            record = dict(chip=0, elf_sha256=identity, files_sha256=files)

            def seal(obj):
                return schedule.sha(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())

            expected = seal(dict(record, files_sha256=dict(files, **{"program.elf": identity})))
            record["metadata_sha256"] = seal(record)
            (unit / "program.json").write_text(json.dumps(record))
            self.assertEqual(schedule.unit_payload(asm, unit, expected, 0), elf)
            (unit / "bindings.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "payload digest mismatch"):
                schedule.unit_payload(asm, unit, expected, 0)



if __name__ == "__main__":
    unittest.main()
