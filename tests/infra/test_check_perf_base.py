"""Explicit baseline directories are complete, distinct source inputs without Git."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / 'tools/check_perf_base.py'
NEEDED = ('Cargo.toml','Cargo.lock','opentsp-iss/crates/tsp-iss/Cargo.toml','opentsp-sw-toolchain/crates/ots-asm/Cargo.toml','tools/iss_bench_prep.py','opentsp-vllm/src/opentsp_vllm/models/tinystories/build_programs.py')
class BaselineDirectory(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'candidate'; self.base=Path(self.tmp.name)/'baseline'
        for directory in (self.root,self.base):
            for name in NEEDED+('opentsp-iss/crates/tsp-iss/src/lib.rs',):
                p=directory/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture\n')
        (self.base/'opentsp-iss/crates/tsp-iss/src/lib.rs').write_text('baseline implementation\n')
    def run_check(self, baseline=None):
        return subprocess.run([sys.executable,str(TOOL),'--root',str(self.root),'--base-dir',str(baseline or self.base)],capture_output=True,text=True)
    def test_complete_distinct_directory_passes_without_git(self):
        result=self.run_check();self.assertEqual(result.returncode,0,result.stderr)
    def test_missing_input_fails(self):
        (self.base/'tools/iss_bench_prep.py').unlink();self.assertNotEqual(self.run_check().returncode,0)
    def test_same_directory_fails(self):
        self.assertNotEqual(self.run_check(self.root).returncode,0)
    def test_identical_iss_sources_fail(self):
        shutil.copyfile(self.root/'opentsp-iss/crates/tsp-iss/src/lib.rs',self.base/'opentsp-iss/crates/tsp-iss/src/lib.rs')
        self.assertNotEqual(self.run_check().returncode,0)
if __name__=='__main__':unittest.main()
