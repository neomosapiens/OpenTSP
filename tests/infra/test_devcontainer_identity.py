"""A resumable development container must identify its source, image and command."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ContainerIdentity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        (self.root / 'tools').mkdir()
        for name in ('devcontainer_common.sh', 'devcontainer_test.sh'):
            shutil.copyfile(ROOT / 'tools' / name, self.root / 'tools' / name)
        fake = self.root / 'bin'
        fake.mkdir()
        docker = fake / 'docker'
        docker.write_text('''#!/usr/bin/env python3
import json,os,sys
args=sys.argv[1:]
with open(os.environ['DOCKER_CALLS'],'a') as out: out.write(json.dumps(args)+'\\n')
if args[:2]==['image','inspect']:
    print('sha256:current')
elif args[0]=='ps':
    if os.environ.get('EXISTING','1')=='1': print('container-id')
elif args[0]=='inspect':
    template=args[2]
    if 'opentsp.recipes' in template: print(os.environ.get('SAVED_RECIPES','test '))
    elif 'opentsp.source' in template: print(os.environ.get('SAVED_SOURCE',os.environ['SOURCE_ROOT']))
    elif 'opentsp.image' in template: print(os.environ.get('SAVED_IMAGE','sha256:current'))
elif args[0]=='wait': print('0')
''')
        docker.chmod(0o755)
        self.calls = self.root / 'calls.jsonl'
        self.env = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ['PATH'], DOCKER_CALLS=str(self.calls), SOURCE_ROOT=str(self.root), OPENTSP_DEV_PLATFORM='linux/amd64')

    def run_launcher(self, **env):
        return subprocess.run(['bash', str(self.root / 'tools/devcontainer_test.sh'), 'test'], env=dict(self.env, **env), capture_output=True, text=True)

    def test_matching_container_can_resume(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_other_source_cannot_resume(self):
        self.assertNotEqual(self.run_launcher(SAVED_SOURCE='/another/source').returncode, 0)

    def test_other_image_cannot_resume(self):
        self.assertNotEqual(self.run_launcher(SAVED_IMAGE='sha256:old').returncode, 0)

    def test_other_recipe_cannot_resume(self):
        self.assertNotEqual(self.run_launcher(SAVED_RECIPES='lint ').returncode, 0)

    def test_new_container_records_source_and_image(self):
        result = self.run_launcher(EXISTING='0')
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [json.loads(line) for line in self.calls.read_text().splitlines()]
        launch = next(args for args in commands if args[0] == 'run')
        self.assertIn('opentsp.source=' + str(self.root), launch)
        self.assertIn('opentsp.image=sha256:current', launch)
        self.assertIn('sha256:current', launch)


if __name__ == '__main__':
    unittest.main()
