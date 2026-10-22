"""Synthetic install roots only. No test touches the operator's live files."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import install


class Install(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='renda-install-synthetic-')
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.home, self.source, self.output = root / 'home', root / 'source', root / 'output'
        self.source.mkdir()
        self.live = self.home / '.local/bin'
        self.live.mkdir(parents=True)
        for name in install.FILES:
            (self.source / name).write_text("print('synthetic test PASS')\n")
        for name in ('renda-harness', 'renda-checked'):
            (self.live / name).write_text('synthetic previous ' + name)
        self.old = {f: install.sha(self.live / f) for f in ('renda-harness', 'renda-checked')}
        self.manifest()
        self.args = argparse.Namespace(expected_harness=self.old['renda-harness'],
            expected_wrapper=self.old['renda-checked'], manifest_sha=install.sha(self.source / 'release.json'),
            output=str(self.output))
        for p in (patch.object(install, '__file__', str(self.source / 'install.py')),
                  patch.object(install.Path, 'home', return_value=self.home)):
            p.start()
            self.addCleanup(p.stop)

    def manifest(self):
        (self.source / 'release.json').write_text(json.dumps(dict(files={f: install.sha(self.source / f) for f in install.FILES})))

    def execute(self):
        with contextlib.redirect_stdout(io.StringIO()):
            install.install(self.args)

    def unchanged(self):
        self.assertEqual({f: install.sha(self.live / f) for f in self.old}, self.old)

    def test_candidate_hash_change_refused(self):
        (self.source / 'local_queue.py').write_text('synthetic tampering')
        with self.assertRaisesRegex(RuntimeError, 'source changed'):
            self.execute()
        self.unchanged()

    def test_live_input_change_refused(self):
        self.args.expected_harness = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'live input changed'):
            self.execute()
        self.unchanged()

    def test_regression_failure_refuses_install(self):
        (self.source / 'test_local_queue.py').write_text('raise SystemExit(1)\n')
        self.manifest()
        self.args.manifest_sha = install.sha(self.source / 'release.json')
        with self.assertRaisesRegex(RuntimeError, 'release test failed'):
            self.execute()
        self.unchanged()

    def test_installed_release_and_backups_hash_exact(self):
        self.execute()
        receipt = json.loads((self.output / 'install.json').read_text())
        self.assertEqual(receipt['state'], 'INSTALLED_PENDING_CLEAN_START')
        self.assertFalse(receipt['processes_signalled'])
        for f in self.old:
            self.assertTrue((self.live / f).is_symlink())
            self.assertEqual(install.sha(Path(receipt['backup']) / f), self.old[f])
            self.assertEqual(install.sha(self.live / f), install.sha(self.source / f))


if __name__ == '__main__':
    unittest.main(verbosity=2)
