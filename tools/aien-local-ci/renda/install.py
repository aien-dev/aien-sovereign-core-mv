#!/usr/bin/env python3
"""Hash-pinned atomic on-disk install. Never signals a process or claims activation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

FILES = ('install.py', 'renda-harness', 'local_queue.py', 'review_chunks.py', 'site_policy.py', 'publication_transport.py', 'successor.py', 'test_successor.py', 'test_delivery_recovery.py', 'test_admission.py', 'test_install.py', 'delivery.py', 'test_publication_transport.py', 'test_delivery.py', 'test_reliability.py', 'test_local_queue.py', 'test_review_chunks.py', 'test_review_integration.py', 'test_site_policy.py', 'renda-checked')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install(args):
    source = Path(__file__).resolve().parent
    manifest = source / 'release.json'
    if sha(manifest) != args.manifest_sha:
        raise RuntimeError('candidate manifest changed, refusing install')
    expected = json.loads(manifest.read_text())
    if set(expected['files']) != set(FILES) or any(sha(source / f) != expected['files'][f] for f in FILES):
        raise RuntimeError('candidate source changed, refusing install')
    home = Path.home()
    state = home / '.local/state/aien-local-ci'
    state.mkdir(parents=True, exist_ok=True)
    with open(state / 'renda-install.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        live = home / '.local/bin'
        inputs = {'renda-harness': args.expected_harness, 'renda-checked': args.expected_wrapper}
        if any(sha(live / f) != value for f, value in inputs.items()):
            raise RuntimeError('live input changed, refusing install')
        release = home / '.local/lib/renda/releases' / ('L603-' + args.manifest_sha[:16])
        release.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            target = release / f
            if target.exists() and sha(target) != expected['files'][f]:
                raise RuntimeError('existing release differs, refusing overwrite')
            if not target.exists():
                shutil.copy2(source / f, target)
            if f in ('renda-harness', 'renda-checked'):
                target.chmod(0o755)
        env = dict(os.environ, RENDA_CANDIDATE=str(release / 'renda-harness'), PYTHONDONTWRITEBYTECODE='1')
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        for test in ('test_reliability.py', 'test_local_queue.py', 'test_review_chunks.py', 'test_review_integration.py', 'test_site_policy.py', 'test_publication_transport.py', 'test_delivery.py', 'test_successor.py', 'test_delivery_recovery.py', 'test_admission.py', 'test_install.py'):
            result = subprocess.run(['python3', str(release / test)], env=env, text=True, capture_output=True)
            (output / ('install-' + test + '.log')).write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError('release test failed: ' + test)
        if any(sha(release / f) != expected['files'][f] for f in FILES):
            raise RuntimeError('release changed during tests')
        if any(sha(live / f) != value for f, value in inputs.items()):
            raise RuntimeError('live input raced during tests, refusing install')
        backup = output / ('backup-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
        backup.mkdir()
        for f in inputs:
            shutil.copy2(live / f, backup / f)
        # Both versions of the wrapper can start the new release. Publish the executable first.
        for f in ('renda-harness', 'renda-checked'):
            if sha(live / f) != inputs[f]:
                raise RuntimeError('live input raced at publication')
            tmp = live / ('.' + f + '.L522-' + str(os.getpid()))
            tmp.symlink_to(release / f)
            os.replace(tmp, live / f)
        directory = os.open(live, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        receipt = dict(schema='renda-install/1', state='INSTALLED_PENDING_CLEAN_START',
            installed_at=time.time(), release=str(release), backup=str(backup),
            manifest_sha=args.manifest_sha, previous=inputs,
            installed={f: sha(live / f) for f in inputs}, source_files=expected['files'],
            processes_signalled=False, active_worker_version='NOT_CHANGED_BY_INSTALLER')
        (output / 'install.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--expected-harness', required=True)
    ap.add_argument('--expected-wrapper', required=True)
    ap.add_argument('--manifest-sha', required=True)
    ap.add_argument('--output', required=True)
    install(ap.parse_args())
