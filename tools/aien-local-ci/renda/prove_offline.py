#!/usr/bin/env python3
"""Exercise authentic retained pins with a transport that refuses every remote operation."""
import argparse
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import shutil
import time
from local_queue import Queue, digest, save


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--retained', required=True)
    ap.add_argument('--state', required=True)
    ap.add_argument('--repo', default='omega')
    ap.add_argument('--production-state', default=str(Path.home() / '.local/state/aien-local-ci'))
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('renda_offline_proof', here / 'renda-harness',
        loader=importlib.machinery.SourceFileLoader('renda_offline_proof', str(here / 'renda-harness')))
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    raw = Path(a.retained).read_bytes()
    retained = json.loads(raw)
    # Preserve the whole captured queue; prioritize the parent's exact heads for this bounded proof.
    selected = [e for e in retained['entries'] if e['kind'] == 'pr' and 614 <= e['pr']['number'] <= 618]
    if len(selected) != 5:
        raise RuntimeError('authentic parent pins 614-618 required')
    state = Path(a.state)
    if state.exists():
        raise RuntimeError('proof state must be new, production caches must not be reset')
    for e in selected:
        sha = e['pr']['headRefOid']
        source = Path(a.production_state) / 'bundles' / a.repo / sha / 'bundle.json'
        if source.exists():
            target = state / 'bundles' / a.repo / sha / 'bundle.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    calls = []
    def offline(cmd, **kw):
        calls.append(cmd)
        if cmd[0] == 'gh':
            return 98, '', 'L522 injected transport: network disabled; Retry-After: 900'
        if cmd[0] == 'git' or (Path(cmd[0]).name == 'aien-test' and cmd[1] in ('plan', 'verify-bundle')):
            return r.run(cmd, **kw)
        raise RuntimeError('offline proof forbids operation: ' + repr(cmd))
    queue = Queue(state, a.repo, retained['repository'], offline, r.CLONES, r.WORK)
    save(queue.path, retained)
    doc, synced = queue.sync()
    assert not synced
    outputs = queue.progress(dict(doc, entries=selected), None, None,
        digest((here / 'renda-harness').read_bytes() + (here / 'local_queue.py').read_bytes()), budget=1)
    plans = list((state / 'renda-queue/local').rglob('*.plan.json'))
    assert len(plans) == 1, outputs
    plan = json.loads(plans[0].read_text())
    assert plan['commit'] == selected[0]['pr']['headRefOid'] and plan['remote_eligible'] is False
    assert plan['plan'].get('gates_executed') is False, plan
    assert plan['plan']['output'], plan
    result = dict(schema='renda-offline-proof/1', evidence='AUTHENTIC_RETAINED_METADATA_AND_LOCAL_SOURCE',
        performed_at=time.time(), retained_sha256=digest(raw), metadata_observed_at=retained['observed_at'],
        retained_count=len(doc['entries']), parent_pins={str(e['pr']['number']): e['pr']['headRefOid'] for e in selected},
        transport='injected refusal for every GitHub request; only local git and existing plan/verify-bundle allowed',
        remote_attempts=sum(cmd[0] == 'gh' for cmd in calls), remote_writes=0, gates_executed=0,
        outputs=outputs, plan_path=str(plans[0]), commit=plan['commit'],
        receipt=plan['receipt'], host_plan=plan['plan'], remote_eligible=False, model_review='NOT_RUN')
    save(state.parent / 'offline-proof.json', result)
    print(json.dumps(result, indent=2))
