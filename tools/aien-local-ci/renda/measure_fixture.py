#!/usr/bin/env python3
"""Count actual harness transport calls under one fixed SYNTHETIC workload."""
import argparse
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import local_queue
from test_local_queue import wire


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path),
        loader=importlib.machinery.SourceFileLoader(name, str(path)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count(module, denied=False):
    prs = [dict(number=n, title='Synthetic failing head', isDraft=False, headRefName=f'fix/synthetic-{n}',
        headRefOid=f'{n:040x}', baseRefOid='b' * 40, baseRefName='main', state='OPEN',
        mergeStateStatus='BLOCKED', createdAt='2026-10-22T00:00:00Z',
        statusCheckRollup=[dict(context=module.SPARK, state='FAILURE', description='synthetic failure')])
        for n in range(1, 7)]
    calls = []
    def transport(cmd, **kw):
        if isinstance(cmd, list) and cmd[0] == 'gh':
            calls.append(cmd)
            if denied:
                return 98, '', 'synthetic guard98 Retry-After: 900'
            if cmd[1:3] == ['api', 'graphql']:
                return 0, json.dumps(wire(prs)), ''
            if cmd[1:3] == ['pr', 'list']:
                return 0, json.dumps(prs), ''
            if cmd[1] == 'api' and '/statuses?' in cmd[2]:
                return 0, json.dumps([dict(context=module.SPARK, state='failure', description='synthetic failure')]), ''
            raise AssertionError('unexpected remote call: ' + repr(cmd[:3]))
        return 0, '', ''
    with tempfile.TemporaryDirectory(prefix='renda-call-count-synthetic-') as temp:
        with patch.object(module, 'STATE', temp), patch.object(module, 'CLONES', temp + '/missing'), \
             patch.object(module, 'WORK', temp + '/missing'), patch.object(module, 'lane_owned', return_value=True), \
             patch.object(module, 'held', return_value=None), patch.object(module, 'model_id', return_value=None), \
             patch.object(module, 'run', side_effect=transport), \
             patch.object(module, 'load_block', return_value={'open': None, 'solo': [], 'size': 12}):
            for step in range(10):
                with patch.object(local_queue.time, 'time', return_value=100000 + step * 60):
                    result = module.lane_pass('omega')
                    assert not any(result[k] for k in ('merged', 'rebased', 'rerun', 'operator'))
    return len(calls)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True)
    args = ap.parse_args()
    baseline = load('renda_baseline_count', args.baseline)
    candidate = load('renda_candidate_count', Path(__file__).with_name('renda-harness'))
    result = dict(fixture='SYNTHETIC: 6 failing PRs, 10 passes at 60-second intervals',
        baseline_calls=count(baseline), candidate_calls=count(candidate),
        baseline_denied_calls=count(baseline, True), candidate_denied_calls=count(candidate, True),
        remote_writes=0, live_performance='NOT_MEASURED')
    assert result['baseline_calls'] == 70 and result['candidate_calls'] == 2, result
    assert result['baseline_denied_calls'] == 10 and result['candidate_denied_calls'] == 1, result
    print(json.dumps(result, indent=2))
