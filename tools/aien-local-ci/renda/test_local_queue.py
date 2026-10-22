"""L522 synthetic fixtures. Git/network/model results here are not production receipts."""
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import threading
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
candidate = os.environ.get('RENDA_CANDIDATE', str(HERE / 'renda-harness'))
spec = importlib.util.spec_from_file_location('renda_l522', candidate,
    loader=importlib.machinery.SourceFileLoader('renda_l522', candidate))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
import local_queue as q


def wire(prs, single=False):
    nodes = []
    for p in prs:
        node = dict(p)
        rollup = node.pop('statusCheckRollup')
        node['commits'] = {'nodes': [{'commit': {'oid': p['headRefOid'], 'statusCheckRollup': {'contexts': {
            'pageInfo': {'hasNextPage': False}, 'nodes': rollup}}}}]}
        nodes.append(node)
    return {'data': {'repository': {'pullRequest': nodes[0]} if single else {'pullRequests': {
        'pageInfo': {'hasNextPage': False, 'endCursor': None}, 'nodes': nodes}}}}


class Offline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='renda-synthetic-')
        self.root = Path(self.tmp.name)
        self.clone = self.root / 'clones' / 'omega'
        self.clone.mkdir(parents=True)
        self.git('init', '-q')
        self.git('config', 'user.email', 'synthetic@example.invalid')
        self.git('config', 'user.name', 'Synthetic fixture')
        self.git('remote', 'add', 'origin', 'https://github.com/aien-dev/omega-mv.git')
        (self.clone / 'sample.txt').write_text('before\n')
        self.git('add', '.')
        self.git('commit', '-qm', 'synthetic base')
        self.base = self.git('rev-parse', 'HEAD').strip()
        self.git('update-ref', 'refs/remotes/origin/main', self.base)
        (self.clone / 'sample.txt').write_text('after local queue\n')
        self.git('commit', '-qam', 'synthetic candidate')
        self.sha = self.git('rev-parse', 'HEAD').strip()
        self.state = self.root / 'state'
        queue = self.state / 'renda-queue'
        queue.mkdir(parents=True)
        self.pr = dict(number=614, title='Synthetic local change', isDraft=False,
            headRefName='fix/synthetic', headRefOid=self.sha, baseRefOid=self.base,
            baseRefName='main', mergeStateStatus='BLOCKED', state='OPEN', statusCheckRollup=[])
        self.doc = dict(schema='renda-local-queue/1', repository='aien-dev/omega-mv',
            observed_at=1000, next_sync=0, failures=0, diagnostic='', entries=[dict(
                key=f'aien-dev/omega-mv#614@{self.sha}', kind='pr', pr=self.pr,
                provenance='synthetic retained gh pr list', remote_observed_at=1000)])
        (queue / 'omega.json').write_text(json.dumps(self.doc))
        self.patches = [patch.object(r, 'STATE', str(self.state)),
            patch.object(r, 'CLONES', str(self.root / 'clones')),
            patch.object(r, 'WORK', str(self.root / 'work')),
            patch.object(r, 'PAUSE', str(self.root / 'MERGE-PAUSE')),
            patch.object(r, 'lane_owned', return_value=True),
            patch.object(r, 'held', return_value=None),
            patch.object(r, 'model_id', return_value='synthetic-model'),
            patch.object(r, 'ask_model', return_value=('{"merge":false,"note":"synthetic refusal"}', 'synthetic-model'))]
        for p in self.patches:
            p.start()
        self.addCleanup(self.tmp.cleanup)
        for p in self.patches:
            self.addCleanup(p.stop)

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.clone), *args], text=True)

    def test_network_denied_still_emits_actual_local_diff_and_plan(self):
        original = r.run
        calls = []
        def denied(cmd, **kw):
            if isinstance(cmd, list) and cmd[0] == 'gh':
                calls.append(cmd)
                return 98, '', 'GitHub guard: reads refused; Retry-After: 900'
            return original(cmd, **kw)
        with patch.object(r, 'run', side_effect=denied):
            result = r.lane_pass('omega')
        self.assertTrue(result.get('local'), result)
        artifacts = list((self.state / 'renda-queue').rglob('*.diff'))
        self.assertTrue(artifacts, result)
        self.assertIn('+after local queue', artifacts[0].read_text())
        plans = list((self.state / 'renda-queue').rglob('*.plan.json'))
        self.assertTrue(plans, result)
        plan = json.loads(plans[0].read_text())
        self.assertEqual(plan['commit'], self.sha)
        self.assertEqual(plan['remote_eligible'], False)
        self.assertTrue(result.get('sync_wait'))
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['merged'], [])
        self.assertEqual(result['operator'], [])
        self.assertEqual(result['rebased'], [])

    def queue(self, run=None):
        return q.Queue(str(self.state), 'omega', 'aien-dev/omega-mv', run or r.run,
                       str(self.root / 'clones'), str(self.root / 'work'))

    def test_guard_backoff_shared_across_instances(self):
        with patch.object(r, 'run', return_value=(98, '', 'guard98 Retry-After: 1800')) as transport:
            first = self.queue(r.run)
            d, synced = first.sync(now=2000)
            self.assertFalse(synced)
            self.assertIn('guard98', d['diagnostic'])
            self.assertGreaterEqual(d['next_sync'], 3800)
            self.queue(r.run).sync(now=2200)
            self.assertEqual(transport.call_count, 1)
            self.assertEqual(d['entries'][0]['remote_observed_at'], 1000)

    def test_malformed_snapshot_preserves_pins(self):
        for raw in ('', 'null', '{}', '[{}]', '[', json.dumps([dict(self.pr, headRefOid='bad')])):
            with self.subTest(raw=raw):
                qu = self.queue(lambda *a, **k: (0, raw, ''))
                q.save(qu.path, self.doc)
                doc, synced = qu.sync(now=2000)
                self.assertFalse(synced)
                self.assertEqual(doc['entries'][0]['pr'], self.pr)
                self.assertEqual(doc['observed_at'], 1000)

    def test_parallel_sync_one_transport(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        def slow(*a, **kw):
            calls.append(1)
            entered.set()
            self.assertTrue(release.wait(5))
            return 0, json.dumps(wire([self.pr])), ''
        first, second = self.queue(slow), self.queue(slow)
        thread = threading.Thread(target=lambda: first.sync(now=2000))
        thread.start()
        self.assertTrue(entered.wait(5))
        try:
            self.assertFalse(second.sync(now=2000)[1])
        finally:
            release.set()
            thread.join(5)
        self.assertEqual(len(calls), 1)

    def test_empty_remote_snapshot_deactivates_closed_pr(self):
        qu = self.queue(lambda *a, **kw: (0, json.dumps(wire([])), ''))
        doc, synced = qu.sync(now=2000)
        self.assertTrue(synced)
        self.assertFalse(doc['entries'][0]['active'])
        with patch.object(qu, 'source') as source:
            self.assertEqual(qu.progress(doc, None, None, 'synthetic-policy'), [])
            source.assert_not_called()

    def test_head_move_retains_old_failure_and_invalidates_review(self):
        qu = self.queue()
        with patch.object(qu, 'host_plan', return_value={'code': 2}), patch.object(r, 'ask_model') as ask:
            ask.return_value = ('{"merge":false,"note":"synthetic failed review"}', 'synthetic-model')
            qu.progress(self.doc, 'synthetic-model', ask, 'policy')
            qu.progress(self.doc, 'synthetic-model', ask, 'policy')
            self.assertEqual(ask.call_count, 1)
            (self.clone / 'sample.txt').write_text('changed head\n')
            self.git('commit', '-qam', 'synthetic moved head')
            new = dict(self.pr, headRefOid=self.git('rev-parse', 'HEAD').strip())
            original = qu.run
            qu.run = lambda cmd, **kw: (0, json.dumps(wire([new])), '') if cmd[0] == 'gh' else original(cmd, **kw)
            doc, _ = qu.sync(now=2000)
            self.assertEqual(len(doc['entries']), 2)
            qu.progress(doc, 'synthetic-model', ask, 'policy')
            self.assertEqual(ask.call_count, 2)
            self.assertEqual(qu.review(self.pr, 'synthetic-model', 'policy')[0], False)
            self.assertEqual(qu.review(new, 'synthetic-model', 'policy')[0], False)

    def test_model_and_policy_identity_invalidate_review(self):
        qu = self.queue()
        with patch.object(qu, 'host_plan', return_value={'code': 2}), patch.object(r, 'ask_model') as ask:
            ask.return_value = ('{"merge":false,"note":"synthetic failed review"}', 'synthetic-model')
            for model, policy in [('m1', 'p1'), ('m2', 'p1'), ('m2', 'p2')]:
                qu.progress(self.doc, model, ask, policy)
            self.assertEqual(ask.call_count, 3)
            self.assertEqual(qu.review(self.pr, 'm2', 'p2')[0], False)

    def test_corrupt_wrong_repo_wrong_sha_receipts_never_verify(self):
        qu = self.queue()
        bundle = self.state / 'bundles/omega' / self.sha / 'bundle.json'
        bundle.parent.mkdir(parents=True)
        good = dict(schema='aien-verification-bundle/1', repository='github.com/aien-dev/omega-mv', commit=self.sha)
        for content in ('{', 'null', json.dumps(dict(good, repository='github.com/wrong/repo')),
                        json.dumps(dict(good, commit='a' * 40))):
            with self.subTest(content=content):
                bundle.write_text(content)
                with patch.object(qu, 'run') as run:
                    receipt = qu.receipt(self.clone, 'github.com/aien-dev/omega-mv', self.sha, self.base)
                    self.assertFalse(receipt['verified'])
                    self.assertFalse(receipt['eligible'])
                    run.assert_not_called()

    def test_verified_pass_with_not_run_plan_is_ineligible(self):
        qu = self.queue()
        bundle = self.state / 'bundles/omega' / self.sha / 'bundle.json'
        bundle.parent.mkdir(parents=True)
        for plan in (None, {'verdict': 'NOT_RUN'}, {'verdict': 'PASS', 'unmapped': ['sample.txt']},
                     {'verdict': 'PASS', 'blocked': 'PIPELINE_CONFIG_CHANGED'}):
            with self.subTest(plan=plan):
                q.save(bundle, dict(schema='aien-verification-bundle/1', repository='github.com/aien-dev/omega-mv',
                    commit=self.sha, verdict='PASS', plan=plan))
                def synthetic(cmd, **kw):
                    return (0, 'a' * 64, '') if cmd[0] == 'git' else (0, 'synthetic verifier', '')
                with patch.object(qu, 'run', side_effect=synthetic):
                    result = qu.receipt(self.clone, 'github.com/aien-dev/omega-mv', self.sha, self.base)
                self.assertTrue(result['verified'])
                self.assertFalse(result['eligible'])

    def test_interrupted_atomic_write_keeps_old_bytes(self):
        path = self.root / 'atomic.json'
        q.save(path, {'old': True})
        with patch.object(q.os, 'replace', side_effect=OSError('synthetic interruption')):
            with self.assertRaises(OSError):
                q.save(path, {'new': True})
        self.assertEqual(q.read(path), {'old': True})
        self.assertEqual(list(path.parent.glob('.atomic.json.*')), [])

    def test_wrong_repository_cache_is_ignored(self):
        qu = self.queue()
        q.save(qu.path, dict(self.doc, repository='wrong/repo'))
        self.assertEqual(qu.load()['entries'], [])
        q.atomic(qu.path, b'{')
        self.assertEqual(qu.load()['entries'], [])

    def test_unavailable_source_never_fetches(self):
        qu = self.queue()
        doc = json.loads(json.dumps(self.doc))
        doc['entries'][0]['pr']['headRefOid'] = 'a' * 40
        original = qu.run
        commands = []
        def record(cmd, **kw):
            commands.append(cmd)
            return original(cmd, **kw)
        qu.run = record
        result = qu.progress(doc, None, None, 'synthetic-policy')
        self.assertIn('source unavailable', result[0])
        self.assertTrue(all(c[0] == 'git' and 'fetch' not in c for c in commands))

    def test_local_ready_has_no_remote_pr_identity(self):
        qu = self.queue()
        ready = self.root / 'synthetic-ready.json'
        q.save(ready, [dict(repository=qu.full, sha=self.sha, base=self.base, ref='refs/heads/fix/synthetic')])
        qu.seed_ready(ready)
        entry = next(e for e in qu.load()['entries'] if e['kind'] == 'ready')
        self.assertIsNone(entry['number'])
        self.assertIsNone(entry['remote_observed_at'])
        self.assertNotIn('pr', entry)
        self.assertIn('sha256:', entry['provenance'])
        with patch.object(qu, 'host_plan', return_value={'code': 2}):
            self.assertTrue(qu.progress(dict(entries=[entry]), None, None, 'synthetic'))

    def test_remote_admission_fails_on_moved_closed_or_missing(self):
        for fresh in (None, {}, dict(self.pr, headRefOid='a' * 40),
                      dict(self.pr, baseRefOid='b' * 40), dict(self.pr, state='CLOSED')):
            with self.subTest(fresh=fresh), patch.object(r, 'gh_json', return_value=wire([fresh], single=True) if fresh else fresh), patch.object(r, 'run') as remote:
                self.assertFalse(r.remote_admitted('omega', 'aien-dev/omega-mv', self.pr, merge=True)[0])
                remote.assert_not_called()

    def test_required_gate_failure_cannot_publish(self):
        fresh = dict(self.pr, mergeStateStatus='CLEAN', statusCheckRollup=[
            dict(context=r.SPARK, state='SUCCESS'), dict(name=r.HOSTED, status='COMPLETED', conclusion='FAILURE')])
        with patch.object(r, 'gh_json', side_effect=[wire([fresh], single=True), {'contexts': [r.HOSTED]}]):
            admitted, note = r.remote_admitted('omega', 'aien-dev/omega-mv', self.pr, merge=True)
        self.assertFalse(admitted)
        self.assertIn('not successful', note)

    def test_retry_after_http_date(self):
        delay = q.retry_delay(1, 'Retry-After: Thu, 01 Jan 1970 01:00:00 GMT', 1, 1000)
        self.assertEqual(delay, 2600)


if __name__ == '__main__':
    unittest.main(verbosity=2)
