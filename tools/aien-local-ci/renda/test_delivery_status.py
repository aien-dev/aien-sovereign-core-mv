"""Receipt-to-status delivery uses synthetic transport only; no GitHub writes."""
import importlib.machinery
import importlib.util
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name('renda-harness')
spec = importlib.util.spec_from_file_location('renda_status_delivery_test', PATH,
    loader=importlib.machinery.SourceFileLoader('renda_status_delivery_test', str(PATH)))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


class QueueFixture:
    full = 'aien-dev/interplane-mv'
    repo = 'interplane'

    def __init__(self):
        self.head = 'a' * 40
        self.base = 'b' * 40
        self.review_result = (True, 'complete current review')
        self.plan = {'code': 0, 'output': 'AIEN PLAN\nRepository: github.com/aien-dev/interplane-mv\nBase: ' + self.base + '\nHead: ' + self.head + '\nChanged files: 77\nRequired:\n  ARCH-CRUMB-TOOL-TEST pool=host exactness=EXACT check=4da84606bddf\n  ARCH-DOCTRINE pool=host exactness=EXACT check=fc7a28a7ec75\n  ARCH-NAME-CHECK pool=host exactness=EXACT check=68b21c49aeb8\n  ARCH-QUALIFICATION-STATUS pool=host exactness=EXACT check=6912e608b9f6\n  ARCH-SCRIPT-TESTS pool=host exactness=EXACT check=f8eb95b183d7\n  ARCH-SPARK-BROKER pool=host exactness=EXACT check=b4bf6c95f403\n  CRUMB-VERIFY pool=host exactness=EXACT check=e9f8718de39c\nCached:\n  (none)\nNeeds Spark:\n  (none)\nNeeds GB10:\n  (none)\nAdvisory:\n  (none)\nBlocked:\n  (none)\nUNMAPPED:\n  (none)\nPlan verdict: RUNNABLE\n'}
        self.source_name = 'github.com/' + self.full
        self.receipt_result = {'verified': True, 'eligible': True, 'verdict': 'PASS',
                        'digest': 'e' * 64, 'verifier_codes': [0]}

    def source(self, sha, base):
        if (sha, base) == (self.head, self.base):
            return Path('/trusted/source'), self.source_name
        return None, None

    def review(self, pr, model, policy):
        return self.review_result

    def host_plan(self, source, repository, sha, base, *, operator=False):
        self.plan_modes = getattr(self, 'plan_modes', []) + [operator]
        return self.plan

    def receipt(self, source, repository, sha, base):
        return self.receipt_result

    def sync(self):
        return self.doc, True

    def progress(self, *args, **kwargs):
        return []

    @contextmanager
    def lock(self, suffix='publish'):
        yield True


class ReceiptStatusDelivery(unittest.TestCase):
    def setUp(self):
        self.q = QueueFixture()
        self.pr = dict(number=48, title='qualified exact head', isDraft=False,
            headRefName='fix/receipt', headRefOid=self.q.head,
            baseRefName='main', baseRefOid=self.q.base, state='OPEN',
            mergeStateStatus='BLOCKED', statusCheckRollup=[])
        self.statuses = []
        self.writes = []
        self.fresh = dict(self.pr)
        self.ctx = dict(lane_owned=lambda repo: True, held=lambda repo, number: None,
            SPARK=h.SPARK,
            PAUSE=str(Path(tempfile.gettempdir()) / 'L704-status-test-no-pause'),
            fresh_pr=lambda full, number: self.fresh,
            get_statuses=lambda full, sha: list(self.statuses),
            write_status=self.write_status,
            operator_verdict=lambda repo, sha: None, judging=lambda sha: False)

    def write_status(self, full, sha, state, context, description):
        self.writes.append((full, sha, state, context, description))
        row = dict(context=context, state=state, description=description, sha=sha)
        self.statuses.insert(0, row)
        return True, 'synthetic accepted'

    def deliver(self):
        fn = getattr(h, 'deliver_receipt_status', None)
        self.assertTrue(callable(fn), 'receipt-to-status delivery function missing')
        return fn('interplane', self.q, self.pr, 'synthetic-model', 'synthetic-policy', self.ctx)

    def test_rc0_inconsistent_full_plan_refuses_without_status_write(self):
        self.q.plan = {'code': 0, 'output': 'AIEN PLAN\nRepository: github.com/aien-dev/interplane-mv\nBase: ' + self.q.base + '\nHead: ' + self.q.head + '\nChanged files: 77\nRequired:\n  ARCH-CRUMB-TOOL-TEST pool=host exactness=EXACT check=4da84606bddf\n  ARCH-DOCTRINE pool=host exactness=EXACT check=fc7a28a7ec75\n  ARCH-NAME-CHECK pool=host exactness=EXACT check=68b21c49aeb8\n  ARCH-QUALIFICATION-STATUS pool=host exactness=EXACT check=6912e608b9f6\n  ARCH-SCRIPT-TESTS pool=host exactness=EXACT check=f8eb95b183d7\n  ARCH-SPARK-BROKER pool=host exactness=EXACT check=b4bf6c95f403\n  CRUMB-VERIFY pool=host exactness=EXACT check=e9f8718de39c\nCached:\n  (none)\nNeeds Spark:\n  (none)\nNeeds GB10:\n  (none)\nAdvisory:\n  (none)\nBlocked:\n  (none)\nUNMAPPED:\n  hidden/source.c\nPlan verdict: RUNNABLE\n'}
        ok, note = self.deliver()
        self.assertFalse(ok, note)
        self.assertEqual(self.writes, [])

    def test_signed_qualified_status_uses_authorized_operator_plan_context(self):
        ok, note = self.deliver()
        self.assertTrue(ok, note)
        self.assertEqual(self.q.plan_modes, [True])

    def test_qualified_exact_open_pr_posts_success_and_reads_it_back(self):
        ok, note = self.deliver()
        self.assertTrue(ok, note)
        self.assertEqual(self.writes, [('aien-dev/interplane-mv', self.q.head, 'success',
            h.SPARK, 'Verified signed Spark receipt for exact source head')])
        self.assertIn('read back', note)

    def test_trusted_legacy_repository_source_identity_is_preserved(self):
        self.q.source_name='github.com/aien-dev/interplane'
        ok,note=self.deliver()
        self.assertTrue(ok,note)
        self.assertEqual(self.writes[0][0],'aien-dev/interplane-mv')

    def test_existing_success_is_idempotent_without_duplicate_write(self):
        self.statuses.append(dict(context=h.SPARK, state='success', description='already delivered', sha=self.q.head))
        ok, note = self.deliver()
        self.assertTrue(ok, note)
        self.assertIn('already present', note)
        self.assertEqual(self.writes, [])

    def test_existing_success_does_not_hide_a_retained_local_failure(self):
        self.statuses.append(dict(context=h.SPARK,state='success',description='older success',sha=self.q.head))
        self.ctx['operator_verdict']=lambda repo,sha:('FAIL','new local failure')
        ok,note=self.deliver()
        self.assertFalse(ok,note)
        self.assertEqual(self.writes,[])

    def test_existing_spark_state_is_never_overwritten(self):
        for state in ('failure', 'error', 'pending', 'queued', 'unexpected'):
            with self.subTest(state=state):
                self.statuses[:] = [dict(context=h.SPARK, state=state, description='existing', sha=self.q.head)]
                self.writes.clear()
                ok, note = self.deliver()
                self.assertFalse(ok, note)
                self.assertEqual(self.writes, [])

    def test_unexpected_check_run_named_spark_is_never_replaced(self):
        self.fresh = dict(self.pr, statusCheckRollup=[dict(name=h.SPARK,status='COMPLETED',conclusion='SUCCESS')])
        ok, note = self.deliver()
        self.assertFalse(ok, note)
        self.assertEqual(self.writes, [])

    def test_lane_pass_routes_missing_spark_status_to_receipt_delivery(self):
        doc=dict(entries=[dict(kind='pr',pr=dict(self.pr,mergeStateStatus='UNKNOWN'),active=True,
            key=f'{self.q.full}#48@{self.q.head}',provenance='synthetic',remote_observed_at=1)])
        self.q.doc=doc
        with patch.object(h,'Queue',return_value=self.q), patch.object(h,'policy_hash',return_value='synthetic-policy'), \
                patch.object(h,'model_id',return_value='synthetic-model'), patch.object(h,'lane_owned',return_value=True), patch.object(h,'held',return_value=None), patch.object(h,'load_block',return_value={'open':None,'solo':[48],'size':12}), \
                patch.object(h,'deliver_receipt_status',return_value=(True,'exact status read back')) as deliver:
            result=h.lane_pass('interplane')
        deliver.assert_called_once()
        self.assertEqual(result['status'],['exact status read back'])

    def test_idempotent_success_is_reported_as_status_not_waiting(self):
        doc=dict(entries=[dict(kind='pr',pr=dict(self.pr,mergeStateStatus='UNKNOWN'),active=True,
            key=f'{self.q.full}#48@{self.q.head}',provenance='synthetic',remote_observed_at=1)])
        self.q.doc=doc
        note='exact Spark success already present on exact head'
        with patch.object(h,'Queue',return_value=self.q), patch.object(h,'policy_hash',return_value='synthetic-policy'), \
                patch.object(h,'model_id',return_value='synthetic-model'), patch.object(h,'lane_owned',return_value=True), \
                patch.object(h,'held',return_value=None), patch.object(h,'queued_status',return_value=('none','')), \
                patch.object(h,'load_block',return_value={'open':None,'solo':[48],'size':12}), \
                patch.object(h,'deliver_receipt_status',return_value=(True,note)):
            result=h.lane_pass('interplane')
        self.assertEqual(result['status'],[note])
        self.assertNotIn(note,result['waiting'])

    def test_refuses_changed_pr_identity_and_repository(self):
        for field, value in (('headRefOid', 'c' * 40), ('baseRefOid', 'd' * 40),
                             ('state', 'CLOSED'), ('isDraft', True), ('baseRefName', 'develop')):
            with self.subTest(field=field):
                self.fresh = dict(self.pr, **{field: value})
                ok, _ = self.deliver()
                self.assertFalse(ok)
                self.assertEqual(self.writes, [])
                self.fresh = dict(self.pr)
        self.q.full = 'aien-dev/wrong-mv'
        ok, _ = self.deliver()
        self.assertFalse(ok)
        self.assertEqual(self.writes, [])

    def test_active_judge_for_exact_head_defers_status_delivery(self):
        self.ctx['judging']=lambda sha:True
        ok,note=self.deliver()
        self.assertFalse(ok,note)
        self.assertEqual(self.writes,[])

    def test_refuses_hold_pause_and_local_fail_or_hold(self):
        self.ctx['held'] = lambda repo, number: 'held'
        self.assertFalse(self.deliver()[0])
        self.ctx['held'] = lambda repo, number: None
        pause = Path(self.ctx['PAUSE']); pause.touch()
        try:
            self.assertFalse(self.deliver()[0])
        finally:
            pause.unlink()
        for verdict in (('FAIL', 'gate failed'), ('HOLD', 'not qualified')):
            self.ctx['operator_verdict'] = lambda repo, sha, v=verdict: v
            self.assertFalse(self.deliver()[0])
        self.assertEqual(self.writes, [])

    def test_null_or_malformed_remote_reads_fail_closed(self):
        self.ctx['fresh_pr']=lambda full,number:None
        ok,_=self.deliver()
        self.assertFalse(ok)
        self.ctx['fresh_pr']=lambda full,number:self.fresh
        self.ctx['get_statuses']=lambda full,sha:None
        ok,_=self.deliver()
        self.assertFalse(ok)
        self.assertEqual(self.writes,[])

    def test_refuses_bad_review_receipt_or_current_plan(self):
        self.q.review_result = (False, 'review refused')
        self.assertFalse(self.deliver()[0])
        self.q.review_result = (True, 'review current')
        self.q.receipt_result['eligible'] = False
        self.assertFalse(self.deliver()[0])
        self.q.receipt_result['eligible'] = True
        self.q.receipt_result['verified'] = False
        self.assertFalse(self.deliver()[0])
        self.q.receipt_result['verified'] = True
        self.q.plan = {'code': 1, 'output': 'Plan verdict: BLOCKED\n'}
        self.assertFalse(self.deliver()[0])
        self.assertEqual(self.writes, [])

    def test_malformed_or_noninteger_plan_refuses_without_status_write(self):
        good = dict(self.q.plan)
        for plan in (None, [], {'code': True, 'output': good['output']},
                     {'code': 0, 'output': None}, {'code': '0', 'output': good['output']}):
            with self.subTest(plan=plan):
                self.q.plan = plan
                self.writes.clear()
                ok, note = self.deliver()
                self.assertFalse(ok, note)
                self.assertEqual(self.writes, [])
        self.q.plan = good

    def test_cached_default_operator_hold_cannot_publish_status_without_signed_qualification(self):
        self.q.plan = {'code': 4, 'output': 'Plan verdict: BLOCKED_OPERATOR PIPELINE_CONFIG_CHANGED\n'}
        self.q.receipt_result = {'verified': True, 'eligible': False, 'verdict': 'HOLD', 'digest': 'e' * 64}
        ok, note = self.deliver()
        self.assertFalse(ok, note)
        self.assertEqual(self.writes, [])
        self.assertEqual(getattr(self.q, 'plan_modes', []), [])

    def test_changed_signed_receipt_before_write_is_refused(self):
        calls=[]
        def changed(source, repository, sha, base):
            calls.append(1)
            result=dict(self.q.receipt_result)
            if len(calls) > 1:
                result['digest']='f'*64
            return result
        self.q.receipt=changed
        ok,note=self.deliver()
        self.assertFalse(ok,note)
        self.assertEqual(self.writes,[])

    def test_failed_write_or_missing_readback_never_reports_success(self):
        self.ctx['write_status'] = lambda *args: (False, 'refused')
        ok, _ = self.deliver()
        self.assertFalse(ok)
        self.ctx['write_status'] = lambda *args: (True, 'accepted')
        self.ctx['get_statuses'] = lambda full, sha: []
        ok, note = self.deliver()
        self.assertFalse(ok, note)
        self.assertIn('readback', note.lower())


class GitHubStatusTransport(unittest.TestCase):
    """REST schema captured from GitHub: status rows have url, not sha."""
    full = 'aien-dev/interplane-mv'
    sha = 'a' * 40

    def row(self, **changes):
        row = dict(id=123, context=h.SPARK, state='success', description='verified',
            url=f'https://api.github.com/repos/{self.full}/statuses/{self.sha}')
        row.update(changes)
        return row

    def test_actual_get_schema_binds_exact_url_before_normalizing_sha(self):
        import json
        row = self.row()
        self.assertNotIn('sha', row)
        with patch.object(h, 'run', return_value=(0, json.dumps([row]), '')):
            rows = h.exact_status_rows(self.full, self.sha)
        self.assertEqual(rows[0].get('sha'), self.sha)
        self.assertNotIn('sha', row)

    def test_actual_post_schema_accepts_exact_url_context_and_state(self):
        import json
        with patch.object(h, 'run', return_value=(0, json.dumps(self.row()), '')):
            ok, note = h.post_exact_status(self.full, self.sha, 'success', h.SPARK, 'verified')
        self.assertTrue(ok, note)

    def test_foreign_missing_or_conflicting_identity_is_refused(self):
        import json
        changes = [dict(url=None), dict(url='https://evil.example/statuses/' + self.sha),
            dict(url='https://api.github.com/repos/aien-dev/other/statuses/' + self.sha),
            dict(url=f'https://api.github.com/repos/{self.full}/statuses/' + 'b' * 40),
            dict(sha='b' * 40)]
        for change in changes:
            with self.subTest(change=change):
                row = self.row(**change)
                with patch.object(h, 'run', return_value=(0, json.dumps([row]), '')):
                    with self.assertRaises(ValueError):
                        h.exact_status_rows(self.full, self.sha)
                with patch.object(h, 'run', return_value=(0, json.dumps(row), '')):
                    self.assertFalse(h.post_exact_status(self.full, self.sha, 'success', h.SPARK, 'verified')[0])

    def test_post_wrong_context_or_state_is_refused(self):
        import json
        for change in (dict(context='other'), dict(state='failure')):
            with patch.object(h, 'run', return_value=(0, json.dumps(self.row(**change)), '')):
                self.assertFalse(h.post_exact_status(self.full, self.sha, 'success', h.SPARK, 'verified')[0])

    def test_actual_shaped_transport_completes_delivery_and_idempotent_readback(self):
        import json
        case = ReceiptStatusDelivery()
        case.setUp()
        rows, posts = [], []
        def transport(args, **kwargs):
            if '--method' in args:
                self.assertEqual(args[args.index('--method') + 1], 'POST')
                posts.append(args)
                rows.append(self.row())
                return 0, json.dumps(rows[-1]), ''
            return 0, json.dumps(rows), ''
        case.ctx['get_statuses'] = h.exact_status_rows
        case.ctx['write_status'] = h.post_exact_status
        with patch.object(h, 'run', side_effect=transport):
            ok, note = case.deliver()
            self.assertTrue(ok, note)
            self.assertIn('read back', note)
            ok, note = case.deliver()
            self.assertTrue(ok, note)
            self.assertIn('already present', note)
        self.assertEqual(len(posts), 1)
        self.assertNotIn('sha', rows[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
