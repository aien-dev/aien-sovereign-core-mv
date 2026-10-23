"""Delivery safety regressions. Transport fixtures are synthetic; git/source and queue are real."""
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from local_queue import Queue, digest, read, save

PATH = Path(__file__).with_name('renda-harness')
spec = importlib.util.spec_from_file_location('renda_delivery_test', PATH,
    loader=importlib.machinery.SourceFileLoader('renda_delivery_test', str(PATH)))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


class Delivery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        auth=patch('review_chunks._key_path',return_value=self.root/'service-private/review.key')
        auth.start();self.addCleanup(auth.stop)
        self.source = self.root / 'clones' / 'interplane'
        self.source.mkdir(parents=True)
        self.git('init', '-q')
        self.git('config', 'user.name', 'Synthetic test')
        self.git('config', 'user.email', 'fixture@example.invalid')
        self.git('remote', 'add', 'origin', 'https://github.com/aien-dev/interplane-mv.git')
        (self.source / 'source.txt').write_text('base\n')
        self.git('add', '.')
        self.git('commit', '-qm', 'base')
        self.base = self.git('rev-parse', 'HEAD').strip()
        (self.source / 'source.txt').write_text('candidate\n')
        self.git('commit', '-qam', 'candidate')
        self.head = self.git('rev-parse', 'HEAD').strip()
        self.calls = []
        self.rows = []
        self.receipt = dict(verified=True, eligible=True, verdict='PASS', digest='e'*64)
        self.q = Queue(self.root / 'state', 'interplane', 'aien-dev/interplane-mv',
            self.transport, self.root / 'clones', self.root / 'work')
        ready = self.root / 'ready.json'
        ready.write_text(json.dumps([dict(repository=self.q.full, ref='feat/test', sha=self.head, base=self.base)]))
        self.ack = self.q.seed_ready(ready)
        self.entry = self.q.load()['entries'][0]
        self.diff = self.git('diff', self.base + '...' + self.head, '--')
        self.planfile = self.q.root / 'local' / self.q.repo / (digest(self.entry['key'].encode()) + '.plan.json')
        self.plan = dict(schema='renda-local-plan/1', key=self.entry['key'],
            repository='github.com/' + self.q.full, commit=self.head, base=self.base,
            model='synthetic-model', policy='synthetic-policy', review=dict(decision=True, complete=True, note='synthetic review'),
            diff_digest=digest(self.diff.encode()), plan=dict(code=0, output='Plan verdict: RUNNABLE\n', gates_executed=False))
        save(self.planfile, self.plan)
        import review_chunks
        review_chunks.review_diff(self.diff,repository=self.plan['repository'],key=self.entry['key'],base=self.base,head=self.head,
            model='synthetic-model',policy='synthetic-policy',ask=lambda prompt:('{"merge":true,"note":"synthetic approved"}','synthetic-model'),
            cache_path=self.planfile.with_name(self.planfile.name.replace('.plan.json','.chunks.json')),budget=2)
        self.pr = dict(number=42, state='OPEN', headRefName='feat/test', headRefOid=self.head,
            baseRefName='main', baseRefOid=self.base, mergeStateStatus='BLOCKED')
        for name, value in [('STATE',str(self.q.state)),('PAUSE',str(self.root/'PAUSE')),('LOGDIR',str(self.root/'logs')),
                ('run',self.transport),('lane_owned',lambda repo: True),('held',lambda *args: None),
                ('operator_verdict',lambda *args: None)]:
            p=patch.object(h,name,value);p.start();self.addCleanup(p.stop)
        p=patch.object(self.q,'receipt',side_effect=lambda *args:self.receipt);p.start();self.addCleanup(p.stop)

    def git(self,*args):
        return subprocess.check_output(['git','-C',str(self.source),*args],text=True)

    def transport(self,cmd,**kwargs):
        self.calls.append(cmd)
        if cmd[:2]==['git','check-ref-format']:
            p=subprocess.run(cmd,capture_output=True,text=True);return p.returncode,p.stdout,p.stderr
        if cmd[0]=='git' and 'ls-remote' in cmd:
            ref=cmd[-1];return 0, (self.base if ref=='refs/heads/main' else self.head)+'\t'+ref+'\n',''
        if cmd[0]=='git' or cmd[0]==str(self.q.state/'crumb-tools'/('a'*40)/'crumb'):
            p=subprocess.run(cmd,capture_output=True,text=True,**kwargs);return p.returncode,p.stdout,p.stderr
        if cmd[:3]==['gh','api','graphql']:
            query=next(arg for arg in cmd if arg.startswith('query='))
            def graphql_pr(pr):
                return dict(number=pr['number'],title='fixture',isDraft=False,headRefName=pr['headRefName'],
                    headRefOid=pr['headRefOid'],baseRefName=pr['baseRefName'],baseRefOid=pr['baseRefOid'],
                    mergeStateStatus=pr.get('mergeStateStatus','CLEAN'),state=pr['state'],createdAt='2026-01-01T00:00:00Z',
                    commits=dict(nodes=[dict(commit=dict(oid=pr['headRefOid'],statusCheckRollup=None))]))
            if 'pullRequest(number:$number)' in query:
                return 0,json.dumps(dict(data=dict(repository=dict(pullRequest=graphql_pr(self.pr))))),''
            result=dict(data=dict(repository=dict(pullRequests=dict(
                nodes=[graphql_pr(pr) for pr in self.rows],pageInfo=dict(hasNextPage=False,endCursor=None)))))
            return 0,json.dumps(result),''
        if cmd[:3]==['gh','pr','list']:return 0,json.dumps(self.rows),''
        if cmd[:3]==['gh','pr','create']:
            self.rows=[dict(self.pr)];return 0,'https://github.com/'+self.q.full+'/pull/42\n',''
        if cmd[:3]==['gh','pr','view']:return 0,json.dumps(self.pr),''
        raise AssertionError('unexpected external action: '+str(cmd))

    def publish(self):
        self.assertTrue(callable(getattr(h,'publish_ready',None)), 'generic READY publication path missing')
        return h.publish_ready(self.q.repo,self.q,self.entry,'synthetic-model','synthetic-policy')

    def writes(self):
        return [c for c in self.calls if c[:3]==['gh','pr','create']]

    def test_small_ready_cannot_publish_from_forged_review_cache(self):
        import review_chunks
        path=self.planfile.with_name(self.planfile.name.replace('.plan.json','.chunks.json'))
        path.unlink(missing_ok=True)
        review_chunks.review_diff(self.diff,repository=self.plan['repository'],key=self.entry['key'],base=self.base,head=self.head,
            model='synthetic-model',policy='synthetic-policy',ask=lambda prompt:self.fail('zero-budget model call'),cache_path=path,budget=0)
        data=json.loads(path.read_text())
        for row in data['chunks']:row['decision']=True
        path.write_text(json.dumps(data))
        self.plan['review']['complete']=True;save(self.planfile,self.plan)
        ok,note=self.publish();self.assertFalse(ok,note)
        self.assertEqual(self.writes(),[])

    def test_complete_large_ready_review_is_accepted_by_publication_boundary(self):
        import review_chunks
        (self.source/'large.txt').write_text('large-source-'+('x'*70000))
        self.git('add','.');self.git('commit','-qm','synthetic large ready')
        self.head=self.git('rev-parse','HEAD').strip()
        ready=self.root/'ready.json';ready.write_text(json.dumps([dict(repository=self.q.full,ref='feat/test',sha=self.head,base=self.base)]))
        self.ack=self.q.seed_ready(ready);self.entry=self.q.load()['entries'][-1]
        self.diff=self.git('diff',self.base+'...'+self.head,'--')
        self.plan=dict(self.plan,key=self.entry['key'],commit=self.head,diff_digest=digest(self.diff.encode()),review=dict(decision=True,complete=True))
        self.planfile=self.q.root/'local'/self.q.repo/(digest(self.entry['key'].encode())+'.plan.json');save(self.planfile,self.plan)
        review_chunks.review_diff(self.diff,repository=self.plan['repository'],key=self.entry['key'],base=self.base,head=self.head,
            model='synthetic-model',policy='synthetic-policy',ask=lambda prompt:('{"merge":true,"note":"synthetic approval"}','synthetic-model'),
            cache_path=self.planfile.with_name(self.planfile.name.replace('.plan.json','.chunks.json')),budget=100)
        self.pr['headRefOid']=self.head
        ok,note=self.publish();self.assertTrue(ok,note)
        self.assertEqual(len(self.writes()),1)

    def test_review_policy_identity_covers_chunk_implementation(self):
        self.assertTrue(callable(getattr(h,'policy_hash',None)), 'complete policy identity helper missing')
        package=self.root/'policy-package';package.mkdir()
        for name in ('renda-harness','local_queue.py','site_policy.py','publication_transport.py','successor.py','delivery.py','review_chunks.py'):
            (package/name).write_text('synthetic '+name)
        with patch.object(h,'SOURCE_FILE',str(package/'renda-harness')):
            before=h.policy_hash();(package/'review_chunks.py').write_text('changed chunk semantics')
            self.assertNotEqual(before,h.policy_hash())

    def test_exact_ready_creates_pr_once_readback_and_does_not_merge(self):
        ok,note=self.publish();self.assertTrue(ok,note)
        ok,note=self.publish();self.assertTrue(ok,note)
        self.assertEqual(len(self.writes()),1)
        self.assertIn('42',note)
        self.assertFalse(any('merge' in c[:3] for c in self.calls))
        pubs=list((self.q.root/'published'/self.q.repo).glob('*.json'))
        self.assertEqual(len(pubs),1)
        self.assertEqual(read(pubs[0])['head'],self.head)

    def test_graphql_lookup_failure_holds_before_pr_create(self):
        with patch('publication_transport.list_by_head', side_effect=ValueError('GraphQL errors')):
            ok, note = self.publish()
        self.assertFalse(ok, note)
        self.assertIn('lookup unavailable', note)
        self.assertEqual(self.writes(), [])
        self.assertFalse((self.q.root/'published'/self.q.repo).exists())

    def test_missing_failed_or_unverified_receipt_never_creates(self):
        for receipt in [dict(verified=False,eligible=False),dict(verified=True,eligible=False,verdict='FAIL'),
                dict(verified=True,eligible=False,verdict='PASS',plan=dict(reason='UNMAPPED_PATH'))]:
            self.receipt=receipt
            ok,note=self.publish();self.assertFalse(ok,note)
        self.assertEqual(self.writes(),[])

    def test_refused_or_stale_review_never_creates(self):
        for field,value in [('review',dict(decision=False)),('diff_digest','f'*64),('model','moved'),('policy','old'),('base','0'*40)]:
            changed=dict(self.plan);changed[field]=value;save(self.planfile,changed)
            ok,note=self.publish();self.assertFalse(ok,note)
        self.assertEqual(self.writes(),[])

    def test_malformed_identity_and_unacknowledged_intake_never_create(self):
        self.entry['key']='wrong';ok,_=self.publish();self.assertFalse(ok)
        self.entry=self.q.load()['entries'][0]
        Path(self.ack['ack_path']).unlink();ok,_=self.publish();self.assertFalse(ok)
        self.assertEqual(self.writes(),[])

    def test_fresh_remote_head_base_and_existing_closed_pr_refuse(self):
        self.rows=[dict(self.pr,state='MERGED')]
        ok,_=self.publish();self.assertFalse(ok)
        self.rows=[]
        self.pr['headRefOid']='0'*40
        ok,note=self.publish();self.assertFalse(ok,note)
        self.assertFalse((self.q.root/'published'/self.q.repo).exists())

    def test_pause_and_lost_lane_prevent_publication(self):
        Path(h.PAUSE).write_text('pause')
        ok,_=self.publish();self.assertFalse(ok)
        Path(h.PAUSE).unlink()
        with patch.object(h,'lane_owned',return_value=False):
            ok,_=self.publish();self.assertFalse(ok)
        self.assertEqual(self.writes(),[])

    def test_retained_failure_after_eligible_snapshot_prevents_write(self):
        with patch.object(h,'operator_verdict',return_value=('FAIL','new failure')):
            ok,_=self.publish();self.assertFalse(ok)
        self.assertEqual(self.writes(),[])

    def test_operator_cheap_admission_rejects_unmapped_or_missing_plan(self):
        self.assertTrue(callable(getattr(h,'admit_plan',None)), 'cheap planner admission missing')
        for plan in [None,{},dict(code=4,output='Plan verdict: NOT_RUN UNMAPPED_PATH\n'),
                dict(code=0,output='UNMAPPED:\n  hidden/source.c\nPlan verdict: RUNNABLE\n')]:
            ok,_=h.admit_plan(plan);self.assertFalse(ok,plan)
        for plan in [dict(code=0,output='UNMAPPED:\n  (none)\nPlan verdict: RUNNABLE\n'),
                dict(code=4,output='UNMAPPED:\n  (none)\nPlan verdict: BLOCKED_OPERATOR PIPELINE_CONFIG_CHANGED\n')]:
            ok,note=h.admit_plan(plan);self.assertTrue(ok,note)

    def test_release_installer_and_startup_include_delivery_regressions(self):
        import install
        self.assertIn('publication_transport.py',install.FILES)
        self.assertIn('test_delivery.py',install.FILES)
        self.assertIn('test_publication_transport.py',install.FILES)
        wrapper=Path(__file__).with_name('renda-checked').read_text()
        self.assertIn('test_delivery.py',wrapper)
        self.assertIn('test_publication_transport.py',wrapper)

    def test_legacy_workflow_pin_is_used_when_repository_has_no_crumb_lock(self):
        (self.source/'.aien').mkdir();(self.source/'.aien/always').write_text('CRUMB-VERIFY\n')
        workflow=self.source/'.github/workflows';workflow.mkdir(parents=True)
        (workflow/'aien-verification.yml').write_text('''steps:
  - name: Checkout the pinned crumb tool source (aien-architecture @ aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)
    uses: actions/checkout@v4
    with:
      repository: aien-spark/aien-architecture
      ref: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
      path: .crumb-tool-src
  - name: Build the crumb tool
    run: make
''')
        self.git('add','.');self.git('commit','-qm','synthetic explicit workflow pin')
        head=self.git('rev-parse','HEAD').strip()
        tool=self.q.state/'crumb-tools'/('a'*40)/'crumb';tool.parent.mkdir(parents=True)
        tool.write_text('#!/bin/sh\nexit 0\n');tool.chmod(0o755)
        ok,note=h.cheap_metadata(self.q,self.source,head,self.base);self.assertTrue(ok,note)

    def test_pinned_crumb_preflight_failure_blocks_admission_without_fallback(self):
        self.assertTrue(callable(getattr(h,'cheap_metadata',None)), 'pinned metadata preflight missing')
        (self.source/'.aien').mkdir()
        (self.source/'.aien/always').write_text('CRUMB-VERIFY\n')
        (self.source/'crumb.lock').write_text('a'*40+'\n')
        self.git('add','.');self.git('commit','-qm','synthetic legacy crumb gate')
        head=self.git('rev-parse','HEAD').strip()
        tool=self.q.state/'crumb-tools'/('a'*40)/'crumb'
        tool.parent.mkdir(parents=True)
        tool.write_text('#!/bin/sh\nexit 7\n');tool.chmod(0o755)
        ok,note=h.cheap_metadata(self.q,self.source,head,self.base)
        self.assertFalse(ok,note)
        tool.write_text('#!/bin/sh\nexit 0\n')
        ok,note=h.cheap_metadata(self.q,self.source,head,self.base)
        self.assertTrue(ok,note)
        self.assertEqual(self.git('rev-parse','HEAD').strip(),head)
        tool.unlink()
        ok,note=h.cheap_metadata(self.q,self.source,head,self.base)
        self.assertFalse(ok,note)
        self.assertFalse(any(c[0]=='crumb' for c in self.calls))

    def test_concurrent_operator_admission_starts_only_one_judge(self):
        import threading
        entered=threading.Event();release=threading.Event();calls=[];results=[]
        def plan(*args):
            calls.append('plan')
            if len(calls)==1:
                entered.set();self.assertTrue(release.wait(3))
            return dict(code=0,output='Plan verdict: RUNNABLE\n')
        with patch.object(h,'Queue',return_value=self.q),patch.object(h,'judging',return_value=False), \
                patch.object(self.q,'source',return_value=(self.source,'github.com/'+self.q.full)), \
                patch.object(self.q,'host_plan',side_effect=plan),patch.object(h,'cheap_metadata',return_value=(True,'ok')), \
                patch.object(h,'run',return_value=(0,'','')),patch.object(h.subprocess,'Popen') as launch:
            launch.return_value.pid=123456
            first=threading.Thread(target=lambda:results.append(h.operator_run(self.q.repo,self.head)))
            first.start();self.assertTrue(entered.wait(3))
            second=h.operator_run(self.q.repo,self.head)
            release.set();first.join(3);self.assertFalse(first.is_alive())
            self.assertFalse(second[0],second)
            self.assertEqual(launch.call_count,1)
            self.assertTrue(results[0][0],results)

    def test_operator_admission_lock_lives_with_child_after_parent_returns(self):
        import sys,time,fcntl
        real_popen=subprocess.Popen
        release=self.root/'release-child';ready=self.root/'child-ready';children=[]
        code=('import os,sys,time;from pathlib import Path;'
              'os.fstat(int(sys.argv[1]));Path(sys.argv[3]).write_text("ready");'
              '\nwhile not Path(sys.argv[2]).exists():time.sleep(0.01)\n')
        def spawn(cmd,**kwargs):
            self.assertEqual(cmd[-5:-3],['aien-local-ci','once'])
            child=real_popen([sys.executable,'-c',code,str(kwargs['pass_fds'][0]),str(release),str(ready)],**kwargs)
            children.append(child);return child
        try:
            with patch.object(h,'Queue',return_value=self.q),patch.object(h,'judging',return_value=False), \
                    patch.object(self.q,'source',return_value=(self.source,'github.com/'+self.q.full)), \
                    patch.object(self.q,'host_plan',return_value=dict(code=0,output='Plan verdict: RUNNABLE\n')), \
                    patch.object(h,'cheap_metadata',return_value=(True,'ok')),patch.object(h,'run',return_value=(0,'','')), \
                    patch.object(h.subprocess,'Popen',side_effect=spawn):
                ok,note=h.operator_run(self.q.repo,self.head);self.assertTrue(ok,note)
                deadline=time.monotonic()+3
                while not ready.exists() and time.monotonic()<deadline:time.sleep(0.01)
                self.assertTrue(ready.exists())
                ok,note=h.operator_run(self.q.repo,self.head);self.assertFalse(ok,note)
                self.assertEqual(len(children),1)
        finally:
            release.write_text('release')
            for child in children:self.assertEqual(child.wait(timeout=3),0)
        with open(self.q.state/'renda-operator-admission/interplane.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_judge_guard_holds_admission_but_does_not_leak_fd_to_gate(self):
        import sys,fcntl,time
        self.assertTrue(callable(getattr(h,'guarded_judge_command',None)), 'judge guardian missing')
        lockpath=self.q.state/'fixture-admission.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
        ready=self.root/'gate-ready';release=self.root/'release-gate'
        code=('import os,sys,time;from pathlib import Path;'
              '\ntry:os.fstat(int(sys.argv[1]));raise SystemExit(91)\nexcept OSError:pass\n'
              'Path(sys.argv[2]).write_text("fd closed");'
              '\nwhile not Path(sys.argv[3]).exists():time.sleep(0.01)\n')
        with open(lockpath,'a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            fd=lock.fileno()
            child=subprocess.Popen(h.guarded_judge_command(fd,[sys.executable,'-c',code,str(fd),str(ready),str(release)]),pass_fds=(fd,))
        try:
            deadline=time.monotonic()+3
            while not ready.exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(0.01)
            self.assertTrue(ready.exists(),'actual gate inherited admission fd or failed')
            with open(lockpath,'a') as other:
                with self.assertRaises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:
            release.write_text('release');self.assertEqual(child.wait(timeout=3),0)
        with open(lockpath,'a') as other:fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_forged_ack_shape_without_original_input_provenance_is_refused(self):
        original=read(Path(self.ack['ack_path']))
        Path(self.ack['ack_path']).unlink()
        forged=dict(original)
        forged.pop('provenance_sha256');forged.pop('input')
        save(self.q.root/'acks/fake.json',forged)
        ok,note=self.publish();self.assertFalse(ok,note)
        self.assertEqual(self.writes(),[])

    def test_malformed_unmapped_and_removed_planner_sections_refuse(self):
        for text in ['UNMAPPED:\nhidden.c\nPlan verdict: RUNNABLE\n',
                'REMOVED:\n  deleted.c\nPlan verdict: RUNNABLE\n',
                'UNMAPPED:\n  (none)\nUNMAPPED:\n  hidden.c\nPlan verdict: RUNNABLE\n']:
            ok,note=h.admit_plan(dict(code=0,output=text));self.assertFalse(ok,(text,note))

    def test_retained_verdict_is_not_reported_as_a_new_operator_action(self):
        with patch.object(h,'operator_verdict',return_value=('FAIL','known failure')):
            ok,note=h.operator_run(self.q.repo,self.head)
            self.assertFalse(ok,'retained verdict counted as newly started judge')
            self.assertIn('not run again',note)

    def test_operator_rejects_cheap_invalid_plan_without_launching(self):
        doc=self.q.load()
        with patch.object(h,'Queue',return_value=self.q),patch.object(h,'judging',return_value=False), \
                patch.object(self.q,'host_plan',return_value=dict(code=4,output='Plan verdict: NOT_RUN UNMAPPED_PATH\n')), \
                patch.object(self.q,'source',return_value=(self.source,'github.com/'+self.q.full)), \
                patch.object(h,'run',side_effect=lambda *a,**k:(0,'','')),patch.object(h.subprocess,'Popen') as launch:
            ok,note=h.operator_run(self.q.repo,self.head)
            self.assertFalse(ok,note)
            launch.assert_not_called()

    def test_generic_lane_routes_eligible_ready_through_publisher(self):
        doc=self.q.load()
        with patch.object(h,'Queue',return_value=self.q),patch.object(self.q,'sync',return_value=(doc,True)), \
                patch.object(self.q,'progress',return_value=[]),patch.object(h,'model_id',return_value='synthetic-model'), \
                patch.object(h,'assessment',return_value=((self.source,'github.com/'+self.q.full,self.plan),'ok')), \
                patch.object(h,'publish_ready',return_value=(True,'published exact READY')) as publish, \
                patch.object(h,'load_block',return_value={}):
            result=h.lane_pass(self.q.repo)
            publish.assert_called_once()
            self.assertEqual(result['published'],['published exact READY'])

    def test_hosted_different_published_bundle_refuses_dispatch(self):
        import base64
        with patch.object(h,'Queue',return_value=self.q),patch.object(h,'gh_json',return_value=dict(workflow_runs=[
                dict(id=123,name=h.HOSTED,conclusion='failure',status='completed',run_attempt=1)])):
            self.q.load=lambda:dict(entries=[dict(kind='pr',pr=dict(headRefOid=self.head,baseRefOid=self.base))])
            def transport(cmd,**kwargs):
                self.calls.append(cmd)
                return 0,json.dumps(dict(encoding='base64',content=base64.b64encode(b'wrong bundle').decode())),''
            with patch.object(h,'run',side_effect=transport):
                ok,note=h.hosted_rerun(self.q.full,self.head,self.q.repo)
                self.assertFalse(ok,note)
            self.assertFalse(any(c[:3]==['gh','run','rerun'] for c in self.calls))

    def test_hosted_same_evidence_and_run_attempt_dispatches_only_once(self):
        import base64
        raw=b'synthetic eligible bundle'
        self.receipt['digest']=digest(raw)
        with patch.object(h,'Queue',return_value=self.q),patch.object(h,'gh_json',return_value=dict(workflow_runs=[
                dict(id=123,name=h.HOSTED,conclusion='failure',status='completed',run_attempt=1)])):
            self.q.load=lambda:dict(entries=[dict(kind='pr',pr=dict(headRefOid=self.head,baseRefOid=self.base))])
            def transport(cmd,**kwargs):
                self.calls.append(cmd)
                return 0,json.dumps(dict(encoding='base64',content=base64.b64encode(raw).decode())),''
            with patch.object(h,'run',side_effect=transport):
                first,note=h.hosted_rerun(self.q.full,self.head,self.q.repo)
                self.assertTrue(first,note)
                second,note=h.hosted_rerun(self.q.full,self.head,self.q.repo)
                self.assertFalse(second,'unchanged hosted failure dispatched twice')
            self.assertEqual(len([c for c in self.calls if c[:3]==['gh','run','rerun']]),1)

    def test_repeated_reports_emit_changes_and_heartbeat_not_whole_queue(self):
        result=dict(repo='interplane',merged=[],rebased=[],rerun=[],operator=[],blocked=['#42 '+self.head[:8]+' failed'],waiting=[])
        emitted=[]
        with patch.object(h,'log',side_effect=emitted.append),patch.object(h,'DESKLOG',str(self.root/'desk.log')):
            h.report([result]);first=list(emitted);emitted.clear()
            h.report([result]);self.assertEqual(emitted,[], 'unchanged blocked queue reprinted')
            result['blocked']=['#42 '+self.head[:8]+' NEW failure']
            h.report([result]);self.assertTrue(any('NEW failure' in line for line in emitted))
        self.assertTrue(first)


if __name__=='__main__':unittest.main()
