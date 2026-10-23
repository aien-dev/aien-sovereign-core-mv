"""Stalled block retirement: a failure forbids merge, not verified closure."""
import copy
import importlib.machinery
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

p = os.environ['RENDA_CANDIDATE']
spec = importlib.util.spec_from_file_location('recovery_renda', p,
    loader=importlib.machinery.SourceFileLoader('recovery_renda', p))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def pr():
    return dict(number=620, title='desk block: six PRs', isDraft=False,
        headRefName='desk/block-omega', headRefOid='a'*40, baseRefName='main',
        baseRefOid='b'*40, mergeStateStatus='BLOCKED', state='OPEN',
        statusCheckRollup=[dict(context=r.SPARK,state='ERROR',description='NOT_RUN')])


def graphql(p):
    node = copy.deepcopy(p)
    contexts = node.pop('statusCheckRollup')
    node['commits'] = {'nodes':[{'commit':{'oid':node['headRefOid'],
        'statusCheckRollup':{'contexts':{'pageInfo':{'hasNextPage':False},'nodes':contexts}}}}]}
    return {'data':{'repository':{'pullRequest':node}}}


class Recovery(unittest.TestCase):
    def exercise(self, fresh=None, closed=None, failed=True, owned=True, held=None, pause=False):
        pinned = pr()
        fresh = fresh or pinned
        closed = closed or dict(state='CLOSED', headRefOid='a'*40)
        st = {'open':{'pr':620,'branch':'desk/block-omega','sha':'a'*40,
                     'carried':[484,487], 'heads':{'484':'c'*40,'487':'d'*40}},'size':12,'solo':[]}
        original = copy.deepcopy(st)
        result = dict(waiting=[],blocked=[],merged=[])
        def gh(args, **kw):
            if args[:2] == ['api','graphql']:
                return graphql(fresh)
            gh.reads += 1
            return dict(state='OPEN',headRefOid='a'*40,mergeStateStatus='BLOCKED') if gh.reads == 1 else closed
        gh.reads = 0
        writes=[]
        def run(cmd, **kw):
            writes.append(cmd)
            return 0,'',''
        with tempfile.TemporaryDirectory() as t:
            marker=Path(t)/'PAUSE'
            if pause: marker.touch()
            with patch.object(r,'PAUSE',str(marker)),patch.object(r,'gh_json',side_effect=gh), \
                 patch.object(r,'lane_owned',return_value=owned),patch.object(r,'held',return_value=held), \
                 patch.object(r,'operator_verdict',return_value=('FAIL','OMEGA-M19:rc_nonzero') if failed else None), \
                 patch.object(r,'run',side_effect=run),patch.object(r,'save_block') as save:
                r.finish_block('omega','aien-dev/omega-mv',st,{620:pinned},result)
                return st,original,result,writes,save.call_count

    def test_retained_failure_can_retire_failed_block_without_merging(self):
        st,old,result,writes,saves=self.exercise()
        self.assertIsNone(st['open'], result)
        self.assertEqual(len(writes),1)
        self.assertEqual(writes[0][:4],['gh','pr','close','620'])
        self.assertNotIn('--delete-branch',writes[0])
        self.assertEqual(result['merged'],[])
        self.assertEqual(saves,1)

    def test_missing_closed_readback_retains_local_block(self):
        st,old,result,writes,saves=self.exercise(closed={'state':'OPEN','headRefOid':'a'*40})
        self.assertEqual(len(writes),1)
        self.assertEqual(st,old)
        self.assertEqual(saves,0)
        self.assertTrue(result['waiting'])

    def test_refusal_boundaries_never_close_or_mutate(self):
        for options in ({'owned':False},{'held':'HOLD'},{'pause':True},{'failed':False},
                        {'fresh':dict(pr(),headRefOid='e'*40)},
                        {'fresh':dict(pr(),baseRefOid='e'*40)},
                        {'fresh':dict(pr(),state='MERGED')},
                        {'fresh':dict(pr(),isDraft=True)}):
            with self.subTest(options=options):
                st,old,result,writes,saves=self.exercise(**options)
                self.assertEqual(st,old)
                self.assertEqual(writes,[])
                self.assertEqual(saves,0)

    def test_retained_failure_still_forbids_nonclose_actions(self):
        pinned=pr()
        with patch.object(r,'lane_owned',return_value=True),patch.object(r,'held',return_value=None), \
             patch.object(r,'gh_json',return_value=graphql(pinned)), \
             patch.object(r,'operator_verdict',return_value=('FAIL','OMEGA-M19:rc_nonzero')):
            ok,note=r.remote_admitted('omega','aien-dev/omega-mv',pinned)
            self.assertFalse(ok)
            self.assertIn('retained failure',note)


if __name__=='__main__':
    unittest.main(verbosity=2)
