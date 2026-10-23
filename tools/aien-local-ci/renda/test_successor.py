"""Explicit successor intake must never create a competing PR."""
import copy
import json
import unittest
from unittest.mock import patch
import test_delivery as fixtures
h = fixtures.h


class Successor(unittest.TestCase):
    git = fixtures.Delivery.git
    publish = fixtures.Delivery.publish
    writes = fixtures.Delivery.writes

    def setUp(self):
        fixtures.Delivery.setUp(self)
        self.target = dict(number=42, ref='fix/original', expected_head=self.base)
        self.target_head = self.base
        self.target_state = 'open'
        self.pr_base = self.base
        self.fork = False
        self.pushes=[]
        self.obscure_readback=False
        self._seed()

    def _seed(self):
        ready=self.root/'successor.json'
        ready.write_text(json.dumps([dict(repository=self.q.full,ref='feat/test',
            sha=self.head,base=self.base,target=self.target)]))
        self.ack=self.q.seed_ready(ready)
        self.entry=self.q.load()['entries'][0]

    def transport(self,cmd,**kwargs):
        if cmd[:2]==['gh','api'] and '/pulls/' in cmd[2]:
            self.calls.append(cmd)
            data=dict(number=42,state=self.target_state,draft=False,
                head=dict(ref='fix/original',sha=self.target_head,
                    repo=dict(full_name='other/fork' if self.fork else self.q.full)),
                base=dict(ref='main',sha=self.pr_base,repo=dict(full_name=self.q.full)))
            if self.obscure_readback and self.pushes:data['head']['sha']='e'*40
            return 0,json.dumps(data),''
        if cmd[0]=='git' and 'push' in cmd:
            self.calls.append(cmd);self.pushes.append(cmd)
            self.target_head=self.head
            self.pr_base=self.base
            return 0,'',''
        if cmd[0]=='git' and 'ls-remote' in cmd and cmd[-1]=='refs/heads/fix/original':
            self.calls.append(cmd)
            return 0,self.target_head+'\trefs/heads/fix/original\n',''
        return fixtures.Delivery.transport(self,cmd,**kwargs)

    def test_explicit_old_pr_base_can_advance_to_qualified_current_main(self):
        self.target['expected_base']='c'*40
        self.pr_base='c'*40
        try:
            self._seed()
        except ValueError as exc:
            self.fail('explicit old PR base lease rejected: '+str(exc))
        ok,note=self.publish();self.assertTrue(ok,note)
        self.assertEqual(len(self.pushes),1)
        self.assertEqual(self.writes(),[])
        ok,note=self.publish();self.assertTrue(ok,note)
        self.assertEqual(len(self.pushes),1)

    def test_unexpected_pr_base_change_cannot_advance(self):
        self.pr_base='d'*40
        ok,note=self.publish();self.assertFalse(ok,note)
        self.assertEqual(self.pushes,[])
        self.assertEqual(self.writes(),[])

    def test_successor_target_is_retained_in_trusted_intake(self):
        self.assertEqual(self.entry.get('target'),self.target)

    def test_successor_updates_existing_with_exact_lease_and_reads_back(self):
        ok,note=self.publish()
        self.assertTrue(ok,note)
        self.assertEqual(self.writes(),[], 'must never open competing PR')
        self.assertEqual(len(self.pushes),1)
        self.assertIn('--force-with-lease=refs/heads/fix/original:'+self.base,self.pushes[0])
        self.assertIn(self.head+':refs/heads/fix/original',self.pushes[0])
        ok,note=self.publish()
        self.assertTrue(ok,note)
        self.assertEqual(len(self.pushes),1)
        self.assertIn('42',note)

    def test_successor_intake_target_tampering_cannot_authorize_push(self):
        self.entry['target']=dict(self.target,number=43)
        ok,note=self.publish()
        self.assertFalse(ok,note)
        self.assertEqual(self.pushes,[])
        self.assertEqual(self.writes(),[])

    def test_successor_moved_closed_forked_targets_never_write(self):
        for field,value in [('target_head','e'*40),('target_state','closed'),('fork',True)]:
            with self.subTest(field=field):
                original=getattr(self,field);setattr(self,field,value)
                ok,note=self.publish();self.assertFalse(ok,note)
                setattr(self,field,original)
                self.assertEqual(self.pushes,[])
                self.assertEqual(self.writes(),[])

    def test_successor_readback_failure_is_not_published(self):
        self.obscure_readback=True
        ok,note=self.publish();self.assertFalse(ok,note)
        self.assertEqual(len(self.pushes),1)
        self.assertFalse((self.q.root/'published'/self.q.repo).exists())

    def test_successor_failed_gate_blocks_promotion(self):
        with patch.object(h,'operator_verdict',return_value=('FAIL','actual failure')):
            ok,note=self.publish();self.assertFalse(ok,note)
        self.assertEqual(self.pushes,[])
        self.assertEqual(self.writes(),[])


if __name__=='__main__':unittest.main(verbosity=2)
