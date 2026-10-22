"""Synthetic final-admission controls. No remote operation is executed."""
import copy
import unittest
from unittest.mock import patch
import test_local_queue as fixtures
r, wire = fixtures.r, fixtures.wire


class Admission(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.Offline()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_fresh_head_all_gates_and_verified_receipt_admit(self):
        f = self.fixture
        fresh = dict(f.pr, mergeStateStatus='CLEAN', statusCheckRollup=[
            dict(context=r.SPARK, state='SUCCESS'),
            dict(name=r.HOSTED, status='COMPLETED', conclusion='SUCCESS')])
        with patch.object(r, 'gh_json', side_effect=[wire([fresh], single=True), {'contexts': [r.HOSTED]}]), \
             patch.object(r, 'Queue') as queue:
            queue.return_value.source.return_value = (f.clone, 'github.com/aien-dev/omega-mv')
            queue.return_value.receipt.return_value = {'eligible': True}  # Explicit synthetic verifier boundary.
            self.assertTrue(r.remote_admitted('omega', 'aien-dev/omega-mv', f.pr, merge=True)[0])

    def test_failed_remote_close_preserves_block_and_carried_heads(self):
        f = self.fixture
        state = dict(open=dict(pr=620, sha=f.sha, carried=[614, 615], heads={'614': f.sha, '615': 'b' * 40}), size=12, solo=[])
        previous = copy.deepcopy(state)
        block_pr = dict(f.pr, number=620, statusCheckRollup=[dict(context=r.SPARK, state='FAILURE')])
        result = dict(waiting=[], blocked=[])
        fresh = dict(state='OPEN', headRefOid=f.sha, mergeStateStatus='BLOCKED')
        with patch.object(r, 'gh_json', return_value=fresh), patch.object(r, 'remote_admitted', return_value=(True, 'synthetic fresh check')), \
             patch.object(r, 'operator_verdict', return_value=None), patch.object(r, 'run', return_value=(98, '', 'synthetic guard98')), \
             patch.object(r, 'save_block') as save:
            r.finish_block('omega', 'aien-dev/omega-mv', state, {620: block_pr}, result)
        self.assertEqual(state, previous)
        self.assertIn('state retained', result['waiting'][0])
        save.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
