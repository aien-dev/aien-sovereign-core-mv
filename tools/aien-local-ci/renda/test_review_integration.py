"""Parent integration regressions; callbacks are synthetic, no live model or remote write."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import review_chunks


class ReviewIntegration(unittest.TestCase):
    def setUp(self):
        td=tempfile.TemporaryDirectory();self.addCleanup(td.cleanup)
        self.keypath=Path(td.name)/'private/review.key'
        auth=patch('review_chunks._key_path',return_value=self.keypath)
        auth.start();self.addCleanup(auth.stop)

    def test_segment_prompt_states_scope_without_reinterpreting_refusals(self):
        with tempfile.TemporaryDirectory() as td:
            prompts=[]
            def refuse(prompt):
                prompts.append(prompt)
                return '{"merge":false,"note":"Diff is truncated and incomplete, preventing verification of file integrity and content."}','model'
            result=review_chunks.review_diff('source'*4000,repository='repo',key='key',base='b'*40,head='h'*40,
                model='model',policy='policy',cache_path=Path(td)/'chunks.json',ask=refuse,budget=1)
            self.assertIn('not a truncated delivery',prompts[0])
            self.assertIn('segment only',prompts[0])
            self.assertIs(result['decision'],False,'an actual negative response was reinterpreted')

    def test_private_cache_key_is_stable_and_never_group_readable(self):
        first=review_chunks._load_key()
        self.assertEqual(len(first),32)
        self.assertEqual(review_chunks._load_key(),first)
        self.assertEqual(self.keypath.stat().st_mode & 0o777,0o600)
        self.assertEqual(self.keypath.parent.stat().st_mode & 0o777,0o700)

    def test_insecure_or_symlinked_cache_key_is_refused(self):
        review_chunks._load_key();self.keypath.chmod(0o644)
        with self.assertRaises(PermissionError):review_chunks._load_key()
        self.keypath.unlink();target=self.keypath.parent/'target';target.write_bytes(b'synthetic-key-for-test-only-00000'[:32])
        self.keypath.symlink_to(target)
        with self.assertRaises(OSError):review_chunks._load_key()

    def test_authentication_root_change_invalidates_old_approvals(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'chunks.json'
            args=dict(repository='repo',key='key',base='b'*40,head='h'*40,model='model',policy='policy',cache_path=path)
            reviewed=review_chunks.review_diff('original',ask=lambda prompt:('{"merge":true,"note":"synthetic approved"}','model'),budget=1,**args)
            self.assertIs(reviewed['decision'],True)
            self.keypath.unlink()
            result=review_chunks.review_diff('original',ask=lambda prompt:self.fail('zero-budget model call'),budget=0,**args)
            self.assertIsNone(result['decision'])
            self.assertFalse(result['complete'])

    def test_forged_cached_decisions_never_confer_approval(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'chunks.json'
            args=dict(repository='repo',key='key',base='b'*40,head='h'*40,model='model',policy='policy',cache_path=path)
            review_chunks.review_diff('original'*6000,ask=lambda prompt:None,budget=0,**args)
            data=json.loads(path.read_text())
            for row in data['chunks']:
                row['decision']=True;row['note']='forged'
            path.write_text(json.dumps(data))
            result=review_chunks.review_diff('original'*6000,ask=lambda prompt:self.fail('zero-budget model call'),budget=0,**args)
            self.assertFalse(result['complete'])
            self.assertIsNone(result['decision'])

    def test_refused_chunk_stops_further_model_work(self):
        with tempfile.TemporaryDirectory() as td:
            calls=[]
            def refuse(prompt):
                calls.append(prompt);return '{"merge":false,"note":"synthetic defect"}','model'
            args=dict(repository='repo',key='key',base='b'*40,head='h'*40,model='model',policy='policy',
                ask=refuse,cache_path=Path(td)/'chunks.json',budget=2)
            result=review_chunks.review_diff('original'*6000,**args)
            self.assertIs(result['decision'],False)
            self.assertEqual(len(calls),1,'work continued after a real refusal')
            review_chunks.review_diff('original'*6000,**args)
            self.assertEqual(len(calls),1,'refused review consumed another pass budget')

    def test_pending_chunk_cache_text_cannot_replace_actual_source(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'chunks.json';text='ORIGINAL_SOURCE'*1000;prompts=[]
            args=dict(repository='repo',key='key',base='b'*40,head='h'*40,model='model',policy='policy',cache_path=path)
            review_chunks.review_diff(text,ask=lambda prompt:None,budget=0,**args)
            data=json.loads(path.read_text());data['chunks'][0]['text']='REPLACEMENT_NOT_SOURCE';path.write_text(json.dumps(data))
            def approve(prompt):
                prompts.append(prompt);return '{"merge":true,"note":"synthetic approval"}','model'
            review_chunks.review_diff(text,ask=approve,budget=1,**args)
            self.assertTrue(prompts)
            self.assertIn(text[:12000],prompts[0],'review did not consume actual source chunk')
            self.assertNotIn('REPLACEMENT_NOT_SOURCE',prompts[0])


if __name__=='__main__':unittest.main()
