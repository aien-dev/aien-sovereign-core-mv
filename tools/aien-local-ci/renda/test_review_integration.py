"""Parent integration regressions; callbacks are synthetic, no live model or remote write."""
import json
from pathlib import Path
import tempfile
import unittest
import review_chunks


class ReviewIntegration(unittest.TestCase):
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
