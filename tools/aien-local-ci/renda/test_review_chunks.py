import json
from pathlib import Path
import tempfile
import unittest

import review_chunks as rc


class ChunkReview(unittest.TestCase):
    def run_review(self, diff, ask, cache, **overrides):
        args = dict(repository='github.com/aien-dev/omega-mv', key='omega-mv#9@' + 'a' * 40,
                    base='b' * 40, head='a' * 40, model='synthetic-model', policy='policy-v1',
                    ask=ask, cache_path=cache, budget=2, chunk_chars=11)
        args.update(overrides)
        return rc.review_diff(diff, **args)

    def test_full_diff_is_chunked_with_exact_character_coverage_and_context(self):
        diff = 'diff --git a/first b/first\n' + ('A' * 23) + '\ndiff --git a/second b/second\n' + ('B' * 19)
        with tempfile.TemporaryDirectory() as td:
            prompts = []
            def ask(prompt):
                prompts.append(prompt)
                return ('{"merge":true,"note":"checked"}', 'synthetic-model')
            result = self.run_review(diff, ask, Path(td) / 'chunks.json', budget=100)
            self.assertTrue(result['complete'])
            self.assertIs(result['decision'], True)
            cache = json.loads((Path(td) / 'chunks.json').read_text())
            rows = cache['chunks']
            self.assertEqual(rows[0]['start'], 0)
            self.assertEqual(rows[-1]['end'], len(diff))
            self.assertEqual([r['start'] for r in rows[1:]], [r['end'] for r in rows[:-1]])
            self.assertEqual(''.join(diff[r['start']:r['end']] for r in rows), diff)
            self.assertEqual(len(prompts), len(rows))
            for i, prompt in enumerate(prompts):
                self.assertIn(f'chunk {i + 1}/{len(rows)}', prompt)
                self.assertIn(cache['diff_digest'], prompt)
                self.assertIn(rows[i]['chunk_digest'], prompt)
                self.assertIn(f"offsets {rows[i]['start']}:{rows[i]['end']}", prompt)
            self.assertTrue(any('first' in p for p in prompts))
            self.assertTrue(any('second' in p for p in prompts))

    def test_one_oversized_line_uses_exact_offsets_and_is_never_truncated(self):
        diff = 'diff --git a/single b/single\n' + 'Q' * 65000
        with tempfile.TemporaryDirectory() as td:
            prompts = []
            def ask(prompt):
                prompts.append(prompt)
                return ('{"merge":true}', 'synthetic-model')
            result = self.run_review(diff, ask, Path(td) / 'chunks.json', chunk_chars=9000, budget=20)
            self.assertTrue(result['complete'])
            self.assertEqual(result['total_chunks'], 8)
            self.assertEqual(len(prompts), 8)
            cache = json.loads((Path(td) / 'chunks.json').read_text())
            self.assertEqual(cache['chunks'][0]['start'], 0)
            self.assertEqual(cache['chunks'][-1]['end'], len(diff))
            self.assertEqual(sum(len(diff[c['start']:c['end']]) for c in cache['chunks']), len(diff))
            self.assertTrue(all('file single' in prompt for prompt in prompts))
            self.assertTrue(all('offsets ' in prompt and 'chunk digest ' in prompt for prompt in prompts))

    def test_late_refusal_is_never_retried_and_never_turns_green(self):
        diff = 'abcdefghijABCDEFGHIJ'
        with tempfile.TemporaryDirectory() as td:
            calls = []
            def ask(prompt):
                calls.append(prompt)
                return ('{"merge":' + ('true' if len(calls) == 1 else 'false') + '}', 'synthetic-model')
            result = self.run_review(diff, ask, Path(td) / 'chunks.json', budget=20)
            self.assertTrue(result['complete'])
            self.assertIs(result['decision'], False)
            self.assertEqual(result['actual_calls'], 2)
            self.assertEqual(len(calls), 2)
            again = self.run_review(diff, ask, Path(td) / 'chunks.json', budget=20)
            self.assertIs(again['decision'], False)
            self.assertEqual(len(calls), 2)

    def test_transport_and_unparsable_results_are_incomplete_with_bounded_retry(self):
        diff = 'abcdefghijk'
        with tempfile.TemporaryDirectory() as td:
            calls = []
            def ask(prompt):
                calls.append(prompt)
                if len(calls) == 1:
                    raise OSError('synthetic transport failure')
                return ('not-json', 'synthetic-model')
            result = self.run_review(diff, ask, Path(td) / 'chunks.json', budget=10, max_attempts=2)
            self.assertIsNone(result['decision'])
            self.assertFalse(result['complete'])
            self.assertEqual(result['actual_calls'], 2)
            self.assertEqual(len(calls), 2)
            later = self.run_review(diff, ask, Path(td) / 'chunks.json', budget=1, max_attempts=2)
            self.assertIsNone(later['decision'])
            self.assertEqual(later['actual_calls'], 1)

    def test_non_boolean_and_wrong_model_are_incomplete(self):
        with tempfile.TemporaryDirectory() as td:
            for response in [('{"merge":1}', 'synthetic-model'), ('{"merge":true}', 'other-model')]:
                result = self.run_review('small diff', lambda prompt: response, Path(td) / (str(response[0]) + '.json'),
                                         chunk_chars=100, budget=2)
                self.assertIsNone(result['decision'])
                self.assertFalse(result['complete'])

    def test_exact_identity_invalidates_reuse_and_partial_chunks_resume(self):
        diff = 'abcdefghijklmnopqrstuvwxyz0123456789ABCD'
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'chunks.json'
            calls = []
            def yes(prompt):
                calls.append(prompt)
                return ('{"merge":true}', 'synthetic-model')
            first = self.run_review(diff, yes, path, budget=2)
            self.assertFalse(first['complete'])
            self.assertEqual(first['actual_calls'], 2)
            saved = json.loads(path.read_text())
            self.assertEqual(sum(row.get('decision') is True for row in saved['chunks']), 2)
            second = self.run_review(diff, yes, path, budget=2)
            self.assertTrue(second['complete'])
            self.assertIs(second['decision'], True)
            self.assertEqual(second['actual_calls'], 2)
            self.assertEqual(len(calls), 4)
            changed = self.run_review(diff, yes, path, budget=1, base='c' * 40)
            self.assertFalse(changed['complete'])
            self.assertEqual(changed['actual_calls'], 1)
            self.assertEqual(len(calls), 5)

    def test_each_exact_identity_field_invalidates_chunk_reuse(self):
        original_diff = 'abcdefghijklmnopqrstuvwxyz0123456789ABCD'
        changes = [dict(base='c' * 40), dict(head='d' * 40),
                   dict(policy='policy-v2'), dict(model='other-model'),
                   dict(diff=original_diff + ' changed')]
        with tempfile.TemporaryDirectory() as td:
            calls = []
            current_model = ['synthetic-model']
            def ask(prompt):
                calls.append(prompt)
                return ('{"merge":true}', current_model[0])
            for index, change in enumerate(changes):
                path = Path(td) / f'{index}.json'
                current_model[0] = 'synthetic-model'
                initial = self.run_review(original_diff, ask, path, budget=10)
                self.assertTrue(initial['complete'], initial)
                kwargs = dict(change)
                current_model[0] = kwargs.get('model', 'synthetic-model')
                changed_diff = kwargs.pop('diff', original_diff)
                result = self.run_review(changed_diff, ask, path, budget=1, **kwargs)
                self.assertFalse(result['complete'])
                self.assertEqual(result['actual_calls'], 1)
                self.assertEqual(result['reviewed_chunks'], 1)

    def test_corrupt_cache_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'chunks.json'
            path.write_text('{')
            result = self.run_review('abc', lambda prompt: ('bad', 'synthetic-model'), path,
                                     chunk_chars=100, budget=1, max_attempts=1)
            self.assertIsNone(result['decision'])
            self.assertFalse(result['complete'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
