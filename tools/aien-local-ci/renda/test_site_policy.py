import base64
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import site_policy as sp

_harness_spec = importlib.util.spec_from_file_location('renda_site_test', Path(__file__).with_name('renda-harness'),
    loader=importlib.machinery.SourceFileLoader('renda_site_test', str(Path(__file__).with_name('renda-harness'))))
renda = importlib.util.module_from_spec(_harness_spec)
_harness_spec.loader.exec_module(renda)


class SitePolicy(unittest.TestCase):
    def test_queue_rejects_unregistered_or_mismatched_repository_identity(self):
        import local_queue
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, 'unregistered'):
                local_queue.Queue(td, 'aienos.com', 'aien-dev/aienos.com-mv', lambda *a, **k: (0, '', ''), td, td)

    def test_site_ready_import_returns_exact_structured_local_ack(self):
        import local_queue
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            item = dict(repository='aien-dev/aienos.com', ref='feat/now-renda-speedometers',
                sha='d411e1e030f1c06c6863953cce02bbf432b74cb4', base='8' * 40)
            ready = root / 'ready.json'
            ready.write_text(json.dumps([item]))
            queue = local_queue.Queue(root / 'state', 'aienos.com', 'aien-dev/aienos.com',
                lambda *a, **k: (98, '', 'network denied'), root / 'clones', root / 'work')
            ack = queue.seed_ready(ready)
            self.assertEqual(ack['repository'], 'aien-dev/aienos.com')
            self.assertEqual(ack['items'][0]['head'], item['sha'])
            self.assertEqual(ack['items'][0]['ref'], item['ref'])
            self.assertEqual(ack['state'], 'ACCEPTED_LOCAL_NOT_PUBLISHED')
            self.assertIsNone(ack['items'][0]['number'])
            self.assertIsNone(ack['items'][0]['remote_observed_at'])
            self.assertTrue(Path(ack['ack_path']).is_file())
            self.assertEqual(local_queue.read(queue.path)['entries'][0]['key'],
                f"aien-dev/aienos.com:ready:{item['ref']}@{item['sha']}")

    def test_seed_ready_refuses_unregistered_source_and_no_wrong_queue_write(self):
        import local_queue
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ready = root / 'ready.json'
            ready.write_text(json.dumps([dict(repository='aien-dev/aienos.com-mv', ref='x',
                sha='d' * 40, base='8' * 40)]))
            queue = local_queue.Queue(root / 'state', 'aienos.com', 'aien-dev/aienos.com',
                lambda *a, **k: (98, '', 'network denied'), root / 'clones', root / 'work')
            with self.assertRaisesRegex(ValueError, 'READY identity'):
                queue.seed_ready(ready)
            self.assertFalse(queue.path.exists())

    def test_site_ready_progress_is_cold_start_local_only_without_spark_gate(self):
        import local_queue
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clone = root / 'clones' / 'aienos.com'
            clone.mkdir(parents=True)
            def git(*args):
                return subprocess.check_output(['git', '-C', str(clone), *args], text=True)
            git('init', '-q')
            git('config', 'user.email', 'test@example.invalid')
            git('config', 'user.name', 'Fixture')
            git('remote', 'add', 'origin', 'https://github.com/aien-dev/aienos.com.git')
            (clone / 'page.txt').write_text('base\n')
            git('add', '.')
            git('commit', '-qm', 'base')
            base = git('rev-parse', 'HEAD').strip()
            (clone / 'page.txt').write_text('site candidate\n')
            git('commit', '-qam', 'candidate')
            head = git('rev-parse', 'HEAD').strip()
            state = root / 'state'
            def offline_run(cmd, **kwargs):
                if cmd[0] == 'gh' or 'ls-remote' in cmd:
                    raise AssertionError('network call')
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=kwargs.get('timeout', 30))
                return proc.returncode, proc.stdout, proc.stderr
            queue = local_queue.Queue(state, 'aienos.com', 'aien-dev/aienos.com',
                offline_run, root / 'clones', root / 'work')
            ready = root / 'ready.json'
            ready.write_text(json.dumps([dict(repository='aien-dev/aienos.com', ref='feat/site', sha=head, base=base)]))
            ack = queue.seed_ready(ready)
            self.assertEqual(ack['state'], 'ACCEPTED_LOCAL_NOT_PUBLISHED')
            with patch.object(queue, 'host_plan', side_effect=AssertionError('site must not invoke Spark gates')):
                output = queue.progress(queue.load(), 'test-model', lambda prompt: ('{"merge":true,"note":"reviewed full diff"}', 'test-model'), 'test-policy')
            self.assertTrue(output)
            self.assertTrue(any('Spark gates not run' in line for line in output))
            plans = list((state / 'renda-queue' / 'local' / 'aienos.com').glob('*.plan.json'))
            self.assertEqual(len(plans), 1)
            plan = json.loads(plans[0].read_text())
            self.assertEqual(plan['commit'], head)
            self.assertEqual(plan['base'], base)
            self.assertEqual(plan['review']['decision'], True)
            self.assertFalse(plan['remote_eligible'])
            self.assertFalse(plan.get('qualification', {}).get('eligible', False))
            remote_calls = []
            def network_guard(cmd, **kwargs):
                if cmd[0] == 'gh':
                    remote_calls.append(cmd)
                    return 98, '', 'network denied'
                proc = subprocess.run(cmd, cwd=kwargs.get('cwd'), capture_output=True, text=True,
                    timeout=kwargs.get('timeout', 30))
                return proc.returncode, proc.stdout, proc.stderr
            with patch.dict(os.environ, {}, clear=True), patch.object(renda, 'run', side_effect=network_guard), \
                    patch.object(renda, 'lane_owned', return_value=True):
                entry = queue.load()['entries'][0]
                ok, note = renda.site_publish_ready('aienos.com', queue, entry, 'test-model', 'test-policy')
            self.assertFalse(ok)
            self.assertIn('qualification', note)
            self.assertEqual(remote_calls, [])

    def test_branch_rule_required_checks_are_enforced_but_empty_rules_do_not_replace_mac_receipt(self):
        protection = dict(contexts=['Hosted'], checks=[dict(context='Hosted')])
        rulesets = [dict(enforcement='active', rules=[dict(type='required_status_checks', parameters={
            'required_status_checks':[dict(context='Review')]} )])]
        required = sp.site_required_checks(protection, rulesets)
        self.assertEqual(required, {'Hosted', 'Review'})
        self.assertFalse(sp.required_checks_satisfied(required, []))
        self.assertTrue(sp.required_checks_satisfied(required, [dict(context='Hosted', state='SUCCESS'),
            dict(name='Review', status='COMPLETED', conclusion='SUCCESS')]))
        self.assertFalse(sp.required_checks_satisfied(required, [dict(context='Hosted', state='SUCCESS')]))
        self.assertEqual(sp.site_required_checks(None, []), set())

    def test_branch_rule_malformed_status_contract_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'malformed'):
            sp.site_required_checks({'contexts': None}, [])
        with self.assertRaisesRegex(ValueError, 'malformed'):
            sp.site_required_checks(None, [dict(enforcement='active', rules=None)])

    def test_existing_pr_snapshot_reuses_one_exact_open_pr_and_never_duplicates(self):
        pr = dict(number=17, state='OPEN', headRefName='feat/site', headRefOid='d' * 40, baseRefName='main', baseRefOid='b' * 40)
        result = sp.inspect_site_pr_snapshot([pr], 'feat/site', 'd' * 40, 'b' * 40)
        self.assertEqual(result['action'], 'reuse')
        self.assertEqual(result['number'], 17)
        self.assertEqual(sp.inspect_site_pr_snapshot([], 'feat/site', 'd' * 40, 'b' * 40)['action'], 'create')

    def test_closed_malformed_moved_or_duplicate_pr_snapshot_is_refused(self):
        exact = dict(number=17, state='OPEN', headRefName='feat/site', headRefOid='d' * 40, baseRefName='main', baseRefOid='b' * 40)
        cases = ([dict(exact, state='CLOSED')], [dict(exact, headRefOid='e' * 40)],
            [dict(exact, baseRefOid='c' * 40)], [{}], [exact, exact], None)
        for rows in cases:
            with self.subTest(rows=rows):
                self.assertEqual(sp.inspect_site_pr_snapshot(rows, 'feat/site', 'd' * 40, 'b' * 40)['action'], 'refuse')

    def test_remote_branch_and_base_must_match_exact_ready_pins(self):
        good_branch = f"{'d'*40}{chr(9)}refs/heads/feat/site{chr(10)}"
        good_base = f"{'b'*40}{chr(9)}refs/heads/main{chr(10)}"
        self.assertTrue(sp.exact_remote_refs(good_branch, good_base, 'feat/site', 'd' * 40, 'b' * 40))
        self.assertFalse(sp.exact_remote_refs(good_branch.replace('d'*40, 'e'*40), good_base,
            'feat/site', 'd' * 40, 'b' * 40))
        self.assertFalse(sp.exact_remote_refs(good_branch, good_base.replace('b'*40, 'c'*40),
            'feat/site', 'd' * 40, 'b' * 40))
        self.assertFalse(sp.exact_remote_refs(good_branch + good_branch, good_base,
            'feat/site', 'd' * 40, 'b' * 40))

    def test_signed_exact_site_ready_publishes_once_and_reads_back_clean_pr(self):
        import local_queue
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clone = root / 'clones' / 'aienos.com'; clone.mkdir(parents=True)
            def git(*args):
                return subprocess.check_output(['git', '-C', str(clone), *args], text=True)
            git('init', '-q')
            git('config', 'user.email', 'test@example.invalid')
            git('config', 'user.name', 'Fixture')
            git('remote', 'add', 'origin', 'https://github.com/aien-dev/aienos.com.git')
            (clone / 'page.txt').write_text('base\\n'); git('add', '.'); git('commit', '-qm', 'base')
            base = git('rev-parse', 'HEAD').strip()
            git('checkout', '-qb', 'feat/site')
            (clone / 'page.txt').write_text('qualified site\\n'); git('commit', '-qam', 'site')
            head = git('rev-parse', 'HEAD').strip()
            state = root / 'state'
            queue = local_queue.Queue(state, 'aienos.com', 'aien-dev/aienos.com', renda.run, root / 'clones', root / 'work')
            ready = root / 'ready.json'
            ready.write_text(json.dumps([dict(repository='aien-dev/aienos.com', ref='feat/site', sha=head, base=base)]))
            queue.seed_ready(ready)
            priv, pub = root / 'private.pem', root / 'public.pem'
            subprocess.run(['openssl', 'genpkey', '-algorithm', 'Ed25519', '-out', str(priv)], check=True)
            subprocess.run(['openssl', 'pkey', '-in', str(priv), '-pubout', '-out', str(pub)], check=True)
            tree = git('rev-parse', 'HEAD^{tree}').strip()
            receipt = dict(schema=sp.SITE_RECEIPT_SCHEMA, repository='aien-dev/aienos.com', commit=head,
                tree=tree, recipe_id=sp.SITE_RECIPE_ID, qualified=True, platform='macOS',
                commands=[dict(argv=list(argv), exit_code=0, log_sha256='a' * 64) for argv in sp.SITE_REQUIRED_COMMANDS])
            qual = state / 'renda-queue' / 'site-qualification'; qual.mkdir(parents=True)
            receipt_path, sig_path = qual / (head + '.json'), qual / (head + '.sig')
            receipt_path.write_bytes(sp.canonical_json(receipt))
            subprocess.run(['openssl', 'pkeyutl', '-sign', '-rawin', '-inkey', str(priv),
                '-in', str(receipt_path), '-out', str(root / 'sig.bin')], check=True)
            sig_path.write_text(base64.b64encode((root / 'sig.bin').read_bytes()).decode())
            with patch.dict(os.environ, {'RENDA_SITE_QUALIFICATION_PUBKEY': str(pub)}):
                queue.progress(queue.load(), 'review-model',
                    lambda prompt: ('{"merge":true,"note":"complete diff reviewed"}', 'review-model'), 'test-policy')
                created = []
                require_status = True
                status_success = False
                def fake_run(cmd, **kwargs):
                    if cmd[0] == 'gh':
                        if 'api' in cmd:
                            endpoint = cmd[-1]
                            if '/protection/required_status_checks' in endpoint:
                                return 1, '', 'HTTP 404: no branch protection configured'
                            if '/rulesets?' in endpoint:
                                rulesets = ([dict(enforcement='active', rules=[dict(type='required_status_checks',
                                    parameters={'required_status_checks':[dict(context='Hosted')]} )])]
                                    if require_status else [])
                                return 0, json.dumps(rulesets), ''
                        if 'list' in cmd:
                            if not created:
                                return 0, '[]', ''
                            return 0, json.dumps([dict(number=17, state='OPEN', headRefName='feat/site',
                                headRefOid=head, baseRefName='main', baseRefOid=base)]), ''
                        if 'create' in cmd:
                            created.append(17)
                            return 0, f"https://github.com/aien-dev/aienos.com/pull/17{chr(10)}", ''
                        if 'view' in cmd:
                            return 0, json.dumps(dict(number=17, state='OPEN', headRefName='feat/site',
                                headRefOid=head, baseRefName='main', baseRefOid=base, mergeStateStatus='CLEAN',
                                statusCheckRollup=([dict(context='Hosted', state='SUCCESS')]
                                    if status_success else []))), ''
                    if cmd[0] == 'git' and 'ls-remote' in cmd:
                        ref = cmd[-1]
                        value = head if ref.endswith('feat/site') else base
                        return 0, f"{value}{chr(9)}{ref}{chr(10)}", ''
                    process = subprocess.run(cmd, cwd=kwargs.get('cwd'), capture_output=True, text=True,
                        timeout=kwargs.get('timeout', 30))
                    return process.returncode, process.stdout, process.stderr
                queue.run = fake_run
                with patch.object(renda, 'run', side_effect=fake_run), patch.object(renda, 'lane_owned', return_value=True):
                    entry = queue.load()['entries'][0]
                    policy = 'test-policy'
                    pending = renda.site_publish_ready('aienos.com', queue, entry, 'review-model', policy)
                    status_success = True
                    first = renda.site_publish_ready('aienos.com', queue, entry, 'review-model', policy)
                    second = renda.site_publish_ready('aienos.com', queue, entry, 'review-model', policy)
            self.assertFalse(pending[0], pending)
            self.assertIn('required branch checks', pending[1])
            self.assertTrue(first[0], first)
            self.assertTrue(second[0], second)
            self.assertEqual(created, [17])
            ack_path = list((state / 'renda-queue' / 'acks').glob('*.json'))[0]
            ack = json.loads(ack_path.read_text())
            self.assertEqual(ack['state'], 'PR_OPEN_CLEAN_PUBLICATION_PENDING_REVIEW')
            self.assertEqual(ack['pr_number'], 17)
            self.assertEqual(first[1].count('CLEAN'), 1)

    def test_exact_repository_registry_and_unknown_identity_fail_closed(self):
        self.assertEqual(sp.repository_identity('aienos.com'), 'aien-dev/aienos.com')
        self.assertEqual(sp.repository_identity('aien-dev/aienos.com'), 'aien-dev/aienos.com')
        self.assertEqual(sp.repository_identity('aienos'), 'aien-dev/aienos-mv')
        with self.assertRaisesRegex(ValueError, 'unregistered'):
            sp.repository_identity('aienos.com-mv')

    def test_canonical_signed_site_qualification_is_exact_and_verifiable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            priv, pub = root / 'private.pem', root / 'public.pem'
            subprocess.run(['openssl', 'genpkey', '-algorithm', 'Ed25519', '-out', str(priv)], check=True)
            subprocess.run(['openssl', 'pkey', '-in', str(priv), '-pubout', '-out', str(pub)], check=True)
            repo = root / 'repo'
            repo.mkdir()
            subprocess.run(['git', 'init', '-q', str(repo)], check=True)
            subprocess.run(['git', '-C', str(repo), 'config', 'user.email', 'test@example.invalid'], check=True)
            subprocess.run(['git', '-C', str(repo), 'config', 'user.name', 'Fixture'], check=True)
            (repo / 'source.txt').write_text('exact head\n')
            subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(repo), 'commit', '-qm', 'fixture'], check=True)
            commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
            tree = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD^{tree}'], text=True).strip()
            receipt = dict(schema=sp.SITE_RECEIPT_SCHEMA, repository='aien-dev/aienos.com',
                commit=commit, tree=tree, recipe_id=sp.SITE_RECIPE_ID, qualified=True,
                platform='macOS', commands=[dict(argv=argv, exit_code=0, log_sha256='a' * 64)
                    for argv in sp.SITE_REQUIRED_COMMANDS])
            raw = sp.canonical_json(receipt)
            receipt_path, sig_path = root / 'receipt.json', root / 'receipt.sig'
            receipt_path.write_bytes(raw)
            subprocess.run(['openssl', 'pkeyutl', '-sign', '-rawin', '-inkey', str(priv),
                '-in', str(receipt_path), '-out', str(root / 'signature.bin')], check=True)
            sig_path.write_text(base64.b64encode((root / 'signature.bin').read_bytes()).decode('ascii'))
            with patch.dict(os.environ, {'RENDA_SITE_QUALIFICATION_PUBKEY': str(pub)}):
                result = sp.verify_site_qualification(receipt_path, sig_path, repo,
                    'aien-dev/aienos.com', commit)
            self.assertTrue(result['eligible'], result)
            self.assertEqual(result['commit'], commit)
            self.assertEqual(result['tree'], tree)

    def test_unsigned_or_missing_external_key_qualification_fails_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            result = sp.verify_site_qualification('/missing/receipt', '/missing/signature',
                '/missing/repo', 'aien-dev/aienos.com', 'a' * 40)
        self.assertFalse(result['eligible'])
        self.assertIn('public key', result['diagnostic'])

    def test_receipt_cannot_self_authorize_key_or_wrong_recipe(self):
        receipt = dict(schema=sp.SITE_RECEIPT_SCHEMA, repository='aien-dev/aienos.com',
            commit='a' * 40, tree='b' * 40, recipe_id='author-chosen', qualified=True,
            platform='macOS', commands=[])
        result = sp.validate_site_receipt(receipt, 'aien-dev/aienos.com', 'a' * 40, 'b' * 40)
        self.assertFalse(result['eligible'])
        self.assertIn('recipe', result['diagnostic'])
        receipt['recipe_id'] = sp.SITE_RECIPE_ID
        receipt['public_key'] = 'fake'
        result = sp.validate_site_receipt(receipt, 'aien-dev/aienos.com', 'a' * 40, 'b' * 40)
        self.assertFalse(result['eligible'])
        self.assertIn('verification key', result['diagnostic'])

    def test_receipt_rejects_commands_outside_pinned_recipe(self):
        receipt = dict(schema=sp.SITE_RECEIPT_SCHEMA, repository='aien-dev/aienos.com',
            commit='a' * 40, tree='b' * 40, recipe_id=sp.SITE_RECIPE_ID, qualified=True,
            platform='macOS', commands=[dict(argv=list(argv), exit_code=0, log_sha256='a' * 64)
                for argv in sp.SITE_REQUIRED_COMMANDS] + [dict(argv=['python3', '-c', 'untrusted'],
                    exit_code=0, log_sha256='b' * 64)])
        result = sp.validate_site_receipt(receipt, 'aien-dev/aienos.com', 'a' * 40, 'b' * 40)
        self.assertFalse(result['eligible'])
        self.assertIn('exact recipe', result['diagnostic'])

    def test_receipt_requires_all_exact_commands_zero_exit_and_valid_log_hashes(self):
        receipt = dict(schema=sp.SITE_RECEIPT_SCHEMA, repository='aien-dev/aienos.com',
            commit='a' * 40, tree='b' * 40, recipe_id=sp.SITE_RECIPE_ID, qualified=True,
            platform='macOS', commands=[dict(argv=list(argv), exit_code=0, log_sha256='a' * 64)
                for argv in sp.SITE_REQUIRED_COMMANDS[:-1]])
        result = sp.validate_site_receipt(receipt, 'aien-dev/aienos.com', 'a' * 40, 'b' * 40)
        self.assertFalse(result['eligible'])
        self.assertIn('commands', result['diagnostic'])
        receipt['commands'].append(dict(argv=list(sp.SITE_REQUIRED_COMMANDS[-1]), exit_code=1, log_sha256='a' * 64))
        result = sp.validate_site_receipt(receipt, 'aien-dev/aienos.com', 'a' * 40, 'b' * 40)
        self.assertFalse(result['eligible'])

    def test_moved_commit_tree_repository_and_tampered_signature_refuse(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            priv, pub = root / 'private.pem', root / 'public.pem'
            subprocess.run(['openssl', 'genpkey', '-algorithm', 'Ed25519', '-out', str(priv)], check=True)
            subprocess.run(['openssl', 'pkey', '-in', str(priv), '-pubout', '-out', str(pub)], check=True)
            repo = root / 'repo'; repo.mkdir()
            subprocess.run(['git', 'init', '-q', str(repo)], check=True)
            subprocess.run(['git', '-C', str(repo), 'config', 'user.email', 'test@example.invalid'], check=True)
            subprocess.run(['git', '-C', str(repo), 'config', 'user.name', 'Fixture'], check=True)
            (repo / 'source.txt').write_text('x\n')
            subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(repo), 'commit', '-qm', 'fixture'], check=True)
            commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
            tree = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD^{tree}'], text=True).strip()
            receipt = dict(schema=sp.SITE_RECEIPT_SCHEMA, repository='aien-dev/aienos.com', commit=commit,
                tree=tree, recipe_id=sp.SITE_RECIPE_ID, qualified=True, platform='macOS',
                commands=[dict(argv=argv, exit_code=0, log_sha256='a' * 64) for argv in sp.SITE_REQUIRED_COMMANDS])
            receipt_path, sig_path = root / 'receipt.json', root / 'receipt.sig'
            receipt_path.write_bytes(sp.canonical_json(receipt))
            subprocess.run(['openssl', 'pkeyutl', '-sign', '-rawin', '-inkey', str(priv),
                '-in', str(receipt_path), '-out', str(root / 'signature.bin')], check=True)
            sig_path.write_text(base64.b64encode((root / 'signature.bin').read_bytes()).decode('ascii'))
            with patch.dict(os.environ, {'RENDA_SITE_QUALIFICATION_PUBKEY': str(pub)}):
                self.assertFalse(sp.verify_site_qualification(receipt_path, sig_path, repo,
                    'aien-dev/aienos.com', 'c' * 40)['eligible'])
                receipt['repository'] = 'aien-dev/aienos.com-mv'
                receipt_path.write_bytes(sp.canonical_json(receipt))
                self.assertFalse(sp.verify_site_qualification(receipt_path, sig_path, repo,
                    'aien-dev/aienos.com', commit)['eligible'])
                receipt['repository'] = 'aien-dev/aienos.com'
                receipt['tree'] = 'd' * 40
                receipt_path.write_bytes(sp.canonical_json(receipt))
                self.assertFalse(sp.verify_site_qualification(receipt_path, sig_path, repo,
                    'aien-dev/aienos.com', commit)['eligible'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
