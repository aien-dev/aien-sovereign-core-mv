"""Explicit repository identities and fail-closed site qualification verification."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

SHA = re.compile(r'[0-9a-f]{40}')
HEX256 = re.compile(r'[0-9a-f]{64}')

# Every lane identity is explicit. In particular, the website is aien-dev/aienos.com,
# not an invented aienos.com-mv and not the distinct aienos kernel project.
REPOSITORY_IDENTITIES = {
    'aien-sovereign-core': 'aien-dev/aien-sovereign-core-mv',
    'aien-architecture': 'aien-dev/aien-architecture-mv',
    'aien-protocols': 'aien-dev/aien-protocols-mv',
    'physics': 'aien-dev/physics-mv',
    'benchmarks': 'aien-dev/benchmarks-mv',
    'interplane': 'aien-dev/interplane-mv',
    'omega': 'aien-dev/omega-mv',
    'aienos': 'aien-dev/aienos-mv',
    'spark-rsi': 'aien-dev/spark-rsi-mv',
    'aegis-runtime': 'aien-dev/aegis-runtime-mv',
    'aien-rig': 'aien-dev/aien-rig',
    'aien-train': 'aien-dev/aien-train',
    'synapse': 'aien-dev/synapse',
    'CUDApp': 'aien-dev/CUDApp',
    'aienos.com': 'aien-dev/aienos.com',
}
for _full in tuple(REPOSITORY_IDENTITIES.values()):
    REPOSITORY_IDENTITIES[_full] = _full

SITE_REPOSITORY = 'aien-dev/aienos.com'
SITE_POLICY_ID = 'renda-site-policy/aienos.com-v1'
SITE_RECEIPT_SCHEMA = 'renda-site-qualification/1'
SITE_RECIPE_ID = 'aienos.com/mac-site-tests-build-collector-v1'
SITE_REQUIRED_COMMANDS = (
    ('cargo', 'test', '-q', '--manifest-path', 'crates/site-builder/Cargo.toml'),
    ('cargo', 'run', '-q', '--release', '--manifest-path', 'crates/site-builder/Cargo.toml', '--', 'build', 'dist'),
    ('python3', 'scripts/test_renda_metrics.py'),
)

# Other registered repositories retain the generic L522 policy. Website uses its
# independent, exact-head local Mac qualification and never the Spark gate policy.
REPOSITORY_POLICIES = {identity: {'policy_id': 'renda-generic-pr-policy/1'}
                       for identity in set(REPOSITORY_IDENTITIES.values())}
REPOSITORY_POLICIES[SITE_REPOSITORY] = {'policy_id': SITE_POLICY_ID,
    'base_ref': 'main', 'qualification': SITE_RECIPE_ID, 'auto_merge': False}


def repository_identity(value):
    try:
        return REPOSITORY_IDENTITIES[value]
    except (KeyError, TypeError):
        raise ValueError(f'unregistered repository identity: {value!r}')


def repository_policy(value):
    full = repository_identity(value)
    return full, REPOSITORY_POLICIES[full]



def exact_remote_refs(branch_output, base_output, branch, head, base):
    if not isinstance(branch_output, str) or not isinstance(base_output, str):
        return False
    def parse(raw, expected):
        lines = [line.split() for line in raw.splitlines() if line.strip()]
        return (len(lines) == 1 and len(lines[0]) == 2 and SHA.fullmatch(lines[0][0])
                and lines[0][0] == expected and lines[0][1] == 'refs/heads/' + branch)
    if not SHA.fullmatch(str(head)) or not SHA.fullmatch(str(base)):
        return False
    branch_ok = parse(branch_output, head)
    base_lines = [line.split() for line in base_output.splitlines() if line.strip()]
    base_ok = (len(base_lines) == 1 and len(base_lines[0]) == 2 and SHA.fullmatch(base_lines[0][0])
               and base_lines[0][0] == base and base_lines[0][1] == 'refs/heads/main')
    return bool(branch_ok and base_ok)


def inspect_site_pr_snapshot(rows, branch, head, base):
    if not isinstance(rows, list):
        return {'action': 'refuse', 'reason': 'PR list is malformed'}
    if not rows:
        return {'action': 'create', 'reason': 'no PR exists for exact branch'}
    if len(rows) != 1 or not isinstance(rows[0], dict):
        return {'action': 'refuse', 'reason': 'duplicate or malformed PR identity'}
    pr = rows[0]
    if (type(pr.get('number')) is not int or pr['number'] <= 0
            or pr.get('state') not in ('OPEN', 'CLOSED', 'MERGED')
            or pr.get('headRefName') != branch or not SHA.fullmatch(str(pr.get('headRefOid', '')))
            or pr.get('baseRefName') != 'main' or not SHA.fullmatch(str(pr.get('baseRefOid', '')))):
        return {'action': 'refuse', 'reason': 'PR identity is incomplete or malformed'}
    if pr['state'] != 'OPEN':
        return {'action': 'refuse', 'reason': 'branch already has a closed or merged PR'}
    if pr['headRefOid'] != head or pr['baseRefOid'] != base:
        return {'action': 'refuse', 'reason': 'existing PR head or base moved'}
    return {'action': 'reuse', 'number': pr['number'], 'reason': 'exact open PR already exists'}


def site_required_checks(protection, rulesets):
    """Collect active required status contexts from branch protection and rulesets."""
    required = set()
    if protection is not None:
        if not isinstance(protection, dict):
            raise ValueError('malformed branch protection response')
        contexts, checks = protection.get('contexts', []), protection.get('checks', [])
        if not isinstance(contexts, list) or not isinstance(checks, list):
            raise ValueError('malformed branch protection status contract')
        for context in contexts:
            if not isinstance(context, str) or not context:
                raise ValueError('malformed branch protection status context')
            required.add(context)
        for check in checks:
            if not isinstance(check, dict) or not isinstance(check.get('context'), str) or not check['context']:
                raise ValueError('malformed branch protection check')
            required.add(check['context'])
    if not isinstance(rulesets, list):
        raise ValueError('malformed rulesets response')
    for ruleset in rulesets:
        if not isinstance(ruleset, dict):
            raise ValueError('malformed ruleset')
        if ruleset.get('enforcement') != 'active':
            continue
        rules = ruleset.get('rules', [])
        if not isinstance(rules, list):
            raise ValueError('malformed ruleset rule list')
        for rule in rules:
            if not isinstance(rule, dict):
                raise ValueError('malformed ruleset rule')
            if rule.get('type') != 'required_status_checks':
                continue
            parameters = rule.get('parameters')
            if not isinstance(parameters, dict):
                raise ValueError('malformed required status check rule')
            checks = parameters.get('required_status_checks')
            if not isinstance(checks, list):
                raise ValueError('malformed required status check list')
            for check in checks:
                if not isinstance(check, dict) or not isinstance(check.get('context'), str) or not check['context']:
                    raise ValueError('malformed required status check context')
                required.add(check['context'])
    return required


def required_checks_satisfied(required, rollup):
    """Require every fresh PR status context named by currently active branch rules."""
    if not isinstance(required, set) or not isinstance(rollup, list):
        return False
    observed = {}
    for row in rollup:
        if not isinstance(row, dict):
            return False
        name = row.get('context') or row.get('name')
        if not isinstance(name, str) or not name:
            return False
        state = str(row.get('state') or '').upper()
        conclusion = str(row.get('conclusion') or '').upper()
        status = str(row.get('status') or '').upper()
        observed[name] = state == 'SUCCESS' or (status == 'COMPLETED' and conclusion == 'SUCCESS')
    return all(observed.get(context) is True for context in required)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def validate_site_receipt(receipt, repository, commit, tree):
    """Validate trusted recipe data. This alone is not signature verification."""
    if not isinstance(receipt, dict):
        return {'eligible': False, 'diagnostic': 'receipt must be a JSON object'}
    if any(key in receipt for key in ('public_key', 'pubkey', 'key', 'verification_key')):
        return {'eligible': False, 'diagnostic': 'receipt must not supply its own verification key'}
    if receipt.get('schema') != SITE_RECEIPT_SCHEMA or receipt.get('recipe_id') != SITE_RECIPE_ID:
        return {'eligible': False, 'diagnostic': 'unknown qualification schema or recipe'}
    if repository != SITE_REPOSITORY or receipt.get('repository') != SITE_REPOSITORY:
        return {'eligible': False, 'diagnostic': 'qualification repository identity mismatch'}
    if not SHA.fullmatch(str(commit)) or receipt.get('commit') != commit:
        return {'eligible': False, 'diagnostic': 'qualification commit does not match exact source head'}
    if not SHA.fullmatch(str(tree)) or receipt.get('tree') != tree:
        return {'eligible': False, 'diagnostic': 'qualification tree does not match exact source tree'}
    if receipt.get('qualified') is not True or receipt.get('platform') != 'macOS':
        return {'eligible': False, 'diagnostic': 'qualification is not an affirmative macOS result'}
    commands = receipt.get('commands')
    if not isinstance(commands, list):
        return {'eligible': False, 'diagnostic': 'qualification commands missing'}
    seen = set()
    for row in commands:
        if not isinstance(row, dict) or not isinstance(row.get('argv'), list):
            return {'eligible': False, 'diagnostic': 'malformed qualification command'}
        argv = row['argv']
        if not all(isinstance(arg, str) and arg for arg in argv):
            return {'eligible': False, 'diagnostic': 'malformed qualification argv'}
        key = tuple(argv)
        if key in seen:
            return {'eligible': False, 'diagnostic': 'duplicate qualification command'}
        seen.add(key)
        if type(row.get('exit_code')) is not int or row['exit_code'] != 0:
            return {'eligible': False, 'diagnostic': 'qualification command did not exit zero'}
        if not isinstance(row.get('log_sha256'), str) or not HEX256.fullmatch(row['log_sha256']):
            return {'eligible': False, 'diagnostic': 'qualification command log hash invalid'}
    if [tuple(row['argv']) for row in commands if isinstance(row, dict) and isinstance(row.get('argv'), list)] != list(SITE_REQUIRED_COMMANDS):
        return {'eligible': False, 'diagnostic': 'qualification commands do not match exact recipe'}
    return {'eligible': True, 'diagnostic': 'trusted recipe fields match exact repository, commit and tree',
            'commit': commit, 'tree': tree, 'recipe_id': SITE_RECIPE_ID}


def verify_site_qualification(receipt_path, signature_path, source, repository, commit,
                              public_key=None):
    """Verify canonical JSON with deployment-pinned Ed25519 public key and exact git tree."""
    key_path = public_key or os.environ.get('RENDA_SITE_QUALIFICATION_PUBKEY')
    if not key_path or not Path(key_path).is_file():
        return {'eligible': False, 'diagnostic': 'deployment-pinned site qualification public key unavailable'}
    try:
        raw = Path(receipt_path).read_bytes()
        receipt = json.loads(raw.decode('utf-8'))
        canonical = canonical_json(receipt)
        if raw != canonical:
            raise ValueError('receipt is not canonical JSON bytes')
        tree = subprocess.check_output(['git', '-C', str(source), 'rev-parse', commit + '^{tree}'],
                                       stderr=subprocess.DEVNULL, text=True, timeout=20).strip()
        validated = validate_site_receipt(receipt, repository, commit, tree)
        if not validated['eligible']:
            return validated
        signature = base64.b64decode(Path(signature_path).read_text().strip(), validate=True)
        if len(signature) != 64:
            return {'eligible': False, 'diagnostic': 'Ed25519 signature must be 64 bytes'}
        with tempfile.TemporaryDirectory(prefix='renda-site-qualification-') as td:
            message = Path(td) / 'receipt.canonical.json'
            sig = Path(td) / 'receipt.sig'
            message.write_bytes(canonical)
            sig.write_bytes(signature)
            result = subprocess.run(['openssl', 'pkeyutl', '-verify', '-pubin', '-inkey', str(key_path),
                '-rawin', '-in', str(message), '-sigfile', str(sig)], capture_output=True,
                text=True, timeout=20)
        if result.returncode:
            return {'eligible': False, 'diagnostic': 'detached Ed25519 signature verification failed'}
        validated['signature'] = 'verified against deployment-pinned public key'
        validated['receipt_sha256'] = hashlib.sha256(raw).hexdigest()
        return validated
    except (OSError, ValueError, TypeError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        return {'eligible': False, 'diagnostic': f'qualification verification unavailable: {type(exc).__name__}: {exc}'}
