"""Renda delivery boundaries. Local assessment never grants remote merge authority."""
import json
from pathlib import Path
import re
import tempfile
import time

from local_queue import digest, read, save
from site_policy import exact_remote_refs, inspect_site_pr_snapshot, repository_identity


def cheap_metadata(queue, source, sha, base):
    """Use only the already-installed pinned tool, never candidate build scripts."""
    required = set()
    for rev in (base, sha):
        rc, text, _ = queue.run(['git','-C',str(source),'show',rev+':.aien/always'])
        if not rc:
            required.update(line.split('#',1)[0].strip() for line in text.splitlines())
    if 'CRUMB-VERIFY' not in required:
        return True, 'legacy crumb gate not required'
    rc, text, _ = queue.run(['git','-C',str(source),'show',sha+':crumb.lock'])
    pin = text.strip()
    if rc or not re.fullmatch(r'[0-9a-f]{40}',pin):
        return False, 'legacy crumb preflight needs exact crumb.lock; owner must qualify tool'
    tool = queue.state / 'crumb-tools' / pin / 'crumb'
    if not tool.is_file() or not tool.stat().st_mode & 0o111:
        return False, 'pinned crumb tool not installed; no fallback or candidate build allowed'
    with tempfile.TemporaryDirectory(prefix='admission-',dir=queue.root) as tmp:
        checkout = str(Path(tmp)/'source')
        rc, out, err = queue.run(['git','-C',str(source),'-c','core.hooksPath=/dev/null',
            'worktree','add','--detach',checkout,sha],timeout=120)
        if rc:
            return False, 'metadata worktree unavailable: '+(err or out)[-150:]
        try:
            rc, out, err = queue.run([str(tool),'verify','.'],cwd=checkout,timeout=120)
            if rc:
                return False, 'pinned crumb verification failed: '+(err or out)[-500:]
        finally:
            queue.run(['git','-C',str(source),'worktree','remove','--force',checkout],timeout=120)
    return True, 'pinned crumb verification PASS'


def admit_plan(plan):
    if not isinstance(plan, dict) or type(plan.get('code')) is not int or not isinstance(plan.get('output'), str):
        return False, 'planner result missing or malformed'
    text = plan['output']
    # The existing planner emits text, not JSON. Accept only its explicit terminal
    # verdict, and never accept hidden unmapped paths even with an inconsistent rc.
    for section in ('UNMAPPED', 'REMOVED', 'Removed gates'):
        matches = re.findall(r'^'+re.escape(section)+r':[^\n]*\n(.*?)(?=^[A-Za-z][A-Za-z -]*:|\Z)', text, re.M | re.S)
        if len(matches) > 1 or any(body.strip() != '(none)' for body in matches):
            return False, 'planner has nonempty or malformed ' + section + ' section'
    if 'GATE_REMOVED' in text:
        return False, 'planner has GATE_REMOVED'
    verdicts = re.findall(r'^Plan verdict: (.+)$', text, re.M)
    if len(verdicts) != 1:
        return False, 'planner terminal verdict missing or ambiguous'
    verdict = verdicts[0].strip()
    if plan['code'] == 0 and verdict == 'RUNNABLE':
        return True, 'planner RUNNABLE'
    if plan['code'] == 4 and verdict == 'BLOCKED_OPERATOR PIPELINE_CONFIG_CHANGED':
        return True, 'operator admission required for pipeline configuration'
    return False, 'planner refuses candidate: ' + verdict


def intake_ack_valid(queue, entry, ack, path):
    """Audit local trusted intake provenance, not an authentication boundary against the OS user."""
    if path.is_symlink() or not isinstance(ack, dict) or ack.get('ack_path') != str(path):
        return False
    if (ack.get('schema') != 'renda-intake-ack/1' or ack.get('repository') != queue.full
            or ack.get('state') != 'ACCEPTED_LOCAL_NOT_PUBLISHED' or ack.get('remote_eligible') is not False):
        return False
    try:
        source = Path(ack['input'])
        if not source.is_absolute() or source.stat().st_size > 2_000_000:
            return False
        raw = source.read_bytes()
        if digest(raw) != ack['provenance_sha256']:
            return False
        rows = json.loads(raw)
        if not isinstance(rows, list) or not 0 < len(rows) <= 200:
            return False
        keys, items = [], []
        for row in rows:
            if (not isinstance(row, dict) or row.get('repository') != queue.full
                    or not isinstance(row.get('ref'), str)
                    or any(not re.fullmatch(r'[0-9a-f]{40}',str(row.get(k))) for k in ('sha','base'))):
                return False
            keys.append(f"{queue.full}:ready:{row['ref']}@{row['sha']}")
            items.append(dict(ref=row['ref'],head=row['sha'],base=row['base'],number=None,remote_observed_at=None))
        expected_name = digest(json.dumps([queue.full,keys],sort_keys=True).encode())+'.json'
        return (path.name == expected_name and ack.get('items') == items
                and entry['key'] in keys and entry.get('provenance') == f'local READY {source} sha256:{digest(raw)}')
    except (OSError, ValueError, TypeError, KeyError):
        return False


def assessment(queue, entry, model, policy):
    """Validate exact intake and review without network, mutation of source, or tests."""
    full = queue.full
    if entry.get('kind') != 'ready' or not isinstance(model, str) or not model:
        return None, 'local READY identity or review model unavailable'
    sha, base, ref = (entry.get(k) for k in ('sha', 'base', 'ref'))
    if any(not re.fullmatch(r'[0-9a-f]{40}', str(x)) for x in (sha, base)):
        return None, 'malformed source pins'
    if not isinstance(ref, str) or ref.startswith(('refs/', '-')) or queue.run(['git', 'check-ref-format', '--branch', ref])[0]:
        return None, 'invalid branch ref; submit a literal branch name'
    key = f'{full}:ready:{ref}@{sha}'
    if entry.get('key') != key or entry.get('number') is not None or entry.get('remote_observed_at') is not None:
        return None, 'READY queue identity mismatch'
    # Existing generic batch intake has one aggregate acknowledgement; retain it
    # unchanged and find the exact item rather than inventing another authority.
    acknowledged = False
    for path in (queue.root / 'acks').glob('*.json'):
        ack = read(path)
        if not intake_ack_valid(queue, entry, ack, path):
            continue
        for item in ack.get('items', []):
            if (isinstance(item, dict) and item.get('ref') == ref and item.get('head') == sha
                    and item.get('base') == base and item.get('number') is None
                    and item.get('remote_observed_at') is None):
                acknowledged = True
                break
        if acknowledged:
            break
    if not acknowledged:
        return None, 'exact local intake acknowledgement missing'
    plan_path = queue.root / 'local' / queue.repo / (digest(key.encode()) + '.plan.json')
    record = read(plan_path)
    if (not isinstance(record, dict) or record.get('schema') != 'renda-local-plan/1'
            or record.get('key') != key or record.get('commit') != sha or record.get('base') != base
            or record.get('model') != model or record.get('policy') != policy
            or not isinstance(record.get('review'), dict) or record['review'].get('decision') is not True):
        return None, 'exact complete local review missing or refused'
    source, repository = queue.source(sha, base)
    if source is None or record.get('repository') != repository:
        return None, 'trusted source repository unavailable or changed'
    rc, diff, _ = queue.run(['git', '-C', str(source), 'diff', '--no-ext-diff', '--no-textconv', base + '...' + sha, '--'], timeout=120)
    if rc or len(diff) > 60000 or digest(diff.encode()) != record.get('diff_digest'):
        return None, 'exact full source diff changed or unavailable'
    ok, note = admit_plan(record.get('plan'))
    if not ok:
        return None, note
    return (source, repository, record), 'exact local assessment'


def publish_candidate(repo, queue, entry, model, policy, ctx):
    """Create/reuse a PR only after signed qualification; never merge or post PASS."""
    full = queue.full
    tag = f"{entry.get('ref')}@{str(entry.get('sha', ''))[:12]}"
    def refuse(note):
        return False, f'{tag} publication held: {note}'
    def owned(number=None):
        return (ctx['lane_owned'](repo) and not Path(ctx['PAUSE']).exists()
                and not ctx['held'](repo, number))
    if repository_identity(repo) != full or not owned():
        return refuse('repository ownership, hold or pause changed')
    local, note = assessment(queue, entry, model, policy)
    if local is None:
        return refuse(note)
    source, repository, record = local
    sha, base, ref = entry['sha'], entry['base'], entry['ref']
    def qualified():
        retained = ctx['operator_verdict'](repo, sha)
        return (not (retained and retained[0] in ('FAIL', 'HOLD'))
                and queue.receipt(source, repository, sha, base).get('eligible') is True)
    if not qualified():
        return refuse('verified eligible exact-source receipt required; FAIL is never retried')
    def refs_match():
        rc, branch_out, _ = queue.run(['git','-C',str(source),'ls-remote','origin','refs/heads/'+ref],timeout=60)
        rc2, base_out, _ = queue.run(['git','-C',str(source),'ls-remote','origin','refs/heads/main'],timeout=60)
        return not (rc or rc2) and exact_remote_refs(branch_out, base_out, ref, sha, base)
    if not refs_match():
        return refuse('remote branch or main moved from exact READY pins')
    rc, _, _ = queue.run(['git','-C',str(source),'merge-tree','--write-tree',base,sha],timeout=60)
    if rc:
        return refuse('candidate conflicts with pinned main')
    rc, out, err = queue.run(['gh','pr','list','-R',full,'--head',ref,'--state','all','--json',
        'number,state,headRefName,headRefOid,baseRefName,baseRefOid'],timeout=120)
    if rc:
        return refuse('existing PR lookup unavailable: '+(err or out)[-200:])
    try:
        inspection = inspect_site_pr_snapshot(json.loads(out),ref,sha,base)
    except (ValueError, TypeError):
        return refuse('existing PR response malformed')
    if inspection['action'] == 'refuse':
        return refuse(inspection['reason'])
    number = inspection.get('number')
    if not owned(number) or not qualified() or not refs_match():
        return refuse('ownership, qualification or exact remote refs changed before write')
    if inspection['action'] == 'create':
        body = (f'Renda qualified local READY intake\n\nRepository: {full}\nBranch: {ref}\n'
                f'Head: {sha}\nBase: main@{base}\nFull local diff reviewed by: {model}\n'
                'Signed exact-source Spark receipt verified. Hosted verification and fresh required checks '
                'remain mandatory before merge. No issue closure is claimed.')
        rc, out, err = queue.run(['gh','pr','create','-R',full,'--head',ref,'--base','main',
            '--title',('Renda intake: '+ref)[:240],'--body',body],timeout=120)
        if rc:
            return refuse('guarded PR create refused or ambiguous; reconcile branch next pass: '+(err or out)[-200:])
        match = re.search(r'https://github\.com/'+re.escape(full)+r'/pull/(\d+)(?:\s|$)',out)
        if not match:
            return refuse('create response ambiguous; exact branch reconciliation required')
        number = int(match[1])
    rc, out, err = queue.run(['gh','pr','view',str(number),'-R',full,'--json',
        'number,state,headRefName,headRefOid,baseRefName,baseRefOid,mergeStateStatus'],timeout=120)
    try:
        pr = json.loads(out) if not rc else None
    except (ValueError, TypeError):
        pr = None
    check = inspect_site_pr_snapshot([pr],ref,sha,base)
    if check['action'] != 'reuse' or check.get('number') != number:
        return refuse('created/reused PR read-back missing or moved; publication not recorded')
    path = queue.root / 'published' / repo / (digest(entry['key'].encode())+'.json')
    previous = read(path)
    if (isinstance(previous, dict) and previous.get('repository') == full
            and previous.get('head') == sha and previous.get('base') == base
            and previous.get('number') == number):
        return True, f'{tag} already published as PR #{number}; exact head/base reconfirmed; merge checks pending'
    save(path,dict(schema='renda-ready-publication/1',repository=full,key=entry['key'],head=sha,base=base,
        ref=ref,number=number,state='PR_OPEN_QUALIFIED_AWAITING_HOSTED_AND_MERGE_CHECKS',
        url=f'https://github.com/{full}/pull/{number}',remote_observed_at=time.time(),merge_authorized=False))
    return True,f'{tag} published as PR #{number}; exact head/base read back; merge checks remain mandatory'


def report_transitions(state, name, results, emit, now=None):
    now = time.time() if now is None else now
    path = Path(state) / ('renda-report-'+name+'.json')
    old = read(path) or {}
    rows = []
    for result in results:
        for kind in ('local','sync_wait','merged','rebased','rerun','operator','published','blocked','waiting'):
            rows += [f"{result['repo']} {kind.upper()} {item}" for item in result.get(kind,[])]
    previous = set(old.get('rows',[]))
    for row in rows:
        if row not in previous or any(' '+kind+' ' in row for kind in ('MERGED','REBASED','RERUN','OPERATOR','PUBLISHED')):
            emit(row)
    for row in sorted(previous-set(rows)):
        emit('TRANSITION no longer reported: '+row)
    heartbeat = old.get('heartbeat',0)
    if now-heartbeat >= 900:
        emit(f'HEARTBEAT {len(rows)} current queue observations; unchanged details suppressed')
        heartbeat = now
    save(path,dict(schema='renda-report/1',rows=rows,heartbeat=heartbeat))
