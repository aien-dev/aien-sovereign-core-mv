"""Durable local work, never authority for remote effects. Standard library only."""
import contextlib
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

SCHEMA = 'renda-local-queue/1'
TTL = 300
MAX_ENTRIES = 800
SHA = re.compile(r'[0-9a-f]{40}')
FIELDS = ('number,title,isDraft,headRefName,headRefOid,baseRefName,baseRefOid,'
          'mergeStateStatus,state,statusCheckRollup,createdAt')
PR_FIELDS = '''number title isDraft headRefName headRefOid baseRefName baseRefOid
mergeStateStatus state createdAt commits(last:1){nodes{commit{oid statusCheckRollup{
contexts(first:100){pageInfo{hasNextPage} nodes{__typename
... on StatusContext {context state description}
... on CheckRun {name status conclusion}}}}}}}'''


def graphql_args(full, number=None, cursor=None):
    owner, name = full.split('/')
    if number is None:
        declaration = '$owner:String!,$name:String!,$cursor:String'
        selection = 'pullRequests(first:100,states:OPEN,after:$cursor){pageInfo{hasNextPage endCursor} nodes{' + PR_FIELDS + '}}'
    else:
        declaration = '$owner:String!,$name:String!,$number:Int!'
        selection = 'pullRequest(number:$number){' + PR_FIELDS + '}'
    query = 'query(' + declaration + '){repository(owner:$owner,name:$name){' + selection + '}}'
    args = ['api', 'graphql', '-f', 'query=' + query, '-f', 'owner=' + owner, '-f', 'name=' + name]
    if number is not None:
        args += ['-F', 'number=' + str(number)]
    elif cursor:
        args += ['-f', 'cursor=' + cursor]
    return args


def decode_pr(node):
    p = dict(node)
    commits = p.pop('commits')['nodes']
    if len(commits) != 1:
        raise ValueError('missing exact-head commit context')
    if commits[0]['commit']['oid'] != p['headRefOid']:
        raise ValueError('check contexts are not for exact head')
    rollup = commits[0]['commit']['statusCheckRollup']
    if rollup is None:
        p['statusCheckRollup'] = []
    else:
        contexts = rollup['contexts']
        if contexts['pageInfo']['hasNextPage']:
            raise ValueError('truncated required contexts')
        p['statusCheckRollup'] = contexts['nodes']
    if not valid_pr(p):
        raise ValueError('invalid PR identity')
    return p


def decode_one(value):
    if not isinstance(value, dict) or value.get('errors'):
        raise ValueError('GraphQL errors or invalid response')
    return decode_pr(value['data']['repository']['pullRequest'])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic(path, data):
    """Same-directory replace plus fsync: interruption preserves old or new bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save(path, value):
    atomic(path, (json.dumps(value, sort_keys=True, indent=2) + '\n').encode())


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def valid_pr(p):
    return (isinstance(p, dict) and type(p.get('number')) is int and p['number'] > 0
            and isinstance(p.get('title'), str) and type(p.get('isDraft')) is bool
            and isinstance(p.get('headRefName'), str) and bool(p['headRefName'])
            and SHA.fullmatch(str(p.get('headRefOid', ''))) is not None
            and SHA.fullmatch(str(p.get('baseRefOid', ''))) is not None
            and isinstance(p.get('baseRefName'), str) and bool(p['baseRefName'])
            and p.get('state') in ('OPEN', 'CLOSED', 'MERGED')
            and isinstance(p.get('mergeStateStatus'), str)
            and isinstance(p.get('statusCheckRollup'), list))


def status(p, context='AIEN Verification (Spark)'):
    for s in p.get('statusCheckRollup', []):
        if isinstance(s, dict) and s.get('context') == context:
            state = str(s.get('state', 'none')).lower()
            return state if state in ('success', 'failure', 'error', 'pending') else 'none', str(s.get('description') or '')
    return 'none', ''


def retry_delay(rc, diagnostic, failures, now):
    delay = min(3600, TTL * 2 ** min(failures - 1, 4))
    if rc == 98:
        delay = max(delay, 900)
    match = re.search(r'Retry-After:\s*([^\r\n]+)', diagnostic, re.I)
    if match:
        value = match[1].strip()
        if value.isdigit():
            delay = max(delay, min(int(value), 86400))
        else:
            import email.utils
            try:
                delay = max(delay, min(86400, email.utils.parsedate_to_datetime(value).timestamp() - now))
            except (ValueError, TypeError, OverflowError):
                pass
    reset = re.search(r'resets at (\d\d):(\d\d)Z', diagnostic)
    if reset:
        dt = datetime.datetime.fromtimestamp(now, datetime.timezone.utc)
        target = dt.replace(hour=int(reset[1]), minute=int(reset[2]), second=0, microsecond=0).timestamp()
        if target < now:
            target += 86400
        delay = max(delay, target - now + 30)
    return delay


def origin_slug(url):
    match = re.fullmatch(r'(?:https://github.com/|git@github.com:)([^/]+/[^/]+?)(?:\.git)?', url.strip())
    return match[1] if match else None


class Queue:
    def __init__(self, state, repo, full, run, clones, work):
        if not re.fullmatch(r'[A-Za-z0-9._-]+', repo) or repo in ('.', '..'):
            raise ValueError('invalid queue repository')
        self.state, self.repo, self.full = Path(state), repo, full
        self.run, self.clones, self.work = run, Path(clones), Path(work)
        self.root = self.state / 'renda-queue'
        self.path = self.root / (repo + '.json')
        self.root.mkdir(parents=True, exist_ok=True)
        self.sync_diagnostic = ''

    def empty(self):
        return dict(schema=SCHEMA, repository=self.full, observed_at=None,
                    next_sync=0, failures=0, diagnostic='', entries=[])

    def load(self):
        d = read(self.path)
        if not isinstance(d, dict) or d.get('schema') != SCHEMA or d.get('repository') != self.full:
            return self.empty()
        try:
            assert type(d['next_sync']) in (int, float) and math.isfinite(d['next_sync'])
            assert type(d['failures']) is int and 0 <= d['failures'] <= 1000000
            assert isinstance(d['entries'], list) and len(d['entries']) <= MAX_ENTRIES
            for e in d['entries']:
                assert isinstance(e, dict) and e.get('kind') in ('pr', 'ready')
                assert isinstance(e.get('provenance'), str) and e['provenance']
                if e['kind'] == 'pr':
                    assert valid_pr(e['pr']) and e['key'] == self.key(e['pr'])
                    assert type(e['remote_observed_at']) in (int, float)
                else:
                    assert e.get('number') is None and e.get('remote_observed_at') is None
                    assert SHA.fullmatch(e['sha']) and SHA.fullmatch(e['base'])
        except (KeyError, AssertionError, TypeError):
            return self.empty()
        return d

    def key(self, p):
        return f"{self.full}#{p['number']}@{p['headRefOid']}"

    @contextlib.contextmanager
    def lock(self, suffix='sync'):
        with open(self.root / (self.repo + '.' + suffix + '.lock'), 'a') as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            yield True

    def sync(self, now=None):
        """At most one batch read per repo/TTL across desks, including failures."""
        now = time.time() if now is None else now
        with self.lock() as owned:
            d = self.load()
            if not owned or now < d['next_sync']:
                self.sync_diagnostic = d.get('diagnostic') or ('sync owned elsewhere' if not owned else 'sync TTL')
                return d, False
            # Reserve the window durably before transport. A killed sync cannot cause a retry storm.
            d['next_sync'] = now + TTL
            save(self.path, d)
            rc, diagnostic = 0, ''
            try:
                prs, cursor = [], None
                for page in range(2):
                    rc, out, err = self.run(['gh'] + graphql_args(self.full, cursor=cursor), timeout=120)
                    diagnostic = err or out
                    if rc:
                        raise ValueError(f'gh exit {rc}: {diagnostic}')
                    value = json.loads(out)
                    if not isinstance(value, dict) or value.get('errors'):
                        raise ValueError('GraphQL errors or invalid response')
                    connection = value['data']['repository']['pullRequests']
                    prs.extend(decode_pr(node) for node in connection['nodes'])
                    if not connection['pageInfo']['hasNextPage']:
                        break
                    cursor = connection['pageInfo']['endCursor']
                    if not isinstance(cursor, str) or not cursor or page == 1:
                        raise ValueError('truncated PR snapshot')
                if any(p['state'] != 'OPEN' for p in prs) or len({p['number'] for p in prs}) != len(prs):
                    raise ValueError('duplicate or truncated PR snapshot, publication held')
            except (ValueError, TypeError, KeyError) as exc:
                d['failures'] += 1
                d['diagnostic'] = str(exc)[-4000:]
                d['next_sync'] = now + retry_delay(rc, d['diagnostic'], d['failures'], now)
                save(self.path, d)
                self.sync_diagnostic = d['diagnostic']
                return d, False
            # An absent old head is retained as history, never kept active after a successful sync.
            for e in d['entries']:
                if e['kind'] == 'pr':
                    e['active'] = False
            by_key = {e['key']: e for e in d['entries']}
            for p in prs:
                key = self.key(p)
                by_key[key] = dict(key=key, kind='pr', pr=p, active=True,
                    provenance='gh api graphql PR+contexts batch via guarded transport', remote_observed_at=now)
            entries = list(by_key.values())
            if len(entries) > MAX_ENTRIES:
                d = self.load()
                d['diagnostic'] = 'queue capacity reached; retained work preserved, coordinator archive required'
                d['failures'] += 1
                d['next_sync'] = now + TTL
                save(self.path, d)
                self.sync_diagnostic = d['diagnostic']
                return d, False
            entries.sort(key=lambda e: (bool(e.get('active')), e.get('remote_observed_at') or 0))
            d.update(entries=entries, observed_at=now, next_sync=now + TTL, failures=0, diagnostic='')
            save(self.path, d)
            return d, True

    def seed_ready(self, path):
        """Explicit local READY records never become remote PRs. Input bytes retained by hash."""
        raw = Path(path).read_bytes()
        rows = json.loads(raw)
        if not isinstance(rows, list) or len(rows) > 200:
            raise ValueError('invalid READY list')
        entries = []
        for e in rows:
            if (not isinstance(e, dict) or e.get('repository') != self.full
                    or not SHA.fullmatch(str(e.get('sha', '')))
                    or not SHA.fullmatch(str(e.get('base', '')))
                    or not isinstance(e.get('ref'), str) or not e['ref']):
                raise ValueError('invalid READY identity')
            entries.append(dict(key=f"{self.full}:ready:{e['ref']}@{e['sha']}", kind='ready',
                sha=e['sha'], base=e['base'], ref=e['ref'], number=None, remote_observed_at=None,
                provenance=f'local READY {Path(path).resolve()} sha256:{digest(raw)}'))
        with self.lock() as owned:
            if not owned:
                raise RuntimeError('sync busy')
            d = self.load()
            keyed = {e['key']: e for e in d['entries']}
            keyed.update({e['key']: e for e in entries})
            if len(keyed) > MAX_ENTRIES:
                raise ValueError('queue capacity exceeded')
            d['entries'] = list(keyed.values())
            save(self.path, d)

    def source(self, sha, base):
        for source in (self.clones / self.repo, self.work / self.repo):
            if not source.exists():
                continue
            rc, url, _ = self.run(['git', '-C', str(source), 'remote', 'get-url', 'origin'])
            slug = origin_slug(url) if rc == 0 else None
            # Source identities come from configured trusted clones, never from receipt JSON.
            if slug not in (self.full, 'aien-dev/' + self.repo):
                continue
            if all(self.run(['git', '-C', str(source), 'cat-file', '-e', rev + '^{commit}'])[0] == 0 for rev in (sha, base)):
                return source, 'github.com/' + slug
        return None, None

    def receipt(self, source, repository, sha, base):
        path = self.state / 'bundles' / self.repo / sha / 'bundle.json'
        raw = path.read_bytes() if path.exists() else b''
        try:
            b = json.loads(raw)
            if (not isinstance(b, dict) or b.get('schema') != 'aien-verification-bundle/1'
                    or b.get('repository') != repository or b.get('commit') != sha):
                raise ValueError('receipt identity mismatch')
        except (ValueError, TypeError):
            return dict(verified=False, eligible=False, diagnostic='missing, corrupt or wrong-identity receipt', digest=digest(raw))
        rc, keys, _ = self.run(['git', '-C', str(source), 'show', base + ':.aien/verify.pub'])
        keys = [line.split('#')[0].strip() for line in keys.splitlines()] if rc == 0 else []
        keys = [key for key in keys if re.fullmatch(r'[0-9a-fA-F]{64}', key)]
        _, required, _ = self.run(['git', '-C', str(source), 'show', base + ':.aien/always'])
        required = ','.join(line.strip() for line in required.splitlines() if line.strip() and not line.lstrip().startswith('#'))
        binary = os.environ.get('AIEN_TEST_BIN', str(Path.home() / 'workspace/pipeline/target-sc/release/aien-test'))
        codes = []
        # Verify the same immutable bytes inspected above, even if the worker replaces its bundle concurrently.
        with tempfile.TemporaryDirectory(prefix='receipt-', dir=self.root) as tmp:
            snapshot = Path(tmp) / 'bundle.json'
            snapshot.write_bytes(raw)
            for key in keys:
                cmd = [binary, 'verify-bundle', str(snapshot), '--repo', repository, '--commit', sha, '--pubkey', key]
                if required:
                    cmd += ['--require', required]
                code, _, _ = self.run(cmd)
                codes.append(code)
        code = next((c for c in (0, 1, 2) if c in codes), 3)
        plan = b.get('plan')
        good_plan = (isinstance(plan, dict) and plan.get('verdict') == 'PASS'
                     and not plan.get('blocked') and not plan.get('unmapped') and not plan.get('removed'))
        return dict(verified=code in (0, 1, 2), eligible=code == 0 and good_plan,
            verdict=b.get('verdict'), plan=plan, verifier_codes=codes, digest=digest(raw),
            diagnostic='existing verify-bundle; diagnostic only, fresh remote gates still required')

    def progress(self, doc, model, ask, policy, budget=2):
        """Read pinned local objects; run only the existing planner, never a gate/judge."""
        output = []
        with self.lock('local') as owned:
            if not owned:
                return ['local work owned elsewhere']
            count = 0
            for e in doc['entries']:
                if e['kind'] == 'pr':
                    p = e['pr']
                    if not e.get('active', True) or p['state'] != 'OPEN' or p['isDraft']:
                        continue
                    sha, base = p['headRefOid'], p['baseRefOid']
                else:
                    sha, base = e['sha'], e['base']
                prefix = self.root / 'local' / self.repo / digest(e['key'].encode())
                source, repository = self.source(sha, base)
                if source is None:
                    output.append(f"{e['key']} source unavailable (no fetch)")
                    continue
                rc, diff, err = self.run(['git', '-C', str(source), 'diff', '--no-ext-diff', '--no-textconv', base + '...' + sha, '--'], timeout=120)
                if rc:
                    output.append(f"{e['key']} diff unavailable: {err[-200:]}")
                    continue
                receipt = self.receipt(source, repository, sha, base)
                identity = digest(json.dumps([e['key'], base, digest(diff.encode()), model, policy]).encode())
                plan_path = Path(str(prefix) + '.plan.json')
                old = read(plan_path)
                if isinstance(old, dict) and old.get('identity') == identity:
                    # Receipt and sync eligibility are deliberately not cached with review.
                    old['receipt'] = receipt
                    save(plan_path, old)
                    output.append(f"{e['key']} local review retained ({old['review']['decision']}); receipt={receipt.get('verdict', 'unverified')}; waiting-for-sync")
                    continue
                if count >= budget:
                    continue
                count += 1
                atomic(str(prefix) + '.diff', diff.encode())
                plan = self.host_plan(source, repository, sha, base)
                decision, note = None, 'local model unavailable'
                if len(diff) > 60000:
                    note = 'review held: diff exceeds 60000 characters; full diff retained locally'
                elif model:
                    text, mid = ask('Review this pinned LOCAL diff. Qualification is pending; this review grants no merge authority. '
                        'Report concrete defects, secrets, unrelated changes or destructive edits. '
                        'Reply JSON {"merge": true or false, "note": "one line"}.\n'
                        f'Repository {repository}, commit {sha}, base {base}\n{diff[:60000]}')
                    try:
                        if mid != model:
                            raise ValueError('local model changed during review')
                        parsed = json.loads(re.search(r'\{.*\}', text or '', re.S)[0])
                        if type(parsed.get('merge')) is not bool:
                            raise ValueError('non-boolean review')
                        decision, note = parsed['merge'], str(parsed.get('note', ''))[:300]
                    except (ValueError, TypeError, KeyError):
                        note = 'unparsable local review: ' + str(text or mid)[:200]
                record = dict(schema='renda-local-plan/1', identity=identity, key=e['key'], repository=repository,
                    commit=sha, base=base, diff_digest=digest(diff.encode()), provenance=e['provenance'],
                    remote_observed_at=e['remote_observed_at'], remote_eligible=False,
                    model=model, policy=policy, review=dict(decision=decision, note=note),
                    plan=plan, receipt=receipt, created_at=time.time())
                save(plan_path, record)
                # This is an idempotent local intent for the existing desk, never the pipeline's signed outbox.
                save(self.root / 'pending' / self.repo / (digest(e['key'].encode()) + '.json'), dict(
                    schema='renda-local-intent/1', key=e['key'], identity=identity,
                    state='waiting-for-sync', remote_eligible=False, plan=str(plan_path)))
                output.append(f"{e['key']} local diff={digest(diff.encode())[:12]} review={decision}; host plan exit={plan['code']}; receipt={receipt.get('verdict', 'unverified')}; waiting-for-sync")
        return output

    def host_plan(self, source, repository, sha, base):
        binary = os.environ.get('AIEN_TEST_BIN', str(Path.home() / 'workspace/pipeline/target-sc/release/aien-test'))
        with tempfile.TemporaryDirectory(prefix='plan-' + self.repo + '-', dir=self.root) as tmp:
            wt = str(Path(tmp) / 'checkout')
            rc, out, err = self.run(['git', '-C', str(source), '-c', 'core.hooksPath=/dev/null',
                'worktree', 'add', '--detach', wt, sha], timeout=120)
            if rc:
                return dict(code=rc, output='HOST_PLAN unavailable: ' + (err or out)[-1000:])
            try:
                rc, out, err = self.run([binary, 'plan', '--base', base, '--repo', repository,
                    '--evidence-dir', str(self.state / 'evidence')], cwd=wt, timeout=120)
                return dict(code=rc, output=(out + err)[-20000:], gates_executed=False)
            finally:
                self.run(['git', '-C', str(source), 'worktree', 'remove', '--force', wt], timeout=120)

    def review(self, p, model, policy):
        prefix = self.root / 'local' / self.repo / digest(self.key(p).encode())
        d = read(str(prefix) + '.plan.json')
        if not isinstance(d, dict) or d.get('commit') != p['headRefOid'] or d.get('base') != p['baseRefOid'] or d.get('model') != model or d.get('policy') != policy:
            return None, 'pinned local review pending'
        return d['review']['decision'], d['review']['note']
