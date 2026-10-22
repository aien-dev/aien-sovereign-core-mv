#!/usr/bin/env python3
"""Read-only projection of Renda's queue. No network, execution or authorization."""
import argparse, collections, datetime, hashlib, json, re
from pathlib import Path
STAGES=('Actionable','Running','Blocked','Awaiting publication','Untriaged','Done')
def load(p):
    value=json.loads(p.read_text())
    if not isinstance(value,dict):raise ValueError('expected object')
    return value
def collect(state):
    root=Path(state)/'renda-queue'; rows=[];warnings=[];seen=set()
    if not root.is_dir():return [],['Local queue directory not present. READY log entries are NOT acknowledged intake.']
    for path in sorted(root.glob('*.json')):
        try:
            doc=load(path)
            if doc.get('schema')!='renda-local-queue/1' or not isinstance(doc.get('entries'),list):raise ValueError('unsupported queue schema')
        except (OSError,ValueError) as e:warnings.append(f'{path.name}: {e}');continue
        repo=doc.get('repository','unknown')
        for i,e in enumerate(doc['entries']):
            if not isinstance(e,dict):warnings.append(f'{path.name} row {i}: invalid entry');continue
            key=e.get('key')
            if not isinstance(key,str) or not key:warnings.append(f'{path.name} row {i}: missing item ID');continue
            if (repo,key) in seen:warnings.append(f'duplicate item: {key}');continue
            seen.add((repo,key));p=e.get('pr') or {};kind=e.get('kind')
            sha=p.get('headRefOid') if kind=='pr' else e.get('sha');base=p.get('baseRefOid') if kind=='pr' else e.get('base')
            row=dict(id=key,repo=repo,ref=p.get('headRefName',e.get('ref','unknown')),sha=sha,base=base,stage='Actionable',owner=e.get('owner','unrecorded'),dependencies=e.get('dependencies','unrecorded'),blocker='',next_action='Locate pinned source; plan and review locally',last_progress=None,publication='not authorized',merge_authorized=False,provenance=e.get('provenance','unknown'))
            if kind not in ('pr','ready') or not all(isinstance(x,str) and re.fullmatch('[0-9a-f]{40}',x) for x in (sha,base)):
                row.update(stage='Untriaged',blocker='Invalid or incomplete source identity',next_action='Reconcile original intake; do not guess a SHA');rows.append(row);continue
            planpath=root/'local'/path.stem/(hashlib.sha256(key.encode()).hexdigest()+'.plan.json')
            plan=None
            if planpath.exists():
                try:
                    plan=load(planpath)
                    if plan.get('schema')!='renda-local-plan/1' or plan.get('key')!=key or plan.get('commit')!=sha or plan.get('base')!=base:raise ValueError('stale or mismatched plan')
                    if not all(isinstance(plan.get(k),dict) for k in ('review','receipt','plan')):raise ValueError('incomplete plan')
                except (OSError,ValueError) as ex:warnings.append(f'{key}: {ex}');plan=None
            if plan:
                row['last_progress']=plan.get('created_at');review=plan['review'];receipt=plan['receipt'];hp=plan['plan']
                if review.get('decision') is not True:
                    row.update(stage='Blocked',blocker=review.get('note') or 'Review pending or refused',next_action='Resolve review blocker; preserve full-diff coverage')
                elif receipt.get('verified') is not True or receipt.get('eligible') is not True:
                    row.update(stage='Blocked',blocker='Receipt unqualified: '+str(receipt.get('verdict','missing')),next_action='Resolve exact gate or obtain verified qualification; do not retry FAIL unchanged')
                elif hp.get('code')!=0:
                    row.update(stage='Blocked',blocker='Local planner did not succeed',next_action='Inspect planner output')
                else:
                    row.update(stage='Awaiting publication',next_action='Revalidate current review policy, source and remote gates before publication',publication='pending fresh validation')
            if p.get('isDraft') is True:row.update(stage='Blocked',blocker='Imported draft',next_action='Complete author work; remote metadata may be stale')
            if e.get('active') is False or p.get('state') in ('CLOSED','MERGED'):
                row.update(stage='Untriaged',blocker='Imported inactive/closed state, not proof of local completion',next_action='Reconcile completion and supersession evidence')
            rows.append(row)
    return sorted(rows,key=lambda r:(STAGES.index(r['stage']),str(r['repo']),r['id'])),warnings
def esc(v):return str(v if v is not None else 'unrecorded').replace('\\','\\\\').replace('|','\\|').replace('\n',' ').replace('\r',' ').replace('<','&lt;').replace('>','&gt;')
def render(rows,warnings):
    counts=collections.Counter(r['stage'] for r in rows)
    out=['# Local desk board', '', 'Read-only snapshot. Actionable means a local next step, NOT merge-ready. No remote calls.', 'Generated: '+datetime.datetime.now(datetime.timezone.utc).isoformat(), '',f'Items: {len(rows)}. '+', '.join(f'{s}: {counts[s]}' for s in STAGES), '']
    if warnings:out+=['## Intake warnings']+['- '+esc(w) for w in warnings]+['']
    for stage in STAGES:
        group=[r for r in rows if r['stage']==stage]
        if not group:continue
        out+=['## '+stage,'','| Item / source | Owner | Dependencies | Blocker | Next action | Last progress |','|---|---|---|---|---|---|']
        for r in group:
            source=f"{r['repo']} {r['ref']} @ {r['sha']}"
            values=(source,r['owner'],r['dependencies'],r['blocker'] or 'none recorded',r['next_action'],r['last_progress'])
            out.append('| '+' | '.join(esc(v) for v in values)+' |')
        out.append('')
    out+=['## Interpretation','Unknown owners, dependencies and progress remain explicit. This view does not claim running work without a lease, does not authorize merges, and never infers completion from a PASS headline or a closed imported PR. Full IDs and provenance are available with --json.']
    return '\n'.join(out)+'\n'
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--state',type=Path,default=Path.home()/'.local/state/aien-local-ci');ap.add_argument('--json',action='store_true');a=ap.parse_args();rows,warnings=collect(a.state)
    print(json.dumps({'schema':'renda-board/1','items':rows,'warnings':warnings},indent=2) if a.json else render(rows,warnings))
if __name__=='__main__':main()
