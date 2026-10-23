"""Promote a qualified, explicit successor to an existing PR, never create another."""
import json
import re
import time


def valid_target(target):
    return (isinstance(target, dict) and set(target) in ({'number','ref','expected_head'}, {'number','ref','expected_head','expected_base'})
        and type(target.get('number')) is int and target['number'] > 0
        and isinstance(target.get('ref'), str) and bool(target['ref'])
        and not target['ref'].startswith(('refs/','-'))
        and re.fullmatch(r'[0-9a-f]{40}', str(target.get('expected_head'))) is not None
        and ('expected_base' not in target or re.fullmatch(r'[0-9a-f]{40}', str(target['expected_base'])) is not None))


def promote_successor(repo, queue, entry, source, owned, qualified, refs_match):
    # Called only after assessment binds the entire input manifest, authenticated
    # full-source review, signed qualification, clean merge tree and fresh refs.
    from local_queue import digest, save
    full, sha, base = queue.full, entry['sha'], entry['base']
    target = entry.get('target')
    def refuse(note):
        return False, entry['key'] + ' successor held: ' + note
    if not valid_target(target):
        return refuse('invalid existing-PR target')
    number, ref, expected = target['number'], target['ref'], target['expected_head']
    if queue.run(['git','check-ref-format','--branch',ref])[0] or ref == 'main':
        return refuse('invalid target branch')
    expected_base = target.get('expected_base', base)
    endpoint = f'repos/{full}/pulls/{number}'
    def snapshot():
        rc, out, err = queue.run(['gh','api',endpoint],timeout=120)
        if rc:
            raise ValueError('PR read failed: '+(err or out)[-150:])
        p = json.loads(out)
        if (not isinstance(p,dict) or type(p.get('number')) is not int or p['number'] != number
                or p.get('state') != 'open' or p.get('draft') is not False
                or p.get('head',{}).get('ref') != ref
                or p.get('head',{}).get('repo',{}).get('full_name') != full
                or p.get('base',{}).get('ref') != 'main'
                or p.get('base',{}).get('repo',{}).get('full_name') != full
                or p.get('base',{}).get('sha') != (base if p.get('head',{}).get('sha') == sha else expected_base)):
            raise ValueError('target changed, draft, closed, forked or malformed')
        head = p['head']['sha']
        if head not in (expected, sha):
            raise ValueError('existing PR head moved from expected lease')
        return head
    try:
        current = snapshot()
        if not owned(number) or not qualified() or not refs_match():
            return refuse('ownership, qualification or source refs changed')
        already = current == sha
        if not already:
            # Check the actual branch as well as the PR snapshot. The push uses
            # an explicit old value so concurrent author work cannot be replaced.
            rc,out,_ = queue.run(['git','-C',str(source),'ls-remote','origin','refs/heads/'+ref],timeout=60)
            if rc or out.split() != [expected,'refs/heads/'+ref]:
                return refuse('target branch moved or unavailable')
            if snapshot() != expected or not owned(number) or not qualified() or not refs_match():
                return refuse('target or authority changed at write boundary')
            rc,out,err = queue.run(['git','-C',str(source),'push',
                '--force-with-lease=refs/heads/'+ref+':'+expected,'origin',
                sha+':refs/heads/'+ref],timeout=180)
            if rc:
                return refuse('leased push refused or ambiguous: '+(err or out)[-180:])
        if snapshot() != sha:
            return refuse('updated PR exact-head readback missing')
        rc,out,_ = queue.run(['git','-C',str(source),'ls-remote','origin','refs/heads/'+ref],timeout=60)
        if rc or out.split() != [sha,'refs/heads/'+ref]:
            return refuse('updated remote branch readback missing')
        path = queue.root/'published'/repo/(digest(entry['key'].encode())+'.json')
        save(path,dict(schema='renda-ready-publication/1',repository=full,key=entry['key'],
            head=sha,base=base,ref=ref,source_ref=entry['ref'],number=number,
            state='EXISTING_PR_ADVANCED_AWAITING_HOSTED_AND_MERGE_CHECKS',
            url=f'https://github.com/{full}/pull/{number}',remote_observed_at=time.time(),
            merge_authorized=False,expected_old_head=expected))
        return True, (entry['key'] + (f' already published as PR #{number}' if already else f' advanced existing PR #{number}')
            + '; exact leased head read back; hosted and merge checks remain mandatory')
    except (ValueError,TypeError,KeyError,AttributeError) as exc:
        return refuse(str(exc))
