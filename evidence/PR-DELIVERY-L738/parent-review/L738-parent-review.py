import pathlib, subprocess, tempfile, json, hashlib, os, re, sys, shutil
h=pathlib.Path.home(); w=h/'workspace/hermes-desk/renda-planner-L738'
sha='ce5cd74d68d7b766c5500a400c3e269866fda14b'; prefix='tools/aien-local-ci/renda/'
def git(*a):return subprocess.check_output(['git','-C',str(w),*a])
# Export the immutable reviewed commit even after evidence-only follow-ups.
assert git('cat-file','-t',sha).strip()==b'commit'
assert not git('status','--porcelain').strip()
out=pathlib.Path(tempfile.mkdtemp(prefix='L738-parent-review-',dir=h/'handoffs')); pkg=out/'package';pkg.mkdir()
def blob(name):return git('show',sha+':'+prefix+name)
manifest=json.loads(blob('release.json')); assert len(manifest['files'])==21
(pkg/'release.json').write_bytes(blob('release.json'))
for name,digest in manifest['files'].items():
 assert '/' not in name and hashlib.sha256(blob(name)).hexdigest()==digest
 p=pkg/name;p.write_bytes(blob(name));p.chmod(0o755 if name in ('renda-harness','renda-checked') else 0o644)
env=dict(os.environ,RENDA_CANDIDATE=str(pkg/'renda-harness'),PYTHONDONTWRITEBYTECODE='1')
rows=[]
for p in sorted(pkg.glob('test_*.py')):
 r=subprocess.run([sys.executable,str(p)],cwd=pkg,env=env,text=True,capture_output=True,timeout=180)
 text=r.stdout+r.stderr;(out/(p.stem+'.log')).write_text(text)
 counts=re.findall(r'Ran (\d+) tests?',text)
 rows.append(dict(module=p.name,exit=r.returncode,tests=sum(map(int,counts))))
assert len(rows)==12 and all(r['exit']==0 for r in rows),rows
# Independently show the inconsistent-rc0 counterexample kills the old tuple condition.
mutant=out/'tuple-mutant';shutil.copytree(pkg,mutant)
p=mutant/'delivery.py';text=p.read_text()
old="        admitted, _ = admit_plan(plan)\n        if not admitted or plan.get('code') != 0:"
new="        if not admit_plan(plan) or plan.get('code') != 0:"
assert text.count(old)==1;p.write_text(text.replace(old,new))
menv=dict(env,RENDA_CANDIDATE=str(mutant/'renda-harness'))
r=subprocess.run([sys.executable,'-m','unittest','test_delivery_status.ReceiptStatusDelivery.test_rc0_inconsistent_full_plan_refuses_without_status_write'],cwd=mutant,env=menv,text=True,capture_output=True,timeout=60)
(out/'tuple-mutant.log').write_text(r.stdout+r.stderr)
assert r.returncode!=0 and 'AssertionError: True is not false' in r.stderr,r.stderr
# Exercise existing READY assessment and publication with a cached A4 hold.
sys.path.insert(0,str(pkg));os.environ.update(env)
import test_delivery
c=test_delivery.Delivery('test_cached_pipeline_config_hold_needs_signed_qualification_before_publication')
try:
 c.setUp();c.plan['plan']=dict(code=4,output='UNMAPPED:\n  (none)\nPlan verdict: BLOCKED_OPERATOR PIPELINE_CONFIG_CHANGED\n');test_delivery.save(c.planfile,c.plan)
 local,note=test_delivery.h.assessment(c.q,c.entry,'synthetic-model','synthetic-policy')
 assert local is not None,note
 c.receipt=dict(verified=True,eligible=False,verdict='HOLD',digest='e'*64)
 denied,why=c.publish();assert denied is False and not c.writes(),why
 c.receipt=dict(verified=True,eligible=True,verdict='PASS',digest='e'*64)
 published,why=c.publish();assert published is True and len(c.writes())==1,why
 ready_control=dict(assessment_allows_operator_request=True,unsigned_publication_refused=True,signed_fixture_publication_passed=True,scope='synthetic receipt and remote transport; no live writes')
finally:c.doCleanups()
result=dict(source=sha,manifest_hashes=21,modules=rows,total_tests=sum(r['tests'] for r in rows),tuple_mutant_exit=r.returncode,ready_control=ready_control,live_changes=False)
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print('REVIEW_DIR',out);print(json.dumps(result,indent=2))
