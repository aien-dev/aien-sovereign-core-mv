from pathlib import Path
import subprocess,os,json,hashlib
h=Path.home();r=h/'workspace/hermes-desk/L666-m3';out=h/'handoffs/L666-final-host';out.mkdir(exist_ok=False)
env=os.environ.copy();env['PATH']=str(h/'.local/bin')+':'+env['PATH'];env['CARGO_TARGET_DIR']=str(h/'workspace/targets/L666/fc62b5f2cf576a24ac7200ed1cb660aa9ee1e625');env['CARGO_BUILD_JOBS']='4';tmp=h/'workspace/targets/L666/tmp';tmp.mkdir(exist_ok=True);env['TMPDIR']=str(tmp)
files=['crates/spark-harness/src/scheduler/core.rs','crates/spark-harness/src/scheduler/mod.rs','crates/spark-harness/tests/provisional_commit.rs','crates/aien-runtime/src/provisional.rs','crates/aien-runtime/tests/provisional_prep_test.rs','docs/headstart/ROLLOUT.md']
steps=[('harness',['cargo','test','--locked','-p','spark-harness']),('runtime-full-feature',['cargo','test','--locked','-p','aien-runtime','--features','provisional-prep']),('default',['cargo','check','--locked','-p','aien-runtime','--no-default-features']),('clippy',['cargo','clippy','--locked','-p','spark-harness','-p','aien-runtime','--features','aien-runtime/provisional-prep','--tests','--','-D','warnings']),('fmt',['cargo','fmt','--all','--','--check'])]
s={'checkout_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),'source_inputs':{f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in files},'steps':[],'complete':False};p=out/'state.json'
for name,cmd in steps:
 p.write_text(json.dumps(s,indent=2)+'\n')
 with (out/(name+'.log')).open('w') as f:
  f.write('COMMAND '+repr(cmd)+'\n');f.flush()
  try:z=subprocess.run(['timeout','--signal=TERM','20m','quiet-build',*cmd],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1250);rc=z.returncode
  except subprocess.TimeoutExpired:rc=124
 s['steps'].append({'name':name,'command':cmd,'exit':rc});p.write_text(json.dumps(s,indent=2)+'\n');print(name,rc,flush=True)
 if rc:break
s['complete']=len(s['steps'])==len(steps) and all(x['exit']==0 for x in s['steps']);p.write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s),flush=True)
