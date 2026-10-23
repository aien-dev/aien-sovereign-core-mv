from pathlib import Path
import subprocess,os,json
h=Path.home();shared=h/'workspace/aien-sovereign-core';r=h/'workspace/hermes-desk/L666-m3';bundle=h/'handoffs/L666-source.bundle';sha='fc62b5f2cf576a24ac7200ed1cb660aa9ee1e625'
assert subprocess.check_output(['git','-C',str(shared),'rev-parse','--show-toplevel'],text=True).strip()==str(shared)
assert not r.exists()
subprocess.run(['git','-C',str(shared),'fetch',str(bundle),'refs/heads/qualification/559-m3'],check=True)
subprocess.run(['git','-C',str(shared),'worktree','add','--detach',str(r),sha],check=True)
out=h/'handoffs/L666-linux';out.mkdir(exist_ok=False);env=os.environ.copy();env['PATH']=str(h/'.local/bin')+':'+env['PATH'];env['CARGO_TARGET_DIR']=str(h/'workspace/targets/L666'/sha);env['CARGO_BUILD_JOBS']='4'
steps=[('feature',['cargo','test','--locked','-p','aien-runtime','--features','provisional-prep','--test','provisional_prep_test']),('default',['cargo','check','--locked','-p','aien-runtime','--no-default-features']),('clippy',['cargo','clippy','--locked','-p','spark-harness','-p','aien-runtime','--features','aien-runtime/provisional-prep','--tests','--','-D','warnings'])]
s={'source':sha,'steps':[],'complete':False};p=out/'state.json'
for name,cmd in steps:
 p.write_text(json.dumps(s,indent=2)+'\n')
 with (out/(name+'.log')).open('w') as f:
  f.write('COMMAND '+repr(cmd)+'\n');f.flush()
  try:z=subprocess.run(['timeout','--signal=TERM','20m','quiet-build',*cmd],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1250);rc=z.returncode
  except subprocess.TimeoutExpired:rc=124
 s['steps'].append({'name':name,'command':cmd,'exit':rc});p.write_text(json.dumps(s,indent=2)+'\n');print(name,rc,flush=True)
 if rc:break
s['complete']=len(s['steps'])==len(steps) and all(x['exit']==0 for x in s['steps']);p.write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s),flush=True)
