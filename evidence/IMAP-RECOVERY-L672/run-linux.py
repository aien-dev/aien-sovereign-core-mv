from pathlib import Path
import subprocess,os,json,hashlib
h=Path.home();shared=h/'workspace/aien-sovereign-core';r=h/'workspace/hermes-desk/L672-imap';sha='2ea9ee1c804b84c23dc74651c085c00949110e36'
assert not r.exists();assert subprocess.check_output(['git','-C',str(shared),'rev-parse','--show-toplevel'],text=True).strip()==str(shared)
subprocess.run(['git','-C',str(shared),'fetch',str(h/'handoffs/L672-source.bundle'),'refs/heads/qualification/626-imap'],check=True);subprocess.run(['git','-C',str(shared),'worktree','add','--detach',str(r),sha],check=True)
o=h/'handoffs/L672-linux';o.mkdir(exist_ok=False);e=os.environ.copy();e['PATH']=str(h/'.local/bin')+':'+e['PATH'];e['CARGO_TARGET_DIR']=str(h/'workspace/targets/L672'/sha);e['CARGO_BUILD_JOBS']='4';tmp=h/'workspace/targets/L672/tmp';tmp.mkdir(parents=True,mode=0o700,exist_ok=False);e['TMPDIR']=str(tmp)
s={'source':sha,'input_sha256':hashlib.sha256((r/'crates/spark-mail-rs/src/imap.rs').read_bytes()).hexdigest(),'steps':[],'complete':False}
for name,cmd in [('tests',['cargo','test','--locked','-p','spark-mail-rs']),('clippy',['cargo','clippy','--locked','-p','spark-mail-rs','--all-targets','--','-D','warnings']),('fmt',['cargo','fmt','--all','--','--check'])]:
 with (o/(name+'.log')).open('w') as f:
  try:rc=subprocess.run(['timeout','20m','quiet-build',*cmd],cwd=r,env=e,stdout=f,stderr=subprocess.STDOUT,timeout=1250).returncode
  except subprocess.TimeoutExpired:rc=124
 s['steps'].append({'command':cmd,'exit':rc});(o/'state.json').write_text(json.dumps(s,indent=2)+'\n');print(name,rc,flush=True)
 if rc:break
s['complete']=len(s['steps'])==3 and all(x['exit']==0 for x in s['steps']);(o/'state.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s),flush=True)
