from pathlib import Path
import os,subprocess,json
r=Path.home()/'.hermes/cache/scratch/loop21-imap626';o=r/'evidence/IMAP-RECOVERY-L672/initial';o.mkdir(parents=True,exist_ok=False);env=os.environ.copy();env['CARGO_TARGET_DIR']=str(Path.home()/'.hermes/cache/scratch/L672-target');env['CARGO_BUILD_JOBS']='4'
s={'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),'steps':[],'complete':False}
steps=[('tests',['cargo','test','--locked','-p','spark-mail-rs']),('clippy',['cargo','clippy','--locked','-p','spark-mail-rs','--all-targets','--','-D','warnings']),('fmt',['cargo','fmt','--all','--','--check']),('graph',['cargo','metadata','--locked','--format-version','1'])]
for name,c in steps:
 with (o/(name+'.log')).open('w') as f:
  try:rc=subprocess.run(c,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600).returncode
  except subprocess.TimeoutExpired:rc=124
 s['steps'].append({'name':name,'command':c,'exit':rc});(o/'state.json').write_text(json.dumps(s,indent=2)+'\n');print(name,rc,flush=True)
 if rc:break
s['complete']=len(s['steps'])==len(steps) and all(v['exit']==0 for v in s['steps']);(o/'state.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s),flush=True)
