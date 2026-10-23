from pathlib import Path
import subprocess,os,json,time
r=Path('/Users/drakestapleton/.hermes/cache/scratch/loop19-m3-559');out=r/'evidence/M3-RECOVERY-L666/host';out.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env['CARGO_TARGET_DIR']=str(Path.home()/'.hermes/cache/scratch/L666-target');env['CARGO_BUILD_JOBS']='4'
steps=[('harness',['cargo','test','--locked','-p','spark-harness']),('feature',['cargo','test','--locked','-p','aien-runtime','--features','provisional-prep','--test','provisional_prep_test']),('default',['cargo','check','--locked','-p','aien-runtime','--no-default-features']),('fmt',['cargo','fmt','--all','--','--check']),('clippy',['cargo','clippy','--locked','-p','spark-harness','-p','aien-runtime','--features','aien-runtime/provisional-prep','--tests','--','-D','warnings'])]
s={'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),'steps':[],'complete':False};p=out/'state.json'
for name,cmd in steps:
 p.write_text(json.dumps(s,indent=2)+'\n');print('RUN',name,flush=True)
 with (out/(name+'.log')).open('w') as f:
  f.write('COMMAND '+repr(cmd)+'\n');f.flush()
  try:z=subprocess.run(cmd,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900);rc=z.returncode
  except subprocess.TimeoutExpired:rc=124
 s['steps'].append({'name':name,'command':cmd,'exit':rc});p.write_text(json.dumps(s,indent=2)+'\n');print(name,rc,flush=True)
 if rc:break
s['complete']=len(s['steps'])==len(steps) and all(x['exit']==0 for x in s['steps']);p.write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s),flush=True)
