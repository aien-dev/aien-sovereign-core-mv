from pathlib import Path
import subprocess,os,json,hashlib,re
h=Path.home();r=h/'workspace/hermes-desk/L676-browser';out=h/'handoffs/L676-final-qualification';out.mkdir(exist_ok=False,mode=0o700)
e=os.environ.copy();e.update(PATH=str(h/'.local/bin')+':'+e['PATH'],CARGO_TARGET_DIR=str(h/'workspace/targets/L676/83394576f7bff8f18dd235b2dd58c5e90f8fd493'),CARGO_BUILD_JOBS='4',AIEN_CHROME='/usr/bin/google-chrome',TMPDIR=str(h/'workspace/targets/L676/tmp'))
state={'source':'2f6b5ffeda7dd1aa68f26ab548e732de0de82ee8','inputs':{},'steps':[],'complete':False}
for f in ['crates/aien-cli/src/browser_mirror.rs','skills/self-improvement/SKILL.md']:
 state['inputs'][f]=hashlib.sha256((r/f).read_bytes()).hexdigest()
steps=[('fmt',['cargo','fmt','--all','--','--check']),('cli',['cargo','test','--locked','-p','aien-cli']),('clippy',['cargo','clippy','--locked','-p','aien-cli','--all-targets','--','-D','warnings']),('build-browser',['cargo','build','--locked','-p','aien-cli','--bin','aien-browser-mirror'])]
for name,cmd in steps:
 with (out/(name+'.log')).open('w') as f:
  f.write('COMMAND '+repr(cmd)+'\n');f.flush()
  result=subprocess.run(['timeout','180','quiet-build',*cmd],cwd=r,env=e,stdout=f,stderr=subprocess.STDOUT)
 state['steps'].append({'name':name,'command':cmd,'exit':result.returncode});(out/'state.json').write_text(json.dumps(state,indent=2)+'\n');print(name,result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
rows=re.findall(r'test result: ok. (\d+) passed; (\d+) failed; (\d+) ignored', (out/'cli.log').read_text())
state['cli_counts']={k:sum(int(row[i]) for row in rows) for i,k in enumerate(['passed','failed','ignored'])};state['complete']=True;(out/'state.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
