#!/usr/bin/env python3
"""External qualification only. Build each mutation, then assert exact failure."""
import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path(__file__).resolve().parents[1];out=pathlib.Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
seal=root/'crates/aien-kv-store/src/seal.rs';host=root/'crates/aien-kv-store/src/host_fs.rs'
original={p:p.read_text() for p in [seal,host]}
checks=[('nonce-error',seal,'            source(&mut nonce)?;','            let _ = source(&mut nonce);','seal::nonce_tests::entropy_error_is_transactional','entropy failure must propagate','nonce_tests.rs'),('publication-error',host,'if shared.seal_key.seal_file(&mut file_bytes).is_err() {','if { let _ = shared.seal_key.seal_file(&mut file_bytes); false } {','host_fs::tests::entropy_failure_never_publishes_plaintext_or_partial_file','entropy failure cannot complete durable publication','nonce_host_tests.rs'),('aad-omitted',seal,'Aad::from(&region[..AAD_LEN])','Aad::empty()','seal::nonce_tests::aad_every_byte_and_failed_open_zeroes_payload','must be authenticated','nonce_tests.rs'),('failed-open-not-zeroed',seal,'            payload.fill(0);','            // deliberately unsafe mutant','seal::nonce_tests::aad_every_byte_and_failed_open_zeroes_payload','failed open must zero usable payload','nonce_tests.rs')]
rows=[]
def run(args,name):
 try:r=subprocess.run(args,cwd=root,text=True,capture_output=True,timeout=240)
 except subprocess.TimeoutExpired:return None
 (out/(name+'.log')).write_text(r.stdout+r.stderr);return r
try:
 for name,p,old,new,test,msg,site in checks:
  if old not in original[p]:raise RuntimeError('unmatched mutation '+name)
  p.write_text(original[p].replace(old,new));build=run(['cargo','test','--locked','-p','aien-kv-store','--lib','--no-run','--message-format=json'],name+'-build');status='INVALID';rc=None;digest=None
  if build and build.returncode==0:
   bins=[a['executable'] for l in build.stdout.splitlines() if l.startswith('{') for a in [json.loads(l)] if a.get('reason')=='compiler-artifact' and a.get('executable') and a.get('profile',{}).get('test')]
   if len(bins)!=1:raise RuntimeError('ambiguous test executable')
   digest=hashlib.sha256(pathlib.Path(bins[0]).read_bytes()).hexdigest();r=run([bins[0],'--exact',test,'--nocapture'],name+'-run')
   if r:
    rc=r.returncode;s=r.stdout+r.stderr
    if rc==101 and 'running 1 test' in s and '1 failed' in s and msg in s and site in s:status='PASS'
    elif rc==0:status='FAIL'
  rows.append({'mutant':name,'status':status,'exit':rc,'test':test,'intended_assertion':msg,'binary_sha256':digest});print(json.dumps(rows[-1]),flush=True);p.write_text(original[p])
finally:
 for p,t in original.items():p.write_text(t)
(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
sys.exit(0 if len(rows)==len(checks) and all(r['status']=='PASS' for r in rows) else 1)
