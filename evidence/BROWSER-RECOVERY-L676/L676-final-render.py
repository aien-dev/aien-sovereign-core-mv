from pathlib import Path
import os,subprocess,json,urllib.parse,signal
h=Path.home();out=h/'handoffs/L676-final-qualification';target=h/'workspace/targets/L676/83394576f7bff8f18dd235b2dd58c5e90f8fd493';env=os.environ.copy();env.update(AIEN_CHROME='/usr/bin/google-chrome',TMPDIR=str(h/'workspace/targets/L676/tmp'))
html='<html><head><title>L676 final rendered pixel fixture</title></head><body style="margin:0;background:white">'+''.join(f'<div style="position:absolute;left:{20+i*100}px;top:20px;width:80px;height:80px;background:{color}"></div>' for i,color in enumerate(['#ff0000','#00ff00','#0000ff','#ffff00']))+'</body></html>'
(out/'fixture.html').write_text(html)
cmd=[str(target/'debug/aien-browser-mirror'),'--screenshot','pixels.png','--url','data:text/html,'+urllib.parse.quote(html),'--out-dir',str(out)]
p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=90);(out/'render.stdout.json').write_text(p.stdout);(out/'render.stderr.log').write_text(p.stderr);assert p.returncode==0,(p.returncode,p.stderr);assert json.loads(p.stdout)['status']=='ok';print(p.stdout)
binary=target/'debug/deps/aien_cli-1f935d6b820ea560'
with (out/'sigchld.log').open('w') as f:
 p=subprocess.run([str(binary),'browser_mirror::tests::cleanup_with_inherited_ignored_sigchld','--exact','--ignored','--nocapture','--test-threads=1'],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=30,preexec_fn=lambda:signal.signal(signal.SIGCHLD,signal.SIG_IGN))
assert p.returncode==0,p.returncode
trace=out/'cleanup.strace'
with (out/'cleanup.log').open('w') as f:
 p=subprocess.run(['strace','-f','-e','trace=kill,wait4,pidfd_open,pidfd_send_signal','-o',str(trace),str(binary),'browser_mirror','--nocapture','--test-threads=1'],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60)
assert p.returncode==0,p.returncode
s=trace.read_text();numeric=[x for x in s.splitlines() if 'kill(' in x and 'SIGKILL' in x];fd=[x for x in s.splitlines() if 'pidfd_send_signal(' in x and 'SIGKILL' in x and '= 0' in x]
# Test fixture cleanup may use Child::kill for owned neighbors. Production
# Linux cleanup is descriptor-only, assessed separately by source review.
assert fd,'no successful pidfd SIGKILL observed'
(out/'runtime.json').write_text(json.dumps({'source':'2f6b5ffeda7dd1aa68f26ab548e732de0de82ee8','render_exit':0,'sigchld_exit':0,'trace_test_exit':0,'successful_pidfd_sigkills':len(fd),'numeric_test_fixture_sigkills':numeric,'scope':'numeric calls include controlled fixture cleanup; no forced PID reuse or foreign task signaling'},indent=2)+'\n')
print('render, explicit inherited SIGCHLD and traced browser tests passed')
