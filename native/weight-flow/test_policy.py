#!/usr/bin/env python3
"""Independent policy ABI fixtures; no GPU timing claims."""
import json,struct,subprocess,sys,select

def frame(mode=2,cap=8,depth=2,wait=100,compute=400,previous=400,calm=0,pressure=0,version=1):
    return struct.pack('<10Q',version,mode,cap,depth,wait,compute,previous,calm,pressure,0)

def suite(binary):
    cases=[('sync',frame(mode=0),1,0),('fixed',frame(mode=1),2,0),
      ('grow',frame(),3,0),('no-stall',frame(wait=0),2,1),
      ('shrink-calm',frame(depth=5,wait=0,calm=3),4,0),
      ('pressure',frame(depth=8,pressure=1),4,0),
      ('contention',frame(depth=6,compute=501),3,0),
      ('boundary-contention',frame(depth=6,compute=500),7,0),
      ('capacity-one',frame(cap=1,depth=1),1,0),
      ('at-cap',frame(cap=2),2,0),
      ('wait-boundary',frame(wait=50),2,0),
      ('zero-compute',frame(wait=1,compute=0,previous=0),3,0),
      ('fixed-cap-one',frame(mode=1,cap=1,depth=1),1,0)]
    rows=[]
    for name,data,d,c in cases:
        p=subprocess.run([binary],input=data,capture_output=True,timeout=5)
        rows.append(dict(case=name,rc=p.returncode,pass_=p.returncode==0 and p.stdout==struct.pack('<2Q',d,c) and not p.stderr))
    bad=[frame(version=2),frame(mode=3),frame(cap=0),frame(cap=33),frame(depth=0),frame(depth=9),frame(calm=4),frame(pressure=2),frame(wait=2**63),frame(compute=2**63),frame(previous=2**63),frame()[:-1],frame()+b'x']
    for i,data in enumerate(bad):
        p=subprocess.run([binary],input=data,capture_output=True,timeout=5)
        expected=struct.pack('<2Q',3,0) if i==len(bad)-1 else b''
        rows.append(dict(case='refuse-'+str(i),rc=p.returncode,pass_=p.returncode==65 and p.stdout==expected and not p.stderr))
    p=subprocess.run([binary],input=frame()*100,capture_output=True,timeout=5)
    rows.append(dict(case='repeated-100',pass_=p.returncode==0 and p.stdout==struct.pack('<2Q',3,0)*100))
    p=subprocess.Popen([binary],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    p.stdin.write(frame()[:39]);p.stdin.flush()
    early=bool(select.select([p.stdout],[],[],0.03)[0])
    out,err=p.communicate(frame()[39:],timeout=5)
    rows.append(dict(case='fragmented',pass_=not early and p.returncode==0 and out==struct.pack('<2Q',3,0) and not err))
    return dict(total=len(rows),passed=sum(r['pass_'] for r in rows),cases=rows)
if __name__=='__main__':
    r=suite(sys.argv[1]);print(json.dumps(r,indent=2));sys.exit(r['passed']!=r['total'])
