import pathlib,json,time,os,datetime,subprocess,hashlib
h=pathlib.Path.home();out=h/'handoffs/L679-deploy';r=json.loads((out/'install.json').read_text());deadline=time.monotonic()+1200
boot=int(next(l.split()[1] for l in pathlib.Path('/proc/stat').read_text().splitlines() if l.startswith('btime ')));hz=os.sysconf('SC_CLK_TCK')
def completed(name, workers, logs):
 rows=workers.get(name,[])
 if len(rows)!=1 or not rows[0]['new_revision_start']:return False
 lines=logs.get(name,[])
 starts=[i for i,l in enumerate(lines) if f' {name} start lanes=' in l]
 if not starts:return False
 index=starts[-1]
 started=datetime.datetime.strptime(lines[index][:20],'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()
 if started<int(rows[0]['started_at']):return False
 return any('pass done' in l for l in lines[index+1:])
def snapshot():
 workers={};logs={}
 for d in pathlib.Path('/proc').iterdir():
  if not d.name.isdigit():continue
  try:
   a=[x.decode() for x in (d/'cmdline').read_bytes().split(b'\0') if x]
   if len(a)<4 or a[1]!=str(h/'.local/bin/renda-harness') or '--name' not in a:continue
   name=a[a.index('--name')+1]
   if name not in ('renda1','renda2'):continue
   started=boot+int((d/'stat').read_text().rsplit(')',1)[1].split()[19])/hz
   workers.setdefault(name,[]).append(dict(pid=int(d.name),argv=a,started_at=started,new_revision_start=started>r['installed_at']))
  except (OSError,ValueError,IndexError):continue
 for name in ('renda1','renda2'):
  lines=(h/'handoffs/renda-local'/('harness-'+name+'.log')).read_text().splitlines();latest=[]
  for l in lines:
   try:ts=datetime.datetime.strptime(l[:20],'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()
   except ValueError:continue
   if ts>=r['installed_at']:latest.append(l)
  logs[name]=latest
 valid={n:completed(n,workers,logs) for n in logs}
 return dict(workers=workers,logs=logs,loaded_completed_pass=valid)
while True:
 s=snapshot()
 if all(s['loaded_completed_pass'].values()) or time.monotonic()>=deadline:break
 time.sleep(10)
s['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();s['installed_hashes_match']=all(hashlib.sha256((h/'.local/bin'/f).read_bytes()).hexdigest()==v for f,v in r['installed'].items())
s['state']='LOADED_PASSES_VERIFIED' if all(s['loaded_completed_pass'].values()) and s['installed_hashes_match'] else 'INCOMPLETE'
query='{core:repository(owner:"aien-dev",name:"aien-sovereign-core-mv"){p577:pullRequest(number:577){state headRefOid mergedAt} p609:pullRequest(number:609){state headRefOid mergedAt} p626:pullRequest(number:626){state headRefOid mergedAt}} omega:repository(owner:"aien-dev",name:"omega-mv"){p620:pullRequest(number:620){state headRefOid mergedAt}}}'
p=subprocess.run([str(h/'.local/bin/gh'),'api','graphql','-f','query='+query],capture_output=True,text=True,timeout=120)
s['remote_readback']=json.loads(p.stdout) if p.returncode==0 else dict(error=p.stderr[-500:])
(out/'adoption-readback.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2));raise SystemExit(0 if s['state']=='LOADED_PASSES_VERIFIED' else 2)
