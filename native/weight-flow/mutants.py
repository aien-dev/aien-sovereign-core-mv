#!/usr/bin/env python3
"""Compile and execute native behavioral mutants, never count a compile failure as a kill."""
import json,pathlib,subprocess,sys
from build import build
from test_policy import suite
compiler,source,out=map(pathlib.Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=False)
raw=source.read_text();rows=[]
for name,before,after in [('no-growth','depth = add depth 1','depth = depth'),('no-pressure','eq field 8 1','eq field 8 9'),('serial-fixed','if eq mode 1\ndepth = ulong 2','if eq mode 1\ndepth = ulong 1')]:
    if raw.count(before)!=1:raise RuntimeError('mutation anchor '+name)
    src=out/(name+'.om');src.write_text(raw.replace(before,after));binary=out/name
    build(compiler,src,binary);result=suite(str(binary));rows.append({'mutant':name,'killed':result['total']-result['passed'],'result':result})
    if result['passed']==result['total']:raise RuntimeError('survived '+name)
(out/'receipt.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='result'} for r in rows]))
