#!/usr/bin/env python3
"""Disclosed build orchestration; no downloaded or foreign compiler fallback."""
import hashlib,os,pathlib,struct,subprocess,sys,tempfile
EXPECTED='191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9'
def build(compiler,source,target):
    if hashlib.sha256(compiler.read_bytes()).hexdigest()!=EXPECTED: raise RuntimeError('unqualified compiler fingerprint')
    if target.exists() or target.is_symlink(): raise RuntimeError('output must be absent')
    p=subprocess.run([str(compiler)],input=source.read_bytes(),capture_output=True,timeout=30)
    if p.returncode or p.stderr or len(p.stdout)<64 or p.stdout[:4]!=b'\x7fELF': raise RuntimeError(repr((p.returncode,p.stderr)))
    if struct.unpack_from('<HH',p.stdout,16)!=(2,183): raise RuntimeError('not static AArch64 EXEC')
    with target.open('xb') as f:f.write(p.stdout)
    target.chmod(0o700)
    print(hashlib.sha256(p.stdout).hexdigest(),target)
if __name__=='__main__':build(*map(pathlib.Path,sys.argv[1:]))
