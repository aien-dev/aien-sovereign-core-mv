#!/bin/sh
# prefix exactness gate (L369): LAYERS-layer slice of the 35B, real + mutants 1-3, 5, 6 + per-token GDN / attention controls;
# value check (chip lock, no quiet). A slice is not a full-35B load, so it does not take the q36-extra slot; it waits (up to
# WAIT_S) for MemAvailable >= NEED_GB itself and runs with oom_score_adj 1000 (the kernel takes this run, never 18091).
# Optional args: the tags to run (default all).
L=${LAYERS:-4}; LENS=${LENS:-7,64,129,300}; CH=${CHUNK:-128}; D=${DECODE:-16}; NEED=${NEED_GB:-24}; WAIT_S=${WAIT_S:-3600}
B=${GATE_BIN:-$HOME/workspace/targets/l369/sc/release/qwen36_prefill_gate}; S=${SUF:-}; M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
E=$HOME/workspace/targets/l369/ev; mkdir -p $E; C=$HOME/workspace/targets/l369/corpus.txt
run() { tag=$1; shift
  w=0; while [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -lt "$NEED" ]; do
    [ $w -ge $WAIT_S ] && { echo "$tag rc=75 (MemAvailable < $NEED GB for ${WAIT_S}s)" >> $E/prefix$L$S.result; return; }; sleep 15; w=$((w+15)); done
  env "$@" sh -c 'echo 1000 > /proc/self/oom_score_adj; exec "$@"' x flock /tmp/aien-gb10.lock $B $M $C $LENS $CH $D $E/prefix$L$S-$tag.json $L > $E/prefix$L$S-$tag.log 2>&1
  echo "$tag rc=$? $(grep -c '^len .*PASS' $E/prefix$L$S-$tag.log) PASS $(grep -c '^len .*FAIL' $E/prefix$L$S-$tag.log) FAIL $(grep -o 'CHIP_FAIL[^)]*' $E/prefix$L$S-$tag.log | head -1)" >> $E/prefix$L$S.result; }
TAGS=${*:-real mut1 mut2 mut3 mut5 mut6 gdn0 attn0}
: > $E/prefix$L$S.result; echo "gate $(sha256sum $B | cut -c1-16)" >> $E/prefix$L$S.result
for t in $TAGS; do case $t in
  real) run real X=1;; cn0) run cn0 AIEN_OMEGA_Q36_CN_ROWS=0;; gdn0) run gdn0 AIEN_OMEGA_Q36_PF_GDN=0;; attn0) run attn0 AIEN_OMEGA_Q36_PF_ATTN=0;;
  mut*) run $t AIEN_OMEGA_Q36_PF_MUTANT=${t#mut};; esac; done
cat $E/prefix$L$S.result
