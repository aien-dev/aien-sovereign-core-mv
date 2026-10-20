#!/bin/bash
# After session.sh: speed table, compares vs h0, store exactness, 4-bit fit gate, J/token. Writes runs/summary/*.
set -u; W=$HOME/workspace/speed1-work; R=$W/runs; O=$R/summary; mkdir -p $O
AB=$HOME/workspace/targets/speed1/sc/release/qwen38_ab; TH=$HOME/workspace/sc-speed1-w4default/docs/inference/qwen38/w4-fit-gate.json
last() { ls -d $R/$1-2* 2>/dev/null | tail -1; }
echo "arm round median_chip_ms upload_s weight_GB golden_identical rc" > $O/speed.txt
for d in $(ls -d $R/speed-* | sort); do
  n=$(basename $d); a=$(echo $n | cut -d- -f2); r=$(echo $n | cut -d- -f3)
  f=$d/report.json; rc=$(grep -o 'rc=[0-9]*' $d/run.log | tail -1)
  if [ -f $f ]; then
    m=$(jq '[.steps[].chip_ns_all_windows] | sort | .[length/2|floor] / 1e6' $f)
    echo "$a $r $m $(jq .upload_s $f) $(jq '.weight_bytes/1e9' $f) $(jq .golden_identical $f) $rc" >> $O/speed.txt
  else echo "$a $r NA NA NA NA $rc" >> $O/speed.txt; fi
done
for s in golden license prose codelong code arch; do
  h=$(last dump-$s-H0); c=$(last dump-$s-S)
  [ -n "$h" ] && [ -n "$c" ] && $AB compare $h/out $c/out $O/cmp-$s-S-vs-h0.json 2>> $O/compare.log
done
l=$(last dump-golden-L); c=$(last dump-golden-S); [ -n "$l" ] && [ -n "$c" ] && $AB compare $l/out $c/out $O/cmp-exact-L-vs-S.json 2>> $O/compare.log
full=$(ls -d $R/speed-S-r1-* | tail -1)/report.json
args="exact=$O/cmp-exact-L-vs-S.json full=$full"; for s in golden license prose codelong code arch; do args="$args $s=$O/cmp-$s-S-vs-h0.json"; done
$AB gate4 $TH $O/gate4.json $args 2> $O/gate4.log; echo "gate4 rc=$?" >> $O/gate4.log
for v in F S; do d=$(last gen-prose-$v); echo "$v $($W/bin/jtok.sh $d)" ; done > $O/jtok.txt 2>&1
cat $O/speed.txt; grep QWEN38_AB $O/compare.log | cut -c1-160; cat $O/gate4.log $O/jtok.txt
