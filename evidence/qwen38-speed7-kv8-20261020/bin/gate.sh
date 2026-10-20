#!/bin/bash
# gate.sh A_RUN B_RUN NAME: the drift gate (fixed 02:5xZ, unchanged) on the 6 texts, A = reference (bf16 KV).
# Per text: top-pick agreement >= 98.0%, every flip a near tie (A margin <= 0.5 logit), mean KL <= 0.01,
# |dNLL| <= 0.005 nats/token; golden must agree at every position (47/47). Output runs/cmp-NAME/<text>.json.
set -u; cd $HOME/workspace/speed7-work/runs; AB=$HOME/workspace/targets/speed7/release/qwen38_ab
A=$1; B=$2; O=cmp-$3; mkdir -p $O; fails=0
printf "%-9s %11s %8s %9s %9s %8s %s\n" text agree pct dNLL kl_mean flipmax verdict
for t in golden license prose codelong code arch; do
  $AB compare $A/$t $B/$t $O/$t.json > $O/$t.log 2>&1 || { echo "$t: compare failed"; fails=$((fails+1)); continue; }
  read -r n ag dn kl fm < <(jq -r '[.positions, .argmax_agree, (.b_mean_nll_fed - .a_mean_nll_fed), .kl_mean, ([.flips[].a_margin] | max // 0)] | @tsv' $O/$t.json)
  v=$(awk -v n=$n -v ag=$ag -v dn=$dn -v kl=$kl -v fm=$fm -v t=$t 'BEGIN{ok = ag/n >= 0.98 && fm <= 0.5 && kl <= 0.01 && (dn<0?-dn:dn) <= 0.005 && (t != "golden" || ag == n); print ok ? "pass" : "FAIL"}')
  [ $v = pass ] || fails=$((fails+1))
  printf "%-9s %5s/%-5s %7.2f%% %+9.4f %9.5f %8.3f %s\n" $t $ag $n $(awk -v a=$ag -v n=$n 'BEGIN{print 100*a/n}') $dn $kl $fm $v
done
echo "GATE $3: $([ $fails = 0 ] && echo PASS || echo "FAIL ($fails of 6 texts)")"
