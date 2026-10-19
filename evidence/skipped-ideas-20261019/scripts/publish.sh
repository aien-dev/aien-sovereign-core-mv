#!/bin/bash
# publish.sh "<commit message>" : copy every run dir into the branch evidence folder, commit, push.
# Evidence is append-only: existing files in evidence are never rewritten.
set -u
MSG="$1"
exec 9>/tmp/skipped-publish.lock; flock 9
WT=$HOME/workspace/sc-skipped; W=$HOME/workspace/skipped-work
EV=$WT/evidence/skipped-ideas-20261019
mkdir -p $EV/runs $EV/scripts
for d in $W/runs/*/; do
  n=$(basename $d); mkdir -p $EV/runs/$n
  for f in $d*; do
    b=$(basename $f)
    case $b in *.f32|*.logits) continue;; esac
    [ -e $EV/runs/$n/$b ] && cmp -s $f $EV/runs/$n/$b && continue
    [ -e $EV/runs/$n/$b ] && [ -f $W/runs/$n/.final ] && continue
    cp $f $EV/runs/$n/$b
  done
  # receipts named in run.log
  for r in $(grep -o '/home/[^ ]*/yardsticks/receipts/[^ ]*\.json' $d/run.log 2>/dev/null); do
    cp -n $r $EV/runs/$n/ 2>/dev/null
  done
done
cp $W/bin/*.sh $EV/scripts/ 2>/dev/null
[ -d $W/seq ] && { mkdir -p $EV/seq; cp -n $W/seq/* $EV/seq/; }
[ -d $W/sessions ] && { mkdir -p $EV/sessions; cp $W/sessions/* $EV/sessions/ 2>/dev/null; }
[ -f $W/README.md ] && cp $W/README.md $EV/README.md
cd $WT
git add -A crates/aien-inference-abi/src/qwen38.rs crates/aien-inference-abi/src/qwen38/ crates/aien-inference-abi/src/bin/qwen38_lab.rs evidence/skipped-ideas-20261019
git commit -q -m "$MSG" && git push -q origin research/skipped-ideas 2>&1 | tail -2
git log --oneline -1
