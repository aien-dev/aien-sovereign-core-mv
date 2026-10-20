#!/usr/bin/env bash
# compare-all.sh: builds the two reference folders from the evidence and runs every comparison in the README.
#  ref-A  : prompts 0-2 answered alone (partial-A-0533Z-nice10, bin-946f6fed-h4096: same tree GEMM at 1 row, before the
#           tile table rewrite), prompt 3 alone (A n1-p3, bin-946f6fed-tvtab), prompts 4-31 = A n32 streams.
#  ref-B  : B n1 (prompt 0, FP8 head alone) + B n32 streams.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); cd "$H"
rm -rf ref-A ref-B; mkdir ref-A ref-B
for i in 0 1 2; do cp partial-A-0533Z-nice10/n1-p$i-a1-$i.json ref-A/; echo "n1-p$i n1-p$i-a1" >> ref-A/good.txt; done
cp A-tree-bf16head/n1-p3-a1-3.json ref-A/; echo "n1-p3 n1-p3-a1" >> ref-A/good.txt
for f in A-tree-bf16head/n32-a1-*.json; do case $f in *.req.json) ;; *) cp "$f" ref-A/;; esac; done; echo "n32 n32-a1" >> ref-A/good.txt
cp B-tree-fp8head/n1-a1-0.json ref-B/n1-p0-a1-0.json; echo "n1-p0 n1-p0-a1" >> ref-B/good.txt
for f in B-tree-fp8head/n32-a1-*.json; do case $f in *.req.json) ;; *) cp "$f" ref-B/;; esac; done; echo "n32 n32-a1" >> ref-B/good.txt
./compare.sh ref-A n32 A-tree-bf16head > A-tree-bf16head/compare.txt
for d in B-tree-fp8head C-rowpair-bf16head D-single-door E-tree-fp8head-host8; do ./compare.sh ref-A n32 $d > $d/compare-vs-A.txt; done
./compare.sh ref-B n32 B-tree-fp8head > B-tree-fp8head/compare-self.txt
./compare.sh ref-B n32 E-tree-fp8head-host8 > E-tree-fp8head-host8/compare-vs-B.txt
for f in */compare*.txt; do echo "$f: $(tail -1 "$f")"; done
