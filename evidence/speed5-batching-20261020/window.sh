#!/usr/bin/env bash
# All SPEED-5 sessions in one spark-window (run as: spark-window run ./window.sh <bin>). Order: A, B, E, C, D.
#  A tree GEMM + bf16 head, 32 slots: correctness refs n=1 for prompts 0-7, N = 2,4,8,16,32, a shifted 16 (prompts 16-31), N=1 again.
#  B tree GEMM + FP8 head (one GEMM for all rows), 32 slots: N = 1,2,4,8,16,32, N=1 again.
#  E as B with the per-row SiLU on 8 host threads (AIEN_Q38_HOST_THREADS=8): N = 8, 32.
#  C row-pair GEMM + bf16 head (the first batch cut), 8 slots: N = 1, 8.
#  D one-at-a-time door (decode graph): prompts 0 and 3.
set -euo pipefail
export BIN=$1
HERE=$(cd "$(dirname "$0")" && pwd)
seqc() { seq -s, "$1" "$2"; }
N1=(); for i in 0 1 2 3 4 5 6 7; do N1+=("n1-p$i=$i"); done
SLOTS=32 KERNEL=tree FP8HEAD=0 "$HERE/session.sh" "$HERE/A-tree-bf16head" "${N1[@]}" n2=$(seqc 0 1) n4=$(seqc 0 3) n8=$(seqc 0 7) n16=$(seqc 0 15) n32=$(seqc 0 31) n16s=$(seqc 16 31) n1-last=0 || echo "A failed rc $?"
SLOTS=32 KERNEL=tree FP8HEAD=1 "$HERE/session.sh" "$HERE/B-tree-fp8head" n1=0 n2=$(seqc 0 1) n4=$(seqc 0 3) n8=$(seqc 0 7) n16=$(seqc 0 15) n32=$(seqc 0 31) n1-last=0 || echo "B failed rc $?"
SLOTS=32 KERNEL=tree FP8HEAD=1 AIEN_Q38_HOST_THREADS=8 "$HERE/session.sh" "$HERE/E-tree-fp8head-host8" n8=$(seqc 0 7) n32=$(seqc 0 31) || echo "E failed rc $?"
SLOTS=8 KERNEL=rowpair FP8HEAD=0 "$HERE/session.sh" "$HERE/C-rowpair-bf16head" n1=0 n8=$(seqc 0 7) || echo "C failed rc $?"
SLOTS=0 "$HERE/session.sh" "$HERE/D-single-door" single-p0=0 single-p3=3 || echo "D failed rc $?"
echo "window.sh done $(date -u +%FT%TZ)"
