#!/bin/bash
# L154: for each omega gb10 gate: time its prepare step, then dry-run (MAKEFLAGS=n) its run step and
# count compiler/linker commands left for the locked part. Output: /tmp/l154-t/prep-all.tsv
cd /tmp/l154-t/omega || exit 1
out=/tmp/l154-t/prep-all.tsv; : > "$out"
for g in tests/gates/*.gate; do
  p=$(awk '/^prepare:/{f=1} f && /args:/{gsub(/.*\["|"\].*/,""); print; exit}' "$g"); [ -n "$p" ] || continue
  r=$(awk '/^run:/{f=1} f && /args:/{gsub(/.*\["|"\].*/,""); print; exit}' "$g")
  t0=$(date +%s.%N); tests/gates/run.sh "$p" > "/tmp/l154-t/prep-$r.log" 2>&1; rc=$?; t1=$(date +%s.%N)
  MAKEFLAGS=n tests/gates/run.sh "$r" > "/tmp/l154-t/dry-$r.log" 2>&1
  left=$(grep -cE '^(cc|gcc|clang|aarch64-[a-z-]*gcc|ld|ar) |(^| )(-c|-o) ' "/tmp/l154-t/dry-$r.log")
  printf '%s\t%s\tprep_rc=%s\tprep_s=%.1f\tcompile_cmds_left=%s\n' "$(basename "$g")" "$r" "$rc" "$(echo "$t1 - $t0" | bc)" "$left" >> "$out"
done
echo DONE >> "$out"
