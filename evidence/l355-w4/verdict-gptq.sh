#!/bin/bash
# L355 GPTQ-arm verdict (docs/inference/q36/w4-gate.md, "GPTQ arm"): A = the frozen rule on the five goldens (verdict.sh),
# B = R1 / R2 on held-out h1..h5 against the FP8 arm's own greedy ids, C = paired block bootstrap NLL(arm) - NLL(base).
#   verdict-gptq.sh <arm name, e.g. w4g> <base name, e.g. w4e>
set -u
ARM=${1:?arm}; BASE=${2:?base}; R=$HOME/workspace/sc-hum-state-precision/evidence/hum-state-precision; O=runs
cd $HOME/workspace/sc-l355/evidence/l355-w4
A=$(./verdict.sh $ARM)
args=(); for i in 1 2 3 4 5; do args+=(--slurpfile a$i $O/forced-$ARM-h$i.json --slurpfile f$i $O/forced-fp8-h$i.json); done
B=$(jq -n "${args[@]}" '
  def one(a; f): [range(0; (a.argmax|length))] as $ix |
    { n: ($ix|length), nll_arm: ([$ix[] | a.nll[.]]|add), nll_ref: ([$ix[] | f.nll[.]]|add),
      t01: ($ix|map(select(f.margin[.] >= 0.1))|length), a01: ($ix|map(select(f.margin[.] >= 0.1 and a.argmax[.] == f.ref_ids[.]))|length),
      t1: ($ix|map(select(f.margin[.] >= 1.0))|length), a1: ($ix|map(select(f.margin[.] >= 1.0 and a.argmax[.] == f.ref_ids[.]))|length),
      flips: ($ix|map(select(a.argmax[.] != f.ref_ids[.]))|length), same_ids: (a.ref_ids[:($ix|length)] == f.ref_ids[:($ix|length)]) };
  [one($a1[0];$f1[0]), one($a2[0];$f2[0]), one($a3[0];$f3[0]), one($a4[0];$f4[0]), one($a5[0];$f5[0])] as $g |
  { steps: ($g|map(.n)|add), forced_on_fp8_ids: ($g|all(.same_ids)),
    R1_nll_ratio: (($g|map(.nll_arm)|add) / ($g|map(.nll_ref)|add)),
    R2_pooled: (($g|map(.a01)|add) / ($g|map(.t01)|add)), R2_confident: (($g|map(.a1)|add) / ($g|map(.t1)|add)), per_prompt: $g } |
  . + { R1: (.R1_nll_ratio <= 1.07), R2: (.R2_pooled >= 0.98 and .R2_confident >= 0.998) } |
  . + { pass: (.steps == 2560 and .forced_on_fp8_ids and .R1 and .R2) }')
# per-step rows: set (A|B), arm nll, reference nll, base nll -> blocks of 64
rows() {
  for p in p1 p2 p3 p4 chat-max-24; do
    jq -r --slurpfile r $R/drift-$p.json --slurpfile b $O/forced-$BASE-$p.json '. as $a | range(0; ($a.nll|length)) | "A \($a.nll[.]) \($r[0].states[0].per_row.nll_ref[.]) \($b[0].nll[.])"' $O/forced-$ARM-$p.json
  done
  for i in 1 2 3 4 5; do
    jq -r --slurpfile f $O/forced-fp8-h$i.json --slurpfile b $O/forced-$BASE-h$i.json '. as $a | range(0; ($a.nll|length)) | "B \($a.nll[.]) \($f[0].nll[.]) \($b[0].nll[.])"' $O/forced-$ARM-h$i.json
  done
}
C=$(rows | gawk '
  { k = int((NR - 1) / 64); set[k] = $1; a[k] += $2; r[k] += $3; d[k] += $2 - $4; n[k]++ }
  function pct(v, m, q,   w) { asort(v, w); return w[int(q * (m - 1)) + 1] }
  END {
    nb = k + 1; srand(355); R = 10000
    for (s in set) { if (set[s] == "A") nA++; else nB++ }
    for (i = 0; i < nb; i++) { if (set[i] == "A") ia[++ca] = i; else ib[++cb] = i }
    for (t = 1; t <= R; t++) {
      sd = 0; sn = 0; for (j = 0; j < nb; j++) { q = int(rand() * nb); sd += d[q]; sn += n[q] } md[t] = sd / sn
      sa = 0; sr = 0; for (j = 1; j <= ca; j++) { q = ia[int(rand() * ca) + 1]; sa += a[q]; sr += r[q] } ra[t] = sa / sr
      sa = 0; sr = 0; for (j = 1; j <= cb; j++) { q = ib[int(rand() * cb) + 1]; sa += a[q]; sr += r[q] } rb[t] = sa / sr
    }
    td = 0; tn = 0; for (i = 0; i < nb; i++) { td += d[i]; tn += n[i] }
    lo = pct(md, R, 0.025); hi = pct(md, R, 0.975)
    alo = pct(ra, R, 0.025); ahi = pct(ra, R, 0.975); blo = pct(rb, R, 0.025); bhi = pct(rb, R, 0.975)
    printf "{\"steps\":%d,\"blocks\":%d,\"mean_diff\":%.6f,\"diff_ci95\":[%.6f,%.6f],\"pass\":%s,\"R1_ci95_A\":[%.4f,%.4f],\"R1_ci95_B\":[%.4f,%.4f]}\n", tn, nb, td / tn, lo, hi, (hi < 0 ? "true" : "false"), alo, ahi, blo, bhi
  }')
jq -n --argjson a "$A" --argjson b "$B" --argjson c "$C" --arg arm $ARM --arg base $BASE '
  { arm: $arm, base: $base, A: $a, B: $b, C: $c,
    verdict: (if ($a.verdict | startswith("VOID")) then "VOID" elif ($a.verdict == "ACCEPT" and $b.pass and $c.pass) then "ACCEPT" else "REJECT" end) }'
