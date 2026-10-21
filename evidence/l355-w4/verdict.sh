#!/bin/bash
# L355 verdict by docs/inference/q36/w4-gate.md (frozen 02:15Z). Usage: verdict.sh <arm: w4e|w4et> [mutant-arm]
# Reference per golden = L313 drift-<p>.json (f32 arm: f32_ids, f32_margin, states[0].per_row.nll_ref); equals my fp8 arm on p1
# to the last bit (forced-fp8-p1). AGPL: score-fp8 vs score-<arm>.
set -u
ARM=${1:?arm}; R=$HOME/workspace/sc-hum-state-precision/evidence/hum-state-precision; O=runs
cd $HOME/workspace/sc-l355/evidence/l355-w4
args=(); for p in p1 p2 p3 p4 chat-max-24; do args+=(--slurpfile a_$(echo $p | tr - _) $O/forced-$ARM-$p.json --slurpfile r_$(echo $p | tr - _) $R/drift-$p.json); done
jq -n "${args[@]}" --slurpfile f $O/forced-fp8-p1.json --slurpfile sf $( [ -s $O/score-fp8.json ] && echo $O/score-fp8.json || echo /dev/null) --slurpfile sa $( [ -s $O/score-$ARM.json ] && echo $O/score-$ARM.json || echo /dev/null) --arg arm $ARM '
  def one(a; r): [range(0; (a.argmax|length))] as $ix |
    { n: ($ix|length), nll_arm: ([$ix[] | a.nll[.]]|add), nll_ref: ([$ix[] | r.states[0].per_row.nll_ref[.]]|add),
      t01: ($ix|map(select(r.f32_margin[.] >= 0.1))|length), a01: ($ix|map(select(r.f32_margin[.] >= 0.1 and a.argmax[.] == r.f32_ids[.]))|length),
      t1: ($ix|map(select(r.f32_margin[.] >= 1.0))|length), a1: ($ix|map(select(r.f32_margin[.] >= 1.0 and a.argmax[.] == r.f32_ids[.]))|length),
      flips: ($ix|map(select(a.argmax[.] != r.f32_ids[.]))|length), first_flip: ([$ix[]|select(a.argmax[.] != r.f32_ids[.])][0]) };
  { p1: one($a_p1[0]; $r_p1[0]), p2: one($a_p2[0]; $r_p2[0]), p3: one($a_p3[0]; $r_p3[0]), p4: one($a_p4[0]; $r_p4[0]), chat: one($a_chat_max_24[0]; $r_chat_max_24[0]) } as $g |
  [$g[]] as $all |
  ($f[0].argmax as $fa | $r_p1[0] as $r | [range(0;512)] | {ids_equal: (map(select($fa[.] == $r.f32_ids[.]))|length), max_margin_diff: (map(($f[0].margin[.] - $r.f32_margin[.])|fabs)|max)}) as $ctl |
  { arm: $arm, steps: ($all|map(.n)|add),
    control: ($ctl + {pass: ($ctl.ids_equal == 512 and $ctl.max_margin_diff <= 0.001)}),
    R1_nll_ratio: (($all|map(.nll_arm)|add) / ($all|map(.nll_ref)|add)),
    R1b_agpl_rise: (if ($sa|length) > 0 and ($sf|length) > 0 then $sa[0].mean_nll - $sf[0].mean_nll else null end),
    R2_pooled: (($all|map(.a01)|add) / ($all|map(.t01)|add)),
    R2_confident: (($all|map(.a1)|add) / ($all|map(.t1)|add)),
    per_golden: $g } |
  . + { R1: (.R1_nll_ratio <= 1.07), R1b: (if .R1b_agpl_rise == null then null else .R1b_agpl_rise <= 0.069 end), R2: (.R2_pooled >= 0.98 and .R2_confident >= 0.998) } |
  . + { verdict: (if (.control.pass|not) then "VOID (exactness control failed)" elif (.steps < 2560 or .R1b == null) and .R1 != false and .R2 != false then "INCOMPLETE" elif (.R1 and .R1b and .R2) then "ACCEPT" else "REJECT" end) }'
