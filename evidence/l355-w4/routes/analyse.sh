#!/bin/bash
# L355 NV16 go/no-go (a), w4-gate.md: pooled over gen-p4 + chat, per step: layers whose top-8 set differs (w4e vs FP8) and
# excess = NLL(w4e) - NLL(FP8). NO-GO if mean differing layers at the 20 largest-excess steps >= 2x the rest AND they carry >= half.
cd $(dirname $0)
for p in p4 chat-max-24; do
  jq -r --slurpfile f forced-fp8-$p.json --arg p $p '. as $a | range(0; ($a.nll|length)) as $i |
    ([range(0; ($a.routes[$i]|length))] | map(select(($a.routes[$i][.]|sort) != ($f[0].routes[$i][.]|sort))) | length) as $d |
    "\($p) \($i) \($a.nll[$i] - $f[0].nll[$i]) \($d)"' forced-w4e-$p.json
done > steps.txt
sort -k3,3 -g -r steps.txt | awk -v N=20 '
  { n++; ex[n] = $3; d[n] = $4; tot += $3 }
  END { for (i = 1; i <= n; i++) { if (i <= N) { tx += ex[i]; td += d[i] } else { rd += d[i] } }
        mt = td / N; mr = rd / (n - N); share = tx / tot
        nogo = (mt >= 2 * mr && share >= 0.5)
        printf "{\"steps\":%d,\"total_excess\":%.4f,\"top20_excess\":%.4f,\"top20_share\":%.4f,\"mean_diff_layers_top20\":%.3f,\"mean_diff_layers_rest\":%.3f,\"ratio\":%.3f,\"verdict\":\"%s\"}\n", n, tot, tx, share, mt, mr, (mr > 0 ? mt / mr : 0), (nogo ? "NO-GO (router flips)" : "GO (not router-flip dominated)") }'
