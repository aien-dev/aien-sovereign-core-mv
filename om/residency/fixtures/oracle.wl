(* SPDX-License-Identifier: Apache-2.0 *)
(* Independent LRU and Belady (OPT) oracle for om/residency/residency.om, in the Wolfram Language.
   residencyOracle[e, l, k, t, ids] returns the simulator's expected stdout for a trace with header
   E L K T and the flat step-major id list ids (step t, layer l, slot k). It is written from the
   residency.om contract in the file header, not from its code:
   - lru/opt: each layer replays alone from an empty cache of c = Floor[E p / 100] experts;
   - glru/gopt (only when L E <= 65536): one cache of c = Floor[L E p / 100] for all layers, steps
     taken in serving order (t, l) -> t L + l, virtual expert l E + e;
   - a line exists only when c >= K; a miss with c experts resident evicts the resident expert
     with the largest score that was not already used in the current step, lowest id on ties;
     LRU score: steps since last use; OPT score: step of next use (number of steps when none);
   - warm counts misses at steps s >= Floor[steps / 2]; warm_acc = accesses at those steps. *)

replay[steps_, n_, c_, pol_] := Module[{nT = Length[steps], nK = Length[First[steps]], last, nxt,
    stamp, nextUse, hits = 0, miss = 0, warm = 0, resident = 0, id, cand, sc, best},
  (* nxt[[s, j]]: 0-based step of the next use of steps[[s, j]] after step s, else nT *)
  last = ConstantArray[nT, n];
  nxt = ConstantArray[0, {nT, nK}];
  Do[
    Do[nxt[[s, j]] = last[[steps[[s, j]] + 1]], {j, nK}];
    Do[last[[steps[[s, j]] + 1]] = s - 1, {j, nK}],
    {s, nT, 1, -1}];
  stamp = ConstantArray[0, n];   (* 0: not resident; else 1-based step of last use *)
  nextUse = ConstantArray[0, n];
  Do[
    Do[
      id = steps[[s, j]] + 1;
      If[stamp[[id]] != 0, hits++,
        miss++;
        If[s - 1 >= Floor[nT/2], warm++];
        If[resident >= c,
          cand = Select[Range[n], stamp[[#]] != 0 && stamp[[#]] != s &];
          sc = If[pol == 0, s - stamp[[cand]], nextUse[[cand]]];
          best = cand[[First[FirstPosition[sc, Max[sc]]]]];
          stamp[[best]] = 0,
          resident++]];
      stamp[[id]] = s;
      nextUse[[id]] = nxt[[s, j]],
      {j, nK}],
    {s, nT}];
  {hits, miss, warm}];

line[tag_, c_, p_, {h_, m_, w_}, acc_] := StringJoin[tag, " cap=", ToString[c], " pct=", ToString[p],
  " hits=", ToString[h], " miss=", ToString[m], " warm_miss=", ToString[w], " warm_acc=", ToString[acc]];

residencyOracle[e_, l_, k_, t_, ids_] := Module[{pcts = {25, 40, 50, 60, 64, 70, 80, 90},
    tr = ArrayReshape[ids, {t, l, k}], out, g, gsteps},
  out = {StringJoin["trace E=", ToString[e], " L=", ToString[l], " K=", ToString[k], " T=", ToString[t]]};
  Do[
    With[{c = Floor[e p/100]},
      If[c >= k, AppendTo[out, line[{"lru", "opt"}[[pol + 1]], c, p,
        Total[Table[replay[tr[[All, ll]], e, c, pol], {ll, l}]], l k (t - Floor[t/2])]]]],
    {pol, {0, 1}}, {p, pcts}];
  If[l e <= 65536,
    g = l e;
    gsteps = Flatten[Table[tr[[tt, ll]] + (ll - 1) e, {tt, t}, {ll, l}], 1];
    Do[
      With[{c = Floor[g p/100]},
        If[c >= k, AppendTo[out, line[{"glru", "gopt"}[[pol + 1]], c, p,
          replay[gsteps, g, c, pol], k (t l - Floor[t l/2])]]]],
      {pol, {0, 1}}, {p, pcts}]];
  StringJoin[Riffle[out, FromCharacterCode[10]], FromCharacterCode[10]]];
