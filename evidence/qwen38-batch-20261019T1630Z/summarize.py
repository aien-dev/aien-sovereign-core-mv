#!/usr/bin/env python3
"""summarize.py <batch-dir>: speeds from the server log only (decode time, lock wait and client wall excluded).
A spell = one run of the scheduler with the chip lock held (lines "leave:" ... "spell over:"). The timed
spells are the last 8 (n=1, prompts 0-7) plus 3 rounds of n=2,4,8 in run order; earlier spells are the
correctness pass (c-*) and are not published as speeds. Per user tok/s = that stream tokens / its decode ms;
aggregate tok/s = all tokens of the spell / spell seconds (first arrival to last leave, prefill included)."""
import re, sys, json, statistics as st
log = open(sys.argv[1] + "/server.log").read().splitlines()
spells, cur = [], []
for l in log:
    m = re.search(r"leave: prompt (\d+) tok, prefill (\d+) ms, decode (\d+) tok in (\d+) ms over (\d+) steps, mean rows ([\d.]+)", l)
    if m: cur.append(tuple(float(x) for x in m.groups()))
    m = re.search(r"spell over: (\d+) steps, (\d+) rows, ([\d.]+) ms/step wall, chip gemm (\d+) ms, gdn (\d+) ms, attn (\d+) ms, head (\d+) ms, host (\d+) ms, spell ([\d.]+) s", l)
    if m:
        spells.append((cur, [float(x) for x in m.groups()])); cur = []
print("spells in log:", len(spells), "(timed = last 17)")
timed = spells[-17:]
names = [f"n1-p{i}" for i in range(8)] + [f"n{n}-r{r}" for r in (1,2,3) for n in (2,4,8)]
rows_out = []
for name, (leaves, sp) in zip(names, timed):
    toks = sum(x[2] for x in leaves)
    per_user = [x[2] / (x[3] / 1000) for x in leaves]
    agg = toks / sp[8]
    rows_out.append(dict(call=name, n=len(leaves), tokens=int(toks), steps=int(sp[0]), mean_rows=round(sp[1]/sp[0],2),
        ms_per_step=sp[2], gemm_ms_per_step=round(sp[3]/sp[0],1), head_ms_per_step=round(sp[6]/sp[0],1),
        spell_s=sp[8], agg_tok_s=round(agg,3), per_user_tok_s_mean=round(st.mean(per_user),3)))
    print(rows_out[-1])
by = {}
for r in rows_out: by.setdefault(r["n"], []).append(r)
print("\nn  calls  agg_tok_s(median, min-max)  per_user_tok_s(median)  ms/step(median)  gemm_ms/step")
for n in sorted(by):
    g = by[n]
    print(n, len(g), st.median(x["agg_tok_s"] for x in g), (min(x["agg_tok_s"] for x in g), max(x["agg_tok_s"] for x in g)),
          st.median(x["per_user_tok_s_mean"] for x in g), st.median(x["ms_per_step"] for x in g), st.median(x["gemm_ms_per_step"] for x in g))
json.dump(rows_out, open(sys.argv[1] + "/summary.json", "w"), indent=1)
