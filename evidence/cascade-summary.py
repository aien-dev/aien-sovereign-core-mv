#!/usr/bin/env python3
"""Summarise qwen38_spec gen JSON files: accepted tokens per pass and projected tok/s at the GEMM target.
Usage: cascade-summary.py DIR [DIR...]   (reads every <name>.json that has a runs list)
Projection: tok/s = tokens_per_pass / pass_ms. pass_ms is NOT the measured (loaded box) number; two stated bases:
  plain decode on a quiet box 158.9 ms (qwen38-serve handoff, median of 3 runs of 158.6/158.9/160.3 ms)
  weight-read ceiling 98.5 ms (10.15 tok/s, 26.9 GB at 273 GB/s)
The GEMM target is that an N-row verify pass costs about one plain pass (each weight tile read once)."""
import json, sys, glob, os
BASES = {"plain158.9": 158.9, "ceiling98.5": 98.5}
rows = []
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(f).startswith("receipt-"): continue
        try: j = json.load(open(f))
        except Exception: continue
        if "runs" not in j: continue
        for r in j["runs"]:
            tpp = r["tokens_per_round"]
            rows.append((os.path.basename(f)[:-5], j.get("prompt"), j.get("mutant"), r["k"], r["generated"], r["rounds"], tpp,
                         r["accept_rate"], r.get("ngram_accepted", 0), r["decode_ms_per_token"], j.get("pass")))
print("run,prompt,mutant,k,tokens,passes,tokens_per_pass,accept_rate,ngram_accepted,measured_ms_per_token_loaded,pass_flag," + ",".join("proj_toks_" + b for b in BASES))
for r in rows:
    print(",".join(str(x) if not isinstance(x, float) else "%.3f" % x for x in r) + "," + ",".join("%.2f" % (r[6] / (ms / 1000)) for ms in BASES.values()))
