#!/usr/bin/env bash
# summarize.sh <session-dir>: one line per good call, speeds from the server log only (decode time; lock waits and
# client wall excluded). Batch door: a spell = first arrival to last leave with the chip lock held; aggregate tok/s =
# tokens of the call / spell seconds (prefill included); per-stream tok/s = that stream's tokens / its decode ms;
# per-step terms = spell totals / steps. One-at-a-time door: decode tok / decode ms.
set -euo pipefail
S=$1
printf "%-12s %3s %6s %5s %6s %8s %8s %8s %7s %7s %7s %7s %7s %7s %8s %8s\n" call n tokens steps rows ms/step gemm chipgemm a.b gdn attn head host spell_s agg_tok/s per_tok/s
awk '$2 !~ /FAILED/ {print $1, $2}' "$S/good.txt" | while read -r L A; do
  awk -v call="$L" '
    /qwen38-batch: leave:/ {
      match($0, /decode [0-9]+ tok in [0-9]+ ms/); split(substr($0, RSTART, RLENGTH), d, " ")
      tok += d[2]; if (d[5] > 0) { per += d[2] / (d[5] / 1000); np++ }
    }
    /qwen38-batch: spell over:/ {
      line = $0
      match(line, /spell over: [0-9]+ steps, [0-9]+ rows, [0-9.]+ ms\/step/); split(substr(line, RSTART, RLENGTH), a, " ")
      steps = a[3]; rows = a[5]; msps = a[7]
      g = val(line, "chip gemm "); gd = val(line, "gdn "); at = val(line, "attn "); hd = val(line, "head "); ho = val(line, "host ")
      sp = val(line, "spell "); cg = val(line, "gemm launches "); ab = val(line, "a\\|b ")
      spells++
    }
    /qwen38: lock wait/ {
      match($0, /decode [0-9]+ tok in [0-9]+ ms/); split(substr($0, RSTART, RLENGTH), d, " ")
      single = 1; tok = d[2]; dms = d[5]
      match($0, /wall [0-9.]+ ms\/token/); split(substr($0, RSTART, RLENGTH), w, " ")
    }
    function val(l, key,   t) { if (!match(l, key "[0-9.]+")) return -1; t = substr(l, RSTART, RLENGTH); sub(key, "", t); return t + 0 }
    END {
      if (single) { printf "%-12s %3d %6d %5s %6s %8.1f %8s %8s %7s %7s %7s %7s %7s %7.1f %8.3f %8.3f  single door\n", call, 1, tok, "-", "-", w[2], "-", "-", "-", "-", "-", "-", "-", dms / 1000, tok / (dms / 1000), tok / (dms / 1000); exit }
      if (spells != 1) { printf "%-12s spells %d in this call: not one spell, not summarised\n", call, spells; exit }
      f = 1.0 / steps
      printf "%-12s %3d %6d %5d %6.2f %8.1f %8.1f %8.1f %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f %8.3f %8.3f\n", call, np, tok, steps, rows / steps, msps, g * f, cg * f, ab * f, gd * f, at * f, hd * f, ho * f, sp, tok / sp, per / np
    }' "$S/$A.server.txt"
done
