#!/bin/sh
# Summarise qwen38_spec gen/suite JSON files: accepted tokens per pass and projected tok/s at the GEMM target.
# Usage: cascade-summary.sh DIR [DIR...]   (reads every <name>.json that has a runs list; CSV on stdout)
# Projection: tok/s = tokens_per_pass / pass_ms. pass_ms is NOT the measured (loaded box) number; two stated bases:
#   plain decode on a quiet box 158.9 ms (qwen38-serve handoff, median of 3 runs of 158.6/158.9/160.3 ms)
#   weight-read ceiling 98.5 ms (10.15 tok/s, 26.9 GB at 273 GB/s)
# The GEMM target is that an N-row verify pass costs about one plain pass (each weight tile read once).
echo "run,prompt,mutant,k,tokens,passes,tokens_per_pass,accept_rate,ngram_accepted,measured_ms_per_token_loaded,pass_flag,proj_toks_plain158.9,proj_toks_ceiling98.5"
for d in "$@"; do
  for f in "$d"/*.json; do
    case $(basename "$f") in receipt-*) continue ;; esac
    jq -r --arg n "$(basename "$f" .json)" 'select(.runs != null) | . as $j | .runs[] |
      [$n, $j.prompt, ($j.mutant // ""), .k, .generated, .rounds, (.tokens_per_round * 1000 | round / 1000),
       (.accept_rate * 1000 | round / 1000), (.ngram_accepted // 0), (.decode_ms_per_token * 1000 | round / 1000), $j.pass,
       (.tokens_per_round / 0.1589 * 100 | round / 100), (.tokens_per_round / 0.0985 * 100 | round / 100)] | @csv' "$f" 2>/dev/null
  done
done
