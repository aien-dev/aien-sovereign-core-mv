#!/bin/sh
# Summarise the speed2 qwen38_spec suite JSON files (CSV on stdout).
# Usage: summary.sh WALL_MS FILE.json...
#   WALL_MS: the binding wall of one pass at concurrency 1 = resident weight bytes / 273 GB/s (the
#   chip's memory bandwidth), in ms; give the figure for the configuration the files ran (FP8 or bf16 head).
# Per run: words per pass (tokens per verify pass), measured decode ms per token and tok/s,
# the wall tok/s at that words-per-pass (words_per_pass / WALL_MS), and the measured fraction of it.
# identical_to_k0 is the tool's own check (every k's tokens equal the k = 0 run of the same file).
W=$1; shift
echo "file,prompt,verify,mutant,k,tokens,passes,words_per_pass,ms_per_token,tok_s,wall_tok_s,fraction_of_wall,identical_to_k0"
for f in "$@"; do
  jq -r --arg n "$(basename "$f" .json)" --argjson w "$W" 'select(.runs != null) | . as $j | .runs[] |
    [$n, $j.prompt, ($j.spec_options | capture("kernel: (?<k>[A-Za-z]+)").k), ($j.mutant // ""), .k, .generated, .rounds,
     (.words_per_pass * 1000 | round / 1000), (.decode_ms_per_token * 100 | round / 100),
     (1000 / .decode_ms_per_token * 100 | round / 100),
     (.words_per_pass * 1000 / $w * 100 | round / 100),
     ((1000 / .decode_ms_per_token) / (.words_per_pass * 1000 / $w) * 1000 | round / 1000), $j.pass] | @csv' "$f"
done
