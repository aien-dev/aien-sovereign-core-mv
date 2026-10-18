#!/usr/bin/env bash
# run.sh NAME SLOTS PROMPTS : one paged OWN-30B tokens run under the GB10 lock (quietlock check first).
set -u
EV=$(cd "$(dirname "$0")" && pwd); NAME=$1; SLOTS=$2; PROMPTS=$3
B=$HOME/workspace/targets/ribbon-sc/release/own30b_check
CK=$HOME/models/qwen3-30b-a3b-instruct-2507-fp8
G=$HOME/workspace/sc-408/docs/inference/evidence/own30b-f-20261013T1745Z/golden.json
quietlock check || { echo "quiet flag raised: not running" > "$EV/$NAME.rc"; exit 3; }
{ echo "queued $(date -u +%FT%TZ) sc $(git -C "$EV" rev-parse HEAD) omega $(cat "$EV/../../../../omega.lock")"
  echo "binary sha256 $(sha256sum "$B" | cut -d' ' -f1)"; uptime; free -g | head -2; } > "$EV/$NAME.start"
flock /tmp/aien-gb10.lock sh -c 'echo "lock acquired $(date -u +%FT%TZ)" >> "$0"; shift; exec "$@"' "$EV/$NAME.start" x \
  env AIEN_OMEGA_SPIN_US=2000 "$B" chip "$CK" "$G" "$EV/$NAME" --tokens-only --prompts "$PROMPTS" --new 16 \
  --weights fp8 --attn fused --router device --lm-head gemv --prefill auto --paged-experts "$SLOTS" \
  > "$EV/$NAME.raw" 2> "$EV/$NAME.err"
echo $? > "$EV/$NAME.rc"; date -u +%FT%TZ > "$EV/$NAME.end"
