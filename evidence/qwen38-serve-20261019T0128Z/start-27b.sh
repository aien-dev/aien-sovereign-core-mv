#!/bin/bash
# Start the Qwen3.8-27B server (our aien-serve-http on Omega) detached on 127.0.0.1:18091.
# Binary pinned in ~/.local/state/aien-q38/bin (sha256 in SHA256SUMS), built from sc qwen38/serve.
set -euo pipefail
S=$HOME/.local/state/aien-q38
cd "$S/bin"
sha256sum -c SHA256SUMS >/dev/null
echo "$(date -u +%FT%TZ) start 27B" >> "$S/serve.log"
setsid nohup ./aien-serve-http --engine omega --model-dir "$HOME/models/qwen3.8-27b-fp8-017b9c7a" \
  --port 18091 --max-context 131072 --max-tokens-cap 16384 --thinking off >> "$S/serve.log" 2>&1 < /dev/null &
echo $! > "$S/serve.pid"
echo "pid $(cat "$S/serve.pid"), log $S/serve.log"
