#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Drake Stapleton (AIEN / aien-dev)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# (Re)start the OWN-30B Omega server in the background (OWN-30B cut 8; default binary: the cut 10 build with the device router and bf16 lm_head GEMV,
# on top of the fused one-launch attention and batched prefill).
#   scripts/own30b-serve-restart.sh [binary]
# Stops the previous server only if /tmp/own30b-serve.pid names a live aien-serve-http,
# then starts a fresh one detached: log /tmp/own30b-serve.log, PID /tmp/own30b-serve.pid.
# The server loads FP8-all weights (about 18 s, 30.5 GB resident), serves 127.0.0.1:18091,
# and takes /tmp/aien-gb10.lock only while loading and around each generation.
set -euo pipefail
BIN=${1:-/tmp/own30b-f/target-router/release/aien-serve-http}
MODEL=${OWN30B_MODEL_DIR:-$HOME/models/qwen3-30b-a3b-instruct-2507-fp8}
PIDF=/tmp/own30b-serve.pid LOG=/tmp/own30b-serve.log
if [ -f "$PIDF" ]; then
	old=$(cat "$PIDF")
	if [ -n "$old" ] && [ "$(ps -o comm= -p "$old" 2>/dev/null || true)" = aien-serve-http ]; then
		kill "$old"
		for _ in $(seq 1 100); do kill -0 "$old" 2>/dev/null || break; sleep 0.1; done
	fi
fi
# sh writes its own PID then exec's the server, so the file is right whether or not setsid forks.
setsid sh -c "echo \$\$ > $PIDF; exec '$BIN' --engine omega --model-dir '$MODEL' --port 18091" > "$LOG" 2>&1 < /dev/null &
for _ in $(seq 1 120); do
	grep -q "serving\|load failed\|error" "$LOG" 2>/dev/null && break
	sleep 1
done
tail -2 "$LOG"
echo "pid $(cat $PIDF) ($(ps -o comm= -p "$(cat $PIDF)" 2>/dev/null || echo not-running))"
