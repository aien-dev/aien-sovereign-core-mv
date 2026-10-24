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

# OWN-30B cut 8 serve smoke: talks to a running aien-serve-http on the loopback and records
# each exchange with wall-clock timestamps (time-to-first-token and tok/s come from these).
#
#   scripts/own30b-serve-smoke.sh <evidence-dir> <phase> [port]
#     phase warm : /v1/models, one non-streaming, one streaming, then a 2-turn conversation
#                  with a ~1500-token system prompt (turn 2 should reuse turn 1's KV)
#     phase cold : only turn 2 of that conversation (run against a freshly started server)
# Writes <dir>/<phase>-*.json|.sse|.time and refreshes <dir>/SHA256SUMS.
set -euo pipefail
dir=$1 phase=$2 port=${3:-18091}
base="http://127.0.0.1:$port/v1"
mkdir -p "$dir"
now() { date +%s.%N; }

# A long, dull system prompt (about 1500 tokens): the same paragraph numbered many times.
sys_prompt() {
	local i
	echo "You are the OWN-30B assistant on the AIEN Spark. Answer in plain English, briefly. Rules of the house follow."
	for i in $(seq 1 60); do
		echo "Rule $i: keep answers short, never invent facts, say UNKNOWN when unsure, and prefer numbered steps over prose."
	done
}
SYS=$(sys_prompt)
json_str() { jq -Rn --arg s "$1" '$s'; }
SYSJ=$(json_str "$SYS")

post() { # name, body -> writes name.json and name.time
	local name=$1 body=$2 t0 t1
	t0=$(now)
	curl -s "$base/chat/completions" -H 'content-type: application/json' -d "$body" -o "$dir/$name.json"
	t1=$(now)
	printf 'start %s\nend %s\nwall_s %s\n' "$t0" "$t1" "$(echo "$t1 - $t0" | bc -l)" > "$dir/$name.time"
	jq -c '{finish: .choices[0].finish_reason, usage, text: .choices[0].message.content}' "$dir/$name.json"
}

post_stream() { # name, body -> name.sse (timestamp per line), name.time
	local name=$1 body=$2 t0 t1
	t0=$(now)
	curl -sN "$base/chat/completions" -H 'content-type: application/json' -d "$body" \
		| while IFS= read -r line; do printf '%s %s\n' "$(now)" "$line"; done > "$dir/$name.sse"
	t1=$(now)
	local first last n
	first=$(grep -m1 '"content":"[^"]' "$dir/$name.sse" | cut -d' ' -f1 || true)
	last=$(grep '"content":"[^"]' "$dir/$name.sse" | tail -1 | cut -d' ' -f1 || true)
	n=$(grep -c '"content":"[^"]' "$dir/$name.sse" || true)
	printf 'start %s\nfirst_token %s\nlast_token %s\nend %s\nttft_s %s\nchunks %s\ndecode_tok_s %s\n' \
		"$t0" "$first" "$last" "$t1" "$(echo "$first - $t0" | bc -l)" "$n" \
		"$(echo "($n - 1) / ($last - $first)" | bc -l)" > "$dir/$name.time"
	cat "$dir/$name.time"
}

if [ "$phase" = warm ]; then
	curl -s "$base/models" > "$dir/models.json"; jq -c . "$dir/models.json"
	post nonstream '{"model":"qwen3-30b-a3b-instruct-2507","temperature":0,"max_tokens":48,"messages":[{"role":"user","content":"In one sentence, what is a Grace Blackwell GB10?"}]}'
	post_stream stream '{"model":"qwen3-30b-a3b-instruct-2507","temperature":0,"max_tokens":48,"stream":true,"messages":[{"role":"user","content":"Count from one to ten in words, comma separated."}]}'
	post_stream sampled '{"model":"qwen3-30b-a3b-instruct-2507","temperature":0.7,"top_p":0.8,"seed":7,"max_tokens":32,"stream":true,"messages":[{"role":"user","content":"Name three colours."}]}'
fi

# The 2-turn conversation. Turn 1 reply is read back from turn1.json (warm) or from the
# warm run's turn1.json (cold), so both phases send byte-identical turn 2 prompts.
U1="Give me three numbered steps for checking a bolt is torqued correctly."
U2="Now the same for a hose clamp, two steps only."
if [ "$phase" = warm ]; then
	post_stream turn1 "{\"model\":\"qwen3-30b-a3b-instruct-2507\",\"temperature\":0,\"max_tokens\":96,\"stream\":true,\"messages\":[{\"role\":\"system\",\"content\":$SYSJ},{\"role\":\"user\",\"content\":$(json_str "$U1")}]}"
	A1=$(grep '"content":"' "$dir/turn1.sse" | sed 's/^[0-9.]* data: //' | jq -j '.choices[0].delta.content // empty')
	printf '%s' "$A1" > "$dir/turn1.reply.txt"
fi
A1=$(cat "$dir/turn1.reply.txt")
post_stream "$phase-turn2" "{\"model\":\"qwen3-30b-a3b-instruct-2507\",\"temperature\":0,\"max_tokens\":64,\"stream\":true,\"messages\":[{\"role\":\"system\",\"content\":$SYSJ},{\"role\":\"user\",\"content\":$(json_str "$U1")},{\"role\":\"assistant\",\"content\":$(json_str "$A1")},{\"role\":\"user\",\"content\":$(json_str "$U2")}]}"
grep '"content":"' "$dir/$phase-turn2.sse" | sed 's/^[0-9.]* data: //' | jq -j '.choices[0].delta.content // empty' > "$dir/$phase-turn2.reply.txt"
(cd "$dir" && find . -type f ! -name SHA256SUMS | sort | sed "s|^./||" | xargs sha256sum > SHA256SUMS)
echo "smoke $phase done"
