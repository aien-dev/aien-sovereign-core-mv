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

# OWN-8B stage S sampling gate: is seeded sampling observed and honoured by a running
# aien-serve-http? Shared by the 30B (:18091) and the 8B (cut 2, cut 3); only the URL and the
# model id differ. Needs a server that returns `usage.aien_sampling` (stage S or later).
#
#   scripts/sampling-gate.sh <base-url> <model-id> <evidence-dir> [repro.req.json ...]
#     <base-url>  e.g. http://127.0.0.1:18091 (no /v1)
#   env GREEDY_GOLDEN  optional own30b-serve-smoke.sh nonstream.json: gate (c) also requires the
#                      greedy reply to the shared prompt (prompt 1) to equal its content
#   env REDACT_REPRO=0 keep the repro replies' text (default 1: only usage, finish, content sha256,
#                      because the repro prompts belong to another experiment)
#
# Gates (design 2026-10-15-own8b-design.md section 4.1), 4 fixed prompts, max_tokens 48, stream false.
# Contents are compared byte for byte (sha256 of the exact content string).
#   (a) replay: seed 1, temperature 1.0, top_p 1.0, top_k 0, twice: identical, 4/4
#   (b) spread: seeds 1, 2, 3 at the same settings, 4/4 prompts. Every reply must report
#       mode=sampled and the seed it was sent. A prompt passes with at least 2 distinct replies,
#       or with 3 identical replies whose p_repeat^2 >= 0.01 (an honest sampler repeats them with
#       at least 1% chance: the model is confident, as for a short list prompt).
#   (c) greedy: temperature 0, seeds 1 and 2: identical 4/4 (and prompt 1 equal to GREEDY_GOLDEN)
#   (d) repro: each repro request re-sent as is (its own seed, temperature, top_p) with the
#       record. PASS when the replies differ, or when they are identical, report their own seeds
#       and mode=sampled, and p_repeat^(k-1) >= 0.01 (confidence). Otherwise FAIL (the seed is not
#       honoured). No repro files: (d) is NOT_RUN and does not change the verdict.
# p_repeat (from the record) is the chance that an honest draw with another seed repeats the
# reply's sampled draws exactly. Writes <dir>/<gate>-*.json, <dir>/gate.txt (one line per check,
# then VERDICT) and SHA256SUMS. Exit 0 PASS, 1 FAIL, 2 usage or transport error.
set -euo pipefail
[ $# -ge 3 ] || { sed -n '2,29p' "$0"; exit 2; }
base=${1%/} model=$2 dir=$3
shift 3
repro=("$@")
redact=${REDACT_REPRO:-1}
mkdir -p "$dir"
out="$dir/gate.txt"
: > "$out"
say() { echo "$*" | tee -a "$out"; }

PROMPTS=(
	"In one sentence, what is a Grace Blackwell GB10?"
	"Count from one to ten in words, comma separated."
	"Name three colours."
	"Give me three numbered steps for checking a bolt is torqued correctly."
)

# post NAME BODY: writes NAME.json; retries a 503 server_busy (Retry-After) up to 60 times.
post() {
	local name=$1 body=$2 code i
	for i in $(seq 1 60); do
		code=$(curl -s -o "$dir/$name.json" -w '%{http_code}' "$base/v1/chat/completions" \
			-H 'content-type: application/json' -d "$body") || { echo "sampling-gate: curl failed for $name" >&2; exit 2; }
		[ "$code" = 503 ] || break
		sleep 1
	done
	[ "$code" = 200 ] || { echo "sampling-gate: $name answered HTTP $code: $(head -c 300 "$dir/$name.json")" >&2; exit 2; }
	jq -e '.usage.aien_sampling.mode' "$dir/$name.json" > /dev/null \
		|| { echo "sampling-gate: $name has no usage.aien_sampling (an engine that does not sample, e.g. the stub, or a server older than stage S)" >&2; exit 2; }
}
# Byte-exact content digest (jq -j: no added newline; the shell never strips anything).
digest() { jq -j '.choices[0].message.content' "$dir/$1.json" | sha256sum | cut -d' ' -f1; }
field() { jq -r ".usage.aien_sampling.$2" "$dir/$1.json"; }
body() { # prompt temperature top_p top_k seed
	jq -cn --arg m "$model" --arg p "$1" --argjson t "$2" --argjson tp "$3" --argjson k "$4" --argjson s "$5" \
		'{model: $m, max_tokens: 48, stream: false, temperature: $t, top_p: $tp, top_k: $k, seed: $s,
		  messages: [{role: "user", content: $p}]}'
}
rec() { # name -> one-line sampling summary
	jq -r '.usage.aien_sampling | "mode=\(.mode) seed=\(.seed) sampled=\(.sampled_tokens) greedy=\(.greedy_tokens) kept=\(.kept_min)/\(.kept_median)/\(.kept_mean)/\(.kept_max) kept1=\(.kept_single) top_prob=\(.top_prob_min)/\(.top_prob_mean) p_repeat=\(.p_repeat)" + (if .mutant then " MUTANT=\(.mutant)" else "" end)' "$dir/$1.json"
}
at_least() { awk -v x="$1" -v y="$2" 'BEGIN { exit !(x >= y) }'; } # x >= y (floats)

pass_a=0 pass_b=0 pass_c=0
for i in 0 1 2 3; do
	n=$((i + 1))
	p=${PROMPTS[$i]}
	# (a) replay
	post "a-p$n-r1" "$(body "$p" 1.0 1.0 0 1)"
	post "a-p$n-r2" "$(body "$p" 1.0 1.0 0 1)"
	if [ "$(digest "a-p$n-r1")" = "$(digest "a-p$n-r2")" ]; then pass_a=$((pass_a + 1)); ok=same; else ok=DIFFERENT; fi
	say "a p$n $ok | $(rec "a-p$n-r1")"
	# (b) spread
	honest=1
	for s in 1 2 3; do
		post "b-p$n-s$s" "$(body "$p" 1.0 1.0 0 "$s")"
		[ "$(field "b-p$n-s$s" mode)" = sampled ] && [ "$(field "b-p$n-s$s" seed)" = "$s" ] || honest=0
	done
	distinct=$(for s in 1 2 3; do digest "b-p$n-s$s"; done | sort -u | wc -l)
	p2=$(awk -v p="$(field "b-p$n-s1" p_repeat)" 'BEGIN { printf "%.6g", p * p }')
	if [ "$honest" = 0 ]; then
		why="FAIL (a reply was not sampled with its own seed)"
	elif [ "$distinct" -ge 2 ]; then
		why=PASS pass_b=$((pass_b + 1))
	elif at_least "$p2" 0.01; then
		why="PASS (identical, confident: p_repeat^2=$p2 >= 0.01)" pass_b=$((pass_b + 1))
	else
		why="FAIL (identical with p_repeat^2=$p2 < 0.01)"
	fi
	say "b p$n distinct=$distinct/3 $why | $(rec "b-p$n-s1") | $(rec "b-p$n-s2") | $(rec "b-p$n-s3")"
	# (c) greedy
	post "c-p$n-s1" "$(body "$p" 0 1.0 0 1)"
	post "c-p$n-s2" "$(body "$p" 0 1.0 0 2)"
	ok=1
	[ "$(digest "c-p$n-s1")" = "$(digest "c-p$n-s2")" ] || ok=0
	golden=""
	if [ "$n" = 1 ] && [ -n "${GREEDY_GOLDEN:-}" ]; then
		if [ "$(digest "c-p$n-s1")" = "$(jq -j '.choices[0].message.content' "$GREEDY_GOLDEN" | sha256sum | cut -d' ' -f1)" ]; then
			golden=" golden=equal"
		else
			golden=" golden=DIFFERENT" ok=0
		fi
	fi
	[ "$ok" = 1 ] && pass_c=$((pass_c + 1))
	say "c p$n $([ "$ok" = 1 ] && echo same || echo DIFFERENT)$golden | $(rec "c-p$n-s1")"
done

verdict=PASS
for g in a b c; do
	v=pass_$g
	r=$([ "${!v}" = 4 ] && echo PASS || echo FAIL)
	[ "$r" = PASS ] || verdict=FAIL
	say "gate $g ${!v}/4 $r"
done

# (d) repro
if [ ${#repro[@]} -eq 0 ]; then
	say "gate d NOT_RUN (no repro request files given)"
else
	k=0 honest=1 pmin=1
	hashes=()
	for f in "${repro[@]}"; do
		k=$((k + 1))
		post "d-r$k" "$(jq -c --arg m "$model" '.model = $m | .stream = false' "$f")"
		h=$(digest "d-r$k")
		hashes+=("$h")
		[ "$(field "d-r$k" mode)" = sampled ] && [ "$(field "d-r$k" seed)" = "$(jq -r .seed "$f")" ] || honest=0
		pr=$(field "d-r$k" p_repeat)
		at_least "$pr" "$pmin" || pmin=$pr
		say "d r$k $(jq -r '"seed=\(.seed) temperature=\(.temperature) top_p=\(.top_p) top_k=\(.top_k // "default")"' "$f") content_sha256=${h:0:16} | $(rec "d-r$k")"
		if [ "$redact" = 1 ]; then
			t=$(mktemp)
			jq --arg h "$h" '{usage, finish: .choices[0].finish_reason, content_sha256: $h}' "$dir/d-r$k.json" > "$t" \
				|| { rm -f "$t" "$dir/d-r$k.json"; echo "sampling-gate: redaction of d-r$k failed; reply removed" >&2; exit 2; }
			mv "$t" "$dir/d-r$k.json"
		fi
	done
	distinct=$(printf '%s\n' "${hashes[@]}" | sort -u | wc -l)
	# Kept-set sizes of every sampled step of every repro reply.
	kept_all() { for i in $(seq 1 "$k"); do jq -r '.usage.aien_sampling.kept[]' "$dir/d-r$i.json"; done; }
	mean=$(kept_all | awk '{s += $1; n++} END {if (n) printf "%.4f", s / n; else print "0"}')
	steps=$(kept_all | wc -l)
	choice=$(kept_all | awk '$1 > 1' | wc -l)
	chance=$(awk -v p="$pmin" -v k="$k" 'BEGIN { printf "%.6g", p ^ (k - 1) }')
	stats="kept_mean=$mean steps_with_choice=$choice/$steps p_repeat_min=$pmin"
	if [ "$honest" = 0 ]; then
		say "gate d FAIL a reply was not sampled with its own seed; $stats"
		verdict=FAIL
	elif [ "$distinct" -ge 2 ]; then
		say "gate d PASS distinct=$distinct/$k (the seeds produced different replies) $stats"
	elif at_least "$chance" 0.01; then
		say "gate d PASS identical replies explained by confidence: p_repeat^$((k - 1))=$chance >= 0.01, $stats"
	else
		say "gate d FAIL identical replies with p_repeat^$((k - 1))=$chance < 0.01: the seed is not honoured; $stats"
		verdict=FAIL
	fi
fi

say "VERDICT $verdict"
(cd "$dir" && find . -type f ! -name SHA256SUMS | sort | sed 's|^\./||' | xargs sha256sum > SHA256SUMS)
[ "$verdict" = PASS ]
