#!/usr/bin/env bash
# tools/rig/http-gate.sh: correctness and speed of the dense Qwen3 server THROUGH HTTP (aien-rig), on the
# DGX Spark (GB10). Starts our own aien-serve-http on a spare port (never 18091), sends greedy chat requests,
# reads the exact prompt and reply token ids from AIEN_RIG_IDS_LOG, stops the server, then recomputes the
# same prompts on the host reference (own8b_check golden --prompt-ids, the same weights: FP8 or the same MXFP4
# bytes) and requires the reply ids to be identical. A comparator control (one reference id changed) must
# report MISMATCH, so the gate cannot pass vacuously.
#
#   tools/rig/http-gate.sh <evidence-dir> <name> <port> <checkpoint> [--w4] -- <extra server flags...>
#
# Env: BIN (aien-serve-http), CHECK (own8b_check), NEW (reply tokens per gate prompt, default 64),
#      SPEED_TOKENS (default 256), SPEED_RUNS (default 3). Writes <evidence-dir>/<name>/.
set -uo pipefail
ev=$1; name=$2; port=$3; ckpt=$4; shift 4
w4=()
if [[ ${1:-} == --w4 ]]; then w4=(--w4); shift; fi
[[ ${1:-} == -- ]] && shift
[[ $port == 18091 ]] && { echo "port 18091 is the live server: refused"; exit 2; }
BIN=${BIN:?BIN=aien-serve-http path}; CHECK=${CHECK:?CHECK=own8b_check path}
NEW=${NEW:-64}; SPEED_TOKENS=${SPEED_TOKENS:-256}; SPEED_RUNS=${SPEED_RUNS:-3}
d="$ev/$name"; mkdir -p "$d"; ids="$d/ids.jsonl"; : > "$ids"
repo=$(cd "$(dirname "$0")/../.." && pwd)
{
	echo "start $(date -u +%FT%TZ)"
	echo "chip GB10 (DGX Spark), NOT an RTX 5070"
	echo "sc $(git -C "$repo" rev-parse HEAD) dirty $(git -C "$repo" status --porcelain | wc -l)"
	echo "binary $BIN sha256 $(sha256sum "$BIN" | cut -d' ' -f1)"
	echo "server: $BIN --engine omega --model-dir $ckpt --port $port ${w4[*]:+--weights w4} $*"
	echo "env AIEN_Q38_W4_RULE=${AIEN_Q38_W4_RULE:-unset}"
	echo "loadavg $(cat /proc/loadavg)"; grep MemAvailable /proc/meminfo
} > "$d/ground.txt"

# The server takes the shared chip lock itself (at load and per request): never wrap it in flock, that
# deadlocks (measured: the wrapped server sat 21 minutes in flock(2) on its own lock).
# The load can fail with NV_ERR_NO_MEMORY on the chip channel when other jobs hold system memory at that
# moment (the GB10 shares one memory pool; measured twice). Retry the whole start, each time after
# MemFree (not just MemAvailable, page cache does not help the driver) is at least FREE_GIB.
FREE_GIB=${FREE_GIB:-16}; ATTEMPTS=${ATTEMPTS:-6}
pid=; stop() { [[ -n $pid ]] && { kill -TERM "$pid" 2>/dev/null; for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done; }; }
trap stop EXIT
up=0
for try in $(seq 1 "$ATTEMPTS"); do
	while (( $(awk '/MemFree/{print $2}' /proc/meminfo) < FREE_GIB * 1024 * 1024 )); do sleep 20; done
	echo "attempt $try start $(date -u +%FT%TZ) MemFree $(awk '/MemFree/{print int($2/1048576)}' /proc/meminfo) GiB" >> "$d/ground.txt"
	AIEN_RIG_IDS_LOG="$ids" setsid nohup "$BIN" --engine omega --model-dir "$ckpt" --port "$port" \
		${w4[*]:+--weights w4} "$@" > "$d/server.log" 2>&1 < /dev/null &
	pid=$!; echo "$pid" > "$d/server.pid"
	for _ in $(seq 1 7200); do
		curl -sf "http://127.0.0.1:$port/health" > "$d/health.json" 2>/dev/null && { up=1; break; }
		kill -0 "$pid" 2>/dev/null || break
		sleep 1
	done
	(( up )) && break
	echo "attempt $try: server exited before serving: $(tail -1 "$d/server.log" | cut -c1-200)" | tee -a "$d/ground.txt"
	cp "$d/server.log" "$d/server-attempt$try.log"; pid=
	sleep 30
done
(( up )) || { echo "server never came up"; exit 1; }
model=$(curl -sf "http://127.0.0.1:$port/v1/models" | jq -r '.data[0].id')
echo "model id $model" >> "$d/ground.txt"
echo "serving: $(grep -E 'memory budget|dense load' "$d/server.log" | tr '\n' ' ')"

ask() { # ask <out.json> <max_tokens> <user text>
	jq -n --arg u "$3" --argjson m "$2" --arg model "$model" \
		'{model:$model, messages:[{role:"user", content:$u}], temperature:0, max_tokens:$m}' |
		cat > "$1.req"
	# Other jobs hold the chip lock for minutes: the server answers 503 (GPU busy) after 300 s. A refused request
	# leaves no line in the ids log, so retry until it is answered, or the ids would shift against the prompts.
	for try in $(seq 1 20); do
		curl -s -w '\n%{time_total}\n' -H 'content-type: application/json' --data-binary @"$1.req" \
			"http://127.0.0.1:$port/v1/chat/completions" > "$1"
		head -n1 "$1" | jq -e '.choices[0]' > /dev/null 2>&1 && return 0
		echo "request $1 try $try refused: $(head -n1 "$1" | cut -c1-160)" >> "$d/ground.txt"; mv "$1" "$1.refused$try"; sleep 10
	done
	echo "request $1 never answered"; return 1
}
GATE=("What is the capital of France? Answer in one sentence."
	"Write a Rust function \`fn is_prime(n: u64) -> bool\` that uses trial division up to the square root, with a short doc comment."
	"Explain in plain words why a bigger graphics card memory lets a language model remember a longer conversation.")
for i in "${!GATE[@]}"; do ask "$d/gate-$i.json" "$NEW" "${GATE[$i]}"; done
SPEED="Write a 300-word story about a lighthouse keeper who finds a message in a bottle."
for r in $(seq 1 "$SPEED_RUNS"); do ask "$d/speed-$r.json" "$SPEED_TOKENS" "$SPEED"; done
curl -sf "http://127.0.0.1:$port/health" > "$d/health-end.json"
stop; trap - EXIT
echo "server stopped"

# Host reference on the exact prompt ids the server tokenised (first ${#GATE[@]} requests).
head -n "${#GATE[@]}" "$ids" | jq -s '[to_entries[] | {name: ("gate-" + (.key|tostring)), ids: .value.prompt_ids}]' > "$d/prompt-ids.json"
nice -n 10 "$CHECK" golden "$ckpt" "$d/reference.json" --prompt-ids "$d/prompt-ids.json" --new "$NEW" "${w4[@]}" \
	--max-context 4096 > "$d/reference.log" 2>&1
echo "reference rc=$?"

verdict=PASS
: > "$d/gate.txt"
for i in "${!GATE[@]}"; do
	got=$(sed -n "$((i + 1))p" "$ids" | jq -c '.completion_ids')
	# The reference keeps the stop id it chose; the server never emits it.
	want=$(jq -c --argjson i "$i" '.prompts[$i].golden | map(select(. != 151645 and . != 151643))' "$d/reference.json")
	ctl=$(jq -c --argjson i "$i" '.prompts[$i].golden | map(select(. != 151645 and . != 151643)) | (.[3] += 1)' "$d/reference.json")
	ok=MISMATCH; [[ $got == "$want" ]] && ok=MATCH
	cok=MISMATCH; [[ $got == "$ctl" ]] && cok=MATCH
	echo "gate-$i server ids vs host reference: $ok ($(jq 'length' <<<"$got") tokens); control (reference id 3 changed): $cok" >> "$d/gate.txt"
	[[ $(jq 'length' <<<"$got") -gt 0 ]] || verdict=FAIL # an empty reply is never a pass
	[[ $ok == MATCH && $cok == MISMATCH ]] || verdict=FAIL
done
# Speed: server-side decode wall and chip time per token, from the ids log (requests after the gate prompts).
tail -n "+$((${#GATE[@]} + 1))" "$ids" | jq -r '"speed: decode \(.emitted_tokens) tok in \(.decode_ms) ms = \((.emitted_tokens * 1000 / .decode_ms) * 100 | round / 100) tok/s; chip \((.decode_chip_ns / 1e6 / .emitted_tokens) * 100 | round / 100) ms/token; prefill \(.prompt_tokens) tok (\(.reused_prompt_tokens) reused) in \(.prefill_ms) ms; lock wait \(.lock_wait_ms) ms"' >> "$d/gate.txt"
for r in $(seq 1 "$SPEED_RUNS"); do echo "speed-$r client wall $(tail -1 "$d/speed-$r.json") s" >> "$d/gate.txt"; done
echo "VERDICT $verdict" >> "$d/gate.txt"
{ echo "end $(date -u +%FT%TZ)"; echo "loadavg $(cat /proc/loadavg)"; grep MemAvailable /proc/meminfo; } >> "$d/ground.txt"
( cd "$d" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
cat "$d/gate.txt"
[[ $verdict == PASS ]]
