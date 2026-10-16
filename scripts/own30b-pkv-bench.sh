#!/usr/bin/env bash
# Persistent KV benchmark on the chip (sc#508, contract locked in #500). CHIP-ONLY: never run by
# CI or cargo test. Same safety rules as own30b-pkv-chip-check.sh (see own30b-pkv-lib.sh): own
# server on 127.0.0.1:18098 (PKV_CHECK_PORT), refuses while :18091 answers or the port is taken,
# stops only the PID it started, key via `atlas-vault run AIEN_KV_STORE_KEY --` (never printed).
#
#   scripts/own30b-pkv-bench.sh [--dry-run] <evidence-dir> <aien-serve-http binary>
#
# Workload: 8 two-turn conversations, each with its own ~2048-token prefix (a distinct session
# line plus PKV_BENCH_RULES rules) that turn 2 shares with turn 1; greedy, PKV_BENCH_MAX_TOKENS
# (64) new tokens, streamed so TTFT is measured on the client.
#   seed      persistence on, store S: turn 1 of all 8 conversations, wait for the parks
#   disabled  persistence off (AIEN_KV_STORE_DIR unset)
#   warm      persistence on S, every store file read into the page cache before each batch
#   cold      persistence on S, `sync; echo 3 > /proc/sys/vm/drop_caches` (sudo -n) before each
#             batch; cold is VERIFIED per batch by the server's /proc/<pid>/io read_bytes (reads
#             that reached the block layer) against health restore_bytes; no fincore/vmtouch here
# Each mode: concurrency 1/2/4/8 x PKV_BENCH_REPS (3) batches. A batch first sends one short
# unrelated chat (so the engine's RAM prefix is useless), then turn 2 of conversations 1..N at
# once. The server runs one generation at a time and answers 503 when busy: clients retry every
# 0.2 s (counted), latency and TTFT run from the first attempt, so they include the queue.
# Raw evidence: every SSE stream with client timestamps, health and /proc/<pid>/io before and
# after each batch, RSS every second, GPU memory (nvidia-smi compute-apps) every 5 s, iostat -dxk
# 5, server logs. summary.txt: p50/p95 latency and TTFT, completion tokens/s, restores, restore
# bytes, block reads, decrypt (open) and encrypt (seal) time, RSS/GPU peak, failures.
# Budget: PKV_BENCH_BUDGET_S (2700) wall seconds; past it the remaining batches are skipped and
# the verdict is INCOMPLETE. Rough duration: 4 server starts plus about 180 requests, 15 to 30 min.
set -euo pipefail
. "$(dirname "$0")/own30b-pkv-lib.sh"
pkv_usage_args "$(basename "$0")" "$@"

REPS=${PKV_BENCH_REPS:-3}
MAXTOK=${PKV_BENCH_MAX_TOKENS:-64}
RULES=${PKV_BENCH_RULES:-118}
BUDGET=${PKV_BENCH_BUDGET_S:-2700}
LEVELS="1 2 4 8"
MODES="disabled warm cold"

bench_prereqs() {
	local bad=0 fs
	for t in iostat nvidia-smi; do
		if command -v "$t" >/dev/null; then echo "  ok   tool $t"; else echo "  info tool $t missing (that column stays empty)"; fi
	done
	if sudo -n true 2>/dev/null; then
		echo "  ok   sudo -n works (drop_caches for the cold mode)"
	else
		echo "  MISS sudo -n (the cold mode needs: sync; echo 3 > /proc/sys/vm/drop_caches)"
		bad=1
	fi
	fs=$(findmnt -no FSTYPE -T /tmp)
	if [ "$fs" = tmpfs ] || [ "$fs" = ramfs ]; then
		echo "  MISS /tmp is $fs: the store must be on the NVMe for a cold restore"
		bad=1
	else
		echo "  ok   store parent /tmp is $fs on $(findmnt -no SOURCE -T /tmp)"
	fi
	return $bad
}

if [ "$DRY" = 1 ]; then
	echo "PLAN own30b-pkv-bench (#508): seed, then modes [$MODES] x concurrency [$LEVELS] x $REPS reps"
	echo "     8 conversations, ~2048-token prefix each ($RULES rules), $MAXTOK new tokens, streamed"
	echo "     4 server starts on :$PORT, about $((3 * REPS * (4 + 15) + 8)) requests, estimate 15 to 30 minutes, budget ${BUDGET} s"
	echo "     cold = drop_caches before each batch, verified by server block reads vs restore_bytes"
	echo "     evidence -> $dir"
	echo "PREREQUISITES"
	ok=1
	pkv_prereqs 1 || ok=0
	bench_prereqs || ok=0
	if [ $ok = 1 ]; then echo "DRY RUN: ready"; exit 0; fi
	echo "DRY RUN: NOT ready"
	exit 1
fi
if ! bench_prereqs >/dev/null 2>&1; then
	bench_prereqs >&2 || true
	echo "refusing: a benchmark prerequisite is missing" >&2
	exit 2
fi
pkv_begin

SAMPLER=
IOSTAT=
stop_samplers() {
	if [ -n "$SAMPLER" ]; then kill "$SAMPLER" 2>/dev/null || true; fi
	SAMPLER=
}
trap 'stop_samplers; [ -n "$IOSTAT" ] && kill "$IOSTAT" 2>/dev/null; stop_server || PKV_LEAK=1; rm -rf "$work"; pkv_trap_status' EXIT

if command -v iostat >/dev/null; then
	iostat -dxk -t 5 >"$dir/iostat.txt" 2>&1 &
	IOSTAT=$!
fi

start_samplers() { # <mode>: RSS every 1 s, GPU memory every 5 s
	local mode=$1 p
	p=$(server_pid)
	(
		n=0
		while kill -0 "$p" 2>/dev/null; do
			printf '%s %s\n' "$EPOCHREALTIME" "$(awk '/^VmRSS/ {print $2}' "/proc/$p/status" 2>/dev/null)" >>"$dir/rss-$mode.txt"
			if [ $((n % 5)) = 0 ] && command -v nvidia-smi >/dev/null; then
				printf '%s %s\n' "$EPOCHREALTIME" "$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v p="$p" '$1 == p {print $2}')" >>"$dir/gpu-$mode.txt"
			fi
			n=$((n + 1))
			sleep 1
		done
	) &
	SAMPLER=$!
}

# The conversations: k = 1..8, each prefix its own (the session line makes every chunk distinct).
sys_k() {
	local s="Session $1 of the OWN-30B benchmark. You are the OWN-30B assistant on the AIEN Spark. Answer briefly." i
	for i in $(seq 1 "$RULES"); do s="$s Rule $i: keep answers short, never invent facts, say UNKNOWN when unsure."; done
	printf '%s' "$s"
}
msgs1() { jq -n --arg s "$(sys_k "$1")" --arg q "$Q1" '[{role:"system",content:$s},{role:"user",content:$q}]'; }
msgs2() { # <k> <turn-1 answer> <question>
	jq -n --argjson m "$(msgs1 "$1")" --arg a "$2" --arg q "$3" '$m + [{role:"assistant",content:$a},{role:"user",content:$q}]'
}
FLUSH=$(jq -n '[{role:"user",content:"Say one word about the weather."}]')

# send <name> <messages-json>: one streamed request, retried while the server answers 503.
# Writes <name>.sse (client time + SSE line) and one line of requests.tsv.
send() {
	local name=$1 body="$work/$1.body" h="$work/$1.h" raw="$dir/sse/$1.sse" t0 code retries=0
	jq -n --arg model "$SERVED_MODEL" --argjson m "$2" --argjson n "$MAXTOK" '{model:$model, messages:$m, max_tokens:$n, temperature:0, seed:1, stream:true}' >"$body"
	t0=$EPOCHREALTIME
	while :; do
		rm -f "$h"
		curl -sN -m 900 -D "$h" "$base/v1/chat/completions" -H 'content-type: application/json' -d @"$body" |
			while IFS= read -r line; do [ -n "$line" ] && printf '%s %s\n' "$EPOCHREALTIME" "$line"; done >"$raw" || true
		code=$(awk 'NR == 1 {print $2}' "$h" 2>/dev/null)
		if [ "$code" = 503 ] && [ "$retries" -lt 6000 ]; then
			retries=$((retries + 1))
			sleep 0.2
			continue
		fi
		break
	done
	local row sha
	row=$(awk -v t0="$t0" -v name="$name" -v code="${code:-none}" -v retries="$retries" '
		/ data: \{/ && first == "" && $0 ~ /"content":"[^"]/ { first = $1 }
		/"usage":/ { if (match($0, /"completion_tokens":[0-9]+/)) ct = substr($0, RSTART + 20, RLENGTH - 20)
		             if (match($0, /"prompt_tokens":[0-9]+/)) pt = substr($0, RSTART + 16, RLENGTH - 16) }
		/ data: \[DONE\]/ { done = 1 }
		{ last = $1 }
		END {
			ttft = (first == "") ? "NA" : sprintf("%.4f", first - t0)
			lat = (last == "") ? "NA" : sprintf("%.4f", last - t0)
			ok = (code == "200" && done && first != "" && ct != "") ? "ok" : "FAIL"
			printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s", name, ok, code, retries, ttft, lat, ct + 0, pt + 0
		}' "$raw")
	sha=$(sed -n 's/^[0-9.]* data: //p' "$raw" | grep -v '^\[DONE\]' | jq -rj '.choices[0].delta.content // empty' 2>/dev/null | sha256sum | cut -c1-16)
	# One write per row: the N clients of a batch append at the same time.
	printf '%s\t%s\n' "$row" "$sha" >>"$dir/requests.tsv"
}

# snap <tag>: health and /proc/<pid>/io of our server, and the NVMe line of /proc/diskstats.
snap() {
	local p
	p=$(server_pid)
	curl -s -m 10 "$base/health" >"$dir/snap/$1-health.json"
	cat "/proc/$p/io" >"$dir/snap/$1-io.txt"
	grep -w "$(basename "$(findmnt -no SOURCE -T /tmp)" | sed 's/p[0-9]*$//')" /proc/diskstats >"$dir/snap/$1-diskstats.txt" || true
	awk '/^(Cached|MemAvailable):/' /proc/meminfo >"$dir/snap/$1-meminfo.txt"
}
hv() { jq -r "(.persistent_kv$2) // 0" "$dir/snap/$1-health.json" 2>/dev/null || echo 0; }
iov() { awk -v k="$2:" '$1 == k {v = $2} END {print v + 0}' "$dir/snap/$1-io.txt" 2>/dev/null || echo 0; }

# batch <mode> <N> <rep>
batch() {
	local mode=$1 n=$2 r=$3 tag="$1-c$2-r$3" k t0 t1 pids=()
	send "$tag-flush" "$FLUSH"
	case $mode in
	warm) find "$S" -type f -exec cat {} + >/dev/null 2>&1 || true ;;
	cold)
		sync
		sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'
		;;
	esac
	snap "$tag-pre"
	t0=$EPOCHREALTIME
	for k in $(seq 1 "$n"); do
		send "$tag-k$k" "$(msgs2 "$k" "${A1[$k]}" "Benchmark question c$n r$r: which of those matters most on a bicycle, and why?")" &
		pids+=("$!")
	done
	wait "${pids[@]}"
	t1=$EPOCHREALTIME
	snap "$tag-post"
	printf '%s\t%s\t%s\t%.4f\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$n" "$r" "$(awk -v a="$t0" -v b="$t1" 'BEGIN {print b - a}')" \
		$(($(hv "$tag-post" .restores) - $(hv "$tag-pre" .restores))) \
		$(($(hv "$tag-post" .restore_bytes) - $(hv "$tag-pre" .restore_bytes))) \
		$(($(iov "$tag-post" read_bytes) - $(iov "$tag-pre" read_bytes))) \
		$(($(iov "$tag-post" write_bytes) - $(iov "$tag-pre" write_bytes))) \
		$(($(hv "$tag-post" .seal.open_ns) - $(hv "$tag-pre" .seal.open_ns))) \
		$(($(hv "$tag-post" .seal.open_bytes) - $(hv "$tag-pre" .seal.open_bytes))) \
		$(($(hv "$tag-post" .seal.seal_ns) - $(hv "$tag-pre" .seal.seal_ns))) \
		$(($(hv "$tag-post" .seal.seal_bytes) - $(hv "$tag-pre" .seal.seal_bytes))) \
		"$(jq -c '.persistent_kv.restore_misses // {}' "$dir/snap/$tag-post-health.json")" >>"$dir/batches.tsv"
}

mkdir -p "$dir/sse" "$dir/snap"
printf 'name\tok\thttp\tretries_503\tttft_s\tlatency_s\tcompletion_tokens\tprompt_tokens\tcontent_sha\n' >"$dir/requests.tsv"
printf 'mode\tconc\trep\twall_s\trestores\trestore_bytes\tproc_read_bytes\tproc_write_bytes\topen_ns\topen_bytes\tseal_ns\tseal_bytes\trestore_misses\n' >"$dir/batches.tsv"
{
	echo "reps $REPS max_tokens $MAXTOK rules $RULES budget_s $BUDGET levels $LEVELS"
	echo "store parent $(findmnt -no SOURCE,FSTYPE -T /tmp)"
} >>"$dir/run-info.txt"
S="$work/store"
declare -a A1

echo "== seed: turn 1 of 8 conversations on store S"
start_server seed "$S"
start_samplers seed
for k in $(seq 1 8); do
	send "seed-k$k" "$(msgs1 "$k")"
	A1[k]=$(sed -n 's/^[0-9.]* data: //p' "$dir/sse/seed-k$k.sse" | grep -v '^\[DONE\]' | jq -rj '.choices[0].delta.content // empty')
	printf '%s' "${A1[$k]}" >"$dir/seed-k$k.answer.txt"
done
wait_parks
health seed >/dev/null
stop_samplers
stop_server
prefix=$(awk -F'\t' '$1 == "seed-k1" {print $8}' "$dir/requests.tsv")
echo "turn-1 prompt tokens: $prefix (target ~2048)"
if [ "${prefix:-0}" -lt 1843 ] || [ "${prefix:-0}" -gt 2253 ]; then
	echo "the prefix is $prefix tokens, not ~2048: set PKV_BENCH_RULES (now $RULES) and rerun" | tee "$dir/summary.txt" >&2
	pkv_seal
	exit 1
fi

skipped=0
for mode in $MODES; do
	echo "== $mode"
	if [ "$mode" = disabled ]; then start_server "$mode" -; else start_server "$mode" "$S"; fi
	snap "$mode-start"
	start_samplers "$mode"
	for n in $LEVELS; do
		for r in $(seq 1 "$REPS"); do
			if [ "$SECONDS" -gt "$BUDGET" ]; then
				echo "budget ${BUDGET} s spent: skipping $mode c$n r$r"
				skipped=$((skipped + 1))
				continue
			fi
			batch "$mode" "$n" "$r"
		done
	done
	health "$mode" >/dev/null
	stop_samplers
	stop_server
done

pct() { # <p> on stdin numbers -> nearest rank
	sort -g | awk -v p="$1" '{v[NR] = $1} END {if (NR == 0) {print "NA"; exit} i = int(p * NR + 0.9999); if (i < 1) i = 1; print v[i]}'
}
col() { # <mode> <conc> <column>: values of requests in that cell
	awk -F'\t' -v m="$1" -v c="$2" -v col="$3" '$1 ~ "^" m "-c" c "-r[0-9]+-k[0-9]+$" && $5 != "NA" {print $col}' "$dir/requests.tsv"
}
peak() { awk '$2 != "" && $2 + 0 > m {m = $2 + 0} END {print m + 0}' "$1" 2>/dev/null || echo 0; }

{
	echo "PKV BENCH (#508) $(date -u +%FT%TZ)  binary $(sha256sum "$BIN" | cut -c1-16)  prefix $prefix tokens  max_tokens $MAXTOK  reps $REPS"
	echo "latency and TTFT in seconds from the first attempt (queue included); tok/s = completion tokens / batch wall time"
	printf '%-8s %4s %3s %4s %8s %8s %8s %8s %7s %6s %7s %10s %10s %9s %9s\n' mode conc n fail lat_p50 lat_p95 ttft_p50 ttft_p95 tok_s retry restores restore_MB block_rd_MB open_ms seal_ms
	for mode in $MODES; do
		for n in $LEVELS; do
			cnt=$(col "$mode" "$n" 1 | wc -l)
			fails=$(awk -F'\t' -v m="$mode" -v c="$n" '$1 ~ "^" m "-c" c "-r[0-9]+-k[0-9]+$" && $2 != "ok"' "$dir/requests.tsv" | wc -l)
			read -r wall toks retr <<<"$(awk -F'\t' -v m="$mode" -v c="$n" '
				FILENAME ~ /batches/ && $1 == m && $2 == c {w += $4}
				FILENAME ~ /requests/ && $1 ~ "^" m "-c" c "-r[0-9]+-k[0-9]+$" {t += $7; r += $4}
				END {print w + 0, t + 0, r + 0}' "$dir/batches.tsv" "$dir/requests.tsv")"
			read -r rs rb rd on sn <<<"$(awk -F'\t' -v m="$mode" -v c="$n" '$1 == m && $2 == c {a += $5; b += $6; d += $7; o += $9; s += $11} END {print a + 0, b + 0, d + 0, o + 0, s + 0}' "$dir/batches.tsv")"
			printf '%-8s %4s %3s %4s %8s %8s %8s %8s %7s %6s %7s %10.1f %10.1f %9.1f %9.1f\n' "$mode" "$n" "$cnt" "$fails" \
				"$(col "$mode" "$n" 6 | pct 0.50)" "$(col "$mode" "$n" 6 | pct 0.95)" \
				"$(col "$mode" "$n" 5 | pct 0.50)" "$(col "$mode" "$n" 5 | pct 0.95)" \
				"$(awk -v t="$toks" -v w="$wall" 'BEGIN {printf "%.1f", (w > 0) ? t / w : 0}')" "$retr" "$rs" \
				"$(awk -v b="$rb" 'BEGIN {print b / 1048576}')" "$(awk -v b="$rd" 'BEGIN {print b / 1048576}')" \
				"$(awk -v b="$on" 'BEGIN {print b / 1e6}')" "$(awk -v b="$sn" 'BEGIN {print b / 1e6}')"
		done
	done
	for mode in seed $MODES; do
		echo "memory $mode: RSS peak $(($(peak "$dir/rss-$mode.txt") / 1024)) MiB, GPU (nvidia-smi compute-apps) peak $(peak "$dir/gpu-$mode.txt") MiB"
	done
	for mode in $MODES; do
		echo "server prefill $mode (long prompts): p50 $(grep -ho 'prompt [0-9]* tok, [0-9]* reused from the previous turn, prefill [0-9]* ms' "$dir/$mode.log" | awk '$2 > 1000 {print $(NF - 1)}' | pct 0.50) ms"
	done
	echo "seed health: $(jq -c '.persistent_kv | {parked_chunks, park_bytes, park_failed, seal}' "$dir/seed-health.json")"
	echo "CHECKS"
	check "every request answered (http 200, content, usage)" "$(awk -F'\t' 'NR > 1 && $2 != "ok"' "$dir/requests.tsv" | wc -l)" 0
	check "no batch skipped by the budget" "$skipped" 0
	check "disabled: no persistent_kv in health" "$(jq -r 'has("persistent_kv")' "$dir/disabled-health.json")" false
	for mode in warm cold; do
		check "$mode: every request restored" "$(awk -F'\t' -v m="$mode" '$1 == m && $5 != $2' "$dir/batches.tsv" | wc -l)" 0
		check "$mode: no park failed" "$(jq -r '.persistent_kv.park_failed' "$dir/$mode-health.json")" 0
	done
	check "warm: restores read from the page cache (block reads < 10% of restore bytes)" \
		"$(awk -F'\t' '$1 == "warm" && $7 * 10 >= $6 && $6 > 0' "$dir/batches.tsv" | wc -l)" 0
	check "cold: VERIFIED cold, block reads >= 90% of restore bytes in every batch" \
		"$(awk -F'\t' '$1 == "cold" && !($6 > 0 && $7 * 10 >= $6 * 9)' "$dir/batches.tsv" | wc -l)" 0
	check "answers identical across disabled/warm/cold" "$(awk -F'\t' '$1 ~ /-k[0-9]+$/ && $1 !~ /^seed/ {
			split($1, p, "-"); key = p[2] "-" p[3] "-" p[4]; if (key in s && s[key] != $9) bad++; s[key] = $9 } END {print bad + 0}' "$dir/requests.tsv")" 0
	[ $fail = 0 ] && echo "PKV BENCH: COMPLETE" || echo "PKV BENCH: INCOMPLETE (see FAIL lines)"
} | tee "$dir/summary.txt"
grep -q "PKV BENCH: COMPLETE" "$dir/summary.txt" || fail=1
stop_samplers
if [ -n "$IOSTAT" ]; then kill "$IOSTAT" 2>/dev/null || true; fi
IOSTAT=
pkv_seal
exit $fail
