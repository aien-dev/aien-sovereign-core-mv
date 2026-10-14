#!/usr/bin/env bash
# OWN-30B release candidate 2 (W-RC2): the full-model gate. RUN ONLY IN A PAUSE WINDOW: the live :18091 server must be
# stopped first (this script refuses while anything listens there). It never starts or stops :18091.
#
#   ~/handoffs/rc2-gate.sh            all three steps, in order
#   STEPS="1 3" ~/handoffs/rc2-gate.sh   a subset
#
# Binaries: the prebuilt, sealed RC2 set in ~/handoffs/rc2-bin-a08d2c9 (sc own30b/rc2 a08d2c9, omega own30b/rc2 848a999b,
# archive 1baabd41...); their SHA256SUMS are checked first, nothing is rebuilt in the window.
# Evidence: sc-own30b-rc2/docs/inference/evidence/own30b-rc2-<UTC stamp>/ (step1/ step2/ step3/, SUMMARY.txt, SHA256SUMS).
# One 30 GB load at a time. Every own30b_* run holds /tmp/aien-gb10.lock (flock) for its whole life. The step-3 server
# takes the same lock itself around its load and each generation (an outer flock would deadlock it); the script waits
# for the lock to be free before starting it.
#
# Step 1  own30b_check golden tokens + decode tok/s, prompts p1..p4, 16 new tokens, server load flags
#         (--weights fp8 --attn fused --router device --lm-head gemv --prefill auto), --tokens-only:
#           graph-off   whole-prompt + device router/combine, launched decode     (= RC 1 server path, the baseline)
#           rc2         RC2 defaults: + one-list decode + resident chain (grid 128, row pairs)
#           res-off     RC2 with the resident chain off
#           wp-off      RC2 with the whole-prompt prefill off (chunked prefill)
#           devhost-off RC2 with the device router/combine off
#           a16-on      RC2 with the W8A16 prefill on (the switch that is OFF by default)
#           rc2-b       RC2 defaults again (second speed sample)
#         Rules: T1 rc2, rc2-b, res-off, graph-off, devhost-off produce exactly the RC 1 server-path tokens (sealed
#         own30b-rc-20261014T0153Z/check-wp: p2-p4 = golden, p1 = the known W8A8 rounding flip); T2 wp-off = golden on
#         p1..p4; T3 every graph run ran the one-list decode, and resident runs use fewer launches per graph token
#         than res-off; SPEED GO iff rc2 and rc2-b beat graph-off on all four prompts. a16-on tokens are reported,
#         judged in step 2.
# Step 2  W-L1H's W8A16 full gate, the same six runs and arguments as sc scripts/own30b-l1h-gate.sh (check, wp_gate drift
#         at 1/512/2048 with mutants, TTFT 512/2048) for a8 then a16, with the one-list decode env unset (prefill only).
#         Rules (W-L1H, fixed before any number): R1 check-a16 p1 IDENTICAL and no prompt FAILs the tool's margin rule;
#         R2 at every length DRIFT a16 < DRIFT a8 and top1 AGREE; R3 TTFT a16 <= 1.10 x a8 at 512 and 2048 (on(dev),
#         median of 2 reps); R4 every gate-a16 negative control CAUGHT. W8A16 stays OFF unless all four hold.
# Step 3  Server smoke on 127.0.0.1:18093 with the RC2 defaults (no switch flags): /v1/models, Odysseus-style non-stream
#         turn, SSE follow-up with prefix reuse, a ~1800-token prompt (the RC 1 request), and a sampled (temperature 0.7)
#         turn that reads the logits back from the one-list decode. Stopped by PID.
set -uo pipefail
STEPS=${STEPS:-1 2 3}
SC=$HOME/workspace/sc-own30b-rc2
BIN=$HOME/handoffs/rc2-bin-a08d2c9
CKPT=${OWN30B_CHECKPOINT:-$HOME/models/qwen3-30b-a3b-instruct-2507-fp8}
GOLDEN=$SC/docs/inference/evidence/own30b-f-20261013T1745Z/golden.json
RC1=$SC/docs/inference/evidence/own30b-rc-20261014T0153Z
EV=$SC/docs/inference/evidence/own30b-rc2-$(date -u +%Y%m%dT%H%MZ)
LOCK=/tmp/aien-gb10.lock
SUM=$EV/SUMMARY.txt

die() { echo "RC2_GATE: REFUSED $*"; exit 2; }
listening() { ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN; }
avail_gb() { free -g | awk '/^Mem:/ {print $7}'; }

# ---- preflight ----
listening 18091 && die "something listens on :18091 (the live server): stop it first (pause window)"
listening 18093 && die "something already listens on :18093"
[ "$(avail_gb)" -ge 70 ] || die "only $(avail_gb) GB available (needs 70: one 30 GB model at a time)"
( cd "$BIN" && sha256sum -c --quiet SHA256SUMS ) || die "binary digests in $BIN do not match SHA256SUMS"
for f in "$GOLDEN" "$RC1/check-wp/run.json" "$RC1/server-smoke/req3.json" "$SC/scripts/own30b-chip-run.sh"; do
	[ -f "$f" ] || die "missing $f"
done
mkdir -p "$EV"
{
	echo "OWN-30B RC2 gate $(date -u +%FT%TZ)"
	echo "sc $(git -C "$SC" rev-parse HEAD) omega $(git -C "$HOME/workspace/omega-own30b-rc2" rev-parse HEAD) omega.lock $(cat "$SC/omega.lock")"
	echo "binaries $BIN:"; cat "$BIN/SHA256SUMS"
	echo "checkpoint $CKPT"; echo "golden sha256 $(sha256sum "$GOLDEN" | cut -d' ' -f1)"
	free -g | head -2
} > "$EV/MANIFEST"
echo "evidence $EV"
: > "$SUM"
note() { echo "$*" | tee -a "$SUM"; }
clean_env() {
	unset OWN30B_DECODE_GRAPH OWN30B_DECODE_GRAPH_MUTANT OWN30B_DECODE_GRAPH_STAMPS AIEN_OMEGA_DGRAPH_RESIDENT \
		AIEN_OMEGA_DGRAPH_RESIDENT_GRID AIEN_OMEGA_DGRAPH_RESIDENT_ROWS AIEN_OMEGA_DGRAPH_PREFETCH AIEN_OMEGA_DGRAPH_BARRIER \
		AIEN_SERVE_WHOLE_PROMPT AIEN_SERVE_WP_DEVICE_HOST AIEN_SERVE_PREFILL_ACTS AIEN_SERVE_DECODE_GRAPH AIEN_SERVE_RESIDENT_DECODE
	export AIEN_OMEGA_SPIN_US=2000
}
# per-prompt records of an own30b_check run.json: [{name, chip, first_diff, tok_s, on, gt, gl}]
recs() { jq -c '[.. | objects | select(has("decode_graph")) | {name, chip, first_diff, tok_s, on: .decode_graph.on, gt: .decode_graph.graph_tokens, gl: .decode_graph.graph_launches}]' "$1" 2>/dev/null; }
toks() { recs "$1" | jq -c 'map({name, chip})'; }

# ---- step 1 ----
step1() {
	local d=$EV/step1 r1="--prompts 1,2,3,4 --new 16 --layers 0,23,47 --weights fp8 --attn fused --router device --lm-head gemv --prefill auto --tokens-only"
	mkdir -p "$d"
	note "== STEP 1: golden tokens + decode tok/s, RC2 defaults and each switch flipped"
	# run <name> <graph 0|1> <resident 0|1> <own30b_check extra args...>
	run() {
		local n=$1 g=$2 r=$3; shift 3
		clean_env
		if [ "$g" = 1 ]; then export OWN30B_DECODE_GRAPH=1; fi
		if [ "$r" = 1 ]; then export AIEN_OMEGA_DGRAPH_RESIDENT=1 AIEN_OMEGA_DGRAPH_RESIDENT_GRID=128 AIEN_OMEGA_DGRAPH_RESIDENT_ROWS=2; fi
		echo "$n: OWN30B_DECODE_GRAPH=${OWN30B_DECODE_GRAPH:-} AIEN_OMEGA_DGRAPH_RESIDENT=${AIEN_OMEGA_DGRAPH_RESIDENT:-} GRID=${AIEN_OMEGA_DGRAPH_RESIDENT_GRID:-} ROWS=${AIEN_OMEGA_DGRAPH_RESIDENT_ROWS:-} args $r1 $*" >> "$d/runs.txt"
		OWN30B_WHISPER="W-RC2 gate step 1: $n (30 GB load), ~2 min" "$SC/scripts/own30b-chip-run.sh" "$d" "$n" "$BIN/own30b_check" "$GOLDEN" $r1 "$@" > "$d/$n.chiprun" 2>&1
		tail -1 "$d/$n.chiprun" >/dev/null
	}
	run graph-off 0 0 --whole-prompt --prefill-device-host
	run rc2 1 1 --whole-prompt --prefill-device-host
	run res-off 1 0 --whole-prompt --prefill-device-host
	run wp-off 1 1
	run devhost-off 1 1 --whole-prompt
	run a16-on 1 1 --whole-prompt --prefill-device-host --prefill-acts a16
	run rc2-b 1 1 --whole-prompt --prefill-device-host
	clean_env

	local ref ok=PASS speed=GO
	ref=$(toks "$RC1/check-wp/run.json")
	for n in graph-off rc2 res-off devhost-off rc2-b; do
		local t; t=$(toks "$d/$n/run.json")
		if [ -n "$t" ] && [ "$t" = "$ref" ]; then note "T1 $n tokens = RC 1 server path (all 4 prompts): PASS"
		else note "T1 $n tokens differ from the RC 1 server path (or no run.json): FAIL"; ok=FAIL; fi
	done
	if [ "$(recs "$d/wp-off/run.json" | jq '[.[] | select(.first_diff == null)] | length')" = 4 ]; then note "T2 wp-off golden IDENTICAL p1..p4: PASS"
	else note "T2 wp-off not golden on every prompt: FAIL"; ok=FAIL; fi
	for n in rc2 res-off wp-off devhost-off a16-on rc2-b; do
		if [ "$(recs "$d/$n/run.json" | jq '[.[] | select(.on == true and .gt > 0)] | length')" != 4 ]; then
			note "T3 $n: the one-list decode did not run on every prompt: FAIL"; ok=FAIL; fi
	done
	if [ "$(recs "$d/graph-off/run.json" | jq '[.[] | select(.gt > 0)] | length')" != 0 ]; then note "T3 graph-off ran graph tokens: FAIL"; ok=FAIL; fi
	local lres loff
	lres=$(recs "$d/rc2/run.json" | jq '[.[] | .gl / .gt] | max')
	loff=$(recs "$d/res-off/run.json" | jq '[.[] | .gl / .gt] | min')
	if awk -v a="$lres" -v b="$loff" 'BEGIN { exit !(a + 0 > 0 && a < b) }'; then note "T3 launches per graph token: rc2 $lres < res-off $loff (resident ran): PASS"
	else note "T3 launches per graph token: rc2 $lres vs res-off $loff (resident did not run?): FAIL"; ok=FAIL; fi
	note "a16-on tokens vs golden (info; judged in step 2): $(recs "$d/a16-on/run.json" | jq -c 'map({(.name): (.first_diff // "IDENTICAL")}) | add')"
	{
		echo "decode tok/s     p1     p2     p3     p4"
		for n in graph-off rc2 rc2-b res-off wp-off devhost-off a16-on; do
			printf '%-12s' "$n"; recs "$d/$n/run.json" | jq -r 'map(.tok_s * 10 | round / 10 | tostring) | join("  ")'
		done
	} | tee -a "$SUM"
	for i in 0 1 2 3; do
		local b x y
		b=$(recs "$d/graph-off/run.json" | jq ".[$i].tok_s"); x=$(recs "$d/rc2/run.json" | jq ".[$i].tok_s"); y=$(recs "$d/rc2-b/run.json" | jq ".[$i].tok_s")
		awk -v b="$b" -v x="$x" -v y="$y" 'BEGIN { exit !(x > b && y > b) }' || speed=NO-GO
	done
	note "STEP1: tokens=$ok speed=$speed (rc2 and rc2-b vs graph-off on all four prompts)"
	S1="tokens=$ok speed=$speed"
}

# ---- step 2 (W-L1H's gate, prebuilt binaries) ----
step2() {
	local d=$EV/step2
	mkdir -p "$d"
	note "== STEP 2: W-L1H W8A16 gate (W8A8 vs W8A16)"
	clean_env
	run2() {
		local name=$1 bin=$2; shift 2
		{ echo "start $(date -u +%FT%TZ)"; echo "cmd flock $LOCK env AIEN_OMEGA_SPIN_US=2000 $bin $*"; free -g | head -2; } > "$d/$name.start"
		crumb whisper odysseus-9e "$SC" - "W-RC2 gate step 2: chip run $name (full 30B load)" >/dev/null 2>&1 || true
		flock "$LOCK" env AIEN_OMEGA_SPIN_US=2000 "$bin" "$@" > "$d/$name.raw" 2> "$d/$name.err"
		echo $? > "$d/$name.rc"
		{ echo "end $(date -u +%FT%TZ)"; free -g | head -2; } > "$d/$name.end"
		echo "$name rc=$(cat "$d/$name.rc")"
	}
	for acts in a8 a16; do
		run2 "check-$acts" "$BIN/own30b_check" chip "$CKPT" "$GOLDEN" "$d/check-$acts" \
			--whole-prompt --prefill-device-host --prefill-acts "$acts" --tokens-only --new 16 --prompts 1,2,3,4
		run2 "gate-$acts" "$BIN/own30b_wp_gate" "$CKPT" "$d/gate-$acts" --lens 1,512,2048 --devhost --mutants --acts "$acts"
		run2 "ttft-$acts" "$BIN/own30b_ttft" "$CKPT" "$d/ttft-$acts" --lens 512,2048 --reps 2 --new-tokens 32 --devhost --acts "$acts"
	done
	local r1=PASS r2=PASS r3=PASS r4=PASS
	grep -h "^TOKENS" "$d/check-a8.raw" "$d/check-a16.raw" | sed -E 's/^TOKENS ([^:]+):.*; (IDENTICAL|DIVERGES.*)$/\1 \2/' | sed 's/^/  /' | tee -a "$SUM"
	grep -qE "^TOKENS p1[^;]*;.*; IDENTICAL$|^TOKENS p1.*IDENTICAL$" "$d/check-a16.raw" || r1=FAIL
	[ "$(grep -c '^TOKENS p' "$d/check-a16.raw")" = 4 ] || r1=FAIL
	grep -E "^TOKENS p" "$d/check-a16.raw" | grep -q "FAIL)" && r1=FAIL
	grep -h "DRIFT" "$d/gate-a8.raw" "$d/gate-a16.raw" | sed 's/^/  /' | tee -a "$SUM"
	for len in 1 512 2048; do
		local a b
		a=$(grep -E "^len $len DRIFT acts a8 " "$d/gate-a8.raw" | head -1 | sed -E 's/.*= ([0-9.eE+-]+) top1.*/\1/')
		b=$(grep -E "^len $len DRIFT acts a16 " "$d/gate-a16.raw" | head -1 | sed -E 's/.*= ([0-9.eE+-]+) top1.*/\1/')
		grep -E "^len $len DRIFT acts a16 " "$d/gate-a16.raw" | head -1 | grep -q "AGREE" || r2=FAIL
		awk -v a="$a" -v b="$b" 'BEGIN { exit !(a != "" && b != "" && b + 0 < a + 0) }' || r2=FAIL
	done
	for len in 512 2048; do
		local ta tb
		ta=$(grep -E "^len $len rep [0-9]+ DEVHOST" "$d/ttft-a8.raw" | sed -E 's/.*on\(dev\) ([0-9.]+) ms.*/\1/' | sort -n | awk '{v[NR]=$1} END {if (NR) print (NR%2 ? v[(NR+1)/2] : (v[NR/2]+v[NR/2+1])/2)}')
		tb=$(grep -E "^len $len rep [0-9]+ DEVHOST" "$d/ttft-a16.raw" | sed -E 's/.*on\(dev\) ([0-9.]+) ms.*/\1/' | sort -n | awk '{v[NR]=$1} END {if (NR) print (NR%2 ? v[(NR+1)/2] : (v[NR/2]+v[NR/2+1])/2)}')
		note "  TTFT on(dev) median len $len: a8 ${ta:-?} ms a16 ${tb:-?} ms"
		awk -v a="$ta" -v b="$tb" 'BEGIN { exit !(a != "" && b != "" && b + 0 <= 1.10 * a) }' || r3=FAIL
	done
	local nc cc
	nc=$(grep -c "NEGATIVE CONTROL" "$d/gate-a16.raw"); cc=$(grep "NEGATIVE CONTROL" "$d/gate-a16.raw" | grep -c "CAUGHT (rule fails it)")
	note "  gate-a16 negative controls caught $cc of $nc"
	[ "$nc" -gt 0 ] && [ "$nc" = "$cc" ] || r4=FAIL
	grep -h "SUMMARY" "$d"/*.raw | sort | uniq -c | sed 's/^/  /' >> "$SUM"
	note "STEP2: R1 golden=$r1 R2 drift=$r2 R3 ttft=$r3 R4 controls=$r4 (W8A16 may be turned on only if all PASS)"
	S2="R1=$r1 R2=$r2 R3=$r3 R4=$r4"
}

# ---- step 3 (server smoke on :18093) ----
step3() {
	local d=$EV/step3 url=http://127.0.0.1:18093 pid ok=PASS
	mkdir -p "$d"
	note "== STEP 3: server smoke on 127.0.0.1:18093 (RC2 defaults)"
	clean_env
	listening 18093 && { note "STEP3: :18093 busy: FAIL"; S3=FAIL; return; }
	[ "$(avail_gb)" -ge 70 ] || { note "STEP3: only $(avail_gb) GB available: FAIL"; S3=FAIL; return; }
	flock "$LOCK" true # wait until no chip job holds the lock; the server takes it itself
	crumb whisper odysseus-9e "$SC" - "W-RC2 gate step 3: server smoke on 127.0.0.1:18093 (30 GB load), ~4 min" >/dev/null 2>&1 || true
	echo "cmd $BIN/aien-serve-http --engine omega --model-dir $CKPT --port 18093 (started $(date -u +%FT%TZ))" > "$d/server.cmd"
	"$BIN/aien-serve-http" --engine omega --model-dir "$CKPT" --port 18093 > "$d/server.log" 2>&1 < /dev/null &
	pid=$!
	echo "pid $pid exe $(readlink /proc/$pid/exe)" >> "$d/server.cmd"
	local t=0
	until grep -q "serving" "$d/server.log"; do
		kill -0 "$pid" 2>/dev/null || { note "STEP3: server exited during load: FAIL (see server.log)"; S3=FAIL; return; }
		sleep 2; t=$((t + 2)); [ $t -lt 900 ] || { note "STEP3: not serving after 900 s: FAIL"; kill -TERM "$pid"; S3=FAIL; return; }
	done
	grep -q "whole_prompt=on prefill_device_host=on prefill_acts=a8" "$d/server.log" || { note "  ready log: prefill switches not at RC2 defaults: FAIL"; ok=FAIL; }
	grep -q "decode_graph=on resident_decode=on (grid 128 rows 2)" "$d/server.log" || { note "  ready log: decode switches not at RC2 defaults: FAIL"; ok=FAIL; }
	grep -E "whole_prompt=|decode_graph=|engine ready" "$d/server.log" | sed 's/^/  /' | cut -c1-260 >> "$SUM"
	curl -s -m 30 "$url/v1/models" > "$d/models.json"
	local sys='You are Odysseus, a helpful personal assistant. Answer briefly.' q1='What is the capital of France? Answer in one short sentence.'
	jq -n --arg s "$sys" --arg q "$q1" '{model:"qwen3-30b-a3b-instruct-2507",messages:[{role:"system",content:$s},{role:"user",content:$q}],stream:false,temperature:0,max_tokens:48,session_id:"wrc2-smoke",cache_prompt:true,think:false}' > "$d/req1.json"
	/usr/bin/time -f "wall %e s" -o "$d/resp1.time" curl -s -m 300 -H 'Content-Type: application/json' -d @"$d/req1.json" "$url/v1/chat/completions" > "$d/resp1.json"
	local a1; a1=$(jq -r '.choices[0].message.content // empty' "$d/resp1.json")
	note "  turn 1 (non-stream): \"$a1\" ($(cat "$d/resp1.time"))"
	echo "$a1" | grep -q "Paris" || { note "  turn 1 lacks Paris: FAIL"; ok=FAIL; }
	jq -n --arg s "$sys" --arg q "$q1" --arg a "$a1" '{model:"qwen3-30b-a3b-instruct-2507",messages:[{role:"system",content:$s},{role:"user",content:$q},{role:"assistant",content:$a},{role:"user",content:"And of Italy?"}],stream:true,temperature:0,max_tokens:48,session_id:"wrc2-smoke",cache_prompt:true,think:false}' > "$d/req2.json"
	/usr/bin/time -f "wall %e s" -o "$d/resp2.time" curl -s -N -m 300 -H 'Content-Type: application/json' -d @"$d/req2.json" "$url/v1/chat/completions" > "$d/resp2.sse"
	local a2; a2=$(grep '^data: {' "$d/resp2.sse" | sed 's/^data: //' | jq -rj '.choices[0].delta.content // empty')
	note "  turn 2 (SSE, prefix reuse): \"$a2\" ($(cat "$d/resp2.time"))"
	echo "$a2" | grep -q "Rome" || { note "  turn 2 lacks Rome: FAIL"; ok=FAIL; }
	grep -q '^data: \[DONE\]' "$d/resp2.sse" || { note "  turn 2 has no [DONE]: FAIL"; ok=FAIL; }
	local reuse; reuse=$(grep -E "omega: lock wait" "$d/server.log" | sed -n 2p | sed -E 's/.*prompt ([0-9]+) tok, ([0-9]+) reused.*/\2 of \1/')
	note "  turn 2 reused prompt tokens: ${reuse:-?}"
	case "$reuse" in "" | 0\ *) note "  no prefix reuse on turn 2: FAIL"; ok=FAIL ;; esac
	cp "$RC1/server-smoke/req3.json" "$d/req3.json"
	/usr/bin/time -f "wall %e s" -o "$d/resp3.time" curl -s -m 600 -H 'Content-Type: application/json' -d @"$d/req3.json" "$url/v1/chat/completions" > "$d/resp3.json"
	local a3; a3=$(jq -r '.choices[0].message.content // empty' "$d/resp3.json")
	note "  long prompt ($(jq -r '.usage.prompt_tokens // "?"' "$d/resp3.json") tokens): \"$a3\" ($(cat "$d/resp3.time")); RC 1 answered \"$(jq -r '.choices[0].message.content' "$RC1/server-smoke/resp3.json")\""
	[ -n "$a3" ] && [ "$a3" = "$(jq -r '.choices[0].message.content' "$RC1/server-smoke/resp3.json")" ] || { note "  long prompt answer differs from RC 1: FAIL"; ok=FAIL; }
	jq -n --arg s "$sys" '{model:"qwen3-30b-a3b-instruct-2507",messages:[{role:"system",content:$s},{role:"user",content:"Write two sentences about the sea."}],stream:false,temperature:0.7,top_p:0.8,seed:7,max_tokens:64,session_id:"wrc2-smoke-2",think:false}' > "$d/req4.json"
	/usr/bin/time -f "wall %e s" -o "$d/resp4.time" curl -s -m 300 -H 'Content-Type: application/json' -d @"$d/req4.json" "$url/v1/chat/completions" > "$d/resp4.json"
	local a4; a4=$(jq -r '.choices[0].message.content // empty' "$d/resp4.json")
	note "  sampled turn (temperature 0.7): \"$a4\" finish $(jq -r '.choices[0].finish_reason // "?"' "$d/resp4.json") ($(cat "$d/resp4.time"))"
	[ "${#a4}" -ge 20 ] || { note "  sampled turn empty or too short: FAIL"; ok=FAIL; }
	grep -E "omega: lock wait" "$d/server.log" | sed -E 's/^aien-serve-http: omega: //' | awk '{print} match($0, /decode ([0-9]+) tok in ([0-9]+) ms/, m) { if (m[2] > 0) printf "    = %.1f tok/s\n", m[1] * 1000 / m[2] }' | sed 's/^/  /' | tee -a "$SUM"
	kill -TERM "$pid"
	for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
	if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid"; note "  server needed SIGKILL"; fi
	wait "$pid" 2>/dev/null
	listening 18093 && { note "  :18093 still listening after stop: FAIL"; ok=FAIL; }
	note "  server pid $pid stopped $(date -u +%FT%TZ)"
	note "STEP3: $ok"
	S3=$ok
}

S1=NOT_RUN S2=NOT_RUN S3=NOT_RUN
for s in $STEPS; do
	case $s in 1) step1 ;; 2) step2 ;; 3) step3 ;; *) echo "unknown step $s" ;; esac
done
note "RC2_GATE: step1 $S1 | step2 $S2 | step3 $S3 | evidence $EV"
( cd "$EV" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
echo "summary $SUM"
