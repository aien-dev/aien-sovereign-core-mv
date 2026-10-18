# Shared helpers of the persistent KV chip scripts (sc#507 #508 #509). Sourced, never run.
# CHIP-ONLY callers: own30b-pkv-chip-check.sh, own30b-pkv-bench.sh, own30b-pkv-ops-check.sh.
#
# Rules every caller inherits:
#   - refuses while the live server on :18091 answers (no override: two omega processes on
#     the chip at once is UNKNOWN territory), and refuses an occupied port (PKV_CHECK_PORT,
#     default 18098; never 18091, 18092 or 18095, which belong to other services);
#   - starts its own aien-serve-http only on 127.0.0.1:$PORT, stops only the PID it started;
#   - key: atlas-vault entry AIEN_KV_STORE_KEY by default (`atlas-vault run AIEN_KV_STORE_KEY
#     --`, the key never passes through this shell), or PKV_CHECK_EPHEMERAL_KEY=1 for a
#     throwaway key held only in this shell and the child's environment; never printed or saved;
#   - evidence: raw files in the evidence dir, a summary, SHA256SUMS over every file.
# shellcheck shell=bash disable=SC2034 # DRY, fail, SERVER_JOB, ... are read by the callers

PKV_LIVE_PORT=18091
MODEL=${OWN30B_MODEL_DIR:-$HOME/models/qwen3-30b-a3b-instruct-2507-fp8}
PORT=${PKV_CHECK_PORT:-18098}
PKV_RESERVED_PORTS="18091 18092 18095" # live 30B, spark-mail, spark-cockpit
pkv_port_reserved() { case " $PKV_RESERVED_PORTS " in *" $1 "*) return 0 ;; esac; return 1; }
base="http://127.0.0.1:$PORT"
DRY=0
SERVER_JOB=

pkv_usage_args() { # <script> "$@": sets dir BIN DRY
	local me=$1
	shift
	if [ "${1:-}" = --dry-run ]; then
		DRY=1
		shift
	fi
	if [ $# -ne 2 ]; then
		echo "usage: $me [--dry-run] <evidence-dir> <aien-serve-http binary>" >&2
		exit 64
	fi
	dir=$1 BIN=$2
}

pkv_live_up() { curl -s -m 2 "http://127.0.0.1:$PKV_LIVE_PORT/health" >/dev/null 2>&1; }
pkv_port_busy() { ss -Hltn "sport = :$PORT" | grep -q . || curl -s -m 2 "$base/health" >/dev/null 2>&1; }

# Prerequisites shared by every script. Prints one line per check; returns non-zero on a hard miss.
# pkv_prereqs <needs-key 0|1>
pkv_prereqs() {
	local bad=0 t
	for t in curl jq awk ss sha256sum setsid od; do
		if command -v "$t" >/dev/null; then echo "  ok   tool $t"; else echo "  MISS tool $t"; bad=1; fi
	done
	if [ -x "$BIN" ]; then
		echo "  ok   binary $BIN (sha256 $(sha256sum "$BIN" | cut -c1-16)...)"
	else
		echo "  MISS binary $BIN (not an executable file)"
		bad=1
	fi
	if [ -f "$MODEL/config.json" ]; then echo "  ok   model $MODEL"; else echo "  MISS model $MODEL/config.json"; bad=1; fi
	if pkv_live_up; then
		echo "  STOP live server on :$PKV_LIVE_PORT answers: the real run would refuse (pause it first)"
		bad=1
	else
		echo "  ok   :$PKV_LIVE_PORT does not answer"
	fi
	if pkv_port_reserved "$PORT"; then echo "  STOP :$PORT belongs to another service (reserved: $PKV_RESERVED_PORTS)"; bad=1; elif pkv_port_busy; then echo "  STOP something listens on :$PORT (pick a free PKV_CHECK_PORT)"; bad=1; else echo "  ok   :$PORT is free"; fi
	if [ "${1:-1}" = 1 ]; then
		if [ "${PKV_CHECK_EPHEMERAL_KEY:-0}" = 1 ]; then
			echo "  ok   key: ephemeral (PKV_CHECK_EPHEMERAL_KEY=1, lives only in this run)"
		elif ! command -v atlas-vault >/dev/null; then
			echo "  MISS atlas-vault (or set PKV_CHECK_EPHEMERAL_KEY=1)"
			bad=1
		elif atlas-vault list 2>/dev/null | grep -qx AIEN_KV_STORE_KEY; then
			echo "  ok   key: atlas-vault entry AIEN_KV_STORE_KEY exists (listed only, not read)"
		else
			echo "  MISS atlas-vault entry AIEN_KV_STORE_KEY (or set PKV_CHECK_EPHEMERAL_KEY=1)"
			bad=1
		fi
	fi
	local avail
	avail=$(df -Pk /tmp | awk 'NR==2 {print int($4/1048576)}')
	echo "  info /tmp free: ${avail} GiB on $(df -P /tmp | awk 'NR==2 {print $1}')"
	return $bad
}

# The real run: refuse, then make the private work dir and the key. Sets work PIDF KEY.
pkv_begin() {
	if pkv_live_up; then
		echo "refusing: the live server on :$PKV_LIVE_PORT is up (pause it first)" >&2
		exit 2
	fi
	if pkv_port_reserved "$PORT"; then
		echo "refusing: :$PORT belongs to another service (reserved: $PKV_RESERVED_PORTS)" >&2
		exit 2
	fi
	if pkv_port_busy; then
		echo "refusing: something already listens on :$PORT (18095 is spark-cockpit; pick a free PKV_CHECK_PORT)" >&2
		exit 2
	fi
	[ -x "$BIN" ] || { echo "no executable binary at $BIN" >&2; exit 2; }
	mkdir -p "$dir"
	work=$(mktemp -d /tmp/pkv-chip.XXXXXX)
	chmod 700 "$work"
	PIDF="$work/server.pid"
	trap 'stop_server || PKV_LEAK=1; rm -rf "$work"; pkv_trap_status' EXIT
	KEY=
	if [ "${PKV_CHECK_EPHEMERAL_KEY:-0}" = 1 ]; then
		KEY=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')
	fi
	{
		echo "binary $BIN sha256 $(sha256sum "$BIN" | cut -d' ' -f1)"
		echo "model $MODEL port $PORT key-mode $([ -n "$KEY" ] && echo ephemeral || echo atlas-vault)"
		echo "started $(date -u +%FT%TZ) host $(hostname)"
	} >"$dir/run-info.txt"
}

# The name the server answers to (it refuses any other model name).
SERVED_MODEL=${PKV_SERVED_MODEL:-qwen3-30b-a3b-instruct-2507}

server_pid() { [ -f "$PIDF" ] && cat "$PIDF"; }

# Fails (returns 1) if anything still owns our port after a stop: never start the next server
# beside a leftover one (two 30 GB loads on the chip). Mid-run the callers' set -e stops the run;
# the EXIT trap records it and carries on with its cleanup (an exit inside the trap would end the
# trap there and leave the other server and the work dir behind, sc#523).
port_released() {
	for _ in $(seq 1 50); do ss -Hltn "sport = :$PORT" | grep -q . || return 0; sleep 0.2; done
	echo "something still listens on :$PORT after the stop; stopping the run" >&2
	return 1
}

# The last step of every EXIT trap: a stop that left the port held (PKV_LEAK=1) fails the run,
# after every other cleanup step has run.
pkv_trap_status() { [ -z "${PKV_LEAK:-}" ] || exit 1; }

# Stops our server: SIGTERM (graceful drain, #509), SIGKILL after PKV_STOP_WAIT_S (default 120).
stop_server() {
	[ -n "${PIDF:-}" ] && [ -f "$PIDF" ] || return 0
	local p
	p=$(cat "$PIDF")
	if [ -n "$p" ] && [ "$(ps -o comm= -p "$p" 2>/dev/null || true)" = aien-serve-http ]; then
		kill "$p"
		for _ in $(seq 1 $((${PKV_STOP_WAIT_S:-120} * 10))); do kill -0 "$p" 2>/dev/null || break; sleep 0.1; done
		if kill -0 "$p" 2>/dev/null; then
			echo "server $p ignored SIGTERM for ${PKV_STOP_WAIT_S:-120} s: SIGKILL" >&2
			kill -9 "$p" 2>/dev/null || true
		fi
	fi
	rm -f "$PIDF"
	port_released
}

# kill -9 our server now (the interrupted-write case).
kill9_server() {
	local p
	p=$(cat "$PIDF")
	if [ -n "$p" ] && [ "$(ps -o comm= -p "$p" 2>/dev/null || true)" = aien-serve-http ]; then
		kill -9 "$p"
		for _ in $(seq 1 100); do kill -0 "$p" 2>/dev/null || break; sleep 0.1; done
	fi
	rm -f "$PIDF"
	port_released
}

# start_server <log-name> <store-dir | - (unset) | nokey:<dir> | badkey:<dir> | foreignkey:<dir>> [flags...]
# Sets SERVER_JOB (the background job, for `wait`); the server PID is in $PIDF.
# Extra environment for the child: PKV_SERVER_ENV="A=1 B=2" (no spaces inside values).
start_server() {
	local name=$1 store=$2
	shift 2
	local log="$dir/$name.log"
	local cmd="echo \$\$ > '$PIDF'; exec '$BIN' --engine omega --model-dir '$MODEL' --port $PORT $*"
	local -a extra=()
	# shellcheck disable=SC2206
	[ -n "${PKV_SERVER_ENV:-}" ] && extra=($PKV_SERVER_ENV)
	case $store in
	-) setsid env -u AIEN_KV_STORE_DIR -u AIEN_KV_STORE_KEY "${extra[@]}" sh -c "$cmd" >"$log" 2>&1 </dev/null & ;;
	nokey:*) setsid env -u AIEN_KV_STORE_KEY AIEN_KV_STORE_DIR="${store#nokey:}" "${extra[@]}" sh -c "$cmd" >"$log" 2>&1 </dev/null & ;;
	badkey:*) setsid env AIEN_KV_STORE_DIR="${store#badkey:}" AIEN_KV_STORE_KEY=not-a-key "${extra[@]}" sh -c "$cmd" >"$log" 2>&1 </dev/null & ;;
	foreignkey:*)
		# A valid key that is not the store's: a store written by someone else.
		setsid env AIEN_KV_STORE_DIR="${store#foreignkey:}" AIEN_KV_STORE_KEY="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')" "${extra[@]}" sh -c "$cmd" >"$log" 2>&1 </dev/null &
		;;
	*)
		if [ -n "${KEY:-}" ]; then
			setsid env AIEN_KV_STORE_DIR="$store" AIEN_KV_STORE_KEY="$KEY" "${extra[@]}" sh -c "$cmd" >"$log" 2>&1 </dev/null &
		else
			setsid atlas-vault run AIEN_KV_STORE_KEY -- env AIEN_KV_STORE_DIR="$store" "${extra[@]}" sh -c "$cmd" >"$log" 2>&1 </dev/null &
		fi
		;;
	esac
	SERVER_JOB=$!
	for _ in $(seq 1 "${PKV_START_WAIT_S:-600}"); do
		grep -q "serving \|load failed\|bind failed" "$log" 2>/dev/null && break
		sleep 1
	done
	grep -q "serving " "$log" || { echo "server $name did not start; see $log" >&2; exit 1; }
	# The PID file can hold a wrapper, not the server: atlas-vault re-wraps the command in its own
	# `sh -c "..."`, which expands $$ to the wrapper. Record the process that owns our port.
	local lp
	lp=$(ss -Hltnp "sport = :$PORT" | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
	if [ -z "$lp" ] || [ "$(ps -o comm= -p "$lp" 2>/dev/null || true)" != aien-serve-http ]; then
		echo "server $name: no aien-serve-http owns :$PORT" >&2
		exit 1
	fi
	echo "$lp" >"$PIDF"
	grep "persistent KV" "$log" || true
}

health() { curl -s -m 10 "$base/health" >"$dir/$1-health.json"; cat "$dir/$1-health.json"; }
pk() { jq -r ".persistent_kv.$2" "$dir/$1-health.json"; }

# The same greedy two-turn conversation as #507.
SYS="You are the OWN-30B assistant on the AIEN Spark. Answer briefly."
for i in $(seq 1 40); do SYS="$SYS Rule $i: keep answers short, never invent facts, say UNKNOWN when unsure."; done
Q1="Name three uses of a torque wrench."
Q2="Which of those matters most on a bicycle, and why?"
m1=$(jq -n --arg s "$SYS" --arg q "$Q1" '[{role:"system",content:$s},{role:"user",content:$q}]')
turn2_msgs() { jq -n --argjson m "$m1" --arg a "$1" --arg q "${2:-$Q2}" '$m + [{role:"assistant",content:$a},{role:"user",content:$q}]'; }

chat() { # <out-name> <messages-json>: prints the content
	jq -n --arg model "$SERVED_MODEL" --argjson m "$2" '{model:$model, messages:$m, max_tokens:64, temperature:0, seed:1, stream:false}' |
		curl -s -m 600 "$base/v1/chat/completions" -H 'content-type: application/json' -d @- >"$dir/$1.json"
	local c
	c=$(jq -r '.choices[0].message.content // empty' "$dir/$1.json" 2>/dev/null)
	if [ -z "$c" ]; then
		echo "chat $1: no reply ($(head -c 200 "$dir/$1.json")); stopping the run" >&2
		exit 1
	fi
	printf '%s\n' "$c"
}

wait_parks() { # until parked_chunks is a number > 0 and unchanged for 3 s
	local last=-1 now
	for _ in $(seq 1 60); do
		now=$(curl -s -m 5 "$base/health" | jq -r '.persistent_kv.parked_chunks' 2>/dev/null || true)
		# A failed request, null or a missing counter is not a count (sc#523): never settled.
		case $now in '' | *[!0-9]*) now=-1 ;; esac
		[ "$now" -gt 0 ] && [ "$now" = "$last" ] && return 0
		last=$now
		sleep 3
	done
	echo "parks did not settle" >&2
	exit 1
}

fail=0
# Empty or null on either side is a FAIL: two failed requests must never compare equal.
check() { if [ -n "$2" ] && [ "$2" != null ] && [ "$2" = "$3" ]; then echo "PASS $1"; else echo "FAIL $1 ($2 != $3)"; fail=1; fi; }

pkv_seal() { # writes SHA256SUMS over every evidence file
	local sums
	sums=$(cd "$dir" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum)
	printf '%s\n' "$sums" >"$dir/SHA256SUMS"
}

flip_byte() { # <file>: XOR 0x40 into the middle byte, in place
	local f=$1 sz off b
	sz=$(stat -c %s "$f")
	off=$((sz / 2))
	b=$(dd if="$f" bs=1 skip=$off count=1 2>/dev/null | od -An -tu1 | tr -d ' ')
	printf '%b' "\\$(printf '%03o' $(((b ^ 64) & 255)))" | dd of="$f" bs=1 seek=$off count=1 conv=notrunc 2>/dev/null
}
