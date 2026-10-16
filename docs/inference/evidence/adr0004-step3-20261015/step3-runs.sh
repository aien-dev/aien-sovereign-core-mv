#!/bin/bash
# ADR 0004 migration step 3: G30-tokens and G30-speed with the 30B adapter present (the old loop still runs).
#   check-full: own30b_check chip, the same arguments as own30b-idle-speed-20261015/own30b-idle-gates.sh
#               check-full plus --force-memory (the idle exp445 server on :18091 keeps ~36 GB resident,
#               so MemAvailable is below the tool's 70 GB guard; the guard only refuses, it changes nothing).
#   door:       door-window.sh (G30-door), its own server on PORT.
# One quietlock hold (owner STEP3, 20 min max) per run, released between runs. check-full also runs
# under /tmp/aien-gb10.lock; the door server takes that lock itself around each generation.
#   EV=<evidence dir> SC=<worktree at main> [RUN=<n>] step3-runs.sh   (RUN=n names the outputs check-full-n, door-n); SKIP_DOOR=1 runs check-full only
set -u
SC=${SC:?worktree}
EV=${EV:?evidence dir}
B=$SC/target/release
CK=$HOME/models/qwen3-30b-a3b-instruct-2507-fp8
G=$SC/docs/inference/evidence/own30b-f-20261013T1745Z/golden.json
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$EV"
ground() { # name
	{
		echo "queued $(date -u +%FT%TZ) sc $(git -C "$SC" rev-parse HEAD) omega $(cat "$SC/omega.lock")"
		echo "tracked changes outside this folder: $(git -C "$SC" status --porcelain | grep -v '^?? docs/inference/evidence/adr0004-step3' | grep -cv '^?? target')"
		echo "libomega_gpu.a sha256 $(sha256sum "$B"/build/aien-omega-gpu-*/out/omega-build/libomega_gpu.a | cut -d' ' -f1)"
		echo "driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader) kernel $(uname -r)"
		uptime; grep MemAvailable /proc/meminfo
		nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
		nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw,pstate --format=csv
		echo "listeners: $(ss -ltn | awk 'NR>1{print $4}' | grep -E ':180[0-9][0-9]$' | sort | tr '\n' ' ')"
		echo ":18091 log lines $(wc -l < "$HOME/workspace/exp445-runlogs/own30b-18091-f881710a.log" 2>/dev/null)"
	} > "$EV/$1.start"
}
finish() { # name rc
	echo "$2" > "$EV/$1.rc"
	{
		echo "ended $(date -u +%FT%TZ)"; uptime; grep MemAvailable /proc/meminfo
		echo ":18091 log lines $(wc -l < "$HOME/workspace/exp445-runlogs/own30b-18091-f881710a.log" 2>/dev/null)"
	} > "$EV/$1.end"
	echo "$1 rc=$2"
}
R=${RUN:+-$RUN}
TOK="--prompts 1,2,3,4 --new 16 --layers 0,23,47 --weights fp8 --attn fused --router device --lm-head gemv --prefill auto"

ground check-full$R
echo "binary $B/own30b_check sha256 $(sha256sum "$B/own30b_check" | cut -d' ' -f1)" >> "$EV/check-full$R.start"
echo "cmd quietlock hold --owner STEP3 --minutes 20 -- flock /tmp/aien-gb10.lock env AIEN_OMEGA_SPIN_US=2000 $B/own30b_check chip $CK $G $EV/check-full$R $TOK --force-memory" >> "$EV/check-full$R.start"
# shellcheck disable=SC2086
quietlock hold --owner STEP3 --minutes 20 --reason "ADR 0004 step 3 own30b_check chip (adapter present)" -- \
	flock /tmp/aien-gb10.lock sh -c 'echo "lock acquired $(date -u +%FT%TZ)" >> "$0.start"; shift; exec "$@"' "$EV/check-full$R" x \
	env AIEN_OMEGA_SPIN_US=2000 "$B/own30b_check" chip "$CK" "$G" "$EV/check-full$R" $TOK --force-memory \
	> "$EV/check-full$R.raw" 2> "$EV/check-full$R.err"
finish check-full$R $?

if [ -n "${SKIP_DOOR:-}" ]; then echo "all done $(date -u +%FT%TZ) (door skipped)"; exit 0; fi
ground door$R
echo "binary $B/aien-serve-http sha256 $(sha256sum "$B/aien-serve-http" | cut -d' ' -f1)" >> "$EV/door$R.start"
quietlock hold --owner STEP3 --minutes 20 --reason "ADR 0004 step 3 door seeded sampling" -- \
	"$HERE/../adr0004-step0-baseline-20261015/door-window.sh" "$B/aien-serve-http" "$EV/door$R" 18103 > "$EV/door$R.raw" 2> "$EV/door$R.err"
finish door$R $?
echo "all done $(date -u +%FT%TZ)"
