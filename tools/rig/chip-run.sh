#!/usr/bin/env bash
# tools/rig/chip-run.sh: one measured run for aien-rig on the DGX Spark (GB10), under the shared rules:
# waits until MemAvailable >= NEED_GIB + 12 GiB (12 GiB always left for other jobs), then runs the command
# through spark-receipt (wall/GPU power, load, memory, NVRM errors) under /tmp/aien-gb10.lock, and seals
# stdout, stderr, ground facts and the receipt in <evidence-dir>/<name>/ with SHA256SUMS.
#
#   tools/rig/chip-run.sh <evidence-dir> <name> <need-gib> [--no-lock] -- CMD ARGS...
#
# --no-lock: host-only runs (no chip): no GPU lock, still the memory wait and the receipt.
set -uo pipefail
ev=$1; name=$2; need=$3; shift 3
lock=1
if [[ ${1:-} == --no-lock ]]; then lock=0; shift; fi
[[ ${1:-} == -- ]] && shift
d="$ev/$name"; mkdir -p "$d"
want_kb=$(( (need + 12) * 1024 * 1024 ))
while :; do
	avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
	(( avail >= want_kb )) && break
	echo "$(date -u +%FT%TZ) waiting: MemAvailable $((avail / 1024 / 1024)) GiB < $((need + 12)) GiB" >> "$d/wait.log"
	sleep 30
done
repo=$(cd "$(dirname "$0")/../.." && pwd)
{
	echo "start $(date -u +%FT%TZ)"
	echo "host $(hostname) chip GB10 (DGX Spark), NOT an RTX 5070"
	echo "sc $(git -C "$repo" rev-parse HEAD) branch $(git -C "$repo" rev-parse --abbrev-ref HEAD) dirty $(git -C "$repo" status --porcelain | wc -l)"
	echo "binary $1 sha256 $(sha256sum "$1" | cut -d' ' -f1)"
	echo "omega archive ${AIEN_OMEGA_GPU_LIB:-?} sha256 $(sha256sum "${AIEN_OMEGA_GPU_LIB:-/dev/null}" | cut -d' ' -f1)"
	echo "env AIEN_Q38_W4_RULE=${AIEN_Q38_W4_RULE:-unset}"
	echo "cmd $*"
	echo "loadavg $(cat /proc/loadavg)"
	grep MemAvailable /proc/meminfo
	uname -r
} > "$d/ground.txt"
if (( lock )); then
	~/.local/bin/spark-receipt "rig-$name" -- flock /tmp/aien-gb10.lock "$@" > "$d/stdout.txt" 2> "$d/stderr.txt"
else
	~/.local/bin/spark-receipt "rig-$name" -- nice -n 10 "$@" > "$d/stdout.txt" 2> "$d/stderr.txt"
fi
rc=$?
echo "$rc" > "$d/rc"
{ echo "end $(date -u +%FT%TZ)"; echo "loadavg $(cat /proc/loadavg)"; grep MemAvailable /proc/meminfo; } >> "$d/ground.txt"
r=$(ls -t ~/yardsticks/receipts/*-"rig-$name".json 2>/dev/null | head -1)
[[ -n $r ]] && cp "$r" "$d/receipt.json"
( cd "$d" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
echo "$name rc=$rc"
grep -E "^(LOAD|W4|DEVICE|PARITY|PREFILL|TOKENS|SPEED|SUMMARY)" "$d/stdout.txt" 2>/dev/null
exit "$rc"
