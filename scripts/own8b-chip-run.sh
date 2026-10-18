#!/usr/bin/env bash
# OWN-8B stage 2: one sealed chip run of own8b_check under the GB10 lock protocol.
#
#   scripts/own8b-chip-run.sh <evidence-dir> <run-name> <binary> <checkpoint> <golden.json> [own8b_check chip args...]
#
# Refuses while the quiet flag is up (quietlock check). Takes /tmp/aien-gb10.lock for the whole
# run (a serving process waits, it is never killed), records start/end stamps, the binary and
# golden digests, the exact command, raw output and exit code, then refreshes SHA256SUMS in the
# evidence dir. A control run (mutant flag) is expected to exit 1 (FAIL).
set -uo pipefail
ev=$1; name=$2; bin=$3; ckpt=$4; golden=$5; shift 5
quietlock check || { echo "quiet flag is up: no chip run"; exit 4; }
mkdir -p "$ev/$name"
repo=$(cd "$(dirname "$0")/.." && pwd)
{
	echo "start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
	echo "sc commit $(git -C "$repo" rev-parse HEAD) branch $(git -C "$repo" rev-parse --abbrev-ref HEAD)"
	echo "binary $bin sha256 $(sha256sum "$bin" | cut -d' ' -f1)"
	echo "golden $golden sha256 $(sha256sum "$golden" | cut -d' ' -f1)"
	echo "omega ${AIEN_OMEGA_DIR:-?} at $(git -C "${AIEN_OMEGA_DIR:-/nonexistent}" rev-parse HEAD 2>/dev/null)"
	echo "cmd flock /tmp/aien-gb10.lock $bin chip $ckpt $golden $ev/$name $*"
	free -g | head -2
	uname -r
} > "$ev/$name.start"
flock /tmp/aien-gb10.lock "$bin" chip "$ckpt" "$golden" "$ev/$name" "$@" > "$ev/$name.raw" 2> "$ev/$name.err"
rc=$?
echo "$rc" > "$ev/$name.rc"
{ echo "end $(date -u +%Y-%m-%dT%H:%M:%SZ)"; free -g | head -2; } > "$ev/$name.end"
( cd "$ev" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
echo "$name rc=$rc"
grep -E "^(LOAD|PARITY|PREFILL|TOKENS|SPEED|SUMMARY)" "$ev/$name.raw"
exit "$rc"
