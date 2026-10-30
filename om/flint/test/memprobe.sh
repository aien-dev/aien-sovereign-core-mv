#!/usr/bin/env bash
# Load-memory probe for the native Flint forward (Core #834 item 4). Measurement harness only:
# it runs one command under GNU time -v and samples the child's /proc/<pid>/status,
# /proc/<pid>/smaps_rollup and /proc/<pid>/io every INTERVAL seconds until it exits.
# Usage: bash om/flint/test/memprobe.sh OUTDIR -- COMMAND [ARGS...]
# Writes OUTDIR/time.txt (time -v), OUTDIR/samples.tsv (one row per sample), OUTDIR/maps.first and
# OUTDIR/maps.last (/proc/<pid>/maps), OUTDIR/stdout, OUTDIR/stderr, OUTDIR/rc.
# samples.tsv columns (kB unless named): t_ms rchar_bytes VmPeak VmHWM VmRSS RssAnon RssFile
# RssShmem smaps_Rss smaps_Pss smaps_Anonymous file_maps (mappings of files other than the binary).
set -u
out=$1
shift
[ "${1:-}" = -- ] || { echo "usage: memprobe.sh OUTDIR -- COMMAND [ARGS...]" >&2; exit 2; }
shift
iv=${INTERVAL:-0.1}
mkdir -p "$out"
/usr/bin/time -v -o "$out/time.txt" "$@" > "$out/stdout" 2> "$out/stderr" < /dev/null &
tp=$!
pid=
for _ in $(seq 200); do
  pid=$(cat "/proc/$tp/task/$tp/children" 2> /dev/null | cut -d' ' -f1)
  [ -n "$pid" ] && break
  sleep 0.01
done
[ -n "$pid" ] || { wait "$tp"; echo "memprobe: child of time not found" >&2; exit 3; }
exe=$(readlink "/proc/$pid/exe")
st=$(date +%s%N)
kv() { # file key -> value of "key: N kB" or "key: N"
  local k v rest
  while read -r k v rest; do [ "$k" = "$2:" ] && { echo "$v"; return; }; done < "$1"
  echo -
}
printf 't_ms\trchar\tVmPeak\tVmHWM\tVmRSS\tRssAnon\tRssFile\tRssShmem\tsm_Rss\tsm_Pss\tsm_Anon\tfile_maps\n' > "$out/samples.tsv"
first=1
while [ -r "/proc/$pid/status" ] && [ -n "$(kv "/proc/$pid/status" VmHWM 2> /dev/null | tr -d -)" ]; do
  cat "/proc/$pid/status" > "$out/.st" 2> /dev/null || break
  cat "/proc/$pid/smaps_rollup" > "$out/.sr" 2> /dev/null || break
  cat "/proc/$pid/io" > "$out/.io" 2> /dev/null || break
  cat "/proc/$pid/maps" > "$out/maps.last" 2> /dev/null || break
  [ "$first" = 1 ] && { cat "$out/maps.last" > "$out/maps.first"; first=0; }
  fm=$(awk -v e="$exe" '$6 ~ /^\// && $6 != e' "$out/maps.last" | wc -l)
  now=$(( ($(date +%s%N) - st) / 1000000 ))
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$now" "$(kv "$out/.io" rchar)" \
    "$(kv "$out/.st" VmPeak)" "$(kv "$out/.st" VmHWM)" "$(kv "$out/.st" VmRSS)" \
    "$(kv "$out/.st" RssAnon)" "$(kv "$out/.st" RssFile)" "$(kv "$out/.st" RssShmem)" \
    "$(kv "$out/.sr" Rss)" "$(kv "$out/.sr" Pss)" "$(kv "$out/.sr" Anonymous)" "$fm" >> "$out/samples.tsv"
  sleep "$iv"
done
wait "$tp"
echo $? > "$out/rc"
rm -f "$out/.st" "$out/.sr" "$out/.io"
cat "$out/rc"
