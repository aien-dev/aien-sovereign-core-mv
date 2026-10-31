#!/usr/bin/env bash
# DRA-66 (sovereign #857, PR #863) chip slice library: sourced by sNN-*.sh, never run alone.
# The slice sets SLICE and GATES (gate names, run in order: reds before greens), then sources this.
# See PLAN.md beside it for the gates, the Wolfram wall model and the window script contract.
#
# Run only by ~/handoffs/pdra4-hive/chip/run-queue.sh. Every slice carries '# pdra4-gb10: none', so the
# runner opens `spark-window open pdra4-DRA-66-<slice> --lease 20 --gb10 none --quiet exclusive --wait`
# and runs `timeout 1140 spark-window run env Q36_EXTRA_WAIT=120 ... q36-extra <slice>` (no args).
# Why gb10 none: the server verifies an inherited chip reservation by requiring the FLOCK holder of
# /tmp/aien-gb10.lock to be one of its ANCESTORS (crates/aien-serve-http/src/chip_reservation.rs
# `detect`); spark-window's holder (`setsid spark-window _hold ID`) never is. So the slice takes the
# lock itself through flock(1) (at most 90 s), which is the ancestor of every server it starts, and
# sets AIEN_GB10_LOCK_HELD=1 only for those servers. flock(1) exits, and the lock goes, when the inner
# slice exits; timeout(1) signals the whole process group at the cap. The window's quiet lock (--quiet
# exclusive) is what makes the timings quiet.
#
# Refuses with NOT_RUN (exit 3) before any chip contact unless: SPARK_WINDOW and SPARK_WINDOW_DIR name
# the open, held window; it is gb10=none and quiet=exclusive; AIEN_GB10_LOCK_HELD is not set; the chip
# lock comes free within 90 s and its holder is an ancestor; 18091/18300/18310 are down; no compute
# process is on the chip; binary and golden have their pinned sha256.
# Outside a window it prints NOT_RUN and exits 3 WITHOUT a VERDICT file (the runner would skip a slice
# with a VERDICT); inside one every outcome writes receipts/DRA-66/<slice>/VERDICT.
#
# Budget: a gate starts only if its cap fits before BOTH the lease end and the runner's 1140 s cap less
# 15 s (measured from the start of this process: run-queue.sh -> timeout -> spark-window run -> env ->
# q36-extra -> slice is one exec chain, one pid); otherwise it and the later gates are NOT_RUN.
#
# Host dry path, no GPU and no 35B: `DRY_RUN=1 bash <slice>` checks binary/golden/model/tools, prints the
# plan, and runs the slice's host-side checker gate on the committed host fixtures into /tmp/dra66-dry/.
set -u
umask 022

W=/home/drakestapleton/work/dra-66
BIN=$W/bin-head/aien-serve-http
SHA_HEAD=a33cc2abf545927b293316cca018ff81d492bacfdd8fb996dcc5dccdb02449e4 # sc 9d70dca4 + omega a647b6a9
GOLD=$W/core/evidence/q36-host-20261020/gen-chat-max-24.json
GOLD_SHA=ec9040177d67a3f33b4d8800bc6656990952a8479052468814386d9081e6a664
HOSTEV=$W/core/evidence/dra-66-host-time-20261031 # host fixtures for the dry path
MODEL=/home/drakestapleton/models/qwen3.6-35b-a3b-fp8-95a723d0
PORT=18190
LOCK=/tmp/aien-gb10.lock
WAVE=0 # explicit: the 18091 launcher leaves it unset, the a647b6a9 default (off); proof checked in the log
RUNNER_CAP=1140
LOAD=180 # server load to /health (measured 39-78 s in the GoldenP04 window; 148.6 s at 13:32Z beside Flint)
RQ=/home/drakestapleton/handoffs/pdra4-hive/receipts/DRA-66
R=$RQ/$SLICE
S01=$RQ/s01-tokens # s02 reads the token slice's summaries from here
SELF=$(readlink -f "$0")
COUNT='Count from 1 to 400 in English words, one number per line, nothing else.'

ts() { date -u +%FT%TZ; }

# cap NAME: the gate's hard cap in seconds (PLAN.md, Wolfram).
cap() {
  case $1 in
    mutant4-red) echo $(( LOAD + 60 + 30 )) ;;     # load, golden (60 s curl cap), stop
    golden-identity) echo $(( LOAD + 60 )) ;;      # load, golden
    receipt-1conv) echo $(( 3 * 30 + 2 )) ;;
    receipt-8conv) echo $(( 60 + 2 )) ;;
    seed-t07) echo $(( 3 * 20 )) ;;
    check-tokens) echo $(( 30 + 45 )) ;;           # stop the token server, then the checker
    stamps-1conv) echo $(( LOAD + 30 + 2 + 30 )) ;; # load, 128-token reply, idle, stop
    check-stamps) echo 45 ;;
    *) echo 0 ;;
  esac
}

dry_static() {
  local fail=0 t
  for t in curl jq ss sha256sum flock spark-window nvidia-smi; do command -v $t > /dev/null || { echo "missing tool $t"; fail=1; }; done
  [ "$(sha256sum < "$BIN" | cut -c1-64)" = "$SHA_HEAD" ] || { echo "$BIN sha256 is not $SHA_HEAD"; fail=1; }
  [ "$(sha256sum < "$GOLD" | cut -c1-64)" = "$GOLD_SHA" ] || { echo "$GOLD sha256 is not $GOLD_SHA"; fail=1; }
  [ "$(jq '.generated_ids | length' "$GOLD" 2> /dev/null)" = 24 ] || { echo "golden has no 24 generated_ids"; fail=1; }
  [ -f "$MODEL/config.json" ] || { echo "model dir $MODEL missing"; fail=1; }
  return $fail
}

# ---- host dry path: no window, no lock, no GPU, no server ----
if [ "${DRY_RUN:-0}" = 1 ]; then
  DRY=/tmp/dra66-dry/$SLICE
  rm -rf "$DRY"
  mkdir -p "$DRY"
  dry_static || { echo "DRY $SLICE: static checks FAILED"; exit 3; }
  total=90
  echo "DRY $SLICE: static checks passed (bin $SHA_HEAD, golden $GOLD_SHA); receipts -> $R; WAVE=$WAVE"
  for g in "${GATES[@]}"; do echo "  would run: $g (cap $(cap "$g") s)"; total=$(( total + $(cap "$g") )); done
  echo "  caps + 90 s lock wait = $total s of the $RUNNER_CAP s runner cap"
  [ "$total" -le "$RUNNER_CAP" ] || { echo "DRY $SLICE: caps exceed the runner cap"; exit 3; }
  DRY_RUN_CHECK=1
fi

# ---- outer: inside the window, take the chip lock through flock(1) ----
if [ -z "${DRA66_INNER:-}" ] && [ -z "${DRY_RUN_CHECK:-}" ]; then
  refuse() { # in a window: VERDICT; outside: stdout only
    echo "NOT_RUN: $*"
    if [ -n "${SPARK_WINDOW:-}" ]; then mkdir -p "$R" && echo "NOT_RUN $SLICE: $*" > "$R/VERDICT"; fi
    exit 3
  }
  [ -n "${SPARK_WINDOW:-}" ] && [ -n "${SPARK_WINDOW_DIR:-}" ] \
    || { echo "NOT_RUN: not inside a spark-window; run by ~/handoffs/pdra4-hive/chip/run-queue.sh only"; exit 3; }
  [ -s "$R/VERDICT" ] && { echo "have VERDICT: $(cat "$R/VERDICT")"; exit 0; }
  [ "$(spark-window id 2> /dev/null)" = "$SPARK_WINDOW" ] || refuse "window $SPARK_WINDOW is not the open window holding its locks"
  [ -f "$SPARK_WINDOW_DIR/held" ] || refuse "window $SPARK_WINDOW has no held stamp"
  gb=$(cat "$SPARK_WINDOW_DIR/gb10" 2> /dev/null)
  q=$(cat "$SPARK_WINDOW_DIR/quiet" 2> /dev/null)
  [ "$gb" = none ] \
    || refuse "window gb10=$gb: its holder is not an ancestor of the server, which would refuse the chip; the slice needs '# pdra4-gb10: none'"
  [ "$q" = exclusive ] || refuse "window quiet=$q: the timing gates need --quiet exclusive"
  [ -z "${AIEN_GB10_LOCK_HELD:-}" ] || refuse "AIEN_GB10_LOCK_HELD=$AIEN_GB10_LOCK_HELD already set; this slice takes the chip lock itself"
  # A slice that names NEEDS (s02: s01-tokens) opens no chip until that slice's VERDICT is PASS: its
  # gate files need the token summaries and its identity check needs s01's solo-1 ids.
  if [ -n "${NEEDS:-}" ] && ! grep -q '^PASS' "$RQ/$NEEDS/VERDICT" 2> /dev/null; then
    refuse "$NEEDS has no PASS VERDICT ($(cut -c1-120 "$RQ/$NEEDS/VERDICT" 2> /dev/null || echo none)); remove this VERDICT to run again once it has"
  fi
  lease=$(cat "$SPARK_WINDOW_DIR/lease")
  now=$(date +%s)
  lease_end=$(( $(stat -c %Y "$SPARK_WINDOW_DIR/held") + lease * 60 ))
  run_end=$(( now - $(ps -o etimes= -p $$) + RUNNER_CAP - 15 ))
  DRA66_DEADLINE=$(( lease_end < run_end ? lease_end : run_end ))
  export DRA66_INNER=1 DRA66_DEADLINE
  echo "$(ts) window $SPARK_WINDOW (gb10 $gb, quiet $q, lease $lease min); $(( DRA66_DEADLINE - now )) s to the deadline; taking $LOCK (at most 90 s)"
  flock -w 90 -E 75 "$LOCK" bash "$SELF"
  rc=$?
  # 75 from the slice would read as a q36-extra refusal in run-queue.sh: map it to our own NOT_RUN.
  [ "$rc" = 75 ] && refuse "$LOCK did not come free in 90 s"
  exit "$rc"
fi

# ---- inner: child of flock(1), which holds the chip lock (or the dry path) ----
if [ -n "${DRY_RUN_CHECK:-}" ]; then
  EV=$DRY
  DRA66_DEADLINE=$(( $(date +%s) + RUNNER_CAP ))
  SPARK_WINDOW=dry
  SPARK_WINDOW_DIR=none
else
  EV=$R
fi
SERVER_PID=
SERVED= # set once a server was started: only then does finish query the chip again
HOLDER=
BIN_SHA=
BIN_COMMIT=
G_T0=$(ts)
G_S0=$(date +%s)
T0=$G_S0

say() { echo "$(ts) $*" | tee -a "$EV/window.log"; }
left() { echo $(( DRA66_DEADLINE - $(date +%s) )); }
elapsed() { echo $(( $(date +%s) - T0 )); }

# gate NAME STATUS DETAIL [FILE...]: the gate's receipt, stamped with window, binary and lock.
gate() {
  local name=$1 status=$2 detail=$3 files
  shift 3
  files=$(for f in "$@"; do [ -f "$f" ] && sha256sum "$f"; done \
    | jq -R 'capture("^(?<sha256>[0-9a-f]{64})  (?<path>.*)$")' | jq -s .)
  jq -n --arg slice "$SLICE" --arg gate "$name" --arg status "$status" --arg detail "$detail" \
    --arg window_id "$SPARK_WINDOW" --arg window_dir "$SPARK_WINDOW_DIR" \
    --arg bin "$BIN" --arg bin_sha256 "$BIN_SHA" --arg sc_commit "$BIN_COMMIT" \
    --arg holder "$HOLDER" --arg start "$G_T0" --arg end "$(ts)" --arg wave "$WAVE" \
    --argjson elapsed_s "$(( $(date +%s) - G_S0 ))" --argjson deadline_left_s "$(left)" \
    --argjson files "$files" \
    '{schema: "aien.dra66-slice-gate/1", slice: $slice, gate: $gate, status: $status, detail: $detail,
      window_id: $window_id, window_dir: $window_dir, bin: $bin, bin_sha256: $bin_sha256,
      sc_commit: $sc_commit, gb10_lock_holder_pid: $holder, grouped_wave: $wave, start: $start,
      end: $end, elapsed_s: $elapsed_s, deadline_left_s: $deadline_left_s, files: $files}' \
    > "$EV/gate-$name.json"
  say "gate $name: $status ($detail)"
}

# begin NAME: start a gate only if its cap fits before the deadline.
begin() {
  G_T0=$(ts)
  G_S0=$(date +%s)
  local l c
  l=$(left)
  c=$(cap "$1")
  if [ "$l" -lt "$c" ]; then
    gate "$1" NOT_RUN "deadline in $l s, gate cap $c s; this and later gates not run"
    exit 3
  fi
  say "gate $1 begins (cap $c s, $l s to the deadline)"
}

not_run() {
  echo "NOT_RUN: $*"
  gate preconditions NOT_RUN "$*"
  exit 3
}

# ---- server control ----
stop_server() {
  [ -n "$SERVER_PID" ] || return 0
  kill -TERM "$SERVER_PID" 2> /dev/null
  for _ in $(seq 30); do kill -0 "$SERVER_PID" 2> /dev/null || break; sleep 1; done
  kill -KILL "$SERVER_PID" 2> /dev/null
  wait "$SERVER_PID" 2> /dev/null
  SERVER_PID=
}

# finish: receipt.json over every gate, then the one-line VERDICT (status = worst gate:
# FAIL > NOT_RUN > PASS). The 0.5 ms verdict is Main's Wolfram evaluation, never part of it.
finish() {
  stop_server
  [ -z "$SERVED" ] || nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    > "$EV/compute-apps-after.txt" 2>&1
  jq -s --arg slice "$SLICE" --arg window_id "$SPARK_WINDOW" --arg bin_sha256 "$BIN_SHA" \
    --arg sc_commit "$BIN_COMMIT" --argjson elapsed_s "$(elapsed)" \
    '{schema: "aien.dra66-slice/1", slice: $slice, window_id: $window_id, bin_sha256: $bin_sha256,
      sc_commit: $sc_commit, elapsed_s: $elapsed_s,
      status: (if any(.[]; .status == "FAIL") then "FAIL"
               elif any(.[]; .status == "NOT_RUN") then "NOT_RUN" else "PASS" end),
      gates: .}' \
    $(ls -tr "$EV"/gate-*.json 2> /dev/null) > "$EV/receipt.json"
  local st why
  st=$(jq -r '.status // "NOT_RUN"' "$EV/receipt.json" 2> /dev/null)
  why=$(jq -r '[.gates[] | select(.status != "PASS") | "\(.gate) \(.status): \(.detail)"][0] // empty' "$EV/receipt.json" 2> /dev/null)
  [ -n "$st" ] || st=NOT_RUN
  echo "$st $SLICE: $(jq -r '[.gates[] | "\(.gate)=\(.status)"] | join(" ")' "$EV/receipt.json" 2> /dev/null)${why:+; first non-PASS: $why}; elapsed $(elapsed) s; receipt.json" \
    | cut -c1-600 > "$EV/VERDICT"
  cat "$EV/VERDICT"
  [ "$st" = FAIL ] && exit 4
  [ "$st" = NOT_RUN ] && exit 3
  exit 0
}

# start_server LOG [ENV=VAL ...]: the live 18091 flags on a loopback test port, inheriting the chip
# lock this slice's flock(1) holds; at most LOAD s to health.
start_server() {
  local log=$1
  shift
  env AIEN_GB10_LOCK_HELD=1 AIEN_Q36_STROKE=1 AIEN_STROKE_IN_WINDOW=1 AIEN_OMEGA_FP8_GROUPED_WAVE=$WAVE "$@" \
    "$BIN" --engine omega --model-dir "$MODEL" --port $PORT --bind 127.0.0.1 \
    --max-context 49152 --max-tokens-cap 16384 --thinking off --batch-slots 8 \
    > "$log" 2>&1 &
  SERVER_PID=$!
  SERVED=1
  local deadline=$(( $(date +%s) + LOAD ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    kill -0 "$SERVER_PID" 2> /dev/null || { say "server exited during load: $(tail -n 2 "$log" | tr '\n' ' ')"; SERVER_PID=; return 1; }
    curl -s -m 2 127.0.0.1:$PORT/health > /dev/null && break
    sleep 1
  done
  curl -s -m 2 127.0.0.1:$PORT/health > /dev/null || { say "server not healthy in $LOAD s"; stop_server; return 1; }
  grep -q "chip reservation inherited from ancestor pid $HOLDER" "$log" \
    || { say "server did not report the inherited reservation of pid $HOLDER"; stop_server; return 1; }
  ! grep -q '(wave rule)' "$log" \
    || { say "server log shows the grouped wave rule; WAVE=$WAVE wants it off"; stop_server; return 1; }
  local l
  l=$(ss -ltnH "sport = :$PORT" | awk '{print $4}')
  if [ -z "$l" ] || [ -n "$(echo "$l" | grep -v '^127\.0\.0\.1:')" ]; then
    say "listener check FAILED on $PORT: $l"
    stop_server
    return 1
  fi
  say "server pid $SERVER_PID up, listeners: $(echo $l)"
}

# chat OUT PROMPT MAX TEMP SEED CAP_S: one chat request with token ids; prints "tokens wall_s".
chat() {
  local out=$1 prompt=$2 max=$3 temp=$4 seed=$5 cap=$6 body wall
  body=$(jq -nc --arg p "$prompt" --argjson m "$max" --argjson t "$temp" --argjson s "$seed" \
    '{model:"qwen3.6-35b-a3b-fp8",messages:[{role:"user",content:$p}],temperature:$t,seed:$s,max_tokens:$m,stream:false,chat_template_kwargs:{enable_thinking:false},aien_return_token_ids:true}')
  wall=$(curl -s -m "$cap" -o "$out" -w '%{time_total}' -H 'content-type: application/json' \
    -d "$body" 127.0.0.1:$PORT/v1/chat/completions) || { echo "fail 0"; return 1; }
  echo "$(jq -r '.usage.completion_tokens // "fail"' "$out") $wall"
}

# golden LABEL: the gate2 request (24 greedy tokens, 60 s cap); prints IDENTICAL, DIFFER or ERROR.
golden() {
  local out=$EV/golden-$1.json got
  chat "$out" 'Write a C function that returns the larger of two ints.' 24 0 0 60 > /dev/null
  got=$(jq -c '.usage.aien_token_ids // empty' "$out" 2> /dev/null)
  if [ -z "$got" ]; then echo ERROR; elif [ "$got" = "$WANT" ]; then echo IDENTICAL; else echo DIFFER; fi
}

chk() { "$BIN" --check-decode-timeline "$@"; }

# check_tokens TOK A B: the token receipt's checker gate (host only). Writes summary-1conv.wl (the
# 0.5 ms gate's token beats A..B) and summary-red-host.wl (the same beats bent +0.6 ms host: the red).
check_tokens() {
  local tok=$1 a=$2 b=$3 n c0 c1 c2 c3 c4 c5 codes st
  n=$(wc -l < "$tok" 2> /dev/null || echo 0)
  chk "$tok" > "$EV/summary-all.wl" 2> "$EV/check.txt"
  c0=$?
  echo "check exit $c0 (must be 0)" >> "$EV/check.txt"
  chk "$tok" --drop 0:record:0 > /dev/null 2> "$EV/red-drop-first.txt"
  c1=$?
  echo "red control (drop the first record) exit $c1 (must be 1)" >> "$EV/red-drop-first.txt"
  chk "$tok" --drop $(( n / 2 )):record:0 > /dev/null 2> "$EV/red-drop-middle.txt"
  c2=$?
  echo "red control (drop record $(( n / 2 ))) exit $c2 (must be 1)" >> "$EV/red-drop-middle.txt"
  chk "$tok" --bend $(( a + 1 )):chip > /dev/null 2> "$EV/red-bend-chip.txt"
  c3=$?
  echo "red control (record $(( a + 1 )) chip_ns + 1, host share kept) exit $c3 (must be 1)" >> "$EV/red-bend-chip.txt"
  chk "$tok" --bend host:600000 --slice "$a:$b" > "$EV/summary-red-host.wl" 2> "$EV/red-bend-host.txt"
  c4=$?
  echo "synthetic receipt (every beat 0.6 ms more host) exit $c4 (must be 0; its Wolfram verdict must be False)" >> "$EV/red-bend-host.txt"
  chk "$tok" --slice "$a:$b" > "$EV/summary-1conv.wl" 2> "$EV/check-1conv.txt"
  c5=$?
  echo "check --slice $a:$b exit $c5 (must be 0)" >> "$EV/check-1conv.txt"
  echo "$a $b $n" > "$EV/slice.txt"
  codes="$c0 $c1 $c2 $c3 $c4 $c5"
  [ "$codes" = "0 1 1 1 0 0" ] && st=PASS || st=FAIL
  gate check-tokens "$st" "$n token records, slice $a..$b; exits $codes (want 0 1 1 1 0 0: reds before the green slice)" \
    "$tok" "$EV/summary-all.wl" "$EV/summary-red-host.wl" "$EV/summary-1conv.wl" "$EV/check.txt" \
    "$EV/red-drop-first.txt" "$EV/red-drop-middle.txt" "$EV/red-bend-chip.txt" "$EV/red-bend-host.txt" \
    "$EV/check-1conv.txt" "$EV/slice.txt"
}

# check_stamps STA TOKDIR: the stamped timeline's checker gate (host only), then Main's Wolfram inputs:
# gate-red.wl first (the bent token beats + the stamps: must evaluate False), then gate.wl.
check_stamps() {
  local sta=$1 tokdir=$2 ns c0 c1 st detail
  ns=$(wc -l < "$sta" 2> /dev/null || echo 0)
  chk "$sta" --drop 0:record:0 > /dev/null 2> "$EV/red-stamps-drop.txt"
  c1=$?
  echo "red control (drop the first stamped record) exit $c1 (must be 1)" >> "$EV/red-stamps-drop.txt"
  chk "$sta" > "$EV/summary-stamps.wl" 2> "$EV/check-stamps.txt"
  c0=$?
  echo "check exit $c0 (must be 0)" >> "$EV/check-stamps.txt"
  [ "$c1 $c0" = "1 0" ] && st=PASS || st=FAIL
  detail="$ns stamped records; exits $c1 $c0 (want 1 0)"
  if [ -s "$tokdir/summary-red-host.wl" ] && [ -s "$tokdir/summary-1conv.wl" ] \
    && grep -q '^PASS' "$tokdir/VERDICT" 2> /dev/null; then
    cat "$tokdir/summary-red-host.wl" "$EV/summary-stamps.wl" > "$EV/gate-red.wl"
    cat "$tokdir/summary-1conv.wl" "$EV/summary-stamps.wl" > "$EV/gate.wl"
    detail="$detail; gate-red.wl then gate.wl from $tokdir (slice $(cat "$tokdir/slice.txt" 2> /dev/null)); Wolfram: gate-red.wl last result False first, then gate.wl"
  else
    [ "$st" = PASS ] && st=NOT_RUN
    detail="$detail; no PASS token slice in $tokdir, so no gate.wl"
  fi
  gate check-stamps "$st" "$detail" "$sta" "$EV/summary-stamps.wl" "$EV/check-stamps.txt" \
    "$EV/red-stamps-drop.txt" "$EV/gate-red.wl" "$EV/gate.wl" "$tokdir/summary-1conv.wl" "$tokdir/summary-red-host.wl"
}

# ---- host dry path: the slice's checker gate on the committed host fixtures ----
if [ -n "${DRY_RUN_CHECK:-}" ]; then
  BIN_SHA=$SHA_HEAD
  trap 'finish' EXIT
  case $SLICE in
    s01-tokens) G_T0=$(ts); G_S0=$(date +%s); check_tokens "$HOSTEV/host-token-receipt.jsonl" 0 19 ;;
    s02-stamps)
      G_T0=$(ts); G_S0=$(date +%s)
      check_stamps "$HOSTEV/host-stamps.jsonl" /tmp/dra66-dry/s01-tokens ;;
  esac
  exit 0
fi

mkdir -p "$EV" || { echo "NOT_RUN: cannot create $EV"; exit 3; }
rm -f "$EV/VERDICT"
trap 'finish' EXIT
trap 'say "interrupted"; exit 4' INT TERM

# ---- preconditions (no chip contact) ----
[ -e "$LOCK" ] || not_run "$LOCK missing"
ino=$(stat -c %i "$LOCK")
dev=$(printf '%02x:%02x' $(stat -c '%Hd %Ld' "$LOCK"))
# /proc/locks field 6 is MAJ:MIN:INODE (hex device): match the device as well as the inode.
holders=$(awk -v ino="$ino" -v dev="$dev" '$2 == "FLOCK" && $4 == "WRITE" { n = split($6, a, ":"); if (a[n] == ino && a[1] ":" a[2] == dev) print $5 }' /proc/locks | sort -u)
anc=" "
p=$$
while [ "$p" -gt 1 ]; do
  anc="$anc$p "
  p=$(awk '{print $4}' "/proc/$p/stat" 2> /dev/null) || break
done
for h in $holders; do case "$anc" in *" $h "*) HOLDER=$h ;; esac; done
[ -n "$HOLDER" ] || not_run "$LOCK held exclusively by pid(s) '$holders', none an ancestor of $$ ($anc)"
if curl -s -m 5 127.0.0.1:18091/health > "$EV/health-18091-before.json"; then
  not_run "18091 answers /health; the live 35B must be stopped first"
fi
for port in 18091 18300 18310; do
  [ -z "$(ss -ltnH "sport = :$port")" ] || not_run "a listener is still on $port"
done
apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1) \
  || not_run "nvidia-smi failed: $apps"
echo "$apps" > "$EV/compute-apps-before.txt"
[ -z "$apps" ] || not_run "compute apps present: $apps"
dry_static > "$EV/static.txt" 2>&1 || not_run "static checks: $(tr '\n' ';' < "$EV/static.txt")"
BIN_SHA=$SHA_HEAD
"$BIN" --version > "$EV/version.txt" 2>&1 || not_run "$BIN --version failed"
BIN_COMMIT=$(awk '/^sovereign-core commit:/ { print $3 }' "$EV/version.txt")
grep -q '^omega gpu engine: native' "$EV/version.txt" || not_run "$BIN has no native omega gpu engine"
WANT=$(jq -c '.generated_ids' "$GOLD")
[ -z "$(ss -ltnH "sport = :$PORT")" ] || not_run "port $PORT busy"
{
  echo "slice $SLICE window $SPARK_WINDOW $SPARK_WINDOW_DIR deadline $DRA66_DEADLINE ($(left) s left)"
  echo "lock $LOCK holder $HOLDER; ancestors$anc"
  echo "bin $BIN $BIN_SHA commit $BIN_COMMIT; WAVE=$WAVE"; echo "golden $GOLD $GOLD_SHA"
  echo "affinity: $(taskset -pc $$ 2>&1)"
  uname -r; nvidia-smi --query-gpu=driver_version,name --format=csv,noheader; cat /proc/loadavg
} > "$EV/preconditions.txt"
gate preconditions PASS "window $SPARK_WINDOW, lock holder $HOLDER (ancestor), $(left) s to the deadline" \
  "$EV/preconditions.txt" "$EV/version.txt" "$EV/compute-apps-before.txt" "$GOLD"

for g in "${GATES[@]}"; do
  begin "$g"
  case $g in
    mutant4-red) # the negative control goes first
      start_server "$EV/server-mutant4.log" AIEN_Q36_TEST_MUTANT=4 \
        || { gate mutant4-red NOT_RUN "mutant server did not come up" "$EV/server-mutant4.log"; exit 3; }
      m4=$(golden mutant4)
      stop_server
      if [ "$m4" = DIFFER ]; then
        gate mutant4-red PASS "mutant 4 DIFFERs from the golden (the control went red)" "$EV/golden-mutant4.json" "$EV/server-mutant4.log"
      else
        gate mutant4-red FAIL "mutant 4 gave $m4, not DIFFER; no green below would count" "$EV/golden-mutant4.json" "$EV/server-mutant4.log"
        exit 4
      fi ;;
    golden-identity) # honest server, per-token receipt on
      TOK=$EV/tokens.jsonl
      start_server "$EV/server-tokens.log" AIEN_Q36_DECODE_TIMELINE="$TOK" AIEN_Q36_DECODE_TIMELINE_KIND=token \
        || { gate golden-identity NOT_RUN "honest server did not come up" "$EV/server-tokens.log"; exit 3; }
      grep -E 'per-token receipt' "$EV/server-tokens.log" | tee -a "$EV/window.log"
      g=$(golden honest)
      sleep 2 # the door closes the golden's last record at idle; every record is flushed as written
      A=$(wc -l < "$TOK" 2> /dev/null || echo 0)
      [ "$g" = IDENTICAL ] && st=PASS || st=FAIL
      gate golden-identity "$st" "honest golden $g (must be IDENTICAL); receipt records so far A=$A" "$EV/golden-honest.json" "$GOLD" "$TOK" ;;
    receipt-1conv) # records A..B are the 0.5 ms gate's beats
      ok=PASS
      for i in 1 2 3; do
        r=$(chat "$EV/solo-$i.json" "$COUNT" 256 0 0 30) || ok=FAIL
        say "solo $i: tokens wall_s = $r"
      done
      sleep 2 # the door closes the spell at idle
      B=$(wc -l < "$TOK" 2> /dev/null || echo 0)
      [ "$B" -gt $(( A + 1 )) ] || ok=FAIL
      gate receipt-1conv "$ok" "3 sequential greedy replies of 256 tokens; receipt records A..B = $A..$B" "$EV"/solo-*.json "$TOK" ;;
    receipt-8conv)
      pids=()
      for i in 1 2 3 4 5 6 7 8; do
        chat "$EV/eight-$i.json" "Count from $((i * 50)) to $((i * 50 + 300)) in English words, one number per line, nothing else." 256 0 0 60 \
          > "$EV/eight-$i.txt" &
        pids+=($!)
      done
      ok=PASS
      for p in "${pids[@]}"; do wait "$p" || ok=FAIL; done
      for i in 1 2 3 4 5 6 7 8; do say "eight $i: tokens wall_s = $(cat "$EV/eight-$i.txt")"; done
      sleep 2
      gate receipt-8conv "$ok" "8 concurrent greedy replies of 256 tokens; receipt records so far $(wc -l < "$TOK" 2> /dev/null || echo 0)" \
        "$EV"/eight-*.json "$TOK" ;;
    seed-t07)
      for s in 1 2 1b; do chat "$EV/seed-$s.json" "$COUNT" 64 0.7 "${s%b}" 20 > /dev/null; done
      ids() { jq -c '.usage.aien_token_ids' "$EV/seed-$1.json" 2> /dev/null; }
      if [ "$(ids 1)" = "$(ids 1b)" ] && [ "$(ids 1)" != "$(ids 2)" ] && [ -n "$(ids 1)" ] && [ "$(ids 1)" != null ]; then
        gate seed-t07 PASS "seed 1 repeats, seed 2 differs" "$EV"/seed-*.json
      else
        gate seed-t07 FAIL "seed1 $(ids 1 | cut -c1-60) / seed2 $(ids 2 | cut -c1-60) / seed1b $(ids 1b | cut -c1-60)" "$EV"/seed-*.json
      fi ;;
    check-tokens) # server stopped first: the checker reads a closed file
      stop_server
      grep -E 'spell over|per-token receipt' "$EV/server-tokens.log" > "$EV/spells.txt"
      check_tokens "$TOK" "$A" "$B" ;;
    stamps-1conv)
      # chip_ns runs from the first submit to the last completion marker read back on the host clock, so
      # host_overhead_ns = total_ns - chip_ns is a lower bound. The stamped launch timeline at one
      # conversation (the token arm's lone-row graph) gives chip_ns - dev_span = submit + poll per token;
      # the stamps' chip cost sits in both and cancels. Its first 128 greedy ids must equal s01's solo-1.
      STA=$EV/stamps.jsonl
      start_server "$EV/server-stamps.log" AIEN_Q36_DECODE_TIMELINE="$STA" \
        || { gate stamps-1conv NOT_RUN "stamped server did not come up" "$EV/server-stamps.log"; exit 3; }
      grep -E 'decode timeline' "$EV/server-stamps.log" | tee -a "$EV/window.log"
      ok=PASS
      r=$(chat "$EV/stamped.json" "$COUNT" 128 0 0 30) || ok=FAIL
      say "stamped: tokens wall_s = $r"
      sleep 2
      stop_server
      ns=$(wc -l < "$STA" 2> /dev/null || echo 0)
      [ "$ns" -gt 0 ] || ok=FAIL
      got=$(jq -c '.usage.aien_token_ids' "$EV/stamped.json" 2> /dev/null)
      ref=$(jq -c '.usage.aien_token_ids[0:128]' "$S01/solo-1.json" 2> /dev/null)
      if [ -z "$ref" ] || [ "$ref" = null ]; then idn="no s01 solo-1 ids to compare"
      elif [ "$got" = "$ref" ]; then idn="first 128 ids IDENTICAL to s01 solo-1"
      else idn="ids DIFFER from s01 solo-1"; ok=FAIL; fi
      gate stamps-1conv "$ok" "1 greedy reply of 128 tokens, stamped launch timeline; $ns records; $idn" \
        "$EV/stamped.json" "$STA" "$EV/server-stamps.log" "$S01/solo-1.json" ;;
    check-stamps) check_stamps "$STA" "$S01" ;;
    *) gate "$g" FAIL "unknown gate name in GATES" ;;
  esac
done
say "done in $(elapsed) s, $(left) s to the deadline"
