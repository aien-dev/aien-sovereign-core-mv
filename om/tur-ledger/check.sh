#!/usr/bin/env bash
# TUR ledger gate (Aienet-0). Builds om/tur-ledger/tur-ledger.om with the pinned omc, runs the
# suite against it (must PASS), then builds each mutant and runs the same suite (each must FAIL).
# Usage: OMC=/path/to/seed/out/omc bash om/tur-ledger/check.sh   (OMEGA_OMC is accepted too)
# Env: TUR_LEDGER_OUT=path copies the built binary; CRASH_N (200), MUTANT_CRASH_N, CRASH_SEED (7).
# Needs: sha256sum, OpenSSL 3, strace (syscall injection), flock(1), timeout(1). Host only; no network, no GPU.
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
src="$here/tur-ledger.om"
sha="$root/om/vendor/sha256.om"
OMC=${OMC:-${OMEGA_OMC:-}}
: "${OMC:?set OMC to the seed compiler (omega-mv seed/out/omc)}"
PIN=d6290c7663c2b4bafb59eb3fdf812476e977f4c144def71095c6248ef9b96b84
got=$(sha256sum < "$OMC" | cut -d' ' -f1)
[ "$got" = "$PIN" ] || { echo "FAIL omc sha256 $got, pinned $PIN"; exit 1; }
CRASH_N=${CRASH_N:-200}
b=$(mktemp -d); keys=$(mktemp -d /dev/shm/tur-ledger-test-keys.XXXXXX)
trap 'rm -rf "$b" "$keys"' EXIT
umask 077
# Disposable test-only oracle keys, never production credentials or implementation dependencies.
for role in issuer verifier stranger; do
  openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 -out "$keys/$role" || exit 1
done
pubhex() { openssl pkey -in "$keys/$1" -pubout -outform DER 2>/dev/null | od -An -v -tx1 | tr -d ' \n' | cut -c53-; }
issuer=$(pubhex issuer); verifier=$(pubhex verifier)

zero=$(printf '0%.0s' $(seq 64))
hexn() { printf '%064x' "$1"; }
J7=$(printf '7%.0s' $(seq 64)); J8=$(printf '8%.0s' $(seq 64))
RA=$(printf 'a%.0s' $(seq 64)); RB=$(printf 'b%.0s' $(seq 64))

build() { # out source
  { printf 'om2\n'; cat "$sha" "$root/om/vendor/ec.om" "$2"; } | "$OMC" > "$1" 2> "$1.err" || { echo "compile: $(cat "$1.err")"; return 1; }
  chmod 700 "$1"
}

# Trusted provisioning and signing are TEST ORACLES. The ledger never signs and never creates pins.
provision() {
  local d=$1 req=req
  [ "$d" = "$b/pf" ] && req=acct
  mkdir -p "$d"
  [ -f "$d/authority" ] || printf 'genesis tur-ledger-2 %s %s %s %s %s prov ver %s %s %s' "$zero" "$issuer" "$verifier" "$RA" "$req" "$RA" "$RB" "$J7" > "$d/authority"
}
canonical() { # config, unsigned body, optional reservation authorization digest
  cat "$1/authority"; printf '\n%s\n' "$2"
  [ -z "${3:-}" ] || printf '%s\n' "$3"
}
reservehash() {
  local d=$1 job=$2 line rec
  [ -f "$d/journal" ] || return 0
  line=$(grep " reserve $job " "$d/journal" | head -n 1)
  [ -n "$line" ] || return 0
  rec=${line#* }; rec=${rec#* }; rec=${rec% *}; rec=${rec% *}
  canonical "$d" "$rec" | sha256sum | cut -c1-64
}
sign() { # key, dir, unsigned body, reserve digest
  canonical "$2" "$3" "${4:-}" | openssl dgst -sha256 -sign "$keys/$1" | od -An -v -tx1 | tr -d ' \n'
}
prepare() { # d original-command... -> signed argv
  local d=$1; shift; provision "$d"
  argv=("$@")
  local rh= body isig vsig
  case "$1" in
    prefund) ;;
    reserve) argv+=("$RA") ;;
    settle) [ "$#" = 6 ] || return 0; argv+=("$J7" "$RB"); rh=$(reservehash "$d" "$2") ;;
    refund) rh=$(reservehash "$d" "$2") ;;
    *) return 0 ;;
  esac
  body="${argv[*]}"; isig=$(sign issuer "$d" "$body" "$rh"); argv+=("$isig")
  if [ "$1" = settle ]; then vsig=$(sign verifier "$d" "$body" "$rh"); argv+=("$vsig"); fi
}
invoke() { local d=$1; shift; prepare "$d" "$@"; "$T" --journal "$d" "${argv[@]}"; }

# ---- suite: every test prints PASS/FAIL; T is the binary, f counts failures ----
f=0
ok() { echo "PASS $1"; }
no() { echo "FAIL $1: $2"; f=$((f + 1)); }
x() { # dir args... -> sets o (stdout) e (stderr) r (rc)
  local d=$1; shift
  o=$(invoke "$d" "$@" 2> "$b/err"); r=$?; e=$(cat "$b/err")
}
want() { # name rc stdout dir args...
  local n=$1 wr=$2 wo=$3; shift 3
  x "$@"
  if [ "$r" = "$wr" ] && [ "$o" = "$wo" ]; then ok "$n"; else no "$n" "exit $r out '$o' err '$e', wanted exit $wr out '$wo'"; fi
}
nsettle() { grep -c " settle $1 " "$2/journal"; }

# Independent chain check with sha256sum: SEQ counts from 1, PREV is the previous HASH, HASH is the
# sha256 of the bytes before " HASH".
chaincheck() { # journal -> 0 when every line verifies
  local prev=$zero n=0 line body h p s
  while IFS= read -r line; do
    n=$((n + 1)); body=${line% *}; h=${line##* }
    s=${line%% *}; p=${line#* }; p=${p%% *}
    [ "$s" = "$n" ] && [ "$p" = "$prev" ] || return 1
    [ "$(printf '%s' "$body" | sha256sum | cut -c1-64)" = "$h" ] || return 1
    prev=$h
  done < "$1"
}
# forge: write a journal with a valid chain from record bodies on stdin (one per line).
forge() { # journal
  local prev=$zero n=0 body h
  : > "$1"
  while IFS= read -r rec; do
    n=$((n + 1)); body="$n $prev $rec"
    h=$(printf '%s' "$body" | sha256sum | cut -c1-64)
    printf '%s %s\n' "$body" "$h" >> "$1"; prev=$h
  done
}

t_oracle() { # tests/turing-network/tur_r0_oracle.om scenario through the CLI, one JOB per escrow
  local d=$b/oracle; rm -rf "$d"
  want oracle.prefund 0 "prefunded p-req req 100" "$d" prefund p-req req 100
  want oracle.overdraft-101 7 "" "$d" reserve "$J7" req prov ver 101
  want oracle.reserve-7-40 0 "reserved $J7 req prov ver 40" "$d" reserve "$J7" req prov ver 40
  want oracle.req-after-reserve 0 "balance req 60" "$d" balance req
  want oracle.second-reservation 9 "" "$d" reserve "$J7" req prov ver 10
  want oracle.settle-unknown-job 8 "" "$d" settle "$J8" "$RA" 27 3 10
  want oracle.change-mismatch 12 "" "$d" settle "$J7" "$RA" 27 4 10
  want oracle.settle 0 "settled $J7 $RA 27 3 10" "$d" settle "$J7" "$RA" 27 3 10
  want oracle.zero-replay 11 "" "$d" settle "$J7" "$RB" 0 0 0
  want oracle.requester 0 "balance req 70" "$d" balance req
  want oracle.provider 0 "balance prov 27" "$d" balance prov
  want oracle.verifier 0 "balance ver 3" "$d" balance ver
  want oracle.observer 0 "balance obs 0" "$d" balance obs
  x "$d" audit
  case "$o" in "audit ok records 4 "*" prefunded 100 balances 100 escrow 0") ok oracle.audit ;; *) no oracle.audit "exit $r '$o'" ;; esac
  if chaincheck "$d/journal"; then ok oracle.chain-sha256sum; else no oracle.chain-sha256sum "sha256sum disagrees"; fi
}

t_prefund() {
  local d=$b/pf; rm -rf "$d"
  want prefund.first 0 "prefunded p1 acct 5" "$d" prefund p1 acct 5
  want prefund.dup 0 "dup p1 acct 5" "$d" prefund p1 acct 5
  want prefund.conflict-amount 13 "" "$d" prefund p1 acct 6
  want prefund.conflict-account 16 "" "$d" prefund p1 other 5
  want prefund.balance 0 "balance acct 5" "$d" balance acct
  want usage.bad-job 2 "" "$d" reserve "$(printf 'A%.0s' $(seq 64))" acct p v 1
  want usage.leading-zero 2 "" "$d" prefund p2 acct 05
  want usage.zero-amount 2 "" "$d" prefund p2 acct 0
  want usage.arity 2 "" "$d" settle "$J7" "$RA" 1 1
  want usage.command 2 "" "$d" mint acct 1
  want usage.u64-overflow 2 "" "$d" prefund p2 acct 18446744073709551616
}

t_once() { # sequential exactly-once
  local d=$b/once; rm -rf "$d"
  x "$d" prefund p1 req 50; x "$d" reserve "$J7" req prov ver 20
  want once.settle 0 "settled $J7 $RA 15 4 1" "$d" settle "$J7" "$RA" 15 4 1
  cp "$d/journal" "$b/once.before"
  want once.same-receipt 0 "already-settled $J7 $RA 15 4 1" "$d" settle "$J7" "$RA" 15 4 1
  want once.other-receipt 11 "" "$d" settle "$J7" "$RB" 15 4 1
  want once.same-receipt-other-split 11 "" "$d" settle "$J7" "$RA" 16 3 1
  want once.refund-after-settle 10 "" "$d" refund "$J7"
  if cmp -s "$d/journal" "$b/once.before" && [ "$(nsettle "$J7" "$d")" = 1 ]; then ok once.journal-unchanged; else no once.journal-unchanged "journal changed or settle count $(nsettle "$J7" "$d")"; fi
}

t_concurrent() { # 20 concurrent settle processes: exactly one settlement
  local d=$b/conc i n0 n11 nalr
  for mode in distinct same; do
    rm -rf "$d"; x "$d" prefund p1 req 50; x "$d" reserve "$J7" req prov ver 20
    for i in $(seq 1 20); do
      if [ $mode = distinct ]; then rc=$(hexn "$i"); else rc=$RA; fi
      prepare "$d" settle "$J7" "$rc" 15 4 1
      ( "$T" --journal "$d" "${argv[@]}" > "$b/c.$i.out" 2>/dev/null; echo $? > "$b/c.$i.rc" ) &
    done
    wait
    n0=$(cat "$b"/c.*.rc | grep -cx 0); n11=$(cat "$b"/c.*.rc | grep -cx 11)
    nst=$(cat "$b"/c.*.out | grep -c '^settled '); nalr=$(cat "$b"/c.*.out | grep -c '^already-settled ')
    x "$d" audit; ar=$r
    if [ $mode = distinct ]; then
      [ "$n0" = 1 ] && [ "$n11" = 19 ] && [ "$nst" = 1 ]; c=$?
    else
      [ "$n0" = 20 ] && [ "$nst" = 1 ] && [ "$nalr" = 19 ]; c=$?
    fi
    if [ $c = 0 ] && [ "$(nsettle "$J7" "$d")" = 1 ] && [ "$ar" = 0 ]; then ok "concurrent.settle-$mode (settled $nst, already $nalr, refused $n11)"; else no "concurrent.settle-$mode" "exit0 $n0 refused11 $n11 settled $nst already $nalr journal-settles $(nsettle "$J7" "$d") audit $ar"; fi
    rm -f "$b"/c.*
  done
  rm -rf "$d"; x "$d" prefund p1 req 1
  for i in $(seq 1 20); do
    prepare "$d" reserve "$(hexn "$i")" req prov ver 1
    ( "$T" --journal "$d" "${argv[@]}" > /dev/null 2>&1; echo $? > "$b/c.$i.rc" ) &
  done
  wait
  n0=$(cat "$b"/c.*.rc | grep -cx 0); n7=$(cat "$b"/c.*.rc | grep -cx 7); x "$d" audit
  case "$r $o $n0 $n7" in "0 audit ok records 3 "*" jobs 1 open 1 "*" 1 19") ok "concurrent.double-debit (one reserve, nineteen insufficient-funds)" ;; *) no concurrent.double-debit "exit0 $n0 exit7 $n7 audit $r '$o'" ;; esac
  rm -f "$b"/c.*
}

t_lock() { # the whole command waits for flock(LOCK_EX) on the journal
  local d=$b/lock; rm -rf "$d"; x "$d" prefund p1 req 5
  flock "$d/journal" sleep 3 & lp=$!
  sleep 0.5
  timeout 1 "$T" --journal "$d" balance req > /dev/null 2>&1; r1=$?
  wait $lp
  x "$d" balance req
  if [ "$r1" = 124 ] && [ "$r" = 0 ]; then ok "lock.blocks-while-held (timeout exit $r1, then exit $r)"; else no lock.blocks-while-held "while held exit $r1 (want 124), after exit $r"; fi
}

t_fsync() { # the record is pwritten and fsynced before the reply reaches stdout
  local d=$b/fs; rm -rf "$d"; x "$d" prefund p1 req 5
  prepare "$d" reserve "$J7" req prov ver 1
  strace -f -qq -e trace=pwrite64,fsync,write -o "$b/st" "$T" --journal "$d" "${argv[@]}" > /dev/null 2>&1 || true
  # order of the journal pwrite, the fsync of the same fd, and the reply on fd 1
  local wl fl ol fd
  wl=$(grep -n 'pwrite64(' "$b/st" | tail -n 1); fd=$(printf '%s' "$wl" | sed -n 's/.*pwrite64(\([0-9]*\),.*/\1/p'); wl=${wl%%:*}
  fl=$(grep -n "fsync($fd) *= 0" "$b/st" | tail -n 1); fl=${fl%%:*}
  ol=$(grep -n 'write(1, "' "$b/st" | head -n 1); ol=${ol%%:*}
  cp "$b/st" "$b/strace.last"
  if [ -n "$wl" ] && [ -n "$fl" ] && [ -n "$ol" ] && [ "$wl" -lt "$fl" ] && [ "$fl" -lt "$ol" ]; then ok "fsync.before-ack (pwrite64 line $wl < fsync($fd) line $fl < write(1) line $ol)"; else no fsync.before-ack "pwrite '$wl' fsync '$fl' reply '$ol'"; fi
}

t_crash() { # kill -9 at random points over CRASH_N reserve/settle commands, retrying each one
  local c=$b/crash k=$b/ctrl i j op rc0 rc1 killed=0 done0=0 durable=0 lost=0 badaudit=0 badretry=0
  rm -rf "$c" "$k"; x "$c" prefund p0 req 1000000; x "$k" prefund p0 req 1000000
  RANDOM=${CRASH_SEED:-7}
  for i in $(seq 1 "$CRASH_N"); do
    j=$(( (i + 1) / 2 ))
    if [ $((i % 2)) = 1 ]; then op="reserve $(hexn "$j") req prov ver 10"; else op="settle $(hexn "$j") $(hexn $((j + 4096))) 6 3 1"; fi
    invoke "$k" $op > /dev/null 2>&1 || badretry=$((badretry + 1))
    prepare "$c" $op
    "$T" --journal "$c" "${argv[@]}" > /dev/null 2>&1 & p=$!
    sleep "0.00$((RANDOM % 8))$((RANDOM % 10))"
    kill -9 $p 2> /dev/null; { wait $p; } 2> /dev/null; rc0=$?
    if [ "$rc0" = 137 ]; then killed=$((killed + 1)); else done0=$((done0 + 1)); fi
    x "$c" audit; [ "$r" = 0 ] || badaudit=$((badaudit + 1))
    x "$c" $op; rc1=$r
    case "$rc1 $o" in
      "0 already-"*) [ "$rc0" = 137 ] && durable=$((durable + 1)) ;;
      "0 reserved "*|"0 settled "*) [ "$rc0" = 137 ] && lost=$((lost + 1)) ;;
      *) badretry=$((badretry + 1)) ;;
    esac
  done
  local same=1 a
  for a in req prov ver; do
    [ "$("$T" --journal "$c" balance $a)" = "$("$T" --journal "$k" balance $a)" ] || same=0
  done
  local dup; dup=$(grep ' settle ' "$c/journal" | cut -d' ' -f4 | sort | uniq -d | wc -l)
  x "$c" audit; local ca=$o; x "$k" audit
  local ident=no; cmp -s "$c/journal" "$k/journal" && ident=yes
  echo "crash: n $CRASH_N killed $killed (durable-before-kill $durable, not-yet-durable $lost) completed-before-kill $done0 journal-identical-to-control $ident"
  echo "crash: $ca"
  if [ "$badaudit" = 0 ] && [ "$badretry" = 0 ] && [ "$same" = 1 ] && [ "$dup" = 0 ] && [ "${ca#audit ok records }" != "$ca" ] && [ "${ca#* head * }" = "${o#* head * }" ]; then ok "crash.kill9-$CRASH_N (audit after every kill, no double settle, balances equal control)"; else no crash.kill9 "badaudit $badaudit badretry $badretry same $same dup-settles $dup audit '$ca' control '$o'"; fi
}

t_torn() { # a torn last line is truncated and recorded; the records before it stay
  local d=$b/torn; rm -rf "$d"; x "$d" prefund p1 req 9; x "$d" reserve "$J7" req prov ver 4
  cp "$d/journal" "$b/torn.good"; x "$d" balance req; local before=$o
  printf '4 %s settle %s' "$zero" "$J7" >> "$d/journal"
  local want; want=$(tail -c +$(( $(stat -c %s "$b/torn.good") + 1 )) "$d/journal" | sha256sum | cut -c1-64)
  local n; n=$(( $(stat -c %s "$d/journal") - $(stat -c %s "$b/torn.good") ))
  x "$d" balance req
  if [ "$r" = 0 ] && [ "$o" = "$before" ] && [ "$e" = "recovered torn-tail $n bytes sha256 $want" ] && cmp -s "$b/torn.good" <(head -c "$(stat -c %s "$b/torn.good")" "$d/journal") && tail -n 1 "$d/journal" | grep -q " recover $n $want " && chaincheck "$d/journal"; then ok "torn.unterminated ($e)"; else no torn.unterminated "exit $r out '$o' err '$e'"; fi
  # A newline-terminated bad record is corruption, never an automatic rollback.
  local last; last=$(tail -n 1 "$d/journal" | cut -d' ' -f1); local lh; lh=$(tail -n 1 "$d/journal"); lh=${lh##* }
  printf '%s %s refund %s %s\n' $((last + 1)) "$lh" "$J7" "$zero" >> "$d/journal"
  cp "$d/journal" "$b/torn.bad"
  raw_want corrupt.terminated-bad-hash 5 "$d" show "$J7"
  # The same final record cut short without its newline is an incomplete tail.
  local s keep; s=$(stat -c %s "$d/journal"); keep=$(head -n -1 "$d/journal" | wc -c)
  truncate -s $((s - 10)) "$d/journal"
  x "$d" audit
  if [ "$r" = 0 ] && [ "${e#recovered torn-tail $((s - 10 - keep)) bytes}" != "$e" ] && chaincheck "$d/journal"; then ok "torn.cut-last-record ($e)"; else no torn.cut-last-record "exit $r out '$o' err '$e'"; fi
}

t_corrupt() { # a bad line before the last refuses the ledger and changes nothing
  local d=$b/cor; rm -rf "$d"; x "$d" prefund p1 req 100; x "$d" reserve "$J7" req prov ver 40; x "$d" balance req
  cp "$d/journal" "$b/cor.good"
  sed -i '2s/ req 100 / req 900 /' "$d/journal"
  cmp -s "$d/journal" "$b/cor.good" && no corrupt.mutation "sed did not change the journal"
  cp "$d/journal" "$b/cor.bad"
  want corrupt.amount-edited 5 "" "$d" balance req
  if cmp -s "$d/journal" "$b/cor.bad"; then ok corrupt.journal-untouched; else no corrupt.journal-untouched "journal changed"; fi
  cp "$b/cor.good" "$d/journal"; sed -i '2d' "$d/journal"
  want corrupt.line-deleted 5 "" "$d" audit
  { sed -n 1p "$b/cor.good"; sed -n 3p "$b/cor.good"; sed -n 2p "$b/cor.good"; } > "$d/journal"
  want corrupt.lines-swapped 5 "" "$d" audit
  cp "$b/cor.good" "$d/journal"; sed -i '1s/genesis tur-ledger-2/genesis tur-ledger-1/' "$d/journal"
  want corrupt.genesis-edited 5 "" "$d" audit
}

t_conserve() { # valid signatures do not bypass escrow/conservation rules, including replay
  local d=$b/cons; rm -rf "$d"; provision "$d"
  x "$d" prefund p1 req 100; x "$d" reserve "$J7" req prov ver 40
  cp "$d/journal" "$b/cons.before"
  prepare "$d" settle "$J7" "$RA" 30 30 10
  local rec="${argv[*]}" prev n body h
  prev=$(tail -n 1 "$d/journal"); prev=${prev##* }; n=4
  body="$n $prev $rec"; h=$(printf '%s' "$body" | sha256sum | cut -c1-64)
  printf '%s %s\n' "$body" "$h" >> "$d/journal"
  if chaincheck "$d/journal"; then ok conserve.forged-chain-valid; else no conserve.forged-chain-valid "forge broke chain"; fi
  want conserve.payout-over-escrow 6 "" "$d" audit
  want conserve.refused-for-every-command 6 "" "$d" balance prov
}

raw_want() { # name rc d exact args (no oracle signing)
  local name=$1 expected=$2 d=$3; shift 3
  local before after rr
  before=$(sha256sum "$d/journal" 2>/dev/null || true)
  "$T" --journal "$d" "$@" > "$b/raw.out" 2> "$b/raw.err"; rr=$?
  after=$(sha256sum "$d/journal" 2>/dev/null || true)
  if [ "$rr" = "$expected" ] && [ "$before" = "$after" ]; then ok "$name (exit $rr, unchanged journal)"; else no "$name" "exit $rr expected $expected, before '$before' after '$after'"; fi
}
t_unsigned() {
  local d=$b/unsigned; rm -rf "$d"; provision "$d"
  "$T" --journal "$d" audit > /dev/null 2>&1
  raw_want auth.unsigned-prefund 2 "$d" prefund forged req 100
}
t_auth() {
  local d=$b/auth body rh sig
  t_unsigned
  rm -rf "$d"; x "$d" prefund p1 req 100; x "$d" reserve "$J7" req prov ver 40
  raw_want auth.unsigned-reserve 2 "$d" reserve "$J8" req prov ver 5 "$RA"
  raw_want auth.unsigned-settle 2 "$d" settle "$J7" "$RA" 27 3 10 "$J7" "$RB"
  raw_want auth.unsigned-refund 2 "$d" refund "$J7"
  prepare "$d" prefund p2 req 5
  body="prefund p2 req 5"; argv[4]=$(sign stranger "$d" "$body")
  raw_want auth.wrong-issuer-key 16 "$d" "${argv[@]}"
  prepare "$d" prefund p2 req 5; argv[3]=6
  raw_want auth.changed-amount 16 "$d" "${argv[@]}"
  prepare "$d" reserve "$J8" req prov ver 5; argv[6]=$RB
  raw_want auth.changed-quote 16 "$d" "${argv[@]}"
  prepare "$d" reserve "$J8" req thief ver 5
  raw_want auth.account-spoof 16 "$d" "${argv[@]}"
  prepare "$d" settle "$J7" "$RA" 27 3 10
  body="${argv[*]:0:8}"; rh=$(reservehash "$d" "$J7")
  argv[9]=$(sign stranger "$d" "$body" "$rh")
  raw_want auth.wrong-verifier-key 16 "$d" "${argv[@]}"
  prepare "$d" settle "$J7" "$RA" 27 3 10
  argv[8]=$(sign stranger "$d" "$body" "$rh")
  raw_want auth.unauthorized-settle 16 "$d" "${argv[@]}"
  prepare "$d" settle "$J7" "$RA" 27 3 10; argv[1]=$J8
  raw_want auth.changed-job 16 "$d" "${argv[@]}"
  prepare "$d" settle "$J7" "$RA" 27 3 10; argv[2]=$RB
  raw_want auth.forged-receipt-hash 16 "$d" "${argv[@]}"
  prepare "$d" settle "$J7" "$RA" 27 3 10; argv[7]=$RA
  raw_want auth.mutated-attestation-digest 16 "$d" "${argv[@]}"
  prepare "$d" settle "$J7" "$RA" 27 3 10; argv[6]=$RB
  raw_want auth.mutated-signed-job-digest 16 "$d" "${argv[@]}"
  prepare "$d" settle "$J7" "$RA" 27 3 10; argv[9]="${argv[8]}"
  raw_want auth.issuer-is-not-verifier 16 "$d" "${argv[@]}"
  # SIGKILL at stdout write entry is after journal pwrite + fsync and before acknowledgement.
  prepare "$d" settle "$J7" "$RA" 27 3 10
  strace -qq -e trace=pwrite64,fsync,write -e inject=write:signal=SIGKILL -o "$b/kill-after-fsync" "$T" --journal "$d" "${argv[@]}" > /dev/null 2>&1
  local killed=$?
  if [ "$killed" = 137 ] && grep -q 'fsync(.*) *= 0' "$b/kill-after-fsync"; then ok auth.kill-after-fsync-before-ack; else no auth.kill-after-fsync-before-ack "exit $killed, missing fsync"; fi
  want auth.lost-response-retry 0 "already-settled $J7 $RA 27 3 10" "$d" settle "$J7" "$RA" 27 3 10
  want auth.honest-provider 0 "balance prov 27" "$d" balance prov
  prepare "$d" settle "$J7" "$RA" 27 3 10
  body="settle $J7 $RA 27 3 10 $J7 $RA"
  argv[7]=$RA; argv[8]=$(sign issuer "$d" "$body" "$rh"); argv[9]=$(sign verifier "$d" "$body" "$rh")
  raw_want auth.changed-evidence-conflicts-even-when-signed 11 "$d" "${argv[@]}"
  cp "$d/journal" "$b/auth.good"
  { cat "$d/authority"; printf '\nprefund forged req 100\n'; } | forge "$d/journal"
  raw_want auth.unsigned-replay-last 5 "$d" audit
  prepare "$d" prefund p2 req 5; argv[4]=$(sign stranger "$d" "prefund p2 req 5")
  { cat "$d/authority"; printf '\n%s\n' "${argv[*]}"; } | forge "$d/journal"
  raw_want auth.wrong-key-replay-last 5 "$d" audit
  printf 'genesis tur-ledger-1\nprefund p1 req 100\n' | forge "$d/journal"
  raw_want auth.old-journal-refused 5 "$d" audit
  cp "$b/auth.good" "$d/journal"
  local authority; authority=$(cat "$d/authority")
  printf '%s' "${authority/$RA/$J8}" > "$d/authority"
  raw_want auth.changed-policy-pins 5 "$d" audit
}

# Exercise the production allocator/record path with a small, completely full
# key table. Only KCAP changes; crypto, replay, accounting and append are intact.
t_capacity() {
  local source=$1 d=$b/capacity T=$b/capacity-ledger id
  sed 's/^const KCAP ulong 65536$/const KCAP ulong 10/' "$source" > "$b/capacity.om"
  if ! build "$T" "$b/capacity.om"; then no capacity.build "compile failed"; return; fi
  echo "capacity-source $(sha256sum "$source" | cut -c1-64) KCAP=10"
  echo "capacity-bin $(sha256sum "$T" | cut -c1-64)"
  rm -rf "$d"
  want capacity.prefund 0 "prefunded p1 req 100" "$d" prefund p1 req 100
  want capacity.reserve-settle 0 "reserved $J7 req prov ver 40" "$d" reserve "$J7" req prov ver 40
  want capacity.reserve-refund 0 "reserved $J8 req prov ver 40" "$d" reserve "$J8" req prov ver 40
  # Three accounts, two jobs and five prefund IDs occupy all ten slots.
  for id in p2 p3 p4 p5; do
    want "capacity.fill-$id" 0 "prefunded $id req 1" "$d" prefund "$id" req 1
  done
  prepare "$d" prefund overflow req 1
  raw_want capacity.new-prefund-refused 14 "$d" "${argv[@]}"
  prepare "$d" reserve "$RB" req prov ver 1
  raw_want capacity.new-reserve-refused 14 "$d" "${argv[@]}"
  want capacity.settle 0 "settled $J7 $RA 27 3 10" "$d" settle "$J7" "$RA" 27 3 10
  want capacity.refund 0 "refunded $J8" "$d" refund "$J8"
  prepare "$d" settle "$J7" "$RA" 27 3 10
  raw_want capacity.settle-repeat 0 "$d" "${argv[@]}"
  prepare "$d" refund "$J8"
  raw_want capacity.refund-repeat 0 "$d" "${argv[@]}"
  want capacity.provider 0 "balance prov 27" "$d" balance prov
  want capacity.verifier 0 "balance ver 3" "$d" balance ver
  x "$d" audit
  if [ "$r" = 0 ] && [[ "$o" = *" open 0 settled 1 refunded 1 "* ]]; then
    ok capacity.audit-closed
  else no capacity.audit-closed "exit $r out '$o'"; fi
  d=$b/capacity-partial
  rm -rf "$d"
  want capacity.partial-prefund 0 "prefunded p1 req 100" "$d" prefund p1 req 100
  for id in p2 p3 p4 p5 p6 p7 p8; do
    want "capacity.partial-fill-$id" 0 "prefunded $id req 1" "$d" prefund "$id" req 1
  done
  # Only one slot remains, but a first reservation needs a job and two role accounts.
  prepare "$d" reserve "$J7" req prov ver 40
  raw_want capacity.partial-reserve-refused 14 "$d" "${argv[@]}"
  want capacity.last-slot-prefund 0 "prefunded p9 req 1" "$d" prefund p9 req 1
  prepare "$d" prefund overflow req 1
  raw_want capacity.partial-full-prefund-refused 14 "$d" "${argv[@]}"
}

t_audit_io() {
  local d=$b/audit-io attempt before before_size rr jfd fl ol
  rm -rf "$d"; x "$d" prefund base req 100
  before=$(sha256sum "$d/journal")
  before_size=$(stat -c %s "$d/journal")
  prepare "$d" prefund unsynced req 5
  for attempt in 1 2; do
    strace -qq -e trace=pwrite64,fsync,ftruncate,write -e inject=fsync:error=EIO -o "$b/io.$attempt.trace" "$T" --journal "$d" "${argv[@]}" > "$b/io.out" 2> "$b/io.err"; rr=$?
    if [ "$rr" = 3 ] && [ ! -s "$b/io.out" ] && [ "$(sha256sum "$d/journal")" = "$before" ] && [ "$(stat -c %s "$d/journal")" = "$before_size" ]; then
      ok "io.failed-sync-rollback-$attempt (exit 3, original journal, size $before_size)"
    else no "io.failed-sync-rollback-$attempt" "exit $rr out '$(cat "$b/io.out")', journal restored $([ "$(sha256sum "$d/journal")" = "$before" ] && echo yes || echo no)"; fi
  done
  x "$d" audit
  if [ "$r" = 0 ] && [ -z "$e" ]; then ok io.no-torn-tail-after-rollback; else no io.no-torn-tail-after-rollback "exit $r err '$e'"; fi
  prepare "$d" prefund unsynced req 5
  strace -qq -e trace=pwrite64,fsync,write -o "$b/io.retry.trace" "$T" --journal "$d" "${argv[@]}" > "$b/io.out" 2> "$b/io.err"; rr=$?
  if [ "$rr" = 0 ] && [ "$(cat "$b/io.out")" = 'prefunded unsynced req 5' ] && grep -q 'fsync(.*) *= 0' "$b/io.retry.trace"; then
    ok io.retry-writes-and-syncs
  else no io.retry-writes-and-syncs "exit $rr out '$(cat "$b/io.out")'"; fi
  # Each duplicate acknowledgement must cross a fresh durability barrier too.
  local op
  for op in prefund reserve settle refund; do
    case $op in
      prefund) prepare "$d" prefund base req 100 ;;
      reserve) x "$d" reserve "$J7" req prov ver 40; prepare "$d" reserve "$J7" req prov ver 40 ;;
      settle) x "$d" settle "$J7" "$RA" 27 3 10; prepare "$d" settle "$J7" "$RA" 27 3 10 ;;
      refund) x "$d" reserve "$J8" req prov ver 40; x "$d" refund "$J8"; prepare "$d" refund "$J8" ;;
    esac
    before=$(sha256sum "$d/journal")
    strace -qq -e trace=openat,pwrite64,fsync,write -e inject=fsync:error=EIO -o "$b/dup.$op.trace" "$T" --journal "$d" "${argv[@]}" > "$b/io.out" 2> "$b/io.err"; rr=$?
    jfd=$(sed -n 's/.*openat(.*\/journal",.*) *= \([0-9][0-9]*\)$/\1/p' "$b/dup.$op.trace" | head -n 1)
    if [ "$rr" = 3 ] && [ ! -s "$b/io.out" ] && [ "$(sha256sum "$d/journal")" = "$before" ] && [ -n "$jfd" ] && grep -q "fsync($jfd) *= -1 EIO" "$b/dup.$op.trace"; then
      ok "io.duplicate-$op-needs-sync"
    else no "io.duplicate-$op-needs-sync" "exit $rr out '$(cat "$b/io.out")'"; fi
    strace -qq -e trace=openat,pwrite64,fsync,write -o "$b/dup.$op.retry.trace" "$T" --journal "$d" "${argv[@]}" > "$b/io.out" 2> "$b/io.err"; rr=$?
    jfd=$(sed -n 's/.*openat(.*\/journal",.*) *= \([0-9][0-9]*\)$/\1/p' "$b/dup.$op.retry.trace" | head -n 1)
    fl=$(grep -n "fsync($jfd) *= 0" "$b/dup.$op.retry.trace" | tail -n 1); fl=${fl%%:*}
    ol=$(grep -n 'write(1, "' "$b/dup.$op.retry.trace" | head -n 1); ol=${ol%%:*}
    if [ "$rr" = 0 ] && [ -n "$jfd" ] && [ -n "$fl" ] && [ -n "$ol" ] && [ "$fl" -lt "$ol" ] && [ "$(sha256sum "$d/journal")" = "$before" ]; then
      ok "io.duplicate-$op-retry-synced"
    else no "io.duplicate-$op-retry-synced" "exit $rr, missing fsync or changed journal"; fi
  done
}

t_audit_corrupt() {
  local d=$b/audit-tail line body hash first
  rm -rf "$d"; x "$d" prefund base req 100; x "$d" reserve "$J7" req prov ver 40
  want corrupt.acknowledged-settle 0 "settled $J7 $RA 27 3 10" "$d" settle "$J7" "$RA" 27 3 10
  cp "$d/journal" "$b/audit-tail.good"
  sed '$s/ 27 3 10 / 28 3 10 /' "$b/audit-tail.good" > "$d/journal"
  cp "$d/journal" "$b/audit-tail.bad"
  raw_want corrupt.acknowledged-tail-audit 5 "$d" audit
  cp "$b/audit-tail.bad" "$d/journal"
  prepare "$d" refund "$J7"
  raw_want corrupt.acknowledged-tail-refund 5 "$d" "${argv[@]}"
  cp "$b/audit-tail.bad" "$d/journal"
  raw_want corrupt.acknowledged-tail-show 5 "$d" show "$J7"
  # Keep body/signatures valid but damage only the final chain hash.
  line=$(tail -n 1 "$b/audit-tail.good"); body=${line% *}; hash=${line##* }
  first=0; [ "${hash:0:1}" = 0 ] && first=1
  { head -n -1 "$b/audit-tail.good"; printf '%s %s%s\n' "$body" "$first" "${hash:1}"; } > "$d/journal"
  raw_want corrupt.signed-tail-bad-hash 5 "$d" audit
  cp "$b/audit-tail.good" "$d/journal"
  want corrupt.original-settlement-preserved 0 "job $J7 settled req prov ver 40 $RA 27 3 10" "$d" show "$J7"
}

t_audit_argv_phase() {
  local d=$b/audit-argv
  rm -rf "$d"; x "$d" prefund base req 100
  prepare "$d" prefund p2 req 5
  raw_want usage.fused-name-arguments 2 "$d" prefund 'p2 req' 5 "${argv[4]}"
  prepare "$d" prefund p3 req 5
  raw_want usage.fused-command-argument 2 "$d" 'prefund p3' req 5 "${argv[4]}"
  prepare "$d" reserve "$J7" req prov ver 40
  raw_want usage.fused-role-arguments 2 "$d" reserve "$J7" 'req prov' ver 40 "$RA" "${argv[7]}"
  d=$b/audit-phase
  rm -rf "$d"; x "$d" prefund base req 100
  want phase.reserve 0 "reserved $J7 req prov ver 40" "$d" reserve "$J7" req prov ver 40
  want phase.reserve-open-retry 0 "already-reserved $J7 req prov ver 40" "$d" reserve "$J7" req prov ver 40
  x "$d" settle "$J7" "$RA" 27 3 10
  prepare "$d" reserve "$J7" req prov ver 40
  raw_want phase.reserve-after-settle 10 "$d" "${argv[@]}"
  x "$d" reserve "$J8" req prov ver 40; x "$d" refund "$J8"
  prepare "$d" reserve "$J8" req prov ver 40
  raw_want phase.reserve-after-refund 10 "$d" "${argv[@]}"
}

t_audit() { t_audit_io; t_audit_corrupt; t_audit_argv_phase; }

suite() {
  f=0
  t_oracle; t_prefund; t_once; t_concurrent; t_lock; t_fsync; t_torn; t_corrupt; t_conserve; t_auth; t_audit; t_crash
  t_capacity "${1:-$src}"
  [ "$f" = 0 ]
}

echo "omc $(sha256sum "$OMC" | cut -c1-64)"
echo "src $(sha256sum "$src" | cut -c1-64) om/tur-ledger/tur-ledger.om"
echo "sha $(sha256sum "$sha" | cut -c1-64) om/vendor/sha256.om"
echo "ec $(sha256sum "$root/om/vendor/ec.om" | cut -c1-64) omega-mv@6999fa9e6dae4847c155ef4eacaece74f491d3ca"
echo "gate $(sha256sum "$0" | cut -c1-64)"
openssl version
gate=0
if [ -n "${TUR_LEDGER_AUTH_ONLY:-}" ]; then
  T=$TUR_LEDGER_AUTH_ONLY
  t_unsigned
  [ "$f" = 0 ]; exit $?
fi
if [ -n "${TUR_LEDGER_CAPACITY_SOURCE:-}" ]; then
  t_capacity "$TUR_LEDGER_CAPACITY_SOURCE"
  [ "$f" = 0 ]; exit $?
fi
if [ -n "${TUR_LEDGER_AUDIT_SOURCE:-}" ]; then
  T=$b/audit-ledger
  build "$T" "$TUR_LEDGER_AUDIT_SOURCE" || exit 1
  echo "audit-source $(sha256sum "$TUR_LEDGER_AUDIT_SOURCE" | cut -c1-64)"
  t_audit
  [ "$f" = 0 ]; exit $?
fi
T=$b/tur-ledger
if build "$T" "$src"; then
  echo "bin $(sha256sum "$T" | cut -c1-64)"
  [ -n "${TUR_LEDGER_OUT:-}" ] && cp "$T" "$TUR_LEDGER_OUT"
  if suite; then echo "SUITE PASS"; else echo "SUITE FAIL ($f)"; gate=1; fi
else
  echo "FAIL build"; gate=1
fi

mutant() { # name sed-expression: the suite must fail against the mutant
  local m=$b/$1.om
  sed "$2" "$src" > "$m"
  if cmp -s "$src" "$m"; then echo "FAIL mutant $1: mutation did not apply"; gate=1; return; fi
  T=$b/$1
  if ! build "$T" "$m" > "$b/$1.build"; then echo "FAIL mutant $1: $(cat "$b/$1.build")"; gate=1; return; fi
  suite "$m" > "$b/$1.log" 2>&1
  if [ "$f" != 0 ]; then echo "PASS mutant $1 killed ($f failing: $(grep '^FAIL' "$b/$1.log" | cut -d: -f1 | sed 's/^FAIL //' | tr '\n' ' '))"; else echo "FAIL mutant $1 survived"; gate=1; fi
}
CRASH_N=${MUTANT_CRASH_N:-$CRASH_N}
mutant no-fsync '/^sy = fsync fd$/,/^end$/d'
mutant no-lock '/^lk = flock fd 2$/,/^end$/d'
mutant no-exactly-once '/^if eq sph 2$/,/^end$/d'
mutant no-hash-chain '/^if not hsame h$/,/^end$/d;/^if not psame$/,/^end$/d'
mutant no-conservation '/^if not conserve$/,/^end$/d'
mutant no-escrow-rule '/^ret vrule get d3 k pp pv bk$/s/.*/ret 0/'
mutant no-torn-recovery '/^tb = sub get st SFSZ get st SGOOD$/s/.*/tb = ulong 0/'
mutant no-issuer-signature '/^if not authsig add b fields 3$/,/^end$/d'
mutant no-verifier-signature '/^if not authsig add add b fields 1 4$/,/^end$/d'
mutant no-owner-pins '/^if not and cfeq 6 add b 2 and cfeq 7 add b 3 cfeq 8 add b 4$/,/^end$/d'
mutant capacity-lockup '/^if ne wrec 0$/i\if gt add get st SKEYS 4 KCAP\ndie 14\nend'
mutant no-slot-preflight '/^if gt add get st SKEYS needed KCAP$/,/^end$/d'
mutant no-sync-rollback '/^rollback fd off$/d'
mutant no-duplicate-sync '/^if ne fsync int get st SFD 0$/,/^end$/d'
mutant terminated-tail-recovery '/^fn rend e ulong to int$/,/^put st SOVF 0$/s/^ret 5$/if eq e get st SFSZ\nret 0\nend\nret 5/'
mutant argv-whitespace '/^if le c 32$/,/^end$/d'
mutant reserve-closed-phase '/^if ne get d4 sub j 1 1$/,/^end$/d'

[ "$gate" = 0 ] && echo "TUR_LEDGER gate: PASS" || echo "TUR_LEDGER gate: FAIL"
[ "$gate" = 0 ]
