#!/usr/bin/env bash
# TUR ledger gate (Aienet-0). Builds om/tur-ledger/tur-ledger.om with the pinned omc, runs the
# suite against it (must PASS), then builds each mutant and runs the same suite (each must FAIL).
# Usage: OMC=/path/to/seed/out/omc bash om/tur-ledger/check.sh   (OMEGA_OMC is accepted too)
# Env: TUR_LEDGER_OUT=path copies the built binary; CRASH_N (200), MUTANT_CRASH_N, CRASH_SEED (7).
# Needs: sha256sum, strace, flock(1), timeout(1). Host only; no network, no GPU.
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
src="$here/tur-ledger.om"
sha="$root/om/vendor/sha256.om"
OMC=${OMC:-${OMEGA_OMC:-}}
: "${OMC:?set OMC to the seed compiler (omega-mv seed/out/omc)}"
PIN=2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f
got=$(sha256sum < "$OMC" | cut -d' ' -f1)
[ "$got" = "$PIN" ] || { echo "FAIL omc sha256 $got, pinned $PIN"; exit 1; }
CRASH_N=${CRASH_N:-200}
b=$(mktemp -d); trap 'rm -rf "$b"' EXIT

zero=$(printf '0%.0s' $(seq 64))
hexn() { printf '%064x' "$1"; }
J7=$(printf '7%.0s' $(seq 64)); J8=$(printf '8%.0s' $(seq 64))
RA=$(printf 'a%.0s' $(seq 64)); RB=$(printf 'b%.0s' $(seq 64))

build() { # out source
  cat "$sha" "$2" | "$OMC" > "$1" 2> "$1.err" || { echo "compile: $(cat "$1.err")"; return 1; }
  chmod 700 "$1"
}

# ---- suite: every test prints PASS/FAIL; T is the binary, f counts failures ----
f=0
ok() { echo "PASS $1"; }
no() { echo "FAIL $1: $2"; f=$((f + 1)); }
x() { # dir args... -> sets o (stdout) e (stderr) r (rc)
  local d=$1; shift
  o=$("$T" --journal "$d" "$@" 2> "$b/err"); r=$?; e=$(cat "$b/err")
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
  want prefund.conflict-account 13 "" "$d" prefund p1 other 5
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
      ( "$T" --journal "$d" settle "$J7" "$rc" 15 4 1 > "$b/c.$i.out" 2>/dev/null; echo $? > "$b/c.$i.rc" ) &
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
  rm -rf "$d"; x "$d" prefund p1 req 50
  for i in $(seq 1 20); do ( "$T" --journal "$d" reserve "$(hexn "$i")" req prov ver 1 > /dev/null 2>&1; echo $? > "$b/c.$i.rc" ) & done
  wait
  n0=$(cat "$b"/c.*.rc | grep -cx 0); x "$d" audit
  case "$r $o" in "0 audit ok records 22 "*" jobs 20 open 20 "*) ok "concurrent.reserve-20 ($n0 reserved, no lost update)" ;; *) no concurrent.reserve-20 "exit0 $n0 audit $r '$o'" ;; esac
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
  strace -f -qq -e trace=pwrite64,fsync,write -o "$b/st" "$T" --journal "$d" reserve "$J7" req prov ver 1 > /dev/null 2>&1 || true
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
    "$T" --journal "$k" $op > /dev/null 2>&1 || badretry=$((badretry + 1))
    "$T" --journal "$c" $op > /dev/null 2>&1 & p=$!
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
  # a complete last line whose HASH is wrong
  cp "$d/journal" "$b/torn.good2"
  local last; last=$(tail -n 1 "$d/journal" | cut -d' ' -f1); local lh; lh=$(tail -n 1 "$d/journal"); lh=${lh##* }
  printf '%s %s refund %s %s\n' $((last + 1)) "$lh" "$J7" "$zero" >> "$d/journal"
  x "$d" show "$J7"
  if [ "$r" = 0 ] && [ "$o" = "job $J7 reserved req prov ver 4" ] && [ "${e#recovered torn-tail }" != "$e" ] && chaincheck "$d/journal" && cmp -s "$b/torn.good2" <(head -c "$(stat -c %s "$b/torn.good2")" "$d/journal"); then ok "torn.bad-hash-last-line ($e)"; else no torn.bad-hash-last-line "exit $r out '$o' err '$e'"; fi
  # the last record cut short (no newline): the whole partial line is the torn tail
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
  cp "$b/cor.good" "$d/journal"; sed -i '1s/genesis tur-ledger-1/genesis tur-ledger-2/' "$d/journal"
  want corrupt.genesis-edited 5 "" "$d" audit
}

t_conserve() { # a journal with a valid chain whose settle pays out more than its escrow is refused
  local d=$b/cons; rm -rf "$d"; mkdir -p "$d"
  printf '%s\n' "genesis tur-ledger-1" "prefund p1 req 100" "reserve $J7 req prov ver 40" "settle $J7 $RA 27 3 10" | forge "$d/journal"
  want conserve.forged-valid-control 0 "balance prov 27" "$d" balance prov
  printf '%s\n' "genesis tur-ledger-1" "prefund p1 req 100" "reserve $J7 req prov ver 40" "settle $J7 $RA 30 30 10" | forge "$d/journal"
  if chaincheck "$d/journal"; then ok conserve.forged-chain-valid; else no conserve.forged-chain-valid "forge broke the chain"; fi
  want conserve.payout-over-escrow 6 "" "$d" audit
  want conserve.refused-for-every-command 6 "" "$d" balance prov
}

suite() {
  f=0
  t_oracle; t_prefund; t_once; t_concurrent; t_lock; t_fsync; t_torn; t_corrupt; t_conserve; t_crash
  [ "$f" = 0 ]
}

echo "omc $(sha256sum "$OMC" | cut -c1-64)"
echo "src $(sha256sum "$src" | cut -c1-64) om/tur-ledger/tur-ledger.om"
echo "sha $(sha256sum "$sha" | cut -c1-64) om/vendor/sha256.om"
gate=0
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
  suite > "$b/$1.log" 2>&1
  if [ "$f" != 0 ]; then echo "PASS mutant $1 killed ($f failing: $(grep '^FAIL' "$b/$1.log" | cut -d: -f1 | sed 's/^FAIL //' | tr '\n' ' '))"; else echo "FAIL mutant $1 survived"; gate=1; fi
}
CRASH_N=${MUTANT_CRASH_N:-$CRASH_N}
mutant no-fsync '/^sy = fsync fd$/,/^end$/d'
mutant no-lock '/^lk = flock fd 2$/,/^end$/d'
mutant no-exactly-once '/^if eq sph 2$/,/^end$/d'
mutant no-hash-chain '/^if not hsame h$/,/^end$/d;/^if not psame$/,/^end$/d'
mutant no-conservation '/^if not conserve$/,/^end$/d'
mutant no-escrow-rule '/^ret vrule get d3 k pp pv bk$/s/.*/ret 0/'
mutant no-torn-recovery '/^if and eq c 1 eq e get st SFSZ$/s/.*/if eq e 0/'

[ "$gate" = 0 ] && echo "TUR_LEDGER gate: PASS" || echo "TUR_LEDGER gate: FAIL"
[ "$gate" = 0 ]
