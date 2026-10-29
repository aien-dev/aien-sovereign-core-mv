#!/bin/sh
# TUR-R0 gate: build and run the oracle, then require each mutant to fail with its intended exit code.
# Usage: OMC=/path/to/seed/out/omc sh tests/turing-network/check.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
src="$here/tur_r0_oracle.om"
: "${OMC:?set OMC to a seed-born omc}"
b=$(mktemp -d); trap 'rm -rf "$b"' EXIT
fails=0

run() { # name source-file expected-rc expected-stdout
  if ! "$OMC" < "$2" > "$b/$1" 2> "$b/$1.err"; then echo "FAIL $1: compile: $(cat "$b/$1.err")"; fails=$((fails + 1)); return; fi
  chmod 700 "$b/$1"; out=$("$b/$1"); rc=$?
  if [ "$rc" = "$3" ] && [ "$out" = "$4" ]; then echo "PASS $1 (exit $rc)"; else echo "FAIL $1: exit $rc stdout '$out', wanted exit $3 stdout '$4'"; fails=$((fails + 1)); fi
}

mutant() { # name expected-rc sed-expression
  sed "$3" "$src" > "$b/$1.om"
  if cmp -s "$src" "$b/$1.om"; then echo "FAIL $1: mutation did not apply"; fails=$((fails + 1)); return; fi
  run "$1" "$b/$1.om" "$2" ""
}

echo "omc $(sha256sum "$OMC" | cut -c1-64)"
echo "src $(sha256sum "$src" | cut -c1-64)"
run oracle "$src" 0 "TUR_R0_ORACLE_PASS"
# expected provider balance changed 27 -> 28 (README negative control)
mutant assert-provider 14 's/^if ne get bal 1 27$/if ne get bal 1 28/'
# overdraft guard removed: reserving 101 out of 100 must now be caught
mutant no-overdraft-guard 1 '21,23d'
# single-reservation guard removed: a second reservation must now be caught
mutant no-phase-guard 6 '18,20d'
# job-id guard removed: settling against job 8 must now be caught
mutant no-jobid-guard 7 '35,37d'
# replay guard removed: a zero-amount second settlement must now be caught
mutant no-replay-guard 20 '32,34d'
# change-must-match guard removed: wrong refund must now be caught
mutant no-refund-guard 8 '46,48d'

[ "$fails" = 0 ] && echo "TUR_R0 gate: PASS" || echo "TUR_R0 gate: FAIL ($fails)"
[ "$fails" = 0 ]
