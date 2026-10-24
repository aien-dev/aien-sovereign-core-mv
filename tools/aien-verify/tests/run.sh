#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Drake Stapleton (AIEN / aien-dev)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# aien-verify tests: sha256 vectors, both committed fixtures, four mutation tests on copies.
# Usage: tests/run.sh   (set TMPDIR to a scratch folder; fixtures are never written to)
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
repo=$(cd "$here/../.." && pwd)
runs=$repo/docs/campaigns/whole-system-e2e/runs
FA=$runs/RUN-1-dry-20261010T133749Z
FB=$runs/RUN-1-dry-20261010T134202Z
# contract digest of aien-architecture main 3c584a49 (passed on the command line, as the verifier requires)
CS=8ba653848287bc77db253466f6d3ff37c70a42be55601ad47eb1f838a41d20f0
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
bin=$work/aien-verify
pass=0; fail=0
ok()  { pass=$((pass + 1)); echo "ok   $1"; }
bad() { fail=$((fail + 1)); echo "FAIL $1"; }
check() { # name, command...
  n=$1; shift
  if "$@" >/dev/null 2>&1; then ok "$n"; else bad "$n"; fi
}
tree_sum() { (cd "$1" && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1); }

before=$(tree_sum "$runs")
"$here/build.sh" "$bin" >/dev/null 2>&1 || { echo "build failed"; exit 1; }
rustc --edition 2021 --test "$here/src/main.rs" -o "$work/unit" 2>/dev/null && check "unit tests (sha256 vectors incl. empty and abc)" "$work/unit"
check "binary selftest (sha256 vectors)" "$bin" --selftest

rows_of() { grep -E '^(E[1-6]|CTRL-E[1-6]) ' ; }
expect() { # name, run dir, expected rows (here-doc via $3)
  out=$("$bin" "$2" --contract-sha256 $CS 2>/dev/null)
  got=$(printf '%s\n' "$out" | rows_of)
  if [ "$got" = "$3" ]; then ok "$1 rows"; else bad "$1 rows"; printf 'got:\n%s\n' "$got"; fi
}
expect "fixture A" "$FA" "E1 NOT_RUN
E2 NOT_RUN
E3 FAIL
E4 NOT_RUN
E5 NOT_RUN
E6 PASS
CTRL-E1 NOT_RUN
CTRL-E2 NOT_RUN
CTRL-E3 NOT_RUN
CTRL-E4 NOT_RUN
CTRL-E5 NOT_RUN
CTRL-E6 NOT_RUN"
expect "fixture B" "$FB" "E1 NOT_RUN
E2 NOT_RUN
E3 FAIL
E4 FAIL
E5 NOT_RUN
E6 PASS
CTRL-E1 NOT_RUN
CTRL-E2 NOT_RUN
CTRL-E3 NOT_RUN
CTRL-E4 NOT_RUN
CTRL-E5 NOT_RUN
CTRL-E6 NOT_RUN"

vid() { "$bin" "$1" --contract-sha256 $CS 2>/dev/null | sed -n 's/^verdict_id //p'; }
A1=$(vid "$FA"); A2=$(vid "$FA"); B1=$(vid "$FB"); B2=$(vid "$FB")
[ -n "$A1" ] && [ "$A1" = "$A2" ] && ok "fixture A verdict_id stable ($A1)" || bad "fixture A verdict_id stable"
[ -n "$B1" ] && [ "$B1" = "$B2" ] && ok "fixture B verdict_id stable ($B1)" || bad "fixture B verdict_id stable"
[ "$A1" = 728ae9152ef546ab4ccf1b7619fd22d21acb2914870197ac58e8843f9afaaf77 ] && ok "fixture A verdict_id pinned" || bad "fixture A verdict_id pinned"
[ "$B1" = d7d57162ac318ec05f2d97ad9530d9d3a571275409b040f65ccd0e391f6c9615 ] && ok "fixture B verdict_id pinned" || bad "fixture B verdict_id pinned"
for F in "$FA" "$FB"; do
  "$bin" "$F" --contract-sha256 $CS 2>/dev/null | grep -q '^chain_unbroken true$' && ok "$(basename $F) chain unbroken" || bad "$(basename $F) chain unbroken"
  "$bin" "$F" --contract-sha256 $CS >/dev/null 2>&1; [ $? -eq 1 ] && ok "$(basename $F) exit 1 (FAIL rows)" || bad "$(basename $F) exit 1"
done
"$bin" "$FB" --contract-sha256 $CS --objective-id 093e66ec92a6b428b49a0aa2a75678e9fdb155f423e1e7c71c5ae9fef11a60b1 2>/dev/null | grep -q '^chain_unbroken true$' && ok "objective id argument accepted" || bad "objective id argument accepted"
"$bin" "$FB" --contract-sha256 $CS --objective-id 00000000000000000000000000000000000000000000000000000000000000aa 2>/dev/null | grep -q 'OBJECTIVE_ID' && ok "wrong objective id named" || bad "wrong objective id named"
"$bin" "$FB" >/dev/null 2>&1; [ $? -eq 2 ] && ok "missing --contract-sha256 refused" || bad "missing --contract-sha256 refused"

# E5 with a peer: equal base id passes, unequal fails; both machines then print one final id
PASSID=$("$bin" "$FB" --contract-sha256 $CS --peer-verdict-id "$B1" 2>/dev/null | sed -n 's/^verdict_id //p')
"$bin" "$FB" --contract-sha256 $CS --peer-verdict-id "$B1" 2>/dev/null | grep -q '^E5 PASS$' && ok "E5 PASS with matching peer base id" || bad "E5 PASS with matching peer base id"
[ "$PASSID" != "$B1" ] && ok "E5 PASS changes the final id ($PASSID)" || bad "E5 changes final id"
"$bin" "$FB" --contract-sha256 $CS --peer-verdict-id "$A1" 2>/dev/null | grep -q '^E5 FAIL$' && ok "E5 FAIL with differing peer id" || bad "E5 FAIL with differing peer id"

# --write writes both files into a copy, never into the fixture
cp -r "$FB" "$work/w" && "$bin" "$work/w" --contract-sha256 $CS --write >/dev/null 2>&1
[ -s "$work/w/VERIFIER_VERDICT.txt" ] && [ -s "$work/w/VERIFIER_RECEIPT.json" ] && ok "--write wrote verdict and receipt" || bad "--write wrote verdict and receipt"
grep -q '"chain_unbroken": true' "$work/w/VERIFIER_RECEIPT.json" && grep -q '"verifier_sha256": "' "$work/w/VERIFIER_RECEIPT.json" && ok "receipt carries chain_unbroken and verifier_sha256" || bad "receipt fields"
"$bin" "$work/w" --contract-sha256 $CS 2>/dev/null | grep -q '^chain_unbroken true$' && ok "re-verify after --write still unbroken" || bad "re-verify after --write"

mutant() { rm -rf "$work/m"; cp -r "$1" "$work/m"; }
detect() { # name, expected reason text, expected row line or empty
  out=$("$bin" "$work/m" --contract-sha256 $CS 2>/dev/null); rc=$?
  if [ $rc -ne 0 ] && printf '%s\n' "$out" | grep -q -- "$2"; then ok "$1 detected ($2)"; else bad "$1 detected ($2)"; printf '%s\n' "$out" | grep '^# ' | head -5; fi
  if [ -n "${3:-}" ]; then printf '%s\n' "$out" | grep -q -- "$3" && ok "$1 row: $3" || bad "$1 row: $3"; fi
}
# M1: one byte changed in an output file
mutant "$FB"; printf 'X' | dd of="$work/m/artifacts/E3m-report.md" bs=1 seek=3 conv=notrunc 2>/dev/null
detect "M1 one byte changed in an output file" MANIFEST_DIGEST "# (supplementary) E3m FAIL: DIGEST_MISMATCH"
# M2: one chain link broken
mutant "$FB"; sed -i 's/"prev_receipt_sha256": "[0-9a-f]*"/"prev_receipt_sha256": "1111111111111111111111111111111111111111111111111111111111111111"/' "$work/m/chain/006-E3m.json"
detect "M2 chain link broken" LINK_BROKEN
# M3: one file dropped from SHA256SUMS
mutant "$FB"; grep -v 'artifacts/E3m-report.md' "$work/m/chain/SHA256SUMS" >"$work/ms" && cp "$work/ms" "$work/m/chain/SHA256SUMS"
detect "M3 file dropped from SHA256SUMS" MANIFEST_UNLISTED
# M4a: a status edited FAIL to PASS, links and manifest NOT repaired
mutant "$FB"; sed -i 's/"status": "FAIL"/"status": "PASS"/' "$work/m/chain/005-E3.json"
detect "M4a receipt status FAIL->PASS, unrepaired" LINK_BROKEN
# M4b: same edit with every link and manifest line repaired, so only the false status word remains
mutant "$FB"; sed -i 's/"status": "FAIL"/"status": "PASS"/' "$work/m/chain/005-E3.json"
prev=0000000000000000000000000000000000000000000000000000000000000000
for f in $(ls "$work/m/chain"/0*.json | LC_ALL=C sort); do
  sed -i "s/\"prev_receipt_sha256\": \"[0-9a-f]*\"/\"prev_receipt_sha256\": \"$prev\"/" "$f"
  prev=$(sha256sum "$f" | cut -d' ' -f1)
done
: >"$work/newsums"
while read -r h p; do
  if [ -f "$work/m/$p" ]; then printf '%s  %s\n' "$(sha256sum "$work/m/$p" | cut -d' ' -f1)" "$p" >>"$work/newsums"; else printf '%s  %s\n' "$h" "$p" >>"$work/newsums"; fi
done <"$work/m/chain/SHA256SUMS"
cp "$work/newsums" "$work/m/chain/SHA256SUMS"
out=$("$bin" "$work/m" --contract-sha256 $CS 2>/dev/null)
printf '%s\n' "$out" | grep -q '^chain_unbroken true$' && ok "M4b repaired chain is unbroken (only the lie remains)" || bad "M4b repaired chain unbroken"
printf '%s\n' "$out" | grep -q '^E3 FAIL$' && ok "M4b E3 stays FAIL without the facts" || bad "M4b E3 stays FAIL"
printf '%s\n' "$out" | grep -q 'status_disagreement: E3 receipt says PASS but the facts derive FAIL' && ok "M4b status_disagreement named" || bad "M4b status_disagreement named"
# M5: a file added to the folder but not the manifest; M6: a link target removed (sequence gap)
mutant "$FB"; echo extra >"$work/m/artifacts/extra.txt"
detect "M5 unlisted extra file" MANIFEST_UNLISTED
mutant "$FB"; rm "$work/m/chain/007-E4.json"
detect "M6 receipt file dropped" SEQUENCE_GAP

# CTRL-E5 (v1.3-ctrl-e5): judged from the receipt's facts. Built on a copy of fixture B with a receipt appended and the manifest rewritten.
mkE5() { # dir, fields json (merged over a good receipt)
  rm -rf "$1"; cp -r "$FB" "$1"; rm -f "$1/chain/SHA256SUMS"
  last=$(ls "$1"/chain/0*.json | LC_ALL=C sort | tail -1); num=$(basename "$last" | cut -c1-3); next=$(printf '%03d' $((1$num - 999)))
  jq --arg prev "$(sha256sum "$last" | cut -d' ' -f1)" --arg step CTRL-E5 --argjson f "$2" \
    '{objective_id, run_id, step:$step, status:"PASS", prev_receipt_sha256:$prev, evidence:"test"} + $f' "$last" >"$1/chain/$next-CTRL-E5.json"
  (cd "$1" && find . -type f ! -name SHA256SUMS ! -name VERIFIER_VERDICT.txt ! -name VERIFIER_RECEIPT.json -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >chain/SHA256SUMS)
}
GOOD="{\"mutated_output_rejected\":true,\"mutated_chain_unbroken\":false,\"original_verdict_id\":\"$B1\",\"mutated_verdict_id\":\"1111111111111111111111111111111111111111111111111111111111111111\",\"verifier_reason\":\"# E3 FAIL: DIGEST_MISMATCH: x\"}"
c5() { "$bin" "$1" --contract-sha256 $CS 2>/dev/null; }
mkE5 "$work/e5a" "$GOOD"
c5 "$work/e5a" | grep -q "^CTRL-E5 PASS$" && ok "CTRL-E5 PASS from a good receipt" || { bad "CTRL-E5 PASS from a good receipt"; c5 "$work/e5a" | grep "CTRL-E5"; }
c5 "$work/e5a" | grep -q '^chain_unbroken true$' && ok "CTRL-E5 fixture chain unbroken" || bad "CTRL-E5 fixture chain unbroken"
[ "$(c5 "$work/e5a" | sed -n 's/^verdict_id //p')" != "$B1" ] && ok "CTRL-E5 PASS changes the final id" || bad "CTRL-E5 PASS changes the final id"
mkE5 "$work/e5b" "$(printf '%s' "$GOOD" | jq -c '.mutated_output_rejected=false')"
c5 "$work/e5b" | grep -q '^CTRL-E5 FAIL$' && ok "CTRL-E5 FAIL when mutated_output_rejected is false" || bad "CTRL-E5 FAIL when rejected false"
mkE5 "$work/e5c" "$(printf '%s' "$GOOD" | jq -c '.original_verdict_id="2222222222222222222222222222222222222222222222222222222222222222"')"
c5 "$work/e5c" | grep -q '^CTRL-E5 FAIL$' && ok "CTRL-E5 FAIL when original_verdict_id is not the derived pre id" || bad "CTRL-E5 FAIL on wrong original id"
mkE5 "$work/e5d" "$(printf '%s' "$GOOD" | jq -c '.mutated_verdict_id=.original_verdict_id')"
c5 "$work/e5d" | grep -q '^CTRL-E5 FAIL$' && ok "CTRL-E5 FAIL when the verdict id did not change" || bad "CTRL-E5 FAIL on unchanged id"
mkE5 "$work/e5e" "$(printf '%s' "$GOOD" | jq -c '.mutated_chain_unbroken=true')"
c5 "$work/e5e" | grep -q '^CTRL-E5 FAIL$' && ok "CTRL-E5 FAIL when the mutated chain stayed unbroken" || bad "CTRL-E5 FAIL on unbroken mutated chain"
mkE5 "$work/e5f" "{}"
c5 "$work/e5f" | grep -q '^CTRL-E5 NOT_RUN$' && ok "CTRL-E5 NOT_RUN when the receipt lacks facts" || bad "CTRL-E5 NOT_RUN without facts"
c5 "$FB" | grep -q '^CTRL-E5 NOT_RUN$' && ok "CTRL-E5 NOT_RUN when no receipt" || bad "CTRL-E5 NOT_RUN absent"

# regression: the recorded RUN-1 (no facts-bearing CTRL-E5 receipt) still verifies to its recorded ids
R1=$runs/RUN-1-20261010T154724Z
if [ -d "$R1" ]; then
  C9=9e3d5f4e23d65ac17be8f2f4ca48355f0aaa486f67ff5a8a8044dcdd646d34e8
  out=$("$bin" "$R1" --contract-sha256 $C9 --peer-verdict-id a50fccb63363ef9224a19bd8bf4f249754abf5d88667ba0fd6ce93dfb8d63d80 2>/dev/null)
  printf '%s\n' "$out" | grep -q '^verdict_id 437467db4149b013738578c10ef68b7168dfb95feae8dea882de17a6118fc5fd$' && ok "RUN-1 verdict_id unchanged with the Spark base id as peer" || bad "RUN-1 verdict_id unchanged"
  printf '%s\n' "$out" | grep -q '^base_verdict_id a50fccb63363ef9224a19bd8bf4f249754abf5d88667ba0fd6ce93dfb8d63d80$' && ok "RUN-1 base_verdict_id unchanged" || bad "RUN-1 base_verdict_id unchanged"
fi


# VERIFIER-v1.1-e3m (sc#423): E3m D4 follows the receipt's d4_rule; recorded GB10 folders keep their verdict ids
C9=9e3d5f4e23d65ac17be8f2f4ca48355f0aaa486f67ff5a8a8044dcdd646d34e8
R2=$runs/RUN-2-GB10-20261011T030554Z
R3=$runs/RUN-3-GB10-20261011T045450Z
g9() { "$bin" "$1" --contract-sha256 $C9 2>/dev/null; }
if [ -d "$R2" ]; then
  out=$(g9 "$R2")
  printf '%s\n' "$out" | grep -q '^verdict_id af9e744cc3187400e00599159999f357f53c1df867556f62c8f71a552662800c$' && ok "RUN-2-GB10 verdict_id unchanged" || bad "RUN-2-GB10 verdict_id unchanged"
  printf '%s\n' "$out" | grep -q '^# (supplementary) E3m FAIL: .*D4 required name(s) missing: \["windmill"\]' && ok "RUN-2-GB10 E3m FAIL (no d4_rule: exact rule)" || bad "RUN-2-GB10 E3m exact rule"
fi
if [ -d "$R3" ]; then
  out=$(g9 "$R3")
  printf '%s\n' "$out" | grep -q '^verdict_id 9ca9b7af6f9255fc7247acae33492768252d5f2017b40c29c159e04fbca833d5$' && ok "RUN-3-GB10 verdict_id unchanged" || bad "RUN-3-GB10 verdict_id unchanged"
  printf '%s\n' "$out" | grep -q '^# (supplementary) E3m PASS' && ok "RUN-3-GB10 E3m PASS under its d4_rule" || bad "RUN-3-GB10 E3m PASS under its d4_rule"
  printf '%s\n' "$out" | grep -q 'status_disagreement: E3m' && bad "RUN-3-GB10 E3m disagreement gone" || ok "RUN-3-GB10 E3m disagreement gone"
  printf '%s\n' "$out" | grep -q 'status_disagreement: CTRL-E4 receipt says PASS but the facts derive NOT_RUN' && ok "RUN-3-GB10 CTRL-E4 disagreement kept" || bad "RUN-3-GB10 CTRL-E4 disagreement kept"
  relink() { # repair every link and manifest line of $work/m after an edit, so only the edited fact differs
    p=0000000000000000000000000000000000000000000000000000000000000000
    for f in $(ls "$work/m/chain"/0*.json | LC_ALL=C sort); do
      sed -i "s/\"prev_receipt_sha256\": \"[0-9a-f]*\"/\"prev_receipt_sha256\": \"$p\"/" "$f"; p=$(sha256sum "$f" | cut -d' ' -f1)
    done
    : >"$work/newsums"
    while read -r h q; do
      if [ -f "$work/m/$q" ]; then printf '%s  %s\n' "$(sha256sum "$work/m/$q" | cut -d' ' -f1)" "$q" >>"$work/newsums"; else printf '%s  %s\n' "$h" "$q" >>"$work/newsums"; fi
    done <"$work/m/chain/SHA256SUMS"
    cp "$work/newsums" "$work/m/chain/SHA256SUMS"
  }
  mutant "$R3"; jq -c 'del(.d4_rule)' "$work/m/chain/007-E3m.json" >"$work/e" && cp "$work/e" "$work/m/chain/007-E3m.json"; relink
  out=$(g9 "$work/m")
  printf '%s\n' "$out" | grep -q '^chain_unbroken true$' && printf '%s\n' "$out" | grep -q '^# (supplementary) E3m FAIL: .*windmill' && ok "RUN-3 copy without d4_rule: exact rule, E3m FAIL" || bad "RUN-3 copy without d4_rule"
  mutant "$R3"; jq -c '.d4_rule="case-insensitive substring"' "$work/m/chain/007-E3m.json" >"$work/e" && cp "$work/e" "$work/m/chain/007-E3m.json"; relink
  g9 "$work/m" | grep -q '^# (supplementary) E3m NOT_RUN: unknown d4_rule' && ok "RUN-3 copy with an unknown d4_rule: E3m NOT_RUN" || bad "RUN-3 copy unknown d4_rule"
fi
after=$(tree_sum "$runs")
[ "$before" = "$after" ] && ok "fixtures untouched" || bad "fixtures untouched"
echo "passed $pass, failed $fail"
[ $fail -eq 0 ]
