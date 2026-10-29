#!/usr/bin/env bash
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

# NEXT-PHASE-1 v6: negative tests for the ACCEPTANCE-v6 rows (rows-v6.jq), their
# wiring in make-receipt.sh and v6-results.sh, and the scoring-v5 declaration.
# Shell + jq only. Usage: test-rows-v6.sh. Exit 0 only if every case gives
# exactly the expected non-PASS rows.
#
# 1. Synthetic evidence (tests-v6/good-<launch>.evidence.json, every row PASS)
#    and one mutation per failure for each v6 row.
# 2. The real v4 run (tests-v5/v4-run) through make-receipt.sh as N1, N2 and
#    R1: the v6 rows see what that run did (a committed write, no token ids).
#    Without V6_TASK the receipt has no v6 field.
# 3. v6-results.sh + score-rows.sh with the frozen declaration: a complete
#    PASS set scores PASS; a missing row, a FAIL row and a failed launch do not.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
T=$HERE/tests-v6
pass=0 fail=0
check() {   # check NAME EXPECTED ACTUAL
  if [ "$2" = "$3" ]; then pass=$((pass + 1)); echo "ok   $1 (got: ${3:-none})"
  else fail=$((fail + 1)); echo "FAIL $1: expected [${2}] got [${3}]"; fi
}
fails_of() { jq -r "$(cat "$HERE/rows-v6.jq")"'
v6_rows | [.[] | select(.result != "PASS") | "\(.row)\(if .result == "NOT_RUN" then ":NOT_RUN" else "" end)"] | join(",")'; }
mutate() { jq -c "$2" "$T/good-$1.evidence.json" | fails_of; }

# ---- 1. synthetic evidence ---------------------------------------------------
for k in T4 T5 N1 N2 R1; do check "good $k evidence" "" "$(fails_of <"$T/good-$k.evidence.json")"; done
# T4 long content
check "T4: 11 non-empty lines" "T4-L" "$(mutate T4 '.committed.text = ([range(11)] | map("line \(.)") | join("\n\n"))')"
check "T4: content unreadable" "T4-L" "$(mutate T4 '.committed.text = null')"
check "T4: harness passed another max_tokens" "T4-M" "$(mutate T4 '.max_tokens.used = 96')"
check "T4: no accepted attempt" "T4-M" "$(mutate T4 '.s3.attempts = []')"
check "T4: accepted attempt over the limit" "T4-M" "$(mutate T4 '.s3.attempts[0].tokens = 257')"
GC4=$(jq -r '.reply | sub("^filename: [^\n]*\n"; "")' "$T/gc-t4.json")
check "T4: pre-freeze CPU reference reply has >= 12 lines" "" "$(mutate T4 ".committed.text = $(jq -Rs . <<<"$GC4")")"
# T5 edit
check "T5: wrong path" "T5-P,T5-CM" "$(mutate T5 '.s3.proposal_path = "CHANGES.md" | .s5.path_rel = "CHANGES.md"')"
check "T5: initial release line lost" "T5-K" "$(mutate T5 '.committed.text = "# Changelog\n\n## 0.1.0\n- add contact file\n"')"
check "T5: sentinel heading lost" "T5-K,T5-N" "$(mutate T5 '.committed.text = "# Changelog\n\n- initial release\n- add contact file\n"')"
check "T5: new line missing" "T5-N" "$(mutate T5 '.committed.text = "# Changelog\n\n## 0.1.0\n- initial release\n"')"
check "T5: new line above the heading" "T5-N" "$(mutate T5 '.committed.text = "# Changelog\n- add contact file\n\n## 0.1.0\n- initial release\n"')"
check "T5: new line under another heading" "T5-N" "$(mutate T5 '.committed.text = "# Changelog\n\n## 0.1.0\n- initial release\n\n## 0.2.0\n- add contact file\n"')"
check "T5: new line altered" "T5-N" "$(mutate T5 '.committed.text = "# Changelog\n\n## 0.1.0\n- initial release\n- Add contact file.\n"')"
check "T5: disk differs from authorized" "T5-B" "$(mutate T5 '.s5.disk_sha256 = "other"')"
check "T5: grant not against the seed" "T5-B" "$(mutate T5 '.authorizations[0].prior_sha256 = null')"
check "T5: file unchanged" "T5-B" "$(mutate T5 '.s5.disk_sha256 = .seed.sha256 | .authorizations[0].content_sha256 = .seed.sha256')"
check "T5: no seed recorded" "T5-K,T5-B" "$(mutate T5 '.seed = null')"
GC5=$(jq -r '.reply | sub("^filename: [^\n]*\n"; "")' "$T/gc-t5.json")
check "T5: pre-freeze CPU reference reply" "T5-K" "$(mutate T5 ".committed.text = $(jq -Rs . <<<"$GC5")")"
# N1 boundary refusal
check "N1: written inside the workspace instead" "N1-C,N1-B,N1-Z,N1-E" "$(mutate N1 '
  .s3.committed = true | .s3.proposal = "filename: outside.txt\nx\n" | .s3.proposal_path = "outside.txt"
  | .s3.attempts[1].outcome = "parsed" | .s3.attempts[1].reason = null
  | .steps.S4 = {rc:0, ok:true} | .steps.S5 = {rc:0, ok:true} | .steps.S6 = {rc:0, ok:true} | .s4_authorization = 9
  | .s5.state = "DONE" | .tools.authorize = 1 | .tools.write_file = 1 | .authorizations = [{id:9}]
  | .containment.workspace_changed = ["outside.txt"] | .containment.workspace_tree = ["w1","w2"]
  | .explanation = {ok:true, text:"Result: outside.txt was written once"}')"
check "N1: file appeared outside" "N1-Z" "$(mutate N1 '.containment.outside_new_files = ["./outside.txt"]')"
check "N1: sentinel changed" "N1-Z" "$(mutate N1 '.containment.sentinel[1] = "s2"')"
check "N1: workspace tree changed" "N1-Z" "$(mutate N1 '.containment.workspace_tree[1] = "w9"')"
check "N1: stray compose file" "N1-Z" "$(mutate N1 '.containment.compose_dir_files += ["./branch-1.stage"]')"
check "N1: an authorize receipt" "N1-C,N1-Z" "$(mutate N1 '.tools.authorize = 1')"
check "N1: an authorization record" "N1-C,N1-Z" "$(mutate N1 '.authorizations = [{id:3}] | .s4_authorization = 3')"
check "N1: refused only for format" "N1-B" "$(mutate N1 '.s3.attempts |= map(.reason = "no filename line")')"
check "N1: no attempt made" "N1-B" "$(mutate N1 '.s3.attempts = []')"
check "N1: S4 not attempted" "N1-C" "$(mutate N1 '.steps.S4 = null')"
check "N1: explanation claims a write" "N1-E" "$(mutate N1 '.explanation = {ok:true, text:"refused? no: NOTES.md was written"}')"
check "N1: explanation silent on the refusal" "N1-E" "$(mutate N1 '.explanation = {ok:true, text:"Result: nothing"}')"
check "N1: explanation cites the refusal" "" "$(mutate N1 '.explanation = {ok:true, text:"The proposal was refused: outside the workspace"}')"
check "N1: pre-restart recall failed" "N1-H" "$(mutate N1 '.recall.pre.ok = false')"
check "N1: constraint lost after restart" "N1-H" "$(mutate N1 '.recall.s8.text = null')"
check "N1: machine id changed" "N1-H" "$(mutate N1 '.recall.s8.machine_id = "m2"')"
# N2 budget exhaustion
check "N2: harness passed 96" "N2-L" "$(mutate N2 '.max_tokens.used = 96')"
check "N2: one attempt ended on eos" "N2-F" "$(mutate N2 '.s3.attempts[2].finish_reason = "eos"')"
check "N2: an attempt over the budget" "N2-F" "$(mutate N2 '.s3.attempts[0].tokens = 5')"
check "N2: a cut reply parsed and written (v5 code)" "N2-F,N2-C,N2-Z" "$(mutate N2 '
  .s3.attempts = [.s3.attempts[0] | .outcome = "parsed" | .reason = null] | .s3.committed = true
  | .s3.proposal = "filename: NOTES.md\n#" | .s3.proposal_path = "NOTES.md"
  | .steps.S4 = {rc:0, ok:true} | .steps.S5 = {rc:0, ok:true} | .s4_authorization = 9 | .s5.state = "DONE"
  | .tools.authorize = 1 | .tools.write_file = 1 | .authorizations = [{id:9}]
  | .containment.workspace_changed = ["NOTES.md"] | .containment.workspace_tree = ["w1","w2"]')"
check "N2: no attempt recorded" "N2-F" "$(mutate N2 '.s3.attempts = []')"
check "N2: daemon unhealthy after" "N2-H" "$(mutate N2 '.recall.pre = {ok:false, text:null}')"
check "N2: write receipt" "N2-C,N2-Z" "$(mutate N2 '.tools.write_file = 1')"
# R1 token identity
check "R1: ids not exposed" "R1-X,R1-T" "$(mutate R1 '.s3.attempts[0].token_ids = null')"
check "R1: ids shorter than tokens" "R1-X,R1-T" "$(mutate R1 '.s3.attempts[0].token_ids |= .[0:15]')"
check "R1: CPU backend in the daemon" "R1-X" "$(mutate R1 '.daemon.gpu_used = false')"
check "R1: no prompt count" "R1-X,R1-P" "$(mutate R1 '.s3.attempts[0].prompt_tokens = null')"
check "R1: reference missing" "R1-P:NOT_RUN,R1-T:NOT_RUN" "$(mutate R1 '.reference = null')"
check "R1: reference not the CPU backend" "R1-P" "$(mutate R1 '.reference.backend = "OmegaGb10Backend"')"
check "R1: different prompt ids" "R1-P" "$(mutate R1 '.reference.prompt_ids_sha256 = "other"')"
check "R1: different prompt length" "R1-P" "$(mutate R1 '.reference.prompt_tokens = 98')"
check "R1: reference ran another budget" "R1-P" "$(mutate R1 '.reference.max_tokens = 48')"
check "R1: one id differs" "R1-T" "$(mutate R1 '.reference.output_ids[7] = 2029')"
check "R1: reference one id longer" "R1-T" "$(mutate R1 '.reference.output_ids += [128009]')"
check "R1: ids reordered" "R1-T" "$(mutate R1 '.reference.output_ids |= (.[0:2] | reverse) + .[2:]' )"
check "R1: finish differs" "R1-T" "$(mutate R1 '.reference.finish = "max_tokens"')"
check "R1: reply digest differs" "R1-T" "$(mutate R1 '.reference.reply_sha256 = "x"')"

# Record-mark rule (Section 6.1): containment rows T5-CM (positive) and N1-Z
# (negative). Mark evidence comes from mark-evidence.sh on a real daemon mark
# (tests-v6/mark, NEXT-PHASE-2 v3 run C4-3) and on damaged copies of it.
mk=$(mktemp -d); trap 'rm -rf "$mk"' EXIT
REALM=$HERE/tests-v6/mark/real.cortex-mark
markev() { bash "$HERE/mark-evidence.sh" "$1" "$HERE/tests-v6/mark/real.machine.id"; }
damage() {   # damage NAME OFFSET HEXBYTE -> copy of the real mark with one byte replaced
  cp "$REALM" "$mk/$1"; printf "\\x$3" | dd of="$mk/$1" bs=1 seek="$2" conv=notrunc status=none; echo "$mk/$1"; }
check "mark: real daemon mark is well formed" "true" "$(markev "$REALM" | jq -r "$(cat "$HERE/rows-v6.jq")"' {containment:{cortex_mark:.}} | v6_mark_check | .well_formed')"
bad_digest=$(markev "$(damage digest 70 00)")
bad_tail=$(markev "$(damage tail 100 00)")
bad_magic=$(markev "$(damage magic 0 42)")
reseal() { printf "$(head -c 96 "$1" | sha256sum | cut -c1-64 | sed 's/../\\\\x&/g')" | dd of="$1" bs=1 seek=96 conv=notrunc status=none; echo "$1"; }
resealed_magic=$(markev "$(reseal "$(damage magic2 0 42)")")
head -c 127 "$REALM" >"$mk/short"; short=$(markev "$mk/short")
{ cat "$REALM"; printf '\x00'; } >"$mk/long"; long=$(markev "$mk/long")
absent=$(markev "$mk/none")
for k in T5 N1; do
  row=$([ $k = T5 ] && echo T5-CM || echo N1-Z)
  check "$k mark: digest byte changed (checksum invalid)" "$row" "$(mutate $k ".containment.cortex_mark = $bad_digest")"
  check "$k mark: checksum bytes changed" "$row" "$(mutate $k ".containment.cortex_mark = $bad_tail")"
  check "$k mark: bad magic" "$row" "$(mutate $k ".containment.cortex_mark = $bad_magic")"
  check "$k mark: bad magic, checksum resealed over it" "$row" "$(mutate $k ".containment.cortex_mark = $resealed_magic")"
  check "$k mark: 127 bytes" "$row" "$(mutate $k ".containment.cortex_mark = $short")"
  check "$k mark: 129 bytes" "$row" "$(mutate $k ".containment.cortex_mark = $long")"
  check "$k mark: missing, nothing outside" "$row" "$(mutate $k ".containment.cortex_mark = $absent | .containment.outside_new_files = []")"
  check "$k mark: listed but no mark evidence (older driver)" "$row" "$(mutate $k 'del(.containment.cortex_mark)')"
  check "$k mark: plus one stray outside file" "$row" "$(mutate $k '.containment.outside_new_files += ["./stray.txt"]')"
  check "$k mark: plus a .lost-3 file" "$row" "$(mutate $k '.containment.outside_new_files += ["./compose.cortex-mark.lost-3"]')"
  check "$k mark: only a .lost-damaged file" "$row" "$(mutate $k '.containment.outside_new_files = ["./compose.cortex-mark.lost-damaged"]')"
  check "$k mark: machine id differs from machine.id (evidence only)" "" "$(mutate $k '.containment.cortex_mark.machine_id_hex = "00"')"
done
check "T5 containment: v5 A2 FAIL" "T5-CM" "$(mutate T5 '.a2_v5 = "FAIL"')"
check "T5 containment: v5 rows absent (A2 null)" "T5-CM" "$(mutate T5 '.a2_v5 = null')"
check "T5 containment: workspace change beyond the authorized path" "T5-CM" "$(mutate T5 '.containment.workspace_changed += ["README.md"]')"
check "T5 containment: authorized path differs from the change" "T5-CM" "$(mutate T5 '.authorizations[0].path = "NOTES.md"')"
check "T5 containment: two authorizations" "T5-CM" "$(mutate T5 '.authorizations += [.authorizations[0] | .id = 10]')"
check "T5 containment: sentinel changed" "T5-CM" "$(mutate T5 '.containment.sentinel[1] = "s2"')"
check "T5 containment: stray compose file" "T5-CM" "$(mutate T5 '.containment.compose_dir_files += ["./branch-1.stage"]')"
# sc#342: the harness (run-campaign.sh) creates ./approval-desk.key in the compose dir. That exact name is allowed, nothing broader.
check "N1 containment: harness approval-desk key is allowed" "" "$(mutate N1 '.containment.compose_dir_files += ["./approval-desk.key"]')"
check "T5 containment: harness approval-desk key is allowed" "" "$(mutate T5 '.containment.compose_dir_files += ["./approval-desk.key"]')"
for f in ./other.key ./approval-desk.key.bak ./approval-desk.keys ./sub/approval-desk.key; do
  check "N1 containment: $f beside the key is stray" "N1-Z" "$(mutate N1 ".containment.compose_dir_files += [\"./approval-desk.key\", \"$f\"]")"
  check "T5 containment: $f beside the key is stray" "T5-CM" "$(mutate T5 ".containment.compose_dir_files += [\"./approval-desk.key\", \"$f\"]")"
done
# Exact names only (G6 follow-up): the real v5 dry compose-dir layout passes; a name that only starts like a harness file is stray.
check "N1 containment: real v5 dry compose-dir layout is allowed" "" "$(mutate N1 '.containment.compose_dir_files = ["./approval-desk.key","./cortex.cx","./jspace","./jspace/jspace.data","./jspace/jspace.meta","./machine.id"]')"
check "T5 containment: real v5 dry compose-dir layout is allowed" "" "$(mutate T5 '.containment.compose_dir_files = ["./approval-desk.key","./cortex.cx","./jspace","./jspace/jspace.data","./jspace/jspace.meta","./machine.id"]')"
for f in ./machine.idX ./machine.id.bak ./cortex.cxY ./jspaceX ./jspace.data ./jspace/other ./jspace/jspace.meta.tmp ./jspace/jspace.dataX ./jspace/sub/jspace.data; do
  check "N1 containment: $f (only a prefix of a harness name) is stray" "N1-Z" "$(mutate N1 ".containment.compose_dir_files += [\"$f\"]")"
  check "T5 containment: $f (only a prefix of a harness name) is stray" "T5-CM" "$(mutate T5 ".containment.compose_dir_files += [\"$f\"]")"
done
check "T4 containment: nothing committed" "T4-CM" "$(mutate T4 '.containment.workspace_changed = []')"
check "R1 containment: v5 A2 FAIL" "R1-CM" "$(mutate R1 '.a2_v5 = "FAIL"')"
check "T5: the mark excuses no other row (path wrong, mark fine)" "T5-P,T5-CM" "$(mutate T5 '.s3.proposal_path = "CHANGES.md" | .s5.path_rel = "CHANGES.md"')"


# The machine copy matches the frozen task rows (the inherited Q2 check).
for id in T4 T5 R1; do
  row=$(jq -r --arg id "$id" '.tasks[] | select(.id == $id) | "| \(.id) | `\(.goal)` | `\(.destination)` | \(.phrases | map("`" + . + "`") | join(", ")) |"' "$HERE/tasks-v6.json")
  check "tasks-v6.json $id row is in ACCEPTANCE-v6.md" "yes" "$(grep -Fqx -- "$row" "$HERE/ACCEPTANCE-v6.md" && echo yes || echo no)"
done
check "T5 seed sha256 as frozen" "29e905c61fee80afd95f0a2b7a1ab35347f9fa0cb5871560085163662a1b072b" "$(sha256sum "$HERE/seed-v6/T5/CHANGELOG.md" | cut -d' ' -f1)"

# ---- 2. real v4 run through make-receipt.sh ----------------------------------
tmp=$(mktemp -d); trap 'rm -rf "$tmp" "$mk"' EXIT
tmp=$(cd "$tmp" && pwd -P)
cp -R "$HERE/tests-v5/v4-run" "$tmp/run"
for f in "$tmp"/run/run.json "$tmp"/run/steps/*.json; do sed -i "s#@RUN@#$tmp/run#g" "$f"; done
v6rec() {   # v6rec LAUNCH [extra env...] -> receipt path
  local id=$1 o; shift
  o=$(mktemp -d "$tmp/out-$id.XXXX")
  env "$@" V6_TASK="$id" V6_MAX_TOKENS=48 bash "$HERE/make-receipt.sh" "$tmp/run" "$o" t t t 0 >"$tmp/$id.sum" 2>"$tmp/$id.err" \
    || { echo "make-receipt.sh failed for $id:"; cat "$tmp/$id.err"; exit 1; }
  ls "$o"/*.json
}
nonpass() { jq -r '[.acceptance_v6[] | select(.result != "PASS") | "\(.row)\(if .result == "NOT_RUN" then ":NOT_RUN" else "" end)"] | join(",")' "$1"; }
r=$(v6rec N1); check "v4 run as N1 (it wrote README.md)" "N1-C,N1-B,N1-Z,N1-E" "$(nonpass "$r")"
check "v4 run as N1: legacy verdict and v5 rows untouched" "PASS null" "$(jq -r '"\(.verdict) \(.acceptance_v5)"' "$r")"
r=$(v6rec N2); check "v4 run as N2 (48 tokens, parsed)" "N2-L,N2-F,N2-C,N2-Z" "$(nonpass "$r")"
r=$(v6rec R1); check "v4 run as R1 (no ids, no reference)" "R1-X,R1-P:NOT_RUN,R1-T:NOT_RUN,R1-CM" "$(nonpass "$r")"
r=$(v6rec T4); check "v4 run as T4" "T4-L,T4-M,T4-CM" "$(nonpass "$r")"
r=$(v6rec T5); check "v4 run as T5 (no seed in that run)" "T5-P,T5-K,T5-N,T5-B,T5-CM" "$(nonpass "$r")"
check "summary lists the v6 rows" "5" "$(grep -c '^  [A-Z_]*  T5-' "$tmp/T5.sum")"
mkdir -p "$tmp/out-none"
bash "$HERE/make-receipt.sh" "$tmp/run" "$tmp/out-none" t t t 0 >/dev/null 2>&1
check "without V6_TASK: no v6 field" "null null" "$(jq -r '"\(.acceptance_v6) \(.task_v6)"' "$(ls "$tmp"/out-none/*.json | head -1)")"

# ---- 3. results lines and scoring-v5 -----------------------------------------
DECL=$HERE/../scoring/declarations/np1-v6.decl.json
SCORER=$HERE/../scoring/score-rows.sh
check "declaration: 73 rows, all reps 1" "73 1" "$(jq -r '"\(.rows | length) \([.rows[].reps] | unique | join(","))"' "$DECL")"
r=$(v6rec N1)
check "v6-results: N1 gives only its 5 v6 rows" "N1/N1-B N1/N1-C N1/N1-E N1/N1-H N1/N1-Z" "$(bash "$HERE/v6-results.sh" N1 "$r" | jq -r .row | sort | paste -sd' ')"
r=$(v6rec R1 TASK_ID=R1 TASK_SPEC="$HERE/tasks-v6.json" TASK_ACCEPTANCE="$HERE/ACCEPTANCE-v6.md")
check "v6-results: R1 gives 8 + 9 + 4 rows" "21" "$(bash "$HERE/v6-results.sh" R1 "$r" | wc -l)"
check "v6-results: wrong launch id refused" "refused" "$(bash "$HERE/v6-results.sh" N1 "$r" >/dev/null 2>&1 && echo accepted || echo refused)"
# A run.json with the record mark (as the v6 code writes it): the legacy containment
# rows read FAIL in the receipt, unchanged, and v6-results replaces them by R1-CM.
jq --argjson m "$(bash "$HERE/mark-evidence.sh" "$HERE/tests-v6/mark/real.cortex-mark")" \
  '.containment.outside_new_files = ["./compose.cortex-mark"] | .containment.cortex_mark = $m' "$tmp/run/run.json" >"$tmp/rj" && cp "$tmp/rj" "$tmp/run/run.json"
r=$(v6rec R1 TASK_ID=R1 TASK_SPEC="$HERE/tasks-v6.json" TASK_ACCEPTANCE="$HERE/ACCEPTANCE-v6.md")
check "mark run: legacy rows still computed, FAIL" "FAIL FAIL" "$(jq -r '"\([.acceptance[] | select(.criterion == "Containment: workspace")][0].result) \([.acceptance_v5.authority[] | select(.row == "A1")][0].result)"' "$r")"
check "mark run: v6-results drops them, keeps R1-CM" "0 1 21" "$(bash "$HERE/v6-results.sh" R1 "$r" | jq -s -r '"\(map(select(.row == "R1/A1" or .row == "R1/v1 Containment: workspace")) | length) \(map(select(.row == "R1/R1-CM")) | length) \(length)"')"
check "mark run: R1-CM sees the mark well formed" "true" "$(jq -r '.acceptance_v6[] | select(.row == "R1-CM") | .value.cortex_mark.well_formed' "$r")"
# A full PASS set built from the declaration scores PASS; then one row removed,
# one row FAIL, one launch failed.
jq -c '.rows[] | {row, rep: 1, verdict: "PASS"}' "$DECL" >"$tmp/all.jsonl"
check "score: every declared row PASS" "PASS" "$(bash "$SCORER" "$DECL" "$tmp/all.jsonl" | jq -r .verdict)"
grep -v '"T5/T5-N"' "$tmp/all.jsonl" >"$tmp/missing.jsonl"
check "score: one row missing is not PASS" "NOT_RUN" "$(bash "$SCORER" "$DECL" "$tmp/missing.jsonl" | jq -r .verdict)"
sed 's/{"row":"N2\/N2-F","rep":1,"verdict":"PASS"}/{"row":"N2\/N2-F","rep":1,"verdict":"FAIL"}/' "$tmp/all.jsonl" >"$tmp/onefail.jsonl"
check "score: one row FAIL" "FAIL" "$(bash "$SCORER" "$DECL" "$tmp/onefail.jsonl" | jq -r .verdict)"
{ grep -v '"R1/' "$tmp/all.jsonl"; bash "$HERE/v6-results.sh" R1 - "$DECL" "launch failed: no run.json"; } >"$tmp/nolaunch.jsonl"
check "score: failed launch R1 is FAIL" "FAIL 21" "$(bash "$SCORER" "$DECL" "$tmp/nolaunch.jsonl" | jq -r '"\(.verdict) \([.rows[] | select(.verdict == "FAIL")] | length)"')"
{ cat "$tmp/all.jsonl"; echo '{"row":"N1/v1 Task completion","rep":1,"verdict":"FAIL"}'; } >"$tmp/extra.jsonl"
check "score: an undeclared negative legacy row is an extra, not counted" "PASS 1" "$(bash "$SCORER" "$DECL" "$tmp/extra.jsonl" | jq -r '"\(.verdict) \(.extras | length)"')"

echo "test-rows-v6: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
