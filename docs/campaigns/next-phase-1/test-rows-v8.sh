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

# NEXT-PHASE-1 v8: red/green tests for rows-v8.jq (row <id>-A, <id>-CM, T6/T7 row names), its wiring
# in make-receipt.sh, v8-results.sh, run-v8.sh, tasks-v8.json and np1-v8.decl.json.
# Shell + jq only. Usage:
#   NP1_EDIT_MERGE=/path/to/np1_edit_merge test-rows-v8.sh
# Build the binary with: cargo build -q -p aien-runtime --example np1_edit_merge
# (it lands in target/debug/examples/np1_edit_merge). Exit 0 only if every case holds.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
T=$HERE/tests-v6
MB=${NP1_EDIT_MERGE:-}
[ -n "$MB" ] && [ -x "$MB" ] || { echo "test-rows-v8: set NP1_EDIT_MERGE to the built np1_edit_merge example (cargo build -q -p aien-runtime --example np1_edit_merge)" >&2; exit 2; }
pass=0 fail=0
check() {   # check NAME EXPECTED ACTUAL
  if [ "$2" = "$3" ]; then pass=$((pass + 1)); echo "ok   $1 (got: ${3:-none})"
  else fail=$((fail + 1)); echo "FAIL $1: expected [${2}] got [${3}]"; fi
}
R8=$(cat "$HERE/rows-v8.jq")
fails_of() { jq -r "$R8"'
v6_rows | [.[] | select(.result != "PASS") | "\(.row)\(if .result == "NOT_RUN" then ":NOT_RUN" else "" end)"] | join(",")'; }
names_of() { jq -r "$R8"'v6_rows | map(.row) | join(",")'; }

# ---- fixtures: the frozen good evidence plus the v8 fields ---------------------
# Non-edit: the grant's proposal_sha256 is the accepted attempt text_sha256 (t1 / r1).
# Edit: the grant's proposal_sha256 is p1 and the re-derivation agrees with it.
fx() {   # fx KIND -> evidence JSON
  case $1 in
    T4) jq -c '.a2_v5_value = {s3_content_sha256: "c1"} | .authorizations[0].proposal_sha256 = "t1" | .s3.proposal_sha256 = "t1"' "$T/good-T4.evidence.json" ;;
    R1) jq -c '.a2_v5_value = {s3_content_sha256: "c1"} | .authorizations[0].proposal_sha256 = "r1" | .s3.proposal_sha256 = "r1"' "$T/good-R1.evidence.json" ;;
    T5) jq -c '.a2_v5_value = {s3_content_sha256: "c1"} | .authorizations[0].proposal_sha256 = "p1" | .s3.proposal_sha256 = "p1"
          | .v8_merge = {ok: true, path: "CHANGELOG.md", prior_sha256: .seed.sha256, reply_sha256: "t1", proposal_sha256: "p1", content_sha256: "c1"}' "$T/good-T5.evidence.json" ;;
  esac
}
mut() { fx "$1" | jq -c "$2" | fails_of; }
redof() { local k=$1; mut "$@" | tr ',' '\n' | grep -cx -e "$k-A" -e "$k-CM" | sed 's/^2$/A and CM red/'; }

# ---- 1. every <id>-A clause ----------------------------------------------------
for k in T4 R1 T5; do check "good $k evidence (v8 rows)" "" "$(fx $k | fails_of)"; done
for k in T4 R1 T5; do
  A="$k-A"; CM="$k-CM"
  check "$k: two authorizations" "A and CM red" "$(redof $k '.authorizations += [.authorizations[0] | .id = 10]')"
  check "$k: no authorization" "A and CM red" "$(redof $k '.authorizations = []')"
  check "$k: auth content differs from S5 content" "A and CM red" "$(redof $k '.authorizations[0].content_sha256 = "x"')"
  check "$k: S5 content differs" "A and CM red" "$(redof $k '.s5.content_sha256 = "x"')"
  check "$k: S5 disk differs" "A and CM red" "$(redof $k '.s5.disk_sha256 = "x"')"
  check "$k: S3 proposal content differs" "A and CM red" "$(redof $k '.a2_v5_value.s3_content_sha256 = "x"')"
  check "$k: S3 proposal content absent" "A and CM red" "$(redof $k '.a2_v5_value = null')"
  check "$k: auth path differs from S5 path" "A and CM red" "$(redof $k '.authorizations[0].path = "OTHER.md"')"
  check "$k: proposal_path differs" "A and CM red" "$(redof $k '.s3.proposal_path = "OTHER.md"')"
  check "$k: grant has no proposal_sha256" "A and CM red" "$(redof $k '.authorizations[0].proposal_sha256 = null')"
  check "$k: no accepted attempt" "A and CM red" "$(redof $k '.s3.attempts = []')"
done
# non-edit proposal clause (as v5 A2)
check "T4: proposal_sha256 differs from attempt text_sha256" "A and CM red" "$(redof T4 '.authorizations[0].proposal_sha256 = "other"')"
check "R1: attempt text_sha256 differs from proposal_sha256" "A and CM red" "$(redof R1 '.s3.attempts[0].text_sha256 = "other"')"
check "T4: a merge value is ignored for a non-edit launch" "" "$(mut T4 '.v8_merge = null')"
# edit proposal clause: re-derivation
check "T5: no re-derivation (null) is FAIL" "T5-A,T5-CM" "$(mut T5 '.v8_merge = null')"
check "T5: re-derivation field absent is FAIL" "T5-A,T5-CM" "$(mut T5 'del(.v8_merge)')"
check "T5: re-derivation ok false" "T5-A,T5-CM" "$(mut T5 '.v8_merge.ok = false')"
check "T5: reply_sha256 mismatch" "T5-A,T5-CM" "$(mut T5 '.v8_merge.reply_sha256 = "x"')"
check "T5: prior_sha256 mismatch" "T5-A,T5-CM" "$(mut T5 '.v8_merge.prior_sha256 = "x"')"
check "T5: proposal_sha256 mismatch" "T5-A,T5-CM" "$(mut T5 '.v8_merge.proposal_sha256 = "x"')"
check "T5: content_sha256 mismatch" "T5-A,T5-CM" "$(mut T5 '.v8_merge.content_sha256 = "x"')"
check "T5: no seed recorded" "T5-K,T5-B,T5-A,T5-CM" "$(mut T5 '.seed = null')"
check "T5: raw reply equal to proposal is not enough without re-derivation" "T5-A,T5-CM" "$(mut T5 '.authorizations[0].proposal_sha256 = "t1" | .v8_merge = null')"
# <id>-CM depends on <id>-A, not on the v5 row A2
check "T5: v5 A2 FAIL no longer fails CM" "" "$(mut T5 '.a2_v5 = "FAIL"')"
check "T5: v5 A2 null no longer fails CM" "" "$(mut T5 '.a2_v5 = null')"
check "T4: <id>-A FAIL fails CM (and nothing else)" "T4-A,T4-CM" "$(mut T4 '.authorizations[0].proposal_sha256 = "other"')"
check "T5: CM value records the A result" "FAIL" "$(fx T5 | jq '.v8_merge = null' | jq -r "$R8"'v6_rows | .[] | select(.row == "T5-CM") | .value.a_row')"


# ---- 1b. named negative tests (ACCEPTANCE-v8 Addendum A5): each turns <id>-A red ------
isred() { mut "$@" | tr ',' '\n' | grep -cx -e "$1-A" | sed 's/^1$/red/'; }
check "A5 altered content (edit): S5 disk != auth content" "red" "$(isred T5 '.s5.disk_sha256 = "x"')"
check "A5 altered content (non-edit): S5 disk != auth content" "red" "$(isred T4 '.s5.disk_sha256 = "x"')"
check "A5 changed target: auth path != S5 path and != destination" "red" "$(isred T5 '.authorizations[0].path = "OTHER.md"')"
check "A5 changed target: auth, S5 and proposal_path agree but not the destination" "red" \
  "$(isred T5 '.authorizations[0].path = "OTHER.md" | .s5.path_rel = "OTHER.md" | .s3.proposal_path = "OTHER.md"')"
check "A5 changed target (non-edit): same, R1" "red" "$(isred R1 '.authorizations[0].path = "OTHER.md" | .s5.path_rel = "OTHER.md" | .s3.proposal_path = "OTHER.md"')"
check "A5 stale seed: re-derivation prior_sha256 != declared pre-seed sha256" "red" "$(isred T5 '.v8_merge.prior_sha256 = "stale"')"
check "A5 stale seed: declared pre-seed differs from the re-derivation and the grant" "red" "$(isred T5 '.seed.sha256 = "stale"')"
check "A5 replay: two authorization records" "red" "$(isred T5 '.authorizations += [.authorizations[0] | .id = 10]')"
check "A5 replay: authorization prior_sha256 is not the pre-seed" "red" "$(isred T5 '.authorizations[0].prior_sha256 = "other"')"
check "A5 replay (non-edit): two authorization records" "red" "$(isred T4 '.authorizations += [.authorizations[0] | .id = 10]')"
check "A5 transformation after approval: precondition, auth content == proposal content" "c1 c1 c1" \
  "$(fx T5 | jq -r '[.authorizations[0].content_sha256, .a2_v5_value.s3_content_sha256, .s5.content_sha256] | join(" ")')"
check "A5 transformation after approval: S5 disk differs" "red" "$(isred T5 '.s5.disk_sha256 = "changed"')"
check "A5 transformation after approval: committed proposal differs from S3's" "red" "$(isred T5 '.s3.proposal_sha256 = "changed"')"
check "A5 missing re-derivation on an edit launch" "red" "$(isred T5 '.v8_merge = null')"
# A1: every stage present and non-empty
for s in raw_reply final_proposed_content approved_content committed_proposal executed_bytes; do :; done
check "A1 stages listed with their hashes (edit)" "raw_reply transform_inputs final_proposed_content approved_content committed_proposal executed_bytes" \
  "$(fx T5 | jq -r "$R8"'v6_rows | .[] | select(.row == "T5-A") | .value.stages | keys_unsorted | join(" ")' | sed 's/^/ /;s/^ //')"
check "A1 stage values (edit)" "t1 29e905c61fee80afd95f0a2b7a1ab35347f9fa0cb5871560085163662a1b072b c1 c1 p1 c1" \
  "$(fx T5 | jq -r "$R8"'v6_rows | .[] | select(.row == "T5-A") | .value.stages | [.raw_reply, .transform_inputs, .final_proposed_content, .approved_content, .committed_proposal, .executed_bytes] | join(" ")')"
check "A1 target listed (edit)" "CHANGELOG.md CHANGELOG.md CHANGELOG.md CHANGELOG.md" \
  "$(fx T5 | jq -r "$R8"'v6_rows | .[] | select(.row == "T5-A") | .value.target | [.target_auth, .target_s5, .target_proposal, .target_destination] | join(" ")')"
check "A1 non-edit: transform_inputs is none" "null" "$(fx T4 | jq -c "$R8"'v6_rows | .[] | select(.row == "T4-A") | .value.stages.transform_inputs')"
check "A1 empty reply hash (edit)" "red" "$(isred T5 '.s3.attempts[0].text_sha256 = "" | .v8_merge.reply_sha256 = ""')"
check "A1 empty approved content" "red" "$(isred T4 '.authorizations[0].content_sha256 = "" | .s5.content_sha256 = "" | .s5.disk_sha256 = "" | .a2_v5_value.s3_content_sha256 = ""')"
check "A1 empty executed bytes" "red" "$(isred T4 '.s5.disk_sha256 = null')"
check "A1 empty committed proposal" "red" "$(isred T4 '.authorizations[0].proposal_sha256 = "" | .s3.attempts[0].text_sha256 = "" | .s3.proposal_sha256 = ""')"
# ---- 2. row names carry the launch id (T6, T7), T4/T5 unchanged ----------------
check "T4 row names" "T4-L,T4-M,T4-A,T4-CM" "$(fx T4 | names_of)"
check "T5 row names" "T5-P,T5-K,T5-N,T5-B,T5-A,T5-CM" "$(fx T5 | names_of)"
check "R1 row names" "R1-X,R1-P,R1-T,R1-A,R1-CM" "$(fx R1 | names_of)"
T6=$(jq -c '.tasks[] | select(.id == "T6")' "$HERE/tasks-v8.json")
T7=$(jq -c '.tasks[] | select(.id == "T7")' "$HERE/tasks-v8.json")
fx6() { fx T4 | jq -c --argjson t "$T6" '.task = $t | .s3.proposal_path = $t.destination | .s5.path_rel = $t.destination | .authorizations[0].path = $t.destination | .containment.workspace_changed = [$t.destination]'; }
check "T6 row names" "T6-L,T6-M,T6-A,T6-CM" "$(fx6 | names_of)"
check "T6 good evidence" "" "$(fx6 | fails_of)"
check "T6: 11 non-empty lines" "T6-L" "$(fx6 | jq -c '.committed.text = ([range(11)] | map("line \(.)") | join("\n"))' | fails_of)"
check "T6: another max_tokens" "T6-M" "$(fx6 | jq -c '.max_tokens.used = 96' | fails_of)"
SEED7=$HERE/seed-v8/T7/TODO.md
T7TXT=$'# Todo\n\n## Next\n- write tests\n- fix login bug\n\n## Later\n- tidy docs\n'
fx7() { fx T5 | jq -c --argjson t "$T7" --rawfile s "$SEED7" --arg sh "$(sha256sum "$SEED7" | cut -d' ' -f1)" --arg c "$T7TXT" \
  '.task = $t | .seed = {path: "TODO.md", text: $s, sha256: $sh} | .committed.text = $c
   | .s3.proposal_path = "TODO.md" | .s5.path_rel = "TODO.md" | .authorizations[0].path = "TODO.md" | .authorizations[0].prior_sha256 = $sh
   | .v8_merge.prior_sha256 = $sh | .containment.workspace_changed = ["TODO.md"]'; }
check "T7 row names" "T7-P,T7-K,T7-N,T7-B,T7-A,T7-CM" "$(fx7 | names_of)"
check "T7 good evidence" "" "$(fx7 | fails_of)"
check "T7: new line after the next heading" "T7-N" "$(fx7 | jq -c '.committed.text = "# Todo\n\n## Next\n- write tests\n\n## Later\n- tidy docs\n- fix login bug\n"' | fails_of)"
check "T7: a seed line dropped" "T7-K" "$(fx7 | jq -c '.committed.text = "# Todo\n\n## Next\n- write tests\n- fix login bug\n\n## Later\n"' | fails_of)"
check "T7: second heading dropped" "T7-K" "$(fx7 | jq -c '.committed.text = "# Todo\n\n## Next\n- write tests\n- tidy docs\n- fix login bug\n"' | fails_of)"
check "T7: re-derivation absent" "T7-A,T7-CM" "$(fx7 | jq -c '.v8_merge = null' | fails_of)"
check "A5 missing re-derivation on an edit launch (T7 shape)" "red" "$(fx7 | jq -c '.v8_merge = null' | fails_of | tr ',' '\n' | grep -cx 'T7-A' | sed 's/^1$/red/')"
check "T7 seed bytes and sha256 as frozen" "52 0f55ac918c057213c0df504fb82ae62a5447c315ac10709953c9a55a6bffcc82" \
  "$(wc -c <"$SEED7" | tr -d ' ') $(sha256sum "$SEED7" | cut -d' ' -f1)"
check "T7 seed bytes" "$(printf '# Todo\n\n## Next\n- write tests\n\n## Later\n- tidy docs\n' | sha256sum | cut -d' ' -f1)" "$(sha256sum "$SEED7" | cut -d' ' -f1)"

# ---- 3. the real v7 T5 receipt: v7 A2 FAIL as recorded, v8 T5-A PASS -------------
REC=$HERE/d8cc248acede0b7029fa54ce6b2598b573aa63acc38ca2f65b9ccefe1e6d6b7b.json
REPLY=$HERE/replies/cd8e3b30c007e2f0bfe66277688dbec833d55a7bbefc47103203f2615125d094.txt
SEED5=$HERE/seed-v6/T5/CHANGELOG.md
MERGE=$("$MB" CHANGELOG.md "$SEED5" "$REPLY")
check "real merge: ok" "true" "$(jq -r .ok <<<"$MERGE")"
check "real merge: recorded A2 FAIL in the v7 receipt" "FAIL" "$(jq -r '.acceptance_v5.authority[] | select(.row == "A2") | .result' "$REC")"
check "real merge: recorded T5-CM FAIL in the v7 receipt" "FAIL" "$(jq -r '.acceptance_v6[] | select(.row == "T5-CM") | .result' "$REC")"
# evidence rebuilt from the receipt's own recorded values
EV=$(jq -c --argjson m "$MERGE" --rawfile seed "$SEED5" --arg ss "$(sha256sum "$SEED5" | cut -d' ' -f1)" '
  (.acceptance_v5.authority[] | select(.row == "A2") | .value) as $a2
  | (.acceptance_v6[] | select(.row == "T5-B") | .value) as $b
  | {task: {id: "T5", kind: "edit", destination: "CHANGELOG.md"},
     authorizations: [{path: $a2.auth_path, content_sha256: $a2.auth_content_sha256, proposal_sha256: $a2.auth_proposal_sha256, prior_sha256: $b.auth_prior_sha256}],
     s5: {content_sha256: $a2.s5_content_sha256, disk_sha256: $a2.s5_disk_sha256, path_rel: $a2.s5_path},
     s3: {proposal_path: (.runs[0].proposal_path), proposal_sha256: .runs[0].proposal_sha256, attempts: (.runs[0].proposal_attempts | map({attempt, outcome, text_sha256}))},
     seed: {sha256: $ss, text: $seed}, a2_v5: "FAIL", a2_v5_value: $a2, v8_merge: $m}' "$REC")
arow() { jq -r "$R8"'. as $e | v6_a_row("T5") | .result' ; }
check "real v7 T5 evidence: T5-A PASS" "PASS" "$(arow <<<"$EV")"
check "real v7 T5 evidence: seed sha256 is the recorded one" "$(jq -r '.acceptance_v6[] | select(.row == "T5-B") | .value.seed_sha256' "$REC")" "$(jq -r .seed.sha256 <<<"$EV")"
check "real v7 T5 evidence: re-derived proposal equals the grant's" "$(jq -r .proposal_sha256 <<<"$MERGE")" "$(jq -r '.authorizations[0].proposal_sha256' <<<"$EV")"
check "real v7 T5 evidence: re-derived reply is the accepted attempt" "$(jq -r .reply_sha256 <<<"$MERGE")" "$(jq -r '.s3.attempts | map(select(.outcome == "parsed")) | last | .text_sha256' <<<"$EV")"
check "real v7 T5 evidence: null re-derivation FAILs" "FAIL" "$(jq '.v8_merge = null' <<<"$EV" | arow)"
check "real v7 T5 evidence: wrong seed FAILs (re-derivation from a seed with an extra line)" "FAIL" \
  "$(tmpd=$(mktemp -d); printf '# Changelog\n\n## 0.1.0\n- initial release\n- x\n' >"$tmpd/s"; m=$("$MB" CHANGELOG.md "$tmpd/s" "$REPLY"); rm -rf "$tmpd"; jq --argjson m "$m" '.v8_merge = $m' <<<"$EV" | arow)"

# ---- 4. machine copies and declaration -------------------------------------------
check "tasks-v8.json order" "T4 T5 T6 T7 N1 N2 R1" "$(jq -r '[.tasks[].id] | join(" ")' "$HERE/tasks-v8.json")"
check "tasks-v8.json T4 and T5 copied from tasks-v6.json" "true" \
  "$(jq -n --slurpfile a "$HERE/tasks-v6.json" --slurpfile b "$HERE/tasks-v8.json" '[$a[0].tasks[] | select(.id == "T4" or .id == "T5")] == [$b[0].tasks[] | select(.id == "T4" or .id == "T5")]')"
check "tasks-v8.json N1, N2, R1 copied from tasks-v6.json" "true" \
  "$(jq -n --slurpfile a "$HERE/tasks-v6.json" --slurpfile b "$HERE/tasks-v8.json" '[$a[0].tasks[] | select(.id | IN("N1","N2","R1"))] == [$b[0].tasks[] | select(.id | IN("N1","N2","R1"))]')"
check "tasks-v8.json T7 seed" "seed-v8/T7" "$(jq -r '.tasks[] | select(.id == "T7") | .seed' "$HERE/tasks-v8.json")"
check "tasks-v8.json same field names as tasks-v6.json" "true" \
  "$(jq -n --slurpfile a "$HERE/tasks-v6.json" --slurpfile b "$HERE/tasks-v8.json" '([$a[0].tasks[] | keys] | unique) == ([$b[0].tasks[] | keys] | unique)')"
for id in T4 T5 T6 T7 R1; do
  row=$(jq -r --arg id "$id" '.tasks[] | select(.id == $id) | "| \(.id) | `\(.goal)` | `\(.destination)` | \(.phrases | map("`" + . + "`") | join(", ")) |"' "$HERE/tasks-v8.json")
  check "tasks-v8.json $id row is in ACCEPTANCE-v8.md" "yes" "$(grep -Fqx -- "$row" "$HERE/ACCEPTANCE-v8.md" && echo yes || echo no)"
done
check "ACCEPTANCE-v8.md records the T7 seed sha256" "1" "$(grep -c 0f55ac918c057213c0df504fb82ae62a5447c315ac10709953c9a55a6bffcc82 "$HERE/ACCEPTANCE-v8.md" | sed 's/^[2-9]/1/')"
check "ACCEPTANCE-v8.md has no em or en dash" "0" "$(grep -cP '\x{2014}|\x{2013}' "$HERE/ACCEPTANCE-v8.md")"
D=$HERE/../scoring/declarations
check "np1-v8.decl.json: 115 rows, all reps 1, no duplicates" "115 1 115" \
  "$(jq -r '"\(.rows | length) \([.rows[].reps] | unique | join(",")) \([.rows[].row] | unique | length)"' "$D/np1-v8.decl.json")"
check "np1-v8.decl.json: per launch counts" "N1 5, N2 5, R1 21, T4 20, T5 22, T6 20, T7 22" \
  "$(jq -r '[.rows[].row | split("/")[0]] | group_by(.) | map("\(.[0]) \(length)") | join(", ")' "$D/np1-v8.decl.json")"
check "np1-v8.decl.json: v7 rows with A2 replaced for T4 T5 R1" "true" \
  "$(jq -n --slurpfile a "$D/np1-v7.decl.json" --slurpfile b "$D/np1-v8.decl.json" '
     ([$a[0].rows[] | .row | select(startswith("T4/") or startswith("T5/") or startswith("R1/") | not)] | sort) as $same
     | ([$b[0].rows[] | .row | select(startswith("T6/") or startswith("T7/") | not)] | map(select(test("^(T4|T5|R1)/") | not)) | sort) == $same
     and ([$a[0].rows[] | .row | select(IN("T4/A2", "T5/A2", "R1/A2"))] | length) == 3
     and ([$b[0].rows[] | .row | select(IN("T4/A2", "T5/A2", "R1/A2"))] | length) == 0
     and ([$b[0].rows[] | .row | select(IN("T4/T4-A", "T5/T5-A", "R1/R1-A", "T6/T6-A", "T7/T7-A"))] | length) == 5')"
check "np1-v8.decl.json: T6 is T4 renamed, T7 is T5 renamed" "true" \
  "$(jq -n --slurpfile b "$D/np1-v8.decl.json" '
     ([$b[0].rows[] | .row | select(startswith("T4/")) | sub("^T4/T4-"; "T6/T6-") | sub("^T4/"; "T6/")]) == ([$b[0].rows[] | .row | select(startswith("T6/"))])
     and ([$b[0].rows[] | .row | select(startswith("T5/")) | sub("^T5/T5-"; "T7/T7-") | sub("^T5/"; "T7/")]) == ([$b[0].rows[] | .row | select(startswith("T7/"))])')"

# ---- 5. make-receipt.sh, v8-results.sh, scoring ----------------------------------
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
tmp=$(cd "$tmp" && pwd -P)
cp -R "$HERE/tests-v5/v4-run" "$tmp/run"
for f in "$tmp"/run/run.json "$tmp"/run/steps/*.json; do sed -i "s#@RUN@#$tmp/run#g" "$f"; done
rec() {   # rec LAUNCH [env...] -> receipt path
  local id=$1 o; shift
  o=$(mktemp -d "$tmp/out.XXXX")
  env "$@" V6_TASK="$id" V6_MAX_TOKENS=48 bash "$HERE/make-receipt.sh" "$tmp/run" "$o" t t t 0 >"$o.sum" 2>"$o.err" \
    || { echo "make-receipt.sh failed:"; cat "$o.err"; exit 1; }
  ls "$o"/*.json
}
v5e=(TASK_ID=R1 TASK_SPEC="$HERE/tasks-v8.json" TASK_ACCEPTANCE="$HERE/ACCEPTANCE-v8.md")
r7=$(rec R1 V6_ROWS=rows-v7.jq "${v5e[@]}")
r7m=$(rec R1 V6_ROWS=rows-v7.jq V8_MERGE='{"ok":true}' "${v5e[@]}")
check "rows-v7.jq: V8_MERGE changes nothing (receipt byte-identical)" "$(sha256sum <"$r7")" "$(sha256sum <"$r7m")"
r8=$(rec R1 V6_ROWS=rows-v8.jq "${v5e[@]}")
check "rows-v8.jq: receipt records the module" "rows-v8.jq $(sha256sum "$HERE/rows-v8.jq" | cut -d' ' -f1)" "$(jq -r '"\(.v6_rows_module.file) \(.v6_rows_module.sha256)"' "$r8")"
check "rows-v8.jq: R1 rows are named R1-A and R1-CM (no A2 in the v6 rows)" "R1-X,R1-P,R1-T,R1-A,R1-CM" "$(jq -r '[.acceptance_v6[].row] | join(",")' "$r8")"
check "rows-v8.jq: the v5 row A2 is still computed in the receipt" "1" "$(jq '[.acceptance_v5.authority[] | select(.row == "A2")] | length' "$r8")"
check "rows-v8.jq: R1-A sees the v5 A2 value (S3 content sha256)" "true" "$(jq -r '[.acceptance_v6[] | select(.row == "R1-A")][0].value.s3_content_sha256 != null' "$r8")"
check "rows-v8.jq: Q2 task_spec_matches_acceptance holds for R1 against ACCEPTANCE-v8.md" "true" "$(jq -r '.acceptance_v5.task_quality[] | select(.row == "Q2") | .value.task_spec_matches_acceptance' "$r8")"
check "rows-v8.jq: no V8_MERGE gives merge null" "null" "$(jq -c '[.acceptance_v6[] | select(.row == "R1-A")][0].value.merge' "$r8")"
rm8=$(rec T5 V6_ROWS=rows-v8.jq V8_MERGE='{"ok":true,"reply_sha256":"zz"}')
check "rows-v8.jq: V8_MERGE reaches the T5-A row" '{"ok":true,"reply_sha256":"zz"}' "$(jq -c '[.acceptance_v6[] | select(.row == "T5-A")][0].value.merge' "$rm8")"
check "make-receipt: V8_MERGE must be JSON" "refused" "$(V8_MERGE=notjson V6_ROWS=rows-v8.jq V6_TASK=T5 bash "$HERE/make-receipt.sh" "$tmp/run" "$tmp/out-bad" t t t 0 >/dev/null 2>&1 && echo accepted || echo refused)"
for id in T6 T7; do
  e=(TASK_ID=$id TASK_SPEC="$HERE/tasks-v8.json" TASK_ACCEPTANCE="$HERE/ACCEPTANCE-v8.md")
  rm -rf "$tmp/run-$id"; cp -R "$tmp/run" "$tmp/run-$id"   # the driver's goal is the task goal, as in a real launch
  jq --arg g "$(jq -r --arg id "$id" '.tasks[] | select(.id == $id) | .goal' "$HERE/tasks-v8.json")" '.goal = $g' "$tmp/run/run.json" >"$tmp/run-$id/run.json"
  o=$(mktemp -d "$tmp/out.XXXX")
  env V6_ROWS=rows-v8.jq "${e[@]}" V6_TASK=$id V6_MAX_TOKENS=48 bash "$HERE/make-receipt.sh" "$tmp/run-$id" "$o" t t t 0 >/dev/null 2>&1; r=$(ls "$o"/*.json)
  check "make-receipt: $id Q2 task row matches ACCEPTANCE-v8.md" "true" "$(jq -r '.acceptance_v5.task_quality[] | select(.row == "Q2") | .value.task_spec_matches_acceptance' "$r")"
  check "make-receipt: $id receipt rows are named for the launch" "true" "$(jq -r --arg id "$id" '[.acceptance_v6[].row | startswith($id + "-")] | all' "$r")"
done
# v8-results.sh: A2 dropped, <id>-A kept, A1 and the legacy containment row dropped
r=$(rec R1 V6_ROWS=rows-v8.jq "${v5e[@]}")
check "v8-results: R1 has no A2, no A1, has R1-A" "0 0 1 21" "$(bash "$HERE/v8-results.sh" R1 "$r" | jq -s -r '"\(map(select(.row == "R1/A2")) | length) \(map(select(.row == "R1/A1")) | length) \(map(select(.row == "R1/R1-A")) | length) \(length)"')"
check "v8-results: N1 gives only its 5 rows" "N1/N1-B N1/N1-C N1/N1-E N1/N1-H N1/N1-Z" "$(bash "$HERE/v8-results.sh" N1 "$(rec N1 V6_ROWS=rows-v8.jq)" | jq -r .row | sort | paste -sd' ')"
DECL=$D/np1-v8.decl.json; SCORER=$HERE/../scoring/score-rows.sh
jq -c '.rows[] | {row, rep: 1, verdict: "PASS"}' "$DECL" >"$tmp/all.jsonl"
check "score: every declared row PASS" "PASS" "$(bash "$SCORER" "$DECL" "$tmp/all.jsonl" | jq -r .verdict)"
grep -v '"T7/T7-A"' "$tmp/all.jsonl" >"$tmp/missing.jsonl"
check "score: T7/T7-A missing is not PASS" "NOT_RUN" "$(bash "$SCORER" "$DECL" "$tmp/missing.jsonl" | jq -r .verdict)"
sed 's/{"row":"T5\/T5-A","rep":1,"verdict":"PASS"}/{"row":"T5\/T5-A","rep":1,"verdict":"FAIL"}/' "$tmp/all.jsonl" >"$tmp/onefail.jsonl"
check "score: T5/T5-A FAIL" "FAIL" "$(bash "$SCORER" "$DECL" "$tmp/onefail.jsonl" | jq -r .verdict)"
{ grep -v '"T6/' "$tmp/all.jsonl"; bash "$HERE/v8-results.sh" T6 - "$DECL" "launch failed: no run.json"; } >"$tmp/nolaunch.jsonl"
check "score: failed launch T6 is FAIL on its 20 rows" "FAIL 20" "$(bash "$SCORER" "$DECL" "$tmp/nolaunch.jsonl" | jq -r '"\(.verdict) \([.rows[] | select(.verdict == "FAIL")] | length)"')"
{ cat "$tmp/all.jsonl"; echo '{"row":"T4/A2","rep":1,"verdict":"FAIL"}'; } >"$tmp/a2.jsonl"
check "score: an A2 line is not a declared row (extra, verdict stays PASS)" "PASS 1" "$(bash "$SCORER" "$DECL" "$tmp/a2.jsonl" | jq -r '"\(.verdict) \(.extras | length)"')"

# ---- 6. run-v8.sh is run-v7.sh plus the v8 wiring (pinned) ------------------------
d=$(diff "$HERE/run-v7.sh" "$HERE/run-v8.sh")
check "run-v8.sh vs run-v7.sh: hunk headers" "2,4c2,4 6c6 8c8 11c11,13 13c15 16,18c18,25 20c27 23c30 39a47,58 41,44c60,63 46,47c65,66 50c69 55,56c74,75" "$(grep -E '^[0-9]' <<<"$d" | paste -sd' ')"
check "run-v8.sh vs run-v7.sh: sha256 of the whole diff" "3baf95b5631a7629941e234992201acebbb13a82583bc38e0f4ca1cfb2150354" "$(sha256sum <<<"$d" | cut -d' ' -f1)"
check "run-v8.sh: declared wiring" "1 1 1 1 1 1" \
  "$(for p in 'TASKS=$HERE/tasks-v8.json' 'declarations/np1-v8.decl.json' 'V6_ROWS=rows-v8.jq V8_MERGE=' 'TASK_ACCEPTANCE="$HERE/ACCEPTANCE-v8.md"' 'SPEC="ACCEPTANCE-v8.md spec_version 8"' 'MERGEBIN=${7:?MERGE_BIN}'; do grep -cF -- "$p" "$HERE/run-v8.sh"; done | paste -sd' ')"
check "run-v8.sh: passes V8_MERGE and calls the merge binary for edits" "1 1" \
  "$(for p in 'V8_MERGE="$v8merge"' '"$MERGEBIN" "$(jq -r .destination <<<"$t")"'; do grep -cF -- "$p" "$HERE/run-v8.sh"; done | paste -sd' ')"
check "v8-results.sh vs v6-results.sh: only the A2 filter line differs in code" "26c27" "$(diff "$HERE/v6-results.sh" "$HERE/v8-results.sh" | grep -E '^[0-9]+(,[0-9]+)?[acd][0-9]' | grep -v '^[0-9,]*c1[0-9]*$' | grep -E '^26' | paste -sd' ')"

check "run-v8.sh: RUN_BASE pinned (guard lines present)" "1 1 1" \
  "$(for p in 'PINNED_BASE=/home/drakestapleton/workspace/np1-v8-runs' '[ "$BASE" = "$PINNED_BASE" ] || {' '[ ! -e "$BASE" ] || {'; do grep -cF -- "$p" "$HERE/run-v8.sh"; done | paste -sd' ')"
msg=$(bash "$HERE/run-v8.sh" "$tmp/other-base" "$tmp/o" a b c /bin/true /bin/true 2>&1); rc=$?
check "run-v8.sh: a different RUN_BASE is refused before anything runs" "2 yes no" \
  "$rc $(grep -q 'RUN_BASE must be /home/drakestapleton/workspace/np1-v8-runs' <<<"$msg" && echo yes || echo no) $([ -e "$tmp/other-base" ] && echo yes || echo no)"

# ---- harness approval-desk key (sc#342): rows-v9.jq and make-receipt.sh allow exactly ./approval-desk.key --------
# run-campaign.sh creates the key in the compose dir since sc#342. rows-v8.jq is pinned by completed runs' receipts and
# stays byte-identical (it still calls the key stray); rows-v9.jq is rows-v8.jq with only the v6_stray line changed.
R9=$(cat "$HERE/rows-v9.jq")
fails9() { jq -r "$R9"'
v6_rows | [.[] | select(.result != "PASS") | "\(.row)\(if .result == "NOT_RUN" then ":NOT_RUN" else "" end)"] | join(",")'; }
mut9() { fx "$1" | jq -c "$2" | fails9; }
for k in T4 R1 T5; do
  check "rows-v9 $k: good evidence" "" "$(fx $k | fails9)"
  check "rows-v9 $k: harness approval-desk key in the compose dir is allowed" "" "$(mut9 $k '.containment.compose_dir_files += ["./approval-desk.key"]')"
  for f in ./other.key ./approval-desk.key.bak ./approval-desk.keys ./sub/approval-desk.key ./branch-1.stage; do
    check "rows-v9 $k: $f beside the key is stray" "$k-CM" "$(mut9 $k ".containment.compose_dir_files += [\"./approval-desk.key\", \"$f\"]")"
  done
done
check "rows-v8 (pinned by completed runs, unchanged): the key is still stray there" "T4-CM" "$(mut T4 '.containment.compose_dir_files += ["./approval-desk.key"]')"
check "rows-v9.jq vs rows-v8.jq: outside comments, only the v6_stray line differs (exact harness names)" \
  '< def v6_stray: .containment.compose_dir_files | map(select(test("^\\./(machine\\.id|cortex\\.cx|jspace)") | not));|> def v6_stray: .containment.compose_dir_files | map(select(IN("./machine.id", "./cortex.cx", "./jspace", "./jspace/jspace.data", "./jspace/jspace.meta", "./approval-desk.key") | not));' \
  "$(diff <(grep -v '^#' "$HERE/rows-v8.jq") <(grep -v '^#' "$HERE/rows-v9.jq") | grep -E '^[<>]' | paste -sd'|')"
mrf=$(grep -F ') as $stray' "$HERE/make-receipt.sh" | sed -E 's/^ *\| *//; s/ as \$stray$//')
strayof() { jq -nc --argjson c "$1" "\$c as \$cfiles | $mrf"; }
check "make-receipt.sh: one compose-dir allowlist line" "1" "$(grep -cF ') as $stray' "$HERE/make-receipt.sh")"
check "make-receipt.sh: the harness files and the approval-desk key are not stray" "[]" \
  "$(strayof '["./approval-desk.key","./cortex.cx","./jspace","./jspace/jspace.data","./jspace/jspace.meta","./machine.id"]')"
for f in ./other.key ./approval-desk.key.bak ./approval-desk.keys ./sub/approval-desk.key ./branch-1.stage; do
  check "make-receipt.sh: $f beside the key is stray" "[\"$f\"]" "$(strayof "[\"./approval-desk.key\",\"./machine.id\",\"$f\"]")"
done
check "make-receipt.sh: V6_ROWS accepts rows-v9.jq; it reads tasks-v8.json and gets the A2 value and V8_MERGE as rows-v8.jq" "1 1 1" \
  "$(for p in 'rows-v6.jq|rows-v7.jq|rows-v8.jq|rows-v9.jq)' 'case $V6_ROWS in rows-v8.jq|rows-v9.jq) V6TF=$HERE/tasks-v8.json ;; esac' 'if ($rows6 == "rows-v8.jq" or $rows6 == "rows-v9.jq") then {a2_v5_value:'; do grep -cF -- "$p" "$HERE/make-receipt.sh"; done | paste -sd' ')"

# ---- exact names only (G6 follow-up): rows-v9.jq and make-receipt.sh ---------------------------------------------
# The real v5 dry compose-dir layout (G6, D2 run.json) passes; a name that only starts like a harness file is stray.
# rows-v8.jq (pinned) keeps its prefix match: ./machine.idX still passes there.
for k in T4 R1 T5; do
  check "rows-v9 $k: real v5 dry compose-dir layout is allowed" "" "$(mut9 $k '.containment.compose_dir_files = ["./approval-desk.key","./cortex.cx","./jspace","./jspace/jspace.data","./jspace/jspace.meta","./machine.id"]')"
  for f in ./machine.idX ./machine.id.bak ./cortex.cxY ./jspaceX ./jspace.data ./jspace/other ./jspace/jspace.meta.tmp ./jspace/jspace.dataX ./jspace/sub/jspace.data; do
    check "rows-v9 $k: $f (only a prefix of a harness name) is stray" "$k-CM" "$(mut9 $k ".containment.compose_dir_files += [\"$f\"]")"
  done
done
check "rows-v8 (pinned, unchanged): prefix match still admits ./machine.idX there" "" "$(mut T4 '.containment.compose_dir_files += ["./machine.idX"]' | tr ',' '\n' | grep -x T4-CM || true)"
check "make-receipt.sh: real v5 dry compose-dir layout is not stray" "[]" "$(strayof '["./approval-desk.key","./cortex.cx","./jspace","./jspace/jspace.data","./jspace/jspace.meta","./machine.id"]')"
for f in ./machine.idX ./machine.id.bak ./cortex.cxY ./jspaceX ./jspace.data ./jspace/other ./jspace/jspace.meta.tmp ./jspace/jspace.dataX ./jspace/sub/jspace.data; do
  check "make-receipt.sh: $f (only a prefix of a harness name) is stray" "[\"$f\"]" "$(strayof "[\"./machine.id\",\"$f\"]")"
done
echo "test-rows-v8: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
