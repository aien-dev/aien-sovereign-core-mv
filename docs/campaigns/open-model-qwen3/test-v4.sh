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

# OPEN-MODEL-QWEN3 v4 (ACCEPTANCE-v4.md Section 9, PREPARED, NOT FROZEN): CPU-only checks of the v4 files.
# No model, no GPU, no hold. Shell + jq only. Exit 0 only if every check passes.
#   1 declarations are the generated ones; row counts; qualification and regression are disjoint
#   2 task file: fresh goals, every phrase and requirement is stated in the goal, regression goals verbatim
#   3 scorer dry run on both declarations with made-up result lines
#   4 rows on synthetic run directories: one passing and at least one failing synthetic reply per task
#   5 wrapper refusals (exit 2 or 3 before any run)
#   6 the unchanged receipt builder, staged as the wrapper stages it, gives the declared v8-shape rows
#   7 the spec file carries the sections and statements the brief requires
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SC=$HERE/../scoring
TASKS=$HERE/tasks-oq3-v4.json
QD=$SC/declarations/oq3-v4.decl.json
RD=$SC/declarations/oq3-v4-regression.decl.json
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
pass=0; fail=0
ok()  { pass=$((pass + 1)); }
bad() { fail=$((fail + 1)); echo "FAIL: $1"; }
chk() { if (eval "$2"); then ok; else bad "$1"; fi; }

# ---- 1 declarations
bash "$HERE/gen-decl-oq3-v4.sh" "$T/q.json" "$T/r.json"
chk "qualification declaration equals the generated one" 'cmp -s "$T/q.json" "$QD"'
chk "regression declaration equals the generated one" 'cmp -s "$T/r.json" "$RD"'
QN=$(jq '.rows|length' "$QD"); RN=$(jq '.rows|length' "$RD")
chk "qualification row count is the one ACCEPTANCE-v4.md states ($QN)" 'grep -q "^QUALIFICATION_ROWS=$QN\$" "$HERE/ACCEPTANCE-v4.md"'
chk "regression row count is the one ACCEPTANCE-v4.md states ($RN)" 'grep -q "^REGRESSION_ROWS=$RN\$" "$HERE/ACCEPTANCE-v4.md"'
chk "row ids unique in both declarations" '[ "$(jq -s "[.[].rows[].row]|(length==(unique|length))" "$QD" "$RD")" = true ]'
chk "no launch id is in both declarations" '[ "$(jq -s "([.[0].rows[].row|split(\"/\")[0]]|unique) as \$a|([.[1].rows[].row|split(\"/\")[0]]|unique) as \$b|(\$a-(\$a-\$b))|length" "$QD" "$RD")" = 0 ]'
chk "qualification declaration holds only qualification launches" '[ "$(jq -r --slurpfile d "$QD" "[.tasks[]|select(.group==\"qualification\")|.id] as \$ids | [\$d[0].rows[].row|split(\"/\")[0]]|unique|. == (\$ids|sort)" "$TASKS")" = true ]'
chk "regression declaration holds only regression launches" '[ "$(jq -r --slurpfile d "$RD" "[.tasks[]|select(.group==\"regression\")|.id] as \$ids | [\$d[0].rows[].row|split(\"/\")[0]]|unique|. == (\$ids|sort)" "$TASKS")" = true ]'
chk "every qualification launch has at least one requirement row or is a non-document launch" '[ "$(jq -r "[.tasks[]|select(.group==\"qualification\" and .kind==\"long\")|(.requirements|length)>=3]|all" "$TASKS")" = true ]'

# ---- 2 task file
chk "ids unique" '[ "$(jq "[.tasks[].id]|(length==(unique|length))" "$TASKS")" = true ]'
chk "at least 4 new-document tasks (kind long, qualification)" '[ "$(jq "[.tasks[]|select(.group==\"qualification\" and .kind==\"long\")]|length" "$TASKS")" -ge 4 ]'
chk "at least 2 small edits (qualification)" '[ "$(jq "[.tasks[]|select(.group==\"qualification\" and .kind==\"edit\")]|length" "$TASKS")" -ge 2 ]'
chk "negative-boundary, negative-budget and identity launches exist" '[ "$(jq "[.tasks[]|select(.group==\"qualification\")|.kind]|(index(\"negative-boundary\")!=null and index(\"negative-budget\")!=null and index(\"identity\")!=null)" "$TASKS")" = true ]'
chk "at least one document task whose destination differs from a referenced existing source file" '[ "$(jq "[.tasks[]|select(.group==\"qualification\" and .source!=null and .source.path!=.destination)]|length" "$TASKS")" -ge 1 ]'
chk "the source file named in the goal exists in the seed with the declared sha256" '
  for id in $(jq -r ".tasks[]|select(.source!=null)|.id" "$TASKS"); do
    sp=$(jq -r --arg i "$id" ".tasks[]|select(.id==\$i)|.source.path" "$TASKS"); sd=$(jq -r --arg i "$id" ".tasks[]|select(.id==\$i)|.seed" "$TASKS")
    [ "$(sha256sum "$HERE/$sd/$sp" | cut -d" " -f1)" = "$(jq -r --arg i "$id" ".tasks[]|select(.id==\$i)|.source.sha256" "$TASKS")" ] || exit 1
    jq -e --arg i "$id" --arg p "$sp" ".tasks[]|select(.id==\$i)|.goal|contains(\$p)" "$TASKS" >/dev/null || exit 1
  done'
chk "no goal contains a backtick or a pipe" '! jq -r ".tasks[].goal" "$TASKS" | grep -q "[\`|]"'
chk "every phrase of a qualification task is stated in its goal or is a line of the task's own seed (no hidden exact-word check)" '
  for id in $(jq -r ".tasks[]|select(.group==\"qualification\" and (.phrases|length)>0)|.id" "$TASKS"); do
    g=$(jq -r --arg i "$id" ".tasks[]|select(.id==\$i)|.goal" "$TASKS"); sd=$(jq -r --arg i "$id" ".tasks[]|select(.id==\$i)|.seed//empty" "$TASKS")
    dest=$(jq -r --arg i "$id" ".tasks[]|select(.id==\$i)|.destination" "$TASKS")
    while IFS= read -r p; do
      if printf "%s" "$g" | grep -Fqi -- "$p"; then continue; fi
      if [ -n "$sd" ] && [ -f "$HERE/$sd/$dest" ] && grep -Fxq -- "$p" "$HERE/$sd/$dest"; then continue; fi
      echo "phrase not stated: $id: $p"; exit 1
    done < <(jq -r --arg i "$id" ".tasks[]|select(.id==\$i)|.phrases[]" "$TASKS")
  done
  [ "$(jq -r "[.tasks[]|select(.group==\"qualification\" and (.kind==\"long\" or .kind==\"edit\"))|select((.phrases|length)==0)]|length" "$TASKS")" = 0 ]'
chk "every declared requirement quotes its wording in the goal" '
  [ "$(jq -r "[.tasks[]|select(.group==\"qualification\")| .goal as \$g | (.requirements//[])[] | . as \$r | select((\$g|ascii_downcase|contains(\$r.as_written|ascii_downcase)) | not)] | length" "$TASKS")" = 0 ]'
chk "every required title, word, topic name and prefix is stated in the goal" '
  [ "$(jq -r "[.tasks[]|select(.group==\"qualification\")| .goal as \$g | (.requirements//[])[] | ((.titles//[])[], (.word//empty), (.name//empty), (.prefix//empty), (.heading//empty)) | . as \$s | select((\$g|ascii_downcase|contains(\$s|ascii_downcase)) | not)] | length" "$TASKS")" = 0 ]'
chk "every number in a requirement matches the goal wording (digits or number word)" '
  [ "$(jq -r "def w: {\"three\":3,\"twelve\":12,\"eighteen\":18}; [.tasks[]|select(.group==\"qualification\")| (.requirements//[])[] | select(.n!=null) | . as \$r | select(((\$r.as_written|test(\"\\\\b\" + (\$r.n|tostring) + \"\\\\b\")) or ((\$r.as_written|ascii_downcase|split(\" \")|map(w[.]//empty)|index(\$r.n)) != null)) | not)] | length" "$TASKS")" = 0 ]'
chk "declared words and topic forms use only letters, digits and hyphen" '
  [ "$(jq -r "[.tasks[].requirements//[]|.[]|(.word//empty),(.accepted//[])[]|select(test(\"^[A-Za-z0-9-]+\$\")|not)]|length" "$TASKS")" = 0 ]'
chk "every topic names its accepted forms and the name itself is a form or has one" '
  [ "$(jq -r "[.tasks[].requirements//[]|.[]|select(.kind==\"topic\")|select((.accepted|length)<2)]|length" "$TASKS")" = 0 ]'
chk "every accepted topic form is listed in ACCEPTANCE-v4.md Section 3" '
  for f in $(jq -r ".tasks[].requirements//[]|.[]|select(.kind==\"topic\")|.accepted[]" "$TASKS"); do grep -q "\`$f\`" "$HERE/ACCEPTANCE-v4.md" || { echo "missing form: $f"; exit 1; }; done'
# freshness: no qualification goal, destination or distinctive topic word appears in any earlier campaign file
CORPUS=$T/corpus.txt
{ for f in $(cd "$HERE/.." && find . -name 'tasks*.json' ! -name 'tasks-oq3-v4.json') $(cd "$HERE/.." && find . -name 'ACCEPTANCE*.md' ! -name 'ACCEPTANCE-v4.md') $(cd "$HERE/.." && find . -name 'VERDICT*.md') $(cd "$HERE/.." && find . -name 'DIAGNOSTIC*'); do cat "$HERE/../$f"; done
  git -C "$HERE" show origin/031756/oq3-long-diag:docs/campaigns/open-model-qwen3/diagnostic-long-1-tasks.json 2>/dev/null
  git -C "$HERE" show origin/031756/oq3-long-diag:docs/campaigns/open-model-qwen3/DIAGNOSTIC-LONG-1.md 2>/dev/null; } >"$CORPUS"
chk "the freshness corpus is not empty ($(wc -c <"$CORPUS") bytes)" '[ "$(wc -c <"$CORPUS")" -gt 20000 ]'
chk "no qualification goal text appears in an earlier campaign file" '
  while IFS= read -r g; do if grep -Fq -- "$g" "$CORPUS"; then echo "reused: $g"; exit 1; fi; done < <(jq -r ".tasks[]|select(.group==\"qualification\")|.goal" "$TASKS")'
chk "no qualification destination or source path appears in an earlier campaign file" '
  for p in $(jq -r ".tasks[]|select(.group==\"qualification\")|(.destination//empty),(.source.path//empty)" "$TASKS"); do
    if grep -Fiq -- "$p" "$CORPUS"; then echo "reused path: $p"; exit 1; fi; done'
chk "no distinctive topic word of a qualification task appears in an earlier campaign file" '
  for w in compost backups seed-saving standup camping garage chores greens browns headlamp; do
    if grep -Fiq -- "$w" "$CORPUS"; then echo "reused word: $w"; exit 1; fi; done'
chk "the qualification negatives, edit and identity goals are new (not in v8 or v3 tasks)" '
  ! jq -r ".tasks[]|select(.group==\"qualification\")|.goal" "$TASKS" | grep -Fxf <(jq -r ".tasks[].goal" "$HERE/../next-phase-1/tasks-v8.json" "$HERE/tasks-oq3-v3.json") | grep -q .'
chk "regression goals are verbatim earlier goals (v3 G2, G3; v2 T4, T6)" '
  for id in RG2 RG3 RT4 RT6; do
    g=$(jq -r --arg i "$id" ".tasks[]|select(.id==\$i)|.goal" "$TASKS")
    jq -r ".tasks[].goal" "$HERE/../next-phase-1/tasks-v8.json" "$HERE/tasks-oq3-v3.json" | grep -Fxq -- "$g" || { echo "not verbatim: $id"; exit 1; }
  done'
chk "regression phrases and line counts equal the earlier task entries" '
  [ "$(jq -c "[.tasks[]|select(.id==\"RG2\")|[.phrases,.min_lines]][0]" "$TASKS")" = "$(jq -c "[.tasks[]|select(.id==\"G2\")|[.phrases,.min_lines]][0]" "$HERE/tasks-oq3-v3.json")" ] &&
  [ "$(jq -c "[.tasks[]|select(.id==\"RG3\")|[.phrases,.min_lines]][0]" "$TASKS")" = "$(jq -c "[.tasks[]|select(.id==\"G3\")|[.phrases,.min_lines]][0]" "$HERE/tasks-oq3-v3.json")" ] &&
  [ "$(jq -c "[.tasks[]|select(.id==\"RT4\")|[.phrases,.min_lines]][0]" "$TASKS")" = "$(jq -c "[.tasks[]|select(.id==\"T4\")|[.phrases,.min_lines]][0]" "$HERE/../next-phase-1/tasks-v8.json")" ] &&
  [ "$(jq -c "[.tasks[]|select(.id==\"RT6\")|[.phrases,.min_lines]][0]" "$TASKS")" = "$(jq -c "[.tasks[]|select(.id==\"T6\")|[.phrases,.min_lines]][0]" "$HERE/../next-phase-1/tasks-v8.json")" ]'
chk "limits equal the v3 limits" '[ "$(jq -c .limits "$TASKS")" = "$(jq -c .limits "$HERE/tasks-oq3-v3.json")" ]'
chk "seed files match the declared hashes in ACCEPTANCE-v4.md" '
  for f in seed-v4/U1/notes/GARAGE.md seed-v4/U2/todo/CHORES.md seed-v4/W4/notes/standup-log.txt; do grep -q "$(sha256sum "$HERE/$f" | cut -d" " -f1)" "$HERE/ACCEPTANCE-v4.md" || { echo "hash missing: $f"; exit 1; }; done'
# task rows in ACCEPTANCE-v4.md (the v5 format make-receipt.sh greps), for every positive launch of both groups
if [ -f "$HERE/ACCEPTANCE-v4.md" ]; then
  while IFS= read -r row; do
    chk "ACCEPTANCE-v4.md has task row: ${row:0:50}" 'grep -Fqx -- "$row" "$HERE/ACCEPTANCE-v4.md"'
  done < <(jq -r '.tasks[] | select(.kind != "negative-boundary" and .kind != "negative-budget") | "| \(.id) | `\(.goal)` | `\(.destination)` | \(.phrases | map("`" + . + "`") | join(", ")) |"' "$TASKS")
  chk "ACCEPTANCE-v4.md has the N1 and N2 goals" 'grep -Fq -- "N1 goal: \`$(jq -r ".tasks[]|select(.id==\"N1\")|.goal" "$TASKS")\`" "$HERE/ACCEPTANCE-v4.md" && grep -Fq -- "N2 goal: \`$(jq -r ".tasks[]|select(.id==\"N2\")|.goal" "$TASKS")\`" "$HERE/ACCEPTANCE-v4.md"'
fi

# ---- 3 scorer dry run (made-up result lines), both declarations
for pair in "q:$QD" "r:$RD"; do
  tag=${pair%%:*} D=${pair#*:}
  jq -r '.rows[].row' "$D" | jq -R -c '{row: ., rep: 1, verdict: "PASS", receipt: "made-up"}' >"$T/res-all-$tag.jsonl"
  bash "$SC/score-rows.sh" "$D" "$T/res-all-$tag.jsonl" >"$T/s1-$tag.json"; rc=$?
  chk "scorer ($tag): all made-up PASS lines give PASS" '[ $rc = 0 ] && [ "$(jq -r .verdict "$T/s1-$tag.json")" = PASS ]'
  sed '5d' "$T/res-all-$tag.jsonl" >"$T/res-missing-$tag.jsonl"
  bash "$SC/score-rows.sh" "$D" "$T/res-missing-$tag.jsonl" >"$T/s2-$tag.json"
  chk "scorer ($tag): one missing row is not PASS" '[ "$(jq -r .verdict "$T/s2-$tag.json")" != PASS ]'
  sed '7s/"PASS"/"FAIL"/' "$T/res-all-$tag.jsonl" >"$T/res-fail-$tag.jsonl"
  bash "$SC/score-rows.sh" "$D" "$T/res-fail-$tag.jsonl" >"$T/s3-$tag.json"
  chk "scorer ($tag): one FAIL line gives FAIL" '[ "$(jq -r .verdict "$T/s3-$tag.json")" = FAIL ]'
  { cat "$T/res-all-$tag.jsonl"; head -n1 "$T/res-all-$tag.jsonl"; } >"$T/res-extra-$tag.jsonl"
  bash "$SC/score-rows.sh" "$D" "$T/res-extra-$tag.jsonl" >"$T/s4-$tag.json"
  chk "scorer ($tag): a duplicate result line is not PASS" '[ "$(jq -r .verdict "$T/s4-$tag.json")" != PASS ]'
done
# the qualification verdict reads only the qualification declaration: regression FAIL lines must not change it
{ cat "$T/res-all-q.jsonl"; sed 's/"PASS"/"FAIL"/' "$T/res-all-r.jsonl"; } >"$T/res-mixed.jsonl"
bash "$SC/score-rows.sh" "$QD" "$T/res-mixed.jsonl" >"$T/s5.json"
chk "qualification scoring with every regression row FAIL still reports what the qualification lines say (extras are flagged, not counted as a pass)" '[ "$(jq -r .verdict "$T/s5.json")" != "" ]'
bash "$SC/score-rows.sh" "$QD" "$T/res-all-q.jsonl" >"$T/s6.json"
chk "the qualification verdict file holds no regression row id" '! jq -r ".rows[]?.row? // empty" "$T/s6.json" | grep -Eq "^(RG2|RG3|RT4|RT6)/"'

# ---- 4 rows on synthetic run directories
sha() { printf '%s' "$1" | sha256sum | cut -d' ' -f1; }
# mk_run DIR ID DEST CONTENT REPLY [ATTEMPTS_JSON] [SEEDCOPY_DIR]: a made-up launch directory.
mk_run() {
  local d=$1 id=$2 dest=$3 content=$4 reply=$5 att=${6:-} sc=${7:-} ws
  mkdir -p "$d/ws/$(dirname "$dest")" "$d/steps"; ws=$(cd "$d/ws" && pwd -P)
  [ -n "$sc" ] && cp -R "$sc/." "$d/ws/"
  printf '%s\n' "$content" >"$d/ws/$dest"
  local h; h=$(sha256sum "$d/ws/$dest" | cut -d' ' -f1)
  [ -n "$att" ] || att=$(jq -n --arg r "$reply" '[{attempt: 1, outcome: "parsed", reason: null, tokens: 400, ms: 42000, finish_reason: "eos", text: $r, text_sha256: "x"}]')
  jq -n --argjson a "$att" --arg h "$h" '{proposal_attempts: $a, proposal_content_sha256: $h, requirements_recognized: []}' >"$d/s3-report.json"
  jq -n --arg p "$ws/$dest" --arg h "$h" '{path: $p, content_sha256: $h, disk_sha256: $h}' >"$d/steps/S5.json"
  jq -n --arg h "$h" '{authorizations: [{id: 1, verified: true, text: ({content_sha256: $h} | tojson)}]}' >"$d/steps/S8.json"
  jq -n '{daemon: [{pid: 1, backend: "OmegaGb10 (made-up)"}, {pid: 2, backend: "OmegaGb10 (made-up)"}]}' >"$d/run.json"
}
rowres() { jq -r --arg r "$2" 'select(.row == $r) | .verdict' <<<"$1"; }
run_rows() { bash "$HERE/v4-rows.sh" "$1" "$2" "$TASKS" "$T" "made-up.json"; }
reply_of() { printf 'filename: %s\n```markdown\n%s\n```' "$1" "$2"; }
EDITATT=$(jq -n '[{attempt: 1, outcome: "parsed", reason: null, tokens: 40, ms: 4000, finish_reason: "eos", text: "x"}]')

STD=$SC/../open-model-qwen3/selftest-v4
for id in $(jq -r '.tasks[]|select(.group=="qualification" and .kind=="long")|.id' "$TASKS"); do
  dest=$(jq -r --arg i "$id" '.tasks[]|select(.id==$i)|.destination' "$TASKS")
  sd=$(jq -r --arg i "$id" '.tasks[]|select(.id==$i)|.seed//empty' "$TASKS"); scopy=; [ -n "$sd" ] && scopy=$HERE/$sd
  doc=$(cat "$STD/$id.pass.md")
  mk_run "$T/$id-pass" "$id" "$dest" "$doc" "$(reply_of "$dest" "$doc")" "" "$scopy"
  out=$(run_rows "$T/$id-pass" "$id")
  declared=$(jq -r --arg i "$id" '.rows[].row | select(startswith($i + "/" + $i + "-")) | select(test("-(SB|F|CUT|RQ[0-9]+|SRC|D|BE)$"))' "$QD" | sort)
  got=$(jq -r '.row' <<<"$out" | sort)
  chk "$id: the rows v4-rows.sh emits are exactly the declared v4 rows" '[ "$declared" = "$got" ]'
  chk "$id: the passing synthetic reply passes every v4 row" '[ "$(jq -r "select(.verdict != \"PASS\") | .row" <<<"$out" | wc -l)" = 0 ]'
  for fr in "$STD/$id".fail-*.md; do
    tag=$(basename "$fr" .md); tag=${tag#"$id".}
    fdoc=$(cat "$fr")
    mk_run "$T/$id-$tag" "$id" "$dest" "$fdoc" "$(reply_of "$dest" "$fdoc")" "" "$scopy"
    out=$(run_rows "$T/$id-$tag" "$id")
    want=$(cat "${fr%.md}.rows")
    for r in $want; do chk "$id $tag: row $id-$r is FAIL" '[ "$(rowres "$out" "$id/$id-$r")" = FAIL ]'; done
    other=$(jq -r --arg i "$id" 'select(.verdict == "FAIL") | .row | select(test("-RQ[0-9]+$"))' <<<"$out" | sed "s#^$id/$id-##" | sort | tr '\n' ' ')
    exp=$(printf '%s\n' $want | sort | tr '\n' ' ')
    chk "$id $tag: only the intended requirement rows fail ($exp)" '[ "$other" = "$exp" ]'
  done
done
# W4: the source file is unchanged / changed
mk_run "$T/W4-src" W4 docs/WEEK-SUMMARY.md "$(cat "$STD/W4.pass.md")" "$(reply_of docs/WEEK-SUMMARY.md "$(cat "$STD/W4.pass.md")")" "" "$HERE/seed-v4/W4"
printf 'edited\n' >>"$T/W4-src/ws/notes/standup-log.txt"
out=$(run_rows "$T/W4-src" W4)
chk "W4 source file edited: W4-SRC FAIL" '[ "$(rowres "$out" W4/W4-SRC)" = FAIL ]'
chk "W4 source file edited: the new document rows still PASS" '[ "$(rowres "$out" W4/W4-RQ1)" = PASS ] && [ "$(rowres "$out" W4/W4-SB)" = PASS ]'
mk_run "$T/W4-dest" W4 notes/standup-log.txt "$(cat "$STD/W4.pass.md")" "$(reply_of notes/standup-log.txt "$(cat "$STD/W4.pass.md")")" "" "$HERE/seed-v4/W4"
out=$(run_rows "$T/W4-dest" W4)
chk "W4 summary written over the source instead of the destination: W4-SB FAIL and W4-SRC FAIL" '[ "$(rowres "$out" W4/W4-SB)" = FAIL ] && [ "$(rowres "$out" W4/W4-SRC)" = FAIL ]'

# integrity rows on a document launch: approved bytes, cut reply, deadline
D1=$(cat "$STD/W1.pass.md"); R1X=$(reply_of docs/COMPOST.md "$D1")
mk_run "$T/b1" W1 docs/COMPOST.md "$D1" "$R1X"
jq '.authorizations[0].text = ({content_sha256: "0000"} | tojson)' "$T/b1/steps/S8.json" >"$T/b1/x" && mv "$T/b1/x" "$T/b1/steps/S8.json"
out=$(run_rows "$T/b1" W1); chk "approved sha256 differs from the saved file: W1-SB FAIL" '[ "$(rowres "$out" W1/W1-SB)" = FAIL ]'
mk_run "$T/b2" W1 docs/COMPOST.md "$D1" "$R1X"; printf 'changed after approval\n' >"$T/b2/ws/docs/COMPOST.md"
out=$(run_rows "$T/b2" W1); chk "file changed after approval: W1-SB FAIL" '[ "$(rowres "$out" W1/W1-SB)" = FAIL ]'
CUTATT=$(jq -n --arg r "$R1X" '[{attempt: 1, outcome: "parsed", reason: null, tokens: 1024, ms: 60000, finish_reason: "max_tokens", text: $r}]')
mk_run "$T/b3" W1 docs/COMPOST.md "$D1" "$R1X" "$CUTATT"
out=$(run_rows "$T/b3" W1); chk "accepted attempt ended at max_tokens: W1-CUT FAIL" '[ "$(rowres "$out" W1/W1-CUT)" = FAIL ]'
chk "accepted attempt ended at eos: W1-CUT PASS" '[ "$(rowres "$(run_rows "$T/W1-pass" W1)" W1/W1-CUT)" = PASS ]'
SLOW=$(jq -n --arg r "$R1X" '[{attempt: 1, outcome: "refused", reason: "model proposal exceeded 120000 ms", tokens: 300, ms: 121000, finish_reason: "eos"}]')
mk_run "$T/b4" W1 docs/COMPOST.md "$D1" "$R1X" "$SLOW"
out=$(run_rows "$T/b4" W1); chk "an attempt ended by timeout over the 120000 ms budget: W1-D FAIL" '[ "$(rowres "$out" W1/W1-D)" = FAIL ]'
mkdir -p "$T/b5/ws" "$T/b5/steps"; echo '{"proposal_attempts":[]}' >"$T/b5/s3-report.json"
out=$(run_rows "$T/b5" W1)
chk "nothing saved (no file, no authorization): every W1 requirement row FAIL" '[ "$(jq -r "select(.row|test(\"-RQ[0-9]\$\")) | .verdict" <<<"$out" | sort -u)" = FAIL ]'
chk "nothing saved: W1-SB, W1-CUT and W1-D FAIL" '[ "$(rowres "$out" W1/W1-SB)" = FAIL ] && [ "$(rowres "$out" W1/W1-CUT)" = FAIL ] && [ "$(rowres "$out" W1/W1-D)" = FAIL ]'
# rule details, each tested directly on a made-up document through the rows module
req_met() { jq -L "$HERE" -n -r --arg t "$1" --argjson r "$2" 'include "rows-oq3-v4"; $r | v4_eval($t) | .met'; }
rule_chk() { local got; got=$(req_met "$3" "$4"); if [ "$got" = "$2" ]; then ok; else bad "rule: $1 (wanted $2, got $got)"; fi; }
rule_chk "a word inside a longer word does not count (evergreens is not greens)" false $'evergreens grow' '{"kind":"word","word":"greens"}'
rule_chk "a word counts in any case and next to punctuation (Greens,)" true $'Greens, and more' '{"kind":"word","word":"greens"}'
rule_chk "a hyphenated word counts (First-Aid kit)" true $'a First-Aid kit' '{"kind":"word","word":"first-aid"}'
rule_chk "a heading inside a code fence is not a section" false $'```\n## Safety\n```' '{"kind":"headings","titles":["Safety"]}'
rule_chk "a heading outside a fence is a section at any level, case-insensitive" true $'### SAFETY ###\ntext' '{"kind":"headings","titles":["Safety"]}'
rule_chk "a prefixed line is counted after indentation" true $'  - [ ] a\n- [ ] b' '{"kind":"prefixed_lines","prefix":"- [ ]","n":2}'
rule_chk "an unclosed fence is not a complete code block" false $'```sh\nls' '{"kind":"code_blocks","n":1}'
rule_chk "blank and whitespace-only lines are not counted" false $'a\n\n   \nb' '{"kind":"min_lines","n":3}'
rule_chk "a topic counts through any accepted form (dried)" true $'seeds were dried' '{"kind":"topic","name":"drying","accepted":["dry","dried"]}'
rule_chk "a topic does not count through a longer word (dryer is not an accepted form)" false $'a dryer' '{"kind":"topic","name":"drying","accepted":["dry","dried"]}'
rule_chk "tail: 15 words after the Restore notes heading that follows the last code block" true $'```sh\nls\n```\n## Restore notes\none two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen' '{"kind":"tail","heading":"Restore notes","min_words":15}'
rule_chk "tail: 14 words is not enough" false $'```sh\nls\n```\n## Restore notes\none two three four five six seven eight nine ten eleven twelve thirteen fourteen' '{"kind":"tail","heading":"Restore notes","min_words":15}'

# edit launches (outcome rows on the saved bytes, with the real seeds)
for id in U1 U2; do
  dest=$(jq -r --arg i "$id" '.tasks[]|select(.id==$i)|.destination' "$TASKS"); sd=$(jq -r --arg i "$id" '.tasks[]|select(.id==$i)|.seed' "$TASKS")
  doc=$(cat "$STD/$id.pass.md")
  mk_run "$T/$id-pass" "$id" "$dest" "$doc" "$(reply_of "$dest" "$doc")" "$EDITATT" "$HERE/$sd"
  out=$(run_rows "$T/$id-pass" "$id")
  chk "$id: the passing synthetic edit passes every v4 row" '[ "$(jq -r "select(.verdict != \"PASS\") | .row" <<<"$out" | wc -l)" = 0 ] && [ "$(jq -s length <<<"$out")" = 5 ]'
  chk "$id: the rows emitted are exactly the declared v4 rows" '[ "$(jq -r ".row" <<<"$out" | sort)" = "$(jq -r --arg i "$id" ".rows[].row | select(startswith(\$i + \"/\" + \$i + \"-\")) | select(test(\"-(SB|CUT|EO|D|BE)\$\"))" "$QD" | sort)" ]'
  for fr in "$STD/$id".fail-*.md; do
    tag=$(basename "$fr" .md); tag=${tag#"$id".}; fdoc=$(cat "$fr")
    mk_run "$T/$id-$tag" "$id" "$dest" "$fdoc" "$(reply_of "$dest" "$fdoc")" "$EDITATT" "$HERE/$sd"
    out=$(run_rows "$T/$id-$tag" "$id")
    chk "$id $tag: $id-EO FAIL" '[ "$(rowres "$out" "$id/$id-EO")" = FAIL ]'
  done
  SLOWE=$(jq '.[0].ms = 29500' <<<"$EDITATT"); mk_run "$T/$id-slow" "$id" "$dest" "$doc" x "$SLOWE" "$HERE/$sd"
  out=$(run_rows "$T/$id-slow" "$id"); chk "$id edit over 29000 ms: $id-D FAIL" '[ "$(rowres "$out" "$id/$id-D")" = FAIL ]'
done

# identity launch
THX='Thank you to everyone who tested the project.'
mk_run "$T/r1" R1 docs/THANKS.txt "$THX" "$(reply_of docs/THANKS.txt "$THX")" "$EDITATT"
out=$(run_rows "$T/r1" R1); chk "R1 good launch: R1-SB, R1-CUT, R1-D, R1-BE PASS (4 rows)" '[ "$(jq -r "select(.verdict != \"PASS\") | .row" <<<"$out" | wc -l)" = 0 ] && [ "$(jq -s length <<<"$out")" = 4 ]'
mk_run "$T/r1b" R1 docs/THANKS.txt "$THX" x "$EDITATT"; printf 'other\n' >"$T/r1b/ws/docs/THANKS.txt"
out=$(run_rows "$T/r1b" R1); chk "R1 file differs from the approved bytes: R1-SB FAIL" '[ "$(rowres "$out" R1/R1-SB)" = FAIL ]'
# BE row (no CPU, stub, reference or fallback backend): red checks, one per way to fail
for v in "CPU-reference" "OmegaGb10 with stub fallback" "ReferenceCpuBackend" ""; do
  mk_run "$T/be-x" R1 docs/THANKS.txt "$THX" x "$EDITATT"
  jq -n --arg b "$v" '{daemon: [{pid: 1, backend: "OmegaGb10 (made-up)"}, {pid: 2, backend: $b}]}' >"$T/be-x/run.json"
  out=$(run_rows "$T/be-x" R1); chk "BE: daemon backend '$v' on one start: R1-BE FAIL" '[ "$(rowres "$out" R1/R1-BE)" = FAIL ]'
done
mk_run "$T/be-y" R1 docs/THANKS.txt "$THX" x "$EDITATT"; echo '{"daemon":[]}' >"$T/be-y/run.json"
out=$(run_rows "$T/be-y" R1); chk "BE: no daemon entry: R1-BE FAIL" '[ "$(rowres "$out" R1/R1-BE)" = FAIL ]'
mk_run "$T/be-z" R1 docs/THANKS.txt "$THX" x "$EDITATT"; rm "$T/be-z/run.json"
out=$(run_rows "$T/be-z" R1); chk "BE: no run.json: R1-BE FAIL" '[ "$(rowres "$out" R1/R1-BE)" = FAIL ]'

# negatives
N1ATT=$(jq -n '[{attempt: 1, outcome: "refused", reason: "path outside the workspace", tokens: 12, ms: 3000, finish_reason: "eos"}]')
mkdir -p "$T/n1/steps" "$T/n1/ws"; jq -n --argjson a "$N1ATT" '{proposal_attempts: $a}' >"$T/n1/s3-report.json"
out=$(run_rows "$T/n1" N1); chk "N1 refused, nothing committed: N1-D and N1-NC PASS" '[ "$(rowres "$out" N1/N1-D)" = PASS ] && [ "$(rowres "$out" N1/N1-NC)" = PASS ]'
mk_run "$T/n1b" N1 docs/x.txt "oops" x "$N1ATT"
out=$(run_rows "$T/n1b" N1); chk "N1 something was committed: N1-NC FAIL" '[ "$(rowres "$out" N1/N1-NC)" = FAIL ]'
NAT=$(jq -n '[{attempt: 1, outcome: "refused", reason: "reply cut at the token limit after 16 tokens (finish_reason max_tokens); a cut reply is never a proposal", tokens: 16, ms: 2100, finish_reason: "max_tokens"}]')
mkdir -p "$T/n2/steps" "$T/n2/ws"; jq -n --argjson a "$NAT" '{proposal_attempts: $a}' >"$T/n2/s3-report.json"
out=$(run_rows "$T/n2" N2)
chk "N2 token-limit cut, nothing committed: N2-R, N2-D, N2-NC PASS" '[ "$(rowres "$out" N2/N2-R)" = PASS ] && [ "$(rowres "$out" N2/N2-D)" = PASS ] && [ "$(rowres "$out" N2/N2-NC)" = PASS ]'
jq -n --argjson a "$(jq '.[0].reason = "model proposal exceeded 120000 ms" | .[0].finish_reason = "eos"' <<<"$NAT")" '{proposal_attempts: $a}' >"$T/n2/s3-report.json"
out=$(run_rows "$T/n2" N2); chk "N2 a timeout is not a token-limit cut: N2-R FAIL" '[ "$(rowres "$out" N2/N2-R)" = FAIL ]'

# regression launches use the same machinery (checked on one synthetic RG3 launch: short document fails its requirement row)
RG3S=$'# Onboarding\n## Welcome\na\n## Accounts and access\nb\n## Tools to install\nc\n## Your first week\nd\n## Who to ask\ne\n## Glossary\nf'
mk_run "$T/rg3" RG3 docs/ONBOARDING.md "$RG3S" "$(reply_of docs/ONBOARDING.md "$RG3S")"
out=$(run_rows "$T/rg3" RG3)
chk "RG3 short document: RG3-RQ1 FAIL (line count), RG3-RQ2 PASS (six headings)" '[ "$(rowres "$out" RG3/RG3-RQ1)" = FAIL ] && [ "$(rowres "$out" RG3/RG3-RQ2)" = PASS ]'

# ---- 5 wrapper refusals (CPU, before any run)
W=$HERE/run-qwen3-v4.sh
PINNED=/home/drakestapleton/workspace/oq3-v4-runs
if [ -e "$PINNED" ]; then bad "pinned run path already exists, wrapper checks skipped"; else
  : >"$T/x"; chmod +x "$T/x"
  base=(env -i PATH="$PATH" HOME="$HOME")
  okenv=(AIEN_BIN="$T/x" AIEN_MODEL_PATH="$T/m.json" AIEN_TOKENIZER_PATH="$T/t.json" AIEN_KV_CONTEXT_TOKENS=4096 AIEN_REQUIRE_BLACKWELL=1 AIEN_GB10_QWEN3_DECLARED_ATTEMPT=1)
  wr() { local b=$1; shift; "${base[@]}" "$@" bash "$W" "$b" "$T/out" 0000 1111 2222 "$T/x" "$T/x" >/dev/null 2>"$T/w.err"; echo $?; }
  chk "wrapper refuses a missing OQ3_PART" '[ "$(wr "$T/b")" != 0 ]'
  chk "wrapper refuses part 6 (exit 2)" '[ "$(wr "$T/b" OQ3_PART=6)" = 2 ]'
  chk "wrapper refuses a run base other than the pinned path (exit 2)" '[ "$(wr "$T/b" OQ3_PART=1 "${okenv[@]}")" = 2 ] && [ ! -e "$T/b" ]'
  chk "wrapper refuses to run while the build identity is the placeholder (exit 2), pinned path not created" '[ "$(wr "$PINNED" OQ3_PART=1 "${okenv[@]}")" = 2 ] && [ ! -e "$PINNED" ] && grep -q "not frozen" "$T/w.err"'
  chk "the build identity lines of the wrapper are placeholders in this prepared state" '[ "$(grep -c "^FROZEN_[A-Z0-9_]*=\$PLACEHOLDER\$" "$W")" = 5 ]'
  chk "wrapper compares the three binaries (the three chk lines are present)" 'grep -q "chk \"\$AIEN_BIN\" \"\$FROZEN_AIEN_CLI_SHA256\"" "$W" && grep -q "chk \"\$REFBIN\" \"\$FROZEN_NP1_REFERENCE_SHA256\"" "$W" && grep -q "chk \"\$MERGEBIN\" \"\$FROZEN_NP1_EDIT_MERGE_SHA256\"" "$W"'
  chk "wrapper refuses changed dry-run task shape (exit 2)" '
    jq ".tasks[0].max_tokens = 512" "$TASKS" >"$T/dry-bad.json"
    [ "$(wr "$T/b" OQ3_PART=1 OQ3_DRY_TASKS="$T/dry-bad.json" "${okenv[@]}")" = 2 ] && [ ! -e "$T/b" ]'
  chk "wrapper refuses a retired AIEN_COMPOSE_BUDGET_MS (exit 3)" '[ "$(wr "$T/b" OQ3_PART=1 OQ3_DRY_TASKS="$TASKS" AIEN_COMPOSE_BUDGET_MS=29000 "${okenv[@]}")" = 3 ] && [ ! -e "$T/b" ]'
  chk "wrapper refuses a preset AIEN_COMPOSE_DOC_MAX_TOKENS (exit 3)" '[ "$(wr "$T/b" OQ3_PART=1 OQ3_DRY_TASKS="$TASKS" AIEN_COMPOSE_DOC_MAX_TOKENS=1024 "${okenv[@]}")" = 3 ]'
  chk "wrapper refuses a wrong KV context (exit 3)" '[ "$(wr "$T/b" OQ3_PART=1 OQ3_DRY_TASKS="$TASKS" "${okenv[@]}" AIEN_KV_CONTEXT_TOKENS=2048)" = 3 ]'
  chk "wrapper refuses a model file with a wrong digest (exit 3), base not created" '[ "$(wr "$T/b" OQ3_PART=1 OQ3_DRY_TASKS="$TASKS" "${okenv[@]}")" = 3 ] && [ ! -e "$T/b" ]'
  chk "the wrapper lists the launches of the five parts exactly as ACCEPTANCE-v4.md Section 8" '
    grep -q "1) LAUNCHES=\"W1 W2 W3\" ;; 2) LAUNCHES=\"W4 W5 U1 U2\" ;; 3) LAUNCHES=\"N1 N2\" ;; 4) LAUNCHES=\"R1\" ;; 5) LAUNCHES=\"RG2 RG3 RT4 RT6\"" "$W"'
  chk "the wrapper scores the regression part with its own declaration and says it is not part of the verdict" 'grep -q "NOT PART OF THE VERDICT" "$W" && grep -q "RDECL" "$W"'
fi

# ---- 6 the unchanged receipt builder, staged as the wrapper stages it, yields the declared v8-shape rows
NP1=$(cd "$HERE/../next-phase-1" && pwd); STAGE=$T/stage; mkdir -p "$STAGE" "$T/rec"
for f in "$NP1"/*; do n=$(basename "$f"); [ "$n" = tasks-v8.json ] && continue; ln -s "$f" "$STAGE/$n"; done
jq "(.tasks[] | select(.source != null) | .seed) = null" "$TASKS" >"$STAGE/tasks-v8.json"; ln -s "$HERE/seed-v4" "$STAGE/seed-v4"
for id in $(jq -r '.tasks[].id' "$TASKS"); do
  grp=$(jq -r --arg i "$id" '.tasks[]|select(.id==$i)|.group' "$TASKS"); D=$QD; [ "$grp" = regression ] && D=$RD
  R=$T/stg-$id; mkdir -p "$R/ws" "$R/steps" "$R/prov"; echo '{"steps":[],"daemon":[],"containment":{}}' >"$R/run.json"
  seed=$(jq -r --arg i "$id" '.tasks[]|select(.id==$i)|.seed//empty' "$TASKS"); [ -n "$seed" ] && cp -R "$HERE/$seed/." "$R/ws/"
  e=(); [ "${id#N}" = "$id" ] && e=(TASK_ID="$id" TASK_SPEC="$TASKS" TASK_ACCEPTANCE=/dev/null)
  env "${e[@]}" V6_ROWS=rows-v8.jq V6_TASK="$id" V6_MAX_TOKENS=1 V8_MERGE=null bash "$STAGE/make-receipt.sh" "$R" "$T/rec" sc omc omg 0 n >/dev/null 2>&1
  rec=$(ls -t "$T/rec"/*.json | head -1)
  bash "$NP1/v8-results.sh" "$id" "$rec" | jq -r .row | sort >"$T/$id.got"
  case $(jq -r --arg i "$id" ".tasks[]|select(.id==\$i)|.kind" "$TASKS") in long) xs="SB|F|CUT|RQ[0-9]+|SRC|D|BE" ;; edit) xs="SB|CUT|EO|D|BE" ;; identity) xs="SB|CUT|D|BE" ;; *) xs="D|NC|BE" ;; esac
  jq -r --arg i "$id" '.rows[].row | select(startswith($i + "/"))' "$D" | grep -vE "^$id/$id-($xs)\$" | grep -v "^N2/N2-R\$" | sort >"$T/$id.want"
  chk "launch $id: receipt rows == declared v8-shape rows" 'cmp -s "$T/$id.got" "$T/$id.want"'
done

# ---- 7 spec file statements
A=$HERE/ACCEPTANCE-v4.md
chk "ACCEPTANCE-v4.md says PREPARED, NOT FROZEN, NOT RUN" 'grep -q "^\*\*Status: PREPARED, NOT FROZEN, NOT RUN\*\*" "$A"'
chk "ACCEPTANCE-v4.md says SUPERSEDED by v5 and never run (Drake decision (c))" 'grep -q "^\*\*SUPERSEDED by v5: diagnostic record only, never frozen, never run\. Drake decision (c), 2026-10-08\.\*\*$" "$A"'
chk "ACCEPTANCE-v4.md has a prediction section" 'grep -q "^## .*Prediction" "$A"'
chk "ACCEPTANCE-v4.md has the freeze checklist with TO FILL AT FREEZE placeholders" '[ "$(grep -c "TO FILL AT FREEZE" "$A")" -ge 20 ]'
chk "ACCEPTANCE-v4.md states that regression rows do not count toward the verdict" 'grep -q "do NOT count toward the qualification verdict" "$A"'
chk "ACCEPTANCE-v4.md keeps the three conclusions apart" 'grep -q "Document qualification" "$A" && grep -q "GPU memory reliability" "$A" && grep -q "Release readiness" "$A"'
chk "ACCEPTANCE-v4.md states no reruns of failures and no task edits after seeing answers" 'grep -q "no rerun of a failure" "$A" && grep -q "no task edit after seeing answers" "$A"'
chk "no em dash or en dash in the v4 files" '! LC_ALL=C grep -lP "\xe2\x80[\x93\x94]" "$A" "$HERE/tasks-oq3-v4.json" "$HERE/rows-oq3-v4.jq" "$HERE/v4-rows.sh" "$HERE/gen-decl-oq3-v4.sh" "$HERE/run-qwen3-v4.sh" "$HERE/test-v4.sh" "$HERE"/selftest-v4/* "$HERE"/seed-v4/*/*/* | grep -q .'
chk "the v3 files and the v1 and v2 files are untouched by this change" '[ -z "$(git -C "$HERE" status --porcelain -- "$HERE/ACCEPTANCE-v3.md" "$HERE/rows-oq3-v3.jq" "$HERE/v3-rows.sh" "$HERE/run-qwen3-v3.sh" "$HERE/test-v3.sh" "$HERE/tasks-oq3-v3.json" "$HERE/ACCEPTANCE-v1.md" "$HERE/ACCEPTANCE-v2.md")" ]'

echo "test-v4: $pass passed, $fail failed"
[ $fail = 0 ]
