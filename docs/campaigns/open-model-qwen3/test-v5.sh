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

# OPEN-MODEL-QWEN3 v5 (ACCEPTANCE-v5.md Sections 3.5 and 9 gate G5, DRAFT, NOT FROZEN): CPU-only checks of the v5 files.
# No model, no GPU, no hold. Shell + jq only. Derived from test-v4.sh. Exit 0 only if every check passes.
#   1 the declaration is the generated one; row count; per launch counts
#   2 task file: the goals are the authored ones, machine fields agree with the requirements, every requirement is stated in
#     the goal, freshness against every earlier campaign file
#   3 scorer dry run with made-up result lines
#   4 rows on synthetic run directories: a passing launch of every task, and red cases for every v5 row kind
#     (RQ, EP, PRE, DEC, GR, OPR, PIN, MEAS, N1-NR, N1-NM, N1-RC, N1-RH)
#   5 wrapper refusals (exit 2 or 3 before any run)
#   6 the unchanged receipt builder, staged as the wrapper stages it, gives the declared v8-shape rows
#   7 the spec file carries the statements the brief requires; no dashes; v4 files untouched
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SC=$HERE/../scoring
TASKS=$HERE/tasks-oq3-v5.json
QD=$SC/declarations/oq3-v5.decl.json
A=$HERE/ACCEPTANCE-v5.md
FROZEN=$HERE/frozen-v5.json
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
pass=0; fail=0
ok()  { pass=$((pass + 1)); }
bad() { fail=$((fail + 1)); echo "FAIL: $1"; }
chk() { if (eval "$2"); then ok; else bad "$1"; fi; }

# ---- 1 declaration
bash "$HERE/gen-decl-oq3-v5.sh" "$T/q.json"
chk "the declaration equals the generated one" 'cmp -s "$T/q.json" "$QD"'
QN=$(jq '.rows|length' "$QD")
chk "the row count is the one ACCEPTANCE-v5.md states ($QN)" 'grep -q "^QUALIFICATION_ROWS=$QN\$" "$A"'
chk "row ids unique" '[ "$(jq "[.rows[].row]|(length==(unique|length))" "$QD")" = true ]'
chk "the declaration holds exactly the qualification launches" '[ "$(jq -r --slurpfile d "$QD" "[.tasks[]|select(.group==\"qualification\")|.id] as \$ids | [\$d[0].rows[].row|split(\"/\")[0]]|unique|. == (\$ids|sort)" "$TASKS")" = true ]'
chk "per launch row counts (D1 34, D2 36, D3 36, D4 35, D5 36, D6 33, E1 33, E2 34, R1 32, N1 11, N2 14)" '
  [ "$(jq -c "[.rows[].row|split(\"/\")[0]]|group_by(.)|map({(.[0]): length})|add" "$QD")" = "{\"D1\":34,\"D2\":36,\"D3\":36,\"D4\":35,\"D5\":36,\"D6\":33,\"E1\":33,\"E2\":34,\"N1\":11,\"N2\":14,\"R1\":32}" ]'
chk "N1 declares NR, NM, RC and RH and none of N1-B, N1-C, N1-H, N1-D" '
  for r in NR NM RC RH NC BE OPR PIN MEAS Z E; do jq -e --arg r "N1/N1-$r" "any(.rows[]; .row == \$r)" "$QD" >/dev/null || exit 1; done
  ! jq -r ".rows[].row" "$QD" | grep -Eq "^N1/N1-(B|C|H|D|PRE|DEC|GR)\$"'
chk "every committed launch declares PRE, DEC, GR, OPR, PIN, MEAS; N2 declares them without GR" '
  for id in D1 D2 D3 D4 D5 D6 E1 E2 R1; do for r in PRE DEC GR OPR PIN MEAS; do jq -e --arg r "$id/$id-$r" "any(.rows[]; .row == \$r)" "$QD" >/dev/null || exit 1; done; done
  for r in PRE DEC OPR PIN MEAS; do jq -e --arg r "N2/N2-$r" "any(.rows[]; .row == \$r)" "$QD" >/dev/null || exit 1; done
  ! jq -e "any(.rows[]; .row == \"N2/N2-GR\")" "$QD" >/dev/null'
chk "only E2 declares EP" '[ "$(jq -r ".rows[].row|select(endswith(\"-EP\"))" "$QD")" = "E2/E2-EP" ]'
chk "one repetition, role case, no control, none NOT_APPLICABLE" '[ "$(jq "[.rows[]|select(.reps!=1 or .role!=\"case\" or .control!=null or .not_applicable!=false or .expected!=\"PASS\")]|length" "$QD")" = 0 ]'

# ---- 2 task file
GOALS=01ab9f9a26d79242391cf72b5f91b556a06a2b8dea6102604969e257ded9486c
chk "goals, destinations, seeds and phrases are the authored ones (digest $GOALS)" '[ "$(jq -c "[.tasks[]|{id,goal,destination,phrases,seed}]" "$TASKS" | sha256sum | cut -d" " -f1)" = "$GOALS" ]'
# The product refuses to write into a folder that does not exist (ACCEPTANCE-v5 Section 2, known limit): every positive
# destination's folder is in the default workspace that run-campaign.sh builds (the root and docs/) or in the task's seed.
chk "every positive destination's folder exists before the run (root, docs/ or the task's seed; R1 notes/ from seed-v5/R1)" '
  jq -r ".tasks[]|select(.kind|startswith(\"negative\")|not)|\"\(.destination) \(.seed // \"-\")\"" "$TASKS" | while read -r d s; do
    f=$(dirname "$d"); case "$f" in .|docs) continue ;; esac; [ "$s" != - ] && [ -d "$HERE/$s/$f" ] || exit 1; done'
chk "eleven qualification tasks, ids unique, no regression group" '[ "$(jq "[.tasks[]|select(.group==\"qualification\")]|length" "$TASKS")" = 11 ] && [ "$(jq "[.tasks[].id]|(length==(unique|length))" "$TASKS")" = true ] && [ "$(jq "[.tasks[]|select(.group!=\"qualification\")]|length" "$TASKS")" = 0 ]'
chk "N1 is kind negative-boundary and N2 negative-budget" '[ "$(jq -r ".tasks[]|select(.id==\"N1\" or .id==\"N2\")|.kind" "$TASKS" | tr "\n" " ")" = "negative-boundary negative-budget " ]'
chk "document machine fields agree with the declared requirements (min_lines, min_code_blocks; no tail heading)" '
  [ "$(jq "[.tasks[]|select(.kind==\"long\")|. as \$t
     | ((.requirements|map(select(.kind==\"min_lines\"))|first|.n) // 1) as \$ml
     | ((.requirements|map(select(.kind==\"code_blocks\"))|first|.n) // 0) as \$cb
     | select(\$t.min_lines != \$ml or \$t.min_code_blocks != \$cb or \$t.tail_heading != null or \$t.min_tail_words != 0)]|length" "$TASKS")" = 0 ]'
chk "every requirement count is an integer n and every word requirement a non-empty words list" '
  [ "$(jq "[.tasks[].requirements[]|select((.kind|IN(\"min_lines\",\"max_lines\",\"code_blocks\",\"prefixed_lines\",\"paragraphs\",\"tail\",\"sentences_exact\")) and ((.n|type)!=\"number\"))]|length" "$TASKS")" = 0 ] &&
  [ "$(jq "[.tasks[].requirements[]|select(.kind==\"word\" and ((.words|type)!=\"array\" or (.words|length)==0))]|length" "$TASKS")" = 0 ]'
chk "D4 source file exists in its seed with the declared sha256, is named in the goal and differs from the destination" '
  sp=$(jq -r ".tasks[]|select(.id==\"D4\")|.source.path" "$TASKS"); sd=$(jq -r ".tasks[]|select(.id==\"D4\")|.seed" "$TASKS")
  [ "$(sha256sum "$HERE/$sd/$sp" | cut -d" " -f1)" = "$(jq -r ".tasks[]|select(.id==\"D4\")|.source.sha256" "$TASKS")" ] &&
  jq -e --arg p "$sp" ".tasks[]|select(.id==\"D4\")|(.goal|contains(\$p)) and .destination != \$p" "$TASKS" >/dev/null'
chk "no goal contains a backtick or a pipe" '! jq -r ".tasks[].goal" "$TASKS" | grep -q "[\`|]"'
chk "every phrase is a substring of its goal" '[ "$(jq "[.tasks[]|.goal as \$g|.phrases[]|select(. as \$p|\$g|contains(\$p)|not)]|length" "$TASKS")" = 0 ]'
chk "every declared requirement quotes its wording in the goal" '
  [ "$(jq -r "[.tasks[]| .goal as \$g | (.requirements//[])[] | . as \$r | select((\$g|ascii_downcase|contains(\$r.as_written|ascii_downcase)) | not)] | length" "$TASKS")" = 0 ]'
chk "every required title, word, topic name, prefix and heading is stated in the goal" '
  [ "$(jq -r "[.tasks[]| .goal as \$g | (.requirements//[])[] | ((.titles//[])[], (.words//[])[], (.name//empty), (.prefix//empty), (.heading//empty)) | . as \$s | select((\$g|ascii_downcase|contains(\$s|ascii_downcase)) | not)] | length" "$TASKS")" = 0 ]'
chk "every requirement number matches the goal wording (digits or number word)" '
  [ "$(jq -r "def w: {\"one\":1,\"three\":3,\"seven\":7,\"twelve\":12}; [.tasks[]| (.requirements//[])[] | select(.n!=null) | . as \$r | select(((\$r.as_written|test(\"\\\\b\" + (\$r.n|tostring) + \"\\\\b\")) or ((\$r.as_written|ascii_downcase|split(\" \")|map(split(\"-\")[0])|map(w[.]//empty)|index(\$r.n)) != null)) | not)] | length" "$TASKS")" = 0 ]'
chk "declared words and topic forms use only letters, digits and hyphen" '[ "$(jq -r "[.tasks[].requirements[]|(.words//[])[],(.accepted//[])[]|select(test(\"^[A-Za-z0-9-]+\$\")|not)]|length" "$TASKS")" = 0 ]'
chk "every accepted topic form is listed in ACCEPTANCE-v5.md Section 3.1" '
  for f in $(jq -r ".tasks[].requirements[]|select(.kind==\"topic\")|.accepted[]" "$TASKS"); do grep -q "\`$f\`" "$A" || { echo "missing form: $f"; exit 1; }; done'
chk "the goal names the D3 topic forms the product reads (gate G2: withdraw and withdrew)" 'jq -e ".tasks[]|select(.id==\"D3\")|.goal|contains(\"withdraw\") and contains(\"withdrew\")" "$TASKS" >/dev/null'
chk "limits equal the v4 limits; per task max_tokens and budgets as ACCEPTANCE-v5 Section 2 (D6 1536 and 170000, Section 2.1)" '
  [ "$(jq -c .limits "$TASKS")" = "$(jq -c .limits "$HERE/tasks-oq3-v4.json")" ] &&
  [ "$(jq -c "[.tasks[]|[.id,.max_tokens,.budget_ms]]" "$TASKS")" = "[[\"D1\",1024,120000],[\"D2\",1024,120000],[\"D3\",1024,120000],[\"D4\",1024,120000],[\"D5\",1024,120000],[\"D6\",1536,170000],[\"E1\",96,29000],[\"E2\",96,29000],[\"N1\",1024,120000],[\"N2\",16,120000],[\"R1\",96,120000]]" ]'
# freshness: no v5 goal, destination or distinctive word appears in any earlier campaign file
CORPUS=$T/corpus.txt
{ for f in $(cd "$HERE/.." && find . -name 'tasks*.json' ! -name 'tasks-oq3-v5.json') $(cd "$HERE/.." && find . -name 'ACCEPTANCE*.md' ! -name 'ACCEPTANCE-v5.md') $(cd "$HERE/.." && find . -name 'VERDICT*.md') $(cd "$HERE/.." && find . -name 'DIAGNOSTIC*'); do cat "$HERE/../$f"; done
  git -C "$HERE" show origin/031756/oq3-long-diag:docs/campaigns/open-model-qwen3/diagnostic-long-1-tasks.json 2>/dev/null
  git -C "$HERE" show origin/031756/oq3-long-diag:docs/campaigns/open-model-qwen3/DIAGNOSTIC-LONG-1.md 2>/dev/null; } >"$CORPUS"
chk "the freshness corpus is not empty ($(wc -c <"$CORPUS") bytes)" '[ "$(wc -c <"$CORPUS")" -gt 20000 ]'
chk "no v5 goal text appears in an earlier campaign file" '
  while IFS= read -r g; do if grep -Fq -- "$g" "$CORPUS"; then echo "reused: $g"; exit 1; fi; done < <(jq -r ".tasks[].goal" "$TASKS")'
chk "no v5 destination or source path appears in an earlier campaign file" '
  for p in $(jq -r ".tasks[]|(.destination//empty),(.named_path//empty),(.source.path//empty)" "$TASKS"); do
    if grep -Fiq -- "$p" "$CORPUS"; then echo "reused path: $p"; exit 1; fi; done'
chk "no distinctive topic word of a v5 task appears in an earlier campaign file (Priya is the declared exception)" '
  for w in whetstone knife-sharpening rename-photos savings choir carry-on rescue-cat tap-repair scarf marathon swim-tip; do
    if grep -Fiq -- "$w" "$CORPUS"; then echo "reused word: $w"; exit 1; fi; done'
chk "ACCEPTANCE-v5.md has the task row of every positive launch (the format make-receipt.sh greps)" '
  while IFS= read -r row; do grep -Fqx -- "$row" "$A" || { echo "missing row: ${row:0:40}"; exit 1; }; done < <(jq -r ".tasks[] | select(.kind != \"negative-boundary\" and .kind != \"negative-budget\") | \"| \(.id) | \`\(.goal)\` | \`\(.destination)\` | \(.phrases | map(\"\`\" + . + \"\`\") | join(\", \")) |\"" "$TASKS")'
chk "ACCEPTANCE-v5.md has the N1 and N2 goals" 'grep -Fq -- "N1 goal (declared negative, not a Q2 row): \`$(jq -r ".tasks[]|select(.id==\"N1\")|.goal" "$TASKS")\`" "$A" && grep -Fq -- "N2 goal (declared negative, not a Q2 row): \`$(jq -r ".tasks[]|select(.id==\"N2\")|.goal" "$TASKS")\`" "$A"'
chk "frozen-v5.json: the model digest is index+shards and is the manifest of the listed sha256s (not re-hashed)" '
  m=$(jq -j ".model | \"aien-checkpoint-digest v1\nindex \(.index_sha256)\n\" + (.shards | sort_by(.[0]) | map(\"\(.[1])  \(.[0])\n\") | join(\"\"))" "$FROZEN" | sha256sum | cut -d" " -f1)
  [ "$m" = "$(jq -r .model.model_sha256 "$FROZEN")" ] && [ "$(jq -r .model.model_digest_kind "$FROZEN")" = index+shards ] && [ "$(jq -r .model.model_sha256 "$FROZEN")" = 17a78fbba447a4e66a3d886c0998fbcf2f9201d46e5e0c5bfec2d57c975976b7 ] &&
  [ "$(jq -r .model.index_sha256 "$FROZEN")" = d6c42883a895dfef5b0080ed2116a1bcd764f558406b98923d675978a1abf29c ] && [ "$(jq ".model.shards|length" "$FROZEN")" = 3 ] &&
  grep -q "NOT RE-HASHED" <<<"$(jq -r .model.model_sha256_note "$FROZEN")"'
chk "frozen-v5.json: the eight pins are filled (commits 40 hex, sha256 values 64 hex) and the status says FROZEN" 'jq -e "(.pins | [.sovereign_core_commit, .omega_lock_commit, .physics_lock_commit, .aienos_lock_commit] | all(test(\"^[0-9a-f]{40}$\"))) and (.pins | [.cargo_lock_sha256, .aien_cli_sha256, .np1_reference_sha256, .np1_edit_merge_sha256] | all(test(\"^[0-9a-f]{64}$\"))) and (.status | startswith(\"FROZEN\"))" "$FROZEN" >/dev/null'
chk "frozen-v5.json pins equal evidence-v5/build-summary.txt (commits, Cargo.lock and the three binaries)" '
  S=$HERE/evidence-v5/build-summary.txt; h() { awk -v n="$1" "\$2 ~ (\"/\" n \"\$\") {print \$1}" "$S"; }
  [ "$(jq -r .pins.sovereign_core_commit "$FROZEN")" = "$(grep -E "^[0-9a-f]{40}$" "$S" | sed -n 3p)" ] && [ "$(jq -r .pins.omega_lock_commit "$FROZEN")" = "$(grep -E "^[0-9a-f]{40}$" "$S" | sed -n 1p)" ] &&
  [ "$(jq -r .pins.physics_lock_commit "$FROZEN")" = "$(grep -E "^[0-9a-f]{40}$" "$S" | sed -n 2p)" ] && [ "$(jq -r .pins.cargo_lock_sha256 "$FROZEN")" = "$(h Cargo.lock)" ] &&
  [ "$(jq -r .pins.aien_cli_sha256 "$FROZEN")" = "$(h aien-cli)" ] && [ "$(jq -r .pins.np1_reference_sha256 "$FROZEN")" = "$(h np1_reference)" ] &&
  [ "$(jq -r .pins.np1_edit_merge_sha256 "$FROZEN")" = "$(h np1_edit_merge)" ] && grep -q "^aien-cli rc=0$" "$S" && grep -q "^examples rc=0$" "$S"'
chk "evidence-v5/model-rehash-2026-10-08.txt equals the frozen model values and the manifest digest" '
  R=$HERE/evidence-v5/model-rehash-2026-10-08.txt; r() { awk -v n="$1" "\$2 == n {print \$1}" "$R"; }
  [ "$(r model.safetensors.index.json)" = "$(jq -r .model.index_sha256 "$FROZEN")" ] && [ "$(r tokenizer.json)" = "$(jq -r .model.tokenizer_sha256 "$FROZEN")" ] &&
  [ "$(jq -r ".model.shards[] | .[1] + \"  \" + .[0]" "$FROZEN")" = "$(grep -E "^[0-9a-f]{64}  model-0000[0-9]-of-00003.safetensors$" "$R")" ] &&
  grep -q "^$(jq -r .model.model_sha256 "$FROZEN")  model_sha256 (index+shards)$" "$R"'
# The campaign files (ACCEPTANCE-v5 Section 7): the fixed list plus every file under seed-v5/ and selftest-v5/.
CAMPAIGN_FILES="run-qwen3-v5.sh tasks-oq3-v5.json ../scoring/declarations/oq3-v5.decl.json rows-oq3-v3.jq rows-oq3-v4.jq rows-oq3-v5.jq v5-rows.sh gen-decl-oq3-v5.sh test-v5.sh frozen-v5.json ../next-phase-1/run-campaign.sh ../next-phase-1/make-receipt.sh ../next-phase-1/rows-v5.jq ../next-phase-1/rows-v9.jq ../next-phase-1/v8-results.sh ../scoring/score-rows.sh"
chk "evidence-v5/campaign-files.sha256 lists exactly the campaign files and matches each one (regenerate it after any change)" '
  (cd "$HERE" && sha256sum --quiet --strict -c evidence-v5/campaign-files.sha256 >/dev/null 2>&1) &&
  [ "$(cd "$HERE" && { printf "%s\n" $CAMPAIGN_FILES; find seed-v5 selftest-v5 -type f; } | LC_ALL=C sort)" = "$(cut -c67- "$HERE/evidence-v5/campaign-files.sha256" | LC_ALL=C sort)" ]'

# ---- 3 scorer dry run (made-up result lines)
jq -r '.rows[].row' "$QD" | jq -R -c '{row: ., rep: 1, verdict: "PASS", receipt: "made-up"}' >"$T/res-all.jsonl"
bash "$SC/score-rows.sh" "$QD" "$T/res-all.jsonl" >"$T/s1.json"; rc=$?
chk "scorer: all made-up PASS lines give PASS" '[ $rc = 0 ] && [ "$(jq -r .verdict "$T/s1.json")" = PASS ]'
sed '5d' "$T/res-all.jsonl" >"$T/res-missing.jsonl"; bash "$SC/score-rows.sh" "$QD" "$T/res-missing.jsonl" >"$T/s2.json"
chk "scorer: one missing row is not PASS" '[ "$(jq -r .verdict "$T/s2.json")" != PASS ]'
sed '7s/"PASS"/"FAIL"/' "$T/res-all.jsonl" >"$T/res-fail.jsonl"; bash "$SC/score-rows.sh" "$QD" "$T/res-fail.jsonl" >"$T/s3.json"
chk "scorer: one FAIL line gives FAIL" '[ "$(jq -r .verdict "$T/s3.json")" = FAIL ]'
{ cat "$T/res-all.jsonl"; head -n1 "$T/res-all.jsonl"; } >"$T/res-dup.jsonl"; bash "$SC/score-rows.sh" "$QD" "$T/res-dup.jsonl" >"$T/s4.json"
chk "scorer: a duplicate result line is not PASS" '[ "$(jq -r .verdict "$T/s4.json")" != PASS ]'
{ cat "$T/res-all.jsonl"; for r in B C H; do jq -n -c --arg r "N1/N1-$r" '{row: $r, rep: 1, verdict: "FAIL", receipt: "made-up"}'; done; } >"$T/res-extra.jsonl"
bash "$SC/score-rows.sh" "$QD" "$T/res-extra.jsonl" >"$T/s5.json"
chk "scorer: the undeclared v8 rows N1-B, N1-C, N1-H failing are extras and leave the verdict PASS" '[ "$(jq -r .verdict "$T/s5.json")" = PASS ]'

# ---- 4 rows on synthetic run directories
sha() { printf '%s' "$1" | sha256sum | cut -d' ' -f1; }
M=$(jq -c .model "$FROZEN")
MSHA=$(jq -r .model_sha256 <<<"$M"); TSHA=$(jq -r .tokenizer_sha256 <<<"$M")
SHARDS=$(jq -r '"CHECKPOINT_SHARDS model_sha256=\(.model_sha256) model_digest_kind=\(.model_digest_kind) index_sha256=\(.index_sha256) shards=[\(.shards | map("\(.[0]):\(.[1])") | join(","))]"' <<<"$M")
OPS='rmsnorm,apply_rope,matmul_vec,matmul_batch,swiglu,gqa_attention,paged_attention,paged_attention_batch,compute_logits,rmsnorm_heads'
GBE='OmegaGb10Backend (native Omega engine, no CUDA, NVIDIA GB10 sm_121)'
STRICT="  STRICT strict=true dev_fallback_build=false require_checkpoint=true backend=$GBE"
OPR_LINE="OP_REPORT native=[$OPS] reference=[] native_fallbacks=[] reference_runs=[] backend=$GBE"
CT='Project constraints: every change stays inside the authorized workspace; one file per change; plain text only; no network.'
# a frozen file with filled pins, and a launch record that matches it
PINS='{"sovereign_core_commit":"1111111111111111111111111111111111111111","omega_lock_commit":"2222222222222222222222222222222222222222","physics_lock_commit":"3333333333333333333333333333333333333333","aienos_lock_commit":"4444444444444444444444444444444444444444","cargo_lock_sha256":"5555555555555555555555555555555555555555555555555555555555555555","aien_cli_sha256":"6666666666666666666666666666666666666666666666666666666666666666","np1_reference_sha256":"7777777777777777777777777777777777777777777777777777777777777777","np1_edit_merge_sha256":"8888888888888888888888888888888888888888888888888888888888888888"}'
FZ=$T/frozen-filled.json; jq --argjson p "$PINS" '.pins = $p' "$FROZEN" >"$FZ"
# The same values with every pin back to the placeholder (the draft before the freeze fill).
FPH=$T/frozen-placeholder.json; jq '.placeholder as $p | .pins |= map_values($p) | .status = "DRAFT, NOT FROZEN (test copy)"' "$FROZEN" >"$FPH"
# mk_launch DIR WALL_MS: the wrapper's launch record for DIR (DIR.launch-v5.json)
mk_launch() { jq -n --argjson p "$PINS" --argjson w "${2:-600000}" '{launch: "x", wall_ms: $w, wall_clock: "made-up", driver_exit: 0, meminfo_before: {mem_free_kb: 100000000, cached_kb: 2000000}, pins: $p}' >"$1.launch-v5.json"; }
# mk_logs DIR [EXTRA_LOG_LINE]: two daemon logs with the start line, the shards line and one OP_REPORT line each
mk_logs() {
  local k; for k in 1 2; do printf '\033[1mdaemon\033[0m up\n%s\n%s\n%s\n' "$STRICT" "$SHARDS" "$OPR_LINE" >"$1/daemon-$k.log"; done
  [ -z "${2:-}" ] || printf '%s\n' "$2" >>"$1/daemon-2.log"
}
mk_daemons() { jq -n --arg b "$GBE" --arg m "checkpoint loaded from /m/model.safetensors.index.json (tokenizer loaded, model_id=qwen3, config=c, model_sha256=$MSHA, tokenizer_sha256=$TSHA)" \
  '[{pid: 1, backend: $b, model: $m, vmhwm_kb: 9000000, warm_up_ms: 900}, {pid: 2, backend: $b, model: $m, vmhwm_kb: 9100000, warm_up_ms: 950}]'; }
steps_ok() { jq -n '[range(1; 9) as $i | {step: "S\($i)", name: "s", rc: 0, wall_ms: 10, ok: true}]'; }
# one greedy attempt that produced REPLY (parsed), N tokens, MS milliseconds
att1() { jq -n --arg r "$1" --argjson n "${2:-40}" --argjson ms "${3:-42000}" --arg h "$(sha "$1")" \
  '[{attempt: 1, outcome: "parsed", reason: null, tokens: $n, ms: $ms, finish_reason: "eos", text: $r, text_sha256: $h,
     token_ids: [range(0; $n)], decoding: {greedy_tokens: $n, sampled_tokens: 0}}]'; }
# mk_run DIR ID DEST CONTENT REPLY [ATTEMPTS_JSON] [SEEDCOPY_DIR]: a made-up committed launch, every v5 row green.
mk_run() {
  local d=$1 id=$2 dest=$3 content=$4 reply=$5 att=${6:-} sc=${7:-} ws h
  mkdir -p "$d/ws/$(dirname "$dest")" "$d/steps"; ws=$(cd "$d/ws" && pwd -P)
  [ -n "$sc" ] && cp -R "$sc/." "$d/ws/"
  printf '%s\n' "$content" >"$d/ws/$dest"; h=$(sha256sum "$d/ws/$dest" | cut -d' ' -f1)
  [ -n "$att" ] || att=$(att1 "$reply")
  jq -n --arg id "$id" --argjson a "$att" --arg h "$h" '{task: $id, committed: true, compose_commit: 101, generation_record: 100,
     requirements_uncertain: [], proposal_attempts: $a, proposal_content_sha256: $h}' >"$d/s3-report.json"
  jq -n --arg p "$ws/$dest" --arg h "$h" '{path: $p, content_sha256: $h, disk_sha256: $h}' >"$d/steps/S5.json"
  local acc; acc=$(jq -c 'map(select(.outcome == "parsed")) | last' <<<"$att")
  jq -n --arg id "$id" --arg h "$h" --argjson acc "$acc" --arg ms "$MSHA" --arg ts "$TSHA" --arg ct "$CT" '
    {ok: true, machine_id: "m1", constraints: [{id: 6, text: $ct, verified: true}],
     authorizations: [{id: 1, verified: true, text: ({content_sha256: $h, provenance: {generation_record: 100}} | tojson)}],
     recall: {host: [
       {id: 100, note: "generation", verified: true, text: ({generation: 1, origin: "compose_proposal", task: $id, attempt: $acc.attempt,
          output_text_sha256: $acc.text_sha256, model_sha256: $ms, model_digest_kind: "index+shards", tokenizer_sha256: $ts,
          decoding: {mode: "greedy", greedy_tokens: $acc.tokens, sampled_tokens: 0}} | tojson)},
       {id: 101, note: "compose_commit", verified: true, text: ({provenance: {generation_record: 100}} | tojson)}]}}' >"$d/steps/S8.json"
  jq -n --argjson s "$(steps_ok)" --argjson dm "$(mk_daemons)" --arg ct "$CT" '{steps: $s, daemon: $dm, containment: {workspace_changed: []}, constraint_text: $ct}' >"$d/run.json"
  mk_logs "$d"; mk_launch "$d"
}
rowres() { jq -r --arg r "$2" 'select(.row == $r) | .verdict' <<<"$1"; }
run_rows() { bash "$HERE/v5-rows.sh" "$1" "$2" "$TASKS" "$T" "made-up.json" "${3:-$1.launch-v5.json}" "${4:-$FZ}"; }
reply_of() { printf 'filename: %s\n```markdown\n%s\n```' "$1" "$2"; }
STD=$HERE/selftest-v5

# documents: one passing launch each (every v4 and v5 row PASS, exactly the declared extra rows), then the red fixtures
for id in $(jq -r '.tasks[]|select(.kind=="long")|.id' "$TASKS"); do
  dest=$(jq -r --arg i "$id" '.tasks[]|select(.id==$i)|.destination' "$TASKS")
  sd=$(jq -r --arg i "$id" '.tasks[]|select(.id==$i)|.seed//empty' "$TASKS"); scopy=; [ -n "$sd" ] && scopy=$HERE/$sd
  doc=$(cat "$STD/$id.pass.md")
  mk_run "$T/$id-pass" "$id" "$dest" "$doc" "$(reply_of "$dest" "$doc")" "" "$scopy"
  out=$(run_rows "$T/$id-pass" "$id")
  want=$(jq -r --arg i "$id" '.rows[].row | select(startswith($i + "/" + $i + "-")) | select(test("-(SB|F|CUT|RQ[0-9]+|SRC|D|BE|PRE|DEC|GR|OPR|PIN|MEAS)$"))' "$QD" | sort)
  chk "$id: the rows v5-rows.sh emits are exactly the declared v4 and v5 rows" '[ "$want" = "$(jq -r .row <<<"$out" | sort)" ]'
  chk "$id: the passing synthetic launch passes every row" '[ -n "$out" ] && [ "$(jq -r "select(.verdict != \"PASS\") | .row" <<<"$out" | wc -l)" = 0 ]'
  for fr in "$STD/$id".fail-*.md; do
    tag=$(basename "$fr" .md); tag=${tag#"$id".}; fdoc=$(cat "$fr")
    mk_run "$T/$id-$tag" "$id" "$dest" "$fdoc" "$(reply_of "$dest" "$fdoc")" "" "$scopy"
    out=$(run_rows "$T/$id-$tag" "$id")
    exp=$(tr ' ' '\n' <"${fr%.md}.rows" | grep . | sort | tr '\n' ' ')
    got=$(jq -r 'select(.verdict == "FAIL") | .row' <<<"$out" | sed "s#^$id/$id-##" | sort | tr '\n' ' ')
    chk "$id $tag: exactly the intended rows fail ($exp)" '[ "$got" = "$exp" ]'
  done
done
# rule details, each tested directly through the rows module
req_met() { cd "$HERE" && jq -L "$HERE" -n -r --arg t "$1" --argjson r "$2" 'include "rows-oq3-v5"; $r | v5_eval($t) | .met'; }
rule_chk() { local got; got=$(req_met "$3" "$4"); if [ "$got" = "$2" ]; then ok; else bad "rule: $1 (wanted $2, got $got)"; fi; }
rule_chk "a stated word inside a longer word does not count (triangle is not angle)" false $'a triangle' '{"kind":"word","words":["angle"]}'
rule_chk "every stated word is needed (angle without whetstone)" false $'the angle' '{"kind":"word","words":["whetstone","angle"]}'
rule_chk "stated words in any case, next to punctuation" true $'Whetstone, Angle.' '{"kind":"word","words":["whetstone","angle"]}'
rule_chk "a word requirement with no words list is FAIL (fail closed)" false $'angle' '{"kind":"word","word":"angle"}'
rule_chk "a count requirement without n is FAIL (fail closed)" false $'a\nb' '{"kind":"min_lines","min":1}'
rule_chk "an unknown requirement kind is FAIL" false $'a' '{"kind":"vibes","n":1}'
rule_chk "a topic form not in the table does not count (depositor)" false $'a depositor' '{"kind":"topic","name":"deposit","accepted":["deposit","deposits","deposited","depositing"]}'
rule_chk "a listed topic form counts (withdrew)" true $'I withdrew it' '{"kind":"topic","name":"withdraw","accepted":["withdraw","withdrew"]}'
rule_chk "paragraphs: blank lines, headings and list items end a run; fenced lines are not prose (5)" true $'a\nb\n\nc\n## H\nd\n- e\nf\n```\ng\n\nh\n```\ni' '{"kind":"paragraphs","n":5}'
rule_chk "paragraphs: the same text is not 6" false $'a\nb\n\nc\n## H\nd\n- e\nf\n```\ng\n\nh\n```\ni' '{"kind":"paragraphs","n":6}'
rule_chk "paragraphs: a paragraph split only by a heading counts as two" true $'a\n## H\nb' '{"kind":"paragraphs","n":2}'
rule_chk "paragraphs: a whitespace-only line ends a paragraph" true $'a\n   \nb' '{"kind":"paragraphs","n":2}'
rule_chk "paragraphs: numbered items are not prose" false $'1. a\n2) b' '{"kind":"paragraphs","n":1}'
rule_chk "tail: 12 words after Wrap Up following the last code block" true $'```\nx\n```\n## Wrap Up\none two three four five six seven eight nine ten eleven twelve' '{"kind":"tail","heading":"Wrap Up","n":12}'
rule_chk "tail: 11 words is not enough" false $'```\nx\n```\n## Wrap Up\none two three four five six seven eight nine ten eleven' '{"kind":"tail","heading":"Wrap Up","n":12}'
rule_chk "heading order: in order" true $'## A\n## B\n## C' '{"kind":"heading_order","titles":["A","B","C"]}'
rule_chk "heading order: swapped" false $'## A\n## C\n## B' '{"kind":"heading_order","titles":["A","B","C"]}'
rule_chk "heading order: one missing" false $'## A\n## C' '{"kind":"heading_order","titles":["A","B","C"]}'
rule_chk "one sentence: a single line ending with a full stop" true $'Stretch your shoulders before you swim.\n' '{"kind":"sentences_exact","n":1}'
rule_chk "one sentence: two sentences on one line" false $'Stretch. Then swim.' '{"kind":"sentences_exact","n":1}'
rule_chk "one sentence: no final punctuation" false $'Stretch your shoulders' '{"kind":"sentences_exact","n":1}'
rule_chk "one sentence: two lines" false $'Stretch your shoulders\nbefore you swim.' '{"kind":"sentences_exact","n":1}'
rule_chk "one sentence: empty" false '' '{"kind":"sentences_exact","n":1}'
rule_chk "max lines: 31 non-empty lines is over 30" false "$(seq 1 31)" '{"kind":"max_lines","n":30}'

# edit launches (with the real seeds): green, then the red fixtures
EA() { att1 "x" 40 4000; }
for id in E1 E2; do
  dest=$(jq -r --arg i "$id" '.tasks[]|select(.id==$i)|.destination' "$TASKS"); sd=$(jq -r --arg i "$id" '.tasks[]|select(.id==$i)|.seed' "$TASKS")
  doc=$(cat "$STD/$id.pass.md")
  mk_run "$T/$id-pass" "$id" "$dest" "$doc" "$(reply_of "$dest" "$doc")" "$(att1 "$(reply_of "$dest" "$doc")" 40 4000)" "$HERE/$sd"
  out=$(run_rows "$T/$id-pass" "$id")
  want=$(jq -r --arg i "$id" '.rows[].row | select(startswith($i + "/" + $i + "-")) | select(test("-(SB|CUT|EO|EP|D|BE|PRE|DEC|GR|OPR|PIN|MEAS)$"))' "$QD" | sort)
  chk "$id: the rows emitted are exactly the declared v4 and v5 rows" '[ "$want" = "$(jq -r .row <<<"$out" | sort)" ]'
  chk "$id: the passing synthetic edit passes every row" '[ -n "$out" ] && [ "$(jq -r "select(.verdict != \"PASS\") | .row" <<<"$out" | wc -l)" = 0 ]'
  for fr in "$STD/$id".fail-*.md; do
    tag=$(basename "$fr" .md); tag=${tag#"$id".}; fdoc=$(cat "$fr")
    mk_run "$T/$id-$tag" "$id" "$dest" "$fdoc" "$(reply_of "$dest" "$fdoc")" "$(att1 "$(reply_of "$dest" "$fdoc")" 40 4000)" "$HERE/$sd"
    out=$(run_rows "$T/$id-$tag" "$id")
    exp=$(tr ' ' '\n' <"${fr%.md}.rows" | grep . | sort | tr '\n' ' ')
    got=$(jq -r 'select(.verdict == "FAIL") | .row' <<<"$out" | sed "s#^$id/$id-##" | sort | tr '\n' ' ')
    chk "$id $tag: exactly the intended rows fail ($exp)" '[ "$got" = "$exp" ]'
  done
done
# identity launch
R1D=notes/swim-tip.txt
for f in "$STD"/R1.pass.md "$STD"/R1.fail-*.md; do
  tag=$(basename "$f" .md); tag=${tag#R1.}; doc=$(cat "$f")
  mk_run "$T/R1-$tag" R1 "$R1D" "$doc" "$(reply_of "$R1D" "$doc")" "$(att1 "$(reply_of "$R1D" "$doc")" 20 4000)"
  out=$(run_rows "$T/R1-$tag" R1)
  if [ "$tag" = pass ]; then
    chk "R1: the rows emitted are exactly the declared v4 and v5 rows" '[ "$(jq -r ".rows[].row | select(test(\"^R1/R1-(SB|CUT|RQ1|D|BE|PRE|DEC|GR|OPR|PIN|MEAS)\$\"))" "$QD" | sort)" = "$(jq -r .row <<<"$out" | sort)" ]'
    chk "R1: the passing synthetic launch passes every row" '[ -n "$out" ] && [ "$(jq -r "select(.verdict != \"PASS\") | .row" <<<"$out" | wc -l)" = 0 ]'
  else
    chk "R1 $tag: exactly R1-RQ1 fails" '[ "$(jq -r "select(.verdict == \"FAIL\") | .row" <<<"$out")" = R1/R1-RQ1 ]'
  fi
done

# ---- product rows: red cases on the passing D1 launch (each case changes one thing; only the named row may fail)
D1DOC=$(cat "$STD/D1.pass.md"); D1R=$(reply_of docs/knife-sharpening.md "$D1DOC")
jqi() { local f=$1; shift; jq "$@" "$f" >"$f.x" && mv "$f.x" "$f"; }
# red RUNDIR EDIT_CMD ROW...: copy the green D1 launch, apply EDIT_CMD inside it, and expect exactly ROW... to fail
red() {
  local name=$1 cmd=$2; shift 2; local d=$T/red-$name
  rm -rf "$d" "$d.launch-v5.json"; cp -R "$T/D1-pass" "$d"; cp "$T/D1-pass.launch-v5.json" "$d.launch-v5.json"
  jqi "$d/steps/S5.json" --arg p "$(cd "$d/ws" && pwd -P)/docs/knife-sharpening.md" '.path = $p'
  (cd "$d" && eval "$cmd")
  local out got exp; out=$(run_rows "$d" D1 "$d.launch-v5.json" "${FZX:-$FZ}")
  got=$(jq -r 'select(.verdict == "FAIL") | .row' <<<"$out" | sed 's#^D1/D1-##' | sort | tr '\n' ' ')
  exp=$(printf '%s\n' "$@" | sort | tr '\n' ' ')
  if [ "$got" = "$exp" ]; then ok; else bad "red $name: wanted FAIL on [$exp], got FAIL on [$got]"; fi
}
S3R=s3-report.json; S8=steps/S8.json
# PRE
red pre-absent        'jqi $S3R "del(.requirements_uncertain)"' PRE
red pre-uncertain     'jqi $S3R ".requirements_uncertain = [\"between 16 and 30 lines\"]"' PRE
# DEC
red dec-null          'jqi $S3R ".proposal_attempts[0].decoding = null"' DEC
red dec-sampled       'jqi $S3R ".proposal_attempts[0].decoding.sampled_tokens = 1"' DEC
red dec-temperature   'jqi $S3R ".proposal_attempts[0].decoding.temperature = 0.7"' DEC
red dec-mixed         'jqi $S8 "(.recall.host[] | select(.id == 100) | .text) |= (fromjson | .decoding.mode = \"mixed\" | tojson)"' DEC
# GR
red gr-null           'jqi $S3R ".generation_record = null"' DEC GR
red gr-text           'jqi $S8 "(.recall.host[] | select(.id == 100) | .text) |= (fromjson | .output_text_sha256 = \"00\" | tojson)"' GR
red gr-index-only     'jqi $S8 "(.recall.host[] | select(.id == 100) | .text) |= (fromjson | .model_sha256 = \"d6c42883a895dfef5b0080ed2116a1bcd764f558406b98923d675978a1abf29c\" | .model_digest_kind = \"index\" | tojson)"' GR
red gr-tokenizer      'jqi $S8 "(.recall.host[] | select(.id == 100) | .text) |= (fromjson | .tokenizer_sha256 = \"00\" | tojson)"' GR
red gr-unverified     'jqi $S8 "(.recall.host[] | select(.id == 100) | .verified) = false"' GR
red gr-not-in-recall  'jqi $S8 ".recall.host |= map(select(.id != 100))"' DEC GR
red gr-commit-prov    'jqi $S8 "(.recall.host[] | select(.id == 101) | .text) |= (fromjson | .provenance.generation_record = 99 | tojson)"' GR
red gr-auth-prov      'jqi $S8 ".authorizations[0].text |= (fromjson | del(.provenance) | tojson)"' GR
red gr-no-shards      'sed -i "/^CHECKPOINT_SHARDS /d" daemon-2.log' GR
red gr-shard-differs  'sed -i "s/0b48adbb/0b48adbc/" daemon-1.log' GR
red gr-daemon-model   'jqi run.json ".daemon[1].model |= sub(\"model_sha256=[0-9a-f]+\"; \"model_sha256=d6c42883a895dfef5b0080ed2116a1bcd764f558406b98923d675978a1abf29c\")"' GR
# OPR
red opr-fallback      'sed -i "s/native_fallbacks=\[\]/native_fallbacks=[matmul_vec:1]/" daemon-1.log' OPR
red opr-reference-run 'sed -i "s/reference_runs=\[\]/reference_runs=[rmsnorm]/" daemon-2.log' OPR
red opr-no-op-report  'sed -i "/^OP_REPORT /d" daemon-1.log daemon-2.log' OPR
red opr-op-missing    'sed -i "s/,rmsnorm_heads\]/]/" daemon-1.log' OPR
red opr-cpu-backend   'sed -i "s/backend=OmegaGb10Backend.*/backend=ReferenceCpuBackend/" daemon-2.log' OPR
red opr-not-strict    'sed -i "s/STRICT strict=true/STRICT strict=false/" daemon-1.log' OPR
red opr-dev-fallback  'sed -i "s/dev_fallback_build=false/dev_fallback_build=true/" daemon-2.log' OPR
red opr-no-strict     'sed -i "/STRICT strict=/d" daemon-1.log' OPR
red opr-two-strict    'printf "%s\n" "$STRICT" >>daemon-1.log' OPR
red opr-violation     'echo "STRICT_REAL_MODEL_VIOLATION op=matmul_vec" >>daemon-2.log' OPR
red opr-log-missing   'rm daemon-2.log' GR OPR
# PIN
red pin-off-by-one    'jqi ../red-pin-off-by-one.launch-v5.json ".pins.aien_cli_sha256 |= (.[0:-1] + \"7\")"' PIN
red pin-no-record     'jqi ../red-pin-no-record.launch-v5.json "del(.pins)"' PIN
FZX=$FPH red pin-placeholder 'jqi ../red-pin-placeholder.launch-v5.json --argjson p "$(jq .pins "$FPH")" ".pins = \$p"' PIN
# MEAS
red meas-sum-over-wall 'jqi ../red-meas-sum-over-wall.launch-v5.json ".wall_ms = 41999"' MEAS
red meas-no-wall      'jqi ../red-meas-no-wall.launch-v5.json "del(.wall_ms)"' MEAS
red meas-no-meminfo   'jqi ../red-meas-no-meminfo.launch-v5.json "del(.meminfo_before.cached_kb)"' MEAS
red meas-no-vmhwm     'jqi run.json ".daemon[0].vmhwm_kb = null"' MEAS
red meas-tokens-ids   'jqi $S3R ".proposal_attempts[0].token_ids |= .[1:]"' MEAS
red meas-no-token-ids 'jqi $S3R "del(.proposal_attempts[0].token_ids)"' MEAS
red meas-over-cap     'jqi $S3R ".proposal_attempts[0] |= (.tokens = 1025 | .token_ids = [range(0; 1025)] | .decoding.greedy_tokens = 1025)"' MEAS
# a refused attempt before the accepted one with no decoding (timed out or errored): DEC FAIL, the other rows unaffected
red dec-earlier-attempt 'jqi $S3R ".proposal_attempts = [{attempt: 0, outcome: \"refused\", reason: \"reply did not parse\", tokens: 5, ms: 1000, finish_reason: \"eos\", token_ids: [0,1,2,3,4], decoding: null}] + .proposal_attempts"' DEC
# EP on E2 is not the only check of the edit: EO still catches a lost line (fixture E2.fail-b above)

# ---- N1 (negative-boundary): the refusal before the model, as recorded at gate G3
N1ERR='RunComposeTask: destination ../shared-stuff/reminder.txt cannot be written safely (not a plain relative path inside the workspace); refusing'
mk_n1() {
  local d=$1; mkdir -p "$d/ws" "$d/steps"
  jq -n --arg e "$N1ERR" '{error: $e, ok: false, step: "propose"}' >"$d/steps/S3.json"
  jq -n --arg ct "$CT" '{ok: true, machine_id: "m1", constraints: [{id: 6, text: $ct, verified: true}], authorizations: [], recall: {host: [{id: 7, note: "effect", verified: true, text: "{\"arguments_digest\":\"x\"}"}]}}' >"$d/steps/S8.json"
  jq -n --arg ct "$CT" '{ok: true, machine_id: "m1", constraints: [{id: 6, text: $ct, verified: true}], authorizations: []}' >"$d/steps/pre-restart-recall.json"
  jq -n --argjson dm "$(mk_daemons)" --arg ct "$CT" '{steps: [range(1; 9) as $i | {step: "S\($i)", name: "s", rc: (if $i >= 3 and $i <= 6 then 1 else 0 end), wall_ms: 10, ok: ($i < 3 or $i > 6)}],
     daemon: $dm, containment: {workspace_changed: []}, constraint_text: $ct}' >"$d/run.json"
  local k; for k in 1 2; do printf '%s\n%s\n' "$STRICT" "$SHARDS" >"$d/daemon-$k.log"; done
  mk_launch "$d" 9000
}
mk_n1 "$T/n1"; out=$(run_rows "$T/n1" N1)
chk "N1: the rows emitted are exactly the declared extra rows (NC BE NR NM RC RH OPR PIN MEAS)" '[ "$(jq -r ".rows[].row | select(test(\"^N1/N1-(NC|BE|NR|NM|RC|RH|OPR|PIN|MEAS)\$\"))" "$QD" | sort)" = "$(jq -r .row <<<"$out" | sort)" ]'
chk "N1: the refusal before the model passes every row (no OP_REPORT line is needed: no model call)" '[ -n "$out" ] && [ "$(jq -r "select(.verdict != \"PASS\") | .row" <<<"$out" | wc -l)" = 0 ]'
n1red() {
  local name=$1 cmd=$2; shift 2; local d=$T/n1-$name
  rm -rf "$d" "$d.launch-v5.json"; cp -R "$T/n1" "$d"; cp "$T/n1.launch-v5.json" "$d.launch-v5.json"
  (cd "$d" && eval "$cmd")
  local got exp; got=$(run_rows "$d" N1 | jq -r 'select(.verdict == "FAIL") | .row' | sed 's#^N1/N1-##' | sort | tr '\n' ' ')
  exp=$(printf '%s\n' "$@" | sort | tr '\n' ' ')
  if [ "$got" = "$exp" ]; then ok; else bad "N1 red $name: wanted FAIL on [$exp], got FAIL on [$got]"; fi
}
n1red uncertain-req   'jqi steps/S3.json ".error = \"RunComposeTask: the goal has an uncertain requirement (just one line that says); refusing\""' NR
n1red other-dest-text 'jqi steps/S3.json ".error |= sub(\"\\\\.\\\\./shared-stuff\"; \"../shared\")"' NR
n1red s3-ok           'jqi steps/S3.json ".ok = true" ; jqi run.json "(.steps[] | select(.step == \"S3\") | .rc) = 0"' NR RC
n1red s3-missing      'rm steps/S3.json' NR
n1red model-attempt   'jq -n "{proposal_attempts: [{attempt: 1, outcome: \"refused\", reason: \"path outside the workspace\", tokens: 12, ms: 3000, finish_reason: \"eos\", token_ids: [range(0; 12)]}]}" >s3-report.json' NM
n1red gen-record      'jqi steps/S8.json ".recall.host += [{id: 100, note: \"generation\", verified: true, text: ({generation: 1} | tojson)}]"' NM
n1red file-written    'mkdir -p shared-stuff && echo hi >shared-stuff/reminder.txt' NM
n1red authorized      'jqi steps/S8.json ".authorizations = [{id: 1, verified: true, text: \"{}\"}]"' NC NM RC
n1red s4-ran          'jqi run.json "(.steps[] | select(.step == \"S4\") | .rc) = 0"' RC
n1red s5-missing      'jqi run.json ".steps |= map(select(.step != \"S5\"))"' RC
n1red ws-changed      'jqi run.json ".containment.workspace_changed = [\"notes/x.txt\"]"' RC
n1red recall-lost     'jqi steps/S8.json ".constraints = []"' RH
n1red pre-unverified  'jqi steps/pre-restart-recall.json ".constraints[0].verified = false"' RH
n1red machine-differs 'jqi steps/S8.json ".machine_id = \"m2\""' RH
n1red no-pre-recall   'rm steps/pre-restart-recall.json' RH
n1red op-report-bad   'printf "%s\n" "OP_REPORT native=[] reference=[rmsnorm] native_fallbacks=[] reference_runs=[rmsnorm:1] backend=ReferenceCpuBackend" >>daemon-1.log' OPR
n1red cpu-start       'printf "%s\n" "  STRICT strict=true dev_fallback_build=false require_checkpoint=true backend=ReferenceCpuBackend" >daemon-2.log' OPR
n1red no-launch-rec   'rm ../n1-no-launch-rec.launch-v5.json' PIN MEAS
# sc#353 review: a step record with no exit code is not a non-zero exit, and a missing S8 recall is not "no record"
n1red rc-absent       'jqi run.json ".steps |= map(del(.rc))"' NR RC
n1red s8-missing      'rm steps/S8.json' NM RH

# ---- N2 (negative-budget): a token-limit cut, nothing committed, the model was asked
mk_n2() {
  local d=$1; mkdir -p "$d/ws" "$d/steps"
  jq -n '{task: "N2", committed: false, compose_commit: null, generation_record: null, requirements_uncertain: [],
     proposal_attempts: [{attempt: 1, outcome: "refused", reason: "reply cut at the token limit after 16 tokens (finish_reason max_tokens); a cut reply is never a proposal",
       tokens: 16, ms: 2100, finish_reason: "max_tokens", text: "x", text_sha256: "x", token_ids: [range(0; 16)], decoding: {greedy_tokens: 16, sampled_tokens: 0}}]}' >"$d/s3-report.json"
  jq -n --arg ct "$CT" '{ok: true, machine_id: "m1", constraints: [{id: 6, text: $ct, verified: true}], authorizations: [], recall: {host: []}}' >"$d/steps/S8.json"
  jq -n --argjson dm "$(mk_daemons)" --arg ct "$CT" '{steps: [range(1; 9) as $i | {step: "S\($i)", name: "s", rc: (if $i >= 3 and $i <= 6 then 1 else 0 end), wall_ms: 10, ok: ($i < 3 or $i > 6)}],
     daemon: $dm, containment: {workspace_changed: []}, constraint_text: $ct}' >"$d/run.json"
  mk_logs "$d"; mk_launch "$d" 9000
}
mk_n2 "$T/n2"; out=$(run_rows "$T/n2" N2)
chk "N2: the rows emitted are exactly the declared extra rows (D NC R BE PRE DEC OPR PIN MEAS)" '[ "$(jq -r ".rows[].row | select(test(\"^N2/N2-(D|NC|R|BE|PRE|DEC|OPR|PIN|MEAS)\$\"))" "$QD" | sort)" = "$(jq -r .row <<<"$out" | sort)" ]'
chk "N2: a greedy token-limit cut with nothing committed passes every row" '[ -n "$out" ] && [ "$(jq -r "select(.verdict != \"PASS\") | .row" <<<"$out" | wc -l)" = 0 ]'
jqi "$T/n2/s3-report.json" '.requirements_uncertain = ["at least twelve paragraphs"] | .proposal_attempts = []'
out=$(run_rows "$T/n2" N2)
chk "N2 refused before the model (uncertain requirement): N2-PRE FAIL, and N2-D, N2-R, N2-DEC FAIL (no attempt)" '[ "$(jq -r "select(.verdict == \"FAIL\") | .row" <<<"$out" | sort | tr "\n" " ")" = "N2/N2-D N2/N2-DEC N2/N2-PRE N2/N2-R " ]'
chk "N2-OPR reads the daemon logs, not the attempts: still PASS" '[ "$(rowres "$out" N2/N2-OPR)" = PASS ]'

# the rows module is always the one beside the scripts, whatever the working directory holds (jq searches the working
# directory for a top-level include before -L): a decoy module that passes everything must change nothing
mkdir -p "$T/decoy"; printf '%s\n' 'def v5_rows: [{row: "DECOY", criterion: "-", threshold: "-", value: null, result: "PASS"}];' 'def v5_end_of_section_ids: [];' >"$T/decoy/rows-oq3-v5.jq"
out=$(cd "$T/decoy" && run_rows "$T/red-opr-fallback" D1 "$T/red-opr-fallback.launch-v5.json")
chk "v5-rows.sh run from a folder with a decoy rows module still uses its own (D1-OPR FAIL on the fallback case)" '[ "$(rowres "$out" D1/D1-OPR)" = FAIL ] && ! grep -q DECOY <<<"$out"'
(cd "$T/decoy" && bash "$HERE/gen-decl-oq3-v5.sh" decl.json)
chk "gen-decl-oq3-v5.sh run from a folder with a decoy rows module (relative OUT) still gives the committed declaration" 'cmp -s "$T/decoy/decl.json" "$QD"'

# a launch whose driver wrote nothing: every row FAIL
mkdir -p "$T/empty/ws"; out=$(run_rows "$T/empty" D1 - )
chk "a launch with no run.json and no files FAILs every row" '[ -n "$out" ] && [ "$(jq -r "select(.verdict == \"PASS\") | .row" <<<"$out" | wc -l)" = 0 ]'

# ---- 5 wrapper refusals (CPU, before any run)
W=$HERE/run-qwen3-v5.sh
PINNED=/home/drakestapleton/workspace/oq3-v5-runs
if [ -e "$PINNED" ]; then bad "pinned run path already exists, wrapper checks skipped"; else
  : >"$T/x"; chmod +x "$T/x"
  base=(env -i PATH="$PATH" HOME="$HOME")
  okenv=(AIEN_BIN="$T/x" AIEN_MODEL_PATH="$T/m.json" AIEN_TOKENIZER_PATH="$T/t.json" AIEN_KV_CONTEXT_TOKENS=4096 AIEN_REQUIRE_BLACKWELL=1 AIEN_REQUIRE_CHECKPOINT=1 AIEN_GB10_QWEN3_DECLARED_ATTEMPT=1)
  wr() { local b=$1; shift; "${base[@]}" "$@" bash "$W" "$b" "$T/out" "$T" "$T" "$T/x" "$T/x" >/dev/null 2>"$T/w.err"; echo $?; }
  chk "wrapper refuses a missing OQ3_PART (exit 2)" '[ "$(wr "$T/b")" = 2 ]'
  chk "wrapper refuses part 5 (exit 2)" '[ "$(wr "$T/b" OQ3_PART=5)" = 2 ]'
  chk "wrapper refuses a run base other than the pinned path (exit 2), nothing created" '[ "$(wr "$T/b" OQ3_PART=1 "${okenv[@]}")" = 2 ] && [ ! -e "$T/b" ]'
  chk "the wrapper as committed (status FROZEN, pins filled) passes the status and pin checks (stops at the model digest check, exit 3), pinned path not created" '[ "$(wr "$PINNED" OQ3_PART=1 "${okenv[@]}")" = 3 ] && [ ! -e "$PINNED" ] && ! grep -q "status is not FROZEN" "$T/w.err" && ! grep -q "pins are not frozen yet" "$T/w.err" && grep -q "digest mismatch" "$T/w.err"'
  # Filled pins alone do not make a real run: the status line must say FROZEN too (a draft freeze-fill PR fills the pins).
  WC=$T/wcopy/oq3; mkdir -p "$WC" "$T/wcopy/next-phase-1"; cp "$W" "$WC/run-qwen3-v5.sh"
  jq --argjson p "$PINS" '.pins = $p' "$FROZEN" >"$WC/frozen-v5.json"
  wrc() { local b=$1; shift; "${base[@]}" "$@" bash "$WC/run-qwen3-v5.sh" "$b" "$T/out" "$T" "$T" "$T/x" "$T/x" >/dev/null 2>"$T/w.err"; echo $?; }
  cp "$FPH" "$WC/frozen-v5.json"
  chk "wrapper refuses a real run while the pins are the placeholder (exit 2), pinned path not created" '[ "$(wrc "$PINNED" OQ3_PART=1 "${okenv[@]}")" = 2 ] && [ ! -e "$PINNED" ] && grep -q "pins are not frozen yet" "$T/w.err"'
  jq --argjson p "$PINS" '.pins = $p' "$FPH" >"$WC/frozen-v5.json"
  chk "wrapper refuses a real run with every pin filled while the status is not FROZEN (exit 2), pinned path not created" '
    [ "$(wrc "$PINNED" OQ3_PART=1 "${okenv[@]}")" = 2 ] && [ ! -e "$PINNED" ] && grep -q "status is not FROZEN" "$T/w.err"'
  chk "wrapper passes the status check once the pins are filled and the status says FROZEN (stops later, pinned path not created)" '
    jq ".status = \"FROZEN (test copy)\"" "$WC/frozen-v5.json" >"$WC/f.tmp" && mv "$WC/f.tmp" "$WC/frozen-v5.json"
    [ "$(wrc "$PINNED" OQ3_PART=1 "${okenv[@]}")" = 2 ] && [ ! -e "$PINNED" ] && ! grep -q "status is not FROZEN" "$T/w.err" && grep -q "declaration, scorer" "$T/w.err"'
  chk "wrapper refuses changed dry-run task shape (exit 2)" '
    jq ".tasks[0].max_tokens = 512" "$TASKS" >"$T/dry-bad.json"
    [ "$(wr "$T/b" OQ3_PART=1 OQ3_DRY_TASKS="$T/dry-bad.json" "${okenv[@]}")" = 2 ] && [ ! -e "$T/b" ]'
  for v in AIEN_KV_CONTEXT_TOKENS=2048 AIEN_REQUIRE_BLACKWELL=0 AIEN_REQUIRE_CHECKPOINT=0 AIEN_GB10_QWEN3_DECLARED_ATTEMPT=0; do
    chk "wrapper refuses $v (exit 3)" '[ "$(wr "$T/b" OQ3_PART=1 OQ3_DRY_TASKS="$TASKS" "${okenv[@]}" "$v")" = 3 ] && [ ! -e "$T/b" ]'
  done
  for v in AIEN_FORCE_CPU_STUB=1 AIEN_DEV_FALLBACK=1 AIEN_COMPOSE_AUTHORIZE_REQUIRES_DESK=0 AIEN_COMPOSE_BUDGET_MS=29000 AIEN_COMPOSE_DOC_MAX_TOKENS=1024 AIEN_OMEGA_SPIN_US=5; do
    chk "wrapper refuses a preset ${v%%=*} (exit 3)" '[ "$(wr "$T/b" OQ3_PART=1 OQ3_DRY_TASKS="$TASKS" "${okenv[@]}" "$v")" = 3 ] && [ ! -e "$T/b" ]'
  done
  chk "wrapper refuses a model index file with a wrong digest (exit 3), base not created" '[ "$(wr "$T/b" OQ3_PART=1 OQ3_DRY_TASKS="$TASKS" "${okenv[@]}")" = 3 ] && [ ! -e "$T/b" ] && grep -q "digest mismatch" "$T/w.err"'
  chk "wrapper reads the model and tokenizer digests and the pins from frozen-v5.json" 'grep -q "jq -r .model.index_sha256 \"\$FROZEN\"" "$W" && grep -q ".model.shards\[\]" "$W" && grep -q "jq -r .model.tokenizer_sha256 \"\$FROZEN\"" "$W" && grep -q "\.pins == \$p" "$W"'
  chk "wrapper records the launch wall time, the memory before the part and the pins in <id>.launch-v5.json" 'grep -q "launch-v5.json" "$W" && grep -q "meminfo_before" "$W" && grep -q "/proc/uptime" "$W"'
  chk "the wrapper lists the launches of the four parts exactly as ACCEPTANCE-v5.md Section 8" '
    grep -q "1) LAUNCHES=\"D1 D2 D3\" ;; 2) LAUNCHES=\"D4 D5 D6\" ;; 3) LAUNCHES=\"E1 E2 N1\" ;; 4) LAUNCHES=\"N2 R1\"" "$W"'
  chk "the wrapper runs the inherited v8 results and v5-rows.sh for every launch" 'grep -q "v8-results.sh\" \"\$id\" \"\$rec\"" "$W" && grep -q "v5-rows.sh\" \"\$BASE/\$id\" \"\$id\"" "$W"'
fi

# ---- 6 the unchanged receipt builder, staged as the wrapper stages it, yields the declared v8-shape rows
NP1=$(cd "$HERE/../next-phase-1" && pwd); STAGE=$T/stage; mkdir -p "$STAGE" "$T/rec"
for f in "$NP1"/*; do n=$(basename "$f"); [ "$n" = tasks-v8.json ] && continue; ln -s "$f" "$STAGE/$n"; done
jq '(.tasks[] | select(.kind != "edit") | .seed) = null' "$TASKS" >"$STAGE/tasks-v8.json"; ln -s "$HERE/seed-v5" "$STAGE/seed-v5"
chk "the wrapper's staging filter is the one used here" 'grep -qF "STAGE_FILTER='"'"'(.tasks[] | select(.kind != \"edit\") | .seed) = null'"'"'" "$W"'
for id in $(jq -r '.tasks[].id' "$TASKS"); do
  R=$T/stg-$id; mkdir -p "$R/ws" "$R/steps" "$R/prov"; echo '{"steps":[],"daemon":[],"containment":{}}' >"$R/run.json"
  seed=$(jq -r --arg i "$id" '.tasks[]|select(.id==$i)|.seed//empty' "$TASKS"); [ -n "$seed" ] && cp -R "$HERE/$seed/." "$R/ws/"
  e=(); [ "${id#N}" = "$id" ] && e=(TASK_ID="$id" TASK_SPEC="$TASKS" TASK_ACCEPTANCE=/dev/null)
  env "${e[@]}" V6_ROWS=rows-v9.jq V6_TASK="$id" V6_MAX_TOKENS=1 V8_MERGE=null bash "$STAGE/make-receipt.sh" "$R" "$T/rec" sc omc omg 0 n >/dev/null 2>&1
  rec=$(ls -t "$T/rec"/*.json | head -1)
  bash "$NP1/v8-results.sh" "$id" "$rec" | jq -r .row | sort >"$T/$id.got"
  case $(jq -r --arg i "$id" ".tasks[]|select(.id==\$i)|.kind" "$TASKS") in
    long) xs="SB|F|CUT|RQ[0-9]+|SRC|D|BE|PRE|DEC|GR|OPR|PIN|MEAS" ;; edit) xs="SB|CUT|EO|EP|D|BE|PRE|DEC|GR|OPR|PIN|MEAS" ;;
    identity) xs="SB|CUT|RQ[0-9]+|D|BE|PRE|DEC|GR|OPR|PIN|MEAS" ;; negative-boundary) xs="NC|BE|NR|NM|RC|RH|OPR|PIN|MEAS" ;;
    *) xs="D|NC|R|BE|PRE|DEC|OPR|PIN|MEAS" ;; esac
  jq -r --arg i "$id" '.rows[].row | select(startswith($i + "/"))' "$QD" | grep -vE "^$id/$id-($xs)\$" | sort >"$T/$id.want"
  if [ "$id" = N1 ]; then
    chk "launch N1: receipt rows == declared v8-shape rows plus the three undeclared extras N1-B, N1-C, N1-H" '[ "$(grep -vE "^N1/N1-(B|C|H)\$" "$T/N1.got")" = "$(cat "$T/N1.want")" ] && [ "$(grep -cE "^N1/N1-(B|C|H)\$" "$T/N1.got")" = 3 ]'
  else
    chk "launch $id: receipt rows == declared v8-shape rows" 'cmp -s "$T/$id.got" "$T/$id.want"'
  fi
done

# ---- 7 spec file statements
chk "ACCEPTANCE-v5.md says FROZEN, NOT RUN" 'grep -q "^\*\*Status: FROZEN, NOT RUN\*\*" "$A"'
chk "Section 7 states the index+shards digest as computed from the listed sha256s, not re-hashed, and the re-hash as a freeze item" 'grep -q "17a78fbba447a4e66a3d886c0998fbcf2f9201d46e5e0c5bfec2d57c975976b7" "$A" && grep -q "COMPUTED FROM LISTED SHA256s, NOT RE-HASHED" "$A" && grep -q "model_digest_kind=index+shards" "$A"'
chk "the old caveat (shards bound by the wrapper pre-run hash only) is gone" '! grep -q "bound by the wrapper.s pre-run hash only" "$A" && ! grep -q "binds the index file of the checkpoint, not the shards" "$A"'
chk "Section 3.4 carries the STRICT start line, the OP_REPORT format and the ten ops" 'grep -qF "STRICT strict=<b> dev_fallback_build=<b> require_checkpoint=<b> backend=<name>" "$A" && grep -qF "OP_REPORT native=[a,b,..] reference=[..] native_fallbacks=[name:count,..] reference_runs=[..] backend=<name>" "$A" && grep -qF "$(tr , " " <<<"$OPS" | sed "s/ /\`, \`/g")" "$A"'
chk "Section 3.4 carries the verbatim N1-NR string" 'grep -qF "$N1ERR" "$A"'
chk "the rows module carries the same STRICT and OP_REPORT forms and the verbatim N1 string" 'grep -qF "^STRICT strict=true dev_fallback_build=false require_checkpoint=true backend=" "$HERE/rows-oq3-v5.jq" && grep -qF "^OP_REPORT native=" "$HERE/rows-oq3-v5.jq" && grep -qF "$N1ERR" "$HERE/rows-oq3-v5.jq"'
chk "ACCEPTANCE-v5.md declares N1-RC and N1-RH and E2-EP" 'grep -q "\`N1-RC\`" "$A" && grep -q "\`N1-RH\`" "$A" && grep -q "\`E2-EP\`" "$A"'
chk "the G3 and G1 evidence files exist" '[ -s "$HERE/evidence-v5/g3-n1-stub-run.txt" ] && [ -s "$HERE/evidence-v5/requirements-probe-b2cae6d.txt" ] && grep -qF "$N1ERR" "$HERE/evidence-v5/g3-n1-stub-run.txt"'
chk "ACCEPTANCE-v5.md states no reruns of failures and no task edits after seeing answers" 'grep -q "no rerun of a failure" "$A" && grep -q "No task edit after seeing answers" "$A"'
chk "no em dash or en dash in the v5 files" '! LC_ALL=C grep -lP "\xe2\x80[\x93\x94]" "$A" "$TASKS" "$FROZEN" "$HERE/rows-oq3-v5.jq" "$HERE/v5-rows.sh" "$HERE/gen-decl-oq3-v5.sh" "$HERE/run-qwen3-v5.sh" "$HERE/test-v5.sh" "$HERE"/selftest-v5/* "$HERE"/evidence-v5/*.txt | grep -q .'
# The v4 files are compared with main once sc#294 (the v4 record) is on main, and with its head 58ff51c until then.
V4REF=origin/main; git -C "$HERE" cat-file -e origin/main:docs/campaigns/open-model-qwen3/rows-oq3-v4.jq 2>/dev/null || V4REF=58ff51c
chk "the v4 and v3 files are untouched by this change (against $V4REF)" '[ -z "$(git -C "$HERE" diff --name-only "$V4REF" -- "$HERE/ACCEPTANCE-v4.md" "$HERE/rows-oq3-v4.jq" "$HERE/v4-rows.sh" "$HERE/run-qwen3-v4.sh" "$HERE/test-v4.sh" "$HERE/tasks-oq3-v4.json" "$HERE/gen-decl-oq3-v4.sh" "$HERE/selftest-v4" "$HERE/rows-oq3-v3.jq" "$HERE/v3-rows.sh" "$HERE/tasks-oq3-v3.json" "$HERE/ACCEPTANCE-v3.md" "$HERE/../next-phase-1")" ]'

echo "test-v5: $pass passed, $fail failed"
[ $fail = 0 ]
