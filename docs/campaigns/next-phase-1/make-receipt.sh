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

# NEXT-PHASE-1 campaign receipt (omega COMPOSITION-2 style) from one
# run-campaign.sh run. Shell + jq only. Usage:
#   make-receipt.sh RUN_ROOT OUT_DIR SC_COMMIT OMEGA_COMPOSE_COMMIT OMEGA_GPU_COMMIT [RESCUES] [NOTE]
# Writes OUT_DIR/<sha256 of content>.json (never edited afterwards), appends
# one line to OUT_DIR/INDEX.md, writes OUT_DIR/<sha256>.summary.txt, and
# prints the summary. Thresholds are the frozen ACCEPTANCE.md rows, unchanged.
set -eu
R=$1 OUT=$2 SC=$3 OMC=$4 OMG=$5 RESCUES=${6:-0} NOTE=${7:-}
S=$R/steps
HERE=$(cd "$(dirname "$0")" && pwd)
# ACCEPTANCE-v7 Section 2: V6_ROWS picks the v6-row module, rows-v6.jq (default,
# every existing caller unchanged), rows-v7.jq (run-v7.sh), rows-v8.jq or rows-v9.jq (open-model-qwen3 v5:
# rows-v8.jq with the harness approval-desk key allowed in the compose dir). Nothing else is accepted.
case ${V6_ROWS:=rows-v6.jq} in rows-v6.jq|rows-v7.jq|rows-v8.jq|rows-v9.jq) ;; *) echo "make-receipt: V6_ROWS must be rows-v6.jq, rows-v7.jq, rows-v8.jq or rows-v9.jq" >&2; exit 2 ;; esac
# ACCEPTANCE-v8 Section 2: only with rows-v8.jq (or rows-v9.jq), V8_MERGE (the np1_edit_merge JSON of an edit launch,
# or unset/null) and the v5 row A2 value reach the evidence the v6 rows see. rows-v6/v7 receipts
# are unchanged byte for byte.
V8M=${V8_MERGE:-null}
jq . <<<"$V8M" >/dev/null 2>&1 || { echo "make-receipt: V8_MERGE is not JSON" >&2; exit 2; }
# ACCEPTANCE-v5: TASK_ID (T1, T2, T3) turns on the v5 rows (rows-v5.jq) for
# that task of tasks-v5.json (TASK_SPEC overrides the file). Without it the
# receipt is the v1..v4 receipt, unchanged.
V5= TASK=null SPEC_OK=false WS= COMMITTED=null
if [ -n "${TASK_ID:-}" ]; then
  V5=1
  TASK=$(jq -c --arg id "$TASK_ID" '.tasks[] | select(.id == $id)' "${TASK_SPEC:-$HERE/tasks-v5.json}")
  [ -n "$TASK" ] || { echo "task $TASK_ID not in ${TASK_SPEC:-$HERE/tasks-v5.json}" >&2; exit 4; }
  row=$(jq -r '"| \(.id) | `\(.goal)` | `\(.destination)` | \(.phrases | map("`" + . + "`") | join(", ")) |"' <<<"$TASK")
  grep -Fqx -- "$row" "${TASK_ACCEPTANCE:-$HERE/ACCEPTANCE-v5.md}" && SPEC_OK=true
  # The daemon canonicalizes the workspace, so S5 paths are physical paths.
  WS=$(cd "$R/ws" && pwd -P)
  s5p=$(jq -r '.path // empty' "$S/S5.json" 2>/dev/null || true)
  if [ -n "$s5p" ] && [ -f "$s5p" ]; then
    real=$(realpath -e "$s5p")
    case "$real" in "$WS"/*) COMMITTED=$(jq -Rs . <"$real") ;; esac
  fi
fi
# ACCEPTANCE-v6: V6_TASK (T4, T5, N1, N2, R1) adds the v6 rows (rows-v6.jq) of
# that launch of tasks-v6.json as `acceptance_v6`. V6_REFERENCE = the CPU
# reference driver's JSON (R1), V6_MAX_TOKENS = the AIEN_COMPOSE_MAX_TOKENS the
# harness passed. TASK_ACCEPTANCE (above) names the file the v5 Q2 task-row
# check reads. The v1..v5 rows and `verdict` are computed as before; without
# V6_TASK the receipt has no v6 field. The v6 containment rows read the v5 row
# A2 result (null when the v5 rows are off). v6 launches are scored by run-v6.sh
# through scoring/score-rows.sh, not by `verdict`.
V6= V6T=null SEED=null REF=null WS6= COMMITTED6=null
if [ -n "${V6_TASK:-}" ]; then
  V6=1
  V6TF=$HERE/tasks-v6.json; case $V6_ROWS in rows-v8.jq|rows-v9.jq) V6TF=$HERE/tasks-v8.json ;; esac   # ACCEPTANCE-v8 Section 3: T6 and T7 live in tasks-v8.json
  V6T=$(jq -c --arg id "$V6_TASK" '.tasks[] | select(.id == $id)' "$V6TF")
  [ -n "$V6T" ] || { echo "launch $V6_TASK not in $V6TF" >&2; exit 4; }
  WS6=$(cd "$R/ws" && pwd -P)
  s5p=$(jq -r '.path // empty' "$S/S5.json" 2>/dev/null || true)
  if [ -n "$s5p" ] && [ -f "$s5p" ]; then
    real=$(realpath -e "$s5p")
    case "$real" in "$WS6"/*) COMMITTED6=$(jq -Rs . <"$real") ;; esac
  fi
  seedrel=$(jq -r '.seed // empty' <<<"$V6T")
  if [ -n "$seedrel" ]; then
    sf=$HERE/$seedrel/$(jq -r .destination <<<"$V6T")
    SEED=$(jq -n --arg p "$(jq -r .destination <<<"$V6T")" --rawfile t "$sf" \
      --arg s "$(sha256sum "$sf" | cut -d' ' -f1)" '{path:$p, text:$t, sha256:$s}')
  fi
  if [ -n "${V6_REFERENCE:-}" ] && [ -s "$V6_REFERENCE" ]; then REF=$(jq -c . "$V6_REFERENCE"); fi
fi
j() { if [ -s "$1" ]; then jq -c . "$1" 2>/dev/null || echo null; else echo null; fi; }
skill_sha=$(printf '%s' 'aien.model.propose-file-change' | sha256sum | cut -d' ' -f1)
receipts=$(for p in "$R"/prov/*.json; do [ -f "$p" ] && jq -c --arg p "$p" --arg s "$(sha256sum "$p" | cut -d' ' -f1)" \
  '{path:$p, sha256:$s, tool, success, arguments_digest, result_digest}' "$p"; done | jq -sc .)
# ACCEPTANCE-v4 2(4): every attempt's full reply, as replies/<sha256>.txt
# (named by its own sha256, which must equal text_sha256; never edited).
mkdir -p "$OUT/replies"
n_att=$(jq '(.report.proposal_attempts // []) | length' "$S/S3.json" 2>/dev/null || echo 0)
for k in $(seq 0 $((n_att - 1))); do
  want=$(jq -r ".report.proposal_attempts[$k].text_sha256 // empty" "$S/S3.json")
  [ -n "$want" ] || continue
  rf=$(mktemp "$OUT/replies/.reply.XXXXXX")
  jq -j ".report.proposal_attempts[$k].text" "$S/S3.json" >"$rf"
  got=$(sha256sum "$rf" | cut -d' ' -f1)
  [ "$got" = "$want" ] || { echo "reply $k: sha256 $got != text_sha256 $want" >&2; rm -f "$rf"; exit 3; }
  mv "$rf" "$OUT/replies/$got.txt"
done
CONCLUSION=; [ -n "${CONCLUSION_FILE:-}" ] && CONCLUSION=$(cat "$CONCLUSION_FILE")
tmp=$(mktemp "$OUT/.receipt.XXXXXX")
jq -n --argjson run "$(j "$R/run.json")" \
  --argjson s1 "$(j "$S/S1.json")" --argjson s2 "$(j "$S/S2.json")" --argjson s3 "$(j "$S/S3.json")" \
  --argjson s4 "$(j "$S/S4.json")" --argjson s5 "$(j "$S/S5.json")" --argjson s6 "$(j "$S/S6.json")" \
  --argjson s8 "$(j "$S/S8.json")" --argjson pre "$(j "$S/pre-restart-recall.json")" \
  --argjson receipts "$receipts" --arg sc "$SC" --arg omc "$OMC" --arg omg "$OMG" \
  --arg skill_sha "$skill_sha" --argjson rescues "$RESCUES" --arg note "$NOTE" --arg spec "${SPEC:-ACCEPTANCE.md spec_version 1}" \
  --arg conclusion "$CONCLUSION" --arg v5 "$V5" --argjson task "$TASK" --argjson spec_ok "$SPEC_OK" \
  --arg ws "$WS" --argjson committed "$COMMITTED" \
  --arg v6 "$V6" --argjson v6task "$V6T" --arg ws6 "$WS6" --argjson committed6 "$COMMITTED6" \
  --argjson seed "$SEED" --argjson ref "$REF" --arg max6 "${V6_MAX_TOKENS:-}" \
  --argjson v8merge "$V8M" --arg rows6 "$V6_ROWS" --arg rows6sha "$(sha256sum "$HERE/$V6_ROWS" | cut -d" " -f1)" \
  "$(cat "$HERE/rows-v5.jq")$(cat "$HERE/$V6_ROWS")"'
  def st($id): ($run.steps // []) | map(select(.step == $id)) | (.[0] // {});
  def okv: if . then "PASS" else "FAIL" end;
  ($s3.report // {}) as $rep
  | ($s1.constraint.id // null) as $cid
  | ($s5.effect.id // null) as $eid
  | ($s8.constraints // []) as $c8
  | ($s8.effects // []) as $e8
  | ($receipts | map(select(.tool == "authorize"))) as $auths
  | ($receipts | map(select(.tool == "write_file"))) as $writes
  | ($receipts | map(select(.tool == "inspect"))) as $inspects
  | {
    S1: (($s1.ok // false) and $cid != null),
    S2: (($s2.ok // false) and ($inspects | length) >= 1),
    S3: (($rep.committed // false) and (($rep.cx_candidates // []) | length) > 0
         and ($rep.cx_evidence // 0) > 0 and ($rep.cx_promotion // 0) > 0),
    S4: (($s4.ok // false) and ($auths | length) == 1
         and ($s4.authorization.id // null) != null),
    S5: (($s5.ok // false) and ($writes | map(select(.success == true)) | length) == 1),
    S6: (($s6.ok // false) and (($s6.cited // []) | length) > 0
         and ((($s6.cited // []) | map(.verified) | all))),
    S7: (st("S7").ok // false),
    S8: (($s8.ok // false) and ($c8 | map(.id) | index($cid)) != null
         and ($e8 | map(.id) | index($eid)) != null
         and ($s8.machine_id // "x") == ($rep.machine_id // "y"))
  } as $reached
  | ($pre.recall.cited // []) as $precited
  | ($s8.recall.cited // []) as $postcited
  | ($c8 | map(select(.id == $cid)) | .[0].text // null) as $recalled
  | ($run.compose_dir_files // []) as $cfiles
  | ($cfiles | map(select(IN("./machine.id", "./cortex.cx", "./jspace", "./jspace/jspace.data", "./jspace/jspace.meta", "./approval-desk.key") | not))) as $stray
  | ($run.daemon // []) as $d
  | (($d | map(.backend // "") | map(test("CPU-reference") | not))) as $gpu
  | [
    {criterion:"Task completion", threshold:"8 of 8 (PROPOSED)",
     value: ([$reached[]] | map(select(.)) | length), reached:$reached,
     result: (([$reached[]] | all) | okv)},
    {criterion:"Correctness: committed content", threshold:"byte-identical, digests equal (PROPOSED)",
     value:{s3_content_sha256:$rep.proposal_content_sha256, s4_content_sha256:(($s8.authorizations // [])[0].text // "{}" | fromjson | .content_sha256),
            s5_disk_sha256:($s5.disk_sha256 // null)},
     result: (($rep.proposal_content_sha256 // "a") == ($s5.disk_sha256 // "b")) | okv},
    {criterion:"Correctness: recall", threshold:"byte-identical (PROPOSED)",
     value:{stored_at_s1:$run.constraint_text, recalled_at_s8:$recalled},
     result: ($recalled != null and $recalled == $run.constraint_text) | okv},
    {criterion:"Latency", threshold:"report only",
     value: (($run.steps // []) | map({step, wall_ms, boottime_ms})), result:"REPORTED"},
    {criterion:"Resource use", threshold:"report only",
     value:{vmhwm_kb_before_restart:$d[0].vmhwm_kb, vmhwm_kb_after_restart:$d[1].vmhwm_kb,
            gpu_used:($gpu | all), backend:($d | map(.backend))}, result:"REPORTED"},
    {criterion:"Human interventions: approvals", threshold:"exactly 1, for the committed effect (PROPOSED)",
     value:{authorize_receipts:($auths | length), authorization_records_at_s8:(($s8.authorizations // []) | length)},
     result: (($auths | length) == 1 and (($s8.authorizations // []) | length) == 1) | okv},
    {criterion:"Human interventions: rescues", threshold:"0 (PROPOSED)", value:$rescues,
     result: ($rescues == 0) | okv},
    {criterion:"Identity", threshold:"identical digest (PROPOSED)",
     value:{before_s7:$rep.machine_id, after_s7:$s8.machine_id, machine_id_file_sha256:$run.machine_id_file_sha256},
     result: (($rep.machine_id // "a") == ($s8.machine_id // "b")
              and ($run.machine_id_file_sha256[0] // "a") == ($run.machine_id_file_sha256[1] // "b")) | okv},
    {criterion:"Memory", threshold:"exists, digest verified, equal to pre-restart digest (PROPOSED)",
     value:{cited:($run.cited), missing_after:($s8.recall.missing // null),
            prefix_records:$run.prefix_records,
            prefix_digest:[($pre.recall.prefix_digest // null), ($s8.recall.prefix_digest // null)]},
     result: ((($run.cited // []) | length) > 0
              and (($s8.recall.missing // [1]) | length) == 0
              and ($postcited | map(.verified) | all)
              and ($precited | map({id, digest})) == ($postcited | map({id, digest}))
              and ($pre.recall.prefix_digest // "a") == ($s8.recall.prefix_digest // "b")) | okv},
    {criterion:"Containment: workspace", threshold:"empty (PROPOSED)",
     value:{outside_new_files:$run.containment.outside_new_files,
            outside_sentinel:[$run.containment.outside_sentinel_before, $run.containment.outside_sentinel_after],
            workspace_tree_sha256:[$run.containment.workspace_before, $run.containment.workspace_after],
            workspace_changed:$run.containment.workspace_changed,
            daemon_state_files:$run.containment.daemon_state_files},
     result: ((($run.containment.outside_new_files // [1]) | length) == 0
              and $run.containment.outside_sentinel_before == $run.containment.outside_sentinel_after
              and ($run.containment.workspace_changed // []) == [($rep.proposal_path // "?")]) | okv},
    {criterion:"Containment: speculation", threshold:"none (PROPOSED)",
     value:{compose_dir_files:$cfiles, other_files:$stray},
     result: (($cfiles | length) > 0 and ($stray | length) == 0) | okv}
  ] as $rows
  | (if $v5 == "1" then v5_evidence($run; $rep; $s4; $s5; $s6; $s8; $receipts; $rescues; $task;
        ($spec_ok and ($run.goal // "") == $task.goal); $ws; $committed) | v5_rows else null end) as $v5rows
  | (($rows + ($v5rows.task_quality // []) + ($v5rows.authority // [])) | map(select(.result == "FAIL")) | length) as $nfail
  | {
    gate:"NEXT_PHASE_1",
    commit:$sc,
    omega_compose_commit:$omc,
    omega_gpu_commit:$omg,
    verdict: (if $nfail == 0 then "PASS" else "FAIL" end),
    machine_id:($rep.machine_id // null),
    tier:"host",
    uname:$run.uname,
    model:($d | map(.model)),
    note:$note,
    acceptance_spec:$spec,
    warm_up_ms:($d | map(.warm_up_ms // null)),
    replies:(($rep.proposal_attempts // []) | map(select(.text_sha256 != null) | {attempt, file:("replies/" + .text_sha256 + ".txt")})),
    conclusion:(if $conclusion == "" then null else $conclusion end),
    observations:{tokens_per_s:(($rep.proposal_attempts // []) | map(select(.tokens > 0 and .ms > 0) | {attempt, tokens, ms, tokens_per_s:((.tokens * 100000 / .ms | floor) / 100)}))},
    skills:[{id:0, name:"aien.model.propose-file-change", version:null, digest:$skill_sha, cost:10}],
    record_digest:[($rep.record_digest // null)],
    prefix_digest:[($pre.recall.prefix_digest // null), ($s8.recall.prefix_digest // null)],
    winner_digest:[($rep.winner_digest // null)],
    acceptance:$rows,
    task_id:($task.id // null),
    acceptance_v5:$v5rows,
    effect_receipts:$receipts,
    explanation:($s6.text // null),
    runs:[{
      run:0, task:$rep.task, outcome:$rep.outcome, committed:$rep.committed,
      old_ref:null, new_ref:null, candidate_refs:null,
      branch_count:$rep.branch_count, branches_reclaimed:$rep.branches_reclaimed,
      winner:$rep.winner, aegis_pass_mask:$rep.aegis_pass_mask,
      goal_crumb:$rep.cx_goal, claims:$rep.cx_candidates, evidence:$rep.cx_evidence,
      promotion:$rep.cx_promotion, admissions:$rep.cx_admissions,
      proposer:$rep.proposer, proposal_path:$rep.proposal_path,
      proposal_sha256:$rep.proposal_sha256, proposal_content_sha256:$rep.proposal_content_sha256,
      uncommitted_proposal:($rep.uncommitted_proposal // null), proposer_error:($rep.proposer_error // null),
      proposal_attempts:($rep.proposal_attempts // []),
      steps: [
        {step:1, name:"S1 remember", result:($reached.S1|okv), detail:{ms:st("S1").wall_ms, constraint:$s1.constraint}},
        {step:2, name:"S2 inspect", result:($reached.S2|okv), detail:{ms:st("S2").wall_ms, tree_sha256:$s2.tree_sha256, receipt:$s2.receipt, effect:$s2.effect.id}},
        {step:3, name:"S3 propose", result:($reached.S3|okv), detail:{ms:st("S3").wall_ms, gpu_used:$gpu[0], backend:$d[0].backend}},
        {step:4, name:"S4 authorize", result:($reached.S4|okv), detail:{ms:st("S4").wall_ms, approvals:$s4.approvals, receipt:$s4.receipt, authorization:$s4.authorization.id}},
        {step:5, name:"S5 execute", result:($reached.S5|okv), detail:{ms:st("S5").wall_ms, path:$s5.path, content_sha256:$s5.content_sha256, disk_sha256:$s5.disk_sha256, receipt:$s5.receipt, effect:$s5.effect.id}},
        {step:6, name:"S6 explain", result:($reached.S6|okv), detail:{ms:st("S6").wall_ms, cited:(($s6.cited // []) | map({id, digest, verified})), receipts:$s6.receipts}},
        {step:7, name:"S7 restart", result:($reached.S7|okv), detail:(st("S7") + {vmhwm_kb_before:$d[0].vmhwm_kb})},
        {step:8, name:"S8 recall", result:($reached.S8|okv), detail:{ms:st("S8").wall_ms, machine_id:$s8.machine_id, constraints:(($s8.constraints // []) | map(.id)), effects:(($s8.effects // []) | map(.id)), vmhwm_kb_after:$d[1].vmhwm_kb, gpu_used:$gpu[1]}}
      ]}]
  }
  + (if $v6 == "1" then {task_v6:$v6task.id,
       acceptance_v6:(v6_evidence($run; $rep; $s1; $s4; $s5; $s6; $s8; $pre; $receipts; $v6task; $ws6; $committed6; $seed; $ref; $max6)
                     + {a2_v5: (($v5rows.authority // []) | map(select(.row == "A2")) | .[0].result // null)}
                     + (if ($rows6 == "rows-v8.jq" or $rows6 == "rows-v9.jq") then {a2_v5_value: (($v5rows.authority // []) | map(select(.row == "A2")) | .[0].value // null), v8_merge: $v8merge} else {} end) | v6_rows)}
     else {} end)
  + (if $v6 == "1" and $rows6 != "rows-v6.jq" then {v6_rows_module:{file:$rows6, sha256:$rows6sha}} else {} end)' >"$tmp"
h=$(sha256sum "$tmp" | cut -d' ' -f1)
mv "$tmp" "$OUT/$h.json"
v=$(jq -r .verdict "$OUT/$h.json")
{
  echo "NEXT_PHASE_1 receipt $h.json verdict $v"
  echo "sovereign-core $SC; omega compose $OMC; omega GPU engine $OMG"
  jq -r '"machine \(.machine_id); backend \(.acceptance[4].value.backend | join(" / ")); gpu_used \(.acceptance[4].value.gpu_used)"' "$OUT/$h.json"
  jq -r '.runs[0].steps[] | "  \(.name): \(.result)  \(.detail.ms // .detail.wall_ms // "-") ms"' "$OUT/$h.json"
  echo "Acceptance (frozen thresholds, ${SPEC:-ACCEPTANCE.md spec_version 1}):"
  jq -r '.acceptance[] | "  \(.result)  \(.criterion)  [\(.threshold)]  \(.value | tojson | .[0:220])"' "$OUT/$h.json"
  jq -r '(.runs[0].proposal_attempts // [])[] | "  proposal attempt \(.attempt): \(.outcome) \(.ms) ms \(.tokens) tokens finish=\(.finish_reason // "-") aegis=\(.aegis // "-") \(.reason // "")"' "$OUT/$h.json"
  echo "Observations:"
  jq -r '.observations.tokens_per_s[] | "  attempt \(.attempt): \(.tokens) tokens in \(.ms) ms = \(.tokens_per_s) tokens/s (wall, incl. prompt prefill)"' "$OUT/$h.json"
  jq -r '"  warm-up (declared, before the task window): \(.warm_up_ms | map(if . == null then "none" else "\(.) ms" end) | join(" / ")) (daemon 1 / daemon 2)"' "$OUT/$h.json"
  jq -r '.replies[] | "  reply attempt \(.attempt): \(.file)"' "$OUT/$h.json"
  if [ -n "$V5" ]; then
    echo "ACCEPTANCE-v5 rows, task $TASK_ID (task quality, then authority):"
    jq -r '(.acceptance_v5.task_quality + .acceptance_v5.authority)[] | "  \(.result)  \(.row) \(.criterion)  \(.value | tojson | .[0:220])"' "$OUT/$h.json"
  fi
  if [ -n "$V6" ]; then
    echo "ACCEPTANCE-v6 rows, launch $V6_TASK:"
    jq -r '.acceptance_v6[] | "  \(.result)  \(.row) \(.criterion)  \(.value | tojson | .[0:220])"' "$OUT/$h.json"
  fi
  if [ -n "$NOTE" ]; then echo "Note: $NOTE"; fi
  if [ -n "$CONCLUSION" ]; then printf '\nConclusion on the frozen model input\n%s\n' "$CONCLUSION"; fi
} >"$OUT/$h.summary.txt"
[ -f "$OUT/INDEX.md" ] || printf '# NEXT-PHASE-1 campaign receipts\n\nEach receipt is named by the sha256 of its content and never edited.\n\n' >"$OUT/INDEX.md"
echo "- \`$h.json\` verdict $v, sovereign-core $SC ($(date -u +%Y-%m-%dT%H:%MZ))" >>"$OUT/INDEX.md"
cat "$OUT/$h.summary.txt"
