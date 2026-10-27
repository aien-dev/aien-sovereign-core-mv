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

# NEXT-PHASE-2 receipt (ACCEPTANCE-v4 section 6 over v3 section 6, superseding v2) from
# one fixture F0 and one `run-faults.sh cases` run root. Shell + jq only. Usage:
#   make-receipt.sh FIX_ROOT RUN_ROOT OUT_DIR SC_COMMIT OMEGA_COMPOSE_COMMIT GPU_BIN CPU_BIN \
#                   BUILD_ENV_JSON LINK_PROOF_JSON [NOTE]
# FIXTURE_KIND=F0v4 (default) or F0v3-fallback (ACCEPTANCE-v4 section 4: the C3
# control is then NOT_RUN). FIX_OLD = the F0v2 root used by C8a (recorded).
# Writes OUT_DIR/<sha256 of content>.json (never edited afterwards), appends one
# line to OUT_DIR/INDEX.md, writes OUT_DIR/<sha256>.summary.txt, prints it.
# Row verdicts: a row PASSes only if every repetition PASSes and its uninjected
# control passed; a failed control or a variant that could not be injected is
# NOT_RUN; NOT_RUN is never PASS. C6d may be NOT_APPLICABLE (its guard), which
# is named in the verdict and never counted as a PASS.
set -eu
FIX=$1 R=$2 OUT=$3 SC=$4 OMC=$5 GPU=$6 CPU=$7 BENV=$8 LPROOF=$9 NOTE=${10:-}
KIND=${FIXTURE_KIND:-F0v4}
old=null
if [ -n "${FIX_OLD:-}" ]; then old=$(jq -c '{sha256:.files_sha256, s3_committed, machine_id, backend}' "$FIX_OLD/fixture.json"); fi
mkdir -p "$OUT"
sha() { sha256sum "$1" | cut -d' ' -f1; }
fixture=$(jq -c --arg fj "$(sha "$FIX/fixture.json")" --arg s3 "$(sha "$FIX/steps/S3.json")" \
  --arg log "$(sha "$FIX/daemon-fixture.log")" \
  '{sha256:.files_sha256, s3_committed, machine_id, proposal_path, backend, daemon_start_ms,
    evidence_digests:[{file:"fixture.json", sha256:$fj}, {file:"steps/S3.json", sha256:$s3}, {file:"daemon-fixture.log", sha256:$log}]}' \
  "$FIX/fixture.json")
tmp=$(mktemp "$OUT/.receipt.XXXXXX")
jq -n --slurpfile runs "$R/results.jsonl" --argjson fixture "$fixture" --arg sc "$SC" --arg omc "$OMC" \
  --arg gpu "$(sha "$GPU")" --arg cpu "$(sha "$CPU")" --arg note "$NOTE" --arg kind "$KIND" \
  --argjson old "$old" --slurpfile benv "$BENV" --slurpfile lproof "$LPROOF" '
  def control_of($row): if ($row | startswith("C3")) then "F0"
    else "control-" + ($row | capture("^(?<c>C[0-9])").c) end;
  ($runs | map(select(.row | startswith("control-")))) as $controls
  | ($fixture.s3_committed == true) as $f0
  | (($fixture.backend // "") | test("OmegaGb10Backend")) as $c3ok
  | [ "C1a","C1b","C2a","C2b","C2c","C2d","C3a","C3b","C4","C5a","C5b","C5c",
      "C6a","C6b","C6c","C6c-ctl","C6d","C6e","C6f","C6g","C6h","C6i",
      "C7a","C7b","C7c","C8a","C8b" ] as $order
  | ($order | map(. as $row
      | ($runs | map(select(.row == $row))) as $reps
      | (control_of($row)) as $c
      | (if $c == "F0" then $f0 else ($controls | map(select(.row == $c)) | (length > 0 and all(.verdict == "PASS"))) end) as $cok
      | {row:$row, control:$c, control_pass:$cok, reps:($reps | length),
         verdict:(if ($reps | length) == 0 or ($cok | not) or ($reps | any(.verdict == "NOT_RUN")) then "NOT_RUN"
                  elif ($reps | all(.verdict == "PASS")) then "PASS"
                  elif $row == "C6d" and ($reps | all(.verdict == "PASS" or .verdict == "NOT_APPLICABLE")) then "NOT_APPLICABLE"
                  else "FAIL" end),
         outcomes:($reps | map(.outcome) | unique)})) as $rowv
  | ($rowv + [{row:"C3 control", control:"F0", control_pass:$f0, reps:1,
      verdict:(if $kind != "F0v4" then "NOT_RUN" elif $f0 and $c3ok then "PASS" else "FAIL" end),
      outcomes:[(if $kind != "F0v4" then "fixture " + $kind + " (pre-registered fallback): " else "" end)
                + "gpu daemon with AIEN_REQUIRE_BLACKWELL=1: " + ($fixture.backend // "none")]}]) as $all
  | {
    gate:"NEXT_PHASE_2",
    commit:$sc,
    omega_compose_commit:$omc,
    acceptance_spec:"docs/campaigns/next-phase-2/ACCEPTANCE-v4.md spec_version 4 (amends ACCEPTANCE-v3.md, which amends ACCEPTANCE-v2.md)",
    builds:[{name:"gpu", sha256:$gpu}, {name:"cpu-fault", sha256:$cpu}],
    build_env:$benv[0],
    link_proof:$lproof[0],
    fixture:($fixture + {kind:$kind}),
    old_install_fixture_c8a:$old,
    verdict:(if $f0 and ($all | all(.verdict == "PASS" or (.row == "C6d" and .verdict == "NOT_APPLICABLE"))) then "PASS" else "FAIL" end),
    not_applicable:($all | map(select(.verdict == "NOT_APPLICABLE") | .row)),
    row_verdicts:$all,
    controls:($controls | map({row, rep, injected_at, outcome, verdict, checks, evidence_digests})),
    rows:($runs | map(select(.row | startswith("control-") | not))
          | map({row, rep, injected_at, outcome, verdict, checks, evidence_digests})),
    not_proved:[
      "runtime loss of the GPU in a GPU-linked build",
      "omega in-settle crash hooks (A12)",
      "caproot revoke on the effect path (A11)",
      "Cortex and J-Space rolled back together (omega known limit)",
      "torn writes below the file level",
      "tamper resistance of the record mark (journal and mark rewritten together, mark deleted then journal cut, compose dir and mark rolled back together)",
      "a real process kill between an append and its mark update (C6c-ctl reproduces the on-disk state only)",
      "J-Space spill damage while compose never spills (C6d)",
      "the Err(e) arm of server.rs (Reconcile: failed:), unreachable in release builds (panic = abort)",
      "the cause of a real start-up panic (C7c proves only that the daemon stops before serving and writes nothing)"],
    note:(if $note == "" then null else $note end)
  }' >"$tmp"
h=$(sha256sum "$tmp" | cut -d' ' -f1)
mv "$tmp" "$OUT/$h.json"
v=$(jq -r .verdict "$OUT/$h.json")
{
  echo "NEXT_PHASE_2 receipt $h.json verdict $v"
  echo "sovereign-core $SC; omega compose $OMC"
  jq -r '.builds[] | "build \(.name) \(.sha256)"' "$OUT/$h.json"
  jq -r '.fixture | "fixture F0 \(.sha256): s3_committed \(.s3_committed), machine \(.machine_id), backend \(.backend)"' "$OUT/$h.json"
  echo "Rows (ACCEPTANCE-v4 section 5 over v3 and v2; injected variants x3, C6i once, control once per case):"
  jq -r '"build_env: \(.build_env | tojson)"' "$OUT/$h.json"
  jq -r '"link_proof: \(.link_proof | tojson | .[0:400])"' "$OUT/$h.json"
  jq -r 'if (.not_applicable | length) > 0 then "NOT_APPLICABLE (not counted as PASS): \(.not_applicable | join(", "))" else empty end' "$OUT/$h.json"
  jq -r '.row_verdicts[] | "  \(.verdict)  \(.row)  (\(.reps) run(s), control \(.control) \(if .control_pass then "PASS" else "FAIL" end))  \(.outcomes | join(" | ") | .[0:200])"' "$OUT/$h.json"
  echo "Failed checks:"
  jq -r '(.controls + .rows)[] | . as $r | .checks[] | select(.ok | not) | "  \($r.row) rep \($r.rep): \(.check) \(.value | tojson | .[0:160])"' "$OUT/$h.json"
  echo "Not proved (declared): $(jq -r '.not_proved | join("; ")' "$OUT/$h.json")"
  if [ -n "$NOTE" ]; then echo "Note: $NOTE"; fi
} >"$OUT/$h.summary.txt"
[ -f "$OUT/INDEX.md" ] || printf '# NEXT-PHASE-2 campaign receipts\n\nEach receipt is named by the sha256 of its content and never edited.\n\n' >"$OUT/INDEX.md"
echo "- \`$h.json\` verdict $v, sovereign-core $SC ($(date -u +%Y-%m-%dT%H:%MZ))" >>"$OUT/INDEX.md"
cat "$OUT/$h.summary.txt"
