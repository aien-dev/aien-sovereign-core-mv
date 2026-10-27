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

# Scoring contract v5 scorer (docs/campaigns/scoring/SCORING-v5.md). Shell + jq only.
# Usage: score-rows.sh DECLARATION.json RESULTS.jsonl [EXTERNAL.json]
#   DECLARATION  frozen declaration of the case set (see SCORING-v5.md section 2)
#   RESULTS      one JSON object per line, one line per run
#   EXTERNAL     optional JSON object of external controls, e.g. {"F0":true};
#                an external control that is absent counts as false (fail closed)
# Prints one JSON verdict on stdout. Exit 0 only if the campaign verdict is PASS,
# 1 for any other verdict, 2 for a bad declaration, unreadable input or a jq failure.
# Nothing here reads the older make-receipt.sh scorers; they stay as they were.
set -u
DECL=${1:?DECLARATION.json} RES=${2:?RESULTS.jsonl} EXT=${3:-}
[ -f "$DECL" ] && [ -f "$RES" ] || { echo "score-rows: input not found" >&2; exit 2; }
if [ -n "$EXT" ]; then [ -r "$EXT" ] || { echo "score-rows: external file not found" >&2; exit 2; }; else EXT=/dev/null; fi
# Results are read as raw lines so that one bad line is a malformed record, not a crash.
out=$(jq -n --slurpfile decl "$DECL" --rawfile raw "$RES" --slurpfile ext "$EXT" '
  ($decl[0]) as $d
  | ($ext[0] // {}) as $x
  | def isint: type == "number" and . == floor and . >= 1;
    def nonempty: type == "string" and (gsub("\\s"; "") | length) > 0;
  # ---- declaration must be valid, else the whole scoring is an error ----
  ( (($d.rows | map(.row?) + ["F0"]) as $ids2 | if ($d | type) != "object" or $d.contract != "scoring-v5" or ($d.campaign | nonempty | not)
        or ($d.rows | type) != "array" or ($d.rows | length) == 0 then "contract/campaign/rows missing"
    elif ($d.rows | map(.row) | (length != (unique | length))) then "duplicate row id in declaration"
    elif ($d.rows | all(
            (.row | nonempty) and (.reps | isint) and .expected == "PASS"
            and ((.control // null) == null or (.control | type) == "string")
            and ((.not_applicable // false) | type) == "boolean")) | not then "bad row entry (row, reps >= 1, expected PASS)"
    elif ([$d.rows[] | .control | select(. != null and (. as $c | $ids2 | index($c) | not))] | length) > 0
         then "control names an undeclared row"
    else null end) ) as $derr
  | if $derr != null then {contract:"scoring-v5", verdict:"ERROR", declaration_error:$derr, rows:[], extras:[], malformed:[]}
    else
    # ---- parse results line by line ----
    ( $raw | split("\n") | map(select(gsub("\\s"; "") | length > 0)) | to_entries
      | map(. as $e | ($e.value | try fromjson catch null) as $o | {line:($e.key + 1), o:$o}) ) as $lines
    | ($d.rows | map(.row)) as $declared
    | ($lines | map(select((.o | type) != "object" or ((.o.row | type) != "string")
                          ))) as $noid
    | ($lines | map(select((.o | type) == "object" and (.o.row | type) == "string"))) as $withrow
    # a well-formed record: integer rep >= 1, verdict in the enum, NOT_APPLICABLE carries an outcome (the reason)
    | def wellformed: (.rep | isint)
        and (.verdict | IN("PASS","FAIL","NOT_RUN","NOT_APPLICABLE","UNVERIFIED"))
        and (if .verdict == "NOT_APPLICABLE" then (.outcome | nonempty) else true end);
      ($withrow | map(select(.o | wellformed | not))) as $bad
    | ($withrow | map(select(.o | wellformed)) | map(.o + {line:.line})) as $good
    | ($d.rows | map(. as $r
        | ($good | map(select(.row == $r.row))) as $mine
        | ($mine | map(select(.rep <= $r.reps))) as $in
        | ($mine | map(select(.rep > $r.reps)) | map({row, rep, verdict, line})) as $extra
        | ($in | group_by(.rep) | map(select(length > 1) | .[0].rep)) as $dups
        | ($in | map(.rep) | unique) as $have
        | ([range(1; $r.reps + 1)] | map(select(. as $k | $have | index($k) | not))) as $missing
        | ($bad | map(select(.o.row == $r.row)) | map(.line)) as $badlines
        | ($in | map(select(.verdict == "FAIL"))) as $failed
        | ($in | map(select(.verdict == "NOT_APPLICABLE"))) as $na
        | { row:$r.row, role:($r.role // "case"), control:($r.control // null), required_reps:$r.reps,
            counted_reps:($have | length), missing_reps:$missing, duplicate_reps:$dups,
            incomplete_record_lines:$badlines, extra_runs:$extra,
            na_declared:($r.not_applicable // false),
            own:(
              if ($dups | length) > 0 then "ERROR"
              elif ($failed | length) > 0 then "FAIL"
              elif ($na | length) > 0 and (($r.not_applicable // false) | not) then "FAIL"
              elif ($badlines | length) > 0 then "INCOMPLETE"
              elif ($in | any(.verdict == "UNVERIFIED")) then "INCOMPLETE"
              elif ($missing | length) > 0 then (if ($have | length) == 0 then "NOT_RUN" else "INCOMPLETE" end)
              elif ($in | any(.verdict == "NOT_RUN")) then "NOT_RUN"
              elif ($na | length) > 0 then (if ($na | length) == ($in | length) then "NOT_APPLICABLE" else "INCOMPLETE" end)
              else "PASS" end) } )) as $own
    # a control must itself have scored PASS (its declared expected result) for its rows to count
    | ($own | map({(.row): .own}) | add) as $ownmap
    | ($own | map(. as $r
        | (if $r.control == null then true
           elif $r.control == "F0" then ($x.F0 == true)
           else ($ownmap[$r.control] == "PASS") end) as $cok
        | $r + {control_pass:$cok,
                verdict:(if ($r.own == "PASS" or $r.own == "NOT_APPLICABLE") and ($cok | not) then "NOT_RUN" else $r.own end)}
        | del(.own))) as $rows
    | ($d.rows | map(.row)) as $ids
    | ($good | map(select(.row as $k | $ids | index($k) | not)) | map({row, rep, verdict, line})) as $unrequested
    | ( $rows | map(.verdict) ) as $vs
    | { contract:"scoring-v5", campaign:$d.campaign,
        verdict:(if ($noid | length) > 0 then "INCOMPLETE"
                 elif ($vs | any(. == "ERROR")) then "ERROR"
                 elif ($vs | any(. == "FAIL")) then "FAIL"
                 elif ($vs | any(. == "INCOMPLETE")) then "INCOMPLETE"
                 elif ($vs | any(. == "NOT_RUN")) then "NOT_RUN"
                 else "PASS" end),
        rows:$rows,
        not_applicable:($rows | map(select(.verdict == "NOT_APPLICABLE") | .row)),
        extras:($unrequested + ($rows | map(.extra_runs[]))),
        malformed:($noid | map({line, why:"not a JSON object with a string row"})) }
    end' 2>/dev/null) || { echo "score-rows: jq failed (fail closed)" >&2; exit 2; }
printf '%s\n' "$out"
v=$(printf '%s' "$out" | jq -r .verdict 2>/dev/null) || exit 2
[ "$v" = PASS ] && exit 0
exit 1
