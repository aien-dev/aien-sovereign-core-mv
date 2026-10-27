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

# stall-check.sh: finds labelled PRs the desk has been refusing on the same head for too long. Read-only.
#
# Why: the desk only fixes .crumb-only conflicts by itself. A ready-to-merge PR with a real conflict (or a
# trust-surface file, or a failed hosted check) is refused every pass, forever, while the loop still looks
# alive; the lane that wrote the PR has usually exited. On 2026-10-24 and again 2026-10-26/27 the desk ran
# for hours with no merge and every labelled PR refused. Nobody was told.
#
# Reads the desk journals on stdin: train.jsonl ({"utc","repo","pr","head","action","reason"}) and, for merges,
# pipeline.jsonl ({"utc","action":"merge","repo","pr","result":"merged"}). A (repo, pr, head) is STALLED when its
# "blocked" lines span at least --hours, the newest is no older than --fresh minutes (the desk still sees the
# label), and no merge of that PR follows its first block. Prints one JSON line
# per stall (repo, pr, head, since, last, hours, reason: the newest refusal), oldest first.
#
#   cat train.jsonl pipeline.jsonl | stall-check.sh [--hours H] [--fresh M] [--now ISO8601Z]
# Exit: 0 nothing stalled, 1 at least one stall printed, 2 usage or unreadable input.
set -uo pipefail
hours=2 fresh=30 now=
while (($#)); do
  case $1 in
    --hours) hours=${2:-}; shift 2 ;;
    --fresh) fresh=${2:-}; shift 2 ;;
    --now) now=${2:-}; shift 2 ;;
    -h|--help) sed -n '17,30p' "$0"; exit 0 ;;
    *) echo "stall-check: unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ $hours =~ ^[0-9]+([.][0-9]+)?$ && $fresh =~ ^[0-9]+$ ]] || { echo "stall-check: --hours and --fresh take numbers" >&2; exit 2; }
[[ -n $now ]] || now=$(date -u +%FT%TZ)
[[ $now =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]] || { echo "stall-check: --now must look like 2026-10-27T01:40:33Z" >&2; exit 2; }

# Malformed lines are skipped (fromjson?), not fatal: train.jsonl is append-only and a torn last line is normal.
out=$(jq -R -r -n --arg now "$now" --argjson hours "$hours" --argjson fresh "$fresh" '
  ($now | fromdateiso8601) as $t
  | [inputs | fromjson? | select(type == "object" and (.utc | type) == "string")] as $rows
  | ($rows | map(select(.action == "merged" or .result == "merged")) | group_by([.repo, .pr])
     | map({key: (.[0].repo + "#" + .[0].pr), value: (map(.utc) | max)}) | from_entries) as $merged
  | $rows | map(select(.action == "blocked")) | group_by([.repo, .pr, .head])[]
  | {repo: .[0].repo, pr: .[0].pr, head: .[0].head,
     since: (map(.utc) | min), last: (map(.utc) | max), reason: (max_by(.utc).reason)}
  | select(($merged[.repo + "#" + .pr] // "") < .since)
  | ((.last | fromdateiso8601) - (.since | fromdateiso8601)) as $span
  | select($span >= $hours * 3600 and ($t - (.last | fromdateiso8601)) <= $fresh * 60)
  | .hours = (($span / 360 | floor) / 10)
  | [.since, (. | tojson)] | @tsv' 2>/dev/null) || { echo "stall-check: cannot read train.jsonl from stdin" >&2; exit 2; }
[[ -n $out ]] || exit 0
sort <<<"$out" | cut -f2-
exit 1
