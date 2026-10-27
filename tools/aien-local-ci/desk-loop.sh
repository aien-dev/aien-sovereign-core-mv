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

# desk-loop.sh: one continuous merge desk for the Spark (2026-10-24, Queen takeover on Drake's word:
# "make it continuous").
#
# Each pass: rebase-train.sh (move ready-to-merge PRs that are behind or conflict only in generated
# .crumb files onto main, recompile crumbs with the pinned tool, push with lease) and then one
# pipeline.sh round (publish signed bundles, merge ready-to-merge PRs that merge cleanly, tag Alpha builds).
# It never merges anything itself: pipeline.sh decides.
#
# One instance only (flock on $STATE/desk-loop.lock). Stops on $STATE/PAUSE or on removing the
# KEEP-RUNNING flag; the cron tick restarts it when the flag is present and nothing is running.
# Interval defaults to 90 s and is a floor, not a sleep loop: the two scripts take as long as they take.
#
# After each pass, stall-check.sh reads train.jsonl and pipeline.jsonl, and every labelled PR refused on the same head for
# AIEN_STALL_HOURS (2) gets one line on the coordination board (AIEN_BOARD, ~/handoffs/BOARD.md), once per
# head ($STATE/stall-alerted). A refresh that changes the head and stalls again is alerted again.
set -u
STATE=${AIEN_STATE:-$HOME/.local/state/aien-local-ci}
L=${AIEN_LIB:-$HOME/.local/lib/aien-local-ci}
INTERVAL=${AIEN_DESK_INTERVAL:-90}
mkdir -p "$STATE"
exec 9>>"$STATE/desk-loop.lock"
flock -n 9 || { echo "desk-loop: already running"; exit 0; }
echo "$$" > "$STATE/desk-loop.pid"
trap 'rm -f "$STATE/desk-loop.pid"' EXIT
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
export GH_DESK_KEY=$(cat "$STATE/DESK.key" 2>/dev/null)
BOARD=${AIEN_BOARD:-$HOME/handoffs/BOARD.md}
# alert_stalls: one board line per newly stalled (repo, pr, head); a failed check only logs, never stops the desk.
alert_stalls() {
  local stalls line key rc
  [[ -r $STATE/train.jsonl ]] || return 0
  stalls=$(cat "$STATE/train.jsonl" "$STATE/pipeline.jsonl" 2>/dev/null | /bin/bash "$L/stall-check.sh" --hours "${AIEN_STALL_HOURS:-2}"); rc=$?
  ((rc == 1)) || { ((rc == 0)) || echo "desk-loop: stall-check exit $rc"; return 0; }
  touch "$STATE/stall-alerted"
  while IFS= read -r line; do
    [[ -n $line ]] || continue
    key=$(jq -r '"\(.repo)#\(.pr)@\(.head)"' <<<"$line") || continue
    grep -qxF "$key" "$STATE/stall-alerted" && continue
    jq -r --arg utc "$(date -u +%FT%TZ)" '"\($utc) desk: STALLED \(.repo)#\(.pr) head \(.head[:8]) refused \(.hours)h since \(.since): \(.reason). The desk cannot fix this; the owner (or a PR-train lane) merges main into the branch, or removes ready-to-merge."' <<<"$line" >> "$BOARD" \
      && echo "$key" >> "$STATE/stall-alerted" && echo "desk-loop: stall alerted $key"
  done <<<"$stalls"
}
while :; do
  [[ -e $STATE/PAUSE ]] && { echo "$(date -u +%FT%TZ) desk-loop: PAUSE present, stopping"; break; }
  [[ -e $STATE/KEEP-RUNNING ]] || { echo "$(date -u +%FT%TZ) desk-loop: KEEP-RUNNING removed, stopping"; break; }
  echo "$(date -u +%FT%TZ) desk-loop: pass start"
  nice -n 10 /bin/bash "$L/rebase-train.sh" || echo "desk-loop: rebase-train exit $?"
  nice -n 10 /bin/bash "$L/pipeline.sh" round || echo "desk-loop: pipeline round exit $?"
  echo "$(date -u +%FT%TZ) desk-loop: pass done"
  alert_stalls
  sleep "$INTERVAL"
done
