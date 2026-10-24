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

# gc.sh: storage quota + cleanup for the aien-run sandbox (lane INFRA). DRY RUN by default; pass --apply to act.
#
#   gc.sh [--apply] [--days N] [--target-quota-gb G]
#
# Policy (documented here, chosen by lane INFRA):
#   * worktrees under /var/lib/aien-run/wt/<repo>/<sha> older than N days (default 3, by mtime of the root dir)
#     are removed and `git worktree prune` is run on the mirror. A worktree some process still has as its cwd
#     (a running check) is skipped.
#   * target dirs /var/lib/aien-run/target/<repo>/{pr,main}/<sha40>: one per head (sc #605), removed with the head's
#     worktree by wt.sh remove. gc removes head dirs whose worktree is gone (a worker died mid-run) and the old
#     shared cache (anything else under pr/ or main/). The per-repo total is reported against G GB (default 120 GB;
#     one sovereign-core all-targets dev build is ~30 GB). Never touches cargo/registry, cargo/git or rustup.
#   * stale trigger files older than 7 days are reported, not deleted (the worker owns them).
set -euo pipefail
RUN_HOME=/var/lib/aien-run
apply=0; days=3; quota_gb=120
while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) apply=1; shift ;;
    --days) days=$2; shift 2 ;;
    --target-quota-gb) quota_gb=$2; shift 2 ;;
    *) echo "usage: gc.sh [--apply] [--days N] [--target-quota-gb G]" >&2; exit 2 ;;
  esac
done
act() { if (( apply )); then "$@"; else printf 'DRY-RUN: %s\n' "$*"; fi; }
now=$(date +%s)

echo "== worktrees older than $days days"
for d in "$RUN_HOME"/wt/*/*/; do
  [[ -d $d ]] || continue
  age=$(( (now - $(stat -c %Y "$d")) / 86400 ))
  (( age >= days )) || continue
  if sudo -n ls -l /proc/[0-9]*/cwd 2>/dev/null | grep -q -- " -> ${d%/}"; then
    echo "skip (in use): $d"; continue
  fi
  repo=$(basename "$(dirname "$d")")
  act sudo -n rm -rf "$d"
  act git -C "$RUN_HOME/mirror/$repo.git" worktree prune
done

echo "== target dirs: one per head (target/<repo>/pr|main/<sha40>), removed with the worktree by wt.sh remove"
# Left behind only when a worker died mid-run: a head dir whose worktree is gone, older than an hour.
# Anything else directly under pr/ or main/ is the old shared cache (before sc #605 every head of a repo built
# in one dir, which let cargo link one head's build into another); nothing reads it any more.
for c in "$RUN_HOME"/target/*/pr "$RUN_HOME"/target/*/main; do
  [[ -d $c ]] || continue
  repo=$(basename "$(dirname "$c")")
  while IFS= read -r -d '' e; do
    n=${e##*/}
    if [[ $n =~ ^[0-9a-f]{40}$ ]]; then
      [[ -d $RUN_HOME/wt/$repo/$n ]] && continue
      (( now - $(stat -c %Y "$e") >= 3600 )) || continue
      echo "orphan head dir: $e"
    else
      echo "old shared cache: $e"
    fi
    act sudo -n rm -rf "$e"
  done < <(sudo -n find "$c" -mindepth 1 -maxdepth 1 -print0)
done

echo "== target quota ${quota_gb} GB per repo (report; head dirs live only as long as their run)"
for r in "$RUN_HOME"/target/*/; do
  [[ -d $r ]] || continue
  used_gb=$(( $(sudo -n du -sk "$r" | cut -f1) / 1024 / 1024 ))
  printf '%s: %s GB%s\n' "$r" "$used_gb" "$( (( used_gb > quota_gb )) && echo ' OVER QUOTA: heads running in parallel, see wt.sh list')"
done

echo "== stale triggers (>7 days, report only)"
find /home/drakestapleton/.local/state/aien-local-ci/trigger -type f -mtime +7 2>/dev/null || true
(( apply )) || echo "(dry run; add --apply to act)"
