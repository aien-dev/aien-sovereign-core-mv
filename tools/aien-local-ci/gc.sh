#!/usr/bin/env bash
# gc.sh: storage quota + cleanup for the aien-run sandbox (lane INFRA). DRY RUN by default; pass --apply to act.
#
#   gc.sh [--apply] [--days N] [--target-quota-gb G]
#
# Policy (documented here, chosen by lane INFRA):
#   * worktrees under /var/lib/aien-run/wt/<repo>/<sha> older than N days (default 3, by mtime of the root dir)
#     are removed and `git worktree prune` is run on the mirror. A worktree some process still has as its cwd
#     (a running check) is skipped.
#   * target dirs /var/lib/aien-run/target/<repo>/{pr,main}: the per-repo total is capped at G GB (default 120 GB,
#     about 1/3 of the free disk on 2026-10-11; one sovereign-core all-targets dev build is ~30 GB). When over quota
#     the `pr` dir is wiped first (shared cache, rebuilt on the next PR), then `main` (A2 says `main` is wiped after
#     any FAIL/timeout anyway). Wiping never touches cargo/registry, cargo/git or rustup (root-owned, read-only).
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

echo "== target quota ${quota_gb} GB per repo"
for r in "$RUN_HOME"/target/*/; do
  [[ -d $r ]] || continue
  used_gb=$(( $(sudo -n du -sk "$r" | cut -f1) / 1024 / 1024 ))
  printf '%s: %s GB\n' "$r" "$used_gb"
  (( used_gb > quota_gb )) || continue
  for sub in pr main; do
    [[ -d $r/$sub ]] || continue
    act sudo -n find "$r/$sub" -mindepth 1 -delete
    # A wiped pr dir is no longer warm (aien-test worker warm standby record).
    if [[ $sub == pr ]]; then act rm -f "$HOME/.local/state/aien-local-ci/warm/$(basename "$r").json"; fi
    (( apply )) || break
    used_gb=$(( $(sudo -n du -sk "$r" | cut -f1) / 1024 / 1024 ))
    (( used_gb > quota_gb )) || break
  done
done

echo "== stale triggers (>7 days, report only)"
find /home/drakestapleton/.local/state/aien-local-ci/trigger -type f -mtime +7 2>/dev/null || true
(( apply )) || echo "(dry run; add --apply to act)"
