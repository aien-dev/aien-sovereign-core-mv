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

# wt.sh: trusted-side worktree management for aien-run checks (A2: worktrees under /var/lib/aien-run/wt,
# the clone unwritable by aien-run, the worktree's .git link unwritable by aien-run).
#
#   wt.sh sync    <repo> <trusted-clone-dir>   create/refresh the bare mirror /var/lib/aien-run/mirror/<repo>.git
#   wt.sh add     <repo> <sha>                 check out <sha> at /var/lib/aien-run/wt/<repo>/<sha>/<repo> for aien-run
#   wt.sh sibling <repo> <sha40> <dir> <source-repo> <source-sha>
#                                              read-only checkout of <source-repo> at <source-sha> beside it, at
#                                              /var/lib/aien-run/wt/<repo>/<sha40>/<dir> (what ../<dir> means there)
#   wt.sh remove  <repo> <sha>                 remove /var/lib/aien-run/wt/<repo>/<sha> with every sibling in it,
#                                              and the head's target dirs /var/lib/aien-run/target/<repo>/pr|main/<sha>
#   wt.sh list                                 list run directories with age
#
# Why a mirror: the operator's clones live under ~/workspace (0700), which aien-run must not read. The mirror is a
# bare copy owned by the operator (0755, world-readable, not writable by aien-run); the worktree's .git file points
# into the mirror, so aien-run can read objects (git log/rev-parse work) but cannot write refs, objects or the index.
# Layout (SANDBOX lane, 2026-10-11): one run directory per head, /var/lib/aien-run/wt/<repo>/<sha40>/, owned by the
# operator, mode 0755 (aien-run cannot create entries in it). It holds the checkout under test at <repo>/ and the
# sibling checkouts its gates reach as ../<dir>. The exec helper takes --cwd .../<sha40>/<repo> and makes the run
# directory the one writable worktree mount; file ownership inside it decides what aien-run may write.
# Ownership after `add`: checkout dir operator:aien-run mode 1775 (sticky: aien-run may create files but cannot
# remove or replace entries it does not own), `.git` file root:root 0644, everything else aien-run:aien-run.
# Ownership after `sibling`: everything operator-owned, no group/other write: aien-run reads it (and its git history
# through the mirror) but cannot change it.
set -euo pipefail
RUN_USER=aien-run
BASE=/var/lib/aien-run
MIRROR="$BASE/mirror"
WT="$BASE/wt"
OPERATOR=$(id -un)

die() { printf 'wt.sh: %s\n' "$*" >&2; exit 2; }
name_ok() { [[ $1 =~ ^[A-Za-z0-9_-][A-Za-z0-9._-]*$ ]]; }
cmd=${1:-}; shift || true
case "$cmd" in
  sync)
    repo=${1:?repo}; clone=${2:?trusted-clone-dir}
    name_ok "$repo" || die "bad repo name"
    git -C "$clone" rev-parse --git-dir >/dev/null || die "not a git dir: $clone"
    m="$MIRROR/$repo.git"
    if [[ ! -d $m ]]; then
      git init --bare --quiet "$m"
      git -C "$m" config core.sharedRepository 0644
    fi
    git -C "$m" fetch --quiet --prune "$clone" '+refs/heads/*:refs/heads/*' '+refs/remotes/*:refs/remotes/*' '+refs/tags/*:refs/tags/*'
    # Gates may read the origin URL (the sc tier-1 test f62 does, as on a GitHub runner). The mirror takes the
    # trusted clone's URL; nothing fetches through it (the sandbox has no network and cannot write the mirror).
    if u=$(git -C "$clone" remote get-url origin 2>/dev/null); then git -C "$m" config remote.origin.url "$u"; fi
    chmod -R o+rX,o-w "$m"
    printf 'mirror %s: %s refs\n' "$m" "$(git -C "$m" for-each-ref | wc -l)"
    ;;
  add)
    repo=${1:?repo}; sha=${2:?sha}
    name_ok "$repo" && [[ $sha =~ ^[0-9a-f]{7,40}$ ]] || die "bad repo or sha"
    m="$MIRROR/$repo.git"; [[ -d $m ]] || die "no mirror for $repo (run: wt.sh sync $repo <clone>)"
    full=$(git -C "$m" rev-parse --verify "$sha^{commit}") || die "unknown commit $sha"
    c="$WT/$repo/$full"; d="$c/$repo"
    [[ ! -e $c ]] || die "exists: $c"
    install -d -m 0755 "$WT/$repo" "$c"
    git -C "$m" worktree add --quiet --detach "$d" "$full"
    sudo -n chown -R "$RUN_USER:$RUN_USER" "$d"
    sudo -n chown root:root "$d/.git"; sudo -n chmod 0644 "$d/.git"
    sudo -n chown "$OPERATOR:$RUN_USER" "$d"; sudo -n chmod 1775 "$d"
    printf '%s\n' "$d"
    ;;
  sibling)
    repo=${1:?repo}; sha=${2:?sha40}; dir=${3:?dir}; src=${4:?source-repo}; ssha=${5:?source-sha}
    name_ok "$repo" && name_ok "$dir" && name_ok "$src" && [[ $sha =~ ^[0-9a-f]{40}$ && $ssha =~ ^[0-9a-f]{7,40}$ ]] \
      || die "bad repo, dir, source or sha"
    [[ $dir != "$repo" ]] || die "a sibling cannot take the checkout's own name: $dir"
    c="$WT/$repo/$sha"; [[ -d $c/$repo && ! -L $c ]] || die "no checkout at $c/$repo (run: wt.sh add $repo $sha)"
    s="$c/$dir"; [[ ! -e $s ]] || die "exists: $s"
    m="$MIRROR/$src.git"; [[ -d $m ]] || die "no mirror for $src (run: wt.sh sync $src <clone>)"
    full=$(git -C "$m" rev-parse --verify "$ssha^{commit}") || die "unknown commit $ssha in $src"
    git -C "$m" worktree add --quiet --detach "$s" "$full"
    chmod -R a+rX,go-w "$s"
    printf '%s %s\n' "$s" "$full"
    ;;
  remove)
    repo=${1:?repo}; sha=${2:?sha}
    name_ok "$repo" && [[ $sha =~ ^[0-9a-f]{7,40}$ ]] || die "bad repo or sha"
    d=$(ls -d "$WT/$repo/$sha"* 2>/dev/null | head -1); [[ -n $d ]] || die "no worktree for $sha"
    sudo -n rm -rf "$d"
    # The head's own cargo target dirs (one per head and trust class, see aien-run-exec) go with it: nothing
    # reuses them, and a dev build of sovereign-core is ~30 GB.
    full=${d##*/}
    if [[ $full =~ ^[0-9a-f]{40}$ ]]; then
      for t in "$BASE/target/$repo/pr/$full" "$BASE/target/$repo/main/$full"; do
        if [[ -d $t ]]; then sudo -n rm -rf "$t"; printf 'removed %s\n' "$t"; fi
      done
    fi
    # The checkout and its siblings were worktrees of (possibly) several mirrors.
    for m in "$MIRROR"/*.git; do [[ -d $m ]] && git -C "$m" worktree prune; done
    printf 'removed %s\n' "$d"
    ;;
  list)
    for d in "$WT"/*/*/; do
      [[ -d $d ]] || continue
      printf '%s  %s days\n' "$d" "$(( ( $(date +%s) - $(stat -c %Y "$d") ) / 86400 ))"
    done
    ;;
  *) sed -n '2,12p' "$0" >&2; exit 2 ;;
esac
