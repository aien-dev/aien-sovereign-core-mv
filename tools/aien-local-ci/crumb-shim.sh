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

# crumb (operator shim, installed as ~/.local/bin/crumb on the Spark).
#
# Select authority from the target repository's origin/main, never from its
# working tree or HEAD. A candidate may alter crumb.lock or tools/crumb but
# cannot cause the operator shim to build or execute those sources.
#
# The trusted main's crumb.lock pins an ancestor of aien-architecture
# origin/main. Only the architecture clone itself may use its own main's tool.
# Other trees use the installed sandbox tool. No failed pin falls back.
# CRUMB_SHIM_WHICH=1 prints "<tool> <rule> <root>" without running the tool.
# Location overrides are for offline fixtures; operator installs use trusted
# clone/cache/sandbox paths.
set -u
SANDBOX=${CRUMB_SANDBOX_TOOL:-/usr/local/bin/crumb}
CACHE=${CRUMB_TOOL_CACHE:-$HOME/.local/state/aien-local-ci/crumb-tools}
ARCH=${CRUMB_ARCH_CLONE:-$HOME/workspace/pipeline/clones/aien-architecture}

die() { echo "crumb (shim): $*" >&2; exit 97; }

target=.
for a in "$@"; do
  case $a in -*) continue ;; esac
  if [ -e "$a" ]; then target=$a; break; fi
done
[ -d "$target" ] || target=$(dirname -- "$target")
root=$(git -C "$target" rev-parse --show-toplevel 2>/dev/null || true)
arch_root=$(git -C "$ARCH" rev-parse --show-toplevel 2>/dev/null || true)

build_into() {  # build_into <final-path> <archived trusted tools/crumb>
  local final=$1 src=$2 tmpb
  mkdir -p "$(dirname -- "$final")" || die "cannot create $(dirname -- "$final")"
  make -s -C "$src" crumb >&2 || die "make crumb failed in $src"
  tmpb=$(mktemp "$(dirname -- "$final")/.crumb.XXXXXX") || die "mktemp"
  cp "$src/crumb" "$tmpb" && chmod 0755 "$tmpb" && mv -f "$tmpb" "$final" || { rm -f "$tmpb"; die "install $final"; }
}

archive_tool() { # archive_tool <commit> <cache path>
  local w
  [ -x "$2" ] && return
  w=$(mktemp -d) || die mktemp
  trap 'rm -rf "$w"' EXIT
  git -C "$ARCH" archive "$1" tools/crumb | tar -x -C "$w" || die "git archive $1 tools/crumb"
  [ -f "$w/tools/crumb/Makefile" ] && [ -f "$w/tools/crumb/crumb.c" ] || die "missing tools/crumb at $1"
  build_into "$2" "$w/tools/crumb"
  rm -rf "$w"; trap - EXIT
}

tool=""; rule=""
if [ -n "$root" ]; then
  main=$(git -C "$root" rev-parse --verify -q 'refs/remotes/origin/main^{commit}') \
    || die "no trusted origin/main in $root"
  if lock=$(git -C "$root" show "$main:crumb.lock" 2>/dev/null); then
    pin=$(printf '%s' "$lock" | tr -d '[:space:]' | tr 'A-F' 'a-f')
    [[ $pin =~ ^[0-9a-f]{40}$ ]] || die "origin/main crumb.lock is not one 40-hex commit"
    [ -n "$arch_root" ] || die "no architecture clone at $ARCH"
    git -C "$ARCH" merge-base --is-ancestor "$pin" refs/remotes/origin/main 2>/dev/null \
      || die "origin/main crumb.lock pin $pin is not in architecture origin/main ancestry"
    tool=$CACHE/$pin/crumb; rule="crumb.lock:$pin"
    archive_tool "$pin" "$tool"
  elif [ -n "$arch_root" ] && [ "$root" = "$arch_root" ]; then
    git -C "$ARCH" cat-file -e "$main:tools/crumb/crumb.c" 2>/dev/null \
      && git -C "$ARCH" cat-file -e "$main:tools/crumb/Makefile" 2>/dev/null \
      || die "architecture origin/main has no tools/crumb source"
    tool=$CACHE/main-$main/crumb; rule="architecture-main:$main"
    archive_tool "$main" "$tool"
  fi
fi
if [ -z "$tool" ]; then
  tool=$SANDBOX; rule=sandbox
  [ -x "$tool" ] || die "sandbox tool $tool is missing"
fi

if [ "${CRUMB_SHIM_WHICH:-}" = 1 ]; then printf '%s %s %s\n' "$tool" "$rule" "${root:--}"; exit 0; fi
exec "$tool" "$@"
