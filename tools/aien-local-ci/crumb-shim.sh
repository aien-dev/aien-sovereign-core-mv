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
# Runs the crumb tool that the repository's own CRUMB-VERIFY gate uses, so the person or agent that
# compiles crumbs, the worker's readiness check (aien-test verify runs the bare name "crumb") and the
# gate can never disagree about which tool is right. Root cause it closes (2026-10-18 crumb failures):
# four places picked a crumb tool four different ways, and ~/.local/bin/crumb was a fixed 94fffbcf
# build, so omega/aienos/physics mains (pinned d0f0b95d) and every RFC-0003 Am1 migrated tree read
# "stale" outside their own gate.
#
# Rule, the same order the gates use (tests/gates/run.sh in each repo):
#   1. the tree has crumb.lock      -> the aien-architecture commit it names, built from the trusted
#                                      clone (git archive + make, like the gate), cached by commit
#   2. the tree is aien-architecture -> the tool in that tree (tools/crumb/crumb.c + Makefile), cached
#      (has tools/crumb/crumb.c)        by content hash, like its gate
#   3. anything else                 -> the sandbox tool /usr/local/bin/crumb, which is what
#                                      "${CRUMB:-crumb}" resolves to inside the gate sandbox
# A pin that cannot be built is an error (exit 97). Never a silent fallback: a fallback to some other
# tool is exactly the bug this file exists to stop.
#
# The tree is the git top level of the first argument that names an existing path, else of the
# current directory. Env: CRUMB_SHIM_WHICH=1 prints "<tool> <rule> <root>" and exits 0 (doctor, tests).
# CRUMB_SANDBOX_TOOL, CRUMB_TOOL_CACHE, CRUMB_ARCH_CLONE override locations (tests only).
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

build_into() {  # build_into <final-path> <src-dir containing Makefile+crumb.c>; atomic rename
  local final=$1 src=$2 tmpb
  mkdir -p "$(dirname -- "$final")" || die "cannot create $(dirname -- "$final")"
  make -s -C "$src" crumb >&2 || die "make crumb failed in $src"
  tmpb=$(mktemp "$(dirname -- "$final")/.crumb.XXXXXX") || die "mktemp"
  cp "$src/crumb" "$tmpb" && chmod 0755 "$tmpb" && mv -f "$tmpb" "$final" || { rm -f "$tmpb"; die "install $final"; }
}

tool=""; rule=""
if [ -n "$root" ] && [ -f "$root/crumb.lock" ]; then
  pin=$(tr -d '[:space:]' < "$root/crumb.lock" | tr 'A-F' 'a-f')
  [[ $pin =~ ^[0-9a-f]{40}$ ]] || die "$root/crumb.lock is not one 40-hex commit"
  tool=$CACHE/$pin/crumb; rule="crumb.lock:$pin"
  if [ ! -x "$tool" ]; then
    git -C "$ARCH" cat-file -e "$pin^{commit}" 2>/dev/null || git -C "$ARCH" fetch -q origin 2>/dev/null
    git -C "$ARCH" cat-file -e "$pin^{commit}" 2>/dev/null || die "crumb.lock pin $pin is not in $ARCH"
    w=$(mktemp -d) || die mktemp; trap 'rm -rf "$w"' EXIT
    git -C "$ARCH" archive "$pin" tools/crumb | tar -x -C "$w" || die "git archive $pin tools/crumb"
    build_into "$tool" "$w/tools/crumb"
    rm -rf "$w"; trap - EXIT   # exec below skips EXIT traps
  fi
elif [ -n "$root" ] && [ -f "$root/tools/crumb/crumb.c" ] && [ -f "$root/tools/crumb/Makefile" ]; then
  key=$(cat "$root/tools/crumb/Makefile" "$root/tools/crumb/crumb.c" | sha256sum | cut -c1-16)
  tool=$CACHE/tree-$key/crumb; rule="tree:$key"
  if [ ! -x "$tool" ]; then
    w=$(mktemp -d) || die mktemp; trap 'rm -rf "$w"' EXIT
    cp "$root/tools/crumb/Makefile" "$root/tools/crumb/crumb.c" "$w/" || die "copy tree tool"
    build_into "$tool" "$w"
    rm -rf "$w"; trap - EXIT
  fi
else
  tool=$SANDBOX; rule=sandbox
  [ -x "$tool" ] || die "sandbox tool $tool is missing"
fi

if [ "${CRUMB_SHIM_WHICH:-}" = 1 ]; then printf '%s %s %s\n' "$tool" "$rule" "${root:--}"; exit 0; fi
exec "$tool" "$@"
