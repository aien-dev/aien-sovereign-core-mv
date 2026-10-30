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

# Fail closed unless this tree (or a built package) may be released as the candidate in release/candidate.toml.
#   scripts/check-release-candidate.sh                         tree check; prints "candidate=<id>" on success
#   scripts/check-release-candidate.sh --model F               also write the [model] table to file F
#   scripts/check-release-candidate.sh --package A.tar.gz --native
#                                                              also check a built package (see below)
# Tree checks: candidate is set and not "unknown"; omega.lock equals omega-commit; Cargo.lock holds exactly the
# [pins] revisions (aien-protocols, crumb-spec, spark-crumbs); the [model] table is complete (every digest is 64 hex).
# Package checks: release.toml names this candidate; its [model] table equals the candidate's (model id, safetensors,
# tokenizer, config, fixture digests); aien-cli-sha256 equals the sha256 of bin/aien in the archive; every [files]
# entry matches; --native is REQUIRED (the linux-aarch64 build that links the omega engine): aien-cli-sha256 must equal
# [executables] aien-cli-native-release and every helper binary must equal its sc-<name> entry. A package without --native
# is refused (no candidate digest for that kind). Build recipe: docs/release/BUILD_RECIPE.md.
# Env: AIEN_CANDIDATE_ROOT overrides the tree root (tests).
set -euo pipefail
cd "${AIEN_CANDIDATE_ROOT:-$(dirname "${BASH_SOURCE[0]}")/..}"
M=release/candidate.toml
die() { echo "release gate: $*" >&2; exit 1; }
[[ -f "$M" ]] || die "$M is missing"
MODEL_OUT=""; PKG=""; NATIVE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_OUT="${2:?--model needs a file}"; shift 2 ;;
        --package) PKG="${2:?--package needs an archive}"; shift 2 ;;
        --native) NATIVE=1; shift ;;
        *) die "unknown argument $1" ;;
    esac
done
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
# get <section> <key> <file>: value of key = "value" inside [section] ("" = before the first section)
get() { awk -v s="$1" -v k="$2" '
    /^\[/ { cur=$0; gsub(/[\[\] \t]/,"",cur); next }
    cur==s && match($0, "^" k "[ \t]*=[ \t]*\"") { v=substr($0,RLENGTH+1); sub(/"[ \t]*(#.*)?$/,"",v); print v; exit }' "$3"; }
model_table() { awk '/^\[/ { inm = ($0 ~ /^\[model\]/); next } inm && NF && $0 !~ /^#/ { print }' "$1" | sort; }

cand="$(get "" candidate "$M")"; omega="$(get "" omega-commit "$M")"
[[ -n "$cand" && "$cand" != unknown ]] || die "candidate in $M is missing or 'unknown'"
[[ "$omega" =~ ^[0-9a-f]{40}$ ]] || die "omega-commit in $M is not a full commit hash"
[[ -f omega.lock ]] || die "omega.lock is missing"
lock="$(tr -d '[:space:]' < omega.lock)"
[[ "$lock" == "$omega" ]] || die "omega.lock is $lock but candidate $cand names omega $omega. This tree is not $cand and cannot be released as it."
grep -q '^\[model\]' "$M" || die "no [model] table in $M"
for k in model-id model-safetensors-sha256 tokenizer-json-sha256 config-json-sha256 oracle-fixture-safetensors-sha256 oracle-fixture-manifest-sha256; do
    v="$(get model "$k" "$M")"; [[ -n "$v" ]] || die "[model] $k is missing in $M"
    [[ "$k" == model-id || "$v" =~ ^[0-9a-f]{64}$ ]] || die "[model] $k in $M is not a sha256"
done

# Cargo.lock source pins: each repo must resolve to exactly the candidate's revision.
# crumb-spec and spark-crumbs may use the -mv repo (issue 815). aien-protocols does not:
# this lockfile still carries both aien-protocols and aien-protocols-mv, and folding -mv
# into that pin would report two revisions.
[[ -f Cargo.lock ]] || die "Cargo.lock is missing"
for repo in aien-protocols crumb-spec spark-crumbs; do
    want="$(get pins "$repo" "$M")"
    [[ "$want" =~ ^[0-9a-f]{40}$ ]] || die "[pins] $repo in $M is missing or not a full commit hash"
    case "$repo" in
        crumb-spec|spark-crumbs) repo_re="${repo}(-mv)?" ;;
        *) repo_re="$repo" ;;
    esac
    have="$(grep -E "^source = \"git\+https://github\.com/aien-dev/${repo_re}(\.git)?[?#]" Cargo.lock | grep -oE '#[0-9a-f]{40}"' | cut -c2-41 | sort -u || true)"
    [[ -n "$have" ]] || die "Cargo.lock has no git source for $repo (candidate $cand pins $want)"
    [[ "$(printf '%s\n' "$have" | wc -l)" -eq 1 ]] || die "Cargo.lock has several revisions of $repo: $(echo $have)"
    [[ "$have" == "$want" ]] || die "Cargo.lock consumes $repo $have but candidate $cand pins $want"
done

if [[ -n "$MODEL_OUT" ]]; then sed -n '/^\[model\]/,$p' "$M" | sed '/^$/q' > "$MODEL_OUT"; fi

if [[ -n "$PKG" ]]; then
    [[ -f "$PKG" ]] || die "package $PKG not found"
    X="$(mktemp -d)"; trap 'rm -rf "$X"' EXIT
    tar -xzf "$PKG" -C "$X" || die "package $PKG is not a readable archive"
    R="$X/release.toml"; [[ -f "$R" ]] || die "package has no release.toml"
    [[ "$(get "" schema "$R")" == AienReleaseV1 ]] || die "package release.toml schema is not AienReleaseV1"
    [[ "$(get "" candidate "$R")" == "$cand" ]] || die "package names candidate '$(get "" candidate "$R")', not $cand"
    [[ -z "$(get model model-id "$R")" ]] && die "package release.toml has no [model] table"
    [[ "$(model_table "$R")" == "$(model_table "$M")" ]] || die "package [model] table differs from the candidate's (model id or input digests)"
    want_aien="$(get "" aien-cli-sha256 "$R")"
    [[ "$want_aien" =~ ^[0-9a-f]{64}$ ]] || die "package aien-cli-sha256 is missing or not a sha256"
    [[ -f "$X/bin/aien" && "$(sha "$X/bin/aien")" == "$want_aien" ]] || die "package aien-cli-sha256 is not the digest of bin/aien"
    n=0
    while IFS= read -r line; do
        [[ "$line" =~ ^\"([^\"]+)\"[[:space:]]*=[[:space:]]*\"([0-9a-f]{64})\"$ ]] || die "bad [files] line: $line"
        f="${BASH_REMATCH[1]}"; case "$f" in /*|*..*) die "unsafe path in release.toml: $f" ;; esac
        [[ -f "$X/$f" && "$(sha "$X/$f")" == "${BASH_REMATCH[2]}" ]] || die "package file $f does not match release.toml"
        n=$((n+1))
    done < <(sed -n '/^\[files\]/,$p' "$R" | sed '1d;/^$/d')
    [[ "$n" -gt 0 ]] || die "release.toml lists no files"
    have_n="$(find "$X" -type f ! -name release.toml | grep -c .)"
    [[ "$have_n" -eq "$n" ]] || die "package holds $have_n files but release.toml lists $n"
    # Every shipped executable must be bound to a candidate manifest digest. Only the native build (linux aarch64, omega
    # linked) has digests in [executables]; a package of any other kind has none and is refused, never passed.
    [[ "$NATIVE" -eq 1 ]] || die "no candidate digest for this package kind: $cand has [executables] digests only for the native build (--native); a non-native package cannot be released as $cand"
    exe="$(get executables aien-cli-native-release "$M")"
    [[ "$exe" =~ ^[0-9a-f]{64}$ ]] || die "[executables] aien-cli-native-release missing in $M"
    [[ "$want_aien" == "$exe" ]] || die "aien-cli digest $want_aien is not $cand's aien-cli-native-release $exe (build with scripts/release-build.sh, see docs/release/BUILD_RECIPE.md)"
    for helper in spark-cockpit-rs spark-inquisitor cortex-encoder-rs cortex-rs spark-supervisor spark-debugger; do
        hexe="$(get executables "sc-$helper" "$M")"
        [[ "$hexe" =~ ^[0-9a-f]{64}$ ]] || die "[executables] sc-$helper missing in $M"
        [[ -f "$X/bin/$helper" && "$(sha "$X/bin/$helper")" == "$hexe" ]] || die "package bin/$helper is not $cand's sc-$helper ($hexe)"
    done
fi
echo "candidate=$cand"
