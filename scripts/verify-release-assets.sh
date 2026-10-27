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

# Verify a release asset directory: SHA256SUMS.txt exists, carries a valid signature from the allowed
# signer, lists every sovereign-* archive, and every listed file matches its digest.
#   scripts/verify-release-assets.sh <dir> <allowed_signers>
# Fails (exit 1) on a missing or invalid signature, a signer not in the file, a listed file that is missing
# or differs, and an archive that the signed list does not cover.
set -euo pipefail
d="${1:?asset directory}"; signers="${2:?allowed_signers file}"
die() { echo "release assets: $*" >&2; exit 1; }
[[ -f "$d/SHA256SUMS.txt" ]] || die "SHA256SUMS.txt is missing"
[[ -f "$d/SHA256SUMS.txt.sig" ]] || die "SHA256SUMS.txt.sig is missing: refusing an unsigned release"
[[ -s "$signers" ]] || die "allowed signers file $signers is missing or empty"
grep -qvE '^[[:space:]]*(#|$)' "$signers" || die "no key is pinned in $signers (PLACEHOLDER until the key ceremony, docs/release/KEY_CEREMONY.md): refusing"
ssh-keygen -Y verify -f "$signers" -I aien-release -n aien-release -s "$d/SHA256SUMS.txt.sig" < "$d/SHA256SUMS.txt" >/dev/null 2>&1 \
    || die "SHA256SUMS.txt signature is NOT valid for $signers"
n=0
while read -r want name; do
    name="${name#\*}"; [[ -n "$name" ]] || continue
    [[ -f "$d/$name" ]] || die "listed file $name is missing"
    have="$(cd "$d" && { sha256sum "$name" 2>/dev/null || shasum -a 256 "$name"; } | cut -d' ' -f1)"
    [[ "$have" == "$want" ]] || die "$name does not match the signed checksum"
    n=$((n+1))
done < "$d/SHA256SUMS.txt"
[[ "$n" -gt 0 ]] || die "SHA256SUMS.txt lists nothing"
for a in "$d"/sovereign-*; do
    [[ -e "$a" ]] || continue
    grep -qE "^[0-9a-f]{64}  \*?$(basename "$a")\$" "$d/SHA256SUMS.txt" || die "$(basename "$a") is not covered by the signed checksums"
done
echo "release assets verified ($n files)"
