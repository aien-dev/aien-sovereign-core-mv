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

# Stage 3 of the offline-signing handoff (sc#385, docs/release/RELEASE_SIGNING.md): decide whether a draft release may
# be published. Run by .github/workflows/release-publish.yml on the downloaded draft assets, and by hand.
#   scripts/release-publish-gate.sh <asset-dir> <allowed_signers>
# Passes only when ALL hold:
#   - <allowed_signers> pins a key (the PLACEHOLDER refuses) and that key is not the internal test key
#   - nothing in <asset-dir> is a dry-run asset (DRY-RUN-* files, dry-run-signers) and no packaged release.toml says
#     dry-run = true or candidate NOT-RELEASABLE
#   - scripts/verify-release-assets.sh: SHA256SUMS.txt.sig (made offline by the operator) is valid for the pinned key,
#     every listed file matches, every sovereign-* archive is covered
# The private key is never needed here and never present: this only verifies.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
d="${1:?asset directory}"; signers="${2:?allowed_signers file}"
die() { echo "publish gate: REFUSED: $*" >&2; exit 1; }
[[ -d "$d" ]] || die "asset directory $d is missing"
[[ -f "$signers" ]] || die "signer file $signers is missing"
key="$(grep -vE '^[[:space:]]*(#|$)' "$signers" || true)"
[[ -n "$key" ]] || die "no public-release key is pinned in $signers (PLACEHOLDER until the key ceremony)"
[[ "$(printf '%s\n' "$key" | grep -c .)" -eq 1 ]] || die "$signers must pin exactly one key"
internal="$ROOT/docs/release/internal_test_signers"
if [[ -f "$internal" ]]; then
    ik="$(grep -vE '^[[:space:]]*(#|$)' "$internal" | awk '{ print $3 }')"
    [[ -z "$ik" || "$key" != *"$ik"* ]] || die "$signers pins the internal test key, which is not a release key"
fi
for f in "$d"/DRY-RUN-* "$d"/dry-run-signers; do
    [[ ! -e "$f" ]] || die "$(basename "$f") is a dry-run asset; a dry run is never published"
done
# Signature and checksums first, so nothing unverified is unpacked.
bash "$ROOT/scripts/verify-release-assets.sh" "$d" "$signers" || die "signature or checksums do not verify (see above)"
for a in "$d"/sovereign-*.tar.gz; do
    [[ -e "$a" ]] || continue
    m="$(tar -xzOf "$a" ./release.toml 2>/dev/null || true)"
    [[ -n "$m" ]] || die "$(basename "$a") has no release.toml"
    ! grep -qx 'dry-run = true' <<<"$m" || die "$(basename "$a") is labelled dry-run in its release.toml"
    ! grep -qx 'candidate = "NOT-RELEASABLE"' <<<"$m" || die "$(basename "$a") is a NOT-RELEASABLE package"
done
echo "publish gate: PASSED: signed offline by the pinned release key, every asset matches, no dry-run asset"
