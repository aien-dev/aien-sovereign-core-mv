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

# Post-ceremony step (D3, sc#385, sc#118): turn the PUBLIC folder the operator carried out of the ceremony into the
# release trust configuration. Touches exactly three files and reads only the two public ones:
#   docs/release/allowed_signers      the one key line (with a short header)
#   docs/release/ceremony-record.txt  the public record, copied unchanged
#   install.sh                        AIEN_RELEASE_SIGNER='<same key line>'
#   scripts/apply-ceremony-public-record.sh <public-dir> [--root DIR] [--allow-rehearsal]
# <public-dir> is the ceremony's public/ folder (allowed_signers + ceremony-record.txt). Any other file, any private key
# material, a REHEARSAL record (without --allow-rehearsal) or an inconsistent record stops it. Ends by running
# scripts/check-release-trust-config.sh. It never needs, reads or asks for a private key.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; REH=0; PUB=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) ROOT="$(cd "${2:?--root needs a directory}" && pwd)"; shift ;;
        --allow-rehearsal) REH=1 ;;
        *) [[ -z "$PUB" ]] || { echo "one public folder only" >&2; exit 2; }; PUB="$1" ;;
    esac
    shift
done
die() { echo "apply: REFUSED: $*" >&2; exit 1; }
[[ -d "$PUB" ]] || die "give the ceremony's public folder"
[[ "$(ls -A "$PUB" | sort | tr '\n' ' ')" == "allowed_signers ceremony-record.txt " ]] \
    || die "$PUB must hold exactly allowed_signers and ceremony-record.txt"
! grep -rq 'PRIVATE KEY' "$PUB" || die "private key material in $PUB"
line="$(grep -vE '^[[:space:]]*(#|$)' "$PUB/allowed_signers" || true)"
[[ "$(printf '%s' "$line" | grep -c .)" -eq 1 ]] || die "the public allowed_signers must hold exactly one key line"
[[ "$line" =~ ^aien-release\ ssh-ed25519\ [A-Za-z0-9+/]+=*$ ]] || die "unexpected key line format"
if [[ "$REH" == 0 ]]; then
    ! grep -qi REHEARSAL "$PUB/allowed_signers" "$PUB/ceremony-record.txt" || die "this is a REHEARSAL record, not a ceremony"
fi
{
    echo "# AIEN public-release key (ssh-keygen allowed_signers format, principal aien-release)."
    echo "# Pinned from the witnessed key ceremony; the public record is docs/release/ceremony-record.txt."
    echo "# Only this one public key may ever be listed here. The private half is offline and is not in this repository."
    [[ "$REH" == 0 ]] || echo "# REHEARSAL KEY (test only)"
    echo "$line"
} > "$ROOT/docs/release/allowed_signers"
cp "$PUB/ceremony-record.txt" "$ROOT/docs/release/ceremony-record.txt"
sed -i "s|^AIEN_RELEASE_SIGNER=.*|AIEN_RELEASE_SIGNER='$line'|" "$ROOT/install.sh"
args=(--root "$ROOT"); [[ "$REH" == 0 ]] || args+=(--allow-rehearsal)
bash "$ROOT/scripts/check-release-trust-config.sh" "${args[@]}"
