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

# Consistency check for the release trust configuration (D3, sc#385 and sc#118). Reads PUBLIC files only.
#   scripts/check-release-trust-config.sh [--root DIR] [--allow-rehearsal]
# Two states are valid:
#   PLACEHOLDER (before the key ceremony): docs/release/allowed_signers holds no key line, install.sh has
#       AIEN_RELEASE_SIGNER='' and there is no docs/release/ceremony-record.txt.
#   PINNED (after the ceremony): allowed_signers holds exactly one 'aien-release ssh-ed25519 <key>' line; install.sh
#       AIEN_RELEASE_SIGNER is that same line; the line is not the internal test key; docs/release/ceremony-record.txt
#       exists, says mode: ceremony, its allowed_signers line equals the pinned line, and its fingerprint equals the
#       fingerprint ssh-keygen computes from the pinned line; no private key material is in any of the three files.
# Anything else exits 1 and says why. --allow-rehearsal accepts a REHEARSAL record (used only by the tests).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; REH=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) ROOT="$(cd "${2:?--root needs a directory}" && pwd)"; shift ;;
        --allow-rehearsal) REH=1 ;;
        *) echo "usage: $0 [--root DIR] [--allow-rehearsal]" >&2; exit 2 ;;
    esac
    shift
done
die() { echo "trust config: REFUSED: $*" >&2; exit 1; }
AS="$ROOT/docs/release/allowed_signers"; REC="$ROOT/docs/release/ceremony-record.txt"; INST="$ROOT/install.sh"
INT="$ROOT/docs/release/internal_test_signers"
[[ -f "$AS" && -f "$INST" ]] || die "allowed_signers or install.sh is missing"
keylines() { grep -vE '^[[:space:]]*(#|$)' "$1" || true; }
pinned="$(sed -n "s/^AIEN_RELEASE_SIGNER='\(.*\)'\$/\1/p" "$INST")"
grep -qE '^AIEN_RELEASE_SIGNER=' "$INST" || die "install.sh has no AIEN_RELEASE_SIGNER line"
line="$(keylines "$AS")"; n="$(printf '%s' "$line" | grep -c . || true)"
for f in "$AS" "$INST" "$REC"; do
    [[ ! -f "$f" ]] || ! grep -q 'PRIVATE KEY' "$f" || die "private key material found in $f"
done
if [[ "$n" -eq 0 ]]; then
    [[ -z "$pinned" ]] || die "install.sh pins a release key but allowed_signers holds none"
    [[ ! -e "$REC" ]] || die "a ceremony record exists but allowed_signers is still the placeholder"
    grep -q PLACEHOLDER "$AS" || die "allowed_signers holds no key and is not labelled PLACEHOLDER"
    echo "trust config: OK (PLACEHOLDER: no public-release key pinned, nothing can verify as a release)"; exit 0
fi
[[ "$n" -eq 1 ]] || die "allowed_signers must pin exactly one key (found $n)"
[[ "$line" =~ ^aien-release\ ssh-ed25519\ [A-Za-z0-9+/]+=*$ ]] || die "the pinned line is not 'aien-release ssh-ed25519 <key>' with nothing else on it"
[[ "$pinned" == "$line" ]] || die "install.sh AIEN_RELEASE_SIGNER differs from the key line in allowed_signers"
if [[ -f "$INT" ]]; then
    ik="$(keylines "$INT" | awk '{ print $3 }')"
    [[ -z "$ik" || "$line" != *"$ik"* ]] || die "the pinned key is the internal test key"
fi
fp="$(printf '%s\n' "$line" | awk '{ print $2" "$3 }' | ssh-keygen -lf - 2>/dev/null | awk '{ print $2 }')"
[[ "$fp" == SHA256:* ]] || die "ssh-keygen cannot read the pinned key"
[[ -f "$REC" ]] || die "a key is pinned but docs/release/ceremony-record.txt is missing"
rv() { sed -n "s/^$1: //p" "$REC" | head -1; }
mode="$(rv mode)"
if [[ "$REH" == 1 && "$mode" == rehearsal ]]; then :; else
    [[ "$mode" == ceremony ]] || die "ceremony record says mode '$mode', not 'ceremony' (a rehearsal key is never a release key)"
    ! grep -qi REHEARSAL "$REC" "$AS" || die "REHEARSAL marker found; a rehearsal key is never a release key"
fi
[[ "$(rv allowed_signers)" == "$line" ]] || die "the ceremony record's allowed_signers line differs from the pinned line"
[[ "$(rv fingerprint)" == "$fp" ]] || die "the ceremony record fingerprint ($(rv fingerprint)) is not the fingerprint of the pinned key ($fp)"
for k in date-utc operator witness; do [[ -n "$(rv $k)" ]] || die "ceremony record has no $k"; done
[[ "$(grep -c 'signs and verifies yes; altered file refused yes' "$REC")" -ge 2 ]] || die "ceremony record does not show two verified backups"
grep -q 'altered file refused yes' "$REC" || die "ceremony record has no public-key verification test"
echo "trust config: OK (PINNED $fp; record: $(rv operator) with witness $(rv witness), $(rv date-utc))"
