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

# The offline-signing handoff end to end, offline, with throwaway keys only (Drake ruling 1, sc#385):
#   1. key ceremony rehearsal (scripts/key-ceremony.sh --rehearsal): key, fingerprint, public-key verification test,
#      two backups that each sign; a corrupted backup is caught by verify-backup
#   2. stage 1 draft (archive + SHA256SUMS.txt, no signature): the publish gate refuses it (unsigned)
#   3. stage 2 operator signs offline with the rehearsal key (key-ceremony.sh sign)
#   4. stage 3 publish gate (scripts/release-publish-gate.sh): passes under the key that signed; refuses under the real
#      docs/release/allowed_signers while it is the PLACEHOLDER, under the internal test key, under another key, after a
#      one-byte change, and for dry-run assets or a dry-run-labelled package even when the signature is valid
# No real key is made, read or needed. Every key lives in a temp folder that is deleted on exit.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="$(mktemp -d "${TMPDIR:-/tmp}/aien-test-offline-signing.XXXXXX")"
trap 'rm -rf "$W"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "ok: $*"; }
KC=(env AIEN_CEREMONY_ASSUME_YES=1 bash "$ROOT/scripts/key-ceremony.sh")
GATE=(bash "$ROOT/scripts/release-publish-gate.sh")

# 1. ceremony rehearsal
"${KC[@]}" ceremony --rehearsal --out "$W/offline" --backup "$W/usb1" --backup "$W/usb2" > "$W/cer.log" 2>&1 \
    || { tail -20 "$W/cer.log" >&2; fail "ceremony rehearsal failed"; }
K="$W/offline/REHEARSAL-aien_release_ed25519"; S="$W/offline/public/allowed_signers"
[[ -f "$K" && -f "$S" && -f "$W/offline/public/ceremony-record.txt" ]] || fail "ceremony outputs missing"
grep -q '^REHEARSAL: throwaway key' "$W/offline/public/ceremony-record.txt" || fail "rehearsal record not labelled REHEARSAL"
grep -q '^# REHEARSAL KEY' "$S" || fail "rehearsal allowed_signers not labelled"
[[ "$(grep -c 'signs and verifies yes' "$W/offline/public/ceremony-record.txt")" -eq 2 ]] || fail "record does not show two verified backups"
! grep -rqs 'PRIVATE KEY' "$W/offline/public" || fail "private key material in the public folder"
grep -q 'altered file is refused: yes' "$W/cer.log" || fail "public-key verification test has no negative row"
pass "ceremony rehearsal: key, witness fingerprint check, verification test, two backups that sign, public folder only public"
"${KC[@]}" verify-backup "$W/usb2/$(basename "$K")" "$S" --rehearsal > "$W/vb.log" 2>&1 || fail "intact backup failed verify-backup"
BK="$W/usb2/$(basename "$K")"; C3="$(sed -n 3p "$BK" | cut -c20)"; [[ "$C3" == A ]] && R=B || R=A
sed -i "3s/^\(.\{19\}\)./\1$R/" "$BK"   # one character of the key body changed
if "${KC[@]}" verify-backup "$W/usb2/$(basename "$K")" "$S" --rehearsal > "$W/vb.log" 2>&1; then fail "corrupted backup passed verify-backup"; fi
pass "verify-backup accepts an intact backup and refuses a corrupted one"
if "${KC[@]}" ceremony --rehearsal --out "$ROOT/docs/release/x" --backup "$W/b1" --backup "$W/b2" > "$W/cer2.log" 2>&1; then
    rm -rf "$ROOT/docs/release/x"; fail "ceremony wrote a key inside a git checkout"
fi
grep -q 'inside a git checkout' "$W/cer2.log" || fail "in-repo refusal not named"
[[ ! -e "$ROOT/docs/release/x" ]] || fail "ceremony left files in the repository"
pass "ceremony refuses to put a key inside a git checkout"
if env CI=true bash "$ROOT/scripts/key-ceremony.sh" ceremony --out "$W/ci" --backup "$W/c1" --backup "$W/c2" < /dev/null > "$W/cer3.log" 2>&1; then
    fail "real-mode ceremony ran under CI"
fi
grep -qE 'CI machine|network route' "$W/cer3.log" || fail "real-mode ceremony on this machine not refused by name"
pass "real (non-rehearsal) mode refuses to run on a CI or networked machine"

# fixture release package (no dry-run label), as stage 1 drafts it
mk_assets() { # dir [dry]
    local d="$1" p="$W/pkg"; rm -rf "$d" "$p"; mkdir -p "$d" "$p/bin"
    printf '#!/bin/sh\necho aien\n' > "$p/bin/aien"
    { echo 'schema = "AienReleaseV1"'; echo 'candidate = "CAND-1"'; [[ -z "${2:-}" ]] || echo 'dry-run = true'; } > "$p/release.toml"
    tar -czf "$d/sovereign-linux-aarch64.tar.gz" -C "$p" .
    (cd "$d" && sha256sum sovereign-* > SHA256SUMS.txt)
}
# 2. stage 1 draft is unsigned: refused
mk_assets "$W/draft"
if "${GATE[@]}" "$W/draft" "$S" > "$W/g.log" 2>&1; then fail "unsigned draft passed the publish gate"; fi
grep -q "SHA256SUMS.txt.sig is missing" "$W/g.log" || fail "unsigned draft not refused by name"
pass "stage 3 refuses an unsigned draft"
# 3. stage 2: the operator signs offline
"${KC[@]}" sign "$K" "$S" "$W/draft/SHA256SUMS.txt" --rehearsal > "$W/s.log" 2>&1 || { cat "$W/s.log" >&2; fail "offline sign failed"; }
[[ -f "$W/draft/SHA256SUMS.txt.sig" ]] || fail "no signature produced"
pass "stage 2 signs offline and verifies before upload"
# 4. stage 3 publish gate
"${GATE[@]}" "$W/draft" "$S" > "$W/g.log" 2>&1 || { cat "$W/g.log" >&2; fail "signed draft refused under the key that signed it"; }
pass "stage 3 passes a draft signed by the pinned key"
if "${GATE[@]}" "$W/draft" "$ROOT/docs/release/allowed_signers" > "$W/g.log" 2>&1; then fail "gate passed under docs/release/allowed_signers"; fi
if ! grep -qvE '^[[:space:]]*(#|$)' "$ROOT/docs/release/allowed_signers"; then
    grep -q "PLACEHOLDER" "$W/g.log" || fail "placeholder refusal not named"
    pass "stage 3 refuses everything while allowed_signers is the PLACEHOLDER"
fi
if "${GATE[@]}" "$W/draft" "$ROOT/docs/release/internal_test_signers" > "$W/g.log" 2>&1; then fail "gate passed under the internal test key"; fi
grep -q "internal test key" "$W/g.log" || fail "internal test key refusal not named"
pass "stage 3 refuses the internal test key as a release key"
ssh-keygen -q -t ed25519 -N '' -f "$W/other"; printf 'aien-release %s\n' "$(cut -d' ' -f1,2 "$W/other.pub")" > "$W/other_signers"
if "${GATE[@]}" "$W/draft" "$W/other_signers" > "$W/g.log" 2>&1; then fail "gate passed under another key"; fi
grep -q "NOT valid" "$W/g.log" || fail "wrong key not refused by name"
pass "stage 3 refuses a signature from a key that is not pinned"
cp -r "$W/draft" "$W/flip"; A="$W/flip/sovereign-linux-aarch64.tar.gz"; SZ=$(wc -c < "$A"); OFF=$((SZ / 2))
B="$(dd if="$A" bs=1 skip="$OFF" count=1 2>/dev/null | od -An -tu1 | tr -d ' ')"
printf "$(printf '\\%03o' $(( (B + 1) % 256 )))" | dd of="$A" bs=1 seek="$OFF" count=1 conv=notrunc 2>/dev/null
if "${GATE[@]}" "$W/flip" "$S" > "$W/g.log" 2>&1; then fail "gate passed an archive with one byte changed"; fi
grep -q "does not match the signed checksum" "$W/g.log" || fail "one-byte change not refused by name (checksum)"
pass "stage 3 refuses an archive with one byte changed (checksum)"
cp -r "$W/draft" "$W/dryfile"; echo "DRY RUN" > "$W/dryfile/DRY-RUN-NOT-A-RELEASE.txt"
if "${GATE[@]}" "$W/dryfile" "$S" > "$W/g.log" 2>&1; then fail "gate passed a draft holding a DRY-RUN file"; fi
grep -q "dry-run asset" "$W/g.log" || fail "dry-run file refusal not named"
mk_assets "$W/drypkg" dry; "${KC[@]}" sign "$K" "$S" "$W/drypkg/SHA256SUMS.txt" --rehearsal > /dev/null 2>&1
if "${GATE[@]}" "$W/drypkg" "$S" > "$W/g.log" 2>&1; then fail "gate passed a dry-run-labelled package with a valid signature"; fi
grep -q "labelled dry-run" "$W/g.log" || fail "dry-run package refusal not named"
pass "stage 3 refuses dry-run assets and dry-run-labelled packages even with a valid signature"
echo "PASS: offline-signing handoff: ceremony rehearsal with verified backups, unsigned draft refused, offline signature verified, publish gate refuses placeholder, internal test key, wrong key, one-byte change and dry runs"
