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

# Public-key integration test (D3, sc#385 and sc#118). Proves, from ONLY the public folder a ceremony produces
# (allowed_signers line + ceremony-record.txt), that the post-ceremony trust-config pull request is correct:
#   (a) a release signed by the matching key installs with the real install.sh as signature_check: release, and the
#       publish gate passes it
#   (b) a release signed by a different key is refused by install.sh ("signature is NOT valid") and by the publish gate
#   (c) a tampered archive (checksum) and a tampered checksum list (signature) are refused and install nothing
#   (d) the trust-config change is a reviewable diff: one allowed_signers key line, one AIEN_RELEASE_SIGNER line,
#       one new public record file, no private material
#   (e) the record's fingerprint matches the pinned line, and each kind of mismatch is refused (mutants)
# The "ceremony" is the --rehearsal mode of scripts/key-ceremony.sh: a throwaway key made and deleted inside this test.
# No production key is made, read or needed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="$(mktemp -d "${TMPDIR:-/tmp}/aien-test-pubkey.XXXXXX")"
trap 'rm -rf "$W"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "ok: $*"; }
case "$(uname -s)" in Linux) OS=linux ;; Darwin) OS=macos ;; *) fail "unsupported OS" ;; esac
case "$(uname -m)" in x86_64|amd64) ARCH=x86_64 ;; aarch64|arm64) ARCH=aarch64 ;; *) fail "unsupported arch" ;; esac
[[ "$OS-$ARCH" != macos-aarch64 ]] || ARCH=arm64
ASSET="sovereign-$OS-$ARCH.tar.gz"
KC=(env AIEN_CEREMONY_ASSUME_YES=1 bash "$ROOT/scripts/key-ceremony.sh")

# The rehearsal ceremony. From here on only $PUB (public) is used to configure trust; the private key only signs.
"${KC[@]}" ceremony --rehearsal --out "$W/offline" --backup "$W/u1" --backup "$W/u2" > "$W/cer.log" 2>&1 || { tail -20 "$W/cer.log" >&2; fail "rehearsal ceremony failed"; }
PUB="$W/offline/public"; KEY="$W/offline/REHEARSAL-aien_release_ed25519"
ssh-keygen -q -t ed25519 -N '' -C other -f "$W/other"

# A scratch copy of the repository files that matter, under git, so the change is a real diff.
T="$W/tree"; mkdir -p "$T/docs/release" "$T/scripts"
cp "$ROOT/install.sh" "$T/"; cp "$ROOT/docs/release/allowed_signers" "$ROOT/docs/release/internal_test_signers" "$T/docs/release/"
cp "$ROOT"/scripts/*.sh "$T/scripts/"
git -C "$T" init -q; git -C "$T" add -A; git -C "$T" -c user.name=t -c user.email=t@t commit -qm base
bash "$T/scripts/check-release-trust-config.sh" > "$W/o.log" || fail "baseline (placeholder) rejected"
grep -q PLACEHOLDER "$W/o.log" || fail "baseline not reported as placeholder"
pass "baseline: placeholder state is consistent"

# Negative control: the rehearsal record must not be accepted as a real ceremony.
if bash "$T/scripts/apply-ceremony-public-record.sh" "$PUB" --root "$T" > "$W/o.log" 2>&1; then fail "rehearsal record applied without --allow-rehearsal"; fi
grep -q 'REHEARSAL' "$W/o.log" || fail "rehearsal refusal not named"
git -C "$T" diff --quiet && [[ -z "$(git -C "$T" status --porcelain)" ]] || fail "refused apply still changed the tree"
pass "a REHEARSAL record is refused as a release key (negative control), tree untouched"
mkdir "$W/stray"; cp "$PUB"/* "$W/stray/"; touch "$W/stray/aien_release_ed25519"
if bash "$T/scripts/apply-ceremony-public-record.sh" "$W/stray" --root "$T" --allow-rehearsal > "$W/o.log" 2>&1; then fail "apply accepted a public folder holding an extra file"; fi
pass "a public folder with any extra file (for example a key file) is refused"

# (d) the post-ceremony change
bash "$T/scripts/apply-ceremony-public-record.sh" "$PUB" --root "$T" --allow-rehearsal > "$W/o.log" 2>&1 || { cat "$W/o.log" >&2; fail "apply failed"; }
git -C "$T" add -A
[[ "$(git -C "$T" diff --cached --name-only | sort | tr '\n' ' ')" == "docs/release/allowed_signers docs/release/ceremony-record.txt install.sh " ]] || fail "change touches files other than the three expected"
[[ "$(git -C "$T" diff --cached -U0 install.sh | grep -c '^[-+][^-+]')" -eq 2 ]] || fail "install.sh change is not exactly one line replaced"
git -C "$T" diff --cached -U0 install.sh | grep -q "^+AIEN_RELEASE_SIGNER='aien-release ssh-ed25519 " || fail "install.sh change is not the AIEN_RELEASE_SIGNER line"
[[ "$(git -C "$T" diff --cached -U0 docs/release/allowed_signers | grep '^+[^+]' | grep -vc '^+#')" -eq 1 ]] || fail "allowed_signers change is not exactly one key line"
[[ "$(git -C "$T" diff --cached -U0 docs/release/allowed_signers | grep '^-[^-]' | grep -vc '^-#')" -eq 0 ]] || fail "allowed_signers change removes a key line"
! git -C "$T" diff --cached | grep -q 'PRIVATE KEY' || fail "private key material in the diff"
cmp -s "$PUB/ceremony-record.txt" "$T/docs/release/ceremony-record.txt" || fail "record in the tree is not the ceremony's public record"
pass "(d) trust-config change = one allowed_signers key line + one AIEN_RELEASE_SIGNER value + the public record, no private material"

# (e) fingerprint match and mutants
CHK=(bash "$T/scripts/check-release-trust-config.sh" --root "$T" --allow-rehearsal)
"${CHK[@]}" > "$W/o.log" 2>&1 || { cat "$W/o.log" >&2; fail "consistent tree rejected"; }
FP="$(sed -n 's/^fingerprint: //p' "$PUB/ceremony-record.txt")"
[[ "$FP" == "$(awk '{ print $2" "$3 }' "$T/docs/release/allowed_signers" | grep -v '^#' | ssh-keygen -lf - | awk '{ print $2 }' | head -1)" ]] || fail "record fingerprint is not the pinned key's fingerprint"
if bash "$T/scripts/check-release-trust-config.sh" --root "$T" > "$W/o.log" 2>&1; then fail "rehearsal tree passed the strict check"; fi
pass "(e) record fingerprint $FP equals the pinned line; strict mode refuses the rehearsal tree"
mutant() { # name, then commands run in a fresh copy of the pinned tree; the copy must differ; the checker must refuse
    local name="$1" M="$W/m-$1"; shift
    rm -rf "$M"; cp -r "$T" "$M"; ( cd "$M"; "$@" ) || fail "mutant $name could not be built"
    # A mutant that leaves the tree byte-identical is accepted by the checker for the right reason and would report
    # a false failure (sc#568). A no-op mutant is a bug in this test, so refuse to run it at all.
    if diff -rq -x .git "$T" "$M" > /dev/null; then fail "mutant $name is a no-op: the mutated tree is identical to the pinned tree"; fi
    if bash "$M/scripts/check-release-trust-config.sh" --root "$M" --allow-rehearsal > "$W/o.log" 2>&1; then fail "mutant refused? no: $name passed the checker"; fi
    pass "mutant refused: $name ($(sed -n 's/.*REFUSED: //p' "$W/o.log" | cut -c1-70))"
}
OTHERLINE="aien-release $(cut -d' ' -f1,2 "$W/other.pub")"
# The fingerprint is random per run, so replacing its first character with a fixed one was a no-op about 1 in 64
# runs (sc#568). Swap it to X unless it already is X, then to Y: the value always changes.
mutant wrong-fingerprint-in-record sed -i -E 's/^(fingerprint: SHA256:)X/\1Y/; t; s/^(fingerprint: SHA256:)./\1X/' docs/release/ceremony-record.txt
mutant other-key-in-allowed_signers sed -i "s|^aien-release ssh-ed25519 .*|$OTHERLINE|" docs/release/allowed_signers
mutant installer-pin-differs sed -i "s|^AIEN_RELEASE_SIGNER=.*|AIEN_RELEASE_SIGNER='$OTHERLINE'|" install.sh
mutant installer-pin-empty sed -i "s|^AIEN_RELEASE_SIGNER=.*|AIEN_RELEASE_SIGNER=''|" install.sh
mutant record-missing rm docs/release/ceremony-record.txt
mutant two-keys-pinned sh -c "echo '$OTHERLINE' >> docs/release/allowed_signers"
mutant internal-test-key-pinned sh -c "l=\$(grep -v '^#' docs/release/internal_test_signers); sed -i \"s|^aien-release ssh-ed25519 .*|\$l|\" docs/release/allowed_signers; sed -i \"s|^AIEN_RELEASE_SIGNER=.*|AIEN_RELEASE_SIGNER='\$l'|\" install.sh"
mutant private-material-in-record sh -c "echo '-----BEGIN OPENSSH PRIVATE KEY-----' >> docs/release/ceremony-record.txt"
mutant record-says-one-backup sed -i '0,/signs and verifies yes; altered file refused yes/s//signs and verifies no/' docs/release/ceremony-record.txt

# (a)(b)(c) the real installer and publish gate, configured only from the public record
make_release() { # dir signing-key
    local d="$1" k="$2" p="$W/pkg" f aien
    rm -rf "$p"; mkdir -p "$d" "$p/bin"
    printf '#!/bin/sh\necho aien\nexit 0\n' > "$p/bin/aien"; chmod 755 "$p/bin/aien"
    aien="$(sha256sum "$p/bin/aien" | cut -d' ' -f1)"
    { printf 'schema = "AienReleaseV1"\ncandidate = "CAND-1"\naien-cli-sha256 = "%s"\n\n[files]\n' "$aien"
      printf '"bin/aien" = "%s"\n' "$aien"; } > "$p/release.toml"
    tar -czf "$d/$ASSET" -C "$p" .
    (cd "$d" && sha256sum "$ASSET" > SHA256SUMS.txt)
    ssh-keygen -q -Y sign -f "$k" -n aien-release "$d/SHA256SUMS.txt"
}
inst() { # release-dir tag-suffix
    env HOME="$W/home" AIEN_RELEASE_TAG=v0.0.0-test AIEN_RELEASE_BASE_URL="file://$1" AIEN_BIN_DIR="$W/bin-$2" \
        AIEN_CONFIG_DIR="$W/cfg-$2" AIEN_INSTALL_NO_PROFILE=1 bash "$T/install.sh" > "$W/log" 2>&1
}
mkdir -p "$W/home"
GATE=(bash "$T/scripts/release-publish-gate.sh")
make_release "$W/good" "$KEY"
inst "$W/good" good || { cat "$W/log" >&2; fail "release signed by the matching key did not install"; }
grep -q '"signature_check": "release"' "$W/cfg-good/install-receipt.json" || fail "receipt does not say signature_check release"
grep -q "\"signer_fingerprint\": \"$FP\"" "$W/cfg-good/install-receipt.json" || fail "receipt signer_fingerprint is not the ceremony record's fingerprint"
"${GATE[@]}" "$W/good" "$T/docs/release/allowed_signers" > "$W/g.log" 2>&1 || { cat "$W/g.log" >&2; fail "publish gate refused the matching-key release"; }
pass "(a) signed by the matching key: installs as signature_check release, receipt fingerprint = record fingerprint, publish gate passes"
make_release "$W/wrongkey" "$W/other"
if inst "$W/wrongkey" wk; then fail "release signed by a different key was installed"; fi
grep -q "signature is NOT valid" "$W/log" || fail "different key not refused by name"
[[ ! -e "$W/cfg-wk/install-receipt.json" && ! -e "$W/bin-wk/aien" ]] || fail "refusal left a receipt or binary"
if "${GATE[@]}" "$W/wrongkey" "$T/docs/release/allowed_signers" > "$W/g.log" 2>&1; then fail "publish gate passed a different key"; fi
grep -q "NOT valid" "$W/g.log" || fail "publish gate did not name the signature"
pass "(b) signed by a different key: installer refuses ('signature is NOT valid'), nothing installed; publish gate refuses"
cp -r "$W/good" "$W/flip"; A="$W/flip/$ASSET"; SZ=$(wc -c < "$A"); OFF=$((SZ / 2))
B="$(dd if="$A" bs=1 skip="$OFF" count=1 2>/dev/null | od -An -tu1 | tr -d ' ')"
printf "$(printf '\\%03o' $(( (B + 1) % 256 )))" | dd of="$A" bs=1 seek="$OFF" count=1 conv=notrunc 2>/dev/null
if inst "$W/flip" flip; then fail "tampered archive installed"; fi
grep -qi checksum "$W/log" || fail "tampered archive not refused as a checksum failure"
[[ ! -e "$W/cfg-flip/install-receipt.json" ]] || fail "tampered archive left a receipt"
if "${GATE[@]}" "$W/flip" "$T/docs/release/allowed_signers" > "$W/g.log" 2>&1; then fail "publish gate passed a tampered archive"; fi
cp -r "$W/good" "$W/flipsum"; C1="$(head -c1 "$W/flipsum/SHA256SUMS.txt")"; [[ "$C1" == 0 ]] && R=1 || R=0; sed -i "1s/^./$R/" "$W/flipsum/SHA256SUMS.txt"
if inst "$W/flipsum" fs; then fail "tampered checksum list installed"; fi
grep -q "signature is NOT valid" "$W/log" || fail "tampered checksum list not refused as a signature failure"
pass "(c) tampered archive refused (checksum), tampered checksum list refused (signature), nothing installed"
echo "PASS: public-key integration: from only the public ceremony record, the trust-config change is a 3-file reviewable diff, the record fingerprint equals the pinned key (9 mismatch mutants refused), a release signed by the matching key installs as release, a different key, a tampered archive and a tampered checksum list are refused"
