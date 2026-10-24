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

# Negative and positive tests for the release gate (check-release-candidate.sh, package-release.sh,
# verify-release-assets.sh). Runs offline in a throwaway fixture tree; no cargo, no network.
#   scripts/test-release-checks.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="$(mktemp -d "${TMPDIR:-/tmp}/aien-test-release-checks.XXXXXX")"
trap 'rm -rf "$W"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "ok: $*"; }

# fixture tree
T="$W/tree"; mkdir -p "$T/scripts" "$T/release" "$W/bins"
cp "$ROOT/scripts/check-release-candidate.sh" "$ROOT/scripts/package-release.sh" "$ROOT/scripts/verify-release-assets.sh" "$T/scripts/"
echo licence > "$T/LICENSE"; echo notice > "$T/NOTICE"; echo third > "$T/THIRD_PARTY.md"; echo constitution > "$T/CONSTITUTION.md"; echo readme > "$T/README.md"; echo '#!/bin/sh' > "$T/install.sh"
OM=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
PR=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb; CS=cccccccccccccccccccccccccccccccccccccccc; SC=dddddddddddddddddddddddddddddddddddddddd
H64() { printf '%064d' "$1"; }
for b in aien-cli spark-cockpit-rs spark-inquisitor cortex-encoder-rs cortex-rs spark-supervisor spark-debugger; do
    printf '#!/bin/sh\necho %s\n' "$b" > "$W/bins/$b"; chmod 755 "$W/bins/$b"
done
AIEN_DIGEST="$(sha256sum "$W/bins/aien-cli" | cut -d' ' -f1)"
write_candidate() { # aien-digest
    cat > "$T/release/candidate.toml" <<EOT
candidate = "CAND-T"
omega-commit = "$OM"
source-manifest = "fixture"

[model]
model-id = "Fixture/Model"
format = "safetensors"
model-safetensors-sha256 = "$(H64 1)"
tokenizer-json-sha256 = "$(H64 2)"
config-json-sha256 = "$(H64 3)"
oracle-fixture-safetensors-sha256 = "$(H64 4)"
oracle-fixture-manifest-sha256 = "$(H64 5)"

[pins]
aien-protocols = "$PR"
crumb-spec = "$CS"
spark-crumbs = "$SC"

[executables]
aien-cli-native-release = "$1"
$(for b in spark-cockpit-rs spark-inquisitor cortex-encoder-rs cortex-rs spark-supervisor spark-debugger; do echo "sc-$b = \"$(sha256sum "$W/bins/$b" | cut -d" " -f1)\""; done)
EOT
}
write_locks() {
    echo "$OM" > "$T/omega.lock"
    {
        echo "[[package]]"; echo "source = \"git+https://github.com/aien-dev/aien-protocols?rev=$PR#$PR\""
        echo "[[package]]"; echo "source = \"git+https://github.com/aien-dev/crumb-spec?rev=$CS#$CS\""
        echo "[[package]]"; echo "source = \"git+https://github.com/aien-dev/spark-crumbs?rev=$SC#$SC\""
    } > "$T/Cargo.lock"
}
gate() { (cd "$T" && bash scripts/check-release-candidate.sh "$@" > "$W/log" 2>&1); }
must_pass() { gate "$@" || { cat "$W/log" >&2; fail "gate refused: $*"; }; }
must_fail() { # expected-text args...
    local want="$1"; shift
    if gate "$@"; then fail "gate accepted ($want)"; fi
    grep -q -- "$want" "$W/log" || { cat "$W/log" >&2; fail "refused for the wrong reason, wanted: $want"; }
}
write_candidate "$AIEN_DIGEST"; write_locks
must_pass; pass "matching tree accepted"

# source pins
echo "$(H64 9 | cut -c1-40)" > "$T/omega.lock"; must_fail "omega.lock is"; write_locks; pass "omega.lock mismatch rejected"
sed -i "s/crumb-spec?rev=$CS#$CS/crumb-spec?rev=$SC#$SC/" "$T/Cargo.lock"; must_fail "consumes crumb-spec"; write_locks; pass "Cargo.lock crumb-spec mismatch rejected"
sed -i "s/aien-protocols?rev=$PR#$PR/aien-protocols?rev=$CS#$CS/" "$T/Cargo.lock"; must_fail "consumes aien-protocols"; write_locks; pass "Cargo.lock aien-protocols mismatch rejected"
echo "source = \"git+https://github.com/aien-dev/spark-crumbs?rev=$PR#$PR\"" >> "$T/Cargo.lock"; must_fail "several revisions of spark-crumbs"; write_locks; pass "two revisions of one pin rejected"
grep -v spark-crumbs "$T/Cargo.lock" > "$W/cl" && mv "$W/cl" "$T/Cargo.lock"; must_fail "no git source for spark-crumbs"; write_locks; pass "missing Cargo.lock pin rejected"

# candidate file
cp "$T/release/candidate.toml" "$W/cand.ok"
sed -i 's/^candidate = .*/candidate = "unknown"/' "$T/release/candidate.toml"; must_fail "missing or 'unknown'"; cp "$W/cand.ok" "$T/release/candidate.toml"; pass "candidate unknown rejected"
sed -i '/^config-json-sha256/d' "$T/release/candidate.toml"; must_fail "config-json-sha256 is missing"; cp "$W/cand.ok" "$T/release/candidate.toml"; pass "incomplete [model] rejected"
sed -i 's/^tokenizer-json-sha256 = .*/tokenizer-json-sha256 = "xyz"/' "$T/release/candidate.toml"; must_fail "not a sha256"; cp "$W/cand.ok" "$T/release/candidate.toml"; pass "malformed model digest rejected"

# package: build one with the real packager
pkg() { (cd "$T" && AIEN_RELEASE_BIN_DIR="$W/bins" bash scripts/package-release.sh "$1" > "$W/plog" 2>&1) || { cat "$W/plog" >&2; fail "package-release failed"; }; }
pkg "$W/good.tar.gz"
must_pass --package "$W/good.tar.gz" --native; pass "good package accepted (native digest matches)"
# the packager refuses a tree that fails the gate, and labels a dry run
echo 0000000000000000000000000000000000000000 > "$T/omega.lock"
if (cd "$T" && AIEN_RELEASE_BIN_DIR="$W/bins" bash scripts/package-release.sh "$W/x.tar.gz" >/dev/null 2>&1); then fail "packager packaged a tree that fails the gate"; fi
(cd "$T" && AIEN_DRY_RUN=1 AIEN_RELEASE_BIN_DIR="$W/bins" bash scripts/package-release.sh "$W/dry.tar.gz" >/dev/null 2>&1) || fail "dry-run packaging failed"
write_locks
must_fail "names candidate 'NOT-RELEASABLE'" --package "$W/dry.tar.gz"; pass "dry-run package is not a releasable package"
tar -xzOf "$W/dry.tar.gz" ./release.toml | grep -qx 'dry-run = true' || fail "dry-run package manifest is not labelled dry-run"
! tar -xzOf "$W/good.tar.gz" ./release.toml | grep -q '^dry-run' || fail "release package manifest carries a dry-run label"
pass "dry-run package manifest labelled dry-run = true; release package is not"

# executable digest mismatch: a different aien binary against the candidate's pinned digest
mkdir "$W/bins2"; cp "$W/bins/"* "$W/bins2/"; printf '#!/bin/sh\necho other build\n' > "$W/bins2/aien-cli"
(cd "$T" && AIEN_RELEASE_BIN_DIR="$W/bins2" bash scripts/package-release.sh "$W/otherbin.tar.gz" >/dev/null 2>&1)
must_fail "is not CAND-T's aien-cli-native-release" --package "$W/otherbin.tar.gz" --native; pass "executable digest mismatch rejected (--native)"
must_fail "no candidate digest for this package kind" --package "$W/otherbin.tar.gz"; pass "package without --native refused (no candidate digest for the kind)"
must_fail "no candidate digest for this package kind" --package "$W/good.tar.gz"; pass "even a matching package is refused without --native"
# a helper binary changed after the candidate: its sc-<name> digest differs
mkdir "$W/bins3"; cp "$W/bins/"* "$W/bins3/"; printf "#!/bin/sh\necho other cortex\n" > "$W/bins3/cortex-rs"
(cd "$T" && AIEN_RELEASE_BIN_DIR="$W/bins3" bash scripts/package-release.sh "$W/otherhelper.tar.gz" >/dev/null 2>&1)
must_fail "is not CAND-T's sc-cortex-rs" --package "$W/otherhelper.tar.gz" --native; pass "helper binary digest mismatch rejected"

# repack helper: edit a package then re-tar
repack() { # src dst edit-command (run inside the unpacked dir)
    rm -rf "$W/u"; mkdir "$W/u"; tar -xzf "$1" -C "$W/u"; (cd "$W/u" && eval "$3"); tar -czf "$2" -C "$W/u" .
}
# release.toml lies about aien-cli digest
repack "$W/good.tar.gz" "$W/lie.tar.gz" "sed -i 's/^aien-cli-sha256 = .*/aien-cli-sha256 = \"$(H64 7)\"/' release.toml"
must_fail "not the digest of bin/aien" --package "$W/lie.tar.gz" --native; pass "release.toml digest that is not bin/aien rejected"
# model input mismatch in the package
for key in model-id model-safetensors-sha256 tokenizer-json-sha256 config-json-sha256; do
    repack "$W/good.tar.gz" "$W/m.tar.gz" "sed -i 's/^$key = \"[^\"]*\"/$key = \"$(H64 8)\"/' release.toml"
    must_fail "package \[model\] table differs" --package "$W/m.tar.gz" --native; pass "package $key mismatch rejected"
done
repack "$W/good.tar.gz" "$W/nomodel.tar.gz" "sed -i '/^\[model\]/,/^\$/d' release.toml"
must_fail "no \[model\] table" --package "$W/nomodel.tar.gz"; pass "package without [model] rejected"
# a file changed after packaging
repack "$W/good.tar.gz" "$W/edit.tar.gz" "echo x >> bin/cortex-rs"
must_fail "does not match release.toml" --package "$W/edit.tar.gz"; pass "tampered package file rejected"
repack "$W/good.tar.gz" "$W/extra.tar.gz" "echo x > bin/extra"
must_fail "holds" --package "$W/extra.tar.gz"; pass "unlisted extra file rejected"

# signatures
ssh-keygen -q -t ed25519 -N '' -C t -f "$W/key"; ssh-keygen -q -t ed25519 -N '' -C o -f "$W/other"
printf 'aien-release %s\n' "$(cut -d' ' -f1,2 "$W/key.pub")" > "$W/signers"
assets() { rm -rf "$W/a"; mkdir "$W/a"; cp "$W/good.tar.gz" "$W/a/sovereign-linux-aarch64.tar.gz"; (cd "$W/a" && sha256sum sovereign-* > SHA256SUMS.txt); }
sign() { (cd "$W/a" && rm -f SHA256SUMS.txt.sig && ssh-keygen -q -Y sign -f "$1" -n aien-release SHA256SUMS.txt); }
verify() { bash "$ROOT/scripts/verify-release-assets.sh" "$W/a" "$W/signers" > "$W/log" 2>&1; }
vfail() { if verify; then fail "assets accepted ($1)"; fi; grep -q -- "$1" "$W/log" || { cat "$W/log" >&2; fail "wrong reason, wanted: $1"; }; }
assets; sign "$W/key"; verify || { cat "$W/log" >&2; fail "valid signed assets refused"; }; pass "valid signed assets accepted"
assets; vfail "SHA256SUMS.txt.sig is missing"; pass "missing signature rejected"
assets; sign "$W/other"; vfail "NOT valid"; pass "signature from another key rejected"
assets; sign "$W/key"; echo "garbage" > "$W/a/SHA256SUMS.txt.sig"; vfail "NOT valid"; pass "corrupt signature rejected"
assets; sign "$W/key"; echo tamper >> "$W/a/SHA256SUMS.txt"; vfail "NOT valid"; pass "edited checksum list rejected"
assets; sign "$W/key"; echo x >> "$W/a/sovereign-linux-aarch64.tar.gz"; vfail "does not match the signed checksum"; pass "tampered archive rejected"
assets; (cd "$W/a" && cp sovereign-linux-aarch64.tar.gz sovereign-macos-arm64.tar.gz); sign "$W/key"; vfail "not covered"; pass "archive missing from signed list rejected"
assets; : > "$W/empty"; if bash "$ROOT/scripts/verify-release-assets.sh" "$W/a" "$W/empty" >/dev/null 2>&1; then fail "empty signer file accepted"; fi; pass "empty signer file rejected"
# the production signer file holds exactly one aien-release key, or (until the key ceremony) only a labelled
# PLACEHOLDER that verify-release-assets.sh refuses by name; the internal test key is never in it
PROD="$ROOT/docs/release/allowed_signers"; KEYS="$(grep -cvE '^[[:space:]]*(#|$)' "$PROD" || true)"
if [[ "$KEYS" -eq 0 ]]; then
    grep -q PLACEHOLDER "$PROD" || fail "allowed_signers has no key and no PLACEHOLDER label"
    assets; sign "$W/key"; cp "$PROD" "$W/signers"; vfail "PLACEHOLDER"; pass "placeholder allowed_signers refuses a signed release by name"
else
    [[ "$KEYS" -eq 1 ]] && grep -qE '^aien-release ssh-ed25519 [A-Za-z0-9+/=]+$' "$PROD" || fail "allowed_signers must hold exactly one aien-release ed25519 key"
fi
! grep -qF "$(grep -vE '^[[:space:]]*(#|$)' "$ROOT/docs/release/internal_test_signers" | cut -d' ' -f3)" "$PROD" || fail "the internal test key is in allowed_signers"
echo "PASS: release gate rejects inconsistent source pins, executable digest and model input mismatches, missing or invalid signatures"
