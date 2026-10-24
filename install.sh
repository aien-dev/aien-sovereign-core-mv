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

set -euo pipefail

# ==============================================================================
# AIEN Sovereign Stack Universal Multi-OS Developer Installer
# Single-command install for the entire sovereign monorepo ecosystem.
# Free and open-source for humanity. Zero telemetry. Zero subscriptions.
# ==============================================================================

echo "=================================================================="
echo "      ⚡ AIEN Sovereign Stack: Universal Developer Installer       "
echo "=================================================================="

# Release mode (AIEN_RELEASE_TAG=vX.Y.Z) installs a published, signed release and
# needs no source tree and no network beyond the three release assets.
RELEASE_TAG="${AIEN_RELEASE_TAG:-}"

ACTION="install"
ALLOW_DOWNGRADE=0
for arg in "$@"; do
    case "$arg" in
        --rollback) ACTION="rollback" ;;
        --allow-downgrade) ALLOW_DOWNGRADE=1 ;;
        -h|--help)
            echo "usage: install.sh [--rollback] [--allow-downgrade]"
            echo "  (none)             source build, or with AIEN_RELEASE_TAG=vX.Y.Z a signed release install"
            echo "  --rollback         switch back to the one previous release kept by the last upgrade"
            echo "  --allow-downgrade  accept a release whose candidate is older than the installed one"
            exit 0 ;;
        *) echo "Error: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

BIN_DIR="${AIEN_BIN_DIR:-$HOME/.local/bin}"
CONFIG_DIR="${AIEN_CONFIG_DIR:-$HOME/.config/sovereign}"
# Release layout: $REL_DIR/<first 12 hex of the package aien-cli sha256>/ holds one complete
# release; $REL_DIR/current is a symlink to the live one and is the only thing an install or a
# rollback switches (one rename). $BIN_DIR entries are stable symlinks through current. At most
# the live release and one previous release are kept. $RECORD lists what is installed.
REL_DIR="$CONFIG_DIR/releases"
RECORD="$CONFIG_DIR/installed.toml"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# Test hook: AIEN_TEST_PAUSE_AT=<point> makes the installer stop there until it is killed.
pause_at() {
    [[ "${AIEN_TEST_PAUSE_AT:-}" == "$1" ]] || return 0
    echo "[test] paused at $1" >&2
    while :; do sleep 1; done
}

# toml_get <file> <section|""> <key>: first value of key in the [section] table ("" = top level).
toml_get() {
    [[ -f "$1" ]] || return 0
    awk -v sec="$2" -v key="$3" '
        /^\[.*\]$/ { cur = substr($0, 2, length($0) - 2); next }
        cur == sec {
            i = index($0, " = "); if (i == 0) next
            k = substr($0, 1, i - 1); gsub(/"/, "", k)
            if (k == key) { v = substr($0, i + 3); gsub(/^"|"$/, "", v); print v; exit }
        }' "$1"
}

# toml_lines <file> <section>: the raw lines of one table.
toml_lines() {
    [[ -f "$1" ]] || return 0
    awk -v sec="$2" '
        /^\[.*\]$/ { cur = substr($0, 2, length($0) - 2); next }
        cur == sec && NF { print }' "$1"
}

# verify_release_dir <dir>: every file listed in release.toml [files] exists with its digest,
# nothing else is present, and the recorded aien-cli digest is that of bin/aien.
verify_release_dir() {
    local d="$1" line f want n=0 have aien
    [[ -f "$d/release.toml" ]] || { echo "  no release.toml in $d" >&2; return 1; }
    while IFS= read -r line; do
        f="${line%%\" = \"*}"; f="${f#\"}"
        want="${line##*\" = \"}"; want="${want%\"}"
        case "$f" in /*|*..*) echo "  unsafe path in release.toml: $f" >&2; return 1 ;; esac
        [[ -f "$d/$f" && ! -L "$d/$f" ]] || { echo "  missing file: $f" >&2; return 1; }
        [[ "$(sha256_of "$d/$f")" == "$want" ]] || { echo "  digest mismatch: $f" >&2; return 1; }
        n=$((n + 1))
    done < <(toml_lines "$d/release.toml" files)
    [[ "$n" -gt 0 ]] || { echo "  release.toml lists no files" >&2; return 1; }
    have="$(find "$d" \( -type f -o -type l \) ! -path "$d/release.toml" | grep -c . || true)"
    [[ "$have" -eq "$n" ]] || { echo "  $have files present, $n listed" >&2; return 1; }
    aien="$(toml_get "$d/release.toml" "" aien-cli-sha256)"
    [[ "$aien" =~ ^[0-9a-f]{64}$ && "$(sha256_of "$d/bin/aien")" == "$aien" ]] \
        || { echo "  aien-cli-sha256 in release.toml is not the digest of bin/aien" >&2; return 1; }
}

# Switch $REL_DIR/current to <id> with one rename.
set_current() {
    local id="$1" tmp="$REL_DIR/.current.new.$$"
    rm -f "$REL_DIR"/.current.new.*   # leftovers of an interrupted switch; never live
    rm -f "$tmp"
    ln -s "$id" "$tmp"
    pause_at pre-rename
    if mv --help 2>&1 | grep -q -- ' -T'; then mv -T "$tmp" "$REL_DIR/current"; else mv -fh "$tmp" "$REL_DIR/current"; fi
}

# bin name as installed (the archive keeps crate names for two of them).
installed_name() {
    case "$1" in spark-cockpit-rs) echo spark-cockpit ;; cortex-rs) echo cortex ;; *) echo "$1" ;; esac
}

# link_bins <old-id|""> <new-id>: BIN_DIR entries are symlinks through current; links for
# binaries that the new release no longer ships are removed.
link_bins() {
    local old="$1" new="$2" b name tmp
    mkdir -p "$BIN_DIR"
    if [[ -n "$old" && -d "$REL_DIR/$old/bin" ]]; then
        for b in "$REL_DIR/$old/bin/"*; do
            [[ -e "$REL_DIR/$new/bin/$(basename "$b")" ]] && continue
            name="$(installed_name "$(basename "$b")")"
            if [[ -L "$BIN_DIR/$name" && "$(readlink "$BIN_DIR/$name")" == "$REL_DIR/current/bin/"* ]]; then rm -f "$BIN_DIR/$name"; fi
        done
    fi
    for b in "$REL_DIR/$new/bin/"*; do
        name="$(installed_name "$(basename "$b")")"
        [[ ! -d "$BIN_DIR/$name" || -L "$BIN_DIR/$name" ]] || { echo "Error: $BIN_DIR/$name is a directory" >&2; exit 1; }
        tmp="$BIN_DIR/.$name.new.$$"
        rm -f "$tmp"
        ln -s "$REL_DIR/current/bin/$(basename "$b")" "$tmp"
        mv -f "$tmp" "$BIN_DIR/$name"
        echo "    - $name -> $BIN_DIR/$name"
    done
}

# write_record <current-id> <previous-id|""> <action>: rewrite installed.toml atomically.
write_record() {
    local cur="$1" prev="$2" action="$3" now id at sc tmp="$CONFIG_DIR/.installed.toml.new.$$"
    now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    {
        echo 'schema = "AienInstalledV1"'
        echo "current = \"$cur\""
        echo "previous = \"$prev\""
        echo "last-action = \"$action\""
        echo "last-action-at = \"$now\""
        for id in "$cur" "$prev"; do
            [[ -n "$id" ]] || continue
            at="$(toml_get "$RECORD" "version.$id" installed-at)"
            echo
            echo "[version.$id]"
            echo "candidate = \"$(toml_get "$REL_DIR/$id/release.toml" "" candidate)\""
            echo "aien-cli-sha256 = \"$(toml_get "$REL_DIR/$id/release.toml" "" aien-cli-sha256)\""
            echo "installed-at = \"${at:-$now}\""
            sc="$(toml_get "$RECORD" "version.$id" signature-check)"
            [[ "$action" != install || "$id" != "$cur" || -z "${SIG_CHECK:-}" ]] || sc="$SIG_CHECK"
            [[ -z "$sc" ]] || echo "signature-check = \"$sc\""
            if [[ -n "$(toml_lines "$REL_DIR/$id/release.toml" model)" ]]; then
                echo
                echo "[version.$id.model]"
                toml_lines "$REL_DIR/$id/release.toml" model
            fi
        done
    } > "$tmp"
    mv -f "$tmp" "$RECORD"
}

# Keep only the live and the previous release.
prune_releases() {
    local keep1="$1" keep2="$2" d
    for d in "$REL_DIR"/*/; do
        d="$(basename "$d")"
        [[ "$d" == current || "$d" == "$keep1" || "$d" == "$keep2" ]] || rm -rf "${REL_DIR:?}/$d"
    done
}

cand_num() { [[ "$1" =~ ^CAND-([0-9]+)$ ]] && echo "${BASH_REMATCH[1]}" || true; }

# activate_release <extracted-package-dir>: verify, stage, then switch with one rename.
activate_release() {
    local x="$1" aien id cand cur curcand a b stage
    echo "[*] Verifying every package file against release.toml"
    verify_release_dir "$x" || { echo "Error: package does not match its release.toml. Nothing was installed." >&2; exit 1; }
    aien="$(toml_get "$x/release.toml" "" aien-cli-sha256)"; id="${aien:0:12}"
    cand="$(toml_get "$x/release.toml" "" candidate)"
    mkdir -p "$REL_DIR"
    cur="$(toml_get "$RECORD" "" current)"
    if [[ -n "$cur" ]]; then
        curcand="$(toml_get "$RECORD" "version.$cur" candidate)"
        if [[ "$cand" != "$curcand" && "$ALLOW_DOWNGRADE" != 1 ]]; then
            a="$(cand_num "$cand")"; b="$(cand_num "$curcand")"
            if [[ -z "$a" || -z "$b" || "$a" -lt "$b" ]]; then
                echo "Error: package candidate '$cand' is not known to be newer than installed '$curcand'. Use --allow-downgrade to install it anyway. Nothing was installed." >&2
                exit 1
            fi
        fi
    fi
    rm -rf "$REL_DIR"/.stage.*   # leftovers of an interrupted install; never live
    if [[ -d "$REL_DIR/$id" ]]; then
        if verify_release_dir "$REL_DIR/$id" && cmp -s "$x/release.toml" "$REL_DIR/$id/release.toml"; then
            echo "[+] Release $id is already staged and complete"
        else
            echo "Error: $REL_DIR/$id exists but is incomplete or differs from this package. It was not overwritten. Remove it and rerun. Nothing was installed." >&2
            exit 1
        fi
    else
        stage="$REL_DIR/.stage.$id.$$"
        mkdir -p "$stage"
        local line f first=1
        while IFS= read -r line; do
            f="${line%%\" = \"*}"; f="${f#\"}"
            mkdir -p "$stage/$(dirname "$f")"
            cp "$x/$f" "$stage/$f"
            if [[ "$first" == 1 ]]; then first=0; pause_at mid-copy; fi
        done < <(toml_lines "$x/release.toml" files)
        cp "$x/release.toml" "$stage/release.toml"
        pause_at after-copy
        chmod 755 "$stage/bin/"*
        echo "[*] Verifying the staged copy"
        verify_release_dir "$stage" || { echo "Error: staged copy is not complete. Nothing was installed." >&2; exit 1; }
        mv "$stage" "$REL_DIR/$id"
        pause_at before-swap
    fi
    set_current "$id"
    pause_at after-swap
    echo "[+] Live release is now $id (candidate $cand); binaries:"
    link_bins "$cur" "$id"
    pause_at after-links
    if [[ -n "$cur" && "$cur" != "$id" ]]; then
        write_record "$id" "$cur" install
        prune_releases "$id" "$cur"
    else
        write_record "$id" "$(toml_get "$RECORD" "" previous)" install
        prune_releases "$id" "$(toml_get "$RECORD" "" previous)"
    fi
}

do_rollback() {
    local cur prev
    cur="$(toml_get "$RECORD" "" current)"; prev="$(toml_get "$RECORD" "" previous)"
    [[ -n "$cur" && -n "$prev" ]] || { echo "Error: no previous release is recorded in $RECORD. Nothing to roll back to." >&2; exit 1; }
    [[ -d "$REL_DIR/$prev" ]] || { echo "Error: previous release $prev is missing from $REL_DIR." >&2; exit 1; }
    verify_release_dir "$REL_DIR/$prev" || { echo "Error: previous release $prev failed verification. Nothing was changed." >&2; exit 1; }
    set_current "$prev"
    pause_at after-swap
    echo "[+] Rolled back: live release is now $prev (was $cur); binaries:"
    link_bins "$cur" "$prev"
    write_record "$prev" "$cur" rollback
}

if [[ "$ACTION" == "rollback" ]]; then
    do_rollback
    exit 0
fi

# Release signing keys, pinned here so a release can be verified with nothing but this file and the assets
# (offline). scripts/test-install-release.sh checks both lines against their files in docs/release/.
# AIEN_RELEASE_SIGNER is the dedicated offline public-release key (Drake ruling 1, sc#385). It equals the key line of
# docs/release/allowed_signers. PLACEHOLDER: it is empty until the key ceremony (docs/release/KEY_CEREMONY.md)
# produces the key, so until then nothing can verify, or be recorded, as a public release.
AIEN_RELEASE_SIGNER=''
# AIEN_INTERNAL_TEST_SIGNER is the old key that signed internal test release v0.1.1. It equals
# docs/release/internal_test_signers. Anything it signs installs as signature_check internal-test-key, never release.
AIEN_INTERNAL_TEST_SIGNER='aien-release ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAID6WDVy2x8aV80ag10iPonS4J8k4seWrgsi5cuq8AIZJ'

# Run from a checkout, or bootstrap one when the script is piped to Bash for a
# source build. Release mode runs without a checkout.
INSTALL_WORKSPACE=""
SOURCE_ROOT=""
if [[ -n "${AIEN_SOURCE_DIR:-}" ]]; then
    SOURCE_ROOT="$(cd "$AIEN_SOURCE_DIR" && pwd)"
elif [[ -f "${BASH_SOURCE[0]:-}" && -f "$(dirname "${BASH_SOURCE[0]}")/Cargo.toml" ]]; then
    SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
elif [[ -z "$RELEASE_TAG" ]]; then
    INSTALL_WORKSPACE="$(mktemp -d "${TMPDIR:-/tmp}/aien-install.XXXXXX")"
    trap 'rm -rf "$INSTALL_WORKSPACE"' EXIT
    git clone --quiet https://github.com/aien-dev/aien-sovereign-core.git "$INSTALL_WORKSPACE/core"
    git -C "$INSTALL_WORKSPACE/core" checkout --quiet "${AIEN_REV:-main}"
    SOURCE_ROOT="$INSTALL_WORKSPACE/core"
fi
if [[ -n "$SOURCE_ROOT" ]]; then
    [[ -f "$SOURCE_ROOT/Cargo.toml" ]] || { echo "AIEN source manifest is missing" >&2; exit 1; }
    cd "$SOURCE_ROOT"
    echo "[*] Source commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
else
    echo "[*] Release mode without a source tree: $RELEASE_TAG"
fi
# Where the EN2 imprint comes from: the checkout, or the release archive (set below).
IMPRINT_SRC="${SOURCE_ROOT:+$SOURCE_ROOT/imprints/en2-trinity}"

OS="$(uname -s | tr "[:upper:]" "[:lower:]")"
ARCH="$(uname -m)"

case "$ARCH" in
    x86_64|amd64) TARGET_ARCH="x86_64" ;;
    aarch64|arm64) TARGET_ARCH="aarch64" ;;
    *) echo "Error: Unsupported CPU architecture: $ARCH"; exit 1 ;;
esac

case "$OS" in
    linux) TARGET_OS="linux" ;;
    darwin) TARGET_OS="macos" ;;
    *) echo "Error: Unsupported operating system: $OS"; exit 1 ;;
esac

echo "[*] Detected Platform: $TARGET_OS ($TARGET_ARCH)"

# Release asset names: linux uses x86_64 or aarch64, macOS uses arm64.
RELEASE_ARCH="$TARGET_ARCH"
[[ "$TARGET_OS" != macos || "$TARGET_ARCH" != aarch64 ]] || RELEASE_ARCH="arm64"

# Release mode (AIEN_RELEASE_TAG=vX.Y.Z): install a published, signed release
# instead of building from source. Fails closed: nothing is installed unless
# SHA256SUMS.txt carries a valid signature and the archive matches its listed checksum.
# The signature class is decided by WHICH key verified, never by what the artifact says:
#   release            the public-release key pinned above (AIEN_RELEASE_SIGNER; empty until the key ceremony)
#   internal-test-key  the old internal test key (AIEN_INTERNAL_TEST_SIGNER)
#   dry-run-key        a throwaway signer file the caller names in AIEN_DRY_RUN_SIGNERS (AIEN_ALLOWED_SIGNERS is the
#                      older name of the same switch); only for dry runs and tests
# A package whose release.toml says dry-run = true is refused when it verifies as release.
# The class is written to installed.toml (signature-check) and to $CONFIG_DIR/install-receipt.json.

SIG_CHECK=""
# verify_sums <signers-file> <dir>: true when SHA256SUMS.txt.sig is a valid aien-release signature for that file.
verify_sums() {
    ssh-keygen -Y verify -f "$1" -I aien-release -n aien-release -s "$2/SHA256SUMS.txt.sig" < "$2/SHA256SUMS.txt" >/dev/null 2>&1
}

# write_receipt <tag> <asset> <dir> <files-verified>: install-receipt.json for the live release (atomic).
write_receipt() {
    local tag="$1" asset="$2" d="$3" n="$4" id cli dry tmp="$CONFIG_DIR/.install-receipt.json.new.$$"
    id="$(readlink "$REL_DIR/current")"
    cli="$(toml_get "$REL_DIR/$id/release.toml" "" aien-cli-sha256)"
    dry=false; [[ "$(toml_get "$REL_DIR/$id/release.toml" "" dry-run)" != true ]] || dry=true
    {
        echo '{'
        echo '  "schema": "AienInstallReceiptV1",'
        echo "  \"signature_check\": \"$SIG_CHECK\","
        echo "  \"dry_run\": $dry,"
        echo "  \"tag\": \"$tag\","
        echo "  \"artifact\": \"$asset\","
        echo "  \"artifact_sha256\": \"$(sha256_of "$d/$asset")\","
        echo "  \"sha256sums_sha256\": \"$(sha256_of "$d/SHA256SUMS.txt")\","
        echo "  \"signer_fingerprint\": \"$SIGNER_FP\","
        echo "  \"release_id\": \"$id\","
        echo "  \"candidate\": \"$(toml_get "$REL_DIR/$id/release.toml" "" candidate)\","
        echo "  \"files_verified\": $n,"
        echo "  \"cli_sha256\": \"$cli\","
        echo "  \"daemon_sha256\": \"$cli\","
        echo "  \"installed_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
        echo '}'
    } > "$tmp"
    mv -f "$tmp" "$CONFIG_DIR/install-receipt.json"
}

install_release() {
    local tag="$1"
    [[ "$tag" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Error: release tag '$tag' has characters other than letters, digits, dot, dash, underscore" >&2; exit 1; }
    command -v ssh-keygen >/dev/null 2>&1 || { echo "Error: ssh-keygen is required to verify the release signature" >&2; exit 1; }
    command -v curl >/dev/null 2>&1 || { echo "Error: curl is required to download the release" >&2; exit 1; }
    local base="${AIEN_RELEASE_BASE_URL:-https://github.com/aien-dev/aien-sovereign-core/releases/download/$tag}"
    local asset="sovereign-${TARGET_OS}-${RELEASE_ARCH}.tar.gz"
    DL_DIR=""
    DL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aien-release.XXXXXX")"
    trap 'rm -rf "$DL_DIR"; [[ -z "$INSTALL_WORKSPACE" ]] || rm -rf "$INSTALL_WORKSPACE"' EXIT
    local dry_signers="${AIEN_DRY_RUN_SIGNERS:-${AIEN_ALLOWED_SIGNERS:-}}"
    [[ -z "$dry_signers" || -f "$dry_signers" ]] || { echo "Error: dry-run signer file not found: $dry_signers" >&2; exit 1; }
    echo "[*] Downloading $tag: $asset, SHA256SUMS.txt, SHA256SUMS.txt.sig"
    local f
    for f in SHA256SUMS.txt SHA256SUMS.txt.sig "$asset"; do
        if ! curl --proto '=https,file' --fail --silent --show-error --location -o "$DL_DIR/$f" "$base/$f" 2> "$DL_DIR/.curl.err"; then
            [[ "$f" != SHA256SUMS.txt.sig ]] || { echo "Error: SHA256SUMS.txt.sig is missing: refusing an unsigned release (no signature). Nothing was installed." >&2; exit 1; }
            cat "$DL_DIR/.curl.err" >&2; echo "Error: could not download $f from $base" >&2; exit 1
        fi
    done
    echo "[*] Verifying the signature on SHA256SUMS.txt"
    SIG_CHECK=""; SIGNER_FP=""
    local kf="$DL_DIR/.signer"
    if [[ -n "$AIEN_RELEASE_SIGNER" ]]; then
        printf '%s\n' "$AIEN_RELEASE_SIGNER" > "$kf"
        ! verify_sums "$kf" "$DL_DIR" || SIG_CHECK=release
    fi
    if [[ -z "$SIG_CHECK" ]]; then
        printf '%s\n' "$AIEN_INTERNAL_TEST_SIGNER" > "$kf"
        ! verify_sums "$kf" "$DL_DIR" || SIG_CHECK=internal-test-key
    fi
    if [[ -z "$SIG_CHECK" && -n "$dry_signers" ]]; then
        cp "$dry_signers" "$kf"
        ! verify_sums "$kf" "$DL_DIR" || SIG_CHECK=dry-run-key
    fi
    if [[ -z "$SIG_CHECK" ]]; then
        [[ -n "$AIEN_RELEASE_SIGNER" ]] || echo "    (no public-release key is pinned yet: the key ceremony has not happened, see docs/release/KEY_CEREMONY.md)" >&2
        echo "Error: SHA256SUMS.txt signature is NOT valid for any accepted key. Nothing was installed." >&2; exit 1
    fi
    SIGNER_FP="$(awk 'NF >= 3 && $1 !~ /^#/ { print $2" "$3; exit }' "$kf" | ssh-keygen -lf - 2>/dev/null | awk '{ print $2 }' || true)"
    rm -f "$kf"
    echo "[+] signature_check: $SIG_CHECK"
    local want got
    want="$(awk -v a="$asset" '$2 == a || $2 == "*" a { print $1 }' "$DL_DIR/SHA256SUMS.txt")"
    [[ "$(printf '%s\n' "$want" | grep -c .)" -eq 1 ]] \
        || { echo "Error: $asset is not listed exactly once in the signed SHA256SUMS.txt" >&2; exit 1; }
    got="$(sha256_of "$DL_DIR/$asset")"
    [[ "$want" == "$got" ]] \
        || { echo "Error: checksum mismatch for $asset (signed $want, downloaded $got). Nothing was installed." >&2; exit 1; }
    echo "[+] Signature and checksum verified for $asset"
    mkdir -p "$DL_DIR/x" "$BIN_DIR"
    tar -xzf "$DL_DIR/$asset" -C "$DL_DIR/x"
    [[ -d "$DL_DIR/x/bin" ]] || { echo "Error: archive has no bin/ directory" >&2; exit 1; }
    if [[ "$SIG_CHECK" == release && ( "$(toml_get "$DL_DIR/x/release.toml" "" dry-run)" == true \
          || "$(toml_get "$DL_DIR/x/release.toml" "" candidate)" == NOT-RELEASABLE ) ]]; then
        echo "Error: this package is labelled dry-run but carries a release signature; a dry run is never a release. Nothing was installed." >&2
        exit 1
    fi
    [[ ! -d "$DL_DIR/x/imprints/en2-trinity" ]] || IMPRINT_SRC="$DL_DIR/x/imprints/en2-trinity"
    activate_release "$DL_DIR/x"
    write_receipt "$tag" "$asset" "$DL_DIR" "$(toml_lines "$REL_DIR/current/release.toml" files | grep -c .)"
    echo "[+] Install receipt: $CONFIG_DIR/install-receipt.json (signature_check: $SIG_CHECK)"
}

# Verify Rust Toolchain (source builds only)
if [[ -z "$RELEASE_TAG" ]] && ! command -v cargo >/dev/null 2>&1; then
    echo "[!] Cargo not found. Installing Rust toolchain via rustup..."
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

BIN_DIR="${AIEN_BIN_DIR:-$HOME/.local/bin}"
CONFIG_DIR="${AIEN_CONFIG_DIR:-$HOME/.config/sovereign}"
IMPRINT_DIR="$CONFIG_DIR/imprints/en2"

mkdir -p "$BIN_DIR" "$CONFIG_DIR" "$IMPRINT_DIR"

# Ensure ~/.local/bin is in PATH for current and future shells
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    export PATH="$BIN_DIR:$PATH"
fi

if [[ "${AIEN_INSTALL_NO_PROFILE:-0}" != "1" ]]; then
for RC in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.profile"; do
    if [ -f "$RC" ]; then
        if ! grep -Fq '$HOME/.local/bin' "$RC"; then
            echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$RC"
        fi
    fi
done
fi

if [[ -n "$RELEASE_TAG" ]]; then
    install_release "$RELEASE_TAG"
else
echo "[*] Building Core Monorepo Binaries (High-Performance Release Mode)..."
echo "    - aien-cli (Universal Terminal CLI & Orchestrator)"
echo "    - spark-cockpit-rs (3D Sovereign Glass Cockpit Server)"
echo "    - cortex-rs (Canonical Memory & Vector Store)"
echo "    - spark-supervisor (Resilient Process Supervisor)"
echo "    - spark-debugger (Zero-Leak Health & Telemetry Auditor)"
echo "    - spark-harness (5-Layer Verification Engine)"
echo "    - spark-crumbs (Blake3 Scent & Action Vector Tracker)"
echo "    - spark-aegis (Defensive Boundary & Containment Engine)"

cargo build --locked --release -p aien-cli -p spark-cockpit-rs -p cortex-rs -p spark-supervisor -p spark-debugger -p spark-harness -p spark-crumbs -p spark-aegis

echo "[*] Installing Binaries to $BIN_DIR..."
cp target/release/aien-cli "$BIN_DIR/aien"
cp target/release/spark-cockpit-rs "$BIN_DIR/spark-cockpit"
cp target/release/cortex-rs "$BIN_DIR/cortex"
cp target/release/spark-supervisor "$BIN_DIR/spark-supervisor"
cp target/release/spark-debugger "$BIN_DIR/spark-debugger"
cp target/release/spark-harness "$BIN_DIR/spark-harness"
cp target/release/spark-crumbs "$BIN_DIR/spark-crumbs"
cp target/release/spark-aegis "$BIN_DIR/spark-aegis"
chmod +x "$BIN_DIR/aien" "$BIN_DIR/spark-cockpit" "$BIN_DIR/cortex" "$BIN_DIR/spark-supervisor" "$BIN_DIR/spark-debugger" "$BIN_DIR/spark-harness" "$BIN_DIR/spark-crumbs" "$BIN_DIR/spark-aegis"

echo "[+] Binaries successfully linked:"
echo "    - aien -> $BIN_DIR/aien"
echo "    - spark-cockpit -> $BIN_DIR/spark-cockpit"
echo "    - cortex -> $BIN_DIR/cortex"
echo "    - spark-supervisor -> $BIN_DIR/spark-supervisor"
echo "    - spark-debugger -> $BIN_DIR/spark-debugger"
echo "    - spark-harness -> $BIN_DIR/spark-harness"
echo "    - spark-crumbs -> $BIN_DIR/spark-crumbs"
echo "    - spark-aegis -> $BIN_DIR/spark-aegis"
fi

echo "[*] Initializing Sovereign Operator Profile..."
if [ ! -f "$CONFIG_DIR/operator.toml" ]; then
    GIT_NAME="$(git config user.name 2>/dev/null || echo "Sovereign Operator")"
    GIT_EMAIL="$(git config user.email 2>/dev/null || echo "operator@local")"
    
    cat << CFG > "$CONFIG_DIR/operator.toml"
[operator]
name = "$GIT_NAME"
email = "$GIT_EMAIL"
handle = "operator"
sign_commits = false

[engine]
mode = "max"
api_base_url = "http://127.0.0.1:18006/v1"
api_key = ""
model_id = "atlas-lightning-omni"
max_port = 18006
context_window = 32768
temperature = 0.7
CFG
    echo "[+] Generated default configuration at $CONFIG_DIR/operator.toml"
else
    echo "[+] Existing configuration found at $CONFIG_DIR/operator.toml"
fi

echo "[*] Installing Free EN2 Trinity Imprint..."
if [[ -n "$IMPRINT_SRC" && -d "$IMPRINT_SRC" ]]; then
    cp -r "$IMPRINT_SRC"/* "$IMPRINT_DIR/"
    echo "[+] EN2 Trinity Imprint installed locally to $IMPRINT_DIR"
fi

echo "=================================================================="
echo "  ⚡ Portable binaries installed. Model and access-token provisioning remain."
echo "  The installer does not provision a model; see docs/FIRST_RUN.md."
echo "=================================================================="
echo "  Primary Command: aien"
echo ""
echo "  Quickstart (full steps: docs/FIRST_RUN.md):"
echo "    aien compose desk-key --create 1   # Create the approval desk key; aien start refuses without it"
echo "    aien start      # Starts Cortex, Cockpit, and background daemons"
echo "    aien status     # Real-time health, ports, and memory status"
echo "    aien cockpit    # Launches Sovereign Glass Cockpit in browser"
echo "    aien chat       # Interactive sovereign terminal pairing"
echo "    aien aegis      # Run defensive boundary security audit"
echo "    aien doctor     # System diagnostic and TPM key vault audit"
echo "=================================================================="
