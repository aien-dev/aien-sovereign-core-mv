#!/usr/bin/env bash
# Exercise install.sh release mode offline. A fixture release is signed with a
# throwaway key and served over file://. Checks:
#   1. the key pinned inside install.sh equals docs/release/allowed_signers
#   2. a valid signed release installs from a checkout
#   3. a tampered archive installs nothing (checksum)
#   4. a release signed by another key installs nothing (signature)
#   5. a standalone copy of install.sh (no source tree, clean HOME) installs the
#      valid release offline when the signer file is given, imprint included
#   6. the same standalone copy with no signer override falls back to the key
#      pinned in install.sh, so the fixture (throwaway key) is refused
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/aien-test-install-release.XXXXXX")"
[[ -n "${KEEP:-}" ]] || trap 'rm -rf "$WORK"' EXIT
fail() { echo "FAIL: $*" >&2; [[ -z "${KEEP:-}" ]] || echo "WORK=$WORK" >&2; exit 1; }

case "$(uname -s)" in Linux) OS=linux ;; Darwin) OS=macos ;; *) fail "unsupported OS" ;; esac
case "$(uname -m)" in x86_64|amd64) ARCH=x86_64 ;; aarch64|arm64) ARCH=aarch64 ;; *) fail "unsupported arch" ;; esac
[[ "$OS-$ARCH" != macos-aarch64 ]] || ARCH=arm64
ASSET="sovereign-$OS-$ARCH.tar.gz"

# 1. pinned keys: install.sh AIEN_RELEASE_SIGNER equals the key lines of docs/release/allowed_signers (both empty
#    while the placeholder stands); AIEN_INTERNAL_TEST_SIGNER equals docs/release/internal_test_signers; the
#    internal test key is never in allowed_signers (Drake ruling 1, sc#385: the old key is not reused)
keylines() { grep -vE "^[[:space:]]*(#|\$)" "$1" || true; }
pinned() { sed -n "s/^$1='\(.*\)'\$/\1/p" "$ROOT/install.sh"; }
grep -qE "^AIEN_RELEASE_SIGNER=" "$ROOT/install.sh" || fail "install.sh has no AIEN_RELEASE_SIGNER line"
[[ "$(pinned AIEN_RELEASE_SIGNER)" == "$(keylines "$ROOT/docs/release/allowed_signers")" ]] \
    || fail "AIEN_RELEASE_SIGNER in install.sh differs from the key lines of docs/release/allowed_signers"
INTERNAL="$(pinned AIEN_INTERNAL_TEST_SIGNER)"
[[ -n "$INTERNAL" && "$INTERNAL" == "$(keylines "$ROOT/docs/release/internal_test_signers")" ]] \
    || fail "AIEN_INTERNAL_TEST_SIGNER in install.sh differs from docs/release/internal_test_signers"
! grep -qF "${INTERNAL#aien-release }" "$ROOT/docs/release/allowed_signers" || fail "the internal test key is in allowed_signers"
if [[ -z "$(pinned AIEN_RELEASE_SIGNER)" ]]; then
    grep -q "PLACEHOLDER" "$ROOT/docs/release/allowed_signers" || fail "allowed_signers holds no key and no PLACEHOLDER label"
fi

ssh-keygen -q -t ed25519 -N '' -C test -f "$WORK/key"
ssh-keygen -q -t ed25519 -N '' -C other -f "$WORK/other"
printf 'aien-release %s\n' "$(cut -d' ' -f1,2 "$WORK/key.pub")" > "$WORK/allowed_signers"

make_release() { # dir signing-key [candidate] [variant] [dry: label the manifest dry-run]
    local d="$1" k="$2" cand="${3:-CAND-1}" variant="${4:-a}" dry="${5:-}" f aien
    rm -rf "$WORK/pkg"
    mkdir -p "$d" "$WORK/pkg/bin" "$WORK/pkg/imprints/en2-trinity"
    printf '#!/bin/sh\necho aien-%s\nexit 0\n' "$variant" > "$WORK/pkg/bin/aien"
    printf '#!/bin/sh\nexit 0\n' > "$WORK/pkg/bin/cortex-rs"
    printf 'fixture imprint\n' > "$WORK/pkg/imprints/en2-trinity/soul.md"
    chmod 755 "$WORK/pkg/bin/"*
    aien="$(sha256sum "$WORK/pkg/bin/aien" 2>/dev/null | cut -d' ' -f1 || shasum -a 256 "$WORK/pkg/bin/aien" | cut -d' ' -f1)"
    {
        printf 'schema = "AienReleaseV1"\ncandidate = "%s"\naien-cli-sha256 = "%s"\n' "$cand" "$aien"
        [[ -z "$dry" ]] || printf 'dry-run = true\n'
        printf '\n[files]\n'
        (cd "$WORK/pkg" && find . -type f ! -name release.toml | LC_ALL=C sort | sed 's|^\./||') | while read -r f; do
            printf '"%s" = "%s"\n' "$f" "$(sha256sum "$WORK/pkg/$f" | cut -d' ' -f1)"
        done
    } > "$WORK/pkg/release.toml"
    tar -czf "$d/$ASSET" -C "$WORK/pkg" .
    (cd "$d" && sha256sum "$ASSET" 2>/dev/null > SHA256SUMS.txt || shasum -a 256 "$ASSET" > SHA256SUMS.txt)
    ssh-keygen -q -Y sign -f "$k" -n aien-release "$d/SHA256SUMS.txt"
}

run_install() { # release-dir bin-dir config-dir installer [VAR=value ...]
    local rel="$1" bin="$2" cfg="$3" installer="$4"
    shift 4
    env AIEN_RELEASE_TAG=v0.0.0-test AIEN_RELEASE_BASE_URL="file://$rel" \
        AIEN_BIN_DIR="$bin" AIEN_CONFIG_DIR="$cfg" AIEN_INSTALL_NO_PROFILE=1 "$@" \
        bash "$installer" > "$WORK/log" 2>&1
}

# 2. valid release, from the checkout
make_release "$WORK/good" "$WORK/key"
run_install "$WORK/good" "$WORK/bin-good" "$WORK/cfg-good" "$ROOT/install.sh" \
    AIEN_SOURCE_DIR="$ROOT" AIEN_DRY_RUN_SIGNERS="$WORK/allowed_signers" \
    || { cat "$WORK/log" >&2; fail "valid signed release did not install"; }
[[ -x "$WORK/bin-good/aien" && -x "$WORK/bin-good/cortex" ]] || fail "binaries missing after valid install"

# 3. tampered archive
make_release "$WORK/tamper" "$WORK/key"
printf 'x' >> "$WORK/tamper/$ASSET"
if run_install "$WORK/tamper" "$WORK/bin-tamper" "$WORK/cfg-tamper" "$ROOT/install.sh" \
    AIEN_SOURCE_DIR="$ROOT" AIEN_DRY_RUN_SIGNERS="$WORK/allowed_signers"; then
    fail "tampered archive was installed"
fi
grep -q "checksum mismatch" "$WORK/log" || fail "tampered archive not rejected for checksum"
[[ ! -e "$WORK/bin-tamper/aien" ]] || fail "tampered install left a binary"

# 4. signature from another key
make_release "$WORK/forged" "$WORK/other"
if run_install "$WORK/forged" "$WORK/bin-forged" "$WORK/cfg-forged" "$ROOT/install.sh" \
    AIEN_SOURCE_DIR="$ROOT" AIEN_DRY_RUN_SIGNERS="$WORK/allowed_signers"; then
    fail "release signed by another key was installed"
fi
grep -q "signature is NOT valid" "$WORK/log" || fail "forged signature not rejected"
[[ ! -e "$WORK/bin-forged/aien" ]] || fail "forged install left a binary"

# 5. standalone installer, no source tree, clean HOME, offline, signer file given
mkdir -p "$WORK/standalone" "$WORK/home"
cp "$ROOT/install.sh" "$WORK/standalone/install.sh"
run_install "$WORK/good" "$WORK/bin-alone" "$WORK/cfg-alone" "$WORK/standalone/install.sh" \
    HOME="$WORK/home" AIEN_DRY_RUN_SIGNERS="$WORK/allowed_signers" \
    || { cat "$WORK/log" >&2; fail "standalone offline install did not install"; }
grep -q "Release mode without a source tree" "$WORK/log" || fail "standalone install used a source tree"
[[ -x "$WORK/bin-alone/aien" && -x "$WORK/bin-alone/cortex" ]] || fail "binaries missing after standalone install"
[[ -f "$WORK/cfg-alone/imprints/en2/soul.md" ]] || fail "imprint from the release archive was not installed"
[[ -z "$(ls -A "$WORK/home")" ]] || fail "standalone install wrote into HOME: $(ls -A "$WORK/home")"

# 6. standalone installer, no signer override: the pinned key refuses the fixture
if run_install "$WORK/good" "$WORK/bin-pinned" "$WORK/cfg-pinned" "$WORK/standalone/install.sh" \
    HOME="$WORK/home"; then
    fail "fixture signed by a throwaway key was accepted by the pinned signer"
fi
grep -q "signature is NOT valid" "$WORK/log" || fail "pinned-signer fallback did not reject the fixture"
[[ ! -e "$WORK/bin-pinned/aien" ]] || fail "pinned-signer refusal left a binary"


# 7. upgrade: a newer candidate installs over the first; one previous release is kept; the
#    record names both; operator.toml is untouched
SIGN_ENV=(AIEN_SOURCE_DIR="$ROOT" AIEN_DRY_RUN_SIGNERS="$WORK/allowed_signers")
live() { readlink "$1/releases/current"; }
inst_args() { # bin cfg args... : run install.sh with arguments in explicit dirs
    local bin="$1" cfg="$2"; shift 2
    env AIEN_BIN_DIR="$bin" AIEN_CONFIG_DIR="$cfg" AIEN_INSTALL_NO_PROFILE=1 HOME="$WORK/home" \
        bash "$ROOT/install.sh" "$@" > "$WORK/log" 2>&1
}
echo '# local edit' >> "$WORK/cfg-good/operator.toml"
OP_BEFORE="$(sha256sum "$WORK/cfg-good/operator.toml" | cut -d' ' -f1)"
ID1="$(live "$WORK/cfg-good")"
[[ -n "$ID1" && -f "$WORK/cfg-good/installed.toml" ]] || fail "first install left no live release or record"
[[ "$("$WORK/bin-good/aien")" == "aien-a" ]] || fail "first install does not run"
make_release "$WORK/rel2" "$WORK/key" CAND-2 b
run_install "$WORK/rel2" "$WORK/bin-good" "$WORK/cfg-good" "$ROOT/install.sh" "${SIGN_ENV[@]}" \
    || { cat "$WORK/log" >&2; fail "upgrade did not install"; }
ID2="$(live "$WORK/cfg-good")"
[[ "$ID2" != "$ID1" ]] || fail "upgrade did not change the live release"
[[ "$("$WORK/bin-good/aien")" == "aien-b" ]] || fail "installed aien is not the upgraded one"
grep -q "^previous = \"$ID1\"" "$WORK/cfg-good/installed.toml" || fail "record does not name the previous release"
grep -q 'candidate = "CAND-2"' "$WORK/cfg-good/installed.toml" || fail "record lacks the new candidate"
[[ "$(sha256sum "$WORK/cfg-good/operator.toml" | cut -d' ' -f1)" == "$OP_BEFORE" ]] || fail "upgrade changed operator.toml"
[[ "$(ls "$WORK/cfg-good/releases" | grep -vc '^current$')" -eq 2 ]] || fail "expected exactly two releases kept"

# 8. rollback returns to the first release and records it; a second rollback goes forward again
inst_args "$WORK/bin-good" "$WORK/cfg-good" --rollback || { cat "$WORK/log" >&2; fail "rollback failed"; }
[[ "$(live "$WORK/cfg-good")" == "$ID1" && "$("$WORK/bin-good/aien")" == "aien-a" ]] || fail "rollback did not restore the first release"
grep -q '^last-action = "rollback"' "$WORK/cfg-good/installed.toml" || fail "rollback not recorded"
grep -q "^previous = \"$ID2\"" "$WORK/cfg-good/installed.toml" || fail "record does not name the rolled-back-from release"
inst_args "$WORK/bin-good" "$WORK/cfg-good" --rollback || fail "second rollback failed"
[[ "$(live "$WORK/cfg-good")" == "$ID2" ]] || fail "second rollback did not swap forward"
inst_args "$WORK/bin-good" "$WORK/cfg-good" --rollback || fail "third rollback failed"
if inst_args "$WORK/bin-none" "$WORK/cfg-none" --rollback; then fail "rollback with nothing installed succeeded"; fi
grep -q "no previous release" "$WORK/log" || fail "rollback with nothing installed not explained"

# 9. downgrade guard: an older candidate is refused unless --allow-downgrade
inst_args "$WORK/bin-good" "$WORK/cfg-good" --rollback >/dev/null || true   # live = ID2 (CAND-2)
[[ "$(live "$WORK/cfg-good")" == "$ID2" ]] || fail "setup for the downgrade case failed"
make_release "$WORK/rel0" "$WORK/key" CAND-1 c
if run_install "$WORK/rel0" "$WORK/bin-good" "$WORK/cfg-good" "$ROOT/install.sh" "${SIGN_ENV[@]}"; then
    fail "older candidate installed without --allow-downgrade"
fi
grep -q "allow-downgrade" "$WORK/log" || fail "downgrade refusal not explained"
[[ "$(live "$WORK/cfg-good")" == "$ID2" ]] || fail "refused downgrade changed the live release"
env AIEN_RELEASE_TAG=v0.0.0-test AIEN_RELEASE_BASE_URL="file://$WORK/rel0" AIEN_BIN_DIR="$WORK/bin-good" \
    AIEN_CONFIG_DIR="$WORK/cfg-good" AIEN_INSTALL_NO_PROFILE=1 "${SIGN_ENV[@]}" \
    bash "$ROOT/install.sh" --allow-downgrade > "$WORK/log" 2>&1 \
    || { cat "$WORK/log" >&2; fail "downgrade with --allow-downgrade failed"; }

# 10. package that does not match its own release.toml installs nothing
make_release "$WORK/badpkg" "$WORK/key" CAND-9 d
mkdir -p "$WORK/unp"; tar -xzf "$WORK/badpkg/$ASSET" -C "$WORK/unp"; printf 'x' >> "$WORK/unp/bin/cortex-rs"
tar -czf "$WORK/badpkg/$ASSET" -C "$WORK/unp" .
(cd "$WORK/badpkg" && sha256sum "$ASSET" > SHA256SUMS.txt && rm -f SHA256SUMS.txt.sig \
    && ssh-keygen -q -Y sign -f "$WORK/key" -n aien-release SHA256SUMS.txt)
LIVE_BEFORE="$(live "$WORK/cfg-good")"
if run_install "$WORK/badpkg" "$WORK/bin-good" "$WORK/cfg-good" "$ROOT/install.sh" "${SIGN_ENV[@]}"; then
    fail "package with a file that does not match release.toml was installed"
fi
grep -q "does not match its release.toml" "$WORK/log" || fail "manifest mismatch not reported"
[[ "$(live "$WORK/cfg-good")" == "$LIVE_BEFORE" ]] || fail "rejected package changed the live release"

# 11. interrupted install: kill the installer mid-copy and before the swap; the live install is
#     still the old version, runs, and a rerun completes the upgrade
make_release "$WORK/rel3" "$WORK/key" CAND-3 e
for point in mid-copy after-copy before-swap; do
    OLD="$(live "$WORK/cfg-good")"
    env AIEN_TEST_PAUSE_AT="$point" AIEN_RELEASE_TAG=v0.0.0-test AIEN_RELEASE_BASE_URL="file://$WORK/rel3" \
        AIEN_BIN_DIR="$WORK/bin-good" AIEN_CONFIG_DIR="$WORK/cfg-good" AIEN_INSTALL_NO_PROFILE=1 \
        AIEN_SOURCE_DIR="$ROOT" AIEN_DRY_RUN_SIGNERS="$WORK/allowed_signers" \
        bash "$ROOT/install.sh" > "$WORK/log-int" 2>&1 &
    IPID=$!
    for _ in $(seq 1 100); do grep -q "paused at $point" "$WORK/log-int" 2>/dev/null && break; sleep 0.1; done
    grep -q "paused at $point" "$WORK/log-int" || fail "installer never reached $point"
    pkill -9 -P "$IPID" 2>/dev/null || true; kill -9 "$IPID" 2>/dev/null || true; wait "$IPID" 2>/dev/null || true
    [[ "$(live "$WORK/cfg-good")" == "$OLD" ]] || fail "killed at $point changed the live release"
    out="$("$WORK/bin-good/aien")"; [[ "$out" == "aien-b" || "$out" == "aien-c" ]] || fail "live install does not run after kill at $point ($out)"
done
# 11b. kill after the single rename (current already switched, links and record not yet written):
#      the new release is live and runs, the record still names the old one, a rerun repairs the record
ID_NEW="$(sha256sum "$WORK/pkg/bin/aien" | cut -c1-12)"
OLD="$(live "$WORK/cfg-good")"
[[ "$OLD" != "$ID_NEW" ]] || fail "setup for the after-swap case failed"
env AIEN_TEST_PAUSE_AT=after-swap AIEN_RELEASE_TAG=v0.0.0-test AIEN_RELEASE_BASE_URL="file://$WORK/rel3" \
    AIEN_BIN_DIR="$WORK/bin-good" AIEN_CONFIG_DIR="$WORK/cfg-good" AIEN_INSTALL_NO_PROFILE=1 \
    AIEN_SOURCE_DIR="$ROOT" AIEN_DRY_RUN_SIGNERS="$WORK/allowed_signers" \
    bash "$ROOT/install.sh" > "$WORK/log-int" 2>&1 &
IPID=$!
for _ in $(seq 1 100); do grep -q "paused at after-swap" "$WORK/log-int" 2>/dev/null && break; sleep 0.1; done
grep -q "paused at after-swap" "$WORK/log-int" || fail "installer never reached after-swap"
pkill -9 -P "$IPID" 2>/dev/null || true; kill -9 "$IPID" 2>/dev/null || true; wait "$IPID" 2>/dev/null || true
[[ "$(live "$WORK/cfg-good")" == "$ID_NEW" ]] || fail "after-swap: the rename did not take effect"
[[ "$("$WORK/bin-good/aien")" == "aien-e" ]] || fail "after-swap: the live release does not run"
grep -q "^current = \"$OLD\"" "$WORK/cfg-good/installed.toml" || fail "after-swap: record unexpectedly updated before the kill"
[[ -d "$WORK/cfg-good/releases/$OLD" ]] || fail "after-swap: the previous release was pruned before the record was written"
run_install "$WORK/rel3" "$WORK/bin-good" "$WORK/cfg-good" "$ROOT/install.sh" "${SIGN_ENV[@]}" \
    || { cat "$WORK/log" >&2; fail "rerun after interruption failed"; }
[[ "$("$WORK/bin-good/aien")" == "aien-e" ]] || fail "rerun did not install the new release"
grep -q "^current = \"$ID_NEW\"" "$WORK/cfg-good/installed.toml" || fail "rerun after the after-swap kill did not repair the record"
grep -q "^previous = \"$OLD\"" "$WORK/cfg-good/installed.toml" || fail "rerun after the after-swap kill lost the previous release"
[[ -z "$(ls -A "$WORK/cfg-good/releases" | grep '^\.stage\.' || true)" ]] || fail "leftover stage after rerun"

# 12. an existing release directory that is incomplete is never overwritten
ID3="$(live "$WORK/cfg-good")"
make_release "$WORK/rel4" "$WORK/key" CAND-4 f
ID4="$(sha256sum "$WORK/pkg/bin/aien" | cut -c1-12)"
mkdir -p "$WORK/cfg-good/releases/$ID4"; echo partial > "$WORK/cfg-good/releases/$ID4/junk"
if run_install "$WORK/rel4" "$WORK/bin-good" "$WORK/cfg-good" "$ROOT/install.sh" "${SIGN_ENV[@]}"; then
    fail "install over an incomplete release directory succeeded"
fi
grep -q "was not overwritten" "$WORK/log" || fail "incomplete release directory not reported"
[[ "$(live "$WORK/cfg-good")" == "$ID3" && -f "$WORK/cfg-good/releases/$ID4/junk" ]] || fail "refusal changed state"

# 12b. kill -9 at the final switch itself. install.sh switches releases with one rename(2) of a
#      temp symlink over releases/current. After every kill the install is wholly old or wholly new:
#      it runs, its record is readable, and a rerun (or --rollback) finishes cleanly.
kill_at() { # point release-dir [install.sh args...]: run install.sh held at point, then kill -9 it
    local point="$1" rel="$2" ipid
    shift 2
    env AIEN_TEST_PAUSE_AT="$point" AIEN_RELEASE_TAG=v0.0.0-test AIEN_RELEASE_BASE_URL="file://$rel" \
        AIEN_BIN_DIR="$WORK/bin-good" AIEN_CONFIG_DIR="$WORK/cfg-good" AIEN_INSTALL_NO_PROFILE=1 \
        AIEN_SOURCE_DIR="$ROOT" AIEN_DRY_RUN_SIGNERS="$WORK/allowed_signers" HOME="$WORK/home" \
        bash "$ROOT/install.sh" "$@" > "$WORK/log-int" 2>&1 &
    ipid=$!
    for _ in $(seq 1 100); do grep -q "paused at $point" "$WORK/log-int" 2>/dev/null && break; sleep 0.1; done
    grep -q "paused at $point" "$WORK/log-int" || fail "installer never reached $point"
    pkill -9 -P "$ipid" 2>/dev/null || true; kill -9 "$ipid" 2>/dev/null || true; wait "$ipid" 2>/dev/null || true
}
rec_cur() { sed -n 's/^current = "\(.*\)"$/\1/p' "$WORK/cfg-good/installed.toml"; }
strays() { ls -A "$WORK/cfg-good/releases" | grep -c '^\.current\.new\.' || true; }

#   A. install killed after the temp link exists, before the rename: wholly old
make_release "$WORK/rel5" "$WORK/key" CAND-5 g
ID5="$(sha256sum "$WORK/pkg/bin/aien" | cut -c1-12)"
ID_OLD="$(live "$WORK/cfg-good")"
kill_at pre-rename "$WORK/rel5"
[[ "$(live "$WORK/cfg-good")" == "$ID_OLD" && "$("$WORK/bin-good/aien")" == "aien-e" ]] || fail "pre-rename kill: install is not wholly old"
[[ "$(strays)" -ge 1 ]] || fail "pre-rename: no temp link, the hold is not at the rename"
[[ "$(rec_cur)" == "$ID_OLD" ]] || fail "pre-rename kill: record changed"
run_install "$WORK/rel5" "$WORK/bin-good" "$WORK/cfg-good" "$ROOT/install.sh" "${SIGN_ENV[@]}" \
    || { cat "$WORK/log" >&2; fail "rerun after pre-rename kill failed"; }
[[ "$(live "$WORK/cfg-good")" == "$ID5" && "$("$WORK/bin-good/aien")" == "aien-g" && "$(rec_cur)" == "$ID5" ]] || fail "rerun after pre-rename kill is not wholly new"
[[ "$(strays)" -eq 0 ]] || fail "temp link left after rerun"

#   B. install killed after the rename and the links, before the record: wholly new, record repaired by rerun
make_release "$WORK/rel6" "$WORK/key" CAND-6 h
ID6="$(sha256sum "$WORK/pkg/bin/aien" | cut -c1-12)"
kill_at after-links "$WORK/rel6"
[[ "$(live "$WORK/cfg-good")" == "$ID6" && "$("$WORK/bin-good/aien")" == "aien-h" ]] || fail "after-links kill: install is not wholly new"
[[ "$(rec_cur)" == "$ID5" && -d "$WORK/cfg-good/releases/$ID5" ]] || fail "after-links kill: record or previous release lost"
run_install "$WORK/rel6" "$WORK/bin-good" "$WORK/cfg-good" "$ROOT/install.sh" "${SIGN_ENV[@]}" \
    || { cat "$WORK/log" >&2; fail "rerun after after-links kill failed"; }
[[ "$(rec_cur)" == "$ID6" ]] && grep -q "^previous = \"$ID5\"" "$WORK/cfg-good/installed.toml" || fail "rerun did not repair the record"

#   C. --rollback killed before its rename: wholly old (the release it was leaving); rerun completes
kill_at pre-rename "$WORK/none" --rollback
[[ "$(live "$WORK/cfg-good")" == "$ID6" && "$("$WORK/bin-good/aien")" == "aien-h" && "$(rec_cur)" == "$ID6" ]] || fail "rollback killed pre-rename: not wholly old"
inst_args "$WORK/bin-good" "$WORK/cfg-good" --rollback || { cat "$WORK/log" >&2; fail "rollback rerun failed"; }
[[ "$(live "$WORK/cfg-good")" == "$ID5" && "$("$WORK/bin-good/aien")" == "aien-g" && "$(rec_cur)" == "$ID5" ]] || fail "rollback rerun: not wholly rolled back"
[[ "$(strays)" -eq 0 ]] || fail "temp link left after rollback rerun"

#   D. --rollback killed after its rename, before the record: wholly new (the target); rerun is safe and
#      leaves a consistent record; a further rollback still swaps back and verifies
kill_at after-swap "$WORK/none" --rollback
[[ "$(live "$WORK/cfg-good")" == "$ID6" && "$("$WORK/bin-good/aien")" == "aien-h" ]] || fail "rollback killed after-swap: not wholly switched"
inst_args "$WORK/bin-good" "$WORK/cfg-good" --rollback || { cat "$WORK/log" >&2; fail "rollback rerun after after-swap kill failed"; }
[[ "$(live "$WORK/cfg-good")" == "$(rec_cur)" ]] || fail "record and live release disagree after the rerun"
[[ "$("$WORK/bin-good/aien")" == "aien-h" || "$("$WORK/bin-good/aien")" == "aien-g" ]] || fail "install does not run after the rerun"
inst_args "$WORK/bin-good" "$WORK/cfg-good" --rollback || fail "rollback after recovery failed"
[[ "$(live "$WORK/cfg-good")" == "$(rec_cur)" ]] || fail "record and live release disagree after a further rollback"

# 13. release gate: the candidate must be named and omega.lock must be the candidate's omega commit
GT="$WORK/gate"; mkdir -p "$GT/scripts" "$GT/release"
cp "$ROOT/scripts/check-release-candidate.sh" "$GT/scripts/"; cp "$ROOT/release/candidate.toml" "$GT/release/"; cp "$ROOT/Cargo.lock" "$GT/"
gate() { (cd "$GT" && bash scripts/check-release-candidate.sh "$@" > "$WORK/log-gate" 2>&1); }
OM="$(sed -n 's/^omega-commit *= *"\(.*\)"$/\1/p' "$GT/release/candidate.toml")"
echo "$OM" > "$GT/omega.lock"
gate --model "$WORK/model.toml" || { cat "$WORK/log-gate" >&2; fail "gate refused a matching tree"; }
grep -q '^\[model\]' "$WORK/model.toml" && grep -q '^model-safetensors-sha256' "$WORK/model.toml" || fail "gate did not extract the model table"
echo 0000000000000000000000000000000000000000 > "$GT/omega.lock"
if gate; then fail "gate accepted an omega.lock that is not the candidate's"; fi
grep -q "cannot be released" "$WORK/log-gate" || fail "omega mismatch not explained"
echo "$OM" > "$GT/omega.lock"
sed -i 's/^candidate = .*/candidate = "unknown"/' "$GT/release/candidate.toml"
if gate; then fail "gate accepted candidate unknown"; fi
rm "$GT/release/candidate.toml"
if gate; then fail "gate accepted a missing candidate file"; fi

# 14. offline-signing handoff (sc#385): signature classes, labelled receipts, negative rows. Every key here is a
#     throwaway made above; nothing in this section touches a real key.
rcpt() { sed -n "s/^  \"$2\": \"\{0,1\}\([^\",]*\)\"\{0,1\},\{0,1\}\$/\1/p" "$1/install-receipt.json"; }
#   A. a dry-run artifact (manifest labelled dry-run, signed by a throwaway key given explicitly) installs and is
#      recorded as signature_check dry-run-key in installed.toml and in install-receipt.json
make_release "$WORK/dry" "$WORK/key" CAND-1 a dry
run_install "$WORK/dry" "$WORK/bin-dry" "$WORK/cfg-dry" "$ROOT/install.sh" "${SIGN_ENV[@]}" \
    || { cat "$WORK/log" >&2; fail "dry-run artifact did not install with the throwaway signer"; }
grep -q '^signature-check = "dry-run-key"$' "$WORK/cfg-dry/installed.toml" || fail "installed.toml does not record dry-run-key"
[[ "$(rcpt "$WORK/cfg-dry" signature_check)" == dry-run-key ]] || fail "install receipt does not record signature_check dry-run-key"
[[ "$(rcpt "$WORK/cfg-dry" dry_run)" == true ]] || fail "install receipt does not record the dry-run manifest label"
[[ "$(rcpt "$WORK/cfg-dry" artifact_sha256)" == "$(sha256sum "$WORK/dry/$ASSET" | cut -d' ' -f1)" ]] || fail "receipt artifact_sha256 is not the artifact's digest"
[[ "$(rcpt "$WORK/cfg-dry" files_verified)" =~ ^[1-9][0-9]*$ ]] || fail "receipt files_verified is not a positive count"
[[ "$(rcpt "$WORK/cfg-dry" cli_sha256)" =~ ^[0-9a-f]{64}$ ]] || fail "receipt cli_sha256 missing"
grep -q 'signature_check: dry-run-key' "$WORK/log" || fail "installer output does not say dry-run-key"
#   B. one byte changed in the middle of the archive (same size): refused, names the checksum, nothing installed
make_release "$WORK/flip" "$WORK/key" CAND-1 a dry
SZ="$(wc -c < "$WORK/flip/$ASSET")"; OFF=$((SZ / 2))
B="$(dd if="$WORK/flip/$ASSET" bs=1 skip="$OFF" count=1 2>/dev/null | od -An -tu1 | tr -d ' ')"
printf "$(printf '\\%03o' $(( (B + 1) % 256 )))" | dd of="$WORK/flip/$ASSET" bs=1 seek="$OFF" count=1 conv=notrunc 2>/dev/null
[[ "$(wc -c < "$WORK/flip/$ASSET")" == "$SZ" ]] || fail "byte flip changed the size"
if run_install "$WORK/flip" "$WORK/bin-flip" "$WORK/cfg-flip" "$ROOT/install.sh" "${SIGN_ENV[@]}"; then fail "artifact with one byte changed was installed"; fi
grep -qi "checksum" "$WORK/log" || fail "one-byte change not refused by name (checksum)"
[[ ! -e "$WORK/bin-flip/aien" && ! -e "$WORK/cfg-flip/install-receipt.json" ]] || fail "one-byte refusal left a binary or a receipt"
#   C. one byte changed in the signed checksum list: refused, names the signature
make_release "$WORK/flipsum" "$WORK/key" CAND-1 a dry
C1="$(head -c1 "$WORK/flipsum/SHA256SUMS.txt")"; [[ "$C1" == 0 ]] && R=1 || R=0; sed -i "1s/^./$R/" "$WORK/flipsum/SHA256SUMS.txt"
if run_install "$WORK/flipsum" "$WORK/bin-fs" "$WORK/cfg-fs" "$ROOT/install.sh" "${SIGN_ENV[@]}"; then fail "edited checksum list was installed"; fi
grep -q "signature" "$WORK/log" || fail "edited checksum list not refused by name (signature)"
[[ ! -e "$WORK/bin-fs/aien" ]] || fail "edited checksum list left a binary"
#   D. missing signature: refused, says unsigned, nothing installed
make_release "$WORK/nosig" "$WORK/key" CAND-1 a dry
rm -f "$WORK/nosig/SHA256SUMS.txt.sig"
if run_install "$WORK/nosig" "$WORK/bin-ns" "$WORK/cfg-ns" "$ROOT/install.sh" "${SIGN_ENV[@]}"; then fail "release without a signature was installed"; fi
grep -q "SHA256SUMS.txt.sig is missing" "$WORK/log" || fail "missing signature not refused by name"
[[ ! -e "$WORK/bin-ns/aien" && ! -e "$WORK/cfg-ns/install-receipt.json" ]] || fail "missing-signature refusal left a binary or a receipt"
#   E. wrong key: a dry-run artifact signed by another throwaway key is refused
make_release "$WORK/wrong" "$WORK/other" CAND-1 a dry
if run_install "$WORK/wrong" "$WORK/bin-wk" "$WORK/cfg-wk" "$ROOT/install.sh" "${SIGN_ENV[@]}"; then fail "wrong-key artifact was installed"; fi
grep -q "signature is NOT valid" "$WORK/log" || fail "wrong key not refused by name"
#   F. a dry-run artifact can never be recorded as a release. Simulate a pinned release key with a copy of install.sh
#      whose AIEN_RELEASE_SIGNER is the throwaway key: an unlabelled package signed by it records release (positive
#      control); a dry-run-labelled package signed by the same key is refused and leaves no receipt.
mkdir -p "$WORK/pinned-sim"
sed "s|^AIEN_RELEASE_SIGNER=.*|AIEN_RELEASE_SIGNER='$(cat "$WORK/allowed_signers")'|" "$ROOT/install.sh" > "$WORK/pinned-sim/install.sh"
grep -q "^AIEN_RELEASE_SIGNER='aien-release ssh-ed25519 " "$WORK/pinned-sim/install.sh" || fail "release-key simulation did not apply"
run_install "$WORK/good" "$WORK/bin-rel" "$WORK/cfg-rel" "$WORK/pinned-sim/install.sh" HOME="$WORK/home" \
    || { cat "$WORK/log" >&2; fail "simulated release did not install"; }
[[ "$(rcpt "$WORK/cfg-rel" signature_check)" == release ]] || fail "release-key signature not recorded as release (control)"
if run_install "$WORK/dry" "$WORK/bin-dryrel" "$WORK/cfg-dryrel" "$WORK/pinned-sim/install.sh" HOME="$WORK/home"; then
    fail "dry-run artifact signed by the release key was installed"
fi
grep -q "labelled dry-run" "$WORK/log" || fail "dry-run artifact under the release key not refused by name"
[[ ! -e "$WORK/cfg-dryrel/install-receipt.json" ]] || fail "refused dry-run artifact left a receipt"
#   G. with the placeholder in place (no release key pinned) and no throwaway signer, nothing verifies as a release
if [[ -z "$(pinned AIEN_RELEASE_SIGNER)" ]]; then
    if run_install "$WORK/good" "$WORK/bin-ph" "$WORK/cfg-ph" "$ROOT/install.sh" HOME="$WORK/home"; then fail "placeholder accepted a release"; fi
    grep -q "no public-release key is pinned" "$WORK/log" || fail "placeholder refusal does not say no release key is pinned"
fi
#   H. a release signed by the internal test key would be labelled internal-test-key (classification is by the
#      verifying key; checked by grepping the installer, because that private key is not on this machine)
grep -q 'SIG_CHECK=internal-test-key' "$ROOT/install.sh" || fail "install.sh has no internal-test-key class"
echo "PASS: release gate fails closed, release install verifies signature, checksum and every package file, labels the signature class (release, internal-test-key, dry-run-key) in installed.toml and install-receipt.json, never records a dry-run artifact as a release, refuses a one-byte change, an edited checksum list, a missing signature and a wrong key, rejects tampered and forged releases, installs standalone offline, pinned key matches allowed_signers, upgrades atomically, keeps one previous release, rolls back, refuses downgrades, survives an interrupted install (including kill -9 just before the rename, just after it, and during rollback)"
