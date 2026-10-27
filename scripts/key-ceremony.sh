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

# AIEN public-release key ceremony and offline signing (Drake ruling 1, sc#385; operator checklist sc#118).
# Plain-English guide: docs/release/KEY_CEREMONY.md. This script walks the operator through it on an OFFLINE machine.
#
#   scripts/key-ceremony.sh ceremony --out DIR --backup DIR1 --backup DIR2 [--rehearsal]
#       make the dedicated offline Ed25519 release key, show its fingerprint for the witness, run the public-key
#       verification test (a good signature verifies, an altered file does not), copy the key to two backups and prove
#       each backup signs, and write DIR/public/ (allowed_signers line + ceremony record: public data only)
#   scripts/key-ceremony.sh verify-backup KEY ALLOWED_SIGNERS [--rehearsal]
#       prove a backup copy of the key still signs and matches the pinned public key (run any time)
#   scripts/key-ceremony.sh sign KEY ALLOWED_SIGNERS SHA256SUMS.txt [--rehearsal]
#       stage 2 of a release: sign the checksum file offline and verify the signature before it leaves the machine
#
# --rehearsal: a throwaway key with no passphrase, labelled REHEARSAL everywhere, for practice and for the tests
# (scripts/test-offline-signing.sh). A rehearsal key is never a release key. Without --rehearsal the script refuses to
# run on a machine with a network route, under CI, or with any output inside a git checkout.
# AIEN_CEREMONY_ASSUME_YES=1 answers the prompts; it is honoured only with --rehearsal.
set -euo pipefail
NS=aien-release
die() { echo; echo "STOPPED: $*" >&2; exit 1; }
say() { printf '\n== %s\n' "$*"; }
REHEARSAL=0; OUT=""; BACKUPS=(); ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rehearsal) REHEARSAL=1 ;;
        --out) OUT="${2:?--out needs a directory}"; shift ;;
        --backup) BACKUPS+=("${2:?--backup needs a directory}"); shift ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done
set -- "${ARGS[@]+"${ARGS[@]}"}"
CMD="${1:-}"; shift || true
YES=0; [[ "$REHEARSAL" == 1 && "${AIEN_CEREMONY_ASSUME_YES:-}" == 1 ]] && YES=1
LABEL=""; [[ "$REHEARSAL" == 0 ]] || LABEL="REHEARSAL "

ask() { # ask <question>: the operator types yes to go on
    [[ "$YES" == 0 ]] || { echo "$1 [rehearsal: yes]"; return 0; }
    local a; read -r -p "$1 Type yes to continue: " a < /dev/tty
    [[ "$a" == yes ]] || die "the operator did not type yes"
}
answer() { # answer <prompt> <rehearsal default>
    [[ "$YES" == 0 ]] || { echo "$2"; return 0; }
    local a; read -r -p "$1 " a < /dev/tty; [[ -n "$a" ]] || die "an answer is required"; echo "$a"
}
in_git() { local d="$1"; while [[ ! -d "$d" ]]; do d="$(dirname "$d")"; done; git -C "$d" rev-parse --is-inside-work-tree >/dev/null 2>&1; }
preflight() {
    command -v ssh-keygen >/dev/null || die "ssh-keygen is not installed (it comes with OpenSSH)"
    echo "ssh-keygen: $(ssh -V 2>&1)"
    if [[ "$REHEARSAL" == 1 ]]; then echo "REHEARSAL: throwaway key, not a release key; the offline checks are skipped"; return 0; fi
    [[ -z "${CI:-}${GITHUB_ACTIONS:-}" ]] || die "this looks like a CI machine; the release key is never made or used in CI"
    if command -v ip >/dev/null 2>&1; then
        [[ -z "$(ip route show default 2>/dev/null)$(ip -6 route show default 2>/dev/null)" ]] \
            || die "this machine has a network route. Unplug the network cable and turn Wi-Fi off, then start again"
    elif command -v route >/dev/null 2>&1 && route -n get default >/dev/null 2>&1; then
        die "this machine has a network route. Unplug the network cable and turn Wi-Fi off, then start again"
    fi
    ask "Is this machine offline (no cable, Wi-Fi off) and is the witness watching the screen?"
}
# selftest <key> <allowed_signers> <workdir>: a fresh file signed by key verifies; the same file with one byte
# changed does not. Prints two result lines.
selftest() {
    local k="$1" s="$2" w="$3"
    printf 'AIEN key check %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RANDOM$RANDOM" > "$w/challenge.txt"
    rm -f "$w/challenge.txt.sig"
    ssh-keygen -q -Y sign -f "$k" -n "$NS" "$w/challenge.txt" || die "the key could not sign (wrong passphrase?)"
    ssh-keygen -q -Y verify -f "$s" -I "$NS" -n "$NS" -s "$w/challenge.txt.sig" < "$w/challenge.txt" >/dev/null \
        || die "a signature made by this key does NOT verify against $s"
    echo "  good signature verifies: yes"
    sed -i.bak '1s/^A/B/' "$w/challenge.txt"; rm -f "$w/challenge.txt.bak"
    if ssh-keygen -q -Y verify -f "$s" -I "$NS" -n "$NS" -s "$w/challenge.txt.sig" < "$w/challenge.txt" >/dev/null 2>&1; then
        die "an ALTERED file verified; this verification setup is broken"
    fi
    echo "  altered file is refused: yes"
    rm -f "$w/challenge.txt" "$w/challenge.txt.sig"
}

ceremony() {
    [[ -n "$OUT" ]] || die "give --out DIR (an empty folder on the offline machine)"
    [[ "${#BACKUPS[@]}" -ge 2 ]] || die "give two --backup folders (for example two USB sticks)"
    local d
    for d in "$OUT" "${BACKUPS[@]}"; do ! in_git "$d" || die "$d is inside a git checkout; the private key must never be near a repository"; done
    [[ ! -e "$OUT" || -z "$(ls -A "$OUT")" ]] || die "$OUT is not empty"
    preflight
    say "${LABEL}Step 1 of 6: who is here"
    local op wit; op="$(answer "Operator's full name:" "REHEARSAL operator")"; wit="$(answer "Witness's full name:" "REHEARSAL witness")"
    say "${LABEL}Step 2 of 6: make the key"
    mkdir -p "$OUT/public"; chmod 700 "$OUT"
    local stamp key comment
    stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ "$REHEARSAL" == 1 ]]; then
        key="$OUT/REHEARSAL-aien_release_ed25519"; comment="aien-REHEARSAL-throwaway-NOT-A-RELEASE-KEY"
        ssh-keygen -q -t ed25519 -N '' -C "$comment" -f "$key"
    else
        key="$OUT/aien_release_ed25519"; comment="aien-release-key-${stamp%%T*}"
        echo "ssh-keygen will now ask for a passphrase twice. Choose a long one; the witness does not see it."
        ssh-keygen -t ed25519 -a 100 -C "$comment" -f "$key"
    fi
    chmod 600 "$key"
    local fp line; fp="$(ssh-keygen -lf "$key.pub" | awk '{ print $2 }')"
    line="$NS $(cut -d' ' -f1,2 "$key.pub")"
    say "${LABEL}Step 3 of 6: fingerprint for the witness"
    echo "  Fingerprint: $fp"
    echo "  The witness writes this down on paper, by hand."
    local tail8; tail8="$(answer "Witness: type the LAST 8 characters you wrote down:" "${fp: -8}")"
    [[ "$tail8" == "${fp: -8}" ]] || die "the witness's copy does not match the fingerprint on screen"
    { [[ "$REHEARSAL" == 0 ]] || echo "# REHEARSAL KEY: throwaway, NOT a release key. Never pin this line."; echo "$line"; } > "$OUT/public/allowed_signers"
    say "${LABEL}Step 4 of 6: public-key verification test"
    selftest "$key" "$OUT/public/allowed_signers" "$OUT"
    say "${LABEL}Step 5 of 6: backups (each copy must sign and verify on its own)"
    local b i=0 results=""
    for b in "${BACKUPS[@]}"; do
        i=$((i + 1))
        ask "Backup $i goes to $b. Is that medium plugged in?"
        mkdir -p "$b"; chmod 700 "$b"
        cp "$key" "$key.pub" "$b/"; chmod 600 "$b/$(basename "$key")"; sync
        cmp -s "$key" "$b/$(basename "$key")" || die "backup $i is not an exact copy"
        echo "  backup $i is an exact copy: yes"
        selftest "$b/$(basename "$key")" "$OUT/public/allowed_signers" "$b"
        results="${results}backup-$i: exact copy yes; signs and verifies yes; altered file refused yes"$'\n'
    done
    say "${LABEL}Step 6 of 6: the public record"
    {
        [[ "$REHEARSAL" == 0 ]] || echo "REHEARSAL: throwaway key, not a release key, not a ceremony."
        echo "AIEN public-release key ceremony record"
        echo "date-utc: $stamp"
        echo "mode: $([[ "$REHEARSAL" == 1 ]] && echo rehearsal || echo ceremony)"
        echo "operator: $op"
        echo "witness: $wit"
        echo "tool: $(ssh -V 2>&1)"
        echo "key-type: ed25519"
        echo "key-comment: $comment"
        echo "fingerprint: $fp"
        echo "allowed_signers: $line"
        echo "public-key verification test: good signature verifies yes; altered file refused yes"
        printf '%s' "$results"
        echo "private key: stays on the offline machine and the backups; it is not in this record"
    } > "$OUT/public/ceremony-record.txt"
    cat "$OUT/public/ceremony-record.txt"
    echo
    echo "Copy ONLY the folder $OUT/public to a USB stick for the connected computer. Never copy the key file."
}

verify_backup() {
    local k="${1:?KEY}" s="${2:?ALLOWED_SIGNERS}" w
    [[ -f "$k" && -f "$s" ]] || die "key or signer file missing"
    w="$(mktemp -d)"; trap 'rm -rf "$w"' RETURN
    echo "${LABEL}checking backup $k against $s"
    selftest "$k" "$s" "$w"
    echo "backup OK"
}

sign_sums() {
    local k="${1:?KEY}" s="${2:?ALLOWED_SIGNERS}" f="${3:?SHA256SUMS.txt}"
    [[ -f "$k" && -f "$s" && -f "$f" ]] || die "key, signer file or checksum file missing"
    preflight
    grep -qvE '^[0-9a-f]{64}  \*?[A-Za-z0-9._-]+$' "$f" && die "$f has a line that is not '<sha256>  <file name>'"
    [[ -s "$f" ]] || die "$f is empty"
    say "${LABEL}Files this signature will vouch for"
    awk '{ print "  " $2 }' "$f"
    ask "Are these exactly the files of the release you mean to sign?"
    rm -f "$f.sig"
    ssh-keygen -q -Y sign -f "$k" -n "$NS" "$f" || die "signing failed (wrong passphrase?)"
    ssh-keygen -q -Y verify -f "$s" -I "$NS" -n "$NS" -s "$f.sig" < "$f" >/dev/null || { rm -f "$f.sig"; die "the new signature does not verify against $s; nothing to upload"; }
    echo "${LABEL}signed and verified: $f.sig"
    echo "Carry $f.sig (only that file) to the connected computer and attach it to the draft release."
}

case "$CMD" in
    ceremony) ceremony ;;
    verify-backup) verify_backup "$@" ;;
    sign) sign_sums "$@" ;;
    *) sed -n '2,20p' "$0"; exit 2 ;;
esac
