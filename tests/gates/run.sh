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

# tests/gates/run.sh: the one entry point for the local verification gates of aien-sovereign-core.
# Every step runs the SAME command as the hosted job it mirrors (.github/workflows/ci.yml,
# hardening.yml, crumb-visible-conformance.yml, verify-closure.yml, release-dry-run.yml,
# sovereign-inquisitor.yml). Manifests under tests/gates/ name this script plus one step word,
# so no manifest needs shell expansion (ADR 0028 P6). The PREFILL-E2E-2 gates under
# crates/aien-runtime/tests are separate and untouched.
#
# Pinned sibling sources: the hosted jobs check out omega at omega.lock, physics and aienos at
# omega's own locks, and aien-protocols at two fixed commits (TRN1 and CRB1 vector corpora).
# Here the same commits are materialised under $CARGO_TARGET_DIR/aien-pins (default target/aien-pins) from
# the sibling local clones ../omega ../physics ../aienos ../aien-protocols, offline.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
step="${1:?usage: tests/gates/run.sh <step>}"
# A read-only HOME (the sandbox's) is swapped for a fresh writable one for this step; the read-only
# home's models/ mount stays reachable as AIEN_MODELS_DIR (tests/gates/step-home.sh, which
# scripts/test-gb10-model-dir.sh exercises with a read-only HOME).
# shellcheck source=tests/gates/step-home.sh
. "$ROOT/tests/gates/step-home.sh"
step_home
obs() { printf 'AIEN-OBS %s %s\n' "$1" "$2"; }
# Inside the user sandbox (AIEN_SANDBOX_NESTED=1, set by the worker) five aien-test self-tests that would
# start a sandbox themselves skip (they cannot nest); the receipt says so for the two steps that run them.
nested_obs() {
  if [ "${AIEN_SANDBOX_NESTED:-}" = 1 ]; then
    obs nested_sandbox_skips "\"5 aien-test self-tests skipped inside the user sandbox (AIEN_SANDBOX_NESTED=1)\""
  fi
}
# Outside the worktree when CARGO_TARGET_DIR is, and under target/ otherwise: the hosted test job
# has no pin checkouts in its tree, and test_t2_f65 scans the tree (minus target/) for *.pem files,
# which the vendored crates of a pinned aienos ship as test fixtures.
PINS="${CARGO_TARGET_DIR:-$ROOT/target}/aien-pins"
TARGET="${CARGO_TARGET_DIR:-$ROOT/target}"

pin() {  # pin <dir-name> <sibling repo> <sha40>
  local name="$1" repo="$2" sha="$3" dir="$PINS/$1"
  if [ ! -e "../$repo/.git" ]; then
    echo "run.sh: ../$repo is missing; the hosted job checks out aien-dev/$repo at $sha" >&2
    exit 3
  fi
  if [ ! -d "$dir" ] || [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" != "$sha" ]; then
    rm -rf "$dir"; mkdir -p "$PINS"
    git clone --quiet --shared --no-checkout "../$repo" "$dir"
    git -C "$dir" checkout --quiet --detach "$sha"
  fi
  obs "pin_${name//-/_}" "\"$sha\""
}

case "$step" in
  crumb-verify)
    # sc#602: the crumb tool is pinned. crumb.lock names one aien-architecture commit; the tool is
    # built from the sibling ../aien-architecture (offline, read-only) into $TARGET/crumb-tool/<pin>
    # (outside the tree when CARGO_TARGET_DIR is set, ignored under target/ otherwise) and reused
    # across steps. A crumb on PATH never decides the verdict. Without the sibling (a run directory
    # made from a sibling map that predates this line) the step falls back to the PATH crumb and
    # the receipt says so, so a reader can tell the two apart.
    pin=$(tr -d '[:space:]' < crumb.lock)
    case "$pin" in
      *[!0-9a-f]*|"") echo "run.sh: crumb.lock must hold one 40-hex commit, got '$pin'" >&2; exit 1 ;;
    esac
    [ ${#pin} -eq 40 ] || { echo "run.sh: crumb.lock must hold one 40-hex commit, got '$pin'" >&2; exit 1; }
    if [ -e ../aien-architecture/.git ]; then
      tool="$TARGET/crumb-tool/$pin"
      if [ ! -x "$tool/tools/crumb/crumb" ]; then
        rm -rf "$tool"; mkdir -p "$tool"
        git -C ../aien-architecture archive "$pin" tools/crumb | tar -x -C "$tool"
        make -s -C "$tool/tools/crumb" crumb
      fi
      obs crumb_tool_commit "\"$pin\""
      "$tool/tools/crumb/crumb" verify .
    else
      obs crumb_tool_commit "\"unpinned: ../aien-architecture missing, PATH crumb used (wanted $pin)\""
      "${CRUMB:-crumb}" verify .
    fi
    ;;
  # ci.yml job verify ("Workspace Verification & Invariant Audit")
  fmt)                 cargo fmt --all -- --check ;;
  release-gate-negatives)
    bash scripts/test-install-release.sh
    bash scripts/test-offline-signing.sh
    bash scripts/test-public-key-integration.sh
    bash scripts/check-release-trust-config.sh
    bash scripts/test-release-checks.sh
    ;;
  aien-verify-tests)   sh tools/aien-verify/tests/run.sh ;;
  pins-selftest)       bash scripts/test-pins.sh ;;
  own30b-quiet-run-selftest) bash scripts/bench/own30b-quiet-run-selftest.sh ;;
  native-identity)     bash scripts/test-native-identity.sh ;;
  install-test)        bash scripts/test-install.sh ;;
  scoring-v5)          bash docs/campaigns/scoring/test-scoring-v5.sh ;;
  check)               cargo check --workspace --all-targets ;;
  clippy)              cargo clippy --workspace --all-targets -- -D warnings ;;
  test)
    # GitHub Actions sets CI=true in every job. Tests that reach live services or check the
    # branch name (spark-adapters tier1-4) skip themselves under CI, so the hosted job never runs
    # them; without it they fail here on any branch not named feat/, fix/... or on a detached head.
    nested_obs
    CI=true AIEN_DEV_FALLBACK=1 cargo test --workspace --verbose -- --test-threads=1
    ;;
  strict-mode)
    cargo test -p aien-inference-abi --lib --test attention_wiring --test transformer_parity -- --test-threads=1
    cargo test -p aien-inference-runtime --test strict_real_model -- --test-threads=1
    ;;
  zero-cuda)
    bash scripts/zero-cuda-gate.sh --self-test
    cargo build -p aien-cli
    bash scripts/zero-cuda-gate.sh "$TARGET/debug/aien-cli"
    ;;
  preflight-guard)     bash scripts/test-preflight-branch-guard.sh ;;
  agent-preflight)
    # Hosted: developer context on pull requests, merged-tree on pushes to main.
    # CI=true as on GitHub (see the test step): the script runs cargo test --workspace.
    nested_obs
    CI=true AIEN_PROOF_OFF=1 AIEN_DEV_FALLBACK=1 AIEN_PREFLIGHT_CONTEXT="${AIEN_PREFLIGHT_CONTEXT:-developer}" bash scripts/agent-preflight.sh
    ;;
  zero-disk-secrets)
    echo "Auditing repository for prohibited plaintext secret files..."
    if find . -maxdepth 4 -name ".env*" -not -path "*/.git/*" -not -path "*/target/*" | grep -q .; then
      echo "::error::Prohibited plaintext .env file found. Use an approved AIEN SecretProvider."
      obs env_files_found true; exit 1
    fi
    obs env_files_found false
    echo "Zero plaintext .env files verified."
    ;;
  anti-slop)
    echo "Auditing repository for em dashes, en dashes, and formulaic tropes..."
    # Pattern is em dash OR en dash, built from UTF-8 bytes so this file never carries the
    # characters it scans for. Same pattern and scope as the hosted step.
    DASH_PATTERN="$(printf '\xe2\x80\x94|\xe2\x80\x93')"
    BAD_CHARS=$(grep -rnE --exclude-dir=target --exclude-dir=.git --exclude-dir=aien-protocols --exclude="*.rs" --exclude="*.json" --exclude="*.safetensors" --exclude="FORGEJO_CI.md" "$DASH_PATTERN" crates/ docs/ README.md || true)
    BAD_CHARS=$(printf "%s\n" "$BAD_CHARS" | grep -vE "^docs/campaigns/[^:]*/replies/[0-9a-f]{64}\.txt:" || true)
    if [ -n "$BAD_CHARS" ]; then
      echo "::error::Em dashes or en dashes detected:"; echo "$BAD_CHARS"
      obs dash_hits "$(printf '%s\n' "$BAD_CHARS" | grep -c .)"; exit 1
    fi
    obs dash_hits 0
    echo "Sovereign voice compliance verified."
    ;;
  # ci.yml job linked-compose
  linked-compose)
    omega_sha="$(tr -d '[:space:]' < omega.lock)"
    pin omega omega "$omega_sha"
    pin physics physics "$(tr -d '[:space:]' < "$PINS/omega/physics.lock")"
    pin aienos aienos "$(awk 'NF && $1 !~ /^#/ {print $1; exit}' "$PINS/omega/aienos.lock")"
    bash scripts/test-linked-compose.sh "$PINS/omega" "$PINS/physics" "$PINS/aienos"
    ;;
  # No hosted mirror: GitHub has no GB10. SC-GB10-SERVE-CHAT, the GPU receipt an Alpha needs
  # (verify-bundle --require-full, NO_GPU_RECEIPT). Same pinned sources as linked-compose, from the
  # siblings, offline; scripts/pins.sh --gpu reuses them and writes the env.sh install.sh sources.
  gb10-serve-chat)
    omega_sha="$(tr -d '[:space:]' < omega.lock)"
    pin omega omega "$omega_sha"
    pin physics physics "$(tr -d '[:space:]' < "$PINS/omega/physics.lock")"
    pin aienos aienos "$(awk 'NF && $1 !~ /^#/ {print $1; exit}' "$PINS/omega/aienos.lock")"
    AIEN_PINS_REMOTE_BASE="$(cd .. && pwd)" AIEN_PINS_REMOTE_SUFFIX="" bash scripts/pins.sh --gpu "$PINS"
    bash scripts/gb10-serve-chat.sh "$PINS/env.sh"
    ;;
  # SC-GB10-MODEL-DIR (portable, shell only): where SC-GB10-SERVE-CHAT finds its checkpoint under a
  # read-only HOME, and its NOT_RUN model_missing before any build or GPU start (no cargo, no GPU).
  gb10-model-dir)      bash scripts/test-gb10-model-dir.sh ;;
  # No hosted mirror (the checkpoint is host-only). SC-EXPERT-BYTES (DRA-63, #855): per-expert bytes of the
  # Qwen3.6-35B-A3B-FP8 headers, layers 0-39, by seeded Omega; width 768 and FP32 scales must go red.
  expert-bytes)        bash om/expert-bytes/gate.sh ;;
  # SC-PREFILL-LOGITS (#877, P-DRA-4 Lane P G1): the seeded-Omega last-token logits judge, self-test arms plus
  # TAU-0 and BAND-0 source mutants on synthetic fixtures; a real A/B pair is judged by gate.sh A.f32 B.f32.
  prefill-logits)      bash om/prefill-logits/gate.sh ;;
  # hardening.yml
  kv-props)
    cargo fetch --manifest-path crates/aien-replay/Cargo.toml
    cargo test --manifest-path crates/aien-replay/Cargo.toml --test kv_ownership_props
    sh crates/aien-replay/mutations/run_mutants.sh native
    ;;
  miri)
    cargo +nightly miri setup
    cargo +nightly miri test -p aien-capability
    cargo +nightly miri test -p aien-kv-cache
    sh crates/aien-replay/mutations/run_mutants.sh miri
    ;;
  trn1)
    pin trn1-protocols aien-protocols 353b7d4be3f00169545caec1615cffa61f7d311a
    export TRN1_VECTORS="$PINS/trn1-protocols/specs/execution-transcript/vectors"
    TRN1_REQUIRE=1 cargo test --manifest-path crates/aien-replay/Cargo.toml --test trn1_conformance -- --nocapture
    sh crates/aien-replay/mutations/run_mutants.sh trn1
    ;;
  # crumb-visible-conformance.yml
  crb1)
    pin crb1-protocols aien-protocols b50c00d737fa5c57ea7280e2aa4b0354ff166cf5
    export CRB1_VECTORS="$PINS/crb1-protocols/specs/crumb-visible/vectors"
    CRB1_REQUIRE=1 cargo test -p crumbs --test visible_conformance -- --nocapture
    ;;
  # verify-closure.yml
  verify-closure)
    cargo build --locked -p aien-closure --bin verify-closure
    cargo test --locked -p aien-closure
    (cd crates/aien-closure/tests/fixtures/seeded && "$TARGET/debug/verify-closure" --root . --store store)
    ;;
  # release-dry-run.yml (pull_request only)
  release-dry-run)
    bash scripts/test-dry-run-workflow.sh
    bash scripts/test-release-gate-wiring.sh
    bash scripts/test-release-signing-workflow.sh
    # Report only on the hosted side (continue-on-error): recorded here as an observation.
    if out="$(scripts/check-release-candidate.sh 2>&1)"; then
      echo "Release gate: this tree IS releasable as ${out#candidate=}"; obs tree_releasable true
    else
      echo "Release gate: this tree is NOT releasable as the candidate: $out"; obs tree_releasable false
    fi
    bash scripts/test-release-real-tree.sh
    ;;
  # sovereign-inquisitor.yml (pull_request_target). The hosted job fetched `gh pr diff` with GITHUB_TOKEN only to
  # get the diff; the verdict (exit 1 unless the audit is clean) depends on nothing but the diff text, and the
  # author, number and title only decorate the comment. The same diff is computed here from git: base...HEAD, where
  # base is the merge base with the base line (what `gh pr diff` shows), or for a head already on the base line its
  # first parent (the rule the worker's plan uses). The worker's base line is origin/main (AIEN_BASE_REF overrides).
  # Posting the comment and labels stays a GitHub job; nothing here writes to GitHub.
  inquisitor)
    cargo build --release -p spark-inquisitor
    base_ref="${AIEN_BASE_REF:-origin/main}"
    if ! git rev-parse --verify -q "$base_ref^{commit}" >/dev/null; then
      echo "run.sh: base $base_ref is not in this checkout; cannot compute the pull request diff" >&2
      exit 3
    fi
    head="$(git rev-parse HEAD)"
    base="$(git merge-base HEAD "$base_ref")"
    if [ "$base" = "$head" ]; then base="$(git rev-parse HEAD^1)"; fi
    diff_file="$(mktemp)"
    git diff --no-color --no-ext-diff "$base" "$head" > "$diff_file"
    obs inquisitor_base "\"$base\""
    obs inquisitor_diff_files "$(git diff --name-only "$base" "$head" | grep -c . || true)"
    rc=0
    "$TARGET/release/spark-inquisitor" review --author local --pr 0 \
      --title "$(git log -1 --format=%s "$head")" --diff "$diff_file" || rc=$?
    rm -f "$diff_file"
    obs inquisitor_clean "$([ "$rc" = 0 ] && echo true || echo false)"
    exit "$rc"
    ;;
  *) echo "run.sh: unknown step '$step'" >&2; exit 2 ;;
esac
