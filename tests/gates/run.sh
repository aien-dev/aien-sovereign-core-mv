#!/usr/bin/env bash
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
# GitHub runners give every job a writable HOME; the sandbox's HOME (/var/lib/aien-run) is read-only on
# purpose. Tools that write under HOME (Chrome's crash database in aien-cli's browser_mirror test, which
# otherwise dies with SIGTRAP) get a fresh one for this step. The sandbox's git settings come along:
# its .gitconfig trusts the operator-owned checkout (safe.directory), and without it git refuses the
# tree ("not inside a git repository"). CARGO_HOME and RUSTUP_HOME are set by the sandbox and do not
# follow HOME.
if [ -n "${HOME:-}" ] && [ ! -w "$HOME" ]; then
  ro_home="$HOME"
  HOME="$(mktemp -d)"; export HOME
  trap 'rm -rf "$HOME"' EXIT
  if [ -f "$ro_home/.gitconfig" ]; then cp "$ro_home/.gitconfig" "$HOME/.gitconfig"; fi
fi
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
  crumb-verify)        "${CRUMB:-crumb}" verify . ;;
  # ci.yml job verify ("Workspace Verification & Invariant Audit")
  no-python)           sh scripts/check-no-python.sh ;;
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
  # sovereign-inquisitor.yml (pull_request_target: needs the PR diff and GITHUB_TOKEN)
  inquisitor)
    cargo build --release -p spark-inquisitor
    echo "run.sh: the review step needs a GitHub pull request diff and token (pull_request_target); nothing to run locally" >&2
    exit 3
    ;;
  *) echo "run.sh: unknown step '$step'" >&2; exit 2 ;;
esac
