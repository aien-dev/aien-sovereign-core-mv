# Release build recipe and checks

One recipe builds the release and one script checks it. The workflow, a person reproducing a digest and the
checks all use the same commands.

## Build

One command materializes every pin a native build needs (sovereign-core #92, "clean checkout plus one command"):

```
scripts/pins.sh --gpu      # CUDApp at cudapp.lock, omega at omega.lock, physics and aienos at omega's locks, under target/pins
. target/pins/env.sh       # AIEN_CUDAPP_DIR, AIEN_OMEGA_COMPOSE_DIR, AIEN_PHYSICS_DIR, AIEN_AIENOS_LOCK_REPO, AIEN_REQUIRE_NATIVE_LIBS=1
```

Without `--gpu` only the composition library is linked (the CI gate shape). The GPU engine (`libomega_gpu.a`) is built
from aien-dev/CUDApp at `cudapp.lock`; CUDApp carries its driver client in-tree, so it needs no physics checkout.
`scripts/pins.sh --check` verifies the checkouts are at their pins without touching the network. `AIEN_REQUIRE_NATIVE_LIBS=1` makes a build that
would silently fall back to a CPU stub fail instead; `scripts/golden_path.sh` sources `target/pins/env.sh` when
present. `scripts/test-pins.sh` is the offline self-test (throwaway repositories, no network).

`scripts/release-build.sh` (needs the pinned toolchain from `rust-toolchain.toml`, `cargo fetch --locked` into the
CARGO_HOME in use, for the native GPU engine `AIEN_CUDAPP_DIR` at `cudapp.lock`, and for the composition library
`AIEN_OMEGA_COMPOSE_DIR` at `omega.lock` and `AIEN_PHYSICS_DIR` at omega's `physics.lock`; `AIEN_DEV_FALLBACK` and
`AIEN_FORCE_CPU_STUB` unset, no mojo on PATH):

1. `scripts/repro-build.sh -p spark-cockpit-rs -p spark-inquisitor -p cortex-encoder-rs -p cortex-rs -p spark-supervisor -p spark-debugger`
2. `scripts/repro-build.sh -p aien-cli`

`repro-build.sh` is `cargo build --release --locked` with `--remap-path-prefix` for CARGO_HOME and the repo root.
Cargo unifies features across all `-p` packages in one command, so a binary's digest depends on its group
(`F1-binary-digest-difference.md`: one seven-package command gives a different aien-cli). Packages are therefore built
in fixed groups and `aien-cli` alone and last. That aien-cli digest equals CAND-3's `aien-cli-native-release`
(F1, build A). The CAND-3 manifest `[build]` line lists `-p aien-cli -p aien-proof -p aien-test` in one command;
whether that group gives the same aien-cli digest as `-p aien-cli` alone is UNVERIFIED here, and the release does not
package aien-proof or aien-test.

## Package and check

- `scripts/package-release.sh <out.tar.gz>` runs the release gate itself and writes the candidate id and `[model]`
  table from `release/candidate.toml` into `release.toml`. A failing gate means no package (`AIEN_DRY_RUN=1` only
  labels it `NOT-RELEASABLE`, which the checks reject).
- `scripts/check-release-candidate.sh` (tree): candidate named, `omega.lock` equals `omega-commit`, `cudapp.lock`
  holds a full sha and equals `cudapp-commit` when the candidate names one, `Cargo.lock`
  holds exactly the `[pins]` revisions (aien-protocols, crumb-spec, spark-crumbs), `[model]` complete.
- `scripts/check-release-candidate.sh --package A.tar.gz --native`: candidate id, `[model]` table (model id,
  safetensors, tokenizer, config, oracle fixture digests), `aien-cli-sha256` equals the digest of `bin/aien`, every
  listed file, no unlisted file; `--native` is required (linux aarch64, the GPU engine linked): the digest must equal
  `[executables] aien-cli-native-release`, and each helper binary must equal its `sc-<name>` entry. A package without `--native` is refused (no candidate digest for that kind); the release matrix builds only the native target.
- `scripts/verify-release-assets.sh <dir> <allowed_signers>`: signature on `SHA256SUMS.txt` valid for the pinned
  signer, every listed file matches, every `sovereign-*` archive is covered.
- `scripts/release-publish-gate.sh <dir> <allowed_signers>`: the stage-3 publish decision (see `RELEASE_SIGNING.md`).
- `scripts/test-release-checks.sh` (also in CI) proves each of these rejects: wrong omega.lock, cudapp.lock or Cargo.lock pins,
  executable digest mismatch, model input mismatch, missing, corrupt or foreign signature, tampered files.

## Release workflow modes

- Tag push `v*` (stage 1 of the offline-signing handoff, `RELEASE_SIGNING.md`): build, check, write `SHA256SUMS.txt`
  and create a DRAFT release holding the archives and checksums. No signature is made in CI and no key is in CI or a
  secret. The draft is not public. A manual run with `dry_run` false does the same, and only from a `v*` tag.
- Stage 2, the operator, offline: sign `SHA256SUMS.txt` with the public-release key (`scripts/key-ceremony.sh sign`)
  and attach `SHA256SUMS.txt.sig` to the draft.
- Stage 3, `release-publish.yml` (manual, input `tag`): checks out the tag, downloads the draft, runs
  `scripts/release-publish-gate.sh` against that commit's `docs/release/allowed_signers` and publishes only if it
  passes. It refuses unsigned, mismatched, dry-run, internal-test-key and placeholder-key releases.
- Manual run (`workflow_dispatch`): input `dry_run`, default true. Builds, packages with the manifest labelled
  `dry-run = true`, checks, writes `DRY-RUN-NOT-A-RELEASE.txt` into the signed checksum list, signs with a throwaway
  key generated in the job (public half uploaded as `dry-run-signers`, private half deleted), verifies, shows that the
  publish gate refuses it, uploads workflow artifacts only. Gate or package-check failures are recorded in
  `DRY-RUN-STATUS-<target>.txt` instead of stopping the run. It never creates a release. Installed with
  `AIEN_DRY_RUN_SIGNERS=dry-run-signers`, it is recorded as `signature_check: dry-run-key`.
- Pull requests (`release-dry-run.yml`): the publish guard, `scripts/test-release-gate-wiring.sh` and the refusal suite
  (`scripts/test-release-real-tree.sh`, run on a fixture tree pinned to the candidate) block. The tree gate on the PR's
  own revision only reports "IS / is NOT releasable as <candidate>" in the step summary: main may move ahead of the
  candidate's omega-commit. A release must be cut from a commit whose tree passes the gate; the tag-push path enforces
  it, and `test-release-gate-wiring.sh` fails if that step ever becomes non-blocking (only the manual dry run may
  continue past it).
