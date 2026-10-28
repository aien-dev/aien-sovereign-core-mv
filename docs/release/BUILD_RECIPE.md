# Release build recipe and checks

One recipe builds the release and one script checks it. The workflow, a person reproducing a digest and the
checks all use the same commands.

## Build

Materialize every source pin a native build needs (sovereign-core #92, "clean checkout plus one command"):

```bash
scripts/pins.sh --gpu      # omega, physics, aienos, canonical CUDApp and its unit bridge, under target/pins
. target/pins/env.sh
```

The seeded `OMEGA_OMC` executable must have SHA-256
`2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f`.
Set `OMEGA_OMC` to that verified executable before `pins.sh`; otherwise source materialization
can finish but the script explicitly reports the missing compiler prerequisite. It does not
bootstrap an unchecked working-tree script or silently choose another host compiler.
The bridge comes from CUDApp's `om/omega-bridge.lock`, not Omega's separate CPU-unit bridge lock.
`env.sh` exports the component directories, available verified stage tools and
`AIEN_REQUIRE_NATIVE_LIBS=1`.

Without `--gpu` only the composition library is linked (the CI gate shape).
`scripts/pins.sh --gpu --check` verifies the GPU source pins without touching the network;
omit `--gpu` for composition-only checks. Required-native builds fail instead of quietly
using a stub. `scripts/golden_path.sh` sources `target/pins/env.sh` when present.
`scripts/test-pins.sh` is the offline self-test (throwaway repositories, no network).

### Canonical native admission and host rehearsal

The actual GPU ABI, headers and symbols belong to pinned CUDApp `libcudapp.a`.
Omega `libomega_gpu.a` is a separately locked CPU/evidence companion, linked first;
the canonical archive is linked last. Its ABI result is `none`, not a fabricated GPU
layout pass. A companion exporting any manifested GPU entry is refused. Both source
identities and both archive digests must match their independent `native-artifacts.lock`
records before the pair qualifies. GPU prebuilt overrides are refused, including with
stub mode; composition-only prebuilt development support remains nonqualifying.

The GPU builder authenticates the pinned commit, root tree, `tools` tree and source-checker
blob before executing a private snapshot of that checker. Corrupt object substitutions
are refused before the checker can run. The checker materializes the pinned Omega source,
the canonical CUDApp stage verifies its pinned source, and C layout/symbol checks use only
that canonical stage's headers. The admitted archives are copied into a private link
snapshot and rehashed against the admitted digests. Optional Q36, FP8, MoE, prefill and
other APIs derive their availability from symbols in that same admitted archive, not
retired Omega-local headers. Nothing here opens a GPU.

After sourcing the pinned environment, the host-only acceptance commands are:

```bash
bash scripts/abi-qualify.sh --out /absolute/path/to/new-host-evidence-directory
cargo test --locked --offline -p aien-inference-abi --test omega_backend_parity
cargo test --locked --offline -p aien-serve-http --lib --test loopback --test kv_persist -- --test-threads=1
```

Do not add `--chip`, `--ignored`, or a real model directory to this host rehearsal.
Loopback uses the existing tiny host test engine, real host numerical operations and an
ephemeral listener; it does not launch the resident 35B workload. Persistence binds both
sources, both archive digests, both targets and the clean engine commit in a length-framed,
domain-separated `v2:` SHA-256 identity. The bounded identifier preserves the existing KV
format limit; old single-archive identities cannot hit the new namespace.

This rehearsal still uses the reviewed CUDApp #169 source. The separately verified #173
repair is not a merged combined candidate, and user-owned #174 is pending. Keep the guide
and integration PRs draft and the final 35B candidate `NOT_FROZEN` until the accepted
combined source is supplied and the exact artifacts are rebuilt and independently locked.
No physical 35B correctness, memory, speed, reliability, recovery or neighbor-effect
qualification follows from these host tests. That path remains `NOT_RUN` without a current
operator-authorized GPU window; production resolver defaults remain NULL/count 0.

The present native path still depends on NVIDIA kernel-driver RM/UVM device interfaces,
ioctls, mappings and channels, and the pinned NVIDIA 580.173.02 interface headers.
Firmware dependence is a separate migration obligation. This is neither a kernel-driver
replacement nor a blanket CUDA-free/drop-in compatibility claim.

`scripts/release-build.sh` (needs the pinned toolchain from `rust-toolchain.toml`, `cargo fetch --locked` into the
CARGO_HOME in use, and for the native engine `AIEN_OMEGA_DIR` at `omega.lock` and `AIEN_PHYSICS_DIR` at omega's
`physics.lock`; `AIEN_DEV_FALLBACK` and `AIEN_FORCE_CPU_STUB` unset, no mojo on PATH):

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
- `scripts/check-release-candidate.sh` (tree): candidate named, `omega.lock` equals `omega-commit`, `Cargo.lock`
  holds exactly the `[pins]` revisions (aien-protocols, crumb-spec, spark-crumbs), `[model]` complete.
- `scripts/check-release-candidate.sh --package A.tar.gz --native`: candidate id, `[model]` table (model id,
  safetensors, tokenizer, config, oracle fixture digests), `aien-cli-sha256` equals the digest of `bin/aien`, every
  listed file, no unlisted file; `--native` is required (linux aarch64, omega linked): the digest must equal
  `[executables] aien-cli-native-release`, and each helper binary must equal its `sc-<name>` entry. A package without `--native` is refused (no candidate digest for that kind); the release matrix builds only the native target.
- `scripts/verify-release-assets.sh <dir> <allowed_signers>`: signature on `SHA256SUMS.txt` valid for the pinned
  signer, every listed file matches, every `sovereign-*` archive is covered.
- `scripts/release-publish-gate.sh <dir> <allowed_signers>`: the stage-3 publish decision (see `RELEASE_SIGNING.md`).
- `scripts/test-release-checks.sh` (also in CI) proves each of these rejects: wrong omega.lock or Cargo.lock pins,
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
