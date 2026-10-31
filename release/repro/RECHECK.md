# Recheck kit: the SI engine sha256

This file lets a person outside the project rebuild one receipt and compare it byte for byte, with public URLs only, without our machines, our keys or our word.

**The claim being rechecked.** The Sovereign Core engine `aien-serve-http` built from Core commit `d3dec0eb18818808212bdf99db5c8612bcfcb9ef` with the container recipe in this directory has

```
sha256 6fb04197c272aa1d0235e7644b6e91953ff9b90f5145b250d674ff9fe41806e9  aien-serve-http
```

This is the `ENGINE_SHA256` that sovereign-intelligence pins in `installer/build.sh` (pull request aien-dev/sovereign-intelligence#67, head `be7b3b3f516ef220ae46ead8849790f527211472`).

**Status.** Independent rechecks: 0 (as of 2026-10-31). The one run on record, [recheck-transcripts/2026-10-31-agent.txt](recheck-transcripts/2026-10-31-agent.txt), was made by an agent under the same operator as the original build. It is EXERCISED, not independent, and it does not change the count.

## What a receipt must declare

The Crumb Doctrine defines a receipt (aien-dev/aienos.com `public/research/crumb-doctrine/paper.tex:25` and `:27`, commit `763a8ac1334926338ab3560d8a925a7a51f56288`):

> A receipt must be recheckable by an independent party under fully declared prerequisites, without asking us for permission or for our word. If an independent party cannot recheck it, it is not a receipt.

> A receipt records at minimum: the digest of the verifier that ran, the digest of the specification it checked against, the closure of dependencies the result depends on, the execution environment, the architecture assumptions, the randomness and seeds, the resource bounds, the timeout semantics, the nondeterminism policy, the input corpus or a commitment to it, and the exact result. A receipt that omits any of these is incomplete.

Each field for this receipt:

| Field | This recheck |
|---|---|
| Verifier (digest) | The recipe is the verifier. sha256 `26db7be518f0630fde85d28bfd50007353802be0d03d0e8a945fd09a3a6c14ac` `scripts/repro-container-build.sh`, `2f40abd645d9a21868c8712b06149b8a1fa763bdce3dc9cabc211c626734998f` `release/repro/Dockerfile`, `5a09da775509ace488f4fb732a40837c6636bb7d884d3e24b904d822f37cdae1` `release/repro/inside.sh`. Step 2 checks these before anything runs. The comparison itself is `sha256sum` from GNU coreutils inside the build image. |
| Specification (digest) | The pinned claim: `installer/build.sh` of aien-dev/sovereign-intelligence at `be7b3b3f516ef220ae46ead8849790f527211472`, sha256 `2e060c50dcdd3058a43af5a9805e404e748802ec0b9ccb606c2ff75d2c250953`, which sets `CORE_REV=d3dec0eb...` and `ENGINE_SHA256=6fb04197...`. The build method is `docs/release/BUILD_RECIPE.md` in this repository. |
| Dependency closure | Core `d3dec0eb18818808212bdf99db5c8612bcfcb9ef`; its `Cargo.lock` (sha256 `c62d08a276690de3eeaa25e646be41106d0ad32ba49bd1da6e2abb0ba8e9eb33`) pins every crates.io crate by checksum and every git crate by commit (aien-protocols-mv `28bc49ec`, aien-protocols `7ac6facb`, crumb-spec-mv `24194d1b`, spark-crumbs-mv `9355a92b`, all public under github.com/aien-dev); Omega `a647b6a9821fffcf52ba9ce813f75afa4b84be62` (Core's `omega.lock`); physics `048e5f5e881b092733baaf65d0ace46abda7f908` (Omega's `physics.lock`); base image `ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852`; rustup-init 1.29.1 checked by sha256 in the Dockerfile; Rust 1.98.1 from static.rust-lang.org (rustup checks the dist manifest hashes). Not pinned: the Ubuntu packages (gcc, binutils, make, git, curl), see Nondeterminism. |
| Environment | Linux on arm64 (aarch64) with Docker (the agent run used Docker 29.2.1), git, bash, GNU coreutils and util-linux (`timeout`, `nice`, `ionice`). Network access to github.com, registry-1.docker.io, ports.ubuntu.com, static.rust-lang.org, index.crates.io and static.crates.io. About 3 GB of free disk. No GPU, no NVIDIA software, no account and no key. |
| Architecture assumptions | aarch64 host and aarch64 target (`aarch64-unknown-linux-gnu`). The Dockerfile downloads the aarch64 rustup-init. On an x86_64 host the result is undeclared: emulation is not part of this recipe. The engine's GPU code is compiled in but never executed by this check. |
| Randomness and seeds | None. The build uses no random input. `SOURCE_DATE_EPOCH=1793446268` (the Core commit time) is set by `inside.sh`, and paths are remapped (`mode=remap`) so the work directory does not reach the binary. |
| Resource bounds | CPU only. `REPRO_CPUS=2-4` (three cores, `docker --cpuset-cpus`), `REPRO_JOBS=3`, `REPRO_MEMORY=3g` (`docker --memory` and `--memory-swap`), all under `nice -n 19 ionice -c3`. Any three cores work; the bounds change only whether and how fast the build finishes, never the bytes. If the build is killed for memory, raise `REPRO_MEMORY` and say so in the report. |
| Timeout semantics | `timeout 3h` around the build. On expiry `timeout` exits 124 and the result is NOT_RUN, not a mismatch. Agent runs on the operator's host took between 3 and 7 minutes with a warm image. |
| Nondeterminism policy | Exact bytes. The result is a match only if the full sha256 is identical; there is no tolerance. Known source of variation: the Ubuntu packages are taken from noble, noble-updates and noble-security on the day the image is built, so a later gcc or binutils update can change `libomega_gpu.a` and with it the engine. Step 3 records the versions; the original build used gcc 13.3.0-6ubuntu2~24.04.1, binutils 2.42-4ubuntu2.10, make 4.3-4.1build2, libc6-dev 2.39-0ubuntu8.9. A mismatch with different versions is reported as a mismatch with that difference named, not explained away. |
| Inputs | The three source commits above, fetched by full sha over https by the script, plus the crates named in Core's `Cargo.lock`. Nothing else is read from the host. |
| Exact expected result | `036b7491d85a2d0781360c194402f345ad2f63720dab1b8f6d0cbc5c09f87d3e  libomega_gpu.a` and `6fb04197c272aa1d0235e7644b6e91953ff9b90f5145b250d674ff9fe41806e9  aien-serve-http`, printed in `out/BUILD-INFO.txt`; the script exits 0 only if both match. |
| How to report | See [Report your result](#report-your-result). |

## Steps

Run these in a POSIX shell on the arm64 Linux host. They are the exact commands the agent run followed.

1. Fetch the kit into a fresh directory.

   ```sh
   R=$(mktemp -d /tmp/recheck-XXXXXX); echo "$R"
   git clone https://github.com/aien-dev/aien-sovereign-core-mv.git "$R/kit"
   cd "$R/kit"
   git checkout dhgmonkey94/dra-94-publish-the-independent-recheck-count-and-a-recheck-kit-for
   git rev-parse HEAD
   ```

   Once this kit is merged, check out `main` instead. The branch name does not pin the kit; step 2 does.

2. Check the verifier digests. Stop if any line is not `OK`.

   ```sh
   sha256sum -c <<'EOF'
   26db7be518f0630fde85d28bfd50007353802be0d03d0e8a945fd09a3a6c14ac  scripts/repro-container-build.sh
   2f40abd645d9a21868c8712b06149b8a1fa763bdce3dc9cabc211c626734998f  release/repro/Dockerfile
   5a09da775509ace488f4fb732a40837c6636bb7d884d3e24b904d822f37cdae1  release/repro/inside.sh
   EOF
   ```

3. Build the image from scratch and record what it contains. The script reuses an image with this tag if one exists, so building it here with `--no-cache` makes the run clean.

   ```sh
   TAG="aien-repro-build:$(sha256sum release/repro/Dockerfile | cut -c1-12)"
   nice -n 19 ionice -c3 docker build --pull --no-cache -q -t "$TAG" release/repro
   docker run --rm --network none "$TAG" sh -c 'uname -m; dpkg-query -W -f="\${Package}=\${Version}\n" gcc gcc-13 binutils libc6-dev make; rustc -V; cargo -V'
   ```

4. Rebuild and compare. The script fetches the three commits, builds inside the container and checks both hashes.

   ```sh
   CORE_REV=d3dec0eb18818808212bdf99db5c8612bcfcb9ef \
   OMEGA_REV=a647b6a9821fffcf52ba9ce813f75afa4b84be62 \
   PHYSICS_REV=048e5f5e881b092733baaf65d0ace46abda7f908 \
   EXPECT_LIB_SHA256=036b7491d85a2d0781360c194402f345ad2f63720dab1b8f6d0cbc5c09f87d3e \
   EXPECT_ENGINE_SHA256=6fb04197c272aa1d0235e7644b6e91953ff9b90f5145b250d674ff9fe41806e9 \
   REPRO_CPUS=2-4 REPRO_JOBS=3 REPRO_MEMORY=3g \
   nice -n 19 ionice -c3 timeout 3h bash scripts/repro-container-build.sh "$R/work"; echo "exit=$?"
   ```

5. Read the result.

   ```sh
   cat "$R/work/out/BUILD-INFO.txt"
   sha256sum "$R/work/out/aien-serve-http"
   ```

   `exit=0` and the expected engine hash is a match. A nonzero exit after the build finished, with a different hash in BUILD-INFO.txt, is a mismatch. `exit=124` is a timeout: NOT_RUN.

## Report your result

Open an issue on https://github.com/aien-dev/aien-sovereign-core-mv/issues titled `Recheck: engine sha256 d3dec0eb (match)` or `(mismatch)` and attach:

- the full terminal output of steps 1 to 5 (for example from `script -q recheck.txt`),
- `$R/work/out/BUILD-INFO.txt`,
- `uname -a` and `docker version`,
- whether you are connected to the project (an outside recheck counts only if you are not).

A public gist or a signed note works too; link it in an issue so it can be found. A mismatch is as useful as a match. The independent recheck count changes only when an outside person's recheck is linked.
