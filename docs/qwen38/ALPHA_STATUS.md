# Alpha status: what "alpha" covers and what stays pre-alpha

Last updated 2026-10-19. Plain statement of where the line is, so nobody reads more into the
word "alpha" than it means here.

## What alpha covers

One thing: the Qwen3.8-27B model server built from this repository, running on AIEN's own
Omega engine on a DGX Spark.

- Install: `install.sh --with-server --with-model` from public sources (this repository plus
  the pinned public `omega-mv`, `physics-mv` and `aienos-mv` repositories). Source build only.
- Model: Hugging Face `Qwen/Qwen3.8-27B-FP8`, revision
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, downloaded by `--with-model` and checked against
  the list in [qwen3.8-27b-fp8-017b9c7a.sha256](qwen3.8-27b-fp8-017b9c7a.sha256).
- Run: `aien-serve-27b` (the shipped `tools/serve-27b.sh`), OpenAI-style chat completions on
  `http://127.0.0.1:18091/v1`, text only, one request at a time, no tool calls.
- Check: `tools/qwen38_serve_gate.sh` and `tools/qwen38_serve_speed.sh`, each writing a
  `receipt.json` to compare with our evidence.
- Guide: [TRY_IT_ON_YOUR_SPARK.md](TRY_IT_ON_YOUR_SPARK.md), with the known limits in its
  section 7.

Tested setup: DGX Spark (GB10, aarch64), Ubuntu 24.04.5, NVIDIA open kernel driver 580.173.02.
Nothing else.

## What stays pre-alpha

Everything else, and the wording in their READMEs stays as it is:

- The rest of this repository: the `aien` command line tools, Cortex, the cockpit, the
  supervisor, the daemon and its first-run flow (`docs/FIRST_RUN.md`), the OWN-30B server path,
  releases and release signing (no public key, no aarch64 release archive).
- The AIENOS kernel (`aienos-mv`), the Omega compiler and runtime work beyond the kernels this
  server links (`omega-mv`), physics and FORGE (`physics-mv`), and the other AIEN repositories.
- Anything not measured through this server: engine-only speed figures, first-token times on
  the current build, power, long-context behaviour above about 30,000 prompt tokens.

## The alpha label itself

The server is called alpha only after one run on a brand-new user account on a Spark, with
nothing of ours in it, following [TRY_IT_ON_YOUR_SPARK.md](TRY_IT_ON_YOUR_SPARK.md) as written,
ends in a chat reply. The transcript of that run is the receipt, and it is linked here.

- Fresh-account test receipt: NOT YET RUN. (Placeholder: when the run exists, put the path to
  its transcript and the commit it was run on here. Until then this server is a release
  candidate for alpha, not alpha.)

## Evidence behind the numbers in the guide

All on the live server built from sovereign-core `46134247` with omega `0010f7d3`, binary
sha256 `df936011...` (`evidence/qwen38-serve-fast-20261019/README.md`):

- Decode speed 7.63 tokens per second, median of four runs:
  `evidence/qwen38-serve-fast-20261019/after/runs.tsv`; share of the 10.15 tokens per second
  memory ceiling 0.752: [CEILING.md](CEILING.md) and `calc-receipts/`.
- Ceilings: memory ceiling 10.15 tokens per second for one token at a time (8.14 at our measured
  219 GB/s); compute ceiling about 4,050 tokens per second in total at the 207.7 TFLOPS FP8
  figure measured by a third party, 4,880 at NVIDIA's specification, about twice that with FP4
  weights. Arithmetic, sources and which figure is measured: [CEILING.md](CEILING.md).
- Memory: peak resident memory 36.7 GiB (38,499,160 kB, VmHWM) of the live server on this
  binary: `evidence/qwen38-serve-memory-20261019/proc-status.txt`, calc:39181948. Plan on about
  that much free for a second 27B server on the same Spark. Of the peak resident memory now, 35,469,640 kB is file-backed
  (RssFile in the same file); whether a second server reading the same model folder shares part
  of it has not been measured.
- Model download: `install.sh --with-model` run on our Spark (2 files fetched, 1 resumed, all 81
  pass `sha256sum -c`): `evidence/qwen38-install-with-model-20261019/realrun.log`.
- Correctness gate `GATE PASS`, 24 of 24 greedy ids equal to the host reference:
  `evidence/qwen38-serve-fast-20261019/gate-live/summary.txt`.
- 256-token answer identical across four runs (ids hash `ceb68306ae2d`):
  `evidence/qwen38-serve-fast-20261019/after/runs.tsv`.
- Model checksums: 67 of 67 large files match the hashes Hugging Face publishes
  ([README.md](README.md)).

Not yet in place, said plainly: no signed verification bundle in `aien-receipts` names these
runs (the project's gate list has no entry for the serve gate or the speed test yet), and our
evidence folders predate the `receipt.json` format, so a tester's receipt is compared against
the raw files above rather than against a receipt of ours.
