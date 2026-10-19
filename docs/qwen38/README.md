# Qwen3.8-27B on AIEN's Omega engine: model, checks, ceiling

- Start here: [TRY_IT_ON_YOUR_SPARK.md](TRY_IT_ON_YOUR_SPARK.md), the step by step guide for a Spark
  with nothing of AIEN on it. What "alpha" covers and what stays pre-alpha, and the fresh-account
  test receipt once it exists: [ALPHA_STATUS.md](ALPHA_STATUS.md).
- Report a run, working or not, with the alpha tester issue template
  (`.github/ISSUE_TEMPLATE/alpha-tester-report.md`).
- Model: Hugging Face `Qwen/Qwen3.8-27B-FP8`, revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`,
  Apache 2.0, 81 files, 30,890,049,597 bytes (about 30.9 GB). Download:
  `hf download Qwen/Qwen3.8-27B-FP8 --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a --local-dir ~/models/qwen3.8-27b-fp8-017b9c7a`
- Checksums: `qwen3.8-27b-fp8-017b9c7a.sha256` lists the sha256 of all 81 upstream files. Check a
  download with `cd ~/models/qwen3.8-27b-fp8-017b9c7a && sha256sum -c <repo>/docs/qwen38/qwen3.8-27b-fp8-017b9c7a.sha256`.
  The 67 large (LFS) files match the sha256 Hugging Face publishes for this revision; the 14 small
  files have no published sha256, so their values are the ones we downloaded (sizes matched upstream).
  Verified on the project's Spark on 2026-10-19.
- Start the server: `tools/serve-27b.sh` (installed as `aien-serve-27b` by `install.sh --with-server`).
- Check a running server: `tools/qwen38_serve_gate.sh` (correctness against the shipped host
  reference) and `tools/qwen38_serve_speed.sh` (speed). Both write a `receipt.json` with the
  server binary sha256, its build, the model folder, the machine state, every request's figures
  and a MATCH or MISMATCH line against the references in
  `crates/aien-inference-abi/fixtures/qwen38/`. `AIEN_CHECK_MODEL=1` also checks the model files.
- Speed limit set by memory: [CEILING.md](CEILING.md).
