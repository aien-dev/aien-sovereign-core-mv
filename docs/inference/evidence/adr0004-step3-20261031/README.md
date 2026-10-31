# ADR 0004 step 3, G30 rerun in the quiet window (2026-10-31, CoreChip)

Rerun of the G30 tokens and speed gates of `../adr0004-step3-20261015/` on the merged head, with Flint stopped
(GPU compute-apps empty at start of each step), driver `/tmp/corechip/quiet/driver.sh` under
`systemd-run --user --unit corechip-quiet`.

- Source: PR head `9e6ceefdcbae4b649c61605c134dd253d757ae0c` (adr0004-step3 merged with main ab1084d6).
- Binary: `own30b_check` built natively from that head (`--locked --release`, `AIEN_REQUIRE_NATIVE_LIBS=1`,
  omega a647b6a9 as in `omega.lock`), sha256 `99594b5d6bef2bf33389d9f557709da4807bd53ec39eca113cc9e6dfd8e2517c`.
- Command (both steps): `env AIEN_OMEGA_SPIN_US=2000 own30b_check chip ~/models/qwen3-30b-a3b-instruct-2507-fp8
  ../own30b-f-20261013T1745Z/golden.json <out> --prompts 1,2,3,4 --new 16 --layers 0,23,47 --weights fp8
  --attn fused --router device --lm-head gemv --prefill auto --force-memory`.
- Locks: `g30-tokens` in spark-window 20261031T071217Z-ea2cba (`--gb10 shared --quiet none`) under
  `flock -s /tmp/aien-gb10.lock`; `g30-speed` in spark-window 20261031T071440Z-97707a (`--gb10 none --quiet
  exclusive`, verdict "quiet machine") under `flock -x /tmp/aien-gb10.lock`. Telemetry: `<step>.dmon`
  (`nvidia-smi dmon -s pucm -d 1`). Each step's `.start` records commit, binary sha, admission and box state.

| Step | rc | Result |
|---|---|---|
| g30-tokens | 0 | `SUMMARY parity=PASS tokens=PASS`; `first_diff` null on 4 prompts; sha256 of `jq -c '[.prompts[] \| .chip]' run.json` begins `bb1a0401175ab69d`, the same as `../adr0004-step0-baseline-20261015/check-full-3`: IDENTICAL |
| g30-speed | 0 | same token hash `bb1a0401175ab69d`; decode tok/s p1 44.7, p2 42.2, p3 43.5, p4 42.2 against floors 37.927 / 35.017 / 36.375 / 35.502: all above |

Decode tok/s are the binary's own `TOKENS` lines in `g30-speed.raw` (16 tokens per prompt, p2 stops at 7 on eos).
`SHA256SUMS` lists the receipts.
