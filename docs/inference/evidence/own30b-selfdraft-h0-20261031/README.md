# own30b self-draft H0, provenance rerun in the quiet window (2026-10-31, CoreChip)

The 2026-10-18 receipts in `../own30b-selfdraft-h0-20261018/` came from a binary built from an uncommitted
diff that no longer exists, with no chip lock and no telemetry. This rerun uses committed source only.

- Source: PR head `045c7c5495dab8b2914ab49e2c3f3e2350d91616` (own30b/self-draft-mask merged with main ab1084d6).
- Binary: `own30b_selfdraft` built natively from that head (`--locked --release`, `AIEN_REQUIRE_NATIVE_LIBS=1`,
  omega a647b6a9 as in `omega.lock`), sha256 `9e986a97a9e634efc03d7d5e4516a8bdf35d220b64bdbc3df07ee8f6d1003689`.
- Command: `env AIEN_OMEGA_SPIN_US=2000 own30b_selfdraft ~/models/qwen3-30b-a3b-instruct-2507-fp8 <out>
  --new 256 --rs 16,32,64 --ks 4,8,16,32 --cross-check 3 --force-memory` (the 10-18 `full.start` command).
- Flint stopped, spark-window 20261031T072422Z-7e1d4e (`--gb10 shared --quiet none`), run under
  `flock -s /tmp/aien-gb10.lock`, telemetry `selfdraft-full.dmon` (`nvidia-smi dmon -s pucm -d 1`).
  07:24:22Z to 08:08:13Z, rc 0.

Result: `SUMMARY eval_prompts 20 tokens 4886 cross_check PASS`. All 12 `ACCEPT` rows, all 9 `CROSS` rows and
the `SUMMARY` line are byte-identical to `../own30b-selfdraft-h0-20261018/full.raw`
(`diff <(grep -E '^(ACCEPT|CROSS|SUMMARY)' ...)` empty). The 10-18 numbers are therefore reproduced by a
committed, recorded binary under the chip lock. `SHA256SUMS` lists the receipts.
