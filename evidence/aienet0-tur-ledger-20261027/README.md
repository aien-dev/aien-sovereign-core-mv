# Aienet-0 TUR escrow ledger: host evidence (2026-10-27)

Label: **HOST_TESTED**. One machine (DGX Spark, Linux 7.0.0-1019-nvidia aarch64), no GPU, no network.
Not EXERCISED across two machines; that is the job-path lane's evidence.

Source commit: `5bd01b6` on `feat/tur-ledger-r1` (the files below are byte-identical at the PR head).

| Input | sha256 |
|---|---|
| seeded omc (omega-mv `seed/out/omc`, pin e5b32567) | `2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f` |
| `om/vendor/sha256.om` | `323a71e762c0809f07f53c0e6438023e9111ce2acdeda5cb31eef92f2946f000` |
| `om/tur-ledger/tur-ledger.om` | `ad3b61ac7859250e51a9709a7c6d1751f512db7e3606a0ef962b5677634f09e4` |
| `om/tur-ledger/check.sh` | `8632b0c33379bc92c27ad9b5dbd0281df9c433f8f96cde4bc2ba5743fb0fd4db` |
| `tests/turing-network/tur_r0_oracle.om` (#791) | `92b790d4f63ac01c55ca8e8dd0aab5fbebbef5babbe06e30d24c225cf8c23b6f` |
| built binary (`/tmp/an0/bin/tur-ledger`, 37996 bytes) | `6ea978c7f0ae080c6bb0d0254f9fa56b78da2e74d577c092a98cd50bd945be4a` |

Tools: strace 6.8, util-linux flock 2.39.3, GNU coreutils sha256sum/timeout.

## Files

| File | Command | Result |
|---|---|---|
| `check.out` | `OMC=<seed omc> TUR_LEDGER_OUT=/tmp/an0/bin/tur-ledger bash om/tur-ledger/check.sh` | exit 0, `SUITE PASS` (50 checks), 7 of 7 mutants killed, `TUR_LEDGER gate: PASS` |
| `tur_r0_check.out` | `OMC=<seed omc> sh tests/turing-network/check.sh` | exit 0, `TUR_R0 gate: PASS` (the oracle this ledger follows) |
| `fsync-and-journal.out` | oracle settle under `strace -e trace=pwrite64,fsync,write`, the replayed settle, `audit`, the journal, and each line's HASH recomputed with `sha256sum` | `pwrite64(3, ...)` then `fsync(3) = 0` then `write(1, "settled ...")`; 4 of 4 lines verify |

`SHA256SUMS` covers the three output files.

## What each guard's mutant proves

Each mutant is the source with one guard removed by `sed`; the same suite must then fail.

| Mutant | Guard removed | Failing checks in `check.out` |
|---|---|---|
| no-fsync | `fsync` of the journal before the reply | fsync.before-ack |
| no-lock | `flock(LOCK_EX)` for the whole command | concurrent.reserve-20, lock.blocks-while-held |
| no-exactly-once | the settled-JOB branch of settle | oracle.zero-replay, once.*, concurrent.settle-*, crash.kill9 |
| no-hash-chain | HASH and PREV checks during replay | torn.bad-hash-last-line, corrupt.amount-edited |
| no-conservation | the conservation check after replay | conserve.payout-over-escrow, conserve.refused-for-every-command |
| no-escrow-rule | `PAYP + PAYV + BACK = AMOUNT` on live settle | oracle.change-mismatch and the oracle balances |
| no-torn-recovery | torn-tail truncation of the last line | torn.bad-hash-last-line |

## Limits

- The kill -9 run (200 commands, 95 killed, 80 of them after the record was durable) shows that a
  process kill at any point leaves a journal that audits clean and settles nothing twice. A kill
  cannot tear a single `pwrite`, so torn tails are tested by injecting them (three shapes), not by
  power loss. Power-loss durability rests on `fsync` before the reply (strace order), not on a test.
- Arithmetic values come from the #791 oracle (requester 70, provider 27, verifier 3, observer 0,
  escrow 0 of 100) and from differential runs (crash journal byte-identical to the control journal).
