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

## Authentication v2 addendum (2026-10-28, Core #793)

The sections above and their three receipts describe unsigned v1. They are preserved byte for
byte and do not qualify admission or authorization. This addendum fixes the findings in
[the full #792 review](https://github.com/aien-dev/aien-sovereign-core-mv/pull/792#issuecomment-6466130214)
forward from main `1604a72b248aba8b0e41e5379814372d8fd59304`.

**HOST_TESTED**, same Linux ARM64 host, no GPU, production key, live model, network service,
wallet or cryptocurrency. Source: `om/tur-ledger/tur-ledger.om`; native signature verification:
`om/vendor/ec.om`, copied unchanged from omega-mv
`6999fa9e6dae4847c155ef4eacaece74f491d3ca:om/tls/ec.om`, sha256
`909f37c797e187bd0663bde9f57c75d6ce41eb2c9726a6b50d6fc714e424c060`.
Seeded compiler sha256:
`d6290c7663c2b4bafb59eb3fdf812476e977f4c144def71095c6248ef9b96b84`.
OpenSSL **3.0.13 (30 Jan 2024)** only creates test signatures; it is not linked into, invoked by
or needed to run the ledger. Disposable test private keys were generated in private tmpfs and
removed by the gate. No signer or key-management implementation is shipped here.

The journal genesis and trusted `authority` config pin the issuer, a distinct verifier, ledger
identity, policy and exact requester/provider/verifier owner-account map. Every external
mutation has a canonical-record issuer signature. Settle additionally requires the verifier to
sign the **same complete authorization**, including reservation commitment, published quote,
exact payout split and signed-job/result/AN0V digests. This follows the design review's decisive
point: "The verifier must sign the complete settlement authorization, including reservation
commitment and payout split, not just result digests." Distinct keys alone are not operational
independence. Complete unauthorized records refuse replay rather than becoming recoverable tails.
Version-1 journals refuse unchanged; reconciliation requires an explicitly provisioned new
ledger identity, never silent migration.

### New receipts

`auth-v2-SHA256SUMS` covers only the new receipts and current source inputs. The original
`SHA256SUMS`, `check.out`, `tur_r0_check.out` and `fsync-and-journal.out` are unchanged.

| Receipt / command | Observed |
|---|---|
| `auth-v2-before.out`: `OMC=/tmp/ledger-auth-omega/seed/out/omc TUR_LEDGER_AUTH_ONLY=/tmp/ledger-auth-before bash om/tur-ledger/check.sh` | Exit 1: **FAIL auth.unsigned-prefund**, actual ledger exit 0, expected 2; journal changed. Before binary `6ea978c7f0ae080c6bb0d0254f9fa56b78da2e74d577c092a98cd50bd945be4a`, built from main's v1 source and original SHA unit using the same compiler. |
| `auth-v2-after.out`: same command with `TUR_LEDGER_AUTH_ONLY=/tmp/ledger-auth-bin-final` | Exit 0: **PASS auth.unsigned-prefund**, ledger exit 2, unchanged journal. Exactly the same assertion. |
| `auth-v2-check.out`: `OMC=/tmp/ledger-auth-omega/seed/out/omc CRASH_N=200 MUTANT_CRASH_N=0 TUR_LEDGER_OUT=/tmp/ledger-auth-bin bash om/tur-ledger/check.sh` | Exit 0: **SUITE PASS**, **10/10 mutants killed**, **TUR_LEDGER gate: PASS**. Mutant crash trials omitted explicitly; honest-binary crash trials executed. |

The suite exercised unsigned prefund/reserve/settle/refund refusal; wrong issuer and verifier
keys; altered amount, job, quote, result, signed-job and AN0V digest refusal; account spoofing;
issuer-key substitution for verifier; validly re-signed changed evidence conflicting with an
already-settled job; complete unsigned/wrong-key journal tails; v1 refusal and changed policy.
Each rejection asserts the journal hash is unchanged. The concurrent double-debit control
accepted exactly one reserve with only one unit funded. Honest settle paid provider 27 once;
the same signed semantics replayed as `already-settled`.

A deterministic strace SIGKILL at stdout-write entry occurred **after fsync, before acknowledgement**;
the gate observed exit 137, successful fsync, an already-settled retry and provider balance 27.
The additional random-delay kill run killed all 200 attempted processes before a new durable
write (durable-before-kill 0). Each was audited and retried, final balances matched the control.
Journal bytes did not match the independently signed control because ECDSA signatures differ.
Do not read this run as 200 post-fsync kills: that boundary is the separate deterministic control.
Injected torn tails test framing recovery, not hardware power loss.

The full gate compiled source hash
`960f4bb581e3c63cb193ef164736cf5394441ff580515be563b60cc42a53f4af`.
Only its build-instruction and exit-code comments were corrected afterwards: final source
`b188d4c8324eb1d99f9d5e0d08549c54358afca369171974433b456f59ae4ca6`.
Recompilation and `cmp /tmp/ledger-auth-bin /tmp/ledger-auth-bin-final` exited 0; both binaries
hash `d12d05cda56604410b8716602a7e3988a0e90901d28294f04f6ea7d8081f435c`.
The unsigned-after receipt uses that final binary.

The shared SHA unit changed only its private `hs` state name to `hstate` and its explanatory
header, avoiding a name collision with the vendored EC parameter. Compiling `om/effects/world.om`
with original versus renamed SHA produced byte-identical binaries:
`08e1a63d722250d93dbec0574f902458262eb5e64b0b5e63b2f0eb70f9e3af0d`.
Both executions on `abc`, authorization 2, intent 3, returned 0, state DONE and record digest
`6281e53eed299bdb5467ccbfa7f87ae78c547a08660b9971df362c1680f048d1`.
This is a consumer smoke/parity check, not a claim that its separately pinned full Rust suite ran.

### Scope boundary

There is no `scripts/aienet0/settle-run.sh` or Core ledger RPC in this Core revision.
The external signer must verify the actual signed-job/MJR1/AN0V artifacts and quote/policy before
issuing these typed authorizations. The ledger verifies canonical bytes and signatures, not
those artifact parsers. Mutated-AN0V-digest refusal is **not** an external AN0V parser test;
lost CLI acknowledgement is **not** interrupted-RPC qualification. The application contract is
documented in `om/tur-ledger/README.md`; no compiler or shared guide language feature changed.
No deployment, malicious-filesystem-write/rollback resistance or public economic authority is claimed.
