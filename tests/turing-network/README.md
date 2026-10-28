# TUR-R0 native Omega conservation fixture

Status: SOURCE_COMMITTED / OMEGA_COMPILE_NOT_RUN / PHYSICAL_NETWORK_NOT_RUN.

This test-only program belongs to [Core #738](https://github.com/aien-dev/aien-sovereign-core-mv/issues/738) and the four-peer [Aienet-0 interoperability proposal](https://github.com/aien-dev/aien-protocols-mv/blob/feat/aienet0-foundation-20261027/specs/aienet/P0_INTEROP_PROFILE.md). It exercises deterministic **integer** accounting within one in-memory Omega process. It is an **independent arithmetic oracle for future Core escrow**, not a wallet, persistent ledger, authenticated transport, billed job or network deployment.

File: `tests/turing-network/tur_r0_oracle.om`

## Exact host test

Run from a clean checkout on **Linux AArch64**, with an independently pinned and rebuilt Omega compiler:

```sh
# 1. In your separately checked-out aien-dev/omega-mv worktree:
sh seed/omega-from-seed.sh
sha256sum seed/out/omc
OMC="$PWD/seed/out/omc"

# 2. In this aien-sovereign-core-mv PR worktree:
mkdir -p build/turing-network
"$OMC" < tests/turing-network/tur_r0_oracle.om > build/turing-network/tur_r0_oracle
chmod 700 build/turing-network/tur_r0_oracle
sha256sum tests/turing-network/tur_r0_oracle.om build/turing-network/tur_r0_oracle
build/turing-network/tur_r0_oracle
# Expected stdout, exit 0: TUR_R0_ORACLE_PASS
```

**Important:** the command comments are instructions, not proof that these steps were executed. In the current proposal session no AArch64 compiler binary or Spark execution environment was available; all host and chip execution remains NOT_RUN. Preserve stdout, stderr and status separately when reproducing.

## Expected in-memory transitions

- Opening: requester=100, provider=0, verifier=0, observer=0, escrow=0.
- Refuse overdraft reservation of 101; reserve 40 against logical job 7; refuse a second reservation.
- Refuse settlement against wrong job 8; refuse a 41-unit payout from 40-unit escrow.
- Settle job 7 exactly once: provider 27, verifier 3, requester refund 10.
- Refuse identical settlement on replay. Final requester/provider/verifier/observer=70/27/3/0, escrow=0.
- Every assertion has its own nonzero exit code; only a complete pass prints the PASS marker.

Independent **Wolfram Language** reference evaluated the same transitions, returning all expected boolean verdicts, final balances `{70,27,3,0}`, escrow zero, and conservation TRUE. That confirms the mathematical scenario, **not the Omega source executable**.

To test a deliberately broken expected balance, alter only the assertion for the provider's final balance from `27` to `28` in a disposable copy, rebuild it, and require exit 14 (or report precise compiler failure). Never overwrite the original fixture.

## What remains before an economic or network claim

1. Integrate with **one real, durable authoritative Core fixed-point service-credit journal**, rather than promoting this in-memory reference into a new wallet.
2. Bind ownership and payments to canonical MJB1/MJR1 job and independent verifier receipts, signed peer identities and a crash-safe idempotency key.
3. Qualify each signature, concurrency, cancellation, restart boundary, corruption, overpayment and release/refund negative control.
4. Connect to actual Open Humanity transport **only after** mutual opt-in and separated owners. Complete a two-physical-node work exchange before claiming network operation.
5. Keep profile-bound scientific Turing information separate from pre-funded, spendable TUR test credits. This example issues no credit and has no monetary value.

Do not send jobs to the live 35B server or acquire the GB10 for this CPU-only fixture.
