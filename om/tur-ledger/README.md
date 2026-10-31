# tur-ledger: authenticated Aienet-0 TUR test-credit escrow

The native Omega CLI keeps an append-only, single-writer journal with integer conservation,
exactly-once settlement, torn-tail recovery and fsync before acknowledgement. Version 2 adds
issuer authorization for every external mutation and independent verifier authorization for
settlement. It is a local test-credit ledger, not a wallet, currency or public economic service.

Build with omega-mv `6999fa9e6dae4847c155ef4eacaece74f491d3ca`, seeded omc sha256
`d6290c7663c2b4bafb59eb3fdf812476e977f4c144def71095c6248ef9b96b84`:

```sh
{ printf 'om2\n'; cat om/vendor/sha256.om om/vendor/ec.om om/tur-ledger/tur-ledger.om; } | "$OMC" > tur-ledger && chmod +x tur-ledger
OMC="$OMC" bash om/tur-ledger/check.sh
```

`om/vendor/ec.om` is byte-identical to omega-mv's `om/tls/ec.om` at that revision, sha256
`909f37c797e187bd0663bde9f57c75d6ce41eb2c9726a6b50d6fc714e424c060` (Apache-2.0).
It verifies strict DER ECDSA P-256 signatures entirely in Omega; no host crypto runtime is used.
The SHA-256 unit's private state global is renamed `hstate` to avoid an EC parameter collision.
The gate uses OpenSSL only as an independent signing oracle, plus strace, flock and timeout.
Disposable test keys live in a private tmpfs directory and are removed on exit. No production
private key belongs in this repository or in a ledger directory.

## Provisioning and authority

Before any CLI invocation, the trusted Core service operator provisions `DIR/authority` as
exactly these ASCII bytes, with single spaces and **no trailing LF**:

```text
genesis tur-ledger-2 LEDGER_ID ISSUER_SEC1 VERIFIER_SEC1 POLICY_SHA256 REQUESTER PROVIDER VERIFIER_ACCOUNT REQUESTER_OWNER PROVIDER_NODE VERIFIER_NODE
```

`LEDGER_ID`, the policy digest and three owner/node identity pins are 64 lowercase hex characters.
Each public key is 130 lowercase hex characters: SEC1 uncompressed P-256 (`04 || X || Y`). The
issuer and verifier keys must differ. The three account names must differ and are 1..64 bytes
of `[A-Za-z0-9._:-]`. This bounded policy maps exactly one requester owner, provider node and
verifier node to their account names. Only the pinned requester account can be funded; reserve
must name those exact three accounts. Arbitrary account registration is deliberately absent.

The issuer is the sole Core service authority, not an arbitrary local CLI caller. Its signature
means funding-source approval for the unique funding event `PID`, or approval of the reserved
job, published quote, account-owner/node mapping and policy. `PID` identifies one funding event;
reusing it with different content is refused. The issuer must not assign another PID to that
same funding event. The verifier signature means it independently checked the signed job,
MJR1 result, AN0V attestation, published quote, pinned owner/node identities and policy and
approved this exact payout. Distinct keys do not by themselves establish operational independence.

Both signers are external authorities. The ledger verifies their typed authorizations, not
MJR1/AN0V artifact parsers or signer policy implementation. Signing unverified caller-provided
hashes violates this contract. No arbitrary receipt hash or issuer-only authorization can settle.
The ledger needs neither private keys nor a new wallet/cryptocurrency primitive.

Protect the directory, config, provisioning process and authority keys with the service's OS
permissions. This design does not protect against an actor who can replace both config and
journal, roll back a valid signed history or clone an existing ledger identity. The SHA chain is
integrity/recovery evidence, not a signature or malicious-file-write defense.

## Canonical signed bytes

Algorithm is fixed for v2: ECDSA P-256 over SHA-256, signature encoded as strict DER then lowercase
hex (no whitespace). Sign the SHA-256 digest of these bytes, not an externally supplied digest:

```text
EXACT_AUTHORITY_CONFIG LF
CANONICAL_UNSIGNED_COMMAND LF
```

For `settle` and `refund`, append the 64 lowercase hex SHA-256 digest of the reserved job's
canonical signed bytes (excluding its signature), followed by LF. This commits settlement to
the exact reserve: JOB, requester/provider/verifier accounts, amount, published quote and
config's owner/node/policy pins. Journal SEQ/PREV/HASH and signatures are excluded from signing.
Each CLI argument must be a separate field and must not contain ASCII whitespace/control bytes;
only then are arguments joined with one ASCII space. Amounts are canonical decimal u64 with no
leading zeros. Every field is covered. The ledger recomputes this digest during admission and
replay. Signatures are checked **before** idempotent success, and re-signing identical semantic
content cannot create another economic action.

## Commands

`tur-ledger --journal DIR COMMAND`. `ISIG` is the issuer signature; `VSIG` is the independent
verifier signature over the **same canonical bytes**.

| Command | Success output |
|---|---|
| `prefund PID REQUESTER AMOUNT ISIG` | `prefunded PID REQUESTER AMOUNT`; repeat: `dup ...` |
| `reserve JOB REQUESTER PROVIDER VERIFIER_ACCOUNT AMOUNT QUOTE_SHA256 ISIG` | `reserved JOB REQUESTER PROVIDER VERIFIER_ACCOUNT AMOUNT`; repeat: `already-reserved ...` |
| `settle JOB RECEIPT_SHA256 PAYP PAYV BACK SIGNED_JOB_SHA256 AN0V_SHA256 ISIG VSIG` | `settled JOB RECEIPT_SHA256 PAYP PAYV BACK`; repeat: `already-settled ...` |
| `refund JOB ISIG` | `refunded JOB`; repeat: `already-refunded JOB` |
| `balance ACCT` | `balance ACCT N` |
| `show JOB` | `job JOB reserved\|settled\|refunded REQUESTER PROVIDER VERIFIER_ACCOUNT AMOUNT [RECEIPT PAYP PAYV BACK]` |
| `audit` | `audit ok records R head H accounts A jobs J open O settled S refunded F prefunded P balances B escrow E` |

JOB is the 64-lowercase-hex MJB1 dag_id. QUOTE_SHA256 hashes the exact published quote bytes;
SIGNED_JOB_SHA256 hashes the exact signed MJB1 envelope bytes; RECEIPT_SHA256 hashes the exact
signed MJR1 result bytes; AN0V_SHA256 hashes the exact signed AN0V attestation bytes, including
signatures. All digests are lowercase hex. Amounts are nonzero except PAYP/PAYV/BACK. A duplicate
reserve conflicts if its quote changes; a duplicate settle conflicts if any evidence digest or
payout changes, even when both replacement signatures are valid.
An identical reserve retry returns `already-reserved` only while that job is still open.
After settlement or refund it refuses 10; callers must inspect the actual terminal phase
rather than interpreting an old reservation authorization as current escrow.

## Refusals

Refusals print `refused C reason` on stderr and exit C. On an intact journal, an unsuccessful
admission writes nothing. Recovery is a separate internal operation and can run before dispatch.

| C | reason | when |
|---|---|---|
| 2 | usage | missing signatures, bad arity, whitespace inside an argument, noncanonical fields or unknown command |
| 3 | journal-io | journal open/read/write/fsync or directory-fsync failure; reconcile before treating it as aborted |
| 4 | lock | flock failed |
| 5 | journal-corrupt | old genesis, config mismatch, invalid complete record/signature, bad interior line or linked-chain failure |
| 6 | conservation-violated | replayed balances/open escrow/payout totals disagree |
| 7 | insufficient-funds | reserve exceeds requester balance |
| 8 | unknown-job | authenticated settle/refund/show for unknown JOB |
| 9 | job-conflict | different reservation authorization for JOB |
| 10 | wrong-phase | reserve of a terminal JOB, refund of settled JOB or settle of refunded JOB |
| 11 | settle-receipt-conflict | different settlement authorization for settled JOB |
| 12 | escrow-rule | payout split differs from escrow |
| 13 | prefund-id-conflict | PID reused with different authorization |
| 14 | capacity | an operation would allocate more than 65,536 keys |
| 15 | overflow | u64 total would overflow |
| 16 | authorization | missing/invalid authority config, account policy mismatch or invalid issuer/verifier signature |

## Journal version and recovery

`DIR/journal` has `SEQ PREV KIND FIELDS... HASH` records. HASH is SHA-256 of the line before
` HASH`; PREV is the previous HASH (64 zeros for genesis). Genesis is the exact authority config.
Every mutation's record retains its signature(s). Every invocation locks the journal for the
entire command and verifies every admission signature on replay. Every malformed
**newline-terminated** record is corruption, including a bad hash on the final record.
It refuses 5 without modifying the file; it never rolls back a complete acknowledged
settlement as a torn tail. Missing/invalid admission signatures also refuse, even when
the hash chain is otherwise valid. Only an unterminated final line is recoverable:
truncate those incomplete bytes, fsync, then append unsigned internal `recover N SHA`.
This internal recovery cannot mint, debit or pay credits.

Version-1 journals are refused without modification, not automatically upgraded. There is no
implicit migration: archive the old journal, explicitly reconcile its test credits outside this
CLI, provision a **new unique ledger identity** and authorize fresh funding under the new policy.
This does not confer signatures or economic qualification on historical unsigned receipts.
Changing pins requires a new ledger identity and explicit reconciliation, not editing a running
journal's config.

Every successful mutation acknowledgement, including `dup` and `already-*`, follows a
journal fsync and directory fsync. A failed append fsync attempts to restore the last
good length and sync that rollback, then returns 3 without acknowledgement. Rollback
itself can fail, so an I/O refusal is not proof that an operation is absent. A retry
must verify replayed authority/state and cross a fresh durability barrier before it
can acknowledge an existing record. Callers reconcile the job's actual phase; they
must not blindly refund an already-settled job.

## Key-table capacity cannot trap existing escrow

Accounts, prefund IDs and jobs share a bounded table. Before appending, a new
prefund checks its new PID plus the requester account only if missing. A new
reservation checks its job plus any missing provider/verifier accounts; the
requester must already exist. The pinned role names are distinct. An operation
that would exceed the table refuses with exit 14 and leaves the journal unchanged.
An identical authorized retry needs no new key and still succeeds at capacity.

Settlement and refund use only the job and account slots created by its
reservation. They never allocate a key, so both remain available with a completely
full table. There is deliberately no shared “four spare slots” append guard.

## Tests and evidence limits

The gate retains the arithmetic oracle, exactly-once, lock/fsync, recovery, corruption and
conservation controls. It adds unsigned/wrong-key/mutated-record admission, exact policy/account
pins, independent verifier, evidence digest/quote/job binding, complete unauthorized tail refusal,
old-format refusal and a competing double-debit race. A deterministic strace injection kills the
process at stdout-write entry, after journal fsync and before acknowledgement; retry must pay
nothing twice. Separate mutants remove issuer/verifier/owner guards. `CRASH_N=200` also kills
actual ledger processes and retries signed commands, auditing after every interruption.
`MUTANT_CRASH_N=0` can omit redundant crash trials on mutants, not on the honest binary.

`TUR_LEDGER_AUTH_ONLY=/path/to/binary` runs the unsigned-prefund regression against another binary,
so the same assertion can be shown failing against the pre-fix source and passing after the fix.

The suite also fills a ten-slot build of the same native ledger, checks clean
prefund/reserve refusal, settles one existing job and refunds another, and checks
once-only retries and conservation. Only `KCAP` changes for this fast boundary
fixture; record parsing, signature verification, replay and append are unchanged.
`TUR_LEDGER_CAPACITY_SOURCE=/path/to/source.om` runs just this regression, allowing
the same test to fail before the fix and pass after it. The `capacity-lockup`
mutant restores the blanket append guard and must be killed. Separate evidence
records exercise the production 65,536-slot binary with a full signed journal.

`TUR_LEDGER_AUDIT_SOURCE=/path/to/source.om` runs the independent-audit regressions:
repeated injected fsync failure and rollback size, synchronized duplicate replies
for all four operations, acknowledged final-record corruption, fused argv fields,
and reserve retries after either terminal phase. The corresponding five guard
mutants (rollback, duplicate sync, complete-tail recovery, argv and phase) must die.

HOST_TESTED means local CLI behavior only. No Core ledger RPC exists in this checkout, so lost
CLI output is not an interrupted-RPC qualification. Mutating a signed AN0V digest tests admission
binding; it does not exercise an external signer's AN0V parser. No hardware, deployment, public
participant or economic-authority qualification is claimed. Historical evidence remains under
`evidence/aienet0-tur-ledger-20261027/`; the authentication addendum records the new runs separately.
