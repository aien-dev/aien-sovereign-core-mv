# tur-ledger: the Aienet-0 TUR escrow ledger in seeded Omega

One persistent, crash-safe escrow ledger for prefunded TUR test credits. It applies, per JOB, the
escrow rules of `tests/turing-network/tur_r0_oracle.om` (#791): reserve needs funds and one
reservation per JOB; settle needs a reserved JOB and `PAYP + PAYV + BACK = AMOUNT`, and happens
exactly once. Source: `tur-ledger.om` (built after `om/vendor/sha256.om`). No Rust, C or Python.

Build: `cat om/vendor/sha256.om om/tur-ledger/tur-ledger.om | $OMC > tur-ledger && chmod +x tur-ledger`
with the seeded omc (omega-mv `seed/out/omc`, sha256 `2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f`).

Gate: `OMC=<seed omc> bash om/tur-ledger/check.sh` (needs `strace`, util-linux `flock`, `timeout`).

## Commands

`tur-ledger --journal DIR <cmd>`. One record per line on stdout, exit 0 on success.

| Command | Success output |
|---|---|
| `prefund PID ACCT AMOUNT` | `prefunded PID ACCT AMOUNT`; an identical repeat prints `dup PID ACCT AMOUNT` |
| `reserve JOB REQ PROV VER AMOUNT` | `reserved JOB REQ PROV VER AMOUNT`; an identical repeat prints `already-reserved ...` |
| `settle JOB RECEIPT_SHA256 PAYP PAYV BACK` | `settled JOB RECEIPT PAYP PAYV BACK`; the identical repeat prints `already-settled ...` and writes nothing |
| `refund JOB` | `refunded JOB`; a repeat prints `already-refunded JOB` |
| `balance ACCT` | `balance ACCT N` |
| `show JOB` | `job JOB reserved\|settled\|refunded REQ PROV VER AMOUNT [RECEIPT PAYP PAYV BACK]` |
| `audit` | `audit ok records R head H accounts A jobs J open O settled S refunded F prefunded P balances B escrow E` |

JOB and RECEIPT are 64 lowercase hex (JOB is the MJB1 dag_id). Amounts are decimal u64 without
leading zeros, nonzero except PAYP/PAYV/BACK. Names are 1..64 bytes of `[A-Za-z0-9._:-]`.

## Refusals

A refusal prints `refused C reason` on stderr, writes nothing and exits C.

| C | reason | when |
|---|---|---|
| 2 | usage | bad arguments, field syntax or command |
| 3 | journal-io | the journal cannot be opened, read, written or fsynced |
| 4 | lock | flock on the journal failed |
| 5 | journal-corrupt | a bad line before the last, an intact last line that does not link (SEQ/PREV), or a record the rules reject on replay |
| 6 | conservation-violated | after replay: balances + open escrow != prefunds, open escrow != open JOB amounts, or a settled JOB paid out != its amount |
| 7 | insufficient-funds | reserve beyond the requester's balance |
| 8 | unknown-job | settle/refund/show of a JOB never reserved |
| 9 | job-conflict | a second, different reservation for a JOB |
| 10 | wrong-phase | refund of a settled JOB, settle of a refunded JOB |
| 11 | settle-receipt-conflict | settle of a settled JOB with a different receipt or split |
| 12 | escrow-rule | `PAYP + PAYV + BACK != AMOUNT` |
| 13 | prefund-id-conflict | a PID reused with a different account or amount |
| 14 | capacity | the in-memory index (65536 keys) is full |
| 15 | overflow | a sum would exceed u64 |

## Journal

`DIR/journal`, one line per record: `SEQ PREV KIND FIELDS... HASH`, where HASH is the sha256
(lowercase hex) of the line's bytes before ` HASH` and PREV is the previous line's HASH (64 zeros on
line 1, `genesis tur-ledger-1`). Every command takes `flock(LOCK_EX)` on the journal for its whole
run, replays and checks the chain and conservation, then appends with `pwrite` and `fsync`s before
it prints. A last line that is unterminated or fails its own HASH is a torn write: it is truncated
and a `recover N SHA` record (N bytes, their sha256) is appended, with a note on stderr.
