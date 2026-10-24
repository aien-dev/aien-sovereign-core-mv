# Complete-state reuse selection

The native Omega components in this directory advance Core #718. `policy.om` chooses between
fresh prefill, a compatible RAM snapshot and a compatible disk snapshot by
estimated total completion time. It does not implement snapshots, allocate
memory, authenticate files, restore state or execute a model. `pages.om` now
provides bounded host backing pages as described below. Complete-state
composition and restore remain with Core #713; storage identity and offload
remain with #484/#485. Nothing here is wired into production. Hardware
performance and full-model correctness are NOT_RUN.

Inspected baseline: Core main `d691a4dc`,
`crates/aien-serve-http/src/kv_persist.rs`, `PersistState` storage path.
Its `fetch(prompt, in_ram)` skips disk restoration when the disk prefix is no
longer than the RAM prefix. This component supplies a cost decision for the
future native complete-state owner; it does not change that existing path.

## Calling contract

Concatenate `policy.om` before a seeded `.om` caller. All values are unsigned
64-bit integers. Time estimates are microseconds. Calibration timestamps and
the current timestamp must use one monotonic clock in microseconds. There is
one caller-owned policy instance; concurrent calls need separate instances or
the existing scheduler's serialization. No locks or disk waits occur here.

`rcontrol` has these ten fields:

| Index | Meaning |
| --- | --- |
| 0 | Candidate count, 1 through 16; row zero is always fresh work |
| 1 | Previously selected stable snapshot ID; zero means fresh work |
| 2 | Current monotonic timestamp |
| 3 | Maximum calibration age |
| 4 | Minimum observed samples, 1 through 1,000,000 |
| 5 | Total allowed memory bytes, at most 2^50 |
| 6 | Already allocated backing and shared references, counted once |
| 7 | Absolute switching margin in microseconds, at most 10^12 |
| 8 | Relative switching margin in millionths, at most 1,000,000 |
| 9 | Current request token boundary, at most 131,072 |

Each candidate occupies 22 words in `rrows`:

| Index | Meaning |
| --- | --- |
| 0 | Enabled: exactly one admits the row; other values are skipped |
| 1 | Source kind: zero fresh, one RAM, two disk |
| 2 | Stable snapshot incarnation ID, unique among enabled rows; fresh is zero |
| 3 | Accepted, consumed prefix length; fresh is zero |
| 4 | Emitted but unconsumed token count, zero or one; fresh is zero |
| 5 | Complete-state qualification mask, exactly 31 for reuse |
| 6 | Observed calibration sample count, bounded at 1,000,000 |
| 7 | Last calibration timestamp |
| 8 through 14 | Lookup, read, integrity/decryption/validation, transfer, reconstruction, suffix prefill, and remaining generation time |
| 15 through 19 | Additional backing, metadata, staging, scratch and commit/output reservation bytes |
| 20 | Caller has verified payload integrity, exactly one for reuse |
| 21 | Caller still holds the snapshot and its generation, exactly one for reuse |

The five state bits mean attention, recurrent state, convolution tails,
position/pending-token/drafter state, and sampler/constraint continuation state
are all either present or explicitly qualified as reconstructible under the
same continuation contract. They are assertions from the qualified owner,
not a substitute for inspection or tests of those states. A model without one
of these state families still needs an explicit not-applicable qualification.

`rexpect` contains eight nonzero 256-bit identities, four words per identity:
model, tokenizer, template, precision/layout, backend, namespace, continuation
contract and cost-workload profile. The workload profile binds exact context
shape, output-work estimate and calibration environment. `rident` contains
the same 32-word tuple for each candidate. The caller validates the actual
prefix against request tokens separately, authenticates the source and binds
the tuple to it. This module compares all words; it does not create a digest
or interpret an untrusted ID as a capability.

The owner must include all retained shared backing in control field six, and
all incremental allocations and temporary reservations in each row. Count a
shared allocation once, and never subtract merely freed slots from allocated
backing. The selected reservation must succeed before copying or publishing
tokens. A failed reservation requires a fresh decision with updated accounting.

## Decisions and errors

`rselect` returns zero on a decision. `rresult` contains selected row, predicted
total time, predicted saving against fresh work, prediction-valid flag and
stale-or-missing-baseline fallback flag. Output is cleared before validation.
`rreason` gives each row's admission status; unexamined rows stay 77.

Status 65 means malformed configuration/row, 69 incompatible or missing
identity, 74 missing/stale/future calibration, 75 insufficient memory, 76 an
unqualified state boundary, and 77 disabled or unexamined. Invalid reuse rows
are skipped. An invalid baseline refuses the request, except that missing or
stale calibration selects fresh work with prediction-valid zero. In that
fallback, no predicted time or saving is asserted. The caller can run its
ordinary path only after its own admission checks.

Every phase is bounded at 10^12 microseconds; sums and margin arithmetic fit
64 bits. A zero total estimate is unknown. The cheapest admitted candidate
wins only when its saving strictly exceeds the absolute plus relative margin.
The reference is the admitted current snapshot if cheaper than fresh work,
otherwise fresh work. Exact ties retain that reference; other ties prefer RAM
to disk, then the smaller stable ID. Row ordering therefore does not determine
which of equally priced reusable snapshots wins. An invalid current snapshot
cannot be retained by the switching margin.

`rcalib(row, timestamp)` updates an already admitted profile from the seven
actual elapsed phases in `robs`. All fields are validated before any update.
Each estimate becomes `ceil((7 * old + observed) / 8)`; sample counts saturate.
This bounded smoothing is a policy choice, not a statistical confidence claim.
An expired or changed profile requires a new initial calibration from the
existing measurement owner. Do not feed predicted or fabricated measurements
back as observations. Record actual completion cost and the selected source
through the existing measurement owner, without exporting private prompts.

## Reproduction

The compiler is the existing independently reconstructed source-built Omega
compiler from omega-bridge `0fc7d8662cb799e7c206b17c18ea4c3ec2b8fea9`, built
with its existing `tools/bootstrap.sh --offline`. Required compiler SHA-256:
`191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9`.

Set `OMC` to that verified executable and `REUSE_OUT` to a private output
directory. Existing shell commands below are development orchestration only.
All new implementation and test logic is handwritten seeded `.om`.

```sh
cat native/state_reuse/policy.om native/state_reuse/tests.om > "$REUSE_OUT/tests.om"
quiet-build timeout 30 "$OMC" < "$REUSE_OUT/tests.om" > "$REUSE_OUT/tests.elf"
# Require successful compilation before executing the output.
chmod 700 "$REUSE_OUT/tests.elf"
quiet-build timeout 20 "$REUSE_OUT/tests.elf"
```

The test prints the check count and failure count, returning one on failure.
Its 8,000 synthetic cost triples independently enumerate the expected cheapest
fresh/RAM/disk choice. Other controls cover all identity words, all timing and
memory phases, stale/future profiles, state boundaries, hysteresis and atomic
calibration rejection. Synthetic timings establish policy behavior only.

Remaining #718 gates include actual snapshot retention/allocation, integration
with #713 and native serving, observed profiles on the qualified 27B artifact,
corrupt-file and cancellation recovery, storage overlap, actual RAM/page-cache/
NVMe comparisons, independent final-source review and normal repository gates.
No existing runtime or storage implementation is retired by this component.

## Owned backing pages and copy-on-write

`pages.om` implements the bounded host allocation portion of #718, using
existing L788 managed memory operations. It does not define a second snapshot,
cache, scheduler, persistent store, device arena or model-state format.
Historical Core `cascade/tree-verify-wire` at
`3b2dcd9168d73925d8dc0836adaf9a9e4dca022c` stores `pos`, `tails` and `states`
in Qwen38Snapshot while retaining attention rows in the live arena. That
single-conversation technique does not establish independent complete-state
snapshots. Core #713 must determine the complete accepted-state page set and
qualify the restore boundary, including all recurrent layers and continuation
state, before this allocator can back production reuse.

This code uses the existing proposed Omega compiler extension at functional
source `d80b428c6ed9e3864fd7fc02cd7392bd4ad67f56`, consumed from public draft
PR #676, with compiler SHA256
`1d1aad082539399920bc6c90ff41f0a0c3a1ca3225e518fd0d259e0aeb71732d`.
The qualified Linux AArch64 host has 4096-byte OS pages, verified with
`getconf PAGESIZE`. This accounting profile requires that page size; it is
not portable physical-memory qualification for arbitrary hosts or AienOS.
The module makes real anonymous mappings. It does not use GPU memory or copy
whole context arenas for each snapshot.

`pinit(page_size,slot_limit,budget)` configures an empty pool. Page size is
4096..67108864, a multiple of4096; slot limit is1..64. Budget is17984..2^50.
The17984 fixed bytes account for this module's global allocations, including compiler alignment:
7 configuration words padded to64 bytes, three64-entry tables and two8192-byte copy buffers.
Add compiler mapping-table storage, executable/data page rounding, stack,
other modules, model memory and outstanding reservations separately in the
caller's overall accounting. This component does not claim total process RSS.
All calls require one serialized owner. Internal globals must not be changed
by a consumer, and mappings must not be unmapped outside this module.

- `pnew` returns one owned reference to a zero-filled page as a positive long
  ID, or a negative status. It reserves exactly one page against the budget.
- `phold(id)` adds one reference to the same backing page. The total reference
  count is capped at1048576. Sharing does not add backing bytes.
- `pfree(id)` releases one owned reference. Only the final release unmaps the
  backing; accounting decreases only after that operation succeeds. A returned
  unmap failure preserves the reference and charge for a later retry.
- `pread(id,offset,count)` copies up to8192 bytes into `pio`. It validates
  the ID and both range ends first. On failure, scratch bytes are not a result.
- `pwrite(id,offset,count)` writes the first `count` bytes of `pio` and returns
  the reference the caller must retain. With one reference it modifies the
  owned page and returns the same ID. With shared backing it allocates and
  copies exactly one page, applies the write, then transfers one reference
  from the old page to the new page. Allocation refusal preserves the original
  bytes, references and accounting. A valid zero-length write never copies.
- `pbytes` returns live mapped backing plus17984 fixed module bytes, including
  when uninitialized or empty. Freeing one of several references does not
  lower this value.

Status0 is success for status-returning functions. Status65 means invalid
configuration or range,73 uninitialized/busy/stale/invalid ownership ID,
74 returned unmap error,75 quota/capacity/allocation refusal. `pnew` and
`pwrite` return these statuses negated. The underlying compiler traps on
corrupted raw mapping handles; its checked copies return the requested byte
count or trap. Such corruption and concurrent mutation are outside the owned,
serialized module contract.

IDs encode a slot and generation. Generations survive empty-pool reconfiguration
and increase on reuse. At2^56-1 a slot retires permanently rather than wrapping.
The largest ID remains positive as a signed64-bit value. A released reference
must never be reused, even if other references keep the page alive: the ID
identifies backing, not a distinct session's grant. Callers must balance
references, enforce read-only snapshot semantics and namespace/session
permissions, and update their binding to every successful `pwrite` result.
This primitive does not authenticate those higher-level ownership rules.

Each page is a separate managed mapping. The compiler's128 mapping slots are
shared with other process allocations, so an allocation can refuse before
this pool's own limit. Linux may lazily commit physical pages; allocation
success does not guarantee immunity from an OS out-of-memory kill. Actual
GPU state lifetime, producer synchronization and host/device transfer remain
with the existing state/runtime owners.

## Connecting allocation accounting to the selection policy

Concatenate `policy.om`, `pages.om` and `page-policy.om` for the optional
`rpool(other_allocated_bytes)` entry point. It sets policy control field6 to
`pbytes + other_allocated_bytes`, then runs the unchanged selector. Other
bytes must exclude the pool payload already counted and include every other
retained backing, metadata and reservation. Oversized input refuses through
`rselect`, preserving its cleared-output failure behavior. Serialized use
ensures the measured allocation count remains current during selection.

The caller still fills candidate incremental memory fields. In particular,
a shared-page write reserves a full additional page before mutation. Neither
the adapter nor a successful selection authenticates the candidate, creates
a complete snapshot, or guarantees a later allocation; the caller must handle
allocation refusal without publishing tokens or discarding the prior state.

Native `pages-test.om` passes37905 checks. A separately compiled
`page-policy-test.om` passes40 checks, including the actual change in decisions
when backing is released, one-byte-under allocation headroom, copy-on-write
preservation and excessive external-memory refusal. A compiled adapter that
omits live page bytes is rejected. The unchanged original policy also passes
its16210 checks under the mapping compiler. Independent page tests and controls
are recorded in `evidence/state-reuse-718/pages/`.

Build by concatenating the required module files followed by exactly one test
entrypoint, feeding that source to the pinned compiler, checking compiler
success, then executing the emitted ELF. No source here calls a foreign helper.
Development shell/Node orchestration is disclosed separately from native code.
Model restore-versus-fresh equivalence, namespace/store/cancellation integration,
NVMe timing and full-model improvement remain open issue gates.

## Corrected compiler compatibility

The compiler safety correction in Omega PR676 also reproduces this component
without changing its executable bytes. See [evidence/state-reuse-718/compiler-correction](../../evidence/state-reuse-718/compiler-correction/README.md) for the exact compiler/source pins, binary comparisons and fresh host replays. The original qualification remains preserved.
