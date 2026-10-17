# Regression evidence, sc#408 audit row 8 (receipt continuity across a restart), 2026-10-17

Row 8 of `RECOVERY-AUDIT-2026-10-11.md` was PARTIAL: the engine's journal continuity across a kill was shown by
ledger contents, not by a hash check. This change makes it a hash check at the one place a restart reads the journal
back (the compose home open, `ComposeBridge::open_home_marked`).

## The gap, measured

Each Cortex record digest (omega `cx_digest`) covers only its own record and is unkeyed. omega's open recomputes every
digest and the running chain, but the chain is rebuilt from the file, not compared with anything. The external record
mark (`<home>.cortex-mark`) held only the record count and the newest record's digest. So a record before the newest
one, rewritten with a freshly computed digest, opened as if intact.

## The check

The mark is now version 2 (160 bytes): it also carries the running hash over every record digest #1..#records,
`sha256(chain || digest)` from 32 zero bytes, the same chain omega keeps in memory (`rx_cortex.c` chain_step). sha2
only; no new primitive. The open recomputes it from the digests omega verified and refuses a mismatch:
`E_MARK_CHAIN` (`MarkRefusal::Chain`), "Run RecoverComposeHome". Each later append extends the chain from the last
verified mark (no full rehash). `RecoverComposeHome` sets a chain-refused mark aside as `.lost-<seq>` with a repair
record (`why: records before the newest differ`, `old_chain`), as it does for `E_MARK_DIGEST`. A version 1 mark still
opens (count and digest checked), is rewritten as version 2 and the upgrade is printed once.

## Red then green

`crates/aien-runtime/tests/receipt_continuity_test.rs`, linked omega composition library built from the omega.lock pin
(`scripts/pins.sh`). Each case rewrites `cortex.cx` with valid digests, so omega's own open check passes; the parser
first checks its digest against every record omega wrote (a control).

| case | main da84629 (test commit only) | this branch |
|---|---|---|
| forged middle receipt (one constraint note's text changed, payload sha and record digest recomputed) | accepted, opened with 22 records | refused `E_MARK_CHAIN` |
| two receipts reordered (each keeps its slot's logical time) | accepted | refused `E_MARK_CHAIN` |
| gap with the count restored (one receipt dropped, a later one repeated) | accepted | refused `E_MARK_CHAIN` |
| foreign receipt spliced in from another home | accepted | refused `E_MARK_CHAIN` |
| truncated tail | refused `E_MARK_TRUNCATED`; padded back to the count: `E_REPLAY` (J-Space anchor) | same |
| refusal repairs nothing; recover keeps the old mark | (not reached) | pass |
| version 1 mark upgraded, then a forged receipt refused | (not reached) | pass |

Other checks on this branch: `scripts/test-linked-compose.sh` (CI's linked gate, `LINKED_REPEAT=1`): PASS, 592 linked
tests; `cortex_mark` unit tests 12 passed (v1 read, v2 layout, chain formula against a manual sha256 fold, refusal
text); `crates/aien-cli/tests/recovery_matrix_test.rs` (real SIGKILL restarts of a compose-linked CPU
`fault-hold` daemon, SmolLM2 on the CPU reference): 12 passed; `cargo clippy --workspace --all-targets -- -D warnings`
and `cargo fmt --check` clean.

## Limits (not claimed)

The chain is unkeyed: rewriting the journal and the mark together, deleting the mark (adopted and reported), or
rolling both back together is still not detected (the mark's declared NOT_PROVED limits). Records appended after the
last mark write (a kill between append and mark update) are taken as read, as before. A refused open still lets omega
append its start-up records first, as for `E_MARK_DIGEST`. No fault-harness run, no E2E rerun, no GB10: the other
#408 items stay open.
