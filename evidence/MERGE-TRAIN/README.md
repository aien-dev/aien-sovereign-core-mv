# MERGE-TRAIN evidence (L164, 2026-10-20)

Built on the Spark from the judge's own records. Every number below comes from a file in this folder.

## 1. Gate memo replay (`aien-test memo-replay`)

Command:

    aien-test memo-replay --state-dir ~/.local/state/aien-local-ci \
      --clones ~/workspace/pipeline/clones --since 2026-10-20T00:00:00Z --tsv replay-runs.tsv

Input: every fresh receipt in `evidence/<repo>/<sha>/` started since 00:00Z (omega and aien-sovereign-core
were the only repositories judged today). Output: `replay-summary.txt` (per head) and `replay-runs.tsv`
(per gate run: RAN or REUSED, and from which commit or why not).

| policy   | fresh gate runs | would be reused | gate-minutes before | gate-minutes after | saved |
|----------|-----------------|-----------------|---------------------|--------------------|-------|
| ancestor | 710             | 26              | 15,179.5            | 14,962.3           | 1.43% (calc:a15213b1) |
| any      | 710             | 56              | 15,179.5            | 14,544.3           | 4.18% (calc:7e04a88d) |

"Gate-minutes" is the sum of each gate's own run time (gates run in parallel pools, so this is work, not
wall clock). Per-head before/after minutes and the list of reused runs are in `replay-summary.txt` and
`replay-runs.tsv`. Best single heads (policy any): sc 07228020 99.9 to 36.1 gate-min, sc 16288198 96.3 to 45.3.

Why the memo saves little today (diagnosis over 1,612 receipts, same gate, consecutive judged heads):

- 395 runs had no earlier PASS with the same inputs. The inputs that differed were the declared
  inputs in 354 of 387 compared pairs: omega's gates declare `src/`, `tests/` and `tests/gates/` whole,
  so any code change anywhere changes every gate's key. Pins, machine and environment were never the
  difference.
- 259 runs (4,804.5 gate-min) are not cacheable: `cache: never` silicon gates (the gate authors' rule,
  kept), gates without inputs, or unclean runs.

So the memo is safe and small; the lever is to judge many branches in one run (the train), and, later,
narrower `inputs:` lists in omega's gates (each gate naming the folders it really builds).

Assumption stated: receipts before this branch do not record the toolchain id; the replay assumes the
toolchain did not change during the day (the sandbox rustup and cargo dirs were last changed
2026-10-19 18:03, before the replay window). From this branch on, every fresh receipt records
`volatile.toolchain` and a receipt without it is never a memo source.

## 2. Today's judge runs (`judge-runs-20261020.tsv`)

| repo                | judge runs | runs that ran gates | wall minutes per run (sorted)               | median |
|---------------------|------------|---------------------|---------------------------------------------|--------|
| omega               | 14         | 8                   | 81 95 102 109 112 126 310 350               | 110.5  |
| aien-sovereign-core | 32         | 29                  | 16 .. 41                                    | 22     |

## 3. Train dry run on the current READY list (`dry-run-20261020.txt`)

    aien-test train <repo> --dry-run --state-dir /tmp/l164-train-state

(crumb tools and bundles linked from the live state dir; nothing judged, nothing pushed, the live state
dir not written.) Desk log: `~/handoffs/2026-10-18-pr-desk.md` at 11:17Z.

| repo                | candidate    | branches in | branches out | crumbs |
|---------------------|--------------|-------------|--------------|--------|
| omega               | 8244eb8ef20b | 10          | 8            | 6 recompiled, verify OK |
| aien-sovereign-core | 409cb425b133 | 2           | 14           | 3 recompiled, verify OK |
| aienos              | e066ae5452fc | 2           | 10           | 1 recompiled, verify OK |
| physics             | c3b47b2ae146 | 2           | 1            | current, verify OK |
| aien-architecture   | d8639a6ec5ba | 2           | 2            | no crumb pin found |

Why branches are out (34 in all): 20 conflict with a branch before them, 18 of those in `README.md` (every branch
edits the front page), 6 change the judge's own setup and need an operator train, 4 are stacked on a
branch that is not in the train, 3 are already on main, 1 commit is not in the trusted clone.

Projection, not a measured train run: omega's 10 branches judged one by one at today's median would take
10 x 110.5 = 1,105 judge minutes (calc:0fa2764f); as one candidate it is one run (110.5 at the median),
plus one more run per culprit round if it goes red.

## 4. Unit tests

- memo key and memo: `memo::tests` (6) and `cache::tests::memo_*` (3): a changed input or toolchain flips
  the result to a run, an unrelated file does not; a FAIL is never a source; `cache: never` always runs.
- host verdict note: `runner::tests::host_verdict_is_written_once_host_gates_finish_and_names_chip_gates_left`.
- train: `train::tests` (5): desk-log shapes, later READY replaces earlier, input matching, culprit finder
  (red-on-main blames nobody, no-inputs gate blames everyone), candidate build (in-order rebase, crumb-only
  conflict takes main's copy, real conflict skipped, stacked branch only behind its parent, culprit drop).

## 5. Decision: the memo ships switched off

The full test suite found the cost of the memo: `tests/local_ci.rs`
`worker_never_reuses_a_pass_when_an_undeclared_file_changed_across_commits` builds a gate that reads
`README.md` but declares only `u/`. With the memo on, the PASS of the first commit is handed to the second
commit, which breaks the gate. A gain of 1.4% to 4.2% of gate-minutes does not pay for that, so the worker
default is `--memo off`; `--memo ancestor|any` turns it on. Every fresh receipt now records
`volatile.toolchain`, so the memo can be switched on for a repository once its gates' `inputs:` lists are
shown complete (for example by running each gate in a sandbox that exposes only its declared inputs).
