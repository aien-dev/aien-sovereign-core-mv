# Repo agents and the change board (ledger L088)

Drake, 2026-10-19: "This is a Unix-type environment, so why are we testing the entire architecture when we
only need to test the one unit that is being changed? Maybe we need a repo agent dedicated to each entire repo.
When something changes, it communicates that to a board of some sort. The other repo agents then take that
information and make sure it still works with them, without changing anything."

This document measures today's cost, designs the system and describes the first working slice
(`tools/aien-board`, `contracts/omega.contract`, `scripts/contract-omega.sh`). Evidence:
`evidence/repo-agents-2026-10-19/`.

## 1. What already exists (reuse inventory)

| part | what it already does for "test only what changed" | what it does not do |
|---|---|---|
| `aien-test plan` / `worker` (tools/aien-test, ADR 0033 D12, A1, A4) | Per change, selects the gates whose watched inputs cover a changed path, plus `.aien/always`, closed over dependencies and dependents; reuses cached receipts. The worker runs only that selection. | Works at GATE granularity: a gate that watches `src/` or `crates/` runs for any change under it. Looks at one repository only. Does not record why a gate was selected (recomputed here). |
| siblings map (`tools/aien-local-ci/siblings`) | Lists which repository's gates read which other repository (`../omega` etc.), pinned at a branch. It is the cross-repo dependency graph. | Nothing re-checks a consumer when its producer's main moves; consumers see a producer change only when they bump their own pin file (`omega.lock`). |
| pin files (`omega.lock`, `physics.lock`, `aienos.lock`) + `SC-LINKED-COMPOSE` | A consumer ships with an exact producer commit; the pin bump PR runs the consumer's linked tests against it. | The check happens late (at bump time), after many producer commits. |
| `aien-abi-conformance` + `abi/rxc_host_abi.manifest` | The precise interface contract between omega's C ABI and sovereign-core: header, compiled layout probe, archive symbols. | Only runs inside sovereign-core builds at the pinned omega. |
| spark-crumbs (RFC-0003 Am2, advisory since 2026-10-19) | Directory notes; the planner reads `related`/`links` ancestry when present (today's schema has neither). | No dependency data to plan with; advisory only. |
| ledger, PR desk (`~/.claude/skills/pr-desk`) | Who owns what; one desk merges. READY lines hand branches over. | Not a change feed. |
| L074 shared base | Shared build artifacts by source hash, one quiet window per measurement session. | Open; the repo agents' build caches (`target/<repo>`) are the place it would plug in. |

Searched before building: `gh search code --owner aien-dev "change board"`, `"consumer contract"`,
`"repo agent"` (no hits for a cross-repo change feed); `git branch -r | grep -i board` in
aien-sovereign-core (only `switchboard`, unrelated); `ledger list` (L088 only). The slice reuses the
planner (`aien_test::plan`), the ABI conformance check and the siblings map; it adds only the board,
the contract file and one consumer check.

## 2. Today's cost (measured)

MEASURE_SECTION

## 3. Design

### Repo agent
One agent per repository (11 today, the judge's REPOS list). It owns that repository's checks and
nothing else:
- **Own changes**: keeps running `aien-test` on its own PRs exactly as today (the worker), and when its
  main moves it POSTs to the board: repository, commit, base (its previous post), and every changed path.
- **Other repositories' changes**: for every POST by a repository it consumes (a `contracts/<producer>.contract`
  in its own main), it intersects the changed paths with the contract's watched paths.
  No intersection: answer SKIP, nothing runs. Otherwise it runs only its own contract check against
  the posted producer commit, in scratch checkouts it owns, and answers PASS, FAIL or ERROR with a receipt.
- **Never edits another repository**: the producer is a detached read-only checkout; the agent never
  commits, pushes, or changes a pin. A FAIL is a message to the producer (and the desk), not a fix.

### Contract (consumer side)
`contracts/<producer>.contract` in the consumer repository, read from the consumer's main through git
objects (the A1 rule: a producer change can neither widen nor narrow a consumer's contract). It names the
producer, the pin file, the watched producer paths and the check. The watched list is not guessed: the
check learns it from the build itself (the C objects' `-MMD` dependency files) and writes `watch.learned`;
the consumer commits it as `contracts/<producer>.watch` through its own PR.

### Board
BOARD_SECTION

### What still needs a full run, and why
FULL_SECTION

## 4. First working slice

SLICE_SECTION

## 5. Switch-over plan (the desk's call; nothing here changes the live judge)

SWITCH_SECTION
