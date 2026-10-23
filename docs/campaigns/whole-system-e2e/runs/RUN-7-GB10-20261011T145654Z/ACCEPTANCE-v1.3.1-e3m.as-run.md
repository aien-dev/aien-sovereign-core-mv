# WHOLE-SYSTEM-E2E, harness v1.3.1: E3m topic words (for RUN-3-GB10)

Tracking: aien-dev/aien-sovereign-core#402 (owning issue), aien-dev/aien-architecture#190 milestone R6. Contract:
`ACCEPTANCE_E2E.md` FROZEN at aien-architecture b2400fe, sha256
`9e3d5f4e23d65ac17be8f2f4ca48355f0aaa486f67ff5a8a8044dcdd646d34e8`. This file is committed before the harness change
that implements it. `ACCEPTANCE-v1.md` to `-v1.3-gb10.md` and `-v1.3-ctrl-e5.md` stay as written. RUN-1 and RUN-2-GB10
stay as recorded; RUN-2-GB10 stays FAIL.

## Disclosure

This rule was written **after** RUN-2-GB10 (`runs/RUN-2-GB10-20261011T030554Z/`) showed the mismatch. There, row E3m
failed check D4 because the generated report names the windmill only in its heading, as `Windmill`, and the harness
matched the topic word `windmill` exactly and case-sensitively. The coordinator ruled this a harness defect in matching
a topic word (2026-10-11): `ACCEPTANCE-v1.1.md` row E3m says "Mention the harbour, the orchard and the windmill" (topic
words) with the rule "as E3, with the three topic words in place of the file names", and the harness reused the
check written for file names. The next declared attempt, RUN-3-GB10, is a new attempt under this pre-registered rule.

## What changes, and what does not

Only check D4 of row E3m changes. Everything else of `ACCEPTANCE-v1.3-gb10.md` and the rows it builds on applies
unchanged: the objective text, D1 (UTF-8), D2 (first line starts with `# `), D3 (at most 200 words), the digest and
single-spend checks, every other row, the GB10 rules G1 to G6, the frozen contract.

| check | before (v1.1 to v1.3) | v1.3.1 |
|---|---|---|
| E3 D4: required file names | exact, case-sensitive substring (`grep -F`) | **unchanged** |
| E3m D4: topic words `harbour`, `orchard`, `windmill` | exact, case-sensitive substring (`grep -F`) | case-insensitive whole word: `LC_ALL=C grep -q -i -w -- WORD FILE` |

The exact rule implemented (GNU grep `-w`, ASCII case folding under `LC_ALL=C`): a topic word counts when it occurs
anywhere in the report, in any letter case, with no letter, digit or underscore directly before or after it. So
`Windmill`, `WINDMILL`, `windmill.`, `windmill,` and `windmill's` count; `windmills`, `windmill_x` and `wind-mill` do
not count; the heading counts like any other line. Each E3m receipt gains `d4_rule` naming the rule used.

The independent verifier (`tools/aien-verify`, owned by Lane E) is not changed by this file. Its E3m document row
(supplementary, not a frozen contract row) uses an exact substring match. Unless a merged verifier change applies the
same rule to E3m topic words, an E3m that passes here only through letter case shows as a `status_disagreement` on
E3m in `VERIFIER_VERDICT.txt`; that would be recorded, not hidden.

## Predictions for RUN-3-GB10 (written before the run)

| row | prediction | why |
|---|---|---|
| PRE, CTRL-E3a, E2, CTRL-E2, E3, E6, CTRL-E3b, NET, CTRL-E6, CTRL-E5 | PASS | all PASS in RUN-2-GB10 with fallback 0; unchanged |
| E3m | PASS | UNVERIFIED (confidence moderate): greedy decoding, same model and same workspace path should give RUN-2's text, which passes D1 to D4 under this rule; a merged change on main since RUN-2 could still move the text |
| E4 | PASS or FAIL as measured | first run of E4 on the chip; passed on CPU in RUN-1 |
| CTRL-E4 | PASS | store-loss refusal is backend independent; passed in RUN-1 |
| CTRL-E4b | RECORDED | not judged (v1.2) |
| E1, E5, CTRL-E1 | NOT_RUN | R8, R7 |
| G4, G5 | `fallback_detected: false`; every model-calling row `gpu_executions >= 1` | as RUN-2-GB10 |

## Rules of this run

As `ACCEPTANCE-v1.3-gb10.md`: one declared attempt from the merged main commit, folder `runs/RUN-3-GB10-<utc>/`, a note
in the result naming RUN-2-GB10 as the previous attempt, started detached, never killed, `quietlock check` first,
nothing under `runs/` edited afterwards. No program PASS is claimed by any outcome of this run.
