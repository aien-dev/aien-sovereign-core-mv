# WHOLE-SYSTEM-E2E, harness v1.1: amended rows for the second declared dry run

Tracking: aien-dev/aien-architecture#190, lane L1. `ACCEPTANCE-v1.md` stays as written; its run
RUN-1-dry-20261010T133749Z stays recorded (`RESULT-RUN-1-dry.md`). This file changes three rows for the
second declared run and nothing else. Committed before the harness change that implements it.

Why: in dry run 1 the runtime refused both E3 goals at the propose stage, before the model ran, for reasons
that are its own goal-interpretation limits (aien-sovereign-core#383): "under 200 words" is not a recognized
requirement phrasing, and every file name in a goal is taken as a file to write. E4 and CTRL-E4 were therefore
never measured. The amended rows let the chain reach them while keeping the record honest about what each row
tests.

| row | v1.1 wording and rule | prediction |
|---|---|---|
| E3 | the contract objective in the wording of aien-architecture#196: `Read the three text files in the inbox folder and write a Markdown report named report.md in the outbox folder. Start with a Markdown heading line that begins with "# ". Use at most 200 words. Mention each file by name.` Rule unchanged (DONE, digest match, `# ` first line, at most 200 words, all three file names, second execute `AlreadySpent`). `objective_id` is the sha256 of this text. | FAIL on the file-name check (the proposer cannot read files, sc#382), or a propose refusal; either is recorded |
| E3m | the mechanics objective names topics instead of files: `Write a Markdown report named report.md about the harbour, the orchard and the windmill. Start with a Markdown heading line that begins with "# ". Use at most 200 words. Mention the harbour, the orchard and the windmill.` Rule: as E3, with the three topic words in place of the file names. | PASS |
| E4 | the remembered item is `the topic reported on first is the harbour`; the second objective is `Append one line to report.md naming the topic you reported on first.`; the appended line must contain `harbour` (case-insensitive). Other rules unchanged. | PASS or FAIL as measured |

Also recorded, not judged (no rule change): the folder the engine chose for `done.md` and `ctrl.md` (dry run 1
wrote `done.md` at the workspace root although the goal said "in the outbox folder"; aien-sovereign-core#383
item 3). Every other row, rule, prediction and limit of `ACCEPTANCE-v1.md` applies unchanged.
