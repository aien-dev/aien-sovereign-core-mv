# Regression pre-registration: sc#382 (inputs unreadable) and sc#383 items 2 and 3 (destination)

Written before any test or fix. Program: aien-architecture#190. Branch `e2e/w1-input-reading`.
Evidence is under `runs/` and is not edited.

## Failing behaviour on main, as observed

1. sc#382. `RUN-1-dry-20261010T134202Z/artifacts/E3-propose.json`: for the contract objective the model
   invented `ws-contract.txt` and `ws-contract.md` because the prompt carries only the goal text and the
   workspace path. The three inbox files are never shown to the model.
2. sc#383 item 2. `RUN-1-dry-20261010T133749Z/artifacts/E3m-propose.json`: the goal "Write a Markdown report
   named report.md about the three inbox files named harbour-notes.txt, orchard-ledger.txt and
   windmill-log.txt ..." was refused: `ambiguous destination: the goal names several files to write
   (report.md, harbour-notes.txt, orchard-ledger.txt, windmill-log.txt); name one`.
3. sc#383 item 3. `RUN-1-dry-20261010T134202Z/artifacts/CTRL-E3b-propose.json`: goal "Write a file named ctrl.md
   in the outbox folder containing one line: control." was refused because the proposal wrote
   `outbox/ctrl.md` but the goal-named destination was `ctrl.md`.

## Design (constants, no new knobs)

- Destination: "named X in the Y folder" (also "in Y folder", "in the Y directory") resolves to `Y/X` before the
  model runs. The proposal filename line, the destination check and the written path all use `Y/X`.
- Inputs: paths the goal references but does not write. A bare name that exists in the workspace (or in a
  folder the goal names with "the Y folder") and is not the destination is an input. A folder named as a
  source ("files in the inbox folder") contributes its regular files, directly inside it, sorted by name.
- Bounds: `COMPOSE_INPUTS_MAX_BYTES` = 16384 total, `COMPOSE_INPUTS_MAX_FILES` = 16. Files are taken in
  sorted order; a file that does not fit whole, a non-UTF-8 file or a non-regular entry is not included and
  is listed under `inputs_omitted` with the reason. Never truncated silently.
- Confinement: a named input outside the workspace (absolute, `~`, `..`, or a symlink resolving outside) is
  refused before the model runs. Folder entries that are symlinks leaving the workspace are omitted.
- Report: `compose propose` report gains `inputs` (name, sha256, bytes) and `inputs_omitted` (name, reason),
  both default empty so old reports still parse.

## Tests to add

Unit and integration tests in `crates/aien-runtime/tests/inputs_test.rs` (new), plus a stub-aware
`compose propose` test in the same file.

| Test | On main | After fix |
|---|---|---|
| folder_phrase_resolves_done_ctrl_and_report_destinations (OBJ6, OBJC, OBJ texts) | FAIL | PASS |
| proposal_in_the_folder_is_accepted_for_ctrl_goal (`outbox/ctrl.md` reply accepted) | FAIL | PASS |
| input_names_are_not_write_targets (sc#383 item 2 text, files exist) | FAIL | PASS |
| two_distinct_write_targets_stay_ambiguous (negative control) | PASS | PASS |
| existing_destination_is_still_an_edit (negative control) | PASS | PASS |
| folder_source_phrase_includes_inputs_sorted_with_digests (OBJ, three inbox files) | FAIL | PASS |
| named_files_become_inputs_but_destination_is_not (item 2 text) | FAIL | PASS |
| inputs_are_capped_and_omissions_are_listed | FAIL | PASS |
| non_utf8_and_non_regular_inputs_are_omitted_not_read | FAIL | PASS |
| named_input_escaping_workspace_is_refused_before_model (`..`, absolute, symlink) | FAIL | PASS |
| goal_without_inputs_keeps_prompt_byte_for_byte (negative control) | PASS | PASS |
| compose_propose_report_lists_inputs_with_digests (stub-aware) | FAIL | PASS |

The tests commit adds only empty-returning stubs of the new API so the tests compile; a FAIL is an assertion
failure, not a compile error.

## Must not change

Desk requirement, grants, effect boundary refusals, `AlreadySpent`, requirement phrasing rules, the refusal
for two genuinely distinct write targets, the destination-probe corpus record
(`destination_corpus_test`), the model, token budgets, memory and ALLEN code, the harness script.

## Limits (stated now)

Input text is untrusted prompt content, the same as edit-mode content today. Folder reading is one level
deep. Reading only happens when the goal names the source; the model cannot ask for files.

## Notes added with the fix (rows above unchanged)

- `compose_propose_report_lists_inputs_with_digests` passes on a stub build before and after the fix, because
  the stub refuses every run ("not linked"). The registered FAIL-before holds only with the linked compose
  library; the assertions on `inputs` are exercised by CI's linked job. The same facts are asserted without
  the library by `folder_source_phrase_includes_inputs_sorted_with_digests`, which failed before.
- Inputs found by a directory word: a goal word naming a top-level workspace directory counts only when a
  plural noun follows it ("the three inbox files"), so "about docs and plan.txt" keeps its v5 prompt.
- A folder is read as input only when the goal puts a reading word (files, notes, from, about, ...) within
  five words before "in the Y folder" and the phrase does not directly follow a named path.
