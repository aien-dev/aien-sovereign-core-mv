# ALLEN end-to-end demo v2 (predeclared)

Status: PREDECLARED. Committed before any v2 run. v1 (DEMO-v1.md) and its FAIL result (RESULT-v1.md,
receipts-v1.jsonl) stay as recorded and are not edited.

v2 is DEMO-v1.md with exactly the three changes below. Every other rule, step, label, model and pass
criterion is the same as v1 and is incorporated here by reference.

## Changes from v1, with reasons

1. **The request states the format rules (S3, S8).** In v1 the request was only "Write garden.md in the
   workspace, a garden plan." Constraints C2 (first line a `# ` heading) and C4 (at most 200 words) were
   checked but never told to the model, so a model could not know them. In v2 the request is:

       Write garden.md in the workspace, a garden plan. Start with a Markdown heading line that begins
       with "# ". Use at most 200 words.

   (`garden-b.md` for S8.) C3 (`tomato`) is still NOT in the request: it must come only from memory note
   N-work, so it keeps testing that scoped memory reaches the model. C5 (no personal canary) is unchanged.
   The checker and C1-C5 are unchanged.

2. **S7 checks the model line, not the backend line.** v1 required "the backend line and the model digest
   changed". Both models run on the same CPU reference backend, so the `Backend:` line cannot differ; v1
   recorded this as FAIL as written. In v2, S7 PASS requires: the daemon's `Model:` log line and the model
   weights digest both changed, and the `Backend:` line is recorded in the receipt (not required to change).
   The identity and state criteria of S7 are unchanged.

3. **The run is labelled with the stack it ran on.** v1's S4 and S8 failed because the ledger records of a
   commit did not carry the model digest or the ALLEN LogicalAgentId. That is a product gap, fixed in code,
   not by changing S4. If v2 runs before that fix is on main, every receipt is labelled
   `candidate stack at <commit>, not main`.

No other change. In particular: no retry with a changed prompt beyond change 1, the same models, the same
kill -9 restarts, real-CPU label, GB10 and native Aienux NOT_RUN.
