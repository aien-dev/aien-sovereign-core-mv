# aien-omegese-constraint: the Omegese mask

At every decode step the model may only pick a token that keeps its reply a prefix of a program
`omegese emit` accepts. A position where exactly one token is legal is forced (no sampling); at a
complete program the closing fence and the stop token end the reply. The legal-word list comes from
omega's own Omegese reader (`tools/omegese`), linked into the process as a C library: one function
call per query, no subprocess per token.

## Use it

- In a request to `aien-serve-http` (built with `--features omegese`): add `"constraint": "omegese"`
  to the chat-completions body. The reply carries `usage.aien_constraint` (forced positions, logits
  reads skipped, constraint microseconds, feeds). Without the field the server runs today's loop.
- Build: `AIEN_OMEGESE_DIR=<omega checkout at omegese.lock>` (build.rs checks the checkout's HEAD
  against `omegese.lock` and compiles `tools/omegese/omegese_lib.c`), or
  `AIEN_OMEGESE_LIB=<libomegese.a>`. Without either, `LinkedOracle` reports "not linked" and
  `SubprocessOracle` (one `omegese next --machine` per query) still works.

## What is shown, and where

All on the Spark (DGX Spark, GB10), evidence under `evidence/OMEGESE-MASK/`:

- G1 (host only, no chip): the 14 worked examples are rebuilt exactly (PASS) under a decoy that ranks an
  illegal token first; 2,000 random walks under the mask, each judged by `omegese emit`: 1,999
  accepted, 1 refused (FAIL). The refusal is an off-by-one in omega's reader at the 96-words-per-line
  limit (`next` allows a separator that it then refuses), not in the mask; fixer ledger L156. See
  `evidence/OMEGESE-MASK/README.md`.
- G2 (the 27B on our engine, 10 dev tasks, mask off and on, same seeds): 0 refusals both ways, 8 of 10
  correct both ways, replies byte-identical (`g2-20261020T0202Z/README.md`).
- G3 (cost): the constraint's host time is 14.82 ms per token, 9.0 percent of a decode step (calc:45a596e7);
  the 27B graph computes the output head inside its token call, so a forced position saves only the
  logits read and the sampler, not the model step.

Not shown yet: a trial where the model would refuse without the mask (the dev tasks had none), and a
forced position that skips the model step (needs a no-head variant of the 27B graph).
