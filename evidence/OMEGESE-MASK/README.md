# OMEGESE-MASK: the Omegese mask inside our engine (ledger L112)

Predictions, pushed before any run: `00-prediction.md`.

## G1 correctness (host only, no chip): FAIL, one refusal, cause outside the mask

Receipt `g1-run2-20261020T0255Z-FAIL.txt` (2026-10-20 02:55Z, sc 09e72138 + omega 1c5eb14f, 27B tokenizer).

- Worked examples: **PASS**, 14 of 14 reproduced exactly with a decoy (an illegal token) ranked first at
  every step: the mask removed the decoy every time and the controller wrote each example byte for byte.
- Random walks: **FAIL**, 2,000 walks, 1,999 accepted by `omegese emit`, 1 refused. Walks wrote real text
  (non-blank bytes per walk: min 42, median 394, max 576).
- The one refusal (walk 521) is a contradiction inside omega's reader `tools/omegese/omegese.c`, not in
  the mask: at a line that must end at exactly the 96-words-per-line limit, `omegese next` lists a
  separator as legal, then refuses the same prefix plus that separator, while `omegese complete` on the
  prefix writes a tail that begins with that separator and `omegese emit` accepts the result. The mask
  followed `next`'s answer. Repro files: `~/handoffs/l112-omegese-maxwords/` on the Spark; fixer ledger
  L156 (omega branch fix/omegese-maxwords-boundary). G1 is to be rerun when that fix lands.
- Run 1 (`g1-run1-20261020T0104Z-FAIL.txt`) refused 30 walks; the cause was the test itself (eight walk
  threads shared one scratch file name, and the walks wrote only blank space). Fixed in cc0b1026.

Also green on the same build: the crate's and aien-serve-http's ordinary tests; omega
`make test-omegese-lib` (library answer == command answer on 10,899 prefixes, plus an ASan + UBSan
build, no leak); the negative control (a reset list missing one variable gives 436 mismatches of 489).

## G2 the trial and G3 the cost: `g2-20261020T0202Z/README.md`

- 10 dev tasks on the 27B, mask off and on, same seeds: 0 refusals both ways, 10 of 10 readable both
  ways, 8 of 10 correct both ways, replies byte-identical. The dev tasks gave the mask nothing to prevent.
- Forced positions: 3 per answer. Constraint host time 14.82 ms per token, 9.0 percent of a decode step.
  The 27B graph computes its output head inside the token call, so a forced position skips the logits
  read and the sampler only.
