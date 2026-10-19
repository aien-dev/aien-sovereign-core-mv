# OMEGESE-MASK: predictions, written before any G2 or G3 run (ledger L112)

Written 2026-10-19 before the mask-on server existed and before any trial request. Nothing below was
measured; these are the numbers the runs are judged against.

## G2: the 10 dev tasks, omegese arm, many-shot prompt, one attempt, mask off and on

Model: Qwen3.8-27B FP8 on our engine (aien-serve-http, the 27B door: alpha/serve-path plus this branch,
built with the `omegese` feature), one server binary for both modes. Sampling as tools/exp445/PARAMS
(temperature 0.7, top_p 0.8, max_tokens 3072), seed = 445000 + 1000 + task index, the same seed in both
modes. Driver: omega `tools/exp445/mask_trial.sh` (reader.sh prompts and reader, `ola.sh experiment` for
the verdict).

| count (of 10) | mask off | mask on |
|---|---|---|
| reader refusals (no candidate: omegese emit or omega_emit refused) | 3 (range 1 to 6) | 0 (any refusal is a bug in the mask or in `next`) |
| reader check OK (x445 check: signature, checker2) | 7 | 10 |
| harness ACCEPT (correct behaviour on the dev cases) | 4 (range 2 to 7) | 5 (range 2 to 8), never fewer than off minus 1 |

Completion tokens: mask on writes no prose before or after the fenced block (the fence is forced first and
the stop right after the closing fence), so its mean completion tokens per answer is 60 to 85 percent of
mask off's.

Forced positions (one legal token, no sampling) per mask-on answer: median 12 (range 6 to 30): the opening
fence, the closing fence and stop, plus words the grammar leaves no choice on. Heads skipped (no output
head read): forced minus about 2 (the first token after prefill comes with the prefill's head; a forced
stop is not fed).

## G3: time per token, mask on vs off

Same prompts, one request at a time, under quiet-measure, time as the server reports it.

- Decode step of the 27B today: about 126 ms per token (server log, 2026-10-19, other agents' load).
- Constraint time per masked step (scan of the 248,077-token vocabulary plus the linked reader): median
  5 ms, p95 15 ms. Reference: the 2026-10-15 subprocess oracle took 13.4 ms median per step on the
  151,669-token 30B vocabulary, about 6.6 process starts per step.
- Mask-on wall time per token: within +10 percent of mask off.
- At a forced position: the scan that finds the single legal token costs about as much as any masked
  step, so the mask alone does not pay for itself there; the saving is the output head it skips. Reported
  either way.
