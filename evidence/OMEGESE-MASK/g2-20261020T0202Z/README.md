# OMEGESE-MASK G2 + G3: the 10 dev tasks with the mask off and on (ledger L112)

Window 20261020T021420Z-f6cde2 (spark-window, owner L112), 2026-10-20 02:21Z to 02:54Z.
Predictions were pushed before any run: aien-sovereign-core feat/omegese-mask
evidence/OMEGESE-MASK/00-prediction.md (007730e2).

## Setup

- Server: a second Qwen3.8-27B FP8 server on 127.0.0.1:18191, binary sha256 86c99350 built from
  sc feat/omegese-mask-q38 374f0394 + omega feat/omegese-mask-lib 54d430f9 + libomega_gpu bd6802a5,
  same flags as the live :18091 server. Started at window open, stopped at window close.
- Control: one greedy mask-off request (control.req.json) to :18191 and to the live :18091: the same
  reply (`window.log`: CONTROL SAME).
- Trial: `tools/exp445/mask_trial.sh` off, then on; omegese arm, many-shot prompt, one attempt,
  tools/exp445/PARAMS sampling (temperature 0.7, top_p 0.8, max_tokens 3072), seed 445000 + 1000 + task
  index, the same seed in both modes. Each mode under `spark-receipt` + `quiet-measure` (receipts copied
  here). Verdicts: `ola.sh experiment` (checker2 and the dev cases).

## G2 result

| count (of 10) | predicted off | measured off | predicted on | measured on |
|---|---|---|---|---|
| reader refusals | 3 (1 to 6) | **0** | 0 | **0** |
| reader check OK | 7 | **10** | 10 | **10** |
| harness ACCEPT | 4 (2 to 7) | **8** | 5 (2 to 8) | **8** |

- The two REJECTs are the same tasks in both modes (dev07, dev08): wrong behaviour, valid programs.
- Every mask-on reply is byte-identical to its mask-off reply (`cmp` of r1.content, 10 of 10): at this
  seed the model never picked a token the mask would refuse, so the mask changed nothing.
- Completion tokens: 422 in both modes (predicted on = 60 to 85 percent of off: wrong; the model already
  writes only the fenced block, no prose).
- Forced positions per mask-on answer: **3** in every answer (predicted median 12). Logits reads skipped:
  1 per answer. Fed by decode step: tokens minus 1; batched feeds: 0.

What this trial cannot show: the mask's purpose is the refusals it prevents (46 emit refusals in the
held-out trial, OMEGESE.md section 7). On these 10 dev tasks the model made no refusal even without the
mask, so there was nothing to prevent. The zero-refusal guarantee is shown by G1 (random walks), not here.

## G3 result (from the server's own log lines, `serve-window.log`, table `g3.tsv`, script `g3.awk`)

| | mask off | mask on |
|---|---|---|
| tokens | 422 | 422 (the same text) |
| decode time | 72,127 ms | 69,473 ms |
| per token | 170.9 ms | 164.6 ms |
| constraint (host) time | 0 | 6,255.5 ms = 14.82 ms per token (calc:fee7bd51) |
| worst single constraint step | | 84.4 ms |

- The constraint's own time is 9.0 percent of a mask-on decode step (calc:45a596e7). Predicted 5 ms median
  per step: the measured mean is about 3 times that.
- Mask on was 3.7 percent faster than off over the same 422 tokens (69473/72127 = 0.9632,
  calc:656f588f). That is noise, not a gain: per-answer decode ms/token ranged 157 to 193 off and 155 to
  181 on, and the mask adds host work, it does not remove chip work.
- Forced positions: on this engine the output head and argmax run inside the chip graph's token call,
  so a forced position saves only the logits read and the sampler, not the model step. No model step is
  replaced, so "the mask must not cost more than the step it replaces" has nothing to compare against;
  the closest bound: the worst constraint step (84.4 ms) is below the cheapest model step seen (151.6 ms).
- The box was not quiet: load 35 to 54 (off) and 54 to 66 (on) in the receipts; `quiet-build` lets a
  build waiting over 20 minutes run anyway. The chip lock is shared with the live :18091 server (request
  lock waits up to 131 s, not counted in decode time). Kernel logged NVRM out-of-memory errors at 02:47Z
  (MemAvailable fell to 8.6 GB with two 27B servers resident; receipts: nvrm_errors_delta 112 off, 32 on).
