# SPEED-7: 8-bit attention memory (KV cache) for Qwen3.8-27B decode, whole model (ledger L131, 2026-10-20)

Engine: aien-sovereign-core `qwen38/fp8-kv` 3fc1132e (switch `AIEN_Q38_KV8` = 0 bf16, 1 K and V 8-bit, 2 V only,
3 K only). Kernel: omega `qwen38/fp8-kv` (3efab8c1 split modes, c9c7251f mutant fix); its own evidence folder is
omega `evidence/SPEED7-KV8-3efab8c1-20261020`. An 8-bit row is 256 e4m3 bytes plus one power-of-two scale.

## Verdict

1. Quality (drift gate written down before any run, unchanged since): V only PASS 6 of 6 texts. K and V together
   FAIL 3 of 6. K only FAIL 4 of 6. The loss sits where the text copies an earlier passage (arch text, positions
   1312-1366): 8-bit K blurs the sharp attention that copying needs; 8-bit V does not (INFERRED from where the loss sits).
2. Build repeat: the same bf16 run on two builds is bit-identical on all 6 texts (OBSERVED). This shows the reference
   is stable; it does not measure the gate's noise floor. The pre-registered summation-order null control was NOT RUN
   (the engine has no knob for it).
3. Speed: NOT MEASURED. The pre-registered rule was "ship V only after it is timed at 8k, 32k, 64k and 128k against
   bf16". The timing runs exist but are not valid (below), and the 27B model was retired and its weights deleted on
   2026-10-20 (Drake, 20:28Z: MoE only), so they cannot be repeated. V only is therefore NOT shipped.
4. Paper saving for V only: 772 of 1024 bytes per position per head = 0.754 of the bf16 cache (calc:49c732fde45c).
   At 32k that is 528 MB less read per token (calc:0a8ccce531af), at most 1.94 ms per token at the 273 GB/s ceiling (calc:46f1a8434225),
   and less in practice because the kernel is not limited by memory traffic (omega evidence: bf16 reaches 0.25 of
   the ceiling, 8-bit 0.16).

## Drift gate (drift/gate-table.txt)

Rule per text, reference A = bf16 cache, plain FP8 weights: top-pick agreement >= 98.0%, every flip a near tie
(A margin <= 0.5 logit), mean KL <= 0.01, |dNLL| <= 0.005 nats/token; golden must agree at all 47 positions.

| form | golden | license | prose | codelong | code | arch | gate |
|---|---|---|---|---|---|---|---|
| K+V 8-bit | dNLL +0.0171 FAIL | pass | dNLL +0.0052 FAIL | pass | pass | dNLL +0.0244 FAIL | FAIL 3/6 |
| V only | pass | pass | pass | pass | pass | pass (2036/2047, +0.0011) | PASS 6/6 |
| K only | dNLL +0.0083 FAIL | pass | dNLL +0.0059 FAIL | dNLL +0.0057 FAIL | pass | dNLL +0.0245 FAIL | FAIL 4/6 |
| bf16 build repeat | identical | identical | identical | identical | identical | identical | PASS |

The per-text comparison files are in `drift/<form>/<text>.json`. The raw per-token dumps they were made from were
removed from the Spark during the 15:59Z cleanup; the comparison files are the record.

Separate exactness check (`golden/`, qwen38_omega_full, logit cosine >= 0.9999 is a lossless bar): bf16 PASS 0.999968.
Every 8-bit form keeps all 24 generated tokens identical but misses the cosine bar (V only 0.998937, K only 0.996425,
both 0.998570). A lossy format cannot meet a lossless bar; this is reported, not part of the gate.

## Why the speed runs are not valid (speed/)

Window 20261020T185133Z-3f76bb (windows/), 4-bit weights + FP8 head, 512 steps per point, rounds in A B B A order:

| context | bf16 r1 | V only r1 | V only r2 | bf16 r2 |
|---|---|---|---|---|
| 8k | 93.0 ms | 208.9 ms | 245.1 ms | 239.9 ms |
| 32k | 239.3 ms | 235.4 ms | 118.7 ms | 268.5 ms |
| 64k | 268.3 ms | 262.0 ms | 270.1 ms | 304.9 ms |

(chip median ms per token, copied from `speed/*/summary.txt`)

- The same setting differs 2.58 times between its two rounds (bf16 8k, calc:4c1fab6bddcc), and single steps inside one point
  range from 79.8 to 255.9 ms (bf16 8k r1, 512 steps). No ordering of these numbers says anything about the format.
- At window open the GPU was already 96% busy: the 27B server on port 18091 (34 GiB) and a Python GPU job ran on the
  chip without honouring the window (open.json `gpu_apps`; ledger L113). Load average 45 to 53.
- The window holder was reaped at 20:12Z: V only r2 (started 20:09Z) straddled the close and bf16 r2 (20:18Z) ran wholly outside the window.
- `speed/energy-table-raw.txt` also shows a "fraction of 273 GB/s" above 1.0, which is impossible: the script counts
  every resident weight copy (26.9 GB, FP8 originals beside the 4-bit copies) as read per token. That column is wrong
  for 4-bit runs; the fraction of the memory ceiling for these runs is UNMEASURED. J/token there is wall power of the
  whole loaded box over the steps' span, not this job's energy: UNMEASURED for the job.
- Not run: 128k point, third round pair, 32k long drift for V only (`long/` has only the bf16 baseline: 32767 steps,
  chip median 90.20 ms, mean NLL 0.44700, at load average 113 in an earlier window, so its timing is contaminated too;
  the V only run failed at channel create on low GPU memory).

## What carries over

- 8-bit V with a power-of-two row scale held the gate on every text; 8-bit K did not. Any later model's attention
  cache should try V first and leave K at 16 bits unless a K form passes the same gate.
- The decode attention kernel is the bigger lever: it reaches about a quarter of the memory ceiling at 32k to 128k.
- Kernel modes are shape-specific (head size 256). The served MoE model has a different head size; nothing here runs
  on it without a new kernel shape.

## Files

`drift/` comparison files, gate table, run commands and ground; `golden/` exactness runs; `speed/` timing runs with
command, ground (binary sha, git heads, free memory, load) and wall-power receipts; `long/` bf16 32k baseline;
`windows/` box state at window open and close; `bin/` the scripts that produced them (driver, job wrapper, gate).

CALC-EXEMPT: figures without a calc id are measured readings copied from the files in this folder.
