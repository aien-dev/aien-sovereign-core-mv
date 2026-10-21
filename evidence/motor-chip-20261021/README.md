# evidence/motor-chip-20261021: the motor on the chip (L312, Drake 2026-10-21: the motor IS the 35B server)

The 35B (Qwen3.6-35B-A3B FP8, `~/models/qwen3.6-35b-a3b-fp8-95a723d0`) on L306's Omega chip kernels (omega `q36/serve`
739c80e8, lib sha 678df626), gated for identity against the host goldens (`evidence/q36-host-20261020/`) before it serves,
then served on 18091 behind the OpenAI-style door and gated again through the door. Shared chip (`/tmp/aien-gb10.lock`),
not a quiet window: the per-token times here are receipts of the plain path, not a speed claim.

## Gate 1, the mathematics (`qwen38_omega_full`, one 40-layer window, teacher forcing over the golden's 32 tokens)
Rule (sc `motor/serve` 71ab16c6): the chip's words equal the golden's, except where the host reference's own top-two logit
margin is below a band written in advance (0.1 logit, `MARGIN_MIN`); a flip at a wider margin fails; the chip's greedy choice
must agree at every position outside the band; the shifted-by-one control must be caught; the host reference must equal the
golden. The cosine is recorded, not gated (open precision item, parked with L306). The negative control
(`AIEN_Q36_TEST_MUTANT=4`, the shared expert dropped from COMBINE) must fail the same rule.

### First pass (bin 8f86188a, the band taken from the run itself: WITHDRAWN as a gate, kept as data)
`gate1-p1..p4.json`, `gate1-p1-mutant4.json`, logs beside them, `run.log`. The band was the run's own max logit error
(`max_abs`), so a broken run excused its own flips: the mutant passed with 22 flips "inside the band" at max error 23.8
(01:58:27Z). The coordinator caught it (brief 02:05Z); the rule above replaced it. The per-step data (token, golden,
reference margin, chip error, cosine, chip time) are unaffected and were re-judged under the written band (jq over the JSONs):

| run | words identical | flips, margin < 0.1 (allowed) | flips, margin >= 0.1 (fail) | greedy agrees at every position | cosine(min) | max logit error over the vocabulary |
|---|---|---|---|---|---|---|
| p1 | 32/32 | 0 | 0 | yes | 0.999529 | 0.419 |
| p2 | 32/32 | 0 | 0 | yes | 0.994874 | 2.425 |
| p3 | 32/32 | 0 | 0 | yes | 0.999531 | 0.576 |
| p4 | 31/32 | 1 (pos 11, margin 0.0096) | 0 | yes | 0.998920 | 1.033 |
| p1 mutant 4 | 10/32 | 0 | 22 (smallest margin 0.214; 10 wider than 2.425) | no | 0.081793 | 23.819 |

The correct runs' vocabulary-wide error reaches 2.425 while every word stays identical: the error at the top two logits is
far below the vocabulary max, so a vocabulary-wide error cannot serve as the band, and the written band is the rule.

### Rerun with the fixed rule (bin 7c75dd62, mutant first: red before green)
`gate1b-p1-mutant4.json` then `gate1b-p1..p4.json` (script `~/handoffs/motor-scripts/gate1b-motor.sh`, log `~/handoffs/motor-gate1.log`).

| run (bin 7c75dd62, written band 0.1) | verdict | words identical | flips inside the band | flips beyond (fail) | greedy agrees everywhere outside the band | cosine(min) | max logit error (vocab) | upload s | chip ns per token, median |
|---|---|---|---|---|---|---|---|---|---|
| p1 mutant 4 (negative control, run first) | FAIL exit 1 | 10/32 | 0 | 22 | no | 0.081793 | 23.819 | 54.2 | 22,866,878 |
| p1 | PASS | 32/32 | 0 | 0 | yes | 0.999529 | 0.419 | 53.4 | 23,223,588 |
| p2 | PASS | 32/32 | 0 | 0 | yes | 0.994874 | 2.425 | 46.5 | 22,966,881 |
| p3 | PASS | 32/32 | 0 | 0 | yes | 0.999531 | 0.576 | 55.8 | 23,091,844 |
| p4 | PASS | 31/32 | 1 (pos 11, margin 0.0096) | 0 | yes | 0.998920 | 1.033 | 50.9 | 23,599,980 |

Red before green held: the mutant went red at 02:04:30Z, the four prompts green by 02:08:54Z (`~/handoffs/motor-gate1.log`).
The routes half of the gate rests on L306's router probe (`q36_router_check`, 6,520 rows identical; `sc-q36` branch
`q36/serve`, `evidence/q36-chip-20261020/`): the device keeps one ids slot per router, so this tool sees only the words.

## Gate 2, the door (`~/handoffs/motor-scripts/gate2-http.sh`, server `aien-serve-http --engine omega --thinking off`)
One greedy chat completion (the golden's prompt, 24 tokens, `enable_thinking false`, `aien_return_token_ids`) and its 24
ids against `evidence/q36-host-20261020/gen-chat-max-24.json`. Server binary pinned at `~/.local/state/aien-q36/bin-5b726818`
(sc `motor/serve` 5b726818: the 35B chat template pinned as `ChatTemplate::Qwen36`; L306's `bin-1cf8fb09` died at start
with "unsupported ChatML chat template").

| server | port | verdict | receipt |
|---|---|---|---|
| mutant 4 (`AIEN_Q36_TEST_MUTANT=4`), run first | 18093 | red: ids differ (rc 1) | `gate2-mutant4.json`, 01:55:24Z |
| the real one, `qwen3.6-35b-a3b-fp8` | 18091 | green: 24 ids identical (rc 0) | `gate2-real.json`, 01:58:28Z |

18091 has served since 01:57:36Z (`ledger note L305`, `L306`). Caveat for every caller: the server takes `/tmp/aien-gb10.lock`
around each generation (waits 30 s, then HTTP 503 busy), so while another chip job holds the lock an answer waits or is refused.

## Chip time per token, the plain path (one token per command list; shared chip, no quiet window; receipts, not a claim)
First pass medians of `chip_ns_all_windows`: p1 21.863782 ms = 45.74 tok/s (calc:131a3962, calc:593290c1); p2 22.989067 ms =
43.50 tok/s (calc:d18b9629, calc:e9362744); p3 21.990048 ms = 45.48 tok/s (calc:78354ed1); p4 22.600668 ms = 44.25 tok/s
(calc:dde6add0). The rerun's medians (table above, 22.9 to 23.6 ms) were taken with the 35B server resident beside the gate
tool. The coordinator's read through the door at 02:04Z: 42.0 to 44.4 tok/s decode, 0.67 of the 63.25 tok/s ceiling at
today's bytes.

## Stroke v1, four conversations per command list (`qwen36_stroke_gate`, sc 5637a828 over omega a8036530, lib 18fabc97; bins 26343923 for the two red runs, 73dbaac6 and 789d87af for the two green ones, per row)
One multi-head graph (`Q36Graph::create_heads`, at most 4: a 40-layer token records 11 x 40 + 4 launches and the list holds
2048); p1..p4 live in slots 0..3 with their own DeltaNet state, attention cache and position; every step is ONE stroke
(every head's layer l before layer l + 1, one submit), teacher forcing with each golden's own tokens; a head whose sequence
ended keeps taking its last token and is not judged after its end. Rule per conversation = gate 1's written-band rule
(0.1 logit, fixed in advance). Red before green, in this order:

| run | what | result |
|---|---|---|
| `stroke-mutant4` 02:53Z | `AIEN_Q36_TEST_MUTANT=4`, must FAIL | FAIL (exit 1): p1 10/32, p2 30/32, p3 15/32, p4 13/32; 22, 1, 17, 18 flips beyond the band. Head 0's 36 tokens equal the single-head mutant run (`gate1b-p1-mutant4.log`) token for token: the stroke is the one-token mathematics |
| `stroke-swap-control` 02:58Z | slots 0 and 1 exchange state objects at step 18, feeds stay; must FAIL | FAIL (exit 1): p1 and p2 diverge from step 18 on (25/32, 28/32); the untouched p3 32/32 and p4 31/32 (its known in-band flip at 11) stay identical, greedy agrees, shifted-by-one caught |
| `stroke-p1234` 03:40Z (bin 73dbaac6, built from 5637a828 before the tool stamped its own sha) | the real run, must PASS | PASS (exit 0): p1 32/32, p2 32/32, p3 32/32, p4 31/32 (its one in-band flip at 11, margin 0.0096, is the same flip as the single-head run); every head's chip tokens equal its single-head run (`gate1b-p<n>.json` `chip_ids`) 32/32 on all four; greedy agrees outside the band, shifted-by-one caught; cosine min 0.9949 to 0.9995 recorded, not gated |
| `stroke-swap-feature` 03:45Z (bin 789d87af, sc 3dbbad4d: the tool waits for memory in a loop and stamps its own sha, the gate code is unchanged) | the exchange with the feeds moved along (head swap as a feature), must PASS | PASS (exit 0): p1 ends in slot 1, p2 in slot 0; all four identical to the single-head runs 32/32 (p4 with the same in-band flip): a conversation follows its state and cache, the slot number means nothing |

First run (lib 87ffd31e, bin ee17bec9, `~/handoffs/motor-stroke-run1/`): mutant red, BUT head 0 differed from the single-head
mutant run at every position, which the same mutant cannot do. Cause (omega `omega_gpu_q36_graph.c` `record_head_layer`): the fused
COMBINE|NORM writes the next layer's normalised input into `xq` and the next layer skips its NORM; `xq` was one shared slot, so
with heads interleaved per layer every head read the last head's x'. Fixed per head (a8036530); the other scratch is produced
and consumed inside one head's contiguous launches (barriers between launches by default) and stays shared.

Chip time per stroke (shared chip, no quiet window, four tokens per stroke; a receipt only under `spark-window`): real run median 92.887032 ms per stroke of four tokens (46 strokes, min 90.378, max 101.353 ms) = 43.06 tok/s across the four heads (calc:9c07e26f), 23.22 ms per token-equivalent (calc:3ae78e68); swap-feature median 93.448174 ms = 42.80 tok/s (calc:18064ffe), 23.36 ms per token-equivalent (calc:ab4e0439); the red runs sat at 94.228 and 93.206 ms. Against the single-head medians above (21.86 to 22.99 ms per token, same shared chip) a v1 stroke costs about four single tokens, as built: every head reads the weights once per stroke; v2 batches the dense GEMVs at M = 4. Not a speed receipt: the 18091 server and two other lanes' tools shared the chip, and the `spark-window` line was stuck during these runs (the window run is still owed).

## Eight conversations per command list (`qwen36_stroke_gate`, sc 4e0d36d2 over omega 52dd18a8, lib 580171ef, bin 0d1f796b)
Coordinator 02:22Z: eight at once. omega: list cap 2048 -> 4096 launches, MAX_HEADS 8 (444 launches a token, 3552 for eight; the
submit already splits a list over the 64 KB ring as it needs). Four more goldens from the host reference join p1..p4
(`../q36-host-20261020/gen-chat-{code,prose,chat,review}.json`, chat template, 32 tokens each, `qwen38_check` bin 565cccf3).
Same rule per conversation as above. Two sets, because one prompt fails alone:

| run | heads | result |
|---|---|---|
| `stroke8-mutant4` 03:58Z | 8 (set a: p1-p4, code, prose, chat, review) | FAIL as required: every head diverges (p1 10/32 ... prose 5/32), 1 to 27 flips beyond the band each |
| `stroke8-swap-control` 04:00Z | 8 | FAIL as required: p1 and p2 diverge from step 18 (25/32, 28/32); the untouched heads unchanged |
| `stroke8-real` 04:02Z | 8 | 7 of 8 PASS; `gen-chat-prose` FAILS the band rule at pos 63 (chip 8720, host 26606, margin 0.194, max_abs 0.98; cosine min 0.9914 at step 28) |
| `stroke1-prose` 04:03Z | 1 | the same conversation ALONE: the same 67 per-step lines, the same tokens (31/32, the same flip at 63). One-token chip precision, not the stroke; noted to L313 |
| `stroke1-{code,chat,review,prose2}` 04:05-04:07Z | 1 | each alone 32/32 |
| `stroke8-vs-alone.txt` | | every head of `stroke8-real` equals its run alone, 8 of 8 (p1-p4 against `gate1b-p<n>.json`) |
| `stroke8b-mutant4` 04:10Z | 8 (set b: prose -> prose2) | FAIL as required |
| `stroke8b-swap-control` 04:12Z | 8 | FAIL as required |
| `stroke8b-real` 04:14Z | 8 | PASS: p1, p2, p3 32/32, p4 31/32 (its known in-band flip), code, prose2, chat, review 32/32; every head equals its run alone (`stroke8b-vs-alone.txt`) |
| `stroke8b-swap-feature` 04:16Z | 8 | PASS: p1 ends in slot 1, p2 in slot 0, all eight identical to golden and to their runs alone (`stroke8b-feature-vs-alone.txt`) |

Ladder, the same command and binary back to back (`stroke1-real`, `stroke2-real`, `stroke4-real`, `stroke8b-real`; shared chip next
to other lanes' tools, the 18091 server was down at the time; not a speed receipt, that needs `spark-window`):

| heads | ms per stroke (median) | ms per token | tok/s total |
|---|---|---|---|
| 1 | 24.380 (36 strokes) | 24.38 | 41.02 (calc:eab1e7fb) |
| 2 | 49.096 (46) | 24.55 (calc:946a5354) | 40.74 (calc:35ba22a1) |
| 4 | 97.805 (46) | 24.45 (calc:73a5fb9b) | 40.90 (calc:24df1ea6) |
| 8 | 194.143 (97) | 24.27 (calc:dbf658ca) | 41.21 (calc:2172b972) |

v1 is linear in N, as built: every head reads the weights once per stroke, so eight conversations cost eight single tokens and the
total rate stays at one conversation's. The gain is v2: one weight read per layer for all heads (the dense GEMVs batched at
M = heads through the segment-table GEMM, which is bit-identical to the GEMV per row; the experts merged over the heads' picks).

Expert union per layer per decode step, from the goldens' routes (`scripts/expert-union.sh`, `expert-union-*.json`): 8 different real
prompts 54.7 (set a) / 54.5 (set b), min 33-35, max 63-64 per cell, against 57.42 for independent draws (calc:aa517f3a); 4 chat
prompts 29.7 against 30.53; 2 code prompts 15.3 against 15.75. Different real conversations route almost independently. The 30.5 at
N = 8 in ENGINE.md's stroke table came from 32 traces of one prompt family (correlated rows), the optimistic case; a byte budget for
eight conversations must count about 55 of 256 experts per layer per beat.

## v2, the ignition: one weight read per layer for all heads (`AIEN_STROKE_V2=1`; omega d1fca73c, sc 380e65e9, lib 008f5c1a, bin 751d4ec7)
Each layer is recorded once for every head in the stroke: the dense mixer and o_proj GEMVs run as one FP8 segment-table GEMM at
M = heads (bit-identical to the GEMV per row, `gpu_fp8gemm_test`), the N routers write their picks into per-conversation launch
buffers, a GROUP glue kernel on the chip sorts the N x 9 picks into expert segments with row maps (`q36_dec_sim_test`, byte-identical
to a host sort, three mutants caught), and one gate_up and one down grouped GEMM serve all heads (host-only runs in `v2-gemm-sim.log`).
GDN, attention, the router and the combine stay per head; the final head (lm_head, argmax) stays per head. The same gate, set b:

| run | heads | result |
|---|---|---|
| `stroke8v2-mutant4` 05:07Z | 8 | FAIL as required; its chip tokens equal the v1 mutant run (`stroke8b-mutant4`) on all 8 heads |
| `stroke8v2-swap-control` 05:09Z | 8 | FAIL as required (p1, p2 diverge from step 18) |
| `stroke8v2-real` 05:11Z | 8 | PASS 8/8; chip tokens equal `stroke8b-real` (v1) on all 8 heads and equal every run alone (`stroke8v2-vs-alone.txt`) |
| `stroke8v2-swap-feature` 05:13Z | 8 | PASS (p1 ends in slot 1, p2 in slot 0) |
| `stroke{1,2,4}v2-real` 05:14-05:16Z | 1, 2, 4 | PASS |

Ladder, v2 against v1 (the same command, binary and goldens, back to back; shared chip, not a speed receipt):

| heads | v2 ms per stroke (median) | v2 ms per token | v2 tok/s total | v1 ms per stroke | v1 / v2 |
|---|---|---|---|---|---|
| 1 | 27.294 (36 strokes) | 27.29 | 36.64 (calc:9795d31a) | 24.380 | 0.89 (v2 costs 1.12x alone, calc:1262c85d) |
| 2 | 41.221 (46) | 20.61 (calc:4cbfde45) | 48.52 (calc:4776090d) | 49.096 | 1.19 (calc:3512da13) |
| 4 | 68.037 (46) | 17.01 (calc:62cb0468) | 58.79 (calc:e9287378) | 97.805 | 1.44 (calc:0c8385cb) |
| 8 | 127.753 (97) | 15.97 (calc:1750973d) | 62.62 (calc:941ee66f) | 194.143 | 1.52 (calc:cbb28558) |

v2 is no longer linear in N but it is still close: 14.35 ms per extra head (calc:5397d25b, N=1 to N=8), where the byte model says
0.21 GB = 0.77 ms per row at the 273 GB/s ceiling. The eight-head beat moves about 11.5 GB of weights (union 57 experts) in 127.75 ms,
90 GB/s against the 273 GB/s ceiling (calc:48eb2528). The per-head launches (GDN, attention, router, combine, lm_head: about 240 per
head per stroke) carry the cost now; the next receipt is the chip timeline per launch kind (`AIEN_STROKE_STAMPS=1`, `stroke8v2-stamps`).

## Files
`gate1-*.json/.log` first pass (band withdrawn, data kept), `gate1b-*.json/.log` the rerun, `gate2-*.json` the door, `stroke-*.json/.log`
the four stroke runs (`stroke_chip_ns` per stroke, `conversations[].steps`), `stroke8-*` and `stroke8b-*` the eight-head sets, `stroke1-*`
the runs alone, `stroke{1,2,4}-real` the ladder, `*-vs-alone.txt`, `expert-union-*.json`, `scripts/` the scripts that produced them, `run.log`
the chip log with binary and lib shas; the scripts live in `~/handoffs/motor-scripts/` (not in the repo: they carry home paths).
