# Qwen3.8-27B speculative decoding with the checkpoint's MTP head, and a 4-bit drift study (sc#679)

All runs on the Spark GB10, in-process with `qwen38_spec` (this branch), under `flock /tmp/aien-gb10.lock`,
while the live 27B server on 18091 kept running (it was not ours to stop, so nothing went through it).
Commands: `command-*.txt`. Binaries: gen3 (main runs), gen5 (stop fix: controls on prose, code reruns),
gen6 (task B). Library: omega 37034f03 archive, sha256 6e905209... (see command files).

## Exactness (task A)
The verify path ("exact rows") checks k drafted tokens in one pass, and every row of that pass is bit for bit
what a one-row pass gives (probe-8: 48 rows in one pass == 48 one-row passes, m = 2..5; rollback exact for
j = 1..4; control red). k = 0 is the same path one token at a time. Every k > 0 run in this folder produced
exactly the k = 0 tokens; the decode graph (the live server's path) agrees with the exact path on every
token (30/30 code, 256/256 prose, 256/256 codelong); the golden 24 tokens match.

Negative controls (gen5, prose, n 96, k 2), each must go red:
| fault | result |
|---|---|
| accept the first refused draft (extra) | FAIL, identical_to_k0 false |
| no DeltaNet rollback (norollback) | FAIL |
| check row r against row r+1 (rowshift) | FAIL |
`golden-controls-weak/`: the same faults on the short code prompt; extra and norollback stayed green
because no draft before the end mark was refused there (the faults never fired before the end). Kept as a
record of why the controls moved to prose. `stop-bug-gen3/`: the code run before the stop fix (an accepted
draft that was the stop token did not end generation; extra rounds were thrown away; tokens were right).

## Speed (ms per token, decode only, 10.15 tok/s plain ceiling = calc:9822c2ba)
| prompt | exact k0 | k1 | k2 | k3 | k4 | decode graph (chip) |
|---|---|---|---|---|---|---|
| code, golden (30 tokens, gen5) | 203.53 | 143.35 | 186.88 | 155.52 | 169.48 | 155.36 |
| codelong (256, gen5) | 215.46 | 151.01 | 185.66 | 162.62 | 217.62 | 158.50 |
| prose (256, gen3) | 212.51 | 169.06 | 253.11 | 255.79 | 349.46 | 156.14 |

Acceptance (accepted / drafted): codelong k1 120/136, k2 161/192, k3 179/234, k4 185/288;
prose k1 103/152, k2 127/258, k3 136/363, k4 138/480; code k1 15/15, k3 23/24.

Receipts:
- codelong k1 6.622 tok/s (calc:b5df6210), 0.6524 of the ceiling (calc:128ae612), 1.05x the graph
  (6.309 tok/s, calc:45fde843; ratio calc:ca6d18af), 1.427x exact k0 (calc:db337092). k3 6.149 (calc:ee676172).
- code k1 6.976 tok/s (calc:26a3d46a), 0.6873 of the ceiling (calc:b970cb2a), 1.084x the graph
  (6.437, calc:8f54e237; ratio calc:88ba9f34). 30 tokens only.
- prose k1 5.9152 tok/s (calc:a6629fef), 0.5828 of the ceiling (calc:64075f3d), 0.9236 of the graph's speed
  (calc:47d8f26d; graph 6.4044 tok/s calc:1fe4a96f, 0.631 of the ceiling calc:8bc62008); 1.257x exact k0
  (calc:f5f65fb8). k2 3.951 (calc:081081c3), k3 3.909 (calc:611a4b54), k4 2.862 (calc:62d52a0a).

Why it is not faster: the verify GEMM (omega fp8gemm, the exact-per-row kernel) is compute bound past one
row pair: it takes rows in tiles of 2 (mt 2), so 3 rows cost what 4 do and 5 what 6 do (k2 and k4 are the
slow columns); 5 rows of gate|up take 3.09x the time of 2 (calc:45e310b8, omega evidence
fp8gemm-mt-sweep-20261019T0327Z). The exact path at one row is also 1.36x slower than the graph (prose, calc:75203b44).
Drafting costs 2.4 to 5.1 s per 256 tokens (mostly the bf16 lm_head for each draft). Follow-ups: a small-M
dense FP8 kernel that streams the weights once for 2 to 5 rows at bandwidth, an FP8 lm_head copy for drafts,
and the m-row form inside the graph.


## Task B: 4-bit weights (MXFP4) drift study
What was simulated: every FP8 projection of the 64 layers (400 tensors, 24,326,963,200 weights) converted at
load to OCP MXFP4: 4-bit e2m1 values with one power-of-two scale per 32 weights (qwen38_omega::w4). The
converted values are held exactly by the FP8 kernels (0 weights fell outside a block's range), so the chip
runs the true 4-bit model through the unchanged path. lm_head, embeddings and the small bf16 pieces stay as
they are. Relative RMS change of the weights: 0.1167. Scored: the LICENSE passage, 256 tokens (255 scored),
teacher-forced on the exact rows path; logits of every position compared with the FP8 run's.

| run | mean NLL | perplexity | next-token accuracy | argmax agree with FP8 | KL(FP8 || run) mean / max | smallest top-two margin |
|---|---|---|---|---|---|---|
| record (reference) | 0.08707 | 1.09097 | 0.9882 | | | |
| FP8 (drift-fp8) | 0.08708 | 1.09099 | 0.9882 | | | 0.641 |
| FP8 again (control: same run twice) | 0.08708 | 1.09099 | 0.9882 | 255/255 | 0 / 0 | 0.641 |
| MXFP4 (drift-mxfp4) | 0.09249 | 1.09691 | 0.9882 | 255/255 | 2.818e-3 / 0.204 | 0.296 |

- The FP8 run reproduces the record (within 2 percent: PASS) and repeats bit for bit (KL exactly 0), so the
  MXFP4 differences are the 4-bit weights alone. The two-ahead control scores 21.4 nats per token in every
  run (the scorer tells right from wrong tokens).
- Quality cost on this passage: mean NLL 1.062x the FP8 value (calc:73ed24e2), perplexity 1.005x
  (calc:4d3fe824); no greedy choice flipped, but the closest call moved from a 0.641 to a 0.296 logit margin.
- Limits: this passage is easy for the model (98.8 percent of next tokens predicted); a harder text may
  drift more (NOT MEASURED by this scorer). Free greedy generation with MXFP4 weights: see below.
- Speed upside, on paper only (no 4-bit kernel exists here, NOT MEASURED): MXFP4 reads 17/32 byte per weight
  instead of 1, which takes 11.4 GB off the 26.9 GB read per token (calc:225be907) and would put the
  bandwidth ceiling at 17.62 tok/s instead of 10.15 (calc:6b16920a).
- Conversion takes 1275 s on the host (one thread, f64); that is tooling time, not part of any result.

Greedy generation with MXFP4 weights:
- golden code prompt (gen-golden-mxfp4): the same 30 tokens as FP8, golden 24/24 PASS; the graph agrees on
  all 30. Speed is unchanged (154.39 ms/token graph chip) because the 4-bit values still sit in FP8 bytes.
- story prompt, 256 tokens (gen-prose-mxfp4 against gen-prose k0): DIFFERENT from the first token on. FP8
  starts "Elias had spent forty years listening to the rhythm of the North Sea..."; MXFP4 starts "The fog had
  been clinging to the coast of Blackwood Point for three days...". Both fluent; the 4-bit model is a
  different model on open-ended text. (The tool's PASS line there only means the run itself was consistent;
  it had no FP8 run to compare against, so the comparison above was done on the two JSON files.)

Verdict for task B: MXFP4 keeps confident choices (license passage 255/255, code answer word for word) and
costs 6.2 percent in mean NLL on easy text (calc:73ed24e2), but it changes free writing from the first word. Switching to it
trades some quality for speed.
