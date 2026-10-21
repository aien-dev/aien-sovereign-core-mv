# Small Allen student on the host reference (L389, 2026-10-21)

Student: Qwen/Qwen3.5-0.8B @ 2fc06364715b967f1860aea9cf38778875588b17 (Apache-2.0), local copy
`~/models/qwen3.5-0.8b-2fc06364`, LFS sha256 verified (`~/models/qwen3.5-0.8b-2fc06364.sha256`).
Tokenizer: `tokenizer.json` sha256 5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42, identical to the
teacher Qwen3.6-35B-A3B-FP8 @ 95a723d0 (vocab 248320), so the 35B can check the student token by token.

Reference changes (crates/aien-inference-abi/src/qwen38.rs): tied embeddings read the embedding table as the lm_head;
F32 vectors (A_log, linear_attn.norm.weight in the 0.8B) read exactly.

Receipts (host only, no chip, `target/release/qwen38_check`, Spark CPU):
| file | what | result |
|---|---|---|
| host-score-license.json | Apache LICENSE first 2500 bytes, 256 tokens | perplexity 1.319, next-token acc 0.933 |
| host-score-license-shuffled.json | negative control: same words shuffled | perplexity 442.634, acc 0.125 |
| host-score-readme.json | model README bytes 1000..3000, 256 tokens | perplexity 21.108, acc 0.439 |
| host-gen-hey.json | greedy, "hey allen how are you doing tonight", 24 tokens | "Hey everyone! I'm doing well, thanks for asking. How about you?" |

## Trainer parity (pre-registered 2026-10-21 before the first run)

`allen_train parity` runs the same 0.8B weights through the host reference (`Qwen38Reference`)
and through the trainer's own forward (`aien-train-allen`, whole-sequence, the one the backward
differentiates) on the first 64 tokens of `license-2500.txt`. Pass means all three hold:
max |logit difference| <= 0.02; no argmax change at any position where the reference's top-2 gap
is above 0.02; |mean NLL difference| <= 1e-3. Receipt: `trainer-parity-license64.json`.

### Result

`trainer-parity-license64.json`: **pass**. Max |logit diff| 7.5e-5 over 64 x 248,320 logits, no
argmax change, mean NLL 0.8151688 (trainer) vs 0.8151693 (reference).

## Trainer backward (L389 step 5)

`crates/aien-train-allen`: LoRA (rank 16, alpha 32) on every large linear map of the
Qwen3.5-0.8B student (DeltaNet in_proj_qkv/in_proj_z/out_proj, attention q/k/v/o, MLP
gate/up/down); the base weights, norms, conv, A_log, dt_bias and the tied embedding stay frozen.
The backward is written by hand per op: zero-centred RMSNorm, SwiGLU, gated attention (softmax,
GQA, partial RoPE, q/k norm, sigmoid output gate), Gated DeltaNet (causal conv + SiLU, l2 norm,
the delta-rule recurrence run back in time from recomputed states, the gated RMSNorm), the tied
head and weighted cross-entropy.

- `trainer-gradcheck.txt`: on a tiny random model (4 layers: 2 DeltaNet with 2 key / 4 value
  heads, 2 attention with GQA) every one of the 52 adapter tensors' directional derivative equals
  the central finite difference (eps 1e-2, pass |a - fd| <= 2e-3 + 2e-2 max(|a|, |fd|)); large
  gradients agree to about 1e-4 relative. Negative controls: mutant 1 (DeltaNet key gradient
  without its `S' dkv` term) fails 18 tensors, mutant 2 (RoPE turned forward in the attention
  backward) fails 38. The few small attention q/k values sit inside the f32 finite-difference
  scatter (see the eps 3e-2 and 3e-3 rows: fd moves, the analytic value does not).
- The step-by-step decoder (`decode.rs`, used for the evaluation) equals the whole-sequence
  forward within 1e-4 per logit (test `step_by_step_decode_equals_the_whole_sequence_forward`).

## Trainer speed (host CPU, `trainer-host.txt`, nice 19, 12 threads, GB10 untouched)

`trainer-gemmbench.jsonl`: our packed NEON f32 GEMM, 72 GFLOP/s on one core and 430-500 GFLOP/s
on 12 threads (256 x 1024 x 3584: 2*256*1024*3584 = 1.879 GFLOP in 3.79 ms = 496 GFLOP/s).

`trainer-bench.jsonl`, one training sequence (forward, backward, adapter gradients):
256 tokens in 3.614 s = 70.8 tokens/s; 100 tokens in 1.873 s = 53.4 tokens/s. Counted work
(linear maps forward + input-gradient backward, 4 flops per weight per token, plus the head on
the 128 loss rows): 4 * 0.497e9 * 256 + 4 * 254.3e6 * 128 = 639.7 GFLOP, 177 GFLOP/s effective;
the rest of the time is the DeltaNet recurrence and its backward (about a sixth in `perf`).
