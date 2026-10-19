# Flash-Next M1: CPU reference forward pass on the real NVFP4 weights (2026-10-19)

What this is: the Rust reference (`crates/aien-inference-abi/src/flashnext.rs`, driver `flashnext_check golden`) run end to
end on the downloaded `nvidia/Qwen3.8-Flash-Next-NVFP4 @ fc694b54` checkpoint (all 48 layers, real tensors read straight from
the safetensors files, experts dequantised from NVFP4 per `hf_quant_config.json`: e2m1 nibble x e4m3 group scale (16) x
weight_scale_2). Nothing is converted. The n-gram table rows are read by byte range from the FP8 shards (one 160 byte row per
lookup, 16 per token). Weights are not in this folder and must never be committed; `checkpoint-shards.sha256` names them.

Status labels: MEASURED = in the files here. NOT MEASURED = said so.

## Result (MEASURED, files golden1.json, golden2.json, *.stdout.log)

| | prompt 1 (capitals) | prompt 2 (moon landing) |
|---|---|---|
| tokens | 19 | 38 |
| all logits and lane values finite | yes | yes |
| teacher-forced mean NLL (nats / perplexity) | 1.050 / 2.86 | 1.097 / 2.99 |
| next token is the model's top-1 | 16 of 18 | 29 of 37 |
| whole prompt in one call vs one token at a time: max abs logit difference | 0.0 | 0.0 |
| argmax equal at every position | 19 of 19 | 38 of 38 |
| greedy continuation | ` Rome. The capital of Spain is Madrid. The capital of` | ` But what if he had said, "one small step for a man, one` |
| wall (whole / one at a time / 12 or 16 greedy tokens) | 73 s / 124 s / 85 s | 129 s / 285 s / 109 s |

Plain reading: the model completes "The capital of Italy is" with " Rome" (logprob about 0 at that position), carries on with the
pattern, and on the moon text it knows the quotation is "a man". Per-token loss is about 1 nat on repetitive and quoted text.
A broken port (wrong lane mixing, wrong RoPE, wrong DeltaNet gate) normally gives garbage tokens and a loss of many nats per
token; this is not that. The few large losses are where the text is surprising (prompt 1: the bare first word, 9.4 nats;
prompt 2: the comma after "1969" 7.7 nats, "the" after "walk on" 6.0 nats). This is a plausibility argument, not a proof.

The "one token at a time" pass is a real state test: it recomputes the same tokens through the recurrent Gated DeltaNet state,
the 4-token conv state, the indexed-attention KV and indexer cache, and the n-gram history, and must equal the whole-prompt
pass. It matches to the last bit (0.0), because every op is per token in the same summation order.

## What is NOT shown here
- No independent numeric reference exists on this machine: transformers has not run the 125B model here, and the bf16
  checkpoint (360 GB) was not downloaded. Correctness rests on the spec read (modular_qwen4_exp.py), the unit tests (13 host
  tests in the crate), and the sanity above. A byte-exact comparison against transformers is NOT MEASURED.
- The indexer only changes the mask past about 2051 tokens (top 512 blocks of 4 plus the tail); both prompts are shorter, so the
  selection is "everything visible". The long-context indexer path is covered by unit tests on synthetic keys only.
- Quantised NVFP4 weights are the model here. Drift against the bf16 or FP8 weights is NOT MEASURED on the whole model (per-expert
  weight error against bf16 was measured earlier: evidence/flashnext-m1-20261019T1649Z).
- Timing is CPU, shared box, loadavg 30 to 50 (loadavg_start and loadavg_end in each json). Not a speed claim.

## Files
- `golden1.json`, `golden2.json`: per position input, top-5 (id, logit, text), logprob of the given next token, top-1 hit; the MoE
  routing (10 experts, weights, shared gate) of every layer and token of the whole-prompt pass; indexer selection counts; lane RMS
  entering every layer; timings; loadavg.
- `derived-sha256.txt`: sha256 of the full last-position logits (248320 f32) and mixer output of each run. The files stay in
  `~/flashnext-scratch/runs/golden*/` (derived from the weights, not committed).
- `binary.sha256`, `sc-commit.txt`, `commands.txt`, `checkpoint.SOURCE`, `checkpoint-shards.sha256`,
  `checkpoint-small-files.sha256`: provenance.
