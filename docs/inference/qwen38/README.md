# Qwen3.8-27B in 4 bits by default (ledger L125, SPEED-1)

## What this folder holds

- `w4-fit-gate.json`: the pass marks for the 4-bit accuracy check, frozen before any measurement.
- Results of the L125 measurement window are added under `evidence/` with their receipts.

## The idea in plain words

Writing one word means reading every weight of the model from memory once. Memory moves at most 273 GB per second,
so fewer bytes per weight means more words per second. The 8-bit model reads 26.89 GB per word; the 4-bit model reads
13.65 GB. The earlier A/B work measured 13.82 words per second for 4-bit against 8.29 for 8-bit.

Until now the 4-bit copy was made at every start (93 to 161 s) and the 8-bit copy stayed loaded beside it, so the
model held about 44 GB. L125 changes three things:

1. **Convert once and store.** `qwen38_w4_store write <checkpoint-dir>` converts every projection and the word-scoring
   head to 4 bits and writes one file, `<checkpoint-dir>.mxfp4/aien-w4.store`, beside the checkpoint (never inside
   it). The file is read back and checked before it gets its final name.
2. **4-bit only.** When that file exists, the engine (`Qwen38Omega::open`) loads only the 4-bit tensors: no 8-bit copy,
   no 16-bit head. `AIEN_Q38_W4_STORE=off` goes back to 8 bits; `AIEN_Q38_W4_STORE=<path>` picks another file.
3. **A check made for 4 bits.** The strict check (logits within 0.9999 of the reference) fails every reduced setup, so
   it cannot decide whether 4 bits is good enough to serve. `qwen38_ab gate4` applies the frozen pass marks: same next
   word on at least a set share of six texts, how far the word probabilities moved (KL), how much the text score moved
   (NLL), and no confident disagreement beyond a limit.

## What can never load by mistake

The store records a fingerprint of the checkpoint (config, index, every shard's size and header), a check value per
tensor (verified on every read), and the converter's answer on a fixed test block. A store from another checkpoint, a
damaged store, or a store from a converter that rounds differently is refused with an error. There is no silent
fallback to 8 bits.

## Known limit

With no 8-bit copy, prompts are read one token at a time through the decode path. That is exact but slow for long
prompts (a 2,047-token prompt takes about 148 s). A 4-bit prompt routine is the next piece of work before this is the
live server's default. The live server on port 18091 runs another engine path and is not changed by this work.
