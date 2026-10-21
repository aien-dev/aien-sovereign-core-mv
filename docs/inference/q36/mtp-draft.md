# Qwen3.6-35B-A3B: the MTP layer as a draft for multi-token steps (L355)

Status: host acceptance measured (FP8 main model, teacher-forced on L313's f32 reference ids); chip work specified
below for L312 (multi-row verify) and L313 (k-token DeltaNet state). Words stay identical: every drafted token is
checked by the full model in the verify pass and only the matching prefix is kept (greedy), so the output equals plain
greedy decoding by construction.

## The head (checkpoint `mtp.*`, verified on host)

x = fc([norm_e(embed(next)); norm_h(h_prev)]) (fc [2048][4096] bf16, embedding half first), then one full-attention
layer (`mtp.layers.0.self_attn`, 16 q / 2 KV heads, FP8) and one MoE block (256 experts top-8 + the shared expert,
FP8), then `mtp.norm` and the model's lm_head. h_prev is the main model's residual before the final norm at the
position of the token being extended; the MTP attention cache holds every true position (the prompt included) at the
position of h. Host code: `Qwen38Reference::mtp_step` (qwen38.rs), tool `q36_w4_drift mtp`.

## Accepted tokens per verify pass (host, 512 steps each)

k = drafts chained per step (each draft fed to the next MTP step); tokens per pass = 1 + mean(min(accepted, k)).

| input | k=1 | k=2 | k=3 | k=4 | draft 1 accepted |
|---|---|---|---|---|---|
| p1 | 1.934 | 2.713 | 3.273 | 3.605 | 93.4 % |
| p2 | 1.979 | 2.910 | 3.752 | 4.391 | 97.9 % |
| p3 | 1.895 | 2.596 | 3.109 | 3.428 | 89.5 % |
| p4 | 1.877 | 2.592 | 3.107 | 3.471 | 87.7 % |
| chat-max-24 (chat traffic) | 1.705 | 2.074 | 2.201 | 2.293 | 70.5 % |

## Bytes: when a draft pays (stop rule from the plan review)

Bytes per token today 3.747 GB. Expert slot across 40 layers 125,829,120 B (3 x 512 x 2048 FP8 x 40). Expert union per
layer measured on the 35B (L312 strokes): 8 / 12.7 / 19.8 / 30.5 at 1 / 2 / 4 / 8 rows, used here as an upper bound for
draft rows (drafts of one conversation share experts at least as much as separate conversations).

- One conversation, verify 2 rows costs 1.158x one row (calc:efc2b11f), 4 rows 1.396x (calc:2365c918).
- Four conversations plus one draft each (8 rows) costs 1.257x the 4-row beat (calc:c18b6ce6); two plus one draft
  each 1.206x (calc:c9494aca). These two lines used the correlated union (the 30.5 at 8 rows came from one prompt
  family); different conversations route almost independently (54.7 at 8 rows, Cortex a00808e4).
- Eight different conversations plus one draft each (16 rows), measured from the host goldens' routes
  (evidence/l355-w4/mtp-union/, union-draft.sh: rows s and s+1 of p1-p4 + chat code/prose/chat/review): union 81.2 per
  layer against 54.7 for the 8 rows alone (set b with prose2: 80.7 / 54.5; four conversations 44.9-46.3 / 29.4-29.7; one
  conversation 12.0-12.4 / 8, close to the 12.7 above). The 16-row beat reads 1.347x the 8-row beat (calc:e26d8efc),
  1.460x with the MTP layer and the bf16 head read once for all drafts (calc:0898c6c2); at chat acceptance 1.705 that is
  1.168x more tokens per byte (calc:7b25561d). The draft still pays for Drake's eight conversations, with a smaller margin
  than the correlated lines suggested. Same v2 assumption (each weight read once per beat, lm_head batched); not measured
  on chip.
- The draft itself is not free: per drafted token the MTP layer reads about 73 MB plus the lm_head, 1.017 GB bf16.
  One conversation, k=1, p1 acceptance: 1.335x fewer bytes per emitted token with the bf16 head (calc:483f647d), 1.547x
  with a 4-bit draft head (calc:d91c918b; the draft only needs a top-1 guess, the verify pass keeps words exact; assumes
  FP8 acceptance, 4-bit head not measured).
  Chat traffic (1.705 at k=1): 1.177x with the bf16 head (calc:ad3a09fe), 1.365x with a 4-bit draft head (calc:4ea59c1e; same assumption, not measured).
  Deeper chains on chat lose with the bf16 head (each extra draft reads the 1 GB head for about 0.1 to 0.4 extra tokens): k=1
  is the setting for chat, k=2..4 only with the 4-bit draft head or on code-like text.
  At N conversations the draft head is read once per beat for all rows (batched), so its share falls as 1/N.

Rule: build the chip path because accepted tokens per pass (1.71 on chat, 1.88 to 1.98 on p1..p4 at k=1) exceed the byte ratio at
every measured N. The ratios assume the v2 engine reads each union expert once per beat (L312 v2); with v1 strokes
(each head re-reads its weights) a draft row costs about a full row, so k=1 reads about 2x the bytes for 1.93 tokens:
no gain. MTP rides on v2.

## What the chip needs

1. Verify pass = the multi-row beat with the draft rows as extra rows of the same conversation (L312 v2: union experts
   read once; positions pos..pos+k; attention causal inside the k rows).
2. DeltaNet state for k tokens with rollback (L313): run the k rows serially through the state inside one beat and
   keep the state after each row (or the chunked form), then commit the state of the last accepted row. KV for the
   attention layers: write all k, keep the length of the accepted prefix.
3. MTP head on chip: one attention layer (own KV cache, prompt positions included) + one MoE block (the same grouped
   expert kernel, FP8 or W4) + fc + norms + the lm_head (4-bit screen enough for a draft).
4. Gate: words identical to plain greedy on the goldens (the verify keeps them exact; a mutant that accepts one
   unverified draft must fail), tokens per pass on chip equal to the host numbers above, tok/s before/after.
