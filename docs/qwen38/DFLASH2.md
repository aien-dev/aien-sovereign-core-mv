# DFlash 2 draft model for Qwen3.8-27B: architecture as built in AIEN

Status: written 2026-10-19 on branch cascade/dflash-drafter. Plain English first, exact maths after.

## What it is

A small model (5 layers, 1.92 billion parameters, bf16, 3.85 GB) that guesses the next 7 words of the
27B model in ONE pass. It reads the 27B's own hidden states (five of its 64 layers), so it does not have to
understand the text from scratch. It has no word table and no word-scoring head of its own: it borrows the
27B's embedding table and lm_head. A small "selector" then walks through the top 16 guesses at every
position and picks one chain that hangs together. The 27B checks the chain in one verify pass, exactly as it
checks the MTP head's guesses. The answer does not change: every accepted word is the 27B's own greedy word.

Weights: outside weights (Drake 2026-10-19: outside model weights allowed, inference code all ours).
- Checkpoint: z-lab/Qwen3.8-27B-DFlash2, revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4, Apache-2.0.
- Where, and how it was checked: `qwen3.8-27b-dflash2-50307d4c.SOURCE` and `.sha256` in
  evidence/qwen38-dflash-weights/ (the weights themselves are never committed).
- Read for the maths (not copied, not run): the DFlash reference `dflash/model.py` of github.com/z-lab/dflash
  (MIT) at commit 07ebd93db9f472af339b644bb70221ad8428328a, and the model card / blog (inco.ai/blog/dflash2,
  2026-08-18). Everything in AIEN is written from this description in Rust and Omega.

## config.json (as shipped)

| key | value | meaning here |
|---|---|---|
| architectures | DFlash2DraftModel | |
| num_hidden_layers | 5 | drafter layers |
| hidden_size | 5120 | same as the 27B |
| num_attention_heads / num_key_value_heads / head_dim | 32 / 8 / 128 | grouped query attention, 4 query heads per KV head |
| intermediate_size | 17408 | SwiGLU MLP, same width as the 27B |
| rope_parameters | default, theta 10,000,000 | full rotary over all 128 dims (unlike the 27B's partial rotary) |
| rms_norm_eps | 1e-6 | Qwen3 RMSNorm: `x * rsqrt(mean(x^2) + eps) * w` (plain w, not 1 + w) |
| is_causal | false | the block attends both ways |
| layer_types | 5 x sliding_attention, sliding_window 2048 | a query at p sees keys k with abs(p - k) < 2048 |
| dflash_config.block_size | 8 | 1 anchor (the last real word) + 7 guesses |
| dflash_config.mask_token_id | 248070 | the 27B embedding row used for every guess slot |
| dflash_config.target_layer_ids | 5, 19, 33, 47, 61 | 27B layers whose OUTPUT (residual after the layer) is read |
| dflash_config.conv_kernel_size / conv_group_size | 2 / 16 | two-tap dynamic causal convolution, one dynamic weight per 16 channels |
| dflash_config.selector_rank / selector_top_k | 256 / 16 | |
| vocab_size | 248320 | the 27B's vocabulary |
| tie_word_embeddings | false | no head of its own (borrows the 27B's lm_head) |

## Tensors (model.safetensors, 81 tensors, all BF16, 1,924,404,480 parameters)

| name | shape | params |
|---|---|---|
| fc.weight | [5120, 25600] | 131.1 M |
| hidden_norm.weight | [5120] | |
| norm.weight | [5120] | |
| candidate_selector.hidden_projection.weight | [256, 5120] | 1.3 M |
| candidate_selector.predecessor_codebook | [248320, 256] | 63.6 M |
| candidate_selector.successor_codebook | [248320, 256] | 63.6 M |
| per layer i in 0..5 (15 tensors each): | | 332.9 M per layer |
| layers.i.input_layernorm.weight, post_attention_layernorm.weight | [5120] | |
| layers.i.self_attn.q_proj.weight | [4096, 5120] | 21.0 M |
| layers.i.self_attn.k_proj.weight, v_proj.weight | [1024, 5120] | 5.2 M each |
| layers.i.self_attn.o_proj.weight | [5120, 4096] | 21.0 M |
| layers.i.self_attn.q_norm.weight, k_norm.weight | [128] | |
| layers.i.mlp.gate_proj.weight, up_proj.weight | [17408, 5120] | 89.1 M each |
| layers.i.mlp.down_proj.weight | [5120, 17408] | 89.1 M |
| layers.i.attention_conv.base_kernel, mlp_conv.base_kernel | [2, 2, 5120] | |
| layers.i.attention_conv.kernel_projection.weight, mlp_conv.kernel_projection.weight | [1280, 5120] | 6.6 M each |

Bytes read per draft pass (bf16): layers 3.33 GB, fc 0.26 GB, selector codebooks only the rows used (16 x 7
successor rows, 7 predecessor rows), plus the 27B's lm_head (2.54 GB bf16) for 7 rows.

## The forward pass (one draft round)

Notation: the round starts at position `s`. `t` is the last real word (the 27B's own next word from the last
verify pass, not yet run through the 27B). The drafter keeps a KV cache of "context" rows for positions
`0 .. s-1`.

1. Context features. For every row the 27B committed since the last round (positions `c0 .. s-1`; after the
   prompt that is every prompt position), take the 27B's residual stream after layers 5, 19, 33, 47 and 61
   (layer index from 0; in Hugging Face terms `hidden_states[id + 1]`) and concatenate them: 25600 values.
   `ctx = hidden_norm(fc · concat)`, 5120 values per row (RMSNorm with plain weight).
2. Block input. `x[0] = embed27B(t)`, `x[1..8] = embed27B(248070)`, positions `s .. s+7`.
3. Five layers, each:
   - `a = input_layernorm(x)`
   - attention conv prepare: `dyn = kernel_projection(a)` viewed as `[2 stages][2 taps][320 groups]`;
     `a = conv(a, dyn[stage 0], base_kernel[0])` (see Conv below). Keep `dyn[stage 1]` for the finish.
   - attention: `q = q_norm(q_proj(a))` per head, rope at the block positions. Keys and values: the context
     rows `k_proj(ctx)`, `v_proj(ctx)` AND the block rows `k_proj(a)`, `v_proj(a)`; `k_norm` on all keys,
     rope at each key's position. Context keys/values go into the cache and stay; block keys/values are used
     for this pass only (the cache is cut back to `s` after the pass). Every block query attends to every
     cached context key and every block key inside the 2048 window (no causal mask), softmax scale
     1/sqrt(128), GQA 4 query heads per KV head. `o_proj`.
   - attention conv finish: `out = conv(attn_out, dyn[stage 1], base_kernel[1])`; `x = x + out`.
   - `a = post_attention_layernorm(x)`, mlp conv prepare (same as above with the mlp_conv tensors),
     `m = down(silu(gate(a)) * up(a))`, mlp conv finish, `x = x + m`.
4. `h = norm(x)`; rows 1..7 are the draft rows.
5. Scores: `logits[r] = lm_head27B · h[r]` (no final norm of the 27B, no multiplier, no soft cap).
6. Selector (greedy): for each draft row r = 1..7 take the 16 largest logits (values `u`, ids `c`).
   `p = hidden_projection · h[r]` (256 values). With `prev` = the word chosen at the row before (`t` for
   r = 1): `score_j = u_j + sum_i pred[prev][i] * p[i] * succ[c_j][i]`. The draft word is `c` at the largest
   score (first one on a tie, candidates in descending logit order, ascending id on equal logits). That
   word becomes `prev` for the next row.

Conv (grouped dynamic causal conv, 2 taps, groups of 16 channels). For a sequence `v[0..8]` of 5120-wide
rows, group `g = channel / 16`:
`out[p][c] = (((0 + base[0][c] * v[p][c]) + d[p][0][g] * v[p][c]) + base[1][c] * v[p-1][c]) + d[p][1][g] * v[p-1][c]`
with `v[-1] = 0`, every product rounded to f32 before its add (no fused multiply-add), in exactly this
left-to-right order (the reference accumulates `out = out + base*v; out = out + d*v` per tap).
`d[p][tap][g]` is element `tap * 320 + g` of the stage's half of the 1280-wide projection. The sequence is the block only (8 rows), never the context.
The dynamic weights `d` of the finish come from the prepare's input (the normed residual), not from the
attention or MLP output.

## The round loop (greedy, lossless)

Prompt: the 27B runs the prompt, its next word is `t`; every prompt row's five layer outputs feed the drafter's
context at positions `0 .. n-1`. Each round: draft 7 words as above, the 27B verifies `[t, d1 .. d7]` in one
pass at positions `s .. s+7`, accepts the longest prefix it agrees with (`a` words), and its own word after
them is the new `t`. The `a + 1` verified rows (t and the accepted drafts) are the next round's context rows
at `s .. s+a`. Rejected rows' DeltaNet state is rebuilt exactly as for the MTP head (restore + replay).
Words per pass = a + 1.

## What AIEN builds

- Rust CPU reference: `crates/aien-inference-abi/src/qwen38_dflash.rs` (host only, any size, its own tests).
- Omega chip kernel for the conv (`omega_gpu_dynconv2_f32`, omega branch cascade/dflash-drafter), parity
  bit for bit against the f32 host order above, with a mutant that must fail.
- The draft source on the verify path: `qwen38_omega::dflash` (27B on the chip, layer taps from the exact
  rows path, the drafter's matrices on the chip's bf16 GEMV, the lm_head on the chip).
