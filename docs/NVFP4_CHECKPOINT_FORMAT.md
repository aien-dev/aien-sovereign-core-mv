# NVFP4 checkpoint format (Qwen3.6-35B-A3B, DRA-71)

Source: `nvidia/Qwen3.6-35B-A3B-NVFP4` at revision `1355db6a052410cfd62085d94b58866fd0f2c3c5`. Local copy:
`~/models/qwen3.6-35b-a3b-nvfp4-1355db6a/` (3 safetensors shards, index `total_size` 23407580856, 124468 tensors,
header metadata `{"format":"pt"}`). `hf_quant_config.json`: producer modelopt 0.44.0, `quant_algo` MIXED_PRECISION,
`kv_cache_quant_algo` FP8, one `quantized_layers` entry per layer. Everything below was read from the shard headers;
the family table is in the DRA-71 evidence (`nvfp4-tensor-families.txt`).

## Tensor families

| family | count | tensors (dtype, shape) |
|---|---|---|
| routed experts `model.language_model.layers.L.mlp.experts.E.{gate,up}_proj` | 10240 each | `.weight` U8 [512,1024], `.weight_scale` F8_E4M3 [512,128], `.weight_scale_2` F32 [], `.input_scale` F32 [] |
| routed experts `...experts.E.down_proj` | 10240 | `.weight` U8 [2048,256], `.weight_scale` F8_E4M3 [2048,32], `.weight_scale_2` F32 [], `.input_scale` F32 [] |
| `...mlp.shared_expert.{gate,up,down}_proj` | 40 each | same NVFP4 layout as the routed experts |
| `lm_head` | 1 | `.weight` U8 [248320,1024], `.weight_scale` F8_E4M3 [248320,128], `.weight_scale_2` F32 [], `.input_scale` F32 [] |
| `linear_attn.{in_proj_qkv [8192,2048], in_proj_z [4096,2048], out_proj [2048,4096]}` | per DeltaNet layer | F8_E4M3 weight, per-tensor F32 `weight_scale` and `input_scale` (W8A8, static activation scale) |
| `self_attn.{q_proj [8192,2048], k_proj/v_proj [512,2048], o_proj [2048,4096]}` | per attention layer | F8_E4M3, per-tensor F32 `weight_scale` and `input_scale` |
| BF16 | — | `in_proj_a/b`, `conv1d`, `A_log`, `dt_bias`, norms, `mlp.gate` [256,2048], `shared_expert_gate`, `embed_tokens`, `visual.*`, all of MTP (fused 3-D experts `gate_up_proj` [256,1024,2048], `down_proj` [256,2048,512]) |

`weight_scale_2` is per expert per projection: gate and up of one expert carry different globals, so a fused gate|up
kernel needs a per-row global (4 B per row).

## Dequantisation

    W[r][c] = e2m1(nibble(r, c)) * e4m3(weight_scale[r][c / 16]) * weight_scale_2

- byte `c / 2` of row `r` holds columns `c` (low nibble, even column) and `c + 1` (high nibble).
- e2m1 is sign-magnitude: bit 3 sign, bits 2..1 exponent, bit 0 mantissa: magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6.
- one e4m3 scale per 16 consecutive columns of a row; `weight_scale_2` is an f32 scalar.

The nibble order was established on real tensors, not assumed: with low = even column the dequantised experts sit at
rel-RMS 0.088-0.099 from the FP8 checkpoint's same matrices; with the nibbles swapped they sit at 1.41-1.42
(`real_experts_match_host_matmul_and_the_fp8_checkpoint`, 9 matrices).

Reference implementation: `crates/aien-inference-abi/src/nvfp4.rs` (`Nvfp4Matrix`, `Nvfp4Expert`).

## Bytes model (Wolfram, stateless evaluator)

Per matrix `r x k`: `r*k/2` code bytes + `r*k/16` scale bytes + 4 global bytes.

- `w4Mat[512,2048]*2 + w4Mat[2048,512] -> 1769484` B per expert (FP8 with block-128 bf16 scales: 3146112); with a
  per-row global `+ 4*(512+512+2048) -> 1781772`.
- bytes per weight `1769484/(3*512*2048) -> 0.5625038`; ratio to FP8 `0.56243516`.
- per token, 9 experts x 40 layers: FP8 `1.13260032` GB -> NVFP4 `0.63701424` GB; at 181 GB/s the N=1 saving is
  `2.738` ms.
- per 8-head stroke at U = 55 distinct experts per layer: `6.9214 -> 3.8929` GB, `-16.73` ms at 181 GB/s; at the
  measured U(8) = 55.6: `6.99695 -> 3.93533` GB, `-16.92` ms.
- lm_head: FP8 `0.50856` GB, BF16 `1.01712` GB, NVFP4 `0.286065` GB (the reader measured 286064644 B); saving vs FP8
  `1.229` ms at 181 GB/s.
- step model `t(N) = 9.13 + 3.29 N + c*U(N) + 4.81 ceil(N/4)` ms with the expert term `c` at 181 GB/s:
  FP8 `{42.56, 62.77, 82.53, 95.50}` tok/s at N = 1, 2, 4, 8; NVFP4 experts `{48.19, 74.35, 102.26, 119.70}`;
  zero expert time (ceiling) `58.04` at N = 1 and `177.50` at N = 8.

## Corrections to the DRA-71 text

- "~119 tok/s at one conversation" is not reachable under this step model: the N = 1 ceiling with free experts is
  58.04 tok/s; 119.70 tok/s is the N = 8 aggregate with NVFP4 experts alone.
- "~267 at eight" is above the 177.50 tok/s zero-expert ceiling; it needs levers on the fixed terms that the text
  does not name.
- "6.92 -> 3.89 GB, -16.7 ms" holds at U = 55; at the measured U(8) = 55.6 it is 7.00 -> 3.94 GB, -16.9 ms.
- "-2.7 ms at N = 1" holds (2.738 ms).

## Open format questions

- attention and DeltaNet projections are per-tensor FP8 W8A8 with a static `input_scale`, not block-128 like the FP8
  checkpoint the engine reads today: either load them from the FP8 checkpoint (hybrid) or map them to the block form.
- MTP stays BF16 (fused 3-D); the existing FP8 MTP path does not read it.
