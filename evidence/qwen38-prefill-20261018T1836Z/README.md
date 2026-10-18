# Qwen3.8-27B whole-prompt prefill on Omega (2026-10-18)

Branch qwen38/prefill-ttft. Binary: `qwen38_prefill` (crates/aien-inference-abi/src/bin/qwen38_prefill.rs), archive
omega qwen38/prefill-graph 6e24f7fd (sha 373d122c...). Model ~/models/qwen3.8-27b-fp8-017b9c7a.

| Test | Command | Result |
| --- | --- | --- |
| golden (24 tokens, prefill chunk 8 + decode) | command-golden.txt | tokens 24/24 equal to host reference; logits cosine 0.9929 < 0.999 bar, so pass:false (bar not moved) |
| score (LICENSE 256, chunk 64) | command-score.txt | PASS: mean NLL 0.08772 vs ref 0.08707, accuracy 0.9882 = ref, control 21.25 |
| cold TTFT 32k/64k/128k (chunk 4096) | command-ttft.txt | 32k 197.702 s, 64k 415.255 s, 128k 936.728 s (ttft-chunk4096.json). 128k split: gemm chip 304.7 s, attn 318.5 s, host 109.4 s, gdn 73.1 s, a/b 25.3 s |

Ceiling at 131,072 tokens: about 66 s (GEMMs 30.73 s at 207.7 TFLOPS FP8, calc:fb84329b; causal attention 35.18 s at
our measured 96 TFLOPS bf16, calc:2632671b). Refused runs kept: run1 attn alloc (max_pos not a multiple of 256),
run2 quant guard (Rust limits behind C), ttft memory guard refusals while CI used memory.
