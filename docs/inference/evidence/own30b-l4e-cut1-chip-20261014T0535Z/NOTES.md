# W-L4e cut 1: a numerics mutant the token check can see (small weights, GB10, 3 real layers)

HARDWARE, synthetic (first 3 layers of the real checkpoint; tokens are not the model's, the numerics are).
Code: sc own30b/spec-wp (this commit's parent tree), omega archive 8d5a42e8 (own30b/spec-exact). Script: run-cut1.sh.

Probe (r2 probe.raw, m = 1..5 rows, same KV position, logits of each verify row vs plain step):
- exact verify: bit-identical, logit change 0 (instrument baseline).
- attn-bf16 (W-L4d mutant): max logit change 0.036, no argmax flips.
- residual-bf16: 0.067 to 0.10.
- act-fp8 (W8A8 activations into every verify GEMM, the prefill quantiser): 0.479, about 13x attn-bf16
  (reference top-1/top-2 margin 0.248 on this prompt).

Token check (own30b_spec_check --layers 3, suffix drafter, 4 golden + code/JSON, --verify exact):
- attn-bf16 caught on 1 of 5 prompts; residual-bf16 1 of 5; act-fp8 5 of 5. Honest exact verify identical on every run.

Why act-fp8 is the gate's numerics control: it is the rounding class the whole-prompt path really has, and the
whole-prompt verify (same numerics) diverged from spec-off on 6 of 10 full-model runs (W-L4c cut 2 receipt), so
the full-model token check is known to see it. UNKNOWN until the full-model gate runs: its catch count there.
