# OWN-30B whole-prompt prefill: router logits + MoE combine on the chip (W-L1G-E, 2026-10-14)

All results here are HARDWARE (GB10), one process per tool, under /tmp/aien-gb10.lock. Exact commits, binaries, archive and commands are in ttft.start and gate.start.

Change: the whole-prompt ON path can run its router logits (omega_gpu_pf_router_logits) and MoE weighted combine (omega_gpu_pf_moe_combine) on the chip, and its routing plan token-parallel on the host (plan_par). Switch: engine.set_whole_prompt_device_host(true), OFF by default. Weights read in place (no cached copy).

TTFT (ttft.raw, 32 new tokens, reps 2; rep 0 of ON(host) at 512 includes first-use warm-up):
- 512: OFF 7.6-7.7 s; ON host routing 0.62 s (rep 1); ON device routing 0.31 s.
- 2048: OFF 31.4-37.4 s; ON host routing 2.08 s; ON device routing 0.84-0.86 s.
- 2048 device-path split: chip GEMM+quant+router+combine 621 ms, attention 57 ms, host 144-164 ms (was 1.27 s).
Previous baseline (sc 4e5cfde receipts): ON in place 512 0.62-0.79 s, 2048 2.0-2.1 s.

Agreement: device path vs host path, logits0 bit-identical (0 bits differ) and tokens identical in every rep; traced runs: 0 expert flips of 196608 (512) and 786432 (2048) slots over 48 layers, 0 residual-stream bits differ.
OFF vs ON tokens: 2048 identical; 512 differ at position 7, unchanged from 4e5cfde (activation rounding, gate 1890a38).

Gate (gate.raw, own30b_wp_gate --devhost): ON on the device path, PASS at 512 (0.92 x B) and 2048 (0.98 x B), layer-0 ops within budget; mutants MUT_ROUTING (9.1x / 7.8x B) and MUT_GUSCALES (6.8x / 4.7x B) CAUGHT. SUMMARY PASS.
