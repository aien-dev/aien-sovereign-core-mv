# W-L1I plan (pre-registered 2026-10-14, before any variant number was seen)
Question: after W8A16, what makes the whole-prompt path (ON16) drift from plain decode (OFF), and each source's share.
Candidates: (1) bf16 GEMM inputs (A16, 4 points), (2) bf16 Q in prefill attention, (3) bf16 P, (4) online-softmax order,
(5) GEMM vs GEMV summation order / RMSNorm / router / combine = the residual ON16 - FULL.
Tool: sc own30b/a16b own30b_drift_split (variants in its header). Base: sc rc2 d26b3c6 + this branch; omega rc2 848a999b archive 1baabd41 (rebuilt bit-identical).
Chip now: 3-layer view of the real checkpoint (/tmp/wl1i/ckpt3: config num_hidden_layers 3, shards symlinked), lens 512, 2048.
Full model: prepared script only (pause window).
Rules (fixed now):
 R1 floor: layer 0 x_out relerr(ULP,OFF), relerr(HATT32,OFF) <= 0.1 x relerr(ON16,OFF).
 R2 teeth: MUTQS (score scale x1.02) attention-op error vs chip >= 10 x HQP's at every layer.
 R3 repro: layer 0 x_out relerr(ON16,FULL) <= 0.1 x relerr(ON16,OFF).
 Shares: layer 0 x_out, e_X^2 / sum over {A16,HQ,HP,HATT32}; additivity check e_FULL^2 vs sum (within 30%).
 Attention-op shares (all layers, identical inputs): HostAttn chip_relerr per variant.
 Depth: layer where ULP reaches half of ON16 = saturation (full model only meaningful).
 Decision: if HQ+HP share > 50% of layer-0 drift (or attention-op error dominates the A16 per-op error), build f32 Q/P option.
