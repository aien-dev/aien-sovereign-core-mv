# Qwen3.8-27B decode A/B on Omega: FP8 lm_head vs bf16, and MXFP4 weights (2026-10-19)

Branches: sc qwen38/w4-decode (on qwen38/decode-speed b6c61c54), omega qwen38/w4-gemv (on qwen38/decode-speed 0010f7d3).
Model ~/models/qwen3.8-27b-fp8-017b9c7a. Every chip run went through bin/one.sh (flock /tmp/aien-gb10.lock, nice,
ground.txt with binary sha + MemAvailable + live server pid). Logits files (*.logits) and test binaries are not kept
here; their sha256 are in binaries.sha256 and test-binaries.sha256.

Layout: runs/speedA-* and speedB-* (whole-model speed rounds), runs/dumpA/B/C/D-* (teacher-forced logits dumps),
runs/genA/B/C-* (256-token free runs), runs/cmpA, runs/cmpB (qwen38_ab compare JSON), runs/tl-* (GPU-stamped
timelines), runs/w4err (host scale-rule survey), runs/text (decoded texts), runs/failed-* (refused or failed runs kept).
Variants: h0 bf16 head, h1 FP8 head, F FP8 weights + FP8 head, W MXFP4 weights + FP8 head, WW MXFP4 weights + MXFP4 head,
Wb/WWb the same with converter scale rule BEST.

Results and receipts: see ~/handoffs/2026-10-19-ab-fp8head-4bit.md (final table copied here when the last runs land).
