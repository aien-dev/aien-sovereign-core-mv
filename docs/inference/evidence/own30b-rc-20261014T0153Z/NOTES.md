# OWN-30B release candidate (W-RC, 2026-10-14): chip gates

All results HARDWARE (GB10), one 30 GB model per process, each under /tmp/aien-gb10.lock, AIEN_OMEGA_SPIN_US=2000.
Exact commits, binary sha256 and commands: `*.start`. Raw output `*.raw` / `*.err`, exit code `*.rc`.

Build: sc own30b/rc (merge b449580 + server 4ddc4cd + check tool 37b696f, gate-short tool cdccd5a), omega own30b/rc d0889a94
(l1g-e-int 7e43dd8f + decode-mega 7d47aa87), built with AIEN_OMEGA_DIR + AIEN_REQUIRE_NATIVE_LIBS=1 (no prebuilt
archive); ABI manifest, layout probe and symbols pass; archive sha256 6a6cb4df... (not yet in native-artifacts.lock).
Flags of the token runs = the server's paths (ready log): --weights fp8 --attn fused --router device --lm-head gemv --prefill auto.

| run | what | result |
|---|---|---|
| check-full | own30b_check, whole-prompt OFF, parity + all negative controls | parity PASS (layers 0/23/47), 10 controls CAUGHT, tokens IDENTICAL x4, SUMMARY PASS |
| check-wp | own30b_check --tokens-only --whole-prompt --prefill-device-host (the RC server path) | p2, p3, p4 IDENTICAL; p1 ("Hello", 1 token) DIVERGES at 0 (cpu margin 0.40): tool says FAIL, explained by gate-short len 1 |
| ttft | own30b_ttft --devhost, 32 new tokens, 2 reps | 512: OFF 7.55-7.65 s, ON(dev) 0.31-0.32 s; 2048: OFF 31.9-32.1 s, ON(dev) 0.87-0.88 s; dev vs host routing bit-identical |
| gate | own30b_wp_gate --devhost --mutants, 512 + 2048 | PASS (0.92 B, 0.98 B), both mutants CAUGHT at both lengths |
| gate-short | same gate at 1, 2, 20 tokens (48 refused by the tool's own preamble assert, rc 134; no chip fault) | PASS at 1 (0.01 B), 2 (0.51 B), 20 (0.57 B); mutants CAUGHT |

p1 explained (gate-short, len 1): first-step top-1 OFF 323 (= golden) vs W8A8 control 25 vs ON 25; logits cosine ON-CTRL 0.99995,
ON-OFF 0.891, CTRL-OFF 0.895. The OFF kernels with only the ON path's FP8 activation quantiser make the same flip, so the
difference is activation rounding, not a whole-prompt bug.

Decode tok/s (16 tokens, p1..p4): OFF 39.1 / 36.1 / 37.6 / 37.0; RC path 38.4 / 35.9 / 37.3 / 36.7 (decode is the same code; one-list decode OFF).

Server smoke (server-smoke/, binary /tmp/wrc/bin/aien-serve-http): 18092 is held by spark-mail (bind failed after a clean load,
server-18092-portclash.log), so the smoke ran on 127.0.0.1:18093. Ready log: whole_prompt=on prefill_device_host=on, fused / device
router / gemv lm_head / gemm prefill. /v1/models lists qwen3-30b-a3b-instruct-2507 owned_by aien/omega. Odysseus-style requests
(session_id, cache_prompt, think:false extras): turn 1 non-stream "The capital of France is Paris." (41 prompt tok, prefill 761 ms
incl. first-use, 35.5 tok/s); turn 2 SSE with prefix reuse (48 of 62 reused, prefill 110 ms) "The capital of Italy is Rome." + [DONE];
1836-token prompt answered "Block 60" correctly, prefill 797 ms, 35.1 tok/s. Server stopped by PID (SIGTERM).
