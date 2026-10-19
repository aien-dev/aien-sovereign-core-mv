# Qwen3.8-27B serving evidence (sc #642, 2026-10-19)

- session-8layers.json: chip resume check, 8 layers, FAIL (stale-state control used a cosine bar that 8 layers
  cannot trip: stale vs cold 0.99980). Kept as the red run.
- session-8layers-run2.json: same check with the control on greedy ids: PASS (resumed == cold, max_abs 0.0;
  live == cold; other question differs; stale state caught).
- gate-run1, gate-run2: VOID (every request 503: run 1 the server polled the GPU lock and lost to other jobs'
  flock queue; run 2 another client held the server). Fixed: queued lock (GpuLock::acquire_queued_at), gate retries busy.
- gate-run3: FAIL on my own check (expected Snapshot reuse; the server used the cheaper Live reuse). Ids all equal.
- gate-run4: GATE PASS through the live server on 127.0.0.1:18091 (binary sha b0747cc1, start-27b.sh):
  golden 24/24 ids equal the host reference (gen-chat-max-24.json), repeat equal, control (smaller) differs,
  turn 2b Snapshot(17) == cold, turn 2 Live(47) == cold, turn 3 Live(93) == cold, cold runs logged Cold.
- tools/qwen38_serve_gate.sh reproduces it: `tools/qwen38_serve_gate.sh http://127.0.0.1:18091 <golden.json> <out-dir>`.
- speed-run1: VOID/stopped. The 30k prompt was refused by the tokenizer ("Sequence length 29948 exceeds maximum
  context length 2048", fixed in 4c56abc4) and the script retried the refusal until its runner's time limit.
  Decode rows from it (161.7 / 165.2 / 169.1 ms/token graph chip) were on the earlier binary.
- gate-run5: GATE PASS again on the final binary (sha c8404006, 4c56abc4).
- speed-run2 (final binary, live server, other agents' chip jobs queued on the same GPU lock: lock waits up to 201 s):
  decode 30 in / 256 out greedy x3: graph chip 160.3 / 158.6 / 158.9 ms/token, wall 161.6 / 159.9 / 160.2 ms/token;
  request without lock wait 41.7 / 41.0 / 41.1 s. Median 158.9 ms = 6.29 tok/s (calc:b161649a97e28add384c2d499bab42916862a3254feeaa2b1cad91720e4127a0) = 0.620 of the
  10.15 tok/s ceiling (calc:69ebef69f1eb91913e3870c6ba3ce54c9aee67722e9be39b264300080d3e6fb9); wall 160.2 ms = 0.615 (calc:462f1f37398829d5de1bcd014856af2627da410a7653b824cd49c6bdee028756); whole request 6.23 tok/s (calc:2051089ea27b2c4eb5a3597d234269104e3f297b7d2c5cc2e0f10bc91abc1381).
  TTFT cold 1681 tok: prefill 8.78 s (client 50.6 s incl. 41.8 s lock wait); floor ~0.40 s (calc:c2a6decadef4a2441b95b15d1fbcf5cfca88f2fcb1a8ba30547d1e11cc21b569) -> 4.6%
  (calc:3390b07ed4c12983ae3d840f739d03bf4ee2393f707a272082ddc6b92fdbcb52). TTFT cold 29948 tok: prefill 162.2 s (client 168.1 s, lock 5.9 s); floor ~8.86 s (calc:0e6b020ae4e5e056d0809b326acf3ec7febb9c9c1d0780fb1f0980e9e0b392db)
  -> 5.5% (calc:ccb3259b6401e1483cda21f518161e3429c1897c0fa8aac2c44e79219fbb2de1). Next turn on 29973 tok, Live(29948) reuse, 25 fed: prefill 0.71 s (client 7.69 s incl.
  6.95 s lock wait).
- stream-thinking-on.sse: one streamed request through the live 27B with thinking on (effort low, 12 tokens):
  15 SSE events, the thinking arrives as reasoning_content deltas, then [DONE].
- TTFT figures are the server's prefill time for a max_tokens 1 request (prompt fed, first token sampled), not a
  streamed first-token stamp; client wall is listed beside them.
- keepalive.sh: restarts the 27B through start-27b.sh if 18091 is dead (at most 3 per hour).
