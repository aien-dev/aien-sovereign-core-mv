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
