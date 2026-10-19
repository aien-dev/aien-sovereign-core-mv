# qwen38/serve-fast: faster decode in the live 27B server (2026-10-19)

Binary: aien-serve-http from this branch (46134247), sha256 df9360112fdc815539f3e289fccfd8070845129ef4ccb466bb90c729ec71d0e5,
linked to omega qwen38/decode-speed 0010f7d3 libomega_gpu.a sha256 bd6802a5d493ed6b4589dabb5787db70a37ba10c75d8406f3cce27c4dfa9f9bd.
Live on 127.0.0.1:18091 since 2026-10-19T14:00:47Z (start-27b.sh.new = the installed start script).

- No second server: MemAvailable 45 GiB against a 37.6 GiB need (old VmHWM 40.36 GB) at decision time, under the 12 GiB rule. Swap path.
- swap.log: keepalive stopped, SIGTERM 13:55:26Z, old exited 13:59:43Z (257 s drain of an aien-translate reply), new ready 14:00:47Z (load 64 s).
- gate-live/, gate-live.out: tools/qwen38_serve_gate.sh against the new live server: GATE PASS (golden 24/24, control differs, snapshot/live/cold resume equal).
- after/: 4 runs of the serve agent's short prompt, 256 greedy tokens: chip 129.9 / 129.0 / 134.9 / 132.2 ms/token, all four id lists identical, text identical to the old binary's speed-run2/decode1.json.
- old-live-log-decodes.tsv / new-live-log-decodes.tsv: every decode of 100+ tokens from the server log (mostly aien-translate). Old binary 254 replies, chip median 159.8 ms/token; new 22 replies, median 133.0.
- before/: my same-prompt runs on the old binary never got a turn (aien-translate kept the server busy; one busy reply kept). Old same-prompt figures come from speed-run2 (160.3 / 158.6 / 158.9).
