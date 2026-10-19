# Run 2 (2026-10-19, L068): QWEN38-SERVE FAIL not_ready

Memory was free this time (about 69 GiB), the gate took the chip lock after 21 minutes in the queue and started
its own server on port 18391. The server printed "loading Qwen3.8 onto the Omega engine" and then nothing for the
whole 900 s the gate allows, so the gate stopped it and reported `refusal "not_ready"` (FAIL, not a pass).
The reference load in evidence/qwen38-serve-fast-20261019 took 64 s. Cause not known yet; run 3 samples the
server process every 30 s (state, CPU time, memory, kernel wait) and allows 2400 s.
