# QWEN38-SERVE and QWEN38-SPEED: first runs on the Spark (2026-10-19, ledger L068)

Gates: `tests/gates/qwen38-serve.gate`, `tests/gates/qwen38-speed.gate`; driver
`tools/qwen38_gate_run.sh` through `tests/gates/run.sh qwen38-serve|qwen38-speed`.

## refusal/ (memory short, by hand)
The box had three other model servers up (two more 27B instances and an 8B, other agents'
experiments) and MemAvailable sat at 18 to 30 GiB against the loader's 49,497 MiB
(41,305 MiB model + 8,192 MiB kept free). The serve gate waited the bounded time and then
refused: `QWEN38-SERVE FAIL: memory short ...`, rc 1, in 45 s, no server started, no chip lock
taken (the memory check runs before the lock queue). `refusal/gate.out` is the gate's whole
stdout (AIEN-OBS lines are the bundle fields), `refusal/receipt.json` the receipt. `machine.txt`
is the box at the time, `command.txt` the exact commands.

## serve/ and speed/ (the real measurement)
Queued: `~/handoffs/2026-10-19-qwen38-gates-queued.sh` on the Spark waits for the memory, runs
both steps (each takes /tmp/aien-gb10.lock through quiet-measure for its own server's life),
copies the receipts here, commits and pushes. Until those directories exist, no tokens/s figure
from these gates exists; the reference figure stays the live server's 7.63 tokens/s
(evidence/qwen38-serve-fast-20261019, 131.05 ms per token, median of 4 runs).
