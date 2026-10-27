# Peak memory of the live 27B server on the current build (2026-10-19)

proc-status.txt: /proc/2473716/status and /proc/meminfo read at the time shown, for the live
server on port 18091 (binary sha256 df936011..., the build in evidence/qwen38-serve-fast-20261019).
Read only; no server was started or restarted for this. VmHWM (peak resident memory since start)
38,499,160 kB = 38,499,160 / 1,048,576 = 36.7 GiB (docs/qwen38/calc-receipts, calc:39181948),
after 2 h 48 min of service with --max-context 131072. The earlier build peaked at 40,358,896 kB,
38.5 GiB (calc:f6c02d78). Not measured: a second server started beside it, and whether it would
share the file-backed part (RssFile) of the first one.
