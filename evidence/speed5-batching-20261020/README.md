# SPEED-5 batching (L129) evidence

Branch speed5/batching (sovereign-core), omega lib cascade/tree-verify-gemm 22446aef (sha256 8217447c...).
Scripts: window.sh (all sessions, one spark-window), session.sh, bench.sh, summarize.sh, compare.sh (shell, no Python).

## Log
- 2026-10-20T03:20Z host tests (CPU stub; aien-serve-http, aien-inference-abi, aien-omega-fp8) at 1636f455: 448 passed, 0 failed, 22 ignored (host-tests-1636f455.txt). 7c3b0381 adds par_rows + its test (passed alone).
- Native binary bin-7c3b0381 sha256 058fc0eb... Waiting for a spark-window with MemAvailable >= 52 GiB.
