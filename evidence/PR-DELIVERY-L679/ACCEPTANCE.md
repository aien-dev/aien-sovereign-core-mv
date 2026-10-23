# Live acceptance and limits

The service repair is deployed and exercised, not a claim that the PR backlog is merged.

- Reviewed executable source: `7dfcf3d10f178147ff9e494cce050c54d3fa332b`.
- Immutable installed release: `L603-fbfa8c78dc482892`; installer replayed all packaged suites and matched reviewed file hashes.
- Replacement Renda1 PID3142568: startup 10:52:11Z, completed pass 10:58:18Z.
- Replacement Renda2 PID3198603: startup 10:56:10Z, completed pass 10:59:04Z; lanes omega,synapse,aegis-runtime,spark-rsi.
- Published and read back OPEN: protocols#48 `8bb8034c7b7d1be8504f0d7553023891ad96d57d`; benchmarks#36 `8381a6cd85082f49efc95c51badffa5285ae80a3`; interplane#116 `ad861a22e7015ed5a7fa0a85feed2621f770dc3b`.
- Retired failed Omega#620, read back CLOSED at `7bc3f0b6ff41a83e5ac932d72b5544aaae8a8b52`; new smaller batch#641 OPEN at `c0d29e77fedc7f7debc2f554bc995d69f736470e`, carries #484/#487/#494. The prior FAIL remains evidence, not merge authorization.
- Restored Aegis-runtime exact `76e706cb8bfab65daeeff9be1d27d714ce463e98` and Spark-RSI exact `4617501abc658519fce3c0755dac1ac7f5466b7b` operator runs completed PASS. Raw logs retained.
- Zero merges in those first passes. Core#577/#609/#626 remain on original heads; acknowledged successors still need complete current-policy review and signed qualification. Three publication successes do not prove successor promotion has occurred.
- Other kernel/non-Spark identity or adapter gaps remain held under L686. Actual Omega/architecture gate repairs remain L687 with the existing owners/coordinator, not waived by this repair.

## Monitor correction

`adoption-initial-invalid-prestart.json` is INVALID evidence of completed new-worker passes. Its original monitor accepted any pass after installation, incorrectly including old workers' final passes. Both true flags in that artifact are false positives. The failure was reproduced directly against the captured logs; `watch_adoption.py` now requires a pass strictly after the latest startup marker matching the sole replacement process. `adoption-readback.json` is the subsequent corrected observation. Keep both artifacts; do not treat the earlier file's LOADED_PASSES_VERIFIED label as authoritative.

Watchdog registration also required the collector's literal `cron ` prefix before each complete cron line. Only the two authorized Renda entries were corrected; unrelated job alarms were not approved or cleared.
