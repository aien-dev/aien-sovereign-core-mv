# L598 review resolution

All three findings reproduced as actual RED assertions on source9ccfc410, then fixed; full Renda suite73/73PASS on Mac and Spark.

- Important duplicate admission: repository-scoped nonblocking flock covers the whole check/preflight/launch boundary. Direct argv Popen, not detached shell text, passes the lock FD to the actual judge process. Parent closes but never unlocks the inherited open-file description. Concurrent-thread counterexample now starts one child, and an actual harmless child-process regression confirms the lock stays held after parent returns and releases on child exit. No production judges launched in tests.
- Acknowledgement provenance: medium regraded Important auditability. Require canonical aggregate filename and ack_path, exact original input bytes digest, matching full manifest items and queue-entry provenance. This is an auditable trusted-local control, not a security claim against arbitrary same-user filesystem tampering. Shape-only fake ack refused.
- Malformed planner section: low regraded Important fail-closed admission. Unindented/duplicated unmapped sections and removed gates refused even if exit0/RUNNABLE. No thresholds or official gate assertions changed.

No deferred minors. No further reviewer run substitutes for these reproductions. Official core/model/gate qualification and actual live-delivery acceptance remain separate and pending.
