# Omega live deployment

Goal: one interactive coding model on Spark, using Omega and the fastest qualified existing improvements.

1. Preserve the running bin-19173f9e and its configuration. Assemble in this isolated branch from its documented source dae01e2e. Verify canonical PR status rather than treating old handoffs as current.
2. Carry merged sampling PR 646 (754c07b1). Run the server host tests, sampling equivalence and benchmark, then build against the exact live Omega archive. Evaluate ready prefix reuse separately against its existing acceptance tests.
3. Qualify one candidate against the current server: fixed greedy and sampled responses, streaming, second turn, concurrent isolation, tool call and actual coding output. Record timing with shared-load limits. Stop the candidate on failure.
4. Resolve the relevant outstanding PRs through the existing merge protections with required checks passing. Preserve experimental work and never treat skipped tests as passes.
5. Deploy the qualified binary with an explicit rollback to bin-19173f9e. Verify the running executable, chat/coding access, restart configuration and that no other model process remains.

Constraints: no third-party inference engine, no Python, no speculative precision changes, no driver changes, no skipped merge checks. Current library includes GPU coalesced stores; lower-precision state remains disabled because chunked prefill compatibility is unproven.

Review focus: exact sampled tokens including ties/nonfinite inputs; retained live GPU improvements; conversation isolation; tool/streaming compatibility; rollback and restart persistence.
