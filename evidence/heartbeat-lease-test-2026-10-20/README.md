# writer_waits_for_lease_release: timing fix (ledger L152)

Cause: the test slept 10 ms and assumed the reader thread had taken its lease by then, then asserted the writer waited >= 20 ms. Under load either thread can be late, so the writer sometimes passed straight through.

Fix: the test now waits on events only. A channel says "reader holds the lease"; a test-only counter on the slot (`writers_waiting`, cfg(test), no change to real behaviour) says "writer is blocked inside begin_fill"; a second channel releases the reader. The test then checks the writer was still blocked while the lease was held, and finished after release, and that the reader saw its original bytes.

Other sleeps in crates/aien-heartbeat: none (the only two were in this test). `leased_slot_is_never_overwritten` and `reader_lease_holds_back_the_writer` hold the lease on the test thread and only use a timeout as an upper bound, so they are not timing-dependent.

Evidence (40 busy `yes` loops, started and stopped by pid; stress-ng is not installed; machine load 60-90):
- old-test-stress.txt: old test, 20 runs, 15 pass / 5 FAIL (assertion `waited >= 20 ms`).
- fixed-crate-stress.txt: whole crate with `--test-threads=1`, 20 runs, 20 pass / 0 fail.
- negative-control-writer-ignores-lease.txt: with begin_fill changed to ignore leases, the new test fails (so it still proves the no-overwrite rule).
- stress-loop.sh: the loop used.
