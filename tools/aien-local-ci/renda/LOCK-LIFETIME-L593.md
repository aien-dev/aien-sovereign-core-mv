# Admission lock lifetime

Final service suite74/74PASS Mac and Spark. Added a real-child boundary regression after the reviewed race repair: the admission lock must live as long as the judge but must NOT propagate to gate fixtures or their orphaned servers. A fixed trusted Python waiter holds the inherited lock, executes the actual judge with close_fds=True, and returns its actual exit status. The gate receives no admission FD. The positive test verifies lock exclusion while the real child waits, then release at job completion; the existing concurrent-admission regression still starts exactly one child. This is an extension of the L598 lock-lifetime repair, not a new judge or model service.

The previous389d32bc deployment is superseded only at existing code-client idle boundaries. Active judges and requests remain untouched. New source changes operational launch mechanics, not gate assertions or model output requirements.
