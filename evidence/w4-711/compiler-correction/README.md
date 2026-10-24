# Corrected compiler compatibility

The exact proposed compiler from Omega PR676 head
`1dffceaeb0dd26fb69ea64d367a49a9950c25a62` was consumed from its committed
`seed/out/omc` blob, not from the owner's changing working directory. Its SHA256
is `9f48f66a9ee9f6a166df615a7ab9c48bf5b06ac2ba544ac51e52e9a3d290a6c4`.
Functional source is `078455c9c1862b9145ef892dfba68676204a38da`. The owner's
combined seed reconstruction and branch-safety qualification are recorded in
that PR's `evidence/NATIVE-MAPPING/branch-correction/`; this consumer check
does not claim to repeat the entire compiler bootstrap or independent review.

All listed programs were freshly compiled from the ordered source units in
receipt.json, using the exact compiler above through quiet-build timeout30.
Each produced executable is byte-identical to the previously tested binary.
The listed test replays were executed again in fresh private directories,
with a30-second timeout (10seconds for the classifier). Expected INVALID65
is a rejection test, not a process failure counted as a caught control.

Reconstruct by concatenating each build's source_files in listed order, feed
the result to the pinned compiler on stdin, and require exit0 before executing
its stdout artifact. Compare its SHA256 to both recorded binary digests.
Original qualification/reproduction remains in the parent evidence folders.
No historical receipt was rewritten. Production source and executable digests
are unchanged. GPU, full-model execution, hardware timings and repository
merge gates were not rerun or waived. The compiler remains proposed work.
