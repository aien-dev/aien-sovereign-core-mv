# Cascade MTP chain, n-gram source and draft tree: measurement suites (2026-10-19)

What this is: the speed-up (accepted words per verify pass) and exactness (output identical to plain one-word-at-a-time greedy)
of the cascade branch `cascade/mtp-chain`, measured with `qwen38_spec suite` (one model upload per suite, many runs).

- Binary: `qwen38_spec` built from e166f57f (rustfmt-only change after that, 054377b2), omega library 37034f03 (sha256 6e905209...).
- Plans: `plan-A-mtp.txt` (golden chain + golden tree first, then MTP chain on prose, code edit, quote), `plan-B-merge.txt`
  (merge and n-gram-first sources), `plan-C-mutants.txt` (broken-on-purpose variants that MUST fail), `plan-D-tree.txt` (draft trees).
- Each `<name>.json` is one run; a `PASS` flag there means every k produced the same tokens as k = 0 (and the golden record where given).
  Mutant runs are expected to FAIL; a mutant that passes would mean the test cannot see the bug.
- `command-<suite>.txt` has the exact command, binary sha, load and memory at start and end. `suite-<name>.log.txt` is the tool's output,
  `suite-<name>.refused-N.log.txt` are starts the tool refused for lack of memory (kept, not hidden).
- Timing caveat: the box is shared and loaded (see load and NVRM error counts in each receipt). Milliseconds per token in these files are
  NOT speed claims. The claim is tokens per pass; `../cascade-summary.sh` projects tok/s at the "one verify pass costs one plain pass"
  target from tokens per pass (two stated bases), it is a projection, not a measurement.
- Earlier per-run batches (`../qwen38-cascade-chain-20261019T1650Z`, `../qwen38-cascade-tree-20261019T1650Z`) were replaced by these suites
  because the memory wait hit 20 refusals per run; mtp-codelong from that batch is kept and counts.
