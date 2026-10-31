# Two receipts listed in SHA256SUMS are not in the tree

`SHA256SUMS` lists `g30-tokens/run.log` (sha256 05d851b2...) and `g30-speed/run.log` (sha256 57439099...).
Neither file was committed: the repository `.gitignore` line 3 ignores `*.log`, and the run worktree
(`/tmp/corechip/w557`) was removed after the rerun, so the two logs no longer exist. `sha256sum -c SHA256SUMS`
therefore reports 2 files it cannot read; every other listed file matches (checked by Main, 2026-10-31).

Neither gate result depends on the missing logs. Main re-derived both from committed files:

- `g30-tokens.rc` and `g30-speed.rc` are `0`; both `.raw` files end with `SUMMARY parity=PASS tokens=PASS`.
- `jq -c '[.prompts[] | .chip]' run.json | sha256sum` begins `bb1a0401175ab69d` for `g30-tokens/run.json`,
  `g30-speed/run.json` and `../adr0004-step0-baseline-20261015/check-full-3/run.json`.
- The decode rates come from the `TOKENS` lines in `g30-speed.raw`: p1 44.7, p2 42.2, p3 43.5, p4 42.2 tok/s.

`SHA256SUMS` is left as the lane wrote it. Future reruns should `git add -f` any `run.log` named in a manifest.
