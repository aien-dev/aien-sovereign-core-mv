# sovereign-core #661 / #688 browser mirror on the DGX Spark

- Tested head: `f81de1178021e7b9ce2544b69627ddbb10604bdd` (branch `fix/661-browser-mirror-reap` after merging main `9612b7787fcca2463b1b9737a7faebe91a72552a`). The commit that adds this directory changes nothing else.
- Host: DGX Spark, `Linux 7.0.0-1019-nvidia aarch64`, `Google Chrome 153.0.8010.36` at `/usr/bin/google-chrome`. Runs used CPU cores 6-9 and 15-19 at nice 19; no GPU, no model, no live service.
- `browser-mirror-tests.log`: the library test binary's `browser_mirror` tests with `--test-threads=1 --nocapture`. 8 passed, 1 ignored (the isolated SIGCHLD case), exit 0. `stub_page_screenshot_and_eval` captured a screenshot from real Chrome 153 (the aarch64 hang of #661 does not occur with `--disable-gpu --use-gl=disabled --disable-software-rasterizer`). `dropping_chrome_reaps_every_helper` and `tag_sweep_matches_only_its_own_tag` ran against real Chrome.
- Same log, second run: `cleanup_with_inherited_ignored_sigchld` executed through `perl -e '$SIG{CHLD}="IGNORE"; exec @ARGV'` with `AIEN_CHROME=/usr/bin/google-chrome`, so the test process inherited SIGCHLD=SIG_IGN as the test requires. 1 passed, exit 0.
- After both runs `pgrep -fc 'aien-mirror|profile-fixture|owned-sigchld-fixture'` counted 0 processes.
- `aien-cli-tests.log`: `cargo test --locked -j2 -p aien-cli` at the same head, summary lines only. Exit 0, 0 failed.
- Not covered: Chrome on other architectures, other Chrome versions, and non-Linux Unix cleanup (not qualified, as `skills/self-improvement/SKILL.md` says).
