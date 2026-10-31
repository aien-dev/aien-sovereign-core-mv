# sovereign-core #661 / #688 browser mirror on the DGX Spark

Host for every run: DGX Spark, `Linux 7.0.0-1019-nvidia aarch64`, `Google Chrome 153.0.8010.36` at `/usr/bin/google-chrome`. Runs used CPU cores 5-9 and 15-19 at nice 19; no GPU, no model, no live service.

## First run, head `f81de1178021e7b9ce2544b69627ddbb10604bdd`

- Branch `fix/661-browser-mirror-reap` after merging main `9612b7787fcca2463b1b9737a7faebe91a72552a`. The two commits that add this directory also regenerated the root `.crumb`; they change no code.
- `browser-mirror-tests.log`: the library test binary's `browser_mirror` tests with `--test-threads=1 --nocapture`. 8 passed, 1 ignored (the isolated SIGCHLD case), exit 0. `stub_page_screenshot_and_eval` captured a screenshot from real Chrome 153 (the aarch64 hang of #661 does not occur with `--disable-gpu --use-gl=disabled --disable-software-rasterizer`). `dropping_chrome_reaps_every_helper` ran against real Chrome. `tag_sweep_matches_only_its_own_tag` spawns `sleep` fixtures, not Chrome.
- Same log, second run: `cleanup_with_inherited_ignored_sigchld` executed through `perl -e '$SIG{CHLD}="IGNORE"; exec @ARGV'` with `AIEN_CHROME=/usr/bin/google-chrome`, so the test process inherited SIGCHLD=SIG_IGN as the test requires. 1 passed, exit 0.
- The leftover check after both runs, `pgrep -fc 'aien-mirror|profile-fixture|owned-sigchld-fixture'`, counted 0, but its first alternative cannot match the real tag `aien-browser-mirror-<pid>-<nanos>`, so it did not check for leftover Chrome. The second run below uses the corrected pattern.
- `aien-cli-tests.log`: `cargo test --locked -j2 -p aien-cli` at the same head, summary lines only. Exit 0, 0 failed.

## Second run, head `9f7451605fd4f001fbe987e1f07dc1f15cf2fa92`

Review of `4965d44c` (main merged) found that Linux started Chrome in its own process group although Linux teardown never signals a group. `9f745160` keeps Chrome in the caller's group on Linux.

- `sigint-replay-4965d44c.log` (script `sigint-replay.sh`): `aien-browser-mirror --eval` started as leader of its own process group, as a terminal foreground job or a harness timeout group would, then SIGINT to that group. At `4965d44c`, 13 tagged Chrome processes were still alive 4 s later. (The script's group signal goes through `/bin/kill`; dash's builtin `kill` rejects `-- -PGID`.)
- `reverify-9f745160.log` (script `reverify.sh`), run inside `systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0` on the tree of `9f745160`: the build has no compiler warning from `aien-cli`; the `browser_mirror` tests 8 passed, 1 ignored; the isolated inherited SIGCHLD=SIG_IGN case 1 passed; `pgrep -fc 'aien-browser-mirror-[0-9]|profile-fixture|owned-sigchld-fixture'` counted 0; the same SIGINT replay left 0 tagged processes.
- The full `cargo test --locked -j1 -p aien-cli` under the same 3 GiB cap passed on the tree of `4965d44c` and again on the tree of `9f745160`: exit 0 both times; 223 passed, 0 failed, 21 ignored over 13 test binaries.

Not covered: Chrome on other architectures, other Chrome versions, and non-Linux Unix cleanup (not compiled or run here; not qualified, as `skills/self-improvement/SKILL.md` says).
