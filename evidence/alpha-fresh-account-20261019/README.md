# Alpha fresh-account test: docs/qwen38/TRY_IT_ON_YOUR_SPARK.md

**VERDICT: FAIL as written. Step 2 (`./install.sh --with-server`) exits 101 and installs nothing. With one hand-run workaround, steps 3 to 6 all PASS (checksums 81 of 81, first chat reply, GATE PASS, decode ids MATCH, 130.6 chip ms per token).**

- Test user: `aienalpha` (new Linux user, uid 1003, own HOME, created with sudo on the Spark, 2026-10-19 about 11:28 CDT).
- Guide and code: branch `alpha/try-it-guide`, commit `71f1fe592b57bdebdb51c474f4a7b37e36b0ae90` (clone checked out at exactly that commit, 0 dirty files).
- Test server: port 18391 (the guide says 18091; the live server on 18091 was never touched).
- Receipts are published on branch `alpha/fresh-account-evidence`, folder `evidence/alpha-fresh-account-20261019/` (same files as listed below, no model weights).
- Ledger: L009 (this test), findings filed in L032, L035, L036; new items listed at the end.

## Step by step

| Step | Result | Number (receipt file) |
|---|---|---|
| 1. Needs: driver, devices, programs | PASS | driver 580.173.02, 3 device files readable and writable, git/curl/jq present before the test (so `apt install` was NOT exercised) (`aien-install.log` lines 4 to 8) |
| 2. `./install.sh --with-server` | **FAIL, exit 101** | died after 20.56 s wall; nothing installed (`aien-install.log`, `install.rc` = 101, `install.time`) |
| 2 workaround: the installer's `server_build` steps run by hand | PASS | pins 9 s + cargo build 190 s = 199 s (3:19 wall) on a box with load average 87; `aien-serve-http --version` says `native` on both engine lines; binary sha256 `74aaaaa2...238db5` (`server-build.out`, `run-server-build.sh`) |
| 3. Model files | PARTLY TESTED | `sha256sum -c`: 81 of 81 OK in about 39 s (`model-sha256-check.txt`). But 66 of the 81 files (all the big safetensors) were hard links to the live copy, not downloaded; only the 15 small files came from Hugging Face through the guide's loop. The 31 GB download itself is untested here (L033 times it) |
| 4. `aien-serve-27b` | PASS after a wait | first start refused (below). Second start at 13:19:26 CDT with 52,375 MiB free: ready 452 s later, of which 62.3 s was the model load and about 390 s was queueing for the shared chip lock (`serve-27b.log`) |
| 5. First chat reply | PASS | two sensible sentences about the DGX Spark, finish `stop`, 60 tokens, chip 132.1 ms per token, 7.61 tok/s; client wall 78.2 s of which 69.7 s was lock wait (`first-chat.json`, `first-chat.time`) |
| 6a. Gate | **PASS** | `GATE PASS`, `golden ids against the shipped reference: MATCH`, 24 of 24 ids equal; resumed equals cold on turns 2, 2b and 3; negative control differs. Wall 862 s, almost all lock wait (`gate/receipt.json`, `gate/gate-output.txt`) |
| 6b. Speed | **PASS** | `decode ids against the shipped reference: MATCH`; ids hash `ceb68306ae2d` in all three runs; chip ms per token 136.5, 130.6, 129.9, median **130.6** = 7.657 tok/s = 0.754 of the 10.15 ceiling. Ours in the guide: 131.05 ms, 7.63 tok/s, 0.752 (`speed/receipt.json`) |

Other numbers:

- Memory at the server: log at ready says weights 26.92 GB, KV 8.59 GB, RSS 38.55 GB. Peak resident (VmHWM) before stopping: **40,368,140 kB (38.5 GiB)**. The guide's older figure was 40,358,896 kB, so the current build is within 0.03 percent of it. This fills the guide's "current build's peak has not been recorded" gap.
- Prefill: 1,565-token prompt 12,083 ms; 21,507-token prompt 148,187 ms (gemm chip 52,549 ms); follow-up on the 21,507-token prompt (state reused, 25 new tokens) 610 ms.
- Stop: `kill -TERM` on an idle server ended it in under 1 s, port freed.
- Machine during the runs: load average 35 to 46 (gate and speed receipts), MemAvailable 16 to 23 GiB while the server held 39 GiB. Other people's jobs were holding the chip lock the whole time, which is why wall times are long. Chip ms per token is read from the server log and is not affected by the queue.

## Every break, in order

1. **`install.sh --with-server` fails at the "Building Core Monorepo Binaries" line (install.sh line 567).** Message: ``error: specification `spark-crumbs` is ambiguous``. The workspace has a local crate `crates/spark-crumbs` and six crates that depend on a git copy of `spark-crumbs` at rev 9355a92b, so `-p spark-crumbs` matches two packages. Reproduced on its own with `cargo pkgid --locked spark-crumbs` (same error in the other checkout `sc-alpha-docs`), so it is the code, not this account. The installer runs the core build before `server_build` (line 592), so a failure here means the server is never built and nothing is copied to `~/.local/bin`. Fix to try: use the full package id (`path+file://.../crates/spark-crumbs#0.1.0`) or drop `-p spark-crumbs` for the git copy; also consider building the server first when `--with-server` is set. New ledger item opened.
2. **Memory figure is wrong in the guide, the installer check and the start script.** They warn below 40 GiB. The loader needs 41,305 MiB plus 8,192 MiB kept free, so about 48.3 GiB (49,497 MiB) available. The installer check passed at 53 GiB free; the first start then refused at 46,249 MiB: `qwen38 load failed: needs 41305 MiB plus 8192 MiB kept free, MemAvailable 46249 MiB ... refusing` (`serve-27b.log`, 17:22:10Z). Also measured (L032): a second 27B server does not share page cache with the first; it needs the full amount. Open: L035.
3. **`aien-serve-27b` prints "started, pid N" and exits 0 even when the server then refuses to load.** `serve-start.try1.out` and `serve-start.try2.out` are byte-identical "started" messages; the refusal is only in the log. A newcomer would curl a dead port. Open: L036.
4. **The memory check happens before the wait for the chip lock.** The server checked memory, then queued about 390 s for `/tmp/aien-gb10.lock` (limit 1,800 s). Memory on this box swung between 6 and 59 GiB during the wait (`mem-wait.log`). On a quiet Spark this is harmless; on a shared one a start can pass the check and then not have the memory. New ledger item opened.
5. **Guide wording that did not match the run:** "about 30,000 tokens" is 21,507 tokens in the speed script; "about 1,700" is 1,565. "Loading takes one to two minutes" holds only when the chip is free (62.3 s of load; our wait was far longer). The gate and speed scripts took 14.4 and 11.7 minutes on a busy chip; the guide gives no time. Memory sentences in sections 1 and 7 should now say 40,368,140 kB peak, 48.3 GiB free needed at load.
6. **Not exercised, so not claimed:** `apt install` of the programs (already installed), the large-file download over the network (hard links), `hf download`. The PATH line the installer adds: `aien-serve-http` was found in a new login shell (`bash -l`), which is the only way I checked.

## What I did not break

The live server on 18091 (pid 2473716) was running and answering before and after; I never restarted it or killed any job. The test server on 18391 was stopped at 13:56 CDT after the receipts were collected; port 18391 is free and the 39 GiB it held is returned.

## Files

- Spark: `/home/aienalpha/aien-check/gate/receipt.json`, `/home/aienalpha/aien-check/speed/receipt.json`, `aien-install.log`, `server-build.out`, `model-sha256-check.txt`, `mem-wait.log`, `first-chat.json` (copies in the evidence folder above).
- Scripts I used: `/home/aienalpha/run-server-build.sh` (the workaround, the installer's `server_build` commands by hand), `/home/aienalpha/wait-and-start.sh` (waits for 49,600 MiB free, then runs `PORT=18391 aien-serve-27b`).
