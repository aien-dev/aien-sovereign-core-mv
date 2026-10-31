# DRA-66 chip plan: host time per token (wall − chip) on the served 35B, as P-DRA-4 queue slices

Issue: https://linear.app/drake-stapleton/issue/DRA-66 · aien-dev/aien-sovereign-core-mv#857 · draft PR #863
(branch `dra-66/host-time-per-token`). Owner: DRA66Host (Main lane). Every chip line below is **NOT_RUN**
(host prep only; no GPU opened, no slice run, no GO).

The repo copy (`evidence/dra-66-host-time-20261031/chip/`) and the queue copy
(`~/handoffs/pdra4-hive/chip/DRA-66/`) are byte-identical (`SHA256SUMS`). This replaces the earlier
single-window `window.sh` (removed in PR #863).

## Binary and inputs (pinned by sha256 in `dra66-lib.sh`; a mismatch is NOT_RUN, exit 3)

| file | sha256 | from |
|---|---|---|
| `~/work/dra-66/bin-head/aien-serve-http` | `a33cc2ab…49e4` | sovereign-core `9d70dca4` (binary reports it, dirty 0), omega `a647b6a9`, physics `048e5f5e`, aienos `b84c0a67`; native `libomega_gpu.a` (recipe in `../PROTOCOL.md`) |
| `evidence/q36-host-20261020/gen-chat-max-24.json` | `ec904017…a664` | gate2 golden, 24 `generated_ids` |
| model | — | `/home/drakestapleton/models/qwen3.6-35b-a3b-fp8-95a723d0` (`config.json` checked) |

## Window and chip lock

Both slices carry `# pdra4-gb10: none` (run-queue.sh sha256 `420b6cc7…`, `CHANGES-run-queue.md`). The
runner opens `spark-window open pdra4-DRA-66-<slice> --lease 20 --gb10 none --quiet exclusive --wait` and
runs `timeout 1140 spark-window run env Q36_EXTRA_WAIT=120 Q36_EXTRA_LEDGER=… ~/.local/bin/q36-extra <slice>`.

- An aien-serve-http server accepts an inherited chip reservation (`AIEN_GB10_LOCK_HELD=1`) only when
  the FLOCK holder of `/tmp/aien-gb10.lock` is one of its ancestors (`chip_reservation.rs` `detect`).
  spark-window's holder (`setsid spark-window _hold ID`) never is, so a `--gb10 exclusive` window
  cannot serve the door. The slice takes the lock itself: `flock -w 90 -E 75 /tmp/aien-gb10.lock bash
  <slice>`; flock(1) is the ancestor of every server, `/proc/locks` (device + inode match) must name
  it, and the server log must say `chip reservation inherited from ancestor pid <that pid>`.
  `AIEN_GB10_LOCK_HELD=1` is set only on the server command lines. No new trust path.
- flock exit 75 (lock not free in 90 s) becomes this slice's own `NOT_RUN` VERDICT and exit 3, so the
  runner never mistakes it for a q36-extra refusal. The lock goes when the inner slice exits;
  `timeout(1)` signals the whole process group at the cap (server, inner bash and flock(1) alike).
- The window's quiet lock (`--quiet exclusive`) is what keeps the timings quiet.

Refusals before any chip contact (NOT_RUN, exit 3, VERDICT written inside a window, stdout only outside
one): no `SPARK_WINDOW`/`SPARK_WINDOW_DIR`; `spark-window id` ≠ `SPARK_WINDOW`; no `held` stamp;
`gb10` ≠ none or `quiet` ≠ exclusive; `AIEN_GB10_LOCK_HELD` already set; s02 only: s01-tokens has no
`PASS` VERDICT; lock not free in 90 s or its holder not an ancestor; 18091 answers `/health` or
anything listens on 18091/18300/18310; a compute process is on the chip; sha256 mismatch; the binary
has no native omega gpu engine; port 18190 busy.

## Server

The live 18091 flags on loopback port 18190: `--engine omega --model-dir <model> --port 18190 --bind
127.0.0.1 --max-context 49152 --max-tokens-cap 16384 --thinking off --batch-slots 8`; env
`AIEN_GB10_LOCK_HELD=1 AIEN_Q36_STROKE=1 AIEN_STROKE_IN_WINDOW=1 AIEN_OMEGA_FP8_GROUPED_WAVE=0`.

- `AIEN_OMEGA_FP8_GROUPED_WAVE=0` is explicit. The 18091 launcher
  (`~/.local/state/aien-q36/start-35b-batch.sh`) leaves it unset and no script in
  `~/.local/state/aien-q36/`, `~/.local/bin/` or the user systemd units sets it, so the served 35B
  runs the omega `a647b6a9` default (off): `~/.local/state/aien-q36/serve.log` has its last
  `(wave rule)` proof line at line 13207, none in the four starts at lines 70651-70706.
  `bin/GATEKIT.md` names `=1` as the ship-env lever of `bin-cc45f45e` (its `BUILD.txt` line 5); this
  issue measures the configuration actually served, so 0. A server whose log shows `(wave rule)` is
  stopped and the gate is NOT_RUN.
- `AIEN_STROKE_IN_WINDOW` is read only by `qwen36_stroke_gate`; it has no effect on aien-serve-http
  and is set for the queue convention.
- Load cap 180 s to `/health`: loads measured 39/57/78 s in the GoldenP04 window (2026-10-27, same
  model and flags); the four latest 18091 starts reported `engine ready in` 112.4, 78.6, 32.9 and
  148.6 s (`serve.log` lines 70651-70706; the last a 4-row start).
- Server affinity and priority are inherited from the runner (recorded in `preconditions.txt`).

## Slices (queue order; reds before greens)

Receipts: `~/handoffs/pdra4-hive/receipts/DRA-66/<slice>/`: `gate-<name>.json`
(`aien.dra66-slice-gate/1`: window, binary sha and commit, lock holder pid, wave, start/end,
`deadline_left_s`, status, detail, sha256 of every file), `receipt.json` (`aien.dra66-slice/1`, status
= FAIL > NOT_RUN > PASS over the gates) and the one-line `VERDICT` (`PASS|FAIL|NOT_RUN <slice>: <gate
statuses>; first non-PASS: …`). Exit 0 PASS, 4 FAIL, 3 NOT_RUN. A gate starts only if its cap fits
before both the lease end and 1140 − 15 s after the slice process started; otherwise it and later
gates are NOT_RUN. Board lines come from the runner only.

| slice | gate | content | expected | est. s | cap s |
|---|---|---|---|---|---|
| s01-tokens | (outer) | window checks, `flock -w 90` | lock free at once | ~0 | 90 |
| | `mutant4-red` | server with `AIEN_Q36_TEST_MUTANT=4`, gate2 request (24 greedy tokens), stop | `DIFFER` from the golden; else FAIL, exit 4 | 78 + 0.56 + 5 | 270 |
| | `golden-identity` | honest server, per-token receipt (`AIEN_Q36_DECODE_TIMELINE_KIND=token`), gate2 request, 2 s idle; records A | `IDENTICAL` | 78 + 0.56 + 2 | 240 |
| | `receipt-1conv` | 3 sequential greedy COUNT replies of 256 tokens, 2 s idle; records B > A + 1 | all answer | 18.04 + 2 | 92 |
| | `receipt-8conv` | 8 concurrent greedy replies of 256 tokens | all answer | 21.44 + 2 | 62 |
| | `seed-t07` | seed 1, seed 2, seed 1 at T = 0.7, 64 tokens | 1 repeats, 2 differs | 4.51 | 60 |
| | `check-tokens` | stop; checker on `tokens.jsonl`: check (0), `--drop 0:record:0` (1), `--drop n/2:record:0` (1), `--bend A+1:chip` (1), `--bend host:600000 --slice A:B` → `summary-red-host.wl` (0), `--slice A:B` → `summary-1conv.wl` (0) | exits `0 1 1 1 0 0` | 5 + 5 | 75 |
| s02-stamps | (outer) | as s01, plus s01-tokens `VERDICT` must start `PASS` | | ~0 | 90 |
| | `stamps-1conv` | honest server, stamped launch timeline (`AIEN_Q36_DECODE_TIMELINE`, default kind), 1 greedy COUNT reply of 128 tokens, 2 s idle, stop | answers, ≥ 1 record, ids = first 128 of s01 `solo-1.json` | 78 + 3.01 + 2 + 5 | 242 |
| | `check-stamps` | `--drop 0:record:0` (1), check → `summary-stamps.wl` (0); then `gate-red.wl` = s01 `summary-red-host.wl` + `summary-stamps.wl`, then `gate.wl` = s01 `summary-1conv.wl` + `summary-stamps.wl` | exits `1 0` | 5 | 45 |

Wolfram (MCP evaluator), one query:
`t[n_, u_] := 9.13 + 3.29 n + 0.696 u + 4.81 Ceiling[n/4]; t1 = t[1, 9]; t8 = t[8, 55.6];
cap01 = 90 + (180 + 60 + 30) + (180 + 60) + (3*30 + 2) + (60 + 2) + 3*20 + (30 + 45);
cap02 = 90 + (180 + 30 + 2 + 30) + 45;
s01[L_, st_] := 2 L + 2 st + (2*24 + 3*256 + 3*64) t1/1000 + 256 t8/1000 + 6 + 5;
s02[L_, st_] := L + 128 t1/1000 + 2 + st + 5;
{t1, t8, cap01, cap02, 1140 - 15 - cap01, 1140 - 15 - cap02, s01[78, 5], s01[180, 30], s02[78, 5],
s02[180, 30], cap01 + cap02, s01[78, 5] + s02[78, 5]}`
→ `{23.494, 83.7676, 889, 377, 236, 748, 222.1264576, 476.1264576, 93.007232, 220.007232, 1266,
315.1336896}` (s; t in ms).

- Caps: s01 889 s, s02 377 s, each ≤ 1140 s. s01 runs every gate only if q36-extra's admission
  (slot wait ≤ 120 s, memory wait) leaves 236 s or less used before the slice starts; s02 748 s.
- Expected: s01 222.1 s (load 78 s, stop 5 s), 476.1 s at the 180 s load cap and 30 s stops; s02 93.0 s
  (220.0 s). Both slices 315.1 s expected, the same three loads as the old single window.
- The stamped reply is costed at the unstamped t(1); stamps slow the chip, its cap is 30 s.

## Host dry path (no window, no lock, no GPU, no server; exercised 2026-10-31)

`DRY_RUN=1 bash s01-tokens.sh` then `DRY_RUN=1 bash s02-stamps.sh`: static checks (binary, golden, model,
tools), the plan with caps, then the slice's own checker gate (`check_tokens` / `check_stamps`, the code
the chip run uses) on the committed host fixtures `../host-token-receipt.jsonl` (slice 0..19) and
`../host-stamps.jsonl`, into `/tmp/dra66-dry/<slice>/` (never the receipts directory). Output in
`dry-host.log`. The dry `gate.wl` / `gate-red.wl` differ from `../host-gate.wl` / `../host-gate-red.wl`
only in the fixture path inside the first comment line, and Wolfram gives the same verdicts:
`gate-red.wl` → `{0.715041, 0.02, 0.735041, False}`, `gate.wl` → `{0.115041, 0.02, 0.135041, True}`.

Negative controls in `dry-host.log`: outside a window → `NOT_RUN`, exit 3, no VERDICT written; a fake
window (temp copy, temp receipts) → `NOT_RUN … not the open window`, exit 3, VERDICT; the inner part
without an ancestor holding the chip lock (temp copy) → preconditions `NOT_RUN`, exit 3, no chip query;
the dry path with a corrupted `SHA_HEAD` (temp copy) → exit 3.

## After the window (Main, host)

1. Both VERDICTs; s01 `PASS` means the mutant went red before the honest golden was IDENTICAL, the
   seed control held and every checker red fired.
2. Evaluate `receipts/DRA-66/s02-stamps/gate-red.wl` whole with the Wolfram MCP: the last result must be
   `False`. If it is `True` the gate is broken and `gate.wl` does not count.
3. Evaluate `gate.wl`: last result `{hostLB ms, submit+poll ms, sum ms, sum <= 0.5 ms}`; `True` PASS,
   `False` FAIL for "host time ≤ 0.5 ms per token at one conversation". Quote both replies.
4. Per-token host overhead at 1 and 8 conversations: `s01-tokens/summary-all.wl` (whole receipt).
