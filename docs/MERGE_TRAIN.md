# The merge train

Drake, 2026-10-21: "batch the tests ... make this the standard for doing these from now on. Call it the merge train."
Ledger L384. Every lever on the 35B serving path is tested and merged on the train. No lever loads its own full-model
copy for its own gate.

One cycle is: **one candidate build** holding every READY lever, **one 35B load** (two at most, see load-only switches),
**one shared suite** run against **stored references**, and **one verdict per lever**. A failing lever is switched off and
the rest ride on.

Files:

| what | where |
|---|---|
| the command | `evidence/merge-train/merge-train` (bash + jq + curl) |
| the suite (data) | `evidence/merge-train/suite.json` |
| run folders, verdict tables | `evidence/merge-train/runs/<UTC stamp>-<candidate sha8>/` |
| the reference store | `~/.local/state/aien-q36/merge-train/refs/<model dir>/<reference build sha8>/alone/`, `CURRENT` symlink |
| host goldens (no load needed) | `evidence/q36-host-20261020/gen-*.json` |
| per-request switches | cargo feature `aien-test-switches` in `crates/aien-serve-http` |

## 1. Per-request switches (`aien_test`)

A test build (`cargo build --release -p aien-serve-http --features aien-test-switches`) reads an extra request field:

```json
"aien_test": {"stroke": false, "exact_head": false, "mutant": 4, "row_swap": 2}
```

| switch | effect, this request only | default |
|---|---|---|
| `stroke` | `false`: this row steps on its own one-row graph, never in a stroke with other rows | `true` |
| `exact_head` | `false`: the plain one-pass head instead of the exact two-pass head | `true` |
| `mutant` | graph mutant `k` (the `AIEN_Q36_TEST_MUTANT` numbers) for this row's steps | `0` |
| `row_swap` | at this row's `k`-th step the scheduler trades rows with another decoding conversation (both go red) | off |
| `prefill_chunk`, `prefix_pool`, `state_bf16` | names reserved for L369, L375, L313; refused until their lever wires them | |

Honesty rules, enforced in code:

- A build **without** the feature (the live build) answers **400** to any request carrying `aien_test`. It can never serve a test case.
- A test build answers **400** for a switch its engine does not honor. `/health` lists the honored switches under `aien_test.honored`.
- Every reply that set switches echoes them in `usage.aien_test.applied`. The train records ERROR, never PASS, when the echo is missing.
- Rows without switches take the served path unchanged (the strokes, the exact head). A switched row steps alone that beat. The other rows' words are unchanged, and the suite checks that (`mutant-row-isolated`).

**Load-only switches** (a setting read once at start, such as `AIEN_OMEGA_GDN_STATE_BF16=1` or `AIEN_BOARD_DIR`) are a suite
*arm* with `server_env`. The train loads a second server for them after the first one stops, only when a lever of that arm is in the candidate, and never more than `MERGE_TRAIN_MAX_LOADS` (default 2) loads per cycle. An arm past the cap is SKIP and rides the next cycle. The ready clock (`MERGE_TRAIN_READY_S`, default 1800 s) counts loading only, not time queued in q36-extra.
An arm can name `log_proof`, a word the server log must show, so a build that ignored the setting is marked NOT-ON-TRAIN, not PASS.

## 2. The suite format (`suite.json`)

The suite is data. A lever joins by adding JSON, not code.

- `arms`: named ways to run the suite. `aien_test` arms are request switches on the one server. `server_env` arms are a second server. `enabled: false` keeps an arm off until its lever is in the candidate.
- `levers`: `id`, `name`, `arms` (which arms test it), and `present` (how the train tells whether the lever is in this candidate at all):
  `{"always": true}`, `{"switch": "<aien_test name>"}` (in `/health` honored list), `{"health": ".path"}` (truthy in `/health`), `{"model": "<regex>"}` (a served model id), `{"build": "<regex>"}` (matches the bin's `BUILD.txt`).
- `cases`: each case has an `id`, a `kind`, and optional `arms` (default: every arm) and `levers` (default: every lever whose arms include the arm).

| kind | requests | PASS when |
|---|---|---|
| `golden` | one chat request (`user`, `max_tokens`) | token ids == the host golden's `generated_ids` |
| `together` | the first `n` prompts of `prompts.txt` at once, `stagger_ms` apart in groups of 8, `max_tokens` by `ntok(i)` | every reply == the same request alone (reference store) |
| `turn2` | the stored turn-2 messages | ids == the stored turn 2 |
| `tool` | one request with `tools` | the reply is a parsed tool call named `expect_tool` |
| `script` | the owner's own gate script, run against the train's server (`$PORT`, `$OUT`, `$SERVER_LOG` substituted in `args`) | the script exits 0 (1 FAIL, other: ERROR) |

Red controls (the proof that a comparison can fail): `"expect": "red"` on a `golden` case, or `expect_red` (row list),
`expect_red_min`, `others_red_max` on a `together` case with `aien_test_on` (`{"<row>": {switches}}`). A red control that comes
back identical is **RED-FAIL** and fails every lever it covers.

## 3. The reference store

References are the token ids of every suite prompt served **alone** (one request at a time) plus turn 2, computed **once** per
(model, reference build) and kept under `~/.local/state/aien-q36/merge-train/refs/<model dir>/<sha8>/alone/`. `CURRENT` points
at the one in use. No gate re-runs the reference side unless the reference build changes.

```
merge-train ref <port> <sha8>              # compute from the reference build serving on <port>
merge-train ref-import <alone-dir> <sha8>  # import an existing gate's alone/ folder
merge-train ref-use <sha8>                 # switch CURRENT
```

Goldens (`evidence/q36-host-20261020/`) come from the host reference model and need no load at all.
Today's reference: `5b726818`, imported from L354's `evidence/q36-batch-20261021/alone/` (the single door, 02:42Z).
L354's batch gates on bin-16b0e6f8 matched it 32/32.

## 4. The cycle

1. **Join.** A lever owner pushes the lever branch, posts a READY line on `~/handoffs/BOARD.md`, and adds the lever's cases to `suite.json` by commit on `serve/merge-train` (or by ledger note on L384, and L384 adds them). A new switch goes in `TestSwitches` (`crates/aien-serve-http/src/engine.rs`) and in the engine's honored list, with a host test that the switch changes only its own row.
2. **Build.** L354 (the serving lane) merges every READY lever branch onto the ship branch with this branch, builds once with `--features aien-test-switches`, and pins `~/.local/state/aien-q36/bin-<sha8>/` (binary, lib, `BUILD.txt` naming every lever branch and sha, `SHA256SUMS`).
3. **Run.** `merge-train run ~/.local/state/aien-q36/bin-<sha8> --post`. The train loads the candidate under `q36-extra` (`Q36_EXTRA_LEDGER=L384`, port 18100 by default, never 18091), runs every request arm, stops the server, loads a second server only for enabled load-only arms, and writes `runs/<stamp>-<sha8>/verdict.md` (lever x case table, then every cell) and `receipt.json` (loads, load seconds, suite seconds).
4. **Post.** `--post` appends one BOARD line with every lever's verdict and adds a `ledger note` to each lever's item.
5. **Ship.** The live build is the same merge **without** the feature. It is built from the same commit and gated by the cells that passed.

## 5. When a lever fails

- **FAIL or RED-FAIL** in a lever's row: the lever is dropped from this cycle. If the lever has a switch, its arm stays in the suite and the candidate ships with the switch's served default set to off (the lever's own env switch, such as `AIEN_Q36_STROKE=0`). If not, L354 rebuilds the candidate without that branch. The rest of the levers ride on, and the dropped lever's owner gets the cell folder in the ledger note.
- **ERROR** is not a verdict (no reply, a 400, a missing reference). Fix the cause and rerun. A lever is never merged on ERROR.
- **NOT-ON-TRAIN**: the lever or its switch is not in this candidate. That is not a failure. It just did not ride this cycle.
- A failure that touches the shared cases (goldens, `concurrent-32`, `turn2`) fails every lever on that arm. Bisect by arm first: if `plain` fails and `stroke-off` passes, the stroke lever is the suspect.

## 6. Rules

- Extra 35B loads only through `~/.local/bin/q36-extra`. Never port 18091 (the live server), 18092 or 18095.
- The train's one server replaces each lever's own test copy. A lever owner does not load a copy for a gate the train covers.
- Owners keep their deep evidence (speed receipts, bit-identity at the model level) in their own `evidence/`. The train is the merge gate.
- bash + jq + curl only. No Python.
