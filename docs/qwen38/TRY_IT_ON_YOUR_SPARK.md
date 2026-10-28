# Try it on your DGX Spark: the Qwen3.8-27B model server

This page takes you from a DGX Spark with nothing of AIEN on it to a chat reply from
Qwen3.8-27B, served by AIEN's own Omega engine, using only public sources. It also shows how to
run the same correctness and speed checks we run, so you get your own receipt to compare with ours.

What you get: an HTTP server on your Spark that speaks the OpenAI chat-completions format, text
only, one request at a time. What it is not: a finished product. Read the known limits in
section 7 before you depend on it, and [ALPHA_STATUS.md](ALPHA_STATUS.md) for what "alpha" covers
and what stays pre-alpha.

This page describes the branch it sits on. If `./install.sh --help` in your checkout does not list
`--with-server`, your checkout is older than this page; use the branch or release this page came
from.

## 1. What you need

Hardware and system (the only setup tested):

- An NVIDIA DGX Spark (GB10 chip, aarch64 Linux, 128 GB of unified memory). The installer
  refuses anything that is not aarch64 Linux; no other machine has been tried.
- The NVIDIA open kernel driver, version 580.173.02. This is the version whose headers the
  engine was built against (`AIEN_EXPECTED_NVIDIA_DRIVER` in `install.sh`). The installer warns
  on any other version and the engine may fail to start or give wrong results there.
- Our Spark runs Ubuntu 24.04.5 LTS with kernel `7.0.0-1019-nvidia`, read from
  `/etc/os-release` and `uname -r` on 2026-10-19. Nothing else has been tried.
- A normal user account that can read and write `/dev/nvidiactl`, `/dev/nvidia0` and
  `/dev/nvidia-uvm`. On our Spark these device files are readable and writable by every user
  (mode 666). No root is needed except for `apt`.
- Free memory: 49,497 MiB (48.3 GiB) must be available when the server loads. The loader needs
  41,305 MiB for the model and its working memory (at the default context of 131,072 tokens)
  plus 8,192 MiB it keeps free for other programs, and it refuses to load with less. The installer
  and the start script warn below 49,497 MiB (`install.sh`, `tools/serve-27b.sh`). The loader
  judges memory again after it gets the GPU lock, because the wait for the lock can last many
  minutes and free memory changes meanwhile. A running server holds less than the loader asks for:
  the live server on the current build (binary sha256 `df936011...`) peaked at 38,499,160 kB of
  resident memory, 38,499,160 / 1,048,576 = 36.7 GiB (calc:39181948), read from
  `/proc/<pid>/status` (VmHWM) after 2 hours 48 minutes of service with up to 131,072 tokens of
  context allowed: `evidence/qwen38-serve-memory-20261019/proc-status.txt`. Earlier builds peaked
  at 40,358,896 kB, 38.5 GiB (calc:f6c02d78, `evidence/qwen38-serve-fast-20261019/README.md`), and
  40,368,140 kB on the fresh-account test of 2026-10-19. Close other large programs before you
  start the server.
- Disk: 30.9 GB for the model (30,890,049,597 bytes in 81 files, see
  [README.md](README.md)); `install.sh --with-model` asks for that plus 2 GB of room,
  (30,890,049,597 + 2,000,000,000) / 1,000,000,000 = 32.9 GB (calc:8af38ffd), less whatever an
  earlier run already put in the folder. The sources and the build took under 1.5 GB on our Spark on
  2026-10-19 (clone 321 MB, pinned Omega sources 386 MB, release build 561 MB, measured with
  `du`), plus the Rust toolchain and its download cache (1.6 GB on ours, shared with other
  builds). Have 40 GB free and you are safe.
- Network access to github.com (sources) and huggingface.co (model).

Programs:

- `git`, `curl`, a C compiler, `make`, `ar`, `nm`. On DGX OS:
  `sudo apt install git curl build-essential binutils`. The installer checks for these first.
- `jq` for the check tools in section 6: `sudo apt install jq`.
- Rust 1.98.1, pinned in `rust-toolchain.toml`. If `cargo` is missing the installer installs
  rustup and rustup fetches that version on first use.
- Not needed: the CUDA toolkit, `nvcc`, Python, Docker. The Omega engine talks to the kernel
  driver directly.

Time: the build and the first run have not been timed on a clean account yet (see
[ALPHA_STATUS.md](ALPHA_STATUS.md)). Loading the model into the server took 64 seconds on our
Spark (`evidence/qwen38-serve-fast-20261019/README.md`, "load 64 s").

## 2. Install

```bash
cd ~
git clone https://github.com/aien-dev/aien-sovereign-core-mv
cd aien-sovereign-core-mv
./install.sh --with-server --with-model 2>&1 | tee ~/aien-install.log
```

`--with-model` also downloads and checks the model (section 3). Leave it out if you want to
fetch the model yourself.

For the canonical CUDApp source-cutover candidate, export `OMEGA_OMC` and
`OMEGA_BRIDGE` pointing to the compiler and bridge matching canonical CUDApp's
`om/units.sh` pins before invoking the installer. Missing selections are refused.
This candidate's host source/link checks do not constitute GPU qualification.

Keep `~/aien-install.log`: it is one of the things to paste when you report a problem.

What the installer does with `--with-server --with-model`, in order:

1. Checks the machine (the block that starts with `[*] Model server: checking this machine`):
   Linux on aarch64, the six programs above, the NVIDIA driver version, the three device files,
   free memory. Then free disk for the model (the block that starts with
   `[*] Model download: checking`). A failed check stops the installer before anything is
   cloned, built or downloaded and says what is wrong in plain words.
2. Fetches pinned Omega, canonical CUDApp, physics and aienos sources into `target/pins/`
   (`scripts/pins.sh --gpu`). Core `omega.lock` pins Omega; that committed revision's
   `cudapp.lock` pins CUDApp, alongside its existing physics and aienos locks.
3. Regenerates/checks canonical units with the pinned toolchain, builds the canonical
   `libcudapp.a` and retained Omega `libomega_gpu.a` companion, and links the server
   (`cargo build --locked --release -p aien-serve-http`). Both archive digests must match
   `native-artifacts.lock`; headers or caller-selected archives cannot enable native mode.
4. Runs `aien-serve-http --version` and refuses to install a binary that has only the CPU stub
   instead of the real engine.
5. Installs `aien-serve-http` and the start script `aien-serve-27b` into `~/.local/bin`, along
   with the repository's other command line tools (those tools are not part of the alpha; you do
   not need them for the server).
6. Downloads the model into `~/models/qwen3.8-27b-fp8-017b9c7a` and checks every file
   (section 3).

The installer also appends a line to your `~/.bashrc`, `~/.zshrc` or `~/.profile` so that
`~/.local/bin` is on your `PATH`. Open a new shell afterwards, or run
`export PATH="$HOME/.local/bin:$PATH"` in the current one. Set `AIEN_INSTALL_NO_PROFILE=1` if
you do not want your shell files touched.

Check the result:

```bash
aien-serve-http --version
```

The two lines that matter must say `native`:

```
omega gpu engine: native (...)
qwen38 engine: native (...)
```

If either says `STUB`, the engine is not linked and the server will not work; the installer should
have refused this, so please report it (section 8).

## 3. Download the model and check it

The weights are the public Hugging Face model `Qwen/Qwen3.8-27B-FP8` at revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, Apache 2.0 licence, not gated, no account needed.
The server reads that folder directly; no conversion step.

With `--with-model` (section 2) the installer does all of this section for you:

- It fetches the 81 files named in the shipped checksum list
  (`docs/qwen38/qwen3.8-27b-fp8-017b9c7a.sha256`) one by one with `curl` from
  `https://huggingface.co/Qwen/Qwen3.8-27B-FP8/resolve/<revision>/<file>`, the same place our
  own copy came from.
- A file that is being downloaded is called `<file>.part`. If the download is interrupted, run
  the same install command again: files already complete are kept and each `.part` file carries
  on where it stopped.
- At the end it runs `sha256sum -c` on all 81 files. If any file does not match, the installer
  stops with an error, renames the bad file to `<file>.mismatch` and tells you to rerun it, which
  fetches that file again. It never reports a folder that failed the check as ready.
- The folder is `~/models/qwen3.8-27b-fp8-017b9c7a`; choose another with `--model-dir=DIR` (then
  start the server with `MODEL_DIR=DIR aien-serve-27b`).

To fetch or re-check only the model later, run `./install.sh --with-model` again from the
checkout (it also runs the command line tools build again).

What we ran: on our Spark, `install.sh --model-dir=...` with real downloads from huggingface.co
into a folder missing two small files and holding a third as a partial download fetched 2 files,
resumed the third, and `sha256sum -c` passed for all 81 files; a second run fetched nothing and
passed again (`evidence/qwen38-install-with-model-20261019/realrun.log`). The full 31 GB download
from an empty folder has not been timed. `scripts/test-install.sh` checks resuming, the refusal
on a mismatch and the disk check against a small local stand-in folder.

By hand, if you prefer: this loop does the same download, taking the 81 file names from the
shipped checksum list, and skips files that are already there:

```bash
REV=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
LIST=~/aien-sovereign-core-mv/docs/qwen38/qwen3.8-27b-fp8-017b9c7a.sha256
mkdir -p ~/models/qwen3.8-27b-fp8-017b9c7a && cd ~/models/qwen3.8-27b-fp8-017b9c7a
awk '{print $2}' "$LIST" | while read -r f; do
  [ -s "$f" ] && continue
  curl -fL --retry 5 -o "$f" "https://huggingface.co/Qwen/Qwen3.8-27B-FP8/resolve/$REV/$f" \
    || echo "FAILED $f"
done
```

If you already use the Hugging Face command line tool, this fetches the same revision into the
same folder (we have not run this command ourselves; the loop above is what we ran):

```bash
hf download Qwen/Qwen3.8-27B-FP8 --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --local-dir ~/models/qwen3.8-27b-fp8-017b9c7a
```

Then check every file against the list shipped in this repository (the installer's
`--with-model` already did this; it is safe to run again):

```bash
cd ~/models/qwen3.8-27b-fp8-017b9c7a
sha256sum -c ~/aien-sovereign-core-mv/docs/qwen38/qwen3.8-27b-fp8-017b9c7a.sha256
```

Every line must end in `OK` (81 lines). For the 67 large files the expected values are the ones
Hugging Face publishes for this revision; for the 14 small files they are what we downloaded,
with sizes matching upstream (details in [README.md](README.md)). If a file fails, delete it and
run the download loop again. Do not start the server on a folder that fails this check.

The folder name matters: `aien-serve-27b` looks for `~/models/qwen3.8-27b-fp8-017b9c7a` unless
you set `MODEL_DIR`.

## 4. Start the server

```bash
aien-serve-27b
```

The script checks that the binary has the real engine, that the model folder exists, that the
driver's device files are there, that the port is free and that no earlier copy is running; then
it starts the server in the background with the same flags our own live server uses
(`--engine omega --port 18091 --max-context 131072 --max-tokens-cap 16384 --thinking off`), then
it waits for the model to load. When the server is serving it prints `ready`, the log path,
the stop command and a first chat command, and exits 0. If the server exits instead (for example
the loader refuses because less than 49,497 MiB is free), the script prints the reason from the
log and exits 1; no server is running then. If the model is still loading after 40 minutes
(`READY_TIMEOUT`, in seconds) it exits 2 and leaves the server running. The loader first queues
for the shared GPU lock, so on a busy Spark the wait can be minutes (6.5 minutes in our test).
`aien-serve-27b --no-wait` returns at once and says nothing about the load. The server listens
on `127.0.0.1` only.

You can check by hand that it is ready; this returns the model:

```bash
curl -s http://127.0.0.1:18091/v1/models
```

```json
{"object":"list","data":[{"id":"qwen3.8-27b-fp8","object":"model","created":...,"owned_by":"aien/omega"}]}
```

The log is `~/.local/state/aien/serve-27b.log`; the line `aien-serve-http: qwen38 engine ready in
N s` marks the moment it is ready. `curl -s http://127.0.0.1:18091/health` reports `status`,
`model`, `engine`, `busy` and `uptime_s`.

Stop it with `kill -TERM $(cat ~/.local/state/aien/serve-27b.pid)`. It finishes the reply in
progress first, so stopping can take a few minutes. Nothing restarts the server after a reboot;
run `aien-serve-27b` again. `aien-serve-27b --help` lists the settings (`MODEL_DIR`, `PORT`,
`BIN`, `STATE_DIR`, `LOG`) and `--foreground` for use under your own service manager.

## 5. First chat request

```bash
curl -s http://127.0.0.1:18091/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-fp8",
    "messages": [{"role": "user", "content": "In two sentences, what is a DGX Spark?"}],
    "max_tokens": 128,
    "temperature": 0
  }'
```

The reply is standard chat-completions JSON: the text is in `choices[0].message.content`,
`finish_reason` is `stop` (the model ended) or `length` (`max_tokens` reached), and `usage`
gives prompt and completion token counts. Things to know:

- `model` is optional; if you send it, it must be `qwen3.8-27b-fp8` (anything else is a 404).
- `max_tokens` defaults to 1024 and is capped at 16384 by the start script. `temperature`
  defaults to 0.7; `0` gives the same answer every time for the same request, which is what the
  checks in section 6 rely on.
- `"stream": true` streams the reply as server-sent events (`data: {...}` lines, then
  `data: [DONE]`). Closing the connection stops generation.
- One request at a time. A second request while one is running gets HTTP 503 with
  `{"error":{"type":"server_busy",...}}` and a `Retry-After` header. Wait and retry.
- Thinking is off (the start script sets `--thinking off`). A request can turn it on with
  `"chat_template_kwargs": {"enable_thinking": true}`; our figures were all measured with it off.
- Requests with `tools` or `functions` are refused with 400. Text only.

Any OpenAI-compatible chat client pointed at `http://127.0.0.1:18091/v1` with model
`qwen3.8-27b-fp8` and no tool calling should work; we have only exercised `curl` and the scripts
in this repository.

## 6. Check it yourself

Two scripts in `tools/` send fixed requests to a running server and write a `receipt.json` you
can compare with ours and attach to a report. Run them from the repository folder with the
server started by `aien-serve-27b` (they read its log for the per-token timings). They need
`curl`, `jq`, `sha256sum`, `awk` and (speed only) `iconv`.

Correctness gate:

```bash
cd ~/aien-sovereign-core-mv
tools/qwen38_serve_gate.sh http://127.0.0.1:18091 "" ~/aien-check/gate
```

It sends ten short requests (24 tokens at most each) and checks that the greedy answer to the
reference prompt is the same 24 token ids our host reference produced
(`crates/aien-inference-abi/fixtures/qwen38/gen-chat-max-24.json`, provenance in
`PROVENANCE.md` next to it), that a different question gives different ids (the negative
control), and that a follow-up turn gives the same ids whether the server reuses its state or
starts cold. The last lines must read `GATE PASS` and
`golden ids against the shipped reference: MATCH`. Ours: `GATE PASS`, 24 of 24
(`evidence/qwen38-serve-fast-20261019/gate-live/summary.txt`).

Speed:

```bash
tools/qwen38_serve_speed.sh http://127.0.0.1:18091 "" ~/aien-check/speed
```

It sends three times the same 30-token prompt and reads 256 greedy tokens back, then a prompt of
about 1,700 tokens, one of about 30,000 tokens and a follow-up to that one. The 30,000-token
request takes much longer than the others; let it finish. The two lines to look at:

```
decode ids against the shipped reference: MATCH
median chip ms per token (3 decodes): <your number>
```

`MATCH` means your server wrote exactly the 256 tokens ours did
(`crates/aien-inference-abi/fixtures/qwen38/decode-lighthouses-256.json`). The per-token time
is the chip time from the server's own log line, not the wall time at the client, so other
programs waiting for the chip do not distort it.

What `receipt.json` contains (schema `aien-qwen38-serve-receipt/1`, written by
`tools/qwen38_receipt.sh`):

- `tools`: the commit of the checkout the script ran from, and whether it had uncommitted
  changes.
- `server`: the pid that answered, its binary path and sha256, and its build line from
  `aien-serve-http --version` (the sovereign-core commit it was built from).
- `model`: the folder the server was started with, the shipped checksum list and its sha256, and
  the result of `sha256sum -c` if you ran the script with `AIEN_CHECK_MODEL=1` (that reads the
  whole 31 GB again; otherwise it says `not run`).
- `machine.start` and `machine.end`: time, NVIDIA driver version, kernel, architecture, load
  average and available memory before and after the run.
- `runs`: one entry per request: token counts, client wall time, client ms per token, chip ms
  per token, server wall ms per token, how long the request waited for the chip, a 12-character
  hash of the returned token ids, and the server's log line for that reply.
- Gate only: `verdict` (`PASS` or `FAIL`), `golden_ids` (`MATCH` or `MISMATCH`) and the full
  `summary` lines. Speed only: `decode_ids`, `decode_median_chip_ms_per_token`,
  `decode_tokens_per_s`, `ceiling_tokens_per_s` (10.15) and `share_of_ceiling`.

The receipt contains your user name in file paths and your machine's load and memory figures,
nothing else about you.

How to compare with ours. Our numbers, each with the file it comes from:

| Ours | Value | Where |
|---|---|---|
| Decode speed | 7.63 tokens per second: 1000 / 131.05 ms, the median of four runs at 129.9, 129.0, 134.9 and 132.2 chip ms per token, 256 greedy tokens each | `evidence/qwen38-serve-fast-20261019/after/runs.tsv`; arithmetic in `calc-receipts/64f74f90...json` |
| Share of the memory ceiling | 0.752 of 10.15 tokens per second | [CEILING.md](CEILING.md), `calc-receipts/64f74f90...json` and `9822c2ba...json` |
| Token ids of the 256-token answer | hash `ceb68306ae2d`, identical in all four runs | `evidence/qwen38-serve-fast-20261019/after/runs.tsv` (`ids_sha`) and the fixture above |
| Correctness gate | `GATE PASS`, 24 of 24 greedy ids equal the host reference | `evidence/qwen38-serve-fast-20261019/gate-live/summary.txt` |
| Server that produced them | binary sha256 `df936011...`, sovereign-core `46134247`, omega `0010f7d3` | `evidence/qwen38-serve-fast-20261019/README.md` |

Print the same figures from your receipts:

```bash
jq '{verdict, golden_ids, binary: .server.binary_sha256, build: .server.build}' ~/aien-check/gate/receipt.json
jq '{decode_ids, decode_median_chip_ms_per_token, decode_tokens_per_s, share_of_ceiling,
     driver: .machine.start.nvidia_driver, load: .machine.start.loadavg_1_5_15}' ~/aien-check/speed/receipt.json
jq -r '.runs[] | "\(.name)\t\(.ids_sha256_12)\t\(.chip_ms_per_token) ms/token"' ~/aien-check/speed/receipt.json
```

What to expect: the token ids should match exactly (`MATCH`, and `ceb68306ae2d` for the three
decode runs) because greedy decoding is deterministic; a mismatch is a bug and we want to hear
about it. The chip ms per token should be close to ours on an idle Spark; our four runs were
taken while other jobs were using the machine (the `lock wait` figures in `runs.tsv` show
requests waiting up to 186 seconds for the chip), so a somewhat faster number on a quiet machine
would not surprise us. The ceiling arithmetic is written out in [CEILING.md](CEILING.md), and
`calc-receipts/` stores each expression and its exact result, so you can redo it by hand.

Our evidence folders were produced by an earlier version of these scripts, before they wrote
`receipt.json`, so there is no `receipt.json` of ours to diff against yet; the files in the table
are the raw outputs.

## 7. Known limits

Stated here so nobody is surprised; they are the same limits that sit next to the numbers above.

- One request at a time. The second gets a 503 and must retry.
- Text only. No images, audio or files.
- No tool calling. Requests with `tools` or `functions` are refused, so agents that need function
  calls cannot use this server without an adapter.
- Thinking is off in the shipped start script.
- Source build only. There is no aarch64 release archive and no signed public release; the public
  signing key does not exist yet. Expect a long compile.
- One tested setup: DGX Spark, Ubuntu 24.04.5, NVIDIA open driver 580.173.02. Other
  drivers, kernels or machines are untested.
- Context: the server starts with room for 131,072 tokens, but through the server we have sent
  prompts of about 30,000 tokens at most (on an earlier build:
  `evidence/qwen38-serve-20261019T0128Z/`). Longer prompts are untested and the first token on a
  long prompt takes a long time; first-token times for the current build have not been measured.
- Each reply is capped at 16,384 tokens by the start script.
- Memory: the loader needs 49,497 MiB (48.3 GiB) free; a running server peaked at about 36.7 GiB
  on the current build (section 1). A second 27B server shares nothing with the first and needs
  the full amount. Running
  anything else large on the Spark at the same time can get the server killed by the kernel.
- Nothing restarts the server after a reboot, and there is no authentication; it listens on
  `127.0.0.1` only. Do not expose the port.
- The speed figure is a small sample (four runs) on a busy machine. It is the chip time of the
  decoding loop; the time you see at the client includes reading the prompt and any waiting.
- The model download (`--with-model`, section 3) has been run on our Spark only for a few
  missing files plus the full check, not as a full 31 GB download into an empty folder.
- Everything else in this repository and in the AIEN project (the `aien` command line tools, the
  daemon, the other model paths, the AIENOS kernel, the Omega compiler, physics) stays pre-alpha.
  See [ALPHA_STATUS.md](ALPHA_STATUS.md).

## 8. How to report problems

Open an issue with the alpha tester template:
[github.com/aien-dev/aien-sovereign-core-mv/issues/new?template=alpha-tester-report.md](https://github.com/aien-dev/aien-sovereign-core-mv/issues/new?template=alpha-tester-report.md).
Reports that say "it worked" are as useful as failures: your receipts are what turn this page
from "it works on our Spark" into "it works on a Spark".

What to paste (the template has a slot for each):

- Where it stopped: the step number on this page, and the exact message.
- The commit: `git -C ~/aien-sovereign-core-mv rev-parse HEAD`.
- `aien-serve-http --version` (all lines).
- The driver and system: `cat /proc/driver/nvidia/version`, `uname -a`,
  `grep PRETTY_NAME /etc/os-release`.
- Memory: `free -g`, taken while the server is running if it got that far.
- The installer's machine check: the lines from `[*] Model server: checking this machine` to
  `[+] Model server: machine checks passed` (or the error) in `~/aien-install.log`.
- `~/aien-check/gate/receipt.json` and `~/aien-check/speed/receipt.json` if you got to section 6,
  attached as files.
- The last 50 lines of `~/.local/state/aien/serve-27b.log` if the server started.

Please do not paste the model files, your whole install log, or anything from outside your
AIEN folders.
