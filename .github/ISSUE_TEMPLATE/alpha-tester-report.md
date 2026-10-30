---
name: Alpha tester report (Qwen3.8-27B model server on a DGX Spark)
about: Tell us how the model server behaved on your Spark, whether it worked or not
title: "[ALPHA] "
labels: ["alpha-tester"]
assignees: []
---

Thank you for trying it. Reports that say "it worked" matter as much as failures: your
receipts are what turn "it works on our Spark" into "it works on a Spark". The guide is
docs/qwen38/TRY_IT_ON_YOUR_SPARK.md; section 8 explains each item below.

## Result

- [ ] It worked: I got a chat reply (section 5)
- [ ] It worked and I ran the checks (section 6)
- [ ] It stopped at step ___ of the guide

Exact message where it stopped (copy it, do not retype):

```
```

## The commit you built

Output of `git -C ~/aien-sovereign-core-mv rev-parse HEAD`:

```
```

## The server build

Output of `aien-serve-http --version` (all lines):

```
```

## Driver and system

Output of `cat /proc/driver/nvidia/version`, `uname -a` and `grep PRETTY_NAME /etc/os-release`:

```
```

## Free memory

Output of `free -g` (while the server is running, if it got that far):

```
```

## The installer's machine check

The lines from `[*] Model server: checking this machine` to
`[+] Model server: machine checks passed` (or the error) in `~/aien-install.log`:

```
```

## Receipts

Attach `~/aien-check/gate/receipt.json` and `~/aien-check/speed/receipt.json` as files if you
got to section 6. Paste the summary lines here:

```
jq '{verdict, golden_ids, binary: .server.binary_sha256, build: .server.build}' ~/aien-check/gate/receipt.json
jq '{decode_ids, decode_median_chip_ms_per_token, decode_tokens_per_s, share_of_ceiling}' ~/aien-check/speed/receipt.json
```

```
```

## Server log

The last 50 lines of `~/.local/state/aien/serve-27b.log`, if the server started:

```
```

## Anything else

What you expected, what you saw, anything unusual about your machine (other GPU jobs running,
a different driver, less memory). Please do not paste model files, your whole install log, or
anything from outside your AIEN folders.
