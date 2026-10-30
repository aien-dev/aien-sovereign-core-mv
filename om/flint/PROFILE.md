# Flint frozen host-forward profile

Status: IMPLEMENTED and EXERCISED on Spark host CPU only, 2026-10-29. Not a device backend.
Base: sovereign-core fc3b8626d724ecf9ffb21d573592fc2ea5f44974 (#805).
Main fetched during this work: 65e712b7dc58d15ba1f9284e66e2da507a70c380.
Checkpoint: RecursiveSI/flint@2a247be0c0aec895ccb0ac7ca92229ec6df37288, Apache-2.0.

## Reproduction

```
taskset -c 0-4 nice -n 10 ionice -c3 env \
  OMC=/path/to/omega-mv/seed/out/omc \
  FLINT_SRC=/path/to/pinned-source-capture \
  FLINT_SHARDS=/path/to/downloaded-shards \
  FLINT_PY="env PYTHONPATH=/path/to/transformers-a672 /path/to/python" \
  FLINT_FORWARD=1 bash om/flint/check.sh
```

Compiler source omega-mv bf54b19f; binary SHA256
`62775d2a391cfef7cde64e387655caa9809a267093112910ec76bfa568f3712b`.
This is the existing #805 test compiler, not a production compiler-pin change.
Reference runtime: Python 3.12.3, torch 2.14.0+cu130, transformers 5.21.0.dev0
at a672eb5f474b8b30173e9f96eab66adf292da635, safetensors 0.8.0, numpy 2.5.3.
The config's recorded transformers_version is 5.12.1; this is distinct from the
pinned runtime used for current Flint support. Referee runs single-threaded eager
attention/experts on CPU with deterministic algorithms.

## Frozen identities (SHA256)

| Component | SHA256 |
|---|---|
| config.json | a57c33d11f725099105b657043c113149763066d204fd472d0b2ee399abc2ba7 |
| model.safetensors.index.json | e5e9c492685f84c8d2d8369f36c74674b03410715f23f6179313f6f1feb90311 |
| tokenizer.json | a263cce0b7e2b03f7ee07ce809783ef4b47f160c3f268a4ea1524f638da45b68 |
| tokenizer_config.json | 159ac88217e90ffd0a4a294726048bb302829e82e1958dc85bb203aa79138fbf |
| chat_template.jinja | 7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466 |
| generation_config.json | a499c71895e008abd572558a5ad5904feaed92cf2a3b6e2cb0e9cabf03b96ed3 |
| modeling_flint.py at runtime | f0540832049f0d67eee9eadd2d8ed42e320388b6b3330e74c8821a5c4eac0c09 |
| configuration_flint.py at runtime | 849eee3068846081077a7568663355802c8c4eec3741d92cd578d222bd144d00 |
| conversion_mapping.py | 93a992136d09ac875341cd645294f30d133af46efe073430c85fc20731240834 |
| masking_utils.py | 03912b9f8910358966351495075895ef0e8f5048f15226271d700ea268fef08c |
| transformers source archive | 5d82e0de4bf993484299bcc1f2ef78d1d3f4ff23cee411f42f486b774f5f6615 |
| admitted tensor inventory TSV | 60ff4dbdaa1c3632c934c6935aa60ed5b2def4e15f0ca5083ddc7dd660f083f5 |
| chained forward fixture | b5e4bae28af3c1fa0c8f7454c496e72da1388e17f51bb3ec9e9b0387aac97031 |

Per-shard hashes are in `PROFILE-SHARDS.sha256`; all 15 downloaded shards were
rehashed and matched the source capture before this work. Storage is BF16, with
no quantization descriptor/scales in this checkpoint. No lossy repack was applied.
Generation defaults in the pinned capture: do_sample=true, temperature=1.0,
top_p=0.95, top_k=20. The forward referee does not sample: token IDs [1001,8920]
are fixed, not produced by a chat template. The tokenizer/template are frozen
for future task workloads but not exercised by this tensor-only fixture.

## Architecture audit

Line numbers refer to pinned config.json and modeling_flint.py, respectively.

| Requested property | Observed evidence | Verdict |
|---|---|---|
| 49 layers, 2 dense + 47 MoE | config:73-82 mlp_only_layers=[0,1], num_hidden_layers=49 | Matches |
| 256 experts, top-6 sigmoid | config:80-81,105; reference:90-101 | Matches |
| Selection-only expert bias | reference:97 explicitly says without expert bias; gate.weight only, no gate bias in inventory | **Mismatch: no expert bias** |
| Ungated shared expert | reference:149,155-157 | Matches |
| Learned attention sinks | config:108; reference:225-227, checkpoint softmax_offset tensors | Matches |
| 35 local layers, window 513 | config:21-70,107 | Matches |
| 14 full layers, no RoPE | config:84-99; reference:280-283 | Matches |

Wolfram query for counts and modeled sizes is recorded in test/TOLERANCE.md.
Reference behavior, not the inaccurate bias claim, is implemented. Native storage
remains BF16; compute f64 vs reference f32 is an explicit numerical-profile change.
No BF16-activation, GPU, sampled-token, long-context, batching or task-quality
qualification follows from this candidate forward result.

## Existing GPU contracts

This bounded CPU verifier is not a second GPU runtime, cache, quantization schema
or MoE scheduler. #156's descriptor remains the GPU format authority; #159 remains
the native grouped-MoE authority; #157 composition and closed #158 cache identities
are unchanged and not exercised by this host verifier. Their GPU integration,
native encoded dispatch and physical receipts remain NOT_RUN, not implicitly reused
by a claim of host tensor parity.

## Host-loop profiling and numerical invariants

The initial native CPU profile used `monons` wrappers in an isolated compilation
unit stored with the operator receipts, not in the product source. Timers aggregate
whole calls, not per-element syscalls. They include instrumentation overhead:

| Loop/helper | Accumulated ns | Calls |
|---|---:|---:|
| bounded dot product `dotr` | 30899920888 | 3201792 |
| tensor read/decode `ld_bytes` | 10814711443 | 2478 |
| inventory search `tsv_find` | 8860325941 | 2478 |
| integration comparisons `fw_cmp` | 88531129 | 300535 |
| MLP activation `mlp_h` | 55114892 | 613 |
| RMSNorm `rmsv` | 11444826 | 3726 |

These are instrumented call totals, not disjoint percentages of uninstrumented
wall time. The measured hottest loop was `dotr`. The forward-only `matdot` retains
the exact f64 conversions, products and reduction order but omits the two per-product
error sums that the forward integration comparison does not consume. Those slots
are cleared, not left stale. The forward integration budget and every comparison
are unchanged. Router dots retain the bounded implementation for tie reporting;
the existing per-operator checks retain the original bound accumulation.

`worst_ppm` is the maximum `|native-reference|/(2E)` expressed in millionths.
WolframLanguageEvaluator: `N[694277/1000000*100,6] -> 69.4277`.
Thus the observed `694277` means approximately **69.4% of the comparison budget**
was consumed; it does not mean a 69.4% relative tensor error.

## Load memory: peak backing, mmap residency, duplicate buffers (#834 item 4)

Measured 2026-10-30 on the Spark host CPU (`taskset -c 0-4 nice -n 19 ionice -c3`),
sovereign-core base 85a18581, omc `62775d2a…712b`, ops binary sha256
`29cc5286276f6e4cf35cf8721b05a91e4c9534586910cdc2dbd718e4ab6f5cc9`, inputs the pinned
TSV `60ff4dbd…f5` and forward fixture `b5e4bae2…7031`. The omc-built binary and the Python
referee were run as separate processes, each under `test/memprobe.sh` (GNU `time -v` plus
samples of `/proc/<pid>/status`, `smaps_rollup`, `io` and `maps` every 0.1 s):

```
bash om/flint/test/memprobe.sh OUT -- ./ops forward forward.fx real.tsv SHARD_DIR
bash om/flint/test/memprobe.sh OUT -- env PYTHONPATH=… python om/flint/test/fwdref.py SRC SHARD_DIR forward.fx
```

| Process | Wall | Max RSS (`time -v`) | Major faults |
|---|---:|---:|---:|
| native forward (`ops forward`, 300 samples) | 38.75 s | 60480 kB (VmHWM 60620 kB) | 0 |
| Python referee (`fwdref.py`) | 22.49 s | 7319656 kB | 1 |

Native forward, per sample: VmPeak stays 160832 kB, the static image (the RW segment is
0x9cd1910 bytes, `N[FromDigits["9cd1910",16]/1024] -> 160582.27` kB, plus text and stack);
there is no heap. RssFile stays 88 kB (the binary's own text), RssShmem 0, and `maps` has
no file mapping other than the binary in any sample: shard residency through mmap is 0. The
shards are read with `pread` (rchar 7427081605 bytes, `N[7427081605/2^30] -> 6.917` GiB
for the two tokens; `2*150144*2048*2 -> 1229979648` of it is lm_head); that data sits in
the shared kernel page cache, which belongs to no process and is not counted above.
VmHWM reaches 60144 kB at 0.40 s (layer 0) and 60620 kB at 1.69 s and stays flat until
the last sample at 38.72 s, including the lm_head stream.

Duplicate buffers: `ld_bytes` preads at most 1048576 bytes of BF16 into `gstg`, widens
them into the f32 buffer `gw`, and reuses the chunk; `ld_rows` loads the embedding and
lm_head in 2048-row slices. So no full BF16 tensor is ever resident; the one transient
duplicate is the 1 MiB staging chunk beside the f32 repack. The peak is accounted for by
the largest repacked tensor, the dense MLP 6144x2048 in f32, plus the inputs
(WolframLanguageEvaluator: `6144*2048*4/1024 -> 49152`, `N[5248040/1024] -> 5125.04`
fixture, `N[4269125/1024] -> 4169.07` TSV; `49152+1024+5125.04+4169.07+88 -> 59558.11` kB,
residual `60620-59558.11 -> 1061.89` kB of activations and stack). A co-resident full BF16
copy of that tensor would add `6144*2048*2/1024 -> 24576` kB (`60620+24576 -> 85196`).
The referee's footprint is `N[7319656/60480] -> 121.03` times the native one.

Verdict: no full-BF16-plus-repack duplicate exists, so the loader was not changed. The
gate freezes the bound: with `FLINT_FORWARD=1`, `check.sh` runs the forward under GNU
`time -f %M` and requires max RSS <= 65536 kB (margin `65536-60480 -> 5056` kB), which a
co-resident BF16 copy of any tensor over 5056 kB exceeds. The f32 repack itself
(2x the BF16 bytes of the largest tensor) is the remaining cost; computing directly from
BF16 chunks would remove it and is not done here.

Negative controls, both with the same inputs:

- Bound below the measured peak: the full gate with `FWD_RSS_KB=60000` (temporary copy of
  `check.sh`) printed `FAIL forward peak resident '60160' kB over frozen bound 60000 kB` and
  `OM-FLINT-MANIFEST FAIL (1 of 93)`; every other check passed.
- Real duplicate (`retain-bf16`, now a gate mutant): a 16777216-element ushort global
  (`16777216*2/1024 -> 32768` kB virtual) written beside every f32 widening keeps a full
  BF16 copy of each loaded tensor. Standalone run: forward still PASS (worst_ppm 694277),
  max RSS 84960 kB, `84960-60480 -> 24480` kB above the clean binary, which is the predicted
  BF16 copy of the dense MLP tensor within `24480-24576 -> -96` kB; `84960-65536 -> 19424` kB
  over the bound.

