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
are fixed, not produced by a chat template. Tokenizer and control-token identity
(ids, special flags, bytes, ByteLevel decode) and the encode of ASCII text are
exercised on host by `tok.om`; encode of non-ASCII text (refused, 60) and the chat
template remain NOT_RUN.

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
