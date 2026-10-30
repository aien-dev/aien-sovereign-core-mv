# flint: pinned checkpoint manifest reader (Core #802 P0)

`manifest.om` is the first deliverable of Core #802 (aien-architecture-mv#353): a bounded
reader that admits the Flint BF16 checkpoint at one pinned revision and refuses everything
else. Memory, kernel, training and serving work read its output, the tensor inventory TSV,
instead of parsing safetensors themselves. It does not execute the model.

Pinned source: `RecursiveSI/flint` revision `2a247be0c0aec895ccb0ac7ca92229ec6df37288`,
Apache-2.0 (see `THIRD_PARTY.md`).

| File | sha256 |
|---|---|
| `config.json` (2451 bytes, copied to `test/config.json`) | `a57c33d11f725099105b657043c113149763066d204fd472d0b2ee399abc2ba7` |
| `model.safetensors.index.json` (3329405 bytes) | `e5e9c492685f84c8d2d8369f36c74674b03410715f23f6179313f6f1feb90311` |
| inventory TSV printed by the reader (36734 rows plus the header line) | `60ff4dbdaa1c3632c934c6935aa60ed5b2def4e15f0ca5083ddc7dd660f083f5` |

Build with omega-mv `bf54b19fa590e8ae925c0e80815516dbf41e7326`, seeded omc sha256
`62775d2a391cfef7cde64e387655caa9809a267093112910ec76bfa568f3712b`. `om/vendor/json.om` is
byte-identical to that revision's `seed/json/json.om` (sha256 `456254fc…e2d2`). This is a newer
compiler pin than `om/tur-ledger` (`6999fa9e`, omc `d6290c76…`).

```sh
{ printf 'om2\n'; cat om/vendor/json.om om/flint/manifest.om; } | "$OMC" > flint-manifest && chmod +x flint-manifest
./flint-manifest CONFIG INDEX SHARD_DIR > inventory.tsv
```

On success the exit status is 0, stdout is the TSV and stderr is one line:

```
flint-manifest ok tensors 36734 params 39785287456 routed 37849399296 shards 15 bytes 79570574912
```

TSV columns: `name shard dtype shape begin end`. Shards come in name order; within a shard,
rows follow the order of the header's entries. `begin`/`end` are byte offsets into the shard's data section,
which starts at `8 + header length`. `shape` is `d0xd1` or `d0`.

## What it checks

- **Config.** Every key of the pinned `config.json` is recognised or the file is refused, and
  so is a duplicate key. Every value that affects the forward path is pinned. Examples:
  architecture `FlintForCausalLM`; 49 layers; hidden 2048, head_dim 128, 32 query and 4 KV
  heads; 256 experts, top-6, `norm_topk_prob` true; `router_score_function` `sigmoid`;
  `softmax_type` `learnable`; `sliding_window` 513; the full-attention set {0, 1, 5, 9, ..., 45, 48},
  which must equal `no_rope_layer_indices`; `mlp_only_layers` [0, 1]; untied embeddings; vocab
  150144; `rms_norm_eps` 1e-6 and `rope_theta` 10000, compared as exact decimals; `rope_scaling`
  null. `attention_dropout`, `initializer_range`, `output_router_logits`,
  `router_aux_loss_coef`, `use_cache` and `transformers_version` may be absent; when present
  only their type is checked.
- **Index.** The index is read with a strict JSON subset scanner: ASCII without escapes and
  unsigned integers only. The `weight_map` names must be exactly the layout's 36734 tensors,
  each once. A shard name is 13 to 127 bytes of `[A-Za-z0-9._-]`, does not start with `.`,
  and ends in `.safetensors`; at most 64 distinct shards. `total_size` must equal 2 bytes
  times the parameter count derived from the config.
- **Shards.** Each file is opened with `O_RDONLY|O_CLOEXEC|O_NOFOLLOW` and must be a regular
  file. The 8-byte little-endian header length is checked against the file size. Each header
  entry needs dtype `BF16` and exactly the shape derived from the config for its slot.
  `data_offsets` must satisfy `begin <= end` and `end - begin == 2 * prod(shape)` without
  overflow. Within a shard the extents must tile `[0, data length)` exactly: no gap, no
  overlap, no trailing bytes. Each tensor must sit in the shard the index names, and every
  tensor the index assigns to a shard must be present in it.
- **Totals.** Parameters are 39785287456, of which 37849399296 are routed expert parameters.
  Both match the issue and the WolframLanguageEvaluator reconciliation recorded with the
  source capture.

The reader reads headers with `pread`. It never maps or reads tensor data and keeps no
duplicate buffers. It does not hash shard payloads: the 15 shard sha256 values are pinned
separately (`shards.lfs-sha256` in the source capture) and checked by a separate step.

### Storage dtype versus compute contract

Every tensor is stored as BF16, including the router `mlp.gate.weight` [256x2048]. The FP32
sigmoid router is a compute contract of the forward path, not a storage dtype. Nothing in
the checkpoint is FP32.

### Canonical slot space

Slot `0` is `model.embed_tokens.weight`, `1` is `model.norm.weight` and `2` is
`lm_head.weight`. Every other tensor is slot `3 + layer * 784 + kind`:

| kind | tensor | layers |
|---|---|---|
| 0, 1 | `input_layernorm`, `post_attention_layernorm` | all |
| 2..5 | `q_proj`, `k_proj`, `v_proj`, `o_proj` | all |
| 6, 7 | `q_norm`, `k_norm` | all |
| 8 | `softmax_offset` (the learned sink) | all |
| 9..11 | dense `gate_proj`, `up_proj`, `down_proj` | 0, 1 |
| 12 | router `mlp.gate` | 2..48 |
| 13..15 | `shared_expert.{gate,up,down}_proj` | 2..48 |
| 16 + 3e + j | `experts.e.{gate,up,down}_proj` (j 0, 1, 2) | 2..48 |

There is no `shared_expert_gate` tensor: the shared expert is ungated.

## Refusals

On refusal nothing is written to stdout. stderr carries `refused N reason` and the exit
status is N.

| Exit | Reason | Meaning |
|---|---|---|
| 2 | usage | not exactly three arguments |
| 3 | io | open, read or stat failed, or a shard is absent |
| 4 | not-regular-file | a symlink (`ELOOP` under `O_NOFOLLOW`), directory or other non-regular file |
| 10 | config-json | config is not one JSON object |
| 11 | config-field | unknown, duplicate, missing or wrong-valued config key |
| 20 | index-size | index over 4194304 bytes |
| 21 | index-syntax | index outside the strict subset |
| 22 | shard-name | shard name outside the allowed form, or more than 64 distinct shards |
| 23 | unknown-tensor | a name outside the layout, in the index or a header |
| 24 | index-duplicate | a name twice in `weight_map` |
| 25 | total-size | `total_size` differs from 2 bytes times the derived parameter count |
| 26 | index-missing-tensor | a layout tensor absent from `weight_map` |
| 30 | shard-short | shard shorter than 8 bytes |
| 31 | header-length | header length under 2, over 983040, or past the end of the file |
| 32 | header-json | header is not valid JSON |
| 33 | header-entry | header root or entry has the wrong shape |
| 34 | header-metadata | `__metadata__` present |
| 35 | dtype | dtype other than `BF16` |
| 36 | shape | rank or dimensions differ from the config-derived shape |
| 37 | data-offsets | `begin > end`, out of range or overflowing, a length that is not 2 x elements, or data truncated |
| 38 | data-tiling | extents overlap, leave a gap, or leave trailing bytes |
| 39 | header-duplicate | a tensor in two headers |
| 40 | shard-mismatch | tensor stored in a different shard than the index says |
| 41 | header-missing-tensor | a tensor the index assigns to a shard is absent from its header |
| 42 | shard-capacity | over 4096 tensors in one shard, or a header that fills the JSON tape |
| 43 | output-capacity | TSV over 8388608 bytes; a defensive guard that no admitted layout reaches (the full output is 4269125 bytes) |

Hugging Face cache snapshots are directories of symlinks, so they are refused with 4. Point
`SHARD_DIR` at real files, for example after `huggingface-cli download --local-dir`.
`__metadata__` is refused because the pinned checkpoint has none.

## Gate: `check.sh`

```sh
OMC=/path/to/omega-mv/seed/out/omc bash om/flint/check.sh
```

`test/gen.py` is the fixture referee. It derives the full tensor layout from `config.json`
alone and writes a sparse 15-shard checkpoint with the published shard boundaries. It also
writes the expected TSV and one mutant per refusal. The gate:

1. builds the reader with the pinned omc, refusing any other compiler hash;
2. requires the synthetic checkpoint to load with a TSV byte-equal to the referee's and the
   exact summary line, and requires usage exit 2 for 0, 1, 2 and 4 arguments;
3. runs 38 refusal fixtures, each required to exit with its own code, print `refused <code>`
   and write nothing to stdout;
4. rebuilds the reader 25 times, each with one guard removed (every `fm_die N` for one code,
   plus `O_NOFOLLOW` dropped from the open flags). Each mutant must stop producing its code on
   that guard's fixture, which proves the refusal comes from the named guard.

Optional arms, which print `NOT_RUN` when unset:

- `FLINT_SRC=dir` is the pinned source capture (`config.json`, the index, `headers/<shard>.header.json`
  and `shards.lfs-sha256`). The arm rebuilds sparse shards from the exact published header bytes
  and sizes, and the TSV sha256 must equal `60ff4dbd…f5`.
- `FLINT_SHARDS=dir` is the 15 downloaded shards, with config and index from `FLINT_SRC`. The
  same TSV sha256 is required.
- The forward-operator arm runs when both are set and the `FLINT_SHARDS` arm wrote its TSV.
  `FLINT_PY` is the referee's interpreter (torch plus the transformers `a672eb5f` checkout,
  e.g. `env PYTHONPATH=/tmp/tf-a672 ~/.venv/bin/python`); default `python3`.

Measured 2026-10-29 on the Spark host CPU (`taskset -c 10-14`), reader sha256
`d9caa2ab4ecf2580114cfdc8cedb30eb93e80f9f1fc0c0da1be1f31a8d381ae8`: `OM-FLINT-MANIFEST PASS (70)`
with both optional arms. On the downloaded checkpoint the reader took 0.27 s wall and
9440 kB max RSS, with 0 major faults. Without the optional arms the result is
`PASS (68)` plus two `NOT_RUN` lines. With the operator arm the full gate is `PASS (89)`.

## Forward operators: `ops.om`

```sh
{ printf 'om2\n'; cat om/vendor/json.om om/flint/ops.om; } | "$OMC" > flint-ops && chmod +x flint-ops
./flint-ops selftest
./flint-ops OP FIXTURE INVENTORY SHARD_DIR      # OP: rmsnorm qkv attn router shared expert moe
```

`test/opref.py` is the referee: it runs the pinned reference's own modules
(`modeling_flint.py` sha256 `f0540832…0c09` and `configuration_flint.py` sha256
`849eee30…4d00`, transformers `a672eb5f`) in f32 on deterministic inputs built from the real
BF16 weights and writes one fixture per operator, with the torch version in header word 6.
`check.sh` requires the fixtures, concatenated in a fixed order, to hash to `FX_PIN`
(`2d838552…7b7a`, torch 2.14.0 on the Spark host CPU); any other torch, reference file or
CPU result fails that check. `ops.om` reads the
fixture, loads the named BF16 tensors from the shards through the inventory TSV, recomputes the
output in f64 and prints `flint-op OP layer L elems N worst_ppm P ties K PASS|FAIL` (exit 0, 1,
or 2 and up for a refusal). An element passes within twice a first-order bound on the
reference's own f32 error; `test/TOLERANCE.md` has the bound per operator, the gamma_n table,
the Wolfram queries and the detection floor (systematic errors below about 2 gamma_n of the
output's last accumulation always pass). The RoPE bound allows FULP = 4 ulps (`ops.om`) per
powf, cosf and sinf call of the reference; `opref.py` measures those calls against float64 and
refuses to write fixtures if any exceeds FULP (2026-10-29: worst 0.848 ulp, cosf).

Fixtures: RMSNorm (layer 2); q/k/v with q/k norm and RoPE on sliding layer 2, and on full
layer 5 without RoPE; attention with sinks over 520 keys, the 513-key window and its edge on
layer 2, full attention on layer 5, through o_proj; the sigmoid top-6 renormalized router on
layer 24, plus a control with the reference top-6 swapped (must FAIL) and a near-tie control
(must PASS with ties above 0); the shared expert, routed expert 99 and the whole MoE block on
layer 24.

Mutants, each required to turn its operator's line to FAIL with exit 1: softmax router
(`om_exp` in place of `om_sig`), shared expert gated by sigmoid(0), sink omitted, RoPE on every
layer, expert index permuted (255 - e), and window 514.

## Not covered (NOT_RUN for this unit)

- bf16-activation parity: the referee runs in f32;
- generation, bf16 activations and long-context serving remain NOT_RUN; complete two-token
  chained layer/intermediate/logit parity is now covered by the optional arm below;
- weight streaming and repacking, with peak allocation and residency measured during load;
- per-expert layout checks before fusing;
- tokenizer and control-token identity, and a load report.

This unit has no GPU arm. A GGUF importer, if added, will be a separate optional unit.

## Complete bounded host forward

`FLINT_FORWARD=1` adds a full 49-layer chained computation to the same check command.
It starts from two token IDs, loads real BF16 embedding rows and all selected expert weights,
and compares five intermediate tensors per layer, final normalization and every vocabulary logit.
The reference tensors are comparison targets, never teacher-forced inputs.
The full gate prints `OM-FLINT-MANIFEST PASS (94)` with real checkpoint inputs; wrong-token
and skipped-final-normalization source mutants must fail the comparison.

`flint-ops decode` runs the same fixture as a KV-cache decode: one row per chunk, each layer
appends that row's K/V to a per-layer cache (49 layers x 8 positions x 512) and attention reads
every earlier position only from that cache. It must PASS with the same element count as the
batch forward; the `kv-slot-overwrite` mutant (cache row written at slot 0) must fail.

See [PROFILE.md](PROFILE.md) for the frozen weight/runtime/tokenizer/template identities
and the architecture audit. In particular, this checkpoint has **no expert bias**, unlike
the requested selection-only bias description. [test/TOLERANCE.md](test/TOLERANCE.md)
declares the candidate integration budget separately from the tighter per-operator bounds.
Native f64 computation with BF16 storage is compared to reference f32 computation.
This is host-forward evidence only: no serving API, task completion, quantized model,
paged/ring KV-cache beyond 8 positions, long-context batching or GPU/backend qualification is claimed.
