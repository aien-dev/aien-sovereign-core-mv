# SMOLLM2-Q: SmolLM2-1.7B-Instruct on the complete CAND-4 qualification suite (declared before any run)

```text
campaign_id   = "smollm2-qualification"
spec_version  = 1
purpose       = Drake decision D3 (aien-architecture CURRENT_EXECUTION_PLAN.md, "operator decisions",
                arch#156, merge f469bd13): after the daemon memory-sizing fix (sovereign-core issue #239,
                PR #246) is merged, retest SmolLM2-1.7B-Instruct. If it passes the complete qualification
                suite it becomes the next open-model candidate; if it fails for a reason unrelated to the
                corrected memory sizing, Qwen3-4B is evaluated next. Never a bigger model to dodge the
                allocation bug.
suite         = the CAND-4 qualification suite (cand4-qualification/ACCEPTANCE-CAND4.md, frozen in 130beeb,
                merged 328a7e9), run unchanged except for the substitutions of section 0, plus the memory
                observations of section 8 (they close issue #236 and are not verdict rows).
candidate     = sovereign-core main at the merge commit of #246 (sc_commit below), its omega and aienos
                pins, and the SmolLM2 snapshot of section 2. Not a named candidate: if this campaign
                PASSES, a CAND manifest is frozen afterwards from exactly these values.
status        = DRAFT until the block in section 1 holds no UNFROZEN value. The freeze is the commit that
                fills that block; it changes this file only and is made before any run. Nothing in this
                file changes after the freeze; a change is a new spec_version.
scope         = Linux-hosted, single machine (the Spark, GB10), the sovereign-core daemon path, as CAND-4.
                Licence: SmolLM2 weights keep Apache-2.0 and their upstream notices; AIEN code Apache-2.0.
```

## 0. What differs from the CAND-4 suite (every difference, nothing else)

| item | CAND-4 | SMOLLM2-Q | why |
|------|--------|-----------|-----|
| model | unsloth/Llama-3.2-1B-Instruct 5a8abab | HuggingFaceTB/SmolLM2-1.7B-Instruct 31b70e2e (section 2) | the purpose |
| chat template | Llama 3 (from the snapshot) | ChatML (from the snapshot's `tokenizer_config.json`; sovereign-core #233, merged a39c617) | the model's own template |
| `max_tokens` | 96 | 64 | frozen in SmolLM2 campaign v1 before its run, from the measured GB10 decode speed (open-model-smollm2/ACCEPTANCE-v1.md section 2.1: a 96-token attempt costs about 31.7 s, more than the 29 s skill budget). Keeping 64 is no change after results; raising it now would be tuning after v1's T3 hit the limit. |
| binaries | CAND-4 double build at d5b78ff | a clean double build at `sc_commit` (the CAND-4 recipe, `scripts/release-build.sh`), plus `cpu_fault` rebuilt from `sc_commit` by `build-cpu-fault.sh` | new code (#233, #246 and everything merged between) |
| model pinning | seven named files, exact listing | every regular file of the snapshot as a `model_file` line, exact listing `model_dir_listing` (the snapshot also holds `.cache/`, the download metadata, not read by the loader) | the SmolLM2 snapshot has other files than the Llama one |
| harness | `cand4-qualification/` at 1ce850c (#240 hardening included) | a copy in this directory; renamed labels (`SMOL2Q_*` gates, owner `laneSmol2`), the generic model check above, per-launch memory observations in the Q1 attempt records, `results_sha256` in the Q2w score (CAND-4 review finding e), the `mem` command (section 8), the stale note of `q1.decl.json` corrected (CAND-4 review finding b) | the CAND-4 files are a frozen record and are not edited |
| owner of GPU holds | laneQ | laneSmol2 | session "aien implementation phase transition" runs this campaign |


## 1. Frozen inputs (filled once, at the freeze)

`run-smol2q.sh` reads this block. It refuses a QUALIFICATION run while any value is UNFROZEN and
recomputes every sha256 below before every launch, inside each GPU hold.

```text
>>> SMOLLM2-Q frozen inputs >>>
kind                    = QUALIFICATION
candidate_id            = SMOLLM2-Q (unnamed until a PASS)
sc_commit               = 2e8cef96db9eee426b2e52b5396722294b0e69e0
omega_commit            = c0369e6705a4b0cb800978846e78915126a1b7f7
aienos_commit           = b84c0a67590a934f3f3e001b12ec85ebc086a9eb
physics_commit          = 6d7cf0d4d8eb2cda7b512100ff6058e25dbb3ddf
build_evidence          = /home/drakestapleton/workspace/evidence-out/SMOL2Q-BUILD-A
aien_cli_path           = /home/drakestapleton/.claude/jobs/9bfe8553/tmp/smol2/build/rootA/wt-sc/target-rb/release/aien-cli
aien_cli_sha256         = ce2fff7cb473550f61a7db3b78ef0082ea11753eb6b43d162cbcbd242e186a9d
cpu_fault_path          = /home/drakestapleton/.claude/jobs/9bfe8553/tmp/smol2/build/cpu-fault/aien-cli-cpu-fault
cpu_fault_sha256        = b1b4acb0b23615ef61bbeb76ddbaedc14777be9d616f0ab62ba1482dbf08105d
librx_compose_sha256    = dfe0ffb8904b08cf9b67c857f7e9de34bf493a6d4d763e9fef0cf2b10f114cd6
libomega_gpu_sha256     = ef80e4313d4e678335dceac3493aadec8d0df4352dd5356854b92e0a9e070887
omega_build_dir         = /home/drakestapleton/.claude/jobs/9bfe8553/tmp/smol2/build/rootA/wt-omega/build/rb
rx_r13_host_sha256      = f7b3add700e6b2971a2b1851793529c62549f1ecb3439afb792e18d6ea940cb9
rx_r13_silicon_sha256   = 088a7f2b127e941a420ff855d86349e24fe65b50fe4849accb2300f96cc56e25
rx_r13_testbuild_sha256 = 52e6bd7b6b187174719b84e86aaed99a86ddfe11f35a8eb3419fece169d764cf
omega_src_dir           = /home/drakestapleton/.claude/jobs/9bfe8553/tmp/cand4/src/omega
aienos_repo             = /home/drakestapleton/.claude/jobs/9bfe8553/tmp/cand4/src/aienos
physics_dir             = /home/drakestapleton/.claude/jobs/9bfe8553/tmp/smol2/build/rootA/wt-physics
model_dir               = /home/drakestapleton/models/SmolLM2-1.7B-Instruct-31b70e2e869a
model_dir_listing       = .cache,.gitattributes,PROVENANCE.md,README.md,config.json,generation_config.json,instructions_function_calling.md,merges.txt,model.safetensors,special_tokens_map.json,tokenizer.json,tokenizer_config.json,vocab.json
model_file              = .gitattributes 11766e5171641f6c68b9f14099fdb4e7b3d610f0a69451893e356c8334812257
model_file              = PROVENANCE.md 6e8636821c1ceb98dde07c71dbc2a8bde06dcb367a0f20b0cce573ed5b076b92
model_file              = README.md b00c0570dbc43674a0028d088fc72f40d83eea90f47cd987706ca2f95ed114c0
model_file              = config.json 994f50b16abb4ae00880baefe03c10260b5bd608d2bf586f7056ca05a534feea
model_file              = generation_config.json 87b916edaaab66b3899b9d0dd0752727dff6666686da0504d89ae0a6e055a013
model_file              = instructions_function_calling.md dd266c8ca7f329682d17c1bfb72e2b870ac849f17a79a0bc23ccc897e4917134
model_file              = merges.txt 0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510
model_file              = model.safetensors f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8
model_file              = special_tokens_map.json 2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52
model_file              = tokenizer.json 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
model_file              = tokenizer_config.json 4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df
model_file              = vocab.json 82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79
llama_control_dir       = /home/drakestapleton/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/5a8abab4a5d6f164389b1079fb721cfab8d7126c
llama_control_safetensors = 1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f
llama_control_config    = dfb67fd8afe73a1c75245824ef9d64a6ba8983025447e3bf76aa1ea57ee46152
max_tokens              = 64
attempt_budget_ms       = 12000
skill_budget_ms         = 29000
max_attempts            = 3
q1_a1_record_mark       = record-mark
fix_old_dir             = /home/drakestapleton/.claude/jobs/9bfe8553/tmp/np2-fix
fix_old_files_sha256    = 3e4e26eff45e0f7282c335cb2028cd8c314e11b2669c73c16ab27c530c89eaf8
<<< end of SMOLLM2-Q frozen inputs <<<
```

The model values (sections 1 and 2) are filled now, before the freeze, because they do not depend on the
build; the twelve sha256 values equal the files on disk (computed 2026-10-06) and the four that
SmolLM2 campaign v1 pinned (model, config, generation config, tokenizer) are byte-equal to v1.
The Llama control values equal CAND-4 (`ACCEPTANCE-CAND4.md` section 1). Every UNFROZEN value is
filled from the double build at `sc_commit` in the freeze commit.

## 2. Model, template and budgets

```text
model_id     = HuggingFaceTB/SmolLM2-1.7B-Instruct, revision 31b70e2e869a7173562077fd711b654946d38674
               LlamaForCausalLM, 24 layers, hidden 2048, 32 heads, 32 kv heads, head_dim 64,
               max_position_embeddings 8192 (config.json), BF16 weights, chat template ChatML,
               stop token <|im_end|>, licence Apache-2.0 (model card; PROVENANCE.md in model_dir)
max_tokens   = 64 (section 0)
```

Everything else is ACCEPTANCE-CAND4 section 2 unchanged: greedy decoding stopping on the model's
end-of-sequence token, the proposal prompt with the `filename:` assistant prefix, the 12 000 ms
attempt threshold, the 29 000 ms skill budget, 3 attempts, the retry policy and the launch
environment. `run-smol2q.sh check` reads the budget, attempt and prefix constants from the code at
`sc_commit` and refuses if they differ from section 1.

Stated limit carried from v1 (`open-model-smollm2/ACCEPTANCE-v1.md` section 2.1): GB10 greedy output
can differ from the Hugging Face reference on near-ties (two of six diagnostic prompts).

## 3. Production hygiene (H), run first, no GPU

ACCEPTANCE-CAND4 section 3 unchanged (rows H1, H1c, H2, H3, H4, H4c, H5, H6), against the SMOLLM2-Q
binaries; `hygiene.sh`.

## 4. Q1 useful workflow (GPU)

ACCEPTANCE-CAND4 section 4 unchanged: tasks T1, T2, T3 of `next-phase-1/tasks-v5.json`, 3 repetitions
each, every v1..v5 row, one quietlock hold per round, `q1.decl.json`. Row Q4 already lists the ChatML
markers `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`. A daemon that fails to start makes that launch
FAIL. In addition (observations, not rows) every attempt record carries both daemons' KV pool plan
line, warm-up line, fatal lines, session-open attempt lines and VmHWM (section 8).

### 4.1 Reading of the A1 containment row

ACCEPTANCE-CAND4 section 4.1 unchanged (`q1_a1_record_mark = record-mark`, `a1-read.sh`), with its
two negative controls as rows of `q1.decl.json`.

## 5. Q2 recovery (NEXT-PHASE-2 v4 case set)

ACCEPTANCE-CAND4 section 5 unchanged: F0 on the GB10 in one hold, then the NEXT-PHASE-2 v4 cases on
`cpu_fault` (rebuilt from `sc_commit` by `build-cpu-fault.sh`), `REPS=3`, `q2.decl.json`, scored by
SCORING-v5. The cases unset the model variables and run on the CPU-reference daemon
(`next-phase-2/run-faults.sh`), so only F0, C3-control and Q1 use SmolLM2.

## 6. Coverage of six failure cases

ACCEPTANCE-CAND4 section 6 unchanged: cases 1, 3 and 6 OUT OF SCOPE; case 4 CONDITIONAL on C6d;
case 2 only as W2; case 5 only non-deterministically (W5).

### 6.1 Window rows (Q2w)

ACCEPTANCE-CAND4 section 6.1 unchanged: W-ctl, W2 x3, W5 with 40 trials, `windows.decl.json`,
`run-windows.sh`, on the CPU build.

## 7. Run order, receipts, verdict

Run order and GPU rules:

1. After the coordinator's "GO SmolLM2" (sent when #246 is merged): `sc_commit` = that merge commit
   (or main at GO if later merges landed first, named in the freeze commit). Double build with the
   CAND-4 recipe (`scripts/release-build.sh`), started only while `~/workspace/.spark-quiet` is
   absent (the CAND-4 lesson: a hold taken mid-build makes the GPU-library steps refuse). Rebuild
   `cpu_fault`.
2. Freeze commit (fills section 1, this file only). `run-smol2q.sh check` PASS.
3. `hygiene` (no GPU), then `mem` (section 8, one hold <= 10 min), then `q1` (three holds <= 20 min),
   then `q2` (F0 in one hold, cases on CPU), then `q2w` (CPU).
4. Every hold through `quietlock hold` with start and release whispers. Another lane's hold is waited
   for, never interrupted. No chip test is killed.

Receipts and what is never done: ACCEPTANCE-CAND4 section 7 unchanged (receipts in `receipts/`, no
launch repeated, no threshold changed after results, NOT_RUN / NOT_APPLICABLE / INCOMPLETE /
UNVERIFIED never PASS).

SMOLLM2-Q PASS iff every digest check passed, H PASS (H1, H1c, H2, H3, H4, H4c, H5), Q1 PASS (section
4.1 reading), Q2 PASS and Q2w PASS. The verdict (`VERDICT-SMOLLM2-Q.md`) also states the section 9
class of every failing row and cites the section 8 records.

## 8. Memory observations (close issue #236; observations, not verdict rows)

On the production `aien_cli` of section 1, `run-smol2q.sh mem` makes two daemon starts in one hold:
SmolLM2 (`model_dir`), then a Llama-3.2-1B control (`llama_control_dir`, files checked against CAND-4).
Each start records: `/proc/meminfo` before (MemTotal, MemAvailable, SwapFree, HugePages; NVIDIA DGX
Spark known issues: on this unified-memory machine `nvidia-smi` memory is "Not Supported" and
`cudaMemGetInfo` can under-report, so `/proc/meminfo` is the source), any process holding a GPU device
file, the KV pool plan line and the memory check line (#246), the warm-up line, any fatal or
session-open attempt line, and VmHWM after warm-up from `/proc/<pid>/status`. Every Q1 launch also
records both daemons' plan line, warm-up line, fatal lines and VmHWM (from `run.json`) in its attempt
record.

Expected, from the #246 formula and each `config.json` (bytes = blocks x 16 tokens x layers x kv heads
x head_dim x 2 x 4 bytes):

| model | expected plan line starts | before #246 |
|-------|---------------------------|-------------|
| SmolLM2 | `KV pool: 3221225472 bytes (3.00 GiB) = 512 blocks x 16 tokens x 24 layers x 32 kv heads x 64 head_dim x 2 (K,V) x 4 bytes; context 8192 tokens` | 8192 blocks, 51539607552 bytes (48.00 GiB); v1 repro VmHWM 61.1 GB |
| Llama-3.2-1B | `KV pool: 8589934592 bytes (8.00 GiB) = 8192 blocks x 16 tokens x 16 layers x 8 kv heads x 64 head_dim x 2 (K,V) x 4 bytes; context 131072 tokens` | the same 8.00 GiB (unchanged); v1 repro VmHWM 17.4 GB |

Issue #236 is closed after `mem` runs, citing both records, whatever the Q1 result. If the 0x51 crash
appears in any start, #236 stays open with that record.

## 9. Classification of a FAIL (for decision D3; written before any run)

Every failing row is put in exactly one class in `VERDICT-SMOLLM2-Q.md`:

- **M, memory sizing:** a SmolLM2 daemon start (Q1, F0 or `mem`) that ends in `Fatal: shared KV pool`,
  the KV pool memory refusal, a GPU session open that fails after its attempts, driver status 0x51,
  or an out-of-memory kill; or a SmolLM2 KV pool plan line that differs from the expected one in
  section 8. A row that fails because of such a start is class M.
- **U, unrelated to memory sizing:** every other failing row (task quality Q1 to Q4, authority A1 to
  A6, v1..v4 rows, a recovery or window row, hygiene, a digest).

If any class M event occurs, the verdict says so first: the memory fix did not hold for SmolLM2, the
fix is reworked, and the D3 step to Qwen3-4B is not taken on this result. If every failure is class U,
the verdict says so: D3 then names Qwen3-4B as the next model to evaluate.

## 10. Not in this suite

NEXT-PHASE-1 v6 and v7 (the wider correctness campaigns) are not part of the CAND-4 qualification
suite and are not run here; a PASS here says nothing about them.
