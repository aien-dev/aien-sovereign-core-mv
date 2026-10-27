# Third-Party Model Notices

AIEN code is Apache-2.0 (`LICENSE`, `NOTICE`). The models below are separate
third-party works. Each keeps its publisher's licence; running one on AIEN does not
relicense it. The SmolLM2 attribution detail lives in `ATTRIBUTION.md` and `NOTICE`.

## What the release package contains

Read from `scripts/package-release.sh`: the release tarball holds the seven built
binaries, `CONSTITUTION.md`, `README.md`, `install.sh`, `LICENSE`, `NOTICE`, `THIRD_PARTY.md`, `imprints/en2-trinity/`
(6,275 bytes of text) and a generated `release.toml`. `release.toml` carries a `[model]`
table copied from `release/candidate.toml`; that table is **metadata only** (model id,
snapshot, SHA-256 digests, geometry, licence name). **No model weights are packaged in
any release artifact built by this script.** Weights are fetched or supplied
separately by the operator.

| Model | Licence | Upstream | Weights in a release artifact? | Where referenced |
|---|---|---|---|---|
| SmolLM2-1.7B-Instruct (HuggingFaceTB), rev `31b70e2e869a7173562077fd711b654946d38674` | Apache-2.0 (model card `license: apache-2.0`) | https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct | No | `NOTICE`, `ATTRIBUTION.md`, `docs/campaigns/smollm2-qualification/` |
| Llama 3.2 1B Instruct, as `unsloth/Llama-3.2-1B-Instruct`, snapshot `5a8abab4a5d6f164389b1079fb721cfab8d7126c` (CAND-4) | Llama 3.2 Community License (`licence` in `release/candidate.toml`) | https://huggingface.co/unsloth/Llama-3.2-1B-Instruct (re-host of Meta's model) | No. Digests only in `release.toml` | `release/candidate.toml`, `docs/release/ORACLE-FIXTURE-CAND4.md` |
| Qwen3-4B-Instruct-2507 (Qwen), HF revision `cdbee75f` | Apache-2.0 per `docs/campaigns/open-model-qwen3/VERDICT-v2.md` line 31 (upstream card not re-checked) | https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 | No. Referenced in campaign docs only; not qualified, not distributed, no notice shipped (no Qwen3 entry in `NOTICE`) | `docs/campaigns/open-model-qwen3/` |

## CAND-4 (Llama 3.2 1B): internal only

`release/candidate.toml` records `distribution = "internal-only until an open-licence
model passes"`. CAND-4 is used for internal qualification and is **never distributed**
by this project, in a release archive or otherwise.

If it were ever distributed (weights or a product built on them), the Llama 3.2
Community License would apply and would need at least: a copy of that licence given
to every recipient, and the "Built with Llama" attribution. These requirements are
from general knowledge of the licence and are **UNVERIFIED**: the official page
(`https://www.llama.com/llama3_2/license/`) redirected through developer.meta.com to
dev.meta.ai, which returned HTTP 404 on 2026-10-07, so the text could not be checked.
Re-read the official licence before any distribution. Its text is not reproduced here.

## Follow-up

Packages built after this change carry `LICENSE`, `NOTICE` and this file (sha256 listed in
`release.toml`). `package-release.sh` refuses to package without them, but `install.sh` and
`verify-release-assets.sh` do not yet require `LICENSE` in a package; that would also reject
the published v0.1.x archives. Published v0.1.0 and v0.1.1 are unchanged.

## Third-party code and ideas (not models)

| Project | Licence | Reference | What was adapted | Where |
|---|---|---|---|---|
| LMCache | Apache-2.0 (reference clone `~/workspace/refs/lmcache` at `4d69cdc4`; the clone has a `LICENSE` and no `NOTICE` file) | https://github.com/LMCache/LMCache | Ideas only, no code copied: deterministic rolling prefix keys (`lmcache/v1/multiprocess/token_hasher.py`), completed-object publication by temp file plus rename and lazy disk recovery (`nixl_store_dynamic_l2_adapter.py`, `fs_l2_adapter.py`). Test cases re-expressed in Rust from `tests/v1/distributed/test_nixl_store_dynamic_l2_adapter.py` (names in ADR 0003 context item 24). | `crates/aien-kv-store/` (`src/identity.rs`, the test files that name each adapted case), `docs/adr/0003-persistent-kv-identity-storage-contract-and-ownership.md` |
