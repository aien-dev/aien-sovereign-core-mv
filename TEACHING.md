# A model-free request through generation

I kept this Linux-hosted Rust engine as scaffolding while Omega takes over machine-bound work, as [ADR 0024](https://github.com/aien-dev/aien-architecture-mv/blob/main/docs/adr/0024-rust-scaffolding-omega-destination.md) records. It is research-grade. A fresh reader can reproduce a generation-controller decision without downloading weights or touching a live model. That is narrower than proving a complete deployed inference request.

For a real request, [`generate_text`](crates/aien-runtime/src/server.rs) accepts the text, [`ChatTokenizer::encode`](crates/aien-inference-abi/src/tokenizer.rs) makes token IDs, the [scheduler](crates/aien-scheduler/src/lib.rs) admits and batches sequences, and the [KV manager](crates/aien-kv-cache/src/lib.rs) allocates blocks and copies a shared tail on write. The [transformer backend](crates/aien-inference-abi/src/transformer_backend.rs) supplies forward/decode and Omega-backed variants. The [generation controller](crates/aien-inference-abi/src/generation.rs) advances tokens, checks stop/cancel/length rules and returns an outcome. Its [EchoChainModel tests](crates/aien-inference-abi/src/generation/tests.rs) use deterministic synthetic logits and KV, including a refusal when a cancelled or stopped token would be stepped. The [COW branching fixture](crates/aien-kv-cache/tests/cow_branching_tests.rs) shows why branch forks share full prefix blocks until a tail write needs a private copy. This saves allocation in that test; it is not a model throughput measurement.

## Replay

On AArch64 Linux with Rust `1.98.1` from [rust-toolchain.toml](rust-toolchain.toml), cargo and the repository's locked dependencies already cached, use a fresh checkout of `754c07b118f41ad899607b60329bd851b1ccb13e`:

```sh
git clone https://github.com/aien-dev/aien-sovereign-core-mv.git core-lesson
cd core-lesson
git checkout --detach 754c07b118f41ad899607b60329bd851b1ccb13e
cargo test --locked --offline --lib -p aien-inference-abi generation::tests::generate_stops_on_stop_ids_caps_length_and_cancels -- --exact
```

Expected exit is 0 with one named test passed. [TEACHING-REPLAY.txt](TEACHING-REPLAY.txt) records the actual attempt. The synthetic test exercises one token control path, not the server, tokenizer, scheduler, Omega GPU path or a real generated sentence together. [FIRST_RUN.md](docs/FIRST_RUN.md) is the model and desk setup route; [issue 414](https://github.com/aien-dev/aien-sovereign-core-mv/issues/414) tracks release blockers. Published historical prereleases exist on the [GitHub releases page](https://github.com/aien-dev/aien-sovereign-core-mv/releases), but they do not qualify this source revision or provide evidence of a supported general installer. This tutorial does not enroll a machine or claim the model route passed.
