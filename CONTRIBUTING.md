# Contributing to AIEN Sovereign Core

I welcome contributions from systems engineers, researchers, and autonomous agents advancing open, efficient computing.

## Developer Certificate of Origin (DCO 1.1)

All contributions to AIEN repositories require certification under the Developer Certificate of Origin (DCO 1.1). By submitting a pull request, you certify the statements in [DCO](DCO).

Sign off your commits using standard git tooling:

```bash
git commit -s -m "feat(subsystem): descriptive commit message"
```

## Licensing of Contributions

- All software code, including the inference ABI (`aien-inference-abi`), is contributed under the [GNU Affero General Public License v3.0 or later](LICENSE) (`AGPL-3.0-or-later`).
- Architectural specifications and documentation are contributed under `AGPL-3.0-or-later`.
- Benchmark datasets and telemetry logs are dedicated to the public domain under [CC0-1.0](LICENSES/CC0-1.0.txt).
- Model weights and curated training datasets follow classifications in [ARTIFACT_LICENSING.md](ARTIFACT_LICENSING.md).

For bilateral research cooperation, model weight reciprocity, and open distillation agreements, refer to [OPEN_COOPERATION_COVENANT.md](OPEN_COOPERATION_COVENANT.md).

## Upstream Stewardship Doctrine

AIEN upstream development is guided by [CONSTITUTION.md](CONSTITUTION.md). The Constitution defines my technical discipline and ethical commitment to human advancement. As noted in its Governance Notice, the Constitution does not restrict downstream commercial licensees of the software.

## Submission Standards

1. **Mandatory Branching**: Never push directly to `main`. Create descriptive feature branches (`feat/`, `fix/`, `docs/`, `perf/`).
2. **Sovereign Voice & Anti-Slop**:
   - Zero em dashes (Unicode U+2014) and zero en dashes (Unicode U+2013). Use commas, colons, parentheses, or plain hyphens.
   - Ban formulaic marketing buzzwords and conversational filler.
   - Lead pull request descriptions with technical proof, test output, or benchmarks.
3. **Secret Protection at Rest**: Never commit plaintext `.env` files, API keys, or private tokens. All secrets must resolve through an approved `SecretProvider`.
4. **Verification**: Verify that `cargo check --workspace --all-targets` and `cargo test --workspace` pass with zero warnings before opening a pull request.
