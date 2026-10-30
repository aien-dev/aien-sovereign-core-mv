# Alpha classification regression evidence

Production pipeline tests: merged baseline 87 passed / 4 failed; repaired 91 passed / 0 failed. Independent rerun 91/0 plus 17 extracted-function and two-ref controls passed. Shell syntax and whitespace checks passed. Refs #767 and #768.

Base: 5a5474766e977c70a01c47e6ff7db7046d4a419b. Run `bash tools/aien-local-ci/tests/pipeline_test.sh` on a host with Bash, Git, jq and flock. It uses temporary local Git repositories and fake hosting/verifier commands, no real signing key, hardware, or network. The test verifier specifically models refusal of selected no-GPU fixtures without the waiver; it does not replace Rust cryptographic verifier tests.

The repair retains GPU proof on read errors, recognizes nvrm, and checks both current trusted main and the candidate's committed manifests. JSON Unicode escapes conservatively retain GPU proof because this is a text scan, not the manifest parser. Unrelated Unicode metadata can therefore over-require GPU proof. Existing no-manifest host-only policy stays unchanged.

Local preflight: branch isolation, zero secrets and text checks passed; workspace tests blocked because Cargo is absent, cargo-audit skipped absent. Full Rust workspace tests, supported-host qualification, hardware tests and remote exact-head checks were NOT_RUN locally. No deployment or merge performed. The draft needs exact-head CI and supported-host qualification before readiness.

No trust keys, signature checks, full-scope checks, verdict algebra, merge logic or GPU runner changed. No gate weakened. The local patch only reduces the circumstances where the caller grants an evidence waiver.
