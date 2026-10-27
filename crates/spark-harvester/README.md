# spark-harvester

Rights-gated reasoning extractor. Every bundle (`extract`, `distill`) and every
`HarvesterClient::query_with_grant` call requires an `aien-provenance`
`SourceGrant` signed by the P-256 authority key passed with `--authority-key`,
`--authority-key-file` or the `ATLAS_AUTHORITY_PUBKEY` vault entry.

## SourceGrant v2 digest cutover

The harvester pins `aien-provenance` and `aien-protocol-types` from
`aien-dev/aien-protocols-mv` at `28bc49ec57a89398fcb463cf71c0add7fcd2ef6c`
(aien-protocols-mv#59). From that revision a grant signature covers the framed
transcript

```
"AIEN_SOURCE_GRANT_V2\0" || grant_id(16) || len64le(source_uri) || source_uri
  || len64le(license_spdx) || license_spdx || permissions_u32le
  || len64le(issuer) || issuer || (0x00 | 0x01 || expires_at_u64le)
```

The retired v1 transcript concatenated the string fields without lengths, so
bytes could move between `source_uri` and `license_spdx` under one signature.

Cutover rules:

- A grant signed under v1 fails with `SignatureVerificationFailed` at the rights
  gate, the client and the CLI. There is no legacy verifier or fallback flag.
- Every grant issued before this pin must be reissued: the authority signs the
  unchanged fields again with `SourceGrant::sign` from the pinned
  `aien-provenance`. Keep the old grant file as history; do not edit it.
- Core held no committed signed grants or harvester receipts when the pin moved,
  so no repository evidence was reissued. Bundles exported earlier keep their
  recorded digests; they were admitted under v1 and are not re-verified.
- `crates/spark-mail-rs` still pins `aien-probe` at `7ac6facb`, which brings the
  old `aien-provenance` and `aien-evaluation-protocol` into `Cargo.lock`. Mail
  only runs the probe engine; it neither signs nor verifies grants or evaluation
  receipts, so it is not a digest consumer and its pin is unchanged.

`tests/source_grant_v2_cutover_tests.rs` holds the regression. The fixtures in
`tests/fixtures/` are one grant signed twice by a synthetic test authority
(signing scalar `[42u8; 32]`, public key
`020c901d423c831ca85e27c73c263ba132721bb9d7a84c4f0380b2a6756fd60133`):
`source-grant-v1-legacy.json` over the retired transcript, refused, and
`source-grant-v2-reissued.json` over the v2 transcript, accepted.
