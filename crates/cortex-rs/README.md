# cortex-rs

A memory service for the AIEN agent tools, written in Rust (Axum web server, SQLite storage). It stores entities and claims and answers lexical and semantic recall queries over a loopback HTTP API.

It is not the canonical memory owner. Under [ADR 0022](https://github.com/aien-dev/aien-architecture-mv/blob/main/docs/adr/0022-canonical-cortex-owner.md) the canonical Cortex owner at the Omega layer is omega `rx_cortex`. Under [ADR 0036](https://github.com/aien-dev/aien-architecture-mv/blob/main/docs/adr/0036-release-v1-memory-owner.md) the memory the v1 release tests is `aien-allen-memory`, attached to an engaged ALLEN identity. This crate is a member of the aien-sovereign-core workspace; the old standalone repository `aien-dev/cortex-rs` is archived.

## What it does

- **Lexical search.** SQLite full-text search (FTS5, BM25 ranking) over stored entities, with the database in WAL mode.
- **Semantic search.** Calls a separate local embedding service (`--encoder-url`, default `http://127.0.0.1:18081`; see `crates/cortex-encoder-rs`) and scores results by cosine similarity.
- **Graph links.** Claims and entities (`cortex_claims`, `cortex_entities`) are linked in both directions so recall can follow them.
- **Token required, loopback by default.** Every route except `/health` and `/status` needs a bearer token (`CORTEX_TOKEN`, else the file `~/.config/cortex/token`). The default host is `127.0.0.1`. Another `--host` is allowed but logs a warning.

## Build and run

From the root of the aien-sovereign-core checkout:

```bash
cargo build --release -p cortex-rs
./target/release/cortex-rs --help
./target/release/cortex-rs --bench --bench-records 1000
./target/release/cortex-rs --port 18080 --db-path ~/.config/cortex/cortex.db
```

`--bench` runs a built-in micro benchmark (entity writes, FTS5 search, claim traversal) on a temporary database and prints p50, p95 and p99 latencies. Its numbers depend on the machine and are not published evidence. Earlier copies of this page showed a sample output; that output had no recorded commit or run and has been removed.

`scripts/reproduce.sh` starts the server on a temporary database, checks `/health`, writes an entity and runs a search. It looks for `./target/release/cortex-rs`, so run it from the workspace root.

The `Dockerfile` here dates from the standalone repository. It copies only this crate's `Cargo.toml` and `src`, while that `Cargo.toml` takes settings from the workspace (`license.workspace = true`), so it is not expected to build as it stands. It was not tested for this page. There is no container image for this crate published from this repository.

## API

| Route | Auth | What it does |
|---|---|---|
| `GET /health`, `GET /status` | none | service name, space and version |
| `POST /api/cortex/write` | bearer token | store an entity or claim |
| `GET` or `POST /api/cortex/search` | bearer token | lexical and semantic recall |

Also behind the token: `/api/cortex/recall`, `/api/cortex/get`, `/api/cortex/traverse`, and the session, candidate and context routes under `/api/cortex/v2/` (all listed in `src/main.rs`).

```bash
curl http://127.0.0.1:18080/health
curl -G http://127.0.0.1:18080/api/cortex/search \
  --data-urlencode "q=unified memory" \
  -H "Authorization: Bearer $CORTEX_TOKEN"
```

## License

AGPL-3.0-or-later, like the rest of the workspace (`license.workspace = true` in `Cargo.toml`). See [LICENSE](../../LICENSE).
