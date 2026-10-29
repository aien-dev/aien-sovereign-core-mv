# Multi-stage build for cortex-rs
FROM rust:slim-bookworm AS builder

WORKDIR /build

COPY Cargo.toml Cargo.lock* ./
COPY src ./src

RUN cargo build --release

# Minimal runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/target/release/cortex-rs /usr/local/bin/cortex-rs
COPY scripts/reproduce.sh /usr/local/bin/reproduce.sh
RUN chmod +x /usr/local/bin/reproduce.sh

# Run unprivileged; the default store is $HOME/.config/cortex/cortex.db.
RUN useradd --system --create-home --uid 10001 cortex
USER cortex

EXPOSE 18080

ENTRYPOINT ["cortex-rs"]
CMD ["--host", "0.0.0.0", "--port", "18080"]
