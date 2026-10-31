# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Drake Stapleton (AIEN / aien-dev)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Builder image for scripts/repro-container-build.sh (DRA-43). CPU only: no CUDA, no device.
# Base pinned by digest (ubuntu 24.04 arm64); rustup-init pinned by version and sha256; the Rust toolchain is the
# one rust-toolchain.toml names, with the components it lists, so the build never writes to RUSTUP_HOME.
# Ubuntu's binutils is configured with deterministic archives, so `ar rcs` writes uid/gid 0 and mtime 0.
FROM ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852
ARG RUSTUP_VERSION=1.29.1
ARG RUSTUP_INIT_SHA256=15f6e4ce9f583b929c996c91562bad6d4454f3281de858b02cdfdef615fac433
ARG RUST_TOOLCHAIN=1.98.1
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates curl git make gcc libc6-dev binutils util-linux \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/opt/rustup PATH=/opt/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN curl --proto '=https' --tlsv1.2 -sSfo /tmp/rustup-init \
        "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/aarch64-unknown-linux-gnu/rustup-init" \
    && echo "${RUSTUP_INIT_SHA256}  /tmp/rustup-init" | sha256sum -c - \
    && chmod 755 /tmp/rustup-init \
    && CARGO_HOME=/opt/cargo /tmp/rustup-init -y --no-modify-path --profile minimal \
        --default-toolchain "${RUST_TOOLCHAIN}" --component rustfmt --component clippy \
    && rm /tmp/rustup-init \
    && chmod -R a+rX /opt/rustup /opt/cargo
