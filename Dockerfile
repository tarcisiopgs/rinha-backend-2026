# syntax=docker/dockerfile:1.7

# ---- builder ----
FROM rust:1.82-slim-bookworm AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV CARGO_TERM_COLOR=always
ARG RUSTFLAGS="-C target-cpu=haswell -C target-feature=+avx2,+fma"
ENV RUSTFLAGS=${RUSTFLAGS}

COPY Cargo.toml Cargo.lock rust-toolchain.toml clippy.toml rustfmt.toml ./
COPY crates ./crates

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release --bin api --bin lb --locked \
 && cp target/release/api target/release/lb /usr/local/bin/

# ---- runtime ----
FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/api /usr/local/bin/api
COPY --from=builder /usr/local/bin/lb /usr/local/bin/lb

ENV RUST_LOG=info

ENTRYPOINT ["/usr/local/bin/api"]
