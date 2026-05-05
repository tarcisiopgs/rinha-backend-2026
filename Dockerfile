# syntax=docker/dockerfile:1.7

# ---- builder ----
FROM rust:1.82-slim-bookworm AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV CARGO_TERM_COLOR=always
ENV RUSTFLAGS="-C target-cpu=haswell -C target-feature=+avx2,+fma"

# Cache de dependências
COPY Cargo.toml Cargo.lock* rust-toolchain.toml ./
COPY crates/api/Cargo.toml crates/api/Cargo.toml
COPY crates/lb/Cargo.toml crates/lb/Cargo.toml
COPY crates/common/Cargo.toml crates/common/Cargo.toml
COPY crates/preprocess/Cargo.toml crates/preprocess/Cargo.toml

RUN mkdir -p crates/api/src crates/lb/src crates/common/src crates/preprocess/src \
 && echo 'fn main(){}' > crates/api/src/main.rs \
 && echo 'fn main(){}' > crates/lb/src/main.rs \
 && echo 'fn main(){}' > crates/preprocess/src/main.rs \
 && echo '' > crates/common/src/lib.rs \
 && cargo build --release --bin api --bin lb 2>/dev/null || true

COPY crates ./crates
RUN cargo build --release --bin api --bin lb

# ---- runtime ----
FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/api /usr/local/bin/api
COPY --from=builder /build/target/release/lb /usr/local/bin/lb

# Runtime defaults — override via env no compose
ENV RUST_LOG=info

ENTRYPOINT ["/usr/local/bin/api"]
