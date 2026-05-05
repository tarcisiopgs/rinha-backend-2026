# syntax=docker/dockerfile:1.7

# ---- builder ----
FROM rust:1.82-slim-bookworm AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

ENV CARGO_TERM_COLOR=always
ARG RUSTFLAGS="-C target-cpu=haswell -C target-feature=+avx2,+fma"
ENV RUSTFLAGS=${RUSTFLAGS}

COPY Cargo.toml Cargo.lock rust-toolchain.toml clippy.toml rustfmt.toml ./
COPY crates ./crates

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release --bin api --bin lb --bin preprocess --locked \
 && cp target/release/api target/release/lb target/release/preprocess /usr/local/bin/

# Opcional: embute dataset pré-processado na imagem. Quando ativado, baixa
# references.json.gz do repo do desafio, roda preprocess e descarta o gz.
ARG EMBED_DATASET=false
ARG DATASET_URL=https://github.com/zanfranceschi/rinha-de-backend-2026/raw/main/resources/references.json.gz
RUN if [ "$EMBED_DATASET" = "true" ]; then \
        mkdir -p /data && \
        echo "baixando dataset" && \
        curl -fsSL "$DATASET_URL" -o /data/references.json.gz && \
        echo "preprocessando dataset" && \
        /usr/local/bin/preprocess /data/references.json.gz /data/references.bin && \
        rm -f /data/references.json.gz && \
        ls -lh /data/references.bin ; \
    fi

# ---- runtime ----
FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/api /usr/local/bin/api
COPY --from=builder /usr/local/bin/lb /usr/local/bin/lb
COPY --from=builder /data /data

ENV RUST_LOG=info

ENTRYPOINT ["/usr/local/bin/api"]
