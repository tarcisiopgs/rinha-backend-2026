# rinha-backend-2026

Submissão da [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) — detecção de fraude k-NN com restrição de **1 CPU + 350 MB RAM**.

**Objetivo:** disputar o primeiro lugar do leaderboard (score 6000.0, p99 ≤ 1ms).

## Stack

- **Linguagem:** Rust 1.82 (edition 2021)
- **Runtime async:** [monoio](https://github.com/bytedance/monoio) (io_uring nativo, single-threaded)
- **Comunicação:** Unix Domain Sockets entre LB e APIs (sem TCP loopback)
- **SIMD:** AVX2 manual via `core::arch::x86_64` (target-cpu=haswell)
- **Layout:** SoA (Structure of Arrays) com vetores quantizados i16 + score u8
- **HTTP parser:** manual com `memchr`, sem framework
- **Respostas:** pré-montadas no startup (12 buffers fixos)

## Arquitetura

```
            :9999 TCP
              │
        ┌─────▼─────┐
        │    lb     │  round-robin atômico
        └──┬─────┬──┘
           │     │      Unix Domain Socket
        ┌──▼─┐ ┌─▼──┐
        │api1│ │api2│   monoio + io_uring
        └────┘ └────┘   dataset mmap (SoA i16)
```

## Layout do repositório

```
crates/
├── common/      # Dataset (mmap), SIMD, constantes de protocolo
├── api/         # Servidor HTTP + busca k-NN
├── lb/          # Load balancer TCP → UDS, round-robin
└── preprocess/  # references.json.gz → references.bin (i16 SoA)
data/            # binários do dataset (gitignored)
```

## Setup

### Pré-requisitos

- Rust 1.82+ (instalado via `rust-toolchain.toml`)
- Docker + Docker Compose
- `references.json.gz` do repo do desafio em `data/`

### Build local

```bash
make check       # fmt + clippy + test
make build       # cargo build --release
make preprocess  # gera data/references.bin (uma vez)
make run         # docker compose up --build
```

### Validação

```bash
curl -X POST http://localhost:9999/fraud-score \
  -H 'Content-Type: application/json' \
  -d '{"vector":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.0,0.1,0.2,0.3,0.4]}'
```

## Roadmap pra atingir 6000.0

- [x] Scaffold com workspace, Dockerfile multi-stage, CI
- [x] Parser HTTP manual + respostas pré-montadas
- [x] Dataset SoA + mmap + score quantizado u8
- [x] Brute-force escalar baseline
- [ ] AVX2 dot product i16 → i32 (`_mm256_madd_epi16`) em batch de 16
- [ ] IVF index (kmeans clusters, nlist tuning)
- [ ] Quantização i8 (avaliar perda de recall)
- [ ] Huge pages (2MB) pro mmap do dataset
- [ ] Pin thread em CPU + `SCHED_FIFO` (se permitido)
- [ ] LB com `splice(2)` ao invés de copy
- [ ] Profile com `perf stat`/`flamegraph` em hardware do desafio
- [ ] Submissão final + validação local com gatling do repo oficial

## Decisões técnicas

### Por que monoio e não tokio?

- io_uring nativo → menos syscalls
- Single-threaded por design → casa com 0.4 CPU sem custo de scheduler multi-thread
- Sem `Send` bound em tasks → mais flexível

### Por que UDS e não TCP loopback?

Top 2 do leaderboard explicita: economia de ~40-60µs por request. Em p99 ≤ 1ms, 50µs é 5% do orçamento.

### Por que SoA?

Dot product i16 itera por dimensão. SoA mantém valores da mesma dimensão contíguos → cache-friendly + AVX2 carrega 16 lanes alinhados.

### Por que respostas pré-montadas?

Score discreto em 6 buckets (0.0, 0.2, 0.4, 0.6, 0.8, 1.0). 6 × 2 (approved/denied) = 12 respostas HTTP completas montadas no startup. Hot path só seleciona índice.

## Referências

- [Top 1 — thiagorigonatti (C + io_uring)](https://github.com/thiagorigonatti/rinha-2026)
- [Top 2 — jairoblatt (Rust + AVX2)](https://github.com/jairoblatt/rinha-2026-rust)
- [Top 3 — viniciusdsandrade (C++ + IVF)](https://github.com/viniciusdsandrade/rinha-de-backend-2026)
- [Apollo Rust Best Practices](https://github.com/apollographql/rust-best-practices)

## Licença

MIT.
