# rinha-backend-2026

Submissão da [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) — detecção de fraude em transações de cartão via busca vetorial k-NN, sob restrição de **1 CPU + 350 MB RAM total**.

**Objetivo:** disputar o primeiro lugar do leaderboard (score teto 6000.0, p99 ≤ 1ms, zero erros).

## O desafio em 4 passos

1. Receber payload com `transaction`, `customer`, `merchant`, `terminal`, `last_transaction`
2. Normalizar em vetor de 14 dimensões (`[REGRAS_DE_DETECCAO.md]`)
3. Buscar 5 vizinhos mais próximos no dataset (3M vetores rotulados `fraud`/`legit`)
4. Retornar `{approved: fraud_score < 0.6, fraud_score: n_fraudes/5}`

[REGRAS_DE_DETECCAO.md]: https://github.com/zanfranceschi/rinha-de-backend-2026/blob/main/docs/br/REGRAS_DE_DETECCAO.md

## Stack

- **Linguagem:** Rust 1.82 (edition 2021)
- **Runtime async:** [monoio](https://github.com/bytedance/monoio) (io_uring nativo, single-threaded, fallback legacy)
- **Comunicação LB↔API:** TCP loopback intra-bridge (`tcp:apiN:9000`)
- **Layout dataset:** SoA (Structure of Arrays) i16 (f32 quantizado por `QUANT_SCALE = 4096`), label binário u8, mmap
- **Sentinela:** `-4096` (i16) para `last_transaction: null` (índices 5/6 do vetor)
- **HTTP parser:** manual com `memchr`, sem framework
- **Respostas:** 12 buffers HTTP pré-montados no startup (6 score buckets × 2 approved/denied)
- **SIMD:** AVX2 no hot path do k-NN (`target-cpu=haswell`, `+avx2,+fma`); kernel processa 8 vetores i16 → i32 por iteração com 4 acumuladores independentes

## Arquitetura

```
            :9999 TCP
              │
        ┌─────▼─────┐
        │    lb     │  round-robin atômico, sem lógica
        └──┬─────┬──┘
           │     │      tcp:apiN:9000 (bridge network)
        ┌──▼─┐ ┌─▼──┐
        │api1│ │api2│   monoio + io_uring (fallback legacy)
        └────┘ └────┘   dataset mmap (SoA f32 + AVX2/FMA)
```

## Layout do repositório

```
crates/
├── common/      Dataset (mmap), normalização, MCC, time, SIMD, proto
├── api/         Servidor HTTP + JSON parser + KNN escalar + AVX2
├── lb/          Load balancer TCP → TCP/UDS upstream, round-robin
└── preprocess/  references.json.gz → references.bin (f32 SoA)

data/            Binários do dataset (gitignored)
info.json        Metadados de submissão
```

## Setup

### Pré-requisitos

- Rust 1.82+ (instalado via `rust-toolchain.toml`)
- Docker + Docker Compose
- Arquivo do desafio em `data/`:
  - `references.json.gz`

### Comandos

```bash
make check       # fmt + clippy + test
make build       # cargo build --release
make preprocess  # gera data/references.bin (uma vez)
make run         # docker compose up --build
```

### Dev em Apple Silicon (Docker Desktop)

QEMU em Apple Silicon não emula AVX2 corretamente (autovec do `f32` div gera bits garbage) e Docker Desktop bloqueia `io_uring_setup` via seccomp. Use o override `docker-compose.dev.yml`:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

Override desliga `target-cpu=haswell` e força `MONOIO_DRIVER=legacy`. A imagem de submissão (sem override) mantém AVX2 + FMA + io_uring quando disponível.

### Validação manual

```bash
curl -X POST http://localhost:9999/fraud-score \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "tx-1329056812",
    "transaction": {"amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z"},
    "customer":    {"avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003","MERC-016"]},
    "merchant":    {"id": "MERC-016", "mcc": "5411", "avg_amount": 60.25},
    "terminal":    {"is_online": false, "card_present": true, "km_from_home": 29.23},
    "last_transaction": null
  }'
# → {"approved":true,"fraud_score":0.0}
```

## Decisões técnicas

### MCC risk e normalization constants embedded

A spec garante que `mcc_risk.json` e `normalization.json` não mudam durante o teste. Para zerar I/O no startup, os defaults estão hardcoded em `common::mcc::McCRiskTable::default()` e `common::NormalizationConfig::default()` — espelhando o conteúdo oficial.

### Fallback de erro = `200 approved=true score=0.0`

Pesos da fórmula de detecção: FP=1, FN=3, **HTTP error=5**. Em qualquer falha de parse/normalização, retornar 200 OK com aprovação minimiza o pior caso esperado (FN no lugar de Err).

### Por que monoio e não tokio?

- io_uring nativo → menos syscalls
- Single-threaded por design → casa com 0.4 CPU sem custo de scheduler multi-thread
- Sem `Send` bound em tasks → mais flexível

### Por que TCP loopback e não UDS?

UDS sobre tmpfs falha com `ENOTSUP` no `bind(2)` quando o runner do smoke test cai no driver legacy do monoio (sem io_uring). TCP intra-bridge funciona em qualquer kernel/runner com overhead `~µs` no loopback Docker. Em produção real (kernel 6.x, io_uring nativo) o ganho de UDS é marginal versus o custo de portabilidade.

### Por que i16 SoA quantizado?

- Espaço: 14 × 2 bytes × 3M = **84 MB**. f32 (168 MB) não cabia em 170 MB de RAM por API → OOM kill no bench oficial.
- Quantização: `round(f32 * 4096)`. Pós-normalização f32 ∈ `[-1, 1]` → i16 ∈ `[-4096, 4096]`. Erro de quantização ≈ 0.024%, irrelevante pra k-NN com k=5.
- Cache-friendly: dot product itera por dimensão; SoA mantém cada dimensão contígua. Memory bandwidth cai pela metade vs f32.
- AVX2: subtrai i16 (cabe no range), expande pra i32 com `_mm256_cvtepi16_epi32`, eleva ao quadrado com `_mm256_mullo_epi32`. Acumulador i32 cobre 14 dims sem overflow (max 940 M < 2.1 G).

### Por que respostas pré-montadas?

`fraud_score` discretiza em 6 buckets: `n/5` para n ∈ {0..5}. 6 × 2 (approved/denied) = 12 respostas HTTP completas montadas no startup. Hot path só seleciona índice.

## Roadmap pra atingir 6000.0

- [x] Scaffold com workspace, Dockerfile multi-stage, CI
- [x] Schema do payload (5 sub-objetos + `last_transaction: null`)
- [x] Normalização das 14 dimensões + sentinela `NaN`
- [x] Tabela MCC + constantes embedded
- [x] Parser HTTP manual + 12 respostas pré-montadas
- [x] Dataset SoA i16 quantizado + mmap + label binário u8 (84 MB)
- [x] KNN brute-force escalar baseline
- [x] AVX2 i16 → i32 dot product no hot path
- [x] Fallback `200 approved=true` em erro
- [x] Imagem pública em `ghcr.io/tarcisiopgs/rinha-backend-2026:latest` com dataset embutido
- [x] Branch `submission` + `docker-compose.yml` (TCP intra-bridge)
- [x] PR `participants/tarcisiopgs.json` mergeado no repo do desafio
- [x] Issue `rinha/test` aberta para bench oficial
- [ ] Parser JSON manual no hot path (substituir `serde_json`)
- [ ] IVF index (kmeans clusters, nlist tuning) — brute-force em 3M é caro
- [ ] Quantização i8 (avaliar perda de recall)
- [ ] Huge pages 2MB pro mmap do dataset
- [ ] Pin de thread em CPU (sched_setaffinity) + `SCHED_FIFO` se permitido
- [ ] LB com `splice(2)` ao invés de copy bidirecional
- [ ] Profile com `perf stat` + flamegraph em hardware do desafio

## Submissão

A spec exige duas branches:

- `main` — código-fonte (esta branch)
- `submission` — apenas `docker-compose.yml` + `info.json`, referenciando imagem pública em `ghcr.io`

Detalhes em [SUBMISSAO.md](https://github.com/zanfranceschi/rinha-de-backend-2026/blob/main/docs/br/SUBMISSAO.md).

## Licença

MIT.
