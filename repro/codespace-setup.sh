#!/usr/bin/env bash
# Setup do ambiente de reprodução do bench oficial num GitHub Codespace
# (Ubuntu 24.04 x86_64 nativo, kernel real, sem emulação). Replica o que o
# bot da Rinha faz: clona a submission branch, sobe o compose oficial e
# roda o k6 com test.js do upstream.
#
# Uso (dentro do codespace):
#   bash repro/codespace-setup.sh up      # sobe o stack + warm-up
#   bash repro/codespace-setup.sh bench   # roda o k6 e mostra resultado
#   bash repro/codespace-setup.sh logs    # tail dos logs dos 3 containers
#   bash repro/codespace-setup.sh down    # derruba o stack

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/tarcisiopgs/rinha-backend-2026}"
RINHA_REPO="${RINHA_REPO:-https://github.com/zanfranceschi/rinha-de-backend-2026}"
WORK="${WORK:-/tmp/rinha-repro}"
PARTICIPANT_DIR="$WORK/participant"
RINHA_DIR="$WORK/rinha"
COMPOSE="$PARTICIPANT_DIR/docker-compose.yml"

ensure_tools() {
    local missing=()
    command -v docker >/dev/null || missing+=(docker)
    command -v git    >/dev/null || missing+=(git)
    command -v curl   >/dev/null || missing+=(curl)
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "faltam binários: ${missing[*]}" >&2
        exit 1
    fi

    if ! command -v k6 >/dev/null; then
        echo "instalando k6..."
        sudo gpg -k >/dev/null 2>&1 || true
        sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg \
            --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
        echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" \
            | sudo tee /etc/apt/sources.list.d/k6.list >/dev/null
        sudo apt-get update -qq
        sudo apt-get install -y k6
    fi
}

clone_repos() {
    mkdir -p "$WORK"
    if [ ! -d "$PARTICIPANT_DIR/.git" ]; then
        git clone --branch submission --depth 1 "$REPO_URL" "$PARTICIPANT_DIR"
    else
        git -C "$PARTICIPANT_DIR" fetch origin submission
        git -C "$PARTICIPANT_DIR" reset --hard origin/submission
    fi
    if [ ! -d "$RINHA_DIR/.git" ]; then
        git clone --depth 1 "$RINHA_REPO" "$RINHA_DIR"
    else
        git -C "$RINHA_DIR" pull --ff-only
    fi
}

up() {
    ensure_tools
    clone_repos
    docker compose -f "$COMPOSE" pull
    docker compose -f "$COMPOSE" up -d
    echo "aguardando /ready..."
    for i in $(seq 1 30); do
        if curl -sf -o /dev/null --max-time 2 http://localhost:9999/ready; then
            echo "ready (tentativa $i)"
            return 0
        fi
        sleep 2
    done
    echo "/ready não respondeu em 60s" >&2
    docker compose -f "$COMPOSE" logs --tail=80 || true
    exit 1
}

bench() {
    if ! curl -sf -o /dev/null --max-time 2 http://localhost:9999/ready; then
        echo "/ready não responde — rode '$0 up' primeiro" >&2
        exit 1
    fi
    cd "$RINHA_DIR"
    K6_NO_USAGE_REPORT=true k6 run test/test.js
    echo "---"
    echo "results.json:"
    cat test/results.json 2>/dev/null | python3 -m json.tool || cat test/results.json
}

logs() {
    docker compose -f "$COMPOSE" logs -f --tail=200
}

down() {
    docker compose -f "$COMPOSE" down -v || true
}

case "${1:-help}" in
    up)    up ;;
    bench) bench ;;
    logs)  logs ;;
    down)  down ;;
    *)
        cat <<USAGE
uso: $0 <up|bench|logs|down>

  up     clone submission + rinha, sobe compose, espera /ready
  bench  roda k6 test/test.js do upstream e imprime results.json
  logs   tail -f dos containers do compose
  down   derruba e remove volumes
USAGE
        exit 0
        ;;
esac
