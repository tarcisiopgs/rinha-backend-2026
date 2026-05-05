.PHONY: help fmt lint test build run preprocess clean check ci

DATA_DIR := data
REFERENCES_GZ := $(DATA_DIR)/references.json.gz
REFERENCES_BIN := $(DATA_DIR)/references.bin

help:
	@echo "alvos:"
	@echo "  fmt           rustfmt em todo workspace"
	@echo "  lint          clippy + fmt --check"
	@echo "  test          cargo test --workspace"
	@echo "  build         cargo build --release --bin api --bin lb"
	@echo "  preprocess    converte references.json.gz -> references.bin"
	@echo "  run           docker compose up --build"
	@echo "  check         fmt + lint + test (rápido)"
	@echo "  ci            check + build + preprocess (pipeline completo)"
	@echo "  clean         cargo clean + remove binário do dataset"

fmt:
	cargo fmt --all

lint:
	cargo fmt --all --check
	cargo clippy --workspace --all-targets --locked -- -D warnings

test:
	cargo test --workspace --locked

build:
	cargo build --release --bin api --bin lb --locked

preprocess: $(REFERENCES_BIN)

$(REFERENCES_BIN): $(REFERENCES_GZ)
	cargo run --release --bin preprocess -- $(REFERENCES_GZ) $(REFERENCES_BIN)

$(REFERENCES_GZ):
	@echo "ERRO: $(REFERENCES_GZ) não encontrado." >&2
	@echo "Baixe de https://github.com/zanfranceschi/rinha-de-backend-2026 e coloque em $(DATA_DIR)/" >&2
	@exit 1

run:
	docker compose up --build

check: fmt lint test

ci: check build

clean:
	cargo clean
	rm -f $(REFERENCES_BIN)
