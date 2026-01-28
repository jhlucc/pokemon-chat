.PHONY: help docker-up docker-up-infra docker-up-full docker-up-asr docker-up-all docker-down docker-logs web-install web-dev web-build web-lint web-typecheck api-logs py-lint py-format py-format-check py-test check

PY ?= python

help:
	@echo "Targets:"
	@echo "  docker-up        Build+run app only (web+api)"
	@echo "  docker-up-infra   Run app + infra (neo4j/milvus/mysql/etc)"
	@echo "  docker-up-full    Run app + infra + mcp"
	@echo "  docker-up-asr     Run app + asr (FunASR)"
	@echo "  docker-up-all     Run app + infra + mcp + asr"
	@echo "  docker-down      Stop the compose stack"
	@echo "  docker-logs      Tail logs"
	@echo "  web-install      Install frontend deps (npm ci)"
	@echo "  web-dev          Run Vite dev server"
	@echo "  web-build        Build frontend"
	@echo "  web-lint         Lint frontend (eslint)"
	@echo "  web-typecheck    Typecheck frontend (vue-tsc)"
	@echo "  api-logs         Tail API logs"
	@echo "  py-lint          Lint backend (ruff)"
	@echo "  py-format        Format backend (ruff)"
	@echo "  py-format-check  Check backend formatting (ruff)"
	@echo "  py-test          Run backend tests (pytest)"
	@echo "  check            Run backend + frontend checks (like CI)"

docker-up:
	cd docker && docker compose up -d --build

docker-up-infra:
	cd docker && docker compose --profile infra up -d --build

docker-up-full:
	cd docker && docker compose --profile infra --profile mcp up -d --build

docker-up-asr:
	cd docker && docker compose --profile asr up -d --build

docker-up-all:
	cd docker && docker compose --profile infra --profile mcp --profile asr up -d --build

docker-down:
	cd docker && docker compose down

docker-logs:
	cd docker && docker compose logs -f --tail=200

api-logs:
	cd docker && docker compose logs -f --tail=200 api

web-install:
	cd web && npm ci

web-dev:
	cd web && npm run dev

web-build:
	cd web && npm run build

web-lint:
	cd web && npm run lint:check

web-typecheck:
	cd web && npm run typecheck

py-lint:
	$(PY) -m ruff check server src scripts

py-format:
	$(PY) -m ruff format server src scripts

py-format-check:
	$(PY) -m ruff format --check server src scripts

py-test:
	$(PY) -m pytest

check: py-lint py-format-check py-test web-lint web-typecheck web-build
