#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_DIR="${REPO_ROOT}/docker"

ROOT_ENV="${REPO_ROOT}/.env"
ROOT_ENV_EXAMPLE="${REPO_ROOT}/.env.example"

if [[ ! -f "${ROOT_ENV}" ]]; then
  cp "${ROOT_ENV_EXAMPLE}" "${ROOT_ENV}"
  echo "Created .env from .env.example (please fill llm_api_key)." >&2
fi

cd "${DOCKER_DIR}"
docker compose up -d --build

echo
echo "Neo4j import logs (neo4j-bootstrap):"
docker compose logs --tail=200 neo4j-bootstrap

echo
echo "Open:"
echo "  Web UI:   http://localhost:3100/"
echo "  API Docs: http://localhost:3100/api/docs"
echo "  Neo4j:    http://localhost:7474/"
