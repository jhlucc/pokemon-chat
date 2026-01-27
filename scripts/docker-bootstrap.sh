#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_DIR="${REPO_ROOT}/docker"

cd "${DOCKER_DIR}"

if [[ ! -f ".env" ]]; then
  cp ".env.example" ".env"
  echo "Created docker/.env from docker/.env.example (please fill llm_api_key)." >&2
fi

docker compose up -d --build

echo
echo "Neo4j import logs (neo4j-bootstrap):"
docker compose logs --tail=200 neo4j-bootstrap

echo
echo "Open:"
echo "  Web UI:   http://localhost:3100/"
echo "  API Docs: http://localhost:3100/api/docs"
echo "  Neo4j:    http://localhost:7474/"

