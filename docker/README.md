# Docker Deployment

This folder contains a Docker Compose stack for:
- `web` (Vue build served by Nginx, with `/api` reverse proxy)
- `api` (FastAPI backend)
- infra (default): Neo4j / Milvus / MySQL (and their dependencies)
- optional MCP (`--profile mcp`): FastMCP SSE server
- optional ASR (`--profile asr`): FunASR runtime

## Quick Start (Full Stack)

```bash
# from repo root
# optional: copy .env.example -> .env and fill llm_api_key etc
cp .env.example .env
cd docker
docker compose up -d --build
```

Open:
- Web UI: http://localhost:3100/
- API docs: http://localhost:3100/api/docs
- Direct API: http://localhost:5050/healthz

Containers:
- pk-web
- pk-api

### Auto Neo4j Import

On startup, the one-shot service `neo4j-bootstrap` will import graph data from:

- `../resources/data/kg_data/entities.json`
- `../resources/data/kg_data/relations.json`

This is idempotent: it will **skip** if the marker node exists.

Force re-import (DANGEROUS: wipes the DB):

```bash
cd docker
docker compose run --rm neo4j-bootstrap python scripts/import_graph.py --wait-seconds 120 --force --reset
```

### Optional: Import Map Data (MySQL)

If you need the map feature, import CSV into MySQL:

```bash
cd docker
docker compose exec api python scripts/import_pokemon_map.py
```

## Reset / Clean Start

This stack uses bind-mounted data directories under `docker/volumes/`.

To wipe Neo4j data and re-import:

```bash
cd docker
docker compose down
rm -rf volumes/neo4j/data volumes/neo4j/logs
docker compose up -d --build
```

On Windows PowerShell, you can delete the folders manually or run:

```powershell
cd docker
docker compose down
Remove-Item -Recurse -Force .\\volumes\\neo4j\\data, .\\volumes\\neo4j\\logs
docker compose up -d --build
```

Note:
- Docker Compose loads backend env vars from the repo root `.env` (see `.env.example` / `.env.template`).

## App Only (Without Infra)

```bash
cd docker
docker compose up -d --build api web
```

## With MCP SSE Server

```bash
cd docker
docker compose --profile mcp up -d --build
```

MCP SSE endpoint (inside Docker network): `http://mcp:8000/sse`  
Host access: http://localhost:8000/sse
