# Docker Deployment

This folder contains a Docker Compose stack for:
- `web` (Vue build served by Nginx, with `/api` reverse proxy)
- `api` (FastAPI backend)
- optional infra (`--profile infra`): Neo4j / Milvus / MySQL / FunASR
- optional MCP (`--profile mcp`): FastMCP SSE server

## Quick Start (App Only)

```bash
cd docker
# optional: copy docker/.env.example to docker/.env and fill llm_api_key etc
docker compose up -d --build
```

Open:
- Web UI: http://localhost:3100/
- API docs: http://localhost:3100/api/docs
- Direct API: http://localhost:5050/healthz

## Full Stack (Infra + App)

```bash
cd docker
docker compose --profile infra up -d --build
```

## With MCP SSE Server

```bash
cd docker
docker compose --profile infra --profile mcp up -d --build
```

MCP SSE endpoint (inside Docker network): `http://mcp:8000/sse`  
Host access: http://localhost:8000/sse

