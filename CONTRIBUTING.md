# Contributing

Thanks for contributing to **pokemon-chat**!

## Prerequisites

- Python 3.11+ (recommended)
- Node.js 20+ (for `web/`)
- Docker + Docker Compose (optional, for infra)

## Backend (FastAPI)

Install deps:

```bash
python -m pip install -r requirements.txt
python -m pip install ruff pytest pytest-asyncio
```

Or (recommended for development):

```bash
python -m pip install -r requirements-dev.txt
```

Run lint/format/tests:

```bash
python -m ruff check server src scripts
python -m ruff format server src scripts
python -m pytest
```

Run the API:

```bash
python -m server
# or: python -m server.main
```

## Frontend (Vue3 + Vite)

```bash
cd web
npm ci
npm run lint:check
npm run typecheck
npm run build
npm run dev
```

## Pre-commit (optional)

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run -a
```
