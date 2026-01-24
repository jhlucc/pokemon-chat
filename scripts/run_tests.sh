#!/usr/bin/env bash
set -euo pipefail

export PYTHONUTF8=1

echo "Running offline test suite (Docker + python3.11) ..."
docker run --rm \
  -v "${PWD}:/app" \
  -v "${HOME}/.cache/pip:/root/.cache/pip" \
  -w /app \
  python:3.11-slim \
  bash -lc "\
    apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt pytest pytest-asyncio \
    && pytest -q"
