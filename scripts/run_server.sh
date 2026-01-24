#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-5050}"
export PYTHONUTF8=1

echo "Starting FastAPI server on port ${PORT} ..."
python3 -m uvicorn server.main:app --host 0.0.0.0 --port "${PORT}" --reload

