#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-3100}"

echo "Starting Vite dev server on port ${PORT} ..."
npm --prefix web run dev -- --host 0.0.0.0 --port "${PORT}"

