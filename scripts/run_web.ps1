param(
  [int]$Port = 3100
)

$ErrorActionPreference = "Stop"

Write-Host "Starting Vite dev server on port $Port ..."
npm --prefix web run dev -- --host 0.0.0.0 --port $Port

