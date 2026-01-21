param(
  [int]$Port = 5050
)

$ErrorActionPreference = "Stop"

# Ensure UTF-8 output (helps when logs contain Chinese/emoji)
$env:PYTHONUTF8 = "1"

Write-Host "Starting FastAPI server on port $Port ..."
python -m uvicorn server.main:app --host 0.0.0.0 --port $Port --reload
