$ErrorActionPreference = "Stop"

Write-Host "Starting docker services (Neo4j/Milvus/MySQL/Whisper) ..."
Push-Location (Join-Path $PSScriptRoot "..\\docker")
try {
  docker compose up -d
} finally {
  Pop-Location
}

