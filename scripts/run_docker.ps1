param(
  [switch]$WithMcp
)

$ErrorActionPreference = "Stop"

Write-Host "Starting docker stack ..."
Push-Location (Join-Path $PSScriptRoot "..\\docker")
try {
  if ($WithMcp) {
    docker compose --profile infra --profile mcp up -d --build
  } else {
    docker compose --profile infra up -d --build
  }
} finally {
  Pop-Location
}
