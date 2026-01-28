$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$dockerDir = Join-Path $repoRoot "docker"

$rootEnv = Join-Path $repoRoot ".env"
$rootEnvExample = Join-Path $repoRoot ".env.example"

if (-not (Test-Path $rootEnv)) {
  Copy-Item $rootEnvExample $rootEnv
  Write-Host "Created .env from .env.example (please fill llm_api_key)." -ForegroundColor Yellow
}

Set-Location $dockerDir
docker compose up -d --build

Write-Host ""
Write-Host "Neo4j import logs (neo4j-bootstrap):" -ForegroundColor Cyan
docker compose logs --tail=200 neo4j-bootstrap

Write-Host ""
Write-Host "Open:" -ForegroundColor Cyan
Write-Host "  Web UI:  http://localhost:3100/"
Write-Host "  API Docs: http://localhost:3100/api/docs"
Write-Host "  Neo4j:   http://localhost:7474/"

