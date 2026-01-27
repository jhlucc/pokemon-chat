$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$dockerDir = Join-Path $repoRoot "docker"

Set-Location $dockerDir

if (-not (Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
  Write-Host "Created docker/.env from docker/.env.example (please fill llm_api_key)." -ForegroundColor Yellow
}

docker compose up -d --build

Write-Host ""
Write-Host "Neo4j import logs (neo4j-bootstrap):" -ForegroundColor Cyan
docker compose logs --tail=200 neo4j-bootstrap

Write-Host ""
Write-Host "Open:" -ForegroundColor Cyan
Write-Host "  Web UI:  http://localhost:3100/"
Write-Host "  API Docs: http://localhost:3100/api/docs"
Write-Host "  Neo4j:   http://localhost:7474/"

