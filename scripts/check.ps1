$ErrorActionPreference = "Stop"

Write-Host "== Backend ==" -ForegroundColor Cyan
python -m ruff check server src scripts
python -m ruff format --check server src scripts
python -m pytest

Write-Host "== Frontend ==" -ForegroundColor Cyan
npm --prefix web run lint:check
npm --prefix web run typecheck
npm --prefix web run build

Write-Host "All checks passed." -ForegroundColor Green

