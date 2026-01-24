$ErrorActionPreference = "Stop"

$env:PYTHONUTF8 = "1"

Write-Host "Running offline test suite (integration tests skipped by default) ..."
python -m pytest -q
