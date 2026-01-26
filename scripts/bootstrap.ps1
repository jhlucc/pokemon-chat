$ErrorActionPreference = "Stop"

$env:PYTHONUTF8 = "1"

Write-Host "Installing backend deps (pip) ..."
python -m pip install -r requirements.txt

Write-Host "Installing frontend deps (npm) ..."
npm --prefix web install

Write-Host "Done."

