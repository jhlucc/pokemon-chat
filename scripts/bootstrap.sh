#!/usr/bin/env bash
set -euo pipefail

export PYTHONUTF8=1

echo "Installing backend deps (pip) ..."
python -m pip install -r requirements.txt

echo "Installing frontend deps (npm) ..."
npm --prefix web install

echo "Done."

