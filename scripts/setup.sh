#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-tested.txt
.venv/bin/python -m pip install --no-deps -e .
if [ ! -f artifacts/research.json.gz ]; then
  .venv/bin/python -m orderbook.research --demo --output artifacts/research.json.gz
fi
.venv/bin/python scripts/summarize.py
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
