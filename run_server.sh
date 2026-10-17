#!/usr/bin/env bash
# EKG Reading Assistant FastAPI Server Launcher
# Binds to 0.0.0.0:8000 with 1 worker optimized for CPU workstation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate Python 3.11 Virtual Environment
source "$SCRIPT_DIR/.venv/bin/activate"
export PYTHONPATH="$SCRIPT_DIR"

echo "=================================================================="
echo " Starting 12-Lead EKG AI Reading Assistant API Service"
echo " Host   : 0.0.0.0:8000"
echo " Docs   : http://localhost:8000/docs (Swagger UI)"
echo " Health : http://localhost:8000/api/v1/health"
echo " Ollama : http://localhost:11434"
echo "=================================================================="

exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
