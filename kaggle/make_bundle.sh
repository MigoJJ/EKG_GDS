#!/usr/bin/env bash
# Builds Kaggle upload bundles into dist/
#   dist/ekg_code.zip       - training code (always)
#   dist/ptbxl_100hz.zip    - PTB-XL 100Hz records + CSVs (with --data, ~600MB)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist"
mkdir -p "$DIST"

cd "$ROOT"
rm -f "$DIST/ekg_code.zip"
zip -qr "$DIST/ekg_code.zip" train_improved.py models/export_onnx.py pipeline \
    -x '*/__pycache__/*' '*.pyc'
echo "[OK] $DIST/ekg_code.zip ($(du -h "$DIST/ekg_code.zip" | cut -f1))"

if [[ "${1:-}" == "--data" ]]; then
    PTBXL="$(readlink -f "$ROOT/data/ptb-xl")"
    rm -f "$DIST/ptbxl_100hz.zip"
    (cd "$PTBXL" && zip -qr "$DIST/ptbxl_100hz.zip" records100 ptbxl_database.csv scp_statements.csv)
    echo "[OK] $DIST/ptbxl_100hz.zip ($(du -h "$DIST/ptbxl_100hz.zip" | cut -f1))"
fi
