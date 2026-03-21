#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found in PATH" >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment: $VENV_DIR" >&2
  uv venv "$VENV_DIR"
fi

echo "Installing redact runtime dependencies..." >&2
uv pip install \
  --python "$PYTHON_BIN" \
  pillow \
  pymupdf \
  paddleocr \
  paddlepaddle

echo "$VENV_DIR"
